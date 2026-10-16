"""One-to-one assignment of expected RTC-S1 passes to DIST-S1 products, shared by the coverage and orphan checks."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from dist_s1_prod_check.constants import PASS_MATCH_TOLERANCE_SECONDS


PassKey = tuple[str, int, int]  # (mgrs_tile_id, acq_group_id_within_mgrs_tile, pass_id)
PASS_KEY_COLUMNS = ['mgrs_tile_id', 'acq_group_id_within_mgrs_tile', 'pass_id']

# Two assignments whose total |time difference| differs by less than this many seconds are equally good.
_TIE_SECONDS = 1e-3


@dataclass
class PassAssignment:
    """Result of `assign_passes`.

    Attributes
    ----------
    pass_to_product : dict
        Pass key to the `opera_id` of its matched product. One-to-one with `product_to_pass`.
    product_to_pass : dict
        Product `opera_id` to its matched pass key.
    time_diff_seconds : dict
        Pass key to |product acq_time - pass post_acq_dt| in seconds, for matched passes.
    ambiguous : dict
        Pass key to the `opera_id`s of the products it could equally well be matched to (an optimal assignment
        exists in which it gets a different product, or none). Its entry in `pass_to_product` (if any) is tentative.
    unmatched_passes : list
        Pass keys assigned to their dummy (no product within tolerance left for them), in input order.
    orphan_products : list
        Product `opera_id`s assigned to their dummy, in input order.
    """

    pass_to_product: dict[PassKey, str] = field(default_factory=dict)
    product_to_pass: dict[str, PassKey] = field(default_factory=dict)
    time_diff_seconds: dict[PassKey, float] = field(default_factory=dict)
    ambiguous: dict[PassKey, list[str]] = field(default_factory=dict)
    unmatched_passes: list[PassKey] = field(default_factory=list)
    orphan_products: list[str] = field(default_factory=list)


def pass_key(row: object) -> PassKey:
    """Return the key of an expected pass: ``(mgrs_tile_id, acq_group_id_within_mgrs_tile, pass_id)``."""
    return (row.mgrs_tile_id, int(row.acq_group_id_within_mgrs_tile), int(row.pass_id))


def _cost_matrix(diffs: np.ndarray, tolerance_seconds: float) -> tuple[np.ndarray, float, float]:
    """Square cost matrix for passes (rows) x products (columns), augmented with one dummy per pass and per product.

    Layout, with P passes and Q products::

                     products (Q)        pass dummies (P)
        passes (P)   |diff| or BIG       DUMMY on the diagonal, BIG elsewhere
        prod dummies DUMMY on diagonal   0
                     BIG elsewhere

    A matched pair costs at most `tolerance_seconds`; leaving the pass and the product both unmatched costs
    2 * DUMMY, so every pair within tolerance is preferred to its two dummies. Pairs beyond tolerance cost BIG,
    which exceeds the cost of the all-dummy assignment, so no optimal assignment ever uses one. No infinities.
    """
    n_pass, n_prod = diffs.shape
    dummy = float(tolerance_seconds) + 1.0
    big = (n_pass + n_prod + 1) * dummy * 2
    n = n_pass + n_prod
    cost = np.full((n, n), big)
    cost[:n_pass, :n_prod] = np.where(diffs <= tolerance_seconds, diffs, big)
    cost[np.arange(n_pass), n_prod + np.arange(n_pass)] = dummy
    cost[n_pass + np.arange(n_prod), np.arange(n_prod)] = dummy
    cost[n_pass:, n_prod:] = 0.0
    return cost, dummy, big


def _solve(cost: np.ndarray) -> tuple[np.ndarray, float]:
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)], float(cost[rows, cols].sum())


def _assign_tile(
    keys: list[PassKey],
    post_acq: np.ndarray,
    opera_ids: list[str],
    acq: np.ndarray,
    tolerance_seconds: float,
    result: PassAssignment,
) -> None:
    n_pass, n_prod = len(keys), len(opera_ids)
    if n_prod == 0:
        result.unmatched_passes.extend(keys)
        return
    if n_pass == 0:
        result.orphan_products.extend(opera_ids)
        return
    diffs = np.abs((acq[None, :] - post_acq[:, None]) / np.timedelta64(1, 's')).astype(float)
    cost, _, big = _cost_matrix(diffs, tolerance_seconds)
    col_of_row, best = _solve(cost)

    matched_products = set()
    for i, key in enumerate(keys):
        j = col_of_row[i]
        if j < n_prod:
            result.pass_to_product[key] = opera_ids[j]
            result.product_to_pass[opera_ids[j]] = key
            result.time_diff_seconds[key] = float(diffs[i, j])
            matched_products.add(j)
        else:
            result.unmatched_passes.append(key)
    result.orphan_products.extend(opera_ids[j] for j in range(n_prod) if j not in matched_products)

    # A pass is ambiguous if an equally good assignment gives it a different outcome: re-solve with its current
    # outcome forbidden. Only a pass with a candidate can have one, and a matched pass only if it has a second
    # candidate or its product is a candidate of another pass.
    candidates = diffs <= tolerance_seconds
    for i, key in enumerate(keys):
        j = col_of_row[i]
        n_cand = int(candidates[i].sum())
        if n_cand == 0 or (j < n_prod and n_cand == 1 and candidates[:, j].sum() == 1):
            continue
        alt = cost.copy()
        alt[i, j] = big
        alt_col, alt_total = _solve(alt)
        if alt_total - best < _TIE_SECONDS:
            options = {j, alt_col[i]} - {c for c in (j, alt_col[i]) if c >= n_prod}
            result.ambiguous[key] = [opera_ids[c] for c in sorted(options)]


def assign_passes(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    tolerance_seconds: float = PASS_MATCH_TOLERANCE_SECONDS,
) -> PassAssignment:
    """Assign expected RTC-S1 passes to DIST-S1 products one-to-one within each MGRS tile.

    The assignment minimizes the total |product acq_time - pass post_acq_dt| over the pairs it makes
    (`scipy.optimize.linear_sum_assignment`), after first matching as many pairs as possible. Only pairs within
    `tolerance_seconds` can be matched. A pass left without a product is unmatched (missing); a product left without
    a pass is an orphan. Coverage and orphan checks must consume the same assignment so that every product is either
    matched or an orphan, never both or neither.

    Parameters
    ----------
    df_expected_groups : pd.DataFrame
        Expected passes (`coverage.expected_pass_groups`): `mgrs_tile_id`, `acq_group_id_within_mgrs_tile`,
        `pass_id`, `post_acq_dt` (UTC).
    df_dist_dedup : pd.DataFrame
        Latest DIST-S1 product versions: `opera_id`, `mgrs_tile_id`, `acq_time`.
    tolerance_seconds : float
        Largest |time difference| for a pass and product to be matched.

    Returns
    -------
    PassAssignment
    """
    result = PassAssignment()
    products = {}
    if len(df_dist_dedup):
        acq_utc = pd.to_datetime(df_dist_dedup.acq_time, utc=True)
        for tile, group in df_dist_dedup.groupby('mgrs_tile_id', sort=False):
            products[tile] = (group.opera_id.tolist(), acq_utc[group.index].values.astype('datetime64[us]'))
    passes = {}
    if len(df_expected_groups):
        for tile, group in df_expected_groups.groupby('mgrs_tile_id', sort=False):
            post = pd.to_datetime(group.post_acq_dt, utc=True).values.astype('datetime64[us]')
            passes[tile] = ([pass_key(row) for row in group.itertuples()], post)
    empty_ids, empty_times = [], np.array([], dtype='datetime64[us]')
    for tile in list(passes) + [t for t in products if t not in passes]:
        keys, post = passes.get(tile, ([], empty_times))
        opera_ids, acq = products.get(tile, (empty_ids, empty_times))
        _assign_tile(keys, post, opera_ids, acq, tolerance_seconds, result)

    pass_order = {key: n for n, key in enumerate(k for keys, _ in passes.values() for k in keys)}
    result.unmatched_passes.sort(key=pass_order.__getitem__)
    product_order = {opera_id: n for n, opera_id in enumerate(df_dist_dedup.opera_id)} if len(df_dist_dedup) else {}
    result.orphan_products.sort(key=product_order.__getitem__)
    return result
