import numpy as np
import pandas as pd

from dist_s1_prod_check.status import Status
from dist_s1_prod_check.tags import prior_tag_present


# Deterministic order within a tile/acquisition group: last processed wins, ties broken by the full opera_id.
VERSION_ORDER = ['opera_dedup_id', 'processing_time', 'opera_id']


def product_version(opera_id: str) -> str:
    """Product version token of a DIST-S1 id, e.g. ``v1.0``."""
    return opera_id.rsplit('_', 1)[-1]


def filter_product_versions(df_dist: pd.DataFrame, allowed: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep products whose version is in `allowed`; return them and the number excluded per other version."""
    versions = df_dist.opera_id.map(product_version)
    keep = versions.isin(allowed)
    excluded = {str(v): int(n) for v, n in versions[~keep].value_counts().sort_index().items()}
    return df_dist[keep].reset_index(drop=True), excluded


def attach_version_context(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Add `n_versions`, the number of versions of each product's acquisition in `df_dist`.

    Call it on the full (version-filtered) table before any processing-time filter, so a campaign run still knows
    that an acquisition was produced before (F18, decision 1); `deduplicate_products` then keeps the column.
    """
    return df_dist.assign(n_versions=df_dist.groupby('opera_dedup_id')['opera_id'].transform('size'))


def deduplicate_products(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Keep one product per tile/acquisition: the one processed last (ties: the greatest opera_id).

    `n_versions` is taken from `df_dist` when present (see `attach_version_context`), else counted in `df_dist`.
    """
    df = df_dist.sort_values(VERSION_ORDER, kind='stable').copy()
    if 'n_versions' not in df.columns:
        df['n_versions'] = df.groupby('opera_dedup_id')['opera_id'].transform('size')
    df = df[~df.duplicated(subset='opera_dedup_id', keep='last')]
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)


def check_duplicates(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Flag DIST-S1 granules sharing the same tile/acquisition (reprocessed or duplicated products).

    The last-processed granule of each group is taken as the correct one; the earlier ones are
    superseded.
    """
    dup_mask = df_dist.duplicated(subset='opera_dedup_id', keep=False)
    df_dup = df_dist[dup_mask].sort_values(VERSION_ORDER, kind='stable').copy()
    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'n_versions',
        'is_latest',
        'latest_opera_id',
        'reprocessing_lag_hours',
        'reason',
    ]
    if df_dup.empty:
        return pd.DataFrame(columns=cols)
    grouped = df_dup.groupby('opera_dedup_id')
    df_dup['n_versions'] = grouped['opera_id'].transform('size')
    df_dup['latest_opera_id'] = grouped['opera_id'].transform('last')
    df_dup['is_latest'] = df_dup.opera_id == df_dup.latest_opera_id
    first_processing = grouped['processing_time'].transform('first')
    df_dup['reprocessing_lag_hours'] = (df_dup.processing_time - first_processing).dt.total_seconds() / 3600
    df_dup['reason'] = [
        f'{n} granules share acquisition {d}; ' + ('latest, kept' if latest else f'superseded by {keep}')
        for n, d, latest, keep in zip(
            df_dup.n_versions, df_dup.opera_dedup_id, df_dup.is_latest, df_dup.latest_opera_id
        )
    ]
    extra = [c for c in ['alert_url'] if c in df_dup.columns]
    return df_dup[cols + extra].reset_index(drop=True)


def check_processing_order(df_dist_dedup: pd.DataFrame) -> pd.DataFrame:
    """Check that within each MGRS tile products were processed in acquisition order.

    Reprocessed acquisitions are excluded rather than evaluated: the retained version of a
    duplicate is by construction processed late, so keeping it would flag both it and every
    product acquired after it in the tile. Duplicates are the subject of `check_duplicates`.
    Each product is compared against the previous evaluated acquisition in its tile, so one
    genuinely late product is one failure and not a cascade.
    """
    df = df_dist_dedup.sort_values(['mgrs_tile_id', 'acq_time']).copy()
    n_versions = df.n_versions if 'n_versions' in df.columns else pd.Series(1, index=df.index)
    df['is_reprocessed'] = n_versions > 1

    evaluated = df[~df.is_reprocessed].groupby('mgrs_tile_id')
    df['prev_opera_id'] = evaluated['opera_id'].shift(1)
    df['prev_acq_time'] = evaluated['acq_time'].shift(1)
    df['prev_processing_time'] = evaluated['processing_time'].shift(1)
    df['processing_lag_hours'] = (df.processing_time - df.prev_processing_time).dt.total_seconds() / 3600
    _judge_order(df, 'the previous acquisition in the tile')
    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'prev_opera_id',
        'prev_acq_time',
        'prev_processing_time',
        'processing_lag_hours',
        'is_reprocessed',
        'evaluated',
        'out_of_order',
        'tie',
        'status',
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)


def _judge_order(df: pd.DataFrame, what: str) -> None:
    """Add `evaluated`, `out_of_order`, `tie`, `status` and `reason` from `prev_opera_id` and the processing lag.

    A product processed before its predecessor is out of order (FAIL). Equal processing times are in order but
    counted as `tie` (F36). A product with no predecessor is SKIPPED.
    """
    df['evaluated'] = df.prev_opera_id.notna()
    df['out_of_order'] = df.evaluated & (df.processing_lag_hours < 0)
    df['tie'] = df.evaluated & (df.processing_lag_hours == 0)
    df['status'] = np.select([~df.evaluated, df.out_of_order], [Status.SKIPPED, Status.FAIL], Status.PASS)
    df['reason'] = [
        f'processed {-lag:.1f} h before {what} ({prev})' if bad else ''
        for bad, lag, prev in zip(df.out_of_order, df.processing_lag_hours, df.prev_opera_id)
    ]


def check_first_production_order(df_versions: pd.DataFrame, dedup_ids: pd.Series | None = None) -> pd.DataFrame:
    """Check that each acquisition was *first* produced after the previous acquisition in its tile (F08).

    Takes every version (`df_versions`, before any processing filter) and keeps each acquisition's earliest
    version, so a reprocessing never hides an original inversion. With `dedup_ids`, only those acquisitions are
    reported (their predecessors still come from all of `df_versions`). Every acquisition but the first of its tile
    is evaluated. Columns as `check_processing_order`, without `is_reprocessed`.
    """
    df = df_versions.sort_values(['opera_dedup_id', 'processing_time', 'opera_id'], kind='stable')
    df = df.drop_duplicates(subset='opera_dedup_id', keep='first').sort_values(['mgrs_tile_id', 'acq_time']).copy()
    grouped = df.groupby('mgrs_tile_id')
    df['prev_opera_id'] = grouped['opera_id'].shift(1)
    df['prev_acq_time'] = grouped['acq_time'].shift(1)
    df['prev_processing_time'] = grouped['processing_time'].shift(1)
    df['processing_lag_hours'] = (df.processing_time - df.prev_processing_time).dt.total_seconds() / 3600
    _judge_order(df, 'the first production of the previous acquisition in the tile')
    if dedup_ids is not None:
        df = df[df.opera_dedup_id.isin(set(dedup_ids))]
    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'prev_opera_id',
        'prev_acq_time',
        'prev_processing_time',
        'processing_lag_hours',
        'evaluated',
        'out_of_order',
        'tie',
        'status',
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)


def ordering_by_tile(df_order: pd.DataFrame) -> pd.DataFrame:
    """Per tile: products, evaluated, not evaluated (reprocessed or first), reprocessed, failures, ties (F08)."""
    reprocessed = df_order.is_reprocessed if 'is_reprocessed' in df_order.columns else False
    return (
        df_order.assign(not_evaluated=~df_order.evaluated, reprocessed=reprocessed)
        .groupby('mgrs_tile_id')
        .agg(
            n_products=('opera_id', 'size'),
            n_evaluated=('evaluated', 'sum'),
            n_not_evaluated=('not_evaluated', 'sum'),
            n_reprocessed=('reprocessed', 'sum'),
            n_failures=('out_of_order', 'sum'),
            n_ties=('tie', 'sum'),
        )
        .reset_index()
    )


def summarize_by_tile(df_products: pd.DataFrame, fail_col: str) -> pd.DataFrame:
    def _one(group: pd.DataFrame) -> pd.Series:
        bad = group[group[fail_col]]
        return pd.Series(
            {
                'n_products': len(group),
                'n_failures': len(bad),
                'ok': bad.empty,
                'bad_opera_ids': ';'.join(bad.opera_id),
            }
        )

    return df_products.groupby('mgrs_tile_id').apply(_one, include_groups=False).reset_index()


def _version_index(df_versions: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Index every version: previous acquisition per acquisition, versions per acquisition, time per version."""
    df = df_versions.sort_values(['mgrs_tile_id', 'acq_time', 'processing_time', 'opera_id'], kind='stable')
    acqs = df.drop_duplicates(subset='opera_dedup_id')
    prev = acqs.groupby('mgrs_tile_id')['opera_dedup_id'].shift(1)
    prev_acq = {k: v for k, v in zip(acqs.opera_dedup_id, prev) if isinstance(v, str)}
    versions = {k: list(g.opera_id) for k, g in df.groupby('opera_dedup_id', sort=False)}
    proc = dict(zip(df.opera_id, df.processing_time))
    return prev_acq, versions, proc


def _impossible_prior(prior: str, tile: str, acq_time: pd.Timestamp) -> str:
    """Reason a prior can never be right for a product of `tile` acquired at `acq_time` ('' if it could be)."""
    if prior == 'None':
        return ''
    tokens = prior.split('_')
    try:
        prior_tile, prior_acq = tokens[3].lstrip('T'), pd.Timestamp(tokens[4].rstrip('Z'))
    except (IndexError, ValueError):
        return f'impossible prior {prior}: not a DIST-S1 product id'
    if prior_tile != tile:
        return f'impossible prior {prior}: tile {prior_tile}, not {tile}'
    if prior_acq >= acq_time:
        return f'impossible prior {prior}: acquired at {prior_acq}, not before the product'
    return ''


def _confirm_one(b_id: str, b_proc: pd.Timestamp, prior: str, prev: str, versions: dict, proc: dict) -> tuple:
    """Status, reason and ideal prior for a product `b_id` whose previous acquisition is `prev` (decision 3)."""
    candidates = versions[prev]
    existing = [v for v in candidates if proc[v] <= b_proc]
    ideal = existing[-1] if existing else 'None'
    if prior == 'None':
        return Status.FAIL, f'no prior, but acquisition {prev} precedes the product', ideal
    if prior not in candidates:
        return Status.FAIL, f'prior {prior} is not a version of the previous acquisition {prev}', ideal
    if proc[prior] > b_proc:
        return Status.FAIL, f'prior processed after product: {prior} at {proc[prior]}, product at {b_proc}', ideal
    newer = [v for v in candidates if proc[v] > proc[prior]]
    outdated = [v for v in newer if proc[v] <= b_proc]
    if outdated:
        return Status.FAIL, f'outdated prior: newer version {outdated[-1]} was processed before the product', ideal
    if newer:
        return Status.INFO, f'stale chain: {newer[-1]} was reprocessed after the product', ideal
    return Status.PASS, '', ideal


def check_confirmation(
    df_dist_dedup: pd.DataFrame,
    df_tags: pd.DataFrame,
    expect_none_at_start: bool = False,
    df_versions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Check each product's `prior_dist_s1_product` tag against the previous acquisition in its tile (F07).

    The prior must be the latest version of the previous acquisition that existed when the product was processed,
    compared by **full** id against every version in `df_versions` (default `df_dist_dedup`; pass the
    version-filtered table before any processing filter). Processing time is the proxy for availability: a version
    existed for a product if it was processed at or before the product. For product B and previous acquisition A:

    - FAIL if the prior is not one of A's versions (or is 'None'), or was processed after B;
    - FAIL if a newer version of A was processed before B (outdated prior);
    - INFO ``stale chain`` if a newer version of A was processed only after B;
    - otherwise PASS.

    One row per product of `df_dist_dedup`. UNKNOWN when its tag could not be read; the first acquisition of a tile
    in the window is SKIPPED unless `expect_none_at_start` (then its prior must be 'None'). `confirmed_ok` is False
    for FAIL and UNKNOWN. `prior_expected` is the ideal prior (A's latest version existing at B's processing time).
    """
    df_versions = df_dist_dedup if df_versions is None else df_versions
    prev_acq, versions, proc = _version_index(pd.concat([df_versions, df_dist_dedup]).drop_duplicates('opera_id'))
    tag_cols = ['opera_id', 'prior_dist_s1_product', 'error']
    df_tags = df_tags.assign(prior_tag_present=prior_tag_present(df_tags))
    df = df_dist_dedup.merge(df_tags[[*tag_cols, 'prior_tag_present']], on='opera_id', how='left')
    df = df.sort_values(['mgrs_tile_id', 'acq_time'])

    df['tag_missing'] = df.error.isna() | (df.error != '') | df.prior_dist_s1_product.isna()
    df['tile_incomplete'] = df.groupby('mgrs_tile_id')['tag_missing'].transform('any')
    df['prior_used'] = df.prior_dist_s1_product.fillna('').map(lambda p: p.split('/')[-1] or 'None')
    df['is_first_in_window'] = ~df.opera_dedup_id.isin(prev_acq.keys())

    statuses, reasons, ideals = [], [], []
    for row in df.itertuples():
        prev = prev_acq.get(row.opera_dedup_id)
        if row.tag_missing:
            detail = row.error if isinstance(row.error, str) and row.error else 'no tag row'
            status, reason, ideal = Status.UNKNOWN, f'could not read prior_dist_s1_product tag ({detail})', 'None'
        elif not row.prior_tag_present:
            ideal = versions[prev][-1] if prev else 'None'
            status, reason = Status.FAIL, 'prior tag absent: the STATUS layer has no prior_dist_s1_product key'
        elif prev is None:
            ideal = 'None'
            impossible = _impossible_prior(row.prior_used, row.mgrs_tile_id, row.acq_time)
            if impossible:
                status, reason = Status.FAIL, impossible
            elif not expect_none_at_start:
                status, reason = Status.SKIPPED, 'first acquisition of the tile in the window'
            elif row.prior_used == 'None':
                status, reason = Status.PASS, ''
            else:
                status, reason = Status.FAIL, f'prior {row.prior_used} given, but the window starts the mission'
        else:
            status, reason, ideal = _confirm_one(
                row.opera_id, row.processing_time, row.prior_used, prev, versions, proc
            )
        statuses.append(status)
        reasons.append(reason)
        ideals.append(ideal)
    df['prior_expected'] = ideals
    df['status'] = pd.Series(statuses, index=df.index, dtype=str)
    df['reason'] = reasons
    df['confirmed_ok'] = ~df.status.isin([Status.FAIL, Status.UNKNOWN])

    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'prior_used',
        'prior_expected',
        'is_first_in_window',
        'tag_missing',
        'tile_incomplete',
        'confirmed_ok',
        'status',
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)
