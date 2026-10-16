import numpy as np
import pandas as pd
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids

from dist_s1_prod_check import matching
from dist_s1_prod_check.constants import (
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    DUAL_POLARIZATIONS,
    PASS_MATCH_TOLERANCE_SECONDS,
)
from dist_s1_prod_check.ids import get_burst_id
from dist_s1_prod_check.manifest import QueryWindow, lookback_complete
from dist_s1_prod_check.status import Status


COVERAGE_COLUMNS = [
    'mgrs_tile_id',
    'track_token',
    'post_acq_dt',
    'pass_id',
    'n_bursts',
    'product_found',
    'matched_opera_id',
    'time_diff_seconds',
    'n_bursts_with_baseline',
    'missing_product_expected',
    'status',
    'reason',
    'acq_group_id_within_mgrs_tile',
    'eligible',
]
ORPHAN_COLUMNS = ['opera_id', 'mgrs_tile_id', 'acq_time', 'nearest_pass_diff_seconds', 'reason']


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _burst_pol_key(df: pd.DataFrame) -> pd.Series:
    return df.jpl_burst_id + '|' + df.polarizations


def _dual_pol_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.polarizations.isin(DUAL_POLARIZATIONS)]


def expected_pass_groups(
    df_rtc_frame: pd.DataFrame,
    start_time: str | pd.Timestamp,
    stop_time: str | pd.Timestamp,
) -> pd.DataFrame:
    """Group RTC-S1 metadata into MGRS passes; each group is a DIST-S1 product that should exist.

    Rows are prefiltered to the window (with a 1-day margin so boundary passes group correctly)
    before aggregating, so lookback-era passes are never materialized as groups. Single-polarization
    granules are excluded: they are never DIST-S1 post-images.

    Each group carries `burst_pols`, the post-image `burst_id|polarization` keys, so a burst's
    baseline can only be drawn from acquisitions of that same polarization.
    """
    margin = pd.Timedelta(days=1)
    start_utc = pd.Timestamp(start_time).tz_localize('UTC')
    stop_utc = pd.Timestamp(stop_time).tz_localize('UTC')
    near_window = (df_rtc_frame.acq_dt >= start_utc - margin) & (df_rtc_frame.acq_dt < stop_utc + margin)
    df_post = _dual_pol_only(df_rtc_frame[near_window])
    groups = (
        df_post.assign(burst_pol=_burst_pol_key(df_post))
        .groupby(['mgrs_tile_id', 'acq_group_id_within_mgrs_tile', 'pass_id'])
        .agg(
            post_acq_dt=('acq_dt', 'min'),
            n_bursts=('jpl_burst_id', 'nunique'),
            track_token=('track_token', 'first'),
            burst_ids=('jpl_burst_id', 'unique'),
            burst_pols=('burst_pol', 'unique'),
        )
        .reset_index()
    )
    in_window = (groups.post_acq_dt >= start_utc) & (groups.post_acq_dt < stop_utc)
    return groups[in_window].reset_index(drop=True)


def _baseline_burst_counts(
    acq_by_burst_pol: dict[str, np.ndarray],
    burst_pols: list[str],
    post_acq_dt: pd.Timestamp,
    delta_lookback_days: tuple[int, ...],
    delta_window_days: int,
) -> int:
    """Count post-image bursts with at least one baseline acquisition of the *same* polarization."""
    n_with_baseline = 0
    for burst_pol in burst_pols:
        acqs = acq_by_burst_pol.get(burst_pol)
        if acqs is None:
            continue
        n_pre = 0
        for delta in delta_lookback_days:
            stop = post_acq_dt - pd.Timedelta(days=delta)
            start = stop - pd.Timedelta(days=delta_window_days)
            n_pre += np.searchsorted(acqs, stop.to_datetime64()) - np.searchsorted(acqs, start.to_datetime64())
        n_with_baseline += int(n_pre > 0)
    return n_with_baseline


def _unreadable_products(df_tags: pd.DataFrame | None) -> dict[str, str]:
    if df_tags is None or df_tags.empty:
        return {}
    errors = df_tags.drop_duplicates(subset='opera_id', keep='last').set_index('opera_id').error.fillna('')
    return errors[errors != ''].to_dict()


def _tagged_post_groups(
    df_tags: pd.DataFrame | None, product_to_pass: dict[str, matching.PassKey]
) -> dict[str, set[int | None]]:
    """Acquisition groups of each matched product's tagged post-image bursts, from the enumerator's burst LUT.

    A burst that does not feed the product's tile maps to `None`. Products without readable post tags are omitted.
    """
    if df_tags is None or df_tags.empty or not product_to_pass:
        return {}
    df = df_tags.drop_duplicates(subset='opera_id', keep='last')
    df = df[df.opera_id.isin(product_to_pass.keys()) & (df.post_rtc_opera_ids.fillna('') != '')]
    if df.empty:
        return {}
    tile_of = {opera_id: key[0] for opera_id, key in product_to_pass.items()}
    df_lut = get_lut_by_mgrs_tile_ids(sorted({tile_of[o] for o in df.opera_id}))
    group_of = dict(
        zip(zip(df_lut.mgrs_tile_id, df_lut.jpl_burst_id), df_lut.acq_group_id_within_mgrs_tile.astype(int))
    )
    return {
        row.opera_id: {
            group_of.get((tile_of[row.opera_id], get_burst_id(rtc_id))) for rtc_id in row.post_rtc_opera_ids.split(',')
        }
        for row in df.itertuples()
        if row.post_rtc_opera_ids
    }


def _burst_acq_index(df_rtc_frame: pd.DataFrame | None) -> dict[str, np.ndarray]:
    if df_rtc_frame is None:
        return {}
    cols = ['jpl_burst_id', 'polarizations', 'opera_id', 'acq_dt']
    df_bursts = _dual_pol_only(df_rtc_frame[cols]).drop_duplicates(subset='opera_id')
    return {
        burst_pol: np.sort(group.acq_dt.values) for burst_pol, group in df_bursts.groupby(_burst_pol_key(df_bursts))
    }


def check_coverage(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    df_rtc_frame: pd.DataFrame | None = None,
    tolerance_seconds: float = PASS_MATCH_TOLERANCE_SECONDS,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
    query_windows: list[QueryWindow] | None = None,
    df_tags: pd.DataFrame | None = None,
    assignment: matching.PassAssignment | None = None,
) -> pd.DataFrame:
    """Check every expected MGRS pass has a DIST-S1 product; missing passes are checked for baseline availability.

    Passes and products are matched one-to-one per tile by `matching.assign_passes` (computed here unless the
    caller passes the `assignment` it also gives `check_orphan_products`). A pass that could equally well be matched
    to another product, or to none, is UNKNOWN "ambiguous match".

    Baseline eligibility (`n_bursts_with_baseline`, `eligible`) is evaluated for every pass, found or not. A burst
    only counts as having a baseline when the lookback windows hold acquisitions of the same dual polarization as the
    post-image, matching how the enumerator builds a baseline.

    `query_windows` are the RTC-S1 query windows from the table's manifest. A missing pass is judged only if
    every baseline window it needs was queried to completion (`manifest.lookback_complete`); otherwise its
    status is UNKNOWN ("lookback incomplete"), because an unqueried window is not evidence of "no baseline".
    Without `query_windows` nothing is known to be complete. Status per pass: PASS (product found), FAIL
    (missing, baseline available), SKIPPED (missing, no baseline: not expected), UNKNOWN (lookback incomplete).

    The product under test is always the latest version (`df_dist_dedup`), never an older readable copy. With
    `df_tags`, a pass whose matched product has a tag error (its STATUS layer could not be read, or it has no
    STATUS layer URL) is UNKNOWN "latest product unreadable" (F14): it exists but its delivery is not shown. A
    matched product whose tagged post-image bursts belong to a different acquisition group of the tile than the
    pass is UNKNOWN "group mismatch". The group is a consistency check only; it is not used for matching.
    """
    if assignment is None:
        assignment = matching.assign_passes(df_expected_groups, df_dist_dedup, tolerance_seconds=tolerance_seconds)
    acq_by_burst_pol = _burst_acq_index(df_rtc_frame)
    unreadable = _unreadable_products(df_tags)
    post_groups = _tagged_post_groups(df_tags, assignment.product_to_pass)

    records = []
    for row in df_expected_groups.itertuples():
        key = matching.pass_key(row)
        matched_opera_id = assignment.pass_to_product.get(key)
        product_found = matched_opera_id is not None
        n_baseline = _baseline_burst_counts(
            acq_by_burst_pol, list(row.burst_pols), row.post_acq_dt, delta_lookback_days, delta_window_days
        )
        record = {
            'mgrs_tile_id': row.mgrs_tile_id,
            'track_token': row.track_token,
            'post_acq_dt': row.post_acq_dt,
            'pass_id': row.pass_id,
            'n_bursts': row.n_bursts,
            'product_found': product_found,
            'matched_opera_id': matched_opera_id or '',
            'time_diff_seconds': assignment.time_diff_seconds.get(key),
            'n_bursts_with_baseline': n_baseline,
            'missing_product_expected': False,
            'status': Status.PASS,
            'reason': '',
            'acq_group_id_within_mgrs_tile': key[1],
            'eligible': n_baseline > 0,
        }
        groups = post_groups.get(matched_opera_id, {key[1]})
        if key in assignment.ambiguous:
            record['status'] = Status.UNKNOWN
            candidates = ', '.join(assignment.ambiguous[key])
            record['reason'] = f'ambiguous match: RTC pass at {row.post_acq_dt} matches equally well: {candidates}'
        elif product_found and matched_opera_id in unreadable:
            record['status'] = Status.UNKNOWN
            record['reason'] = f'latest product unreadable: {matched_opera_id} ({unreadable[matched_opera_id]})'
        elif product_found and groups != {key[1]}:
            record['status'] = Status.UNKNOWN
            tagged = ', '.join('not in tile' if g is None else str(g) for g in sorted(groups, key=str))
            record['reason'] = (
                f'group mismatch: {matched_opera_id} matched to acquisition group {key[1]} but its tagged post-image '
                f'bursts are in group(s) {tagged}'
            )
        elif not product_found:
            if not lookback_complete(query_windows, row.post_acq_dt, delta_lookback_days, delta_window_days):
                record['status'] = Status.UNKNOWN
                detail = 'lookback incomplete: a required RTC-S1 baseline window was not queried to completion'
            else:
                record['missing_product_expected'] = n_baseline > 0
                record['status'] = Status.FAIL if n_baseline > 0 else Status.SKIPPED
                detail = (
                    f'{n_baseline}/{row.n_bursts} bursts have baseline imagery'
                    if n_baseline > 0
                    else 'no baseline imagery; product not expected'
                )
            record['reason'] = f'no DIST-S1 product for RTC pass at {row.post_acq_dt} ({detail})'
        records.append(record)
    return pd.DataFrame(records, columns=COVERAGE_COLUMNS)


def check_orphan_products(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    tolerance_seconds: float = PASS_MATCH_TOLERANCE_SECONDS,
    assignment: matching.PassAssignment | None = None,
) -> pd.DataFrame:
    """Find DIST-S1 products left without an RTC-S1 pass by the one-to-one assignment (`matching.assign_passes`).

    Pass the same `assignment` given to `check_coverage` so every product is either matched or an orphan.
    `nearest_pass_diff_seconds` is the distance to the nearest pass of the tile, matched or not.
    """
    if assignment is None:
        assignment = matching.assign_passes(df_expected_groups, df_dist_dedup, tolerance_seconds=tolerance_seconds)
    orphans = set(assignment.orphan_products)
    df_dist = df_dist_dedup[df_dist_dedup.opera_id.isin(orphans)] if len(df_dist_dedup) else df_dist_dedup
    groups_by_tile = {}
    if len(df_expected_groups):
        groups_by_tile = {
            tile: pd.to_datetime(group.post_acq_dt, utc=True).reset_index(drop=True)
            for tile, group in df_expected_groups.groupby('mgrs_tile_id')
        }

    records = []
    for row in df_dist.itertuples():
        passes = groups_by_tile.get(row.mgrs_tile_id)
        acq = pd.Timestamp(row.acq_time)
        acq = acq.tz_localize('UTC') if acq.tzinfo is None else acq.tz_convert('UTC')
        min_diff = None if passes is None else (passes - acq).abs().dt.total_seconds().min()
        reason = (
            'DIST-S1 product exists but no matching RTC pass found in expected groups'
            if min_diff is None or min_diff > tolerance_seconds
            else 'DIST-S1 product within tolerance of an RTC pass that was matched to another product'
        )
        records.append(
            {
                'opera_id': row.opera_id,
                'mgrs_tile_id': row.mgrs_tile_id,
                'acq_time': row.acq_time,
                'nearest_pass_diff_seconds': min_diff,
                'reason': reason,
            }
        )
    return pd.DataFrame(records, columns=ORPHAN_COLUMNS)
