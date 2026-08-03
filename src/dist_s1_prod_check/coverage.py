import numpy as np
import pandas as pd

from dist_s1_prod_check.constants import DELTA_LOOKBACK_DAYS, DELTA_WINDOW_DAYS


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def expected_pass_groups(
    df_rtc_frame: pd.DataFrame,
    start_time: str | pd.Timestamp,
    stop_time: str | pd.Timestamp,
) -> pd.DataFrame:
    """Group RTC-S1 metadata into MGRS passes; each group is a DIST-S1 product that should exist.

    Rows are prefiltered to the window (with a 1-day margin so boundary passes group correctly)
    before aggregating, so lookback-era passes are never materialized as groups.
    """
    margin = pd.Timedelta(days=1)
    start_utc = pd.Timestamp(start_time).tz_localize('UTC')
    stop_utc = pd.Timestamp(stop_time).tz_localize('UTC')
    near_window = (df_rtc_frame.acq_dt >= start_utc - margin) & (df_rtc_frame.acq_dt < stop_utc + margin)
    groups = (
        df_rtc_frame[near_window]
        .groupby(['mgrs_tile_id', 'acq_group_id_within_mgrs_tile', 'pass_id'])
        .agg(
            post_acq_dt=('acq_dt', 'min'),
            n_bursts=('jpl_burst_id', 'nunique'),
            track_token=('track_token', 'first'),
            burst_ids=('jpl_burst_id', 'unique'),
        )
        .reset_index()
    )
    in_window = (groups.post_acq_dt >= start_utc) & (groups.post_acq_dt < stop_utc)
    return groups[in_window].reset_index(drop=True)


def _baseline_burst_counts(
    acq_by_burst: dict[str, np.ndarray],
    burst_ids: list[str],
    post_acq_dt: pd.Timestamp,
    delta_lookback_days: tuple[int, ...],
    delta_window_days: int,
) -> int:
    n_with_baseline = 0
    for burst_id in burst_ids:
        acqs = acq_by_burst.get(burst_id)
        if acqs is None:
            continue
        n_pre = 0
        for delta in delta_lookback_days:
            stop = post_acq_dt - pd.Timedelta(days=delta)
            start = stop - pd.Timedelta(days=delta_window_days)
            n_pre += np.searchsorted(acqs, stop.to_datetime64()) - np.searchsorted(acqs, start.to_datetime64())
        n_with_baseline += int(n_pre > 0)
    return n_with_baseline


def check_coverage(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    df_rtc_frame: pd.DataFrame | None = None,
    tolerance_seconds: int = 3600,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
) -> pd.DataFrame:
    """Check every expected MGRS pass has a DIST-S1 product; missing passes are checked for baseline availability."""
    df_dist = df_dist_dedup.copy()
    df_dist['acq_time_utc'] = _to_utc(df_dist.acq_time)
    products_by_tile = {tile: group.reset_index(drop=True) for tile, group in df_dist.groupby('mgrs_tile_id')}

    acq_by_burst: dict[str, np.ndarray] = {}
    if df_rtc_frame is not None:
        df_bursts = df_rtc_frame[['jpl_burst_id', 'opera_id', 'acq_dt']].drop_duplicates(subset='opera_id')
        acq_by_burst = {burst: np.sort(group.acq_dt.values) for burst, group in df_bursts.groupby('jpl_burst_id')}

    records = []
    for row in df_expected_groups.itertuples():
        products = products_by_tile.get(row.mgrs_tile_id)
        matched_opera_id, time_diff_s = None, None
        if products is not None:
            diffs = (products.acq_time_utc - row.post_acq_dt).abs().dt.total_seconds()
            best = diffs.idxmin()
            if diffs[best] <= tolerance_seconds:
                matched_opera_id, time_diff_s = products.opera_id[best], diffs[best]
        product_found = matched_opera_id is not None

        record = {
            'mgrs_tile_id': row.mgrs_tile_id,
            'track_token': row.track_token,
            'post_acq_dt': row.post_acq_dt,
            'pass_id': row.pass_id,
            'n_bursts': row.n_bursts,
            'product_found': product_found,
            'matched_opera_id': matched_opera_id or '',
            'time_diff_seconds': time_diff_s,
            'n_bursts_with_baseline': None,
            'missing_product_expected': False,
            'reason': '',
        }
        if not product_found:
            if acq_by_burst:
                n_baseline = _baseline_burst_counts(
                    acq_by_burst, list(row.burst_ids), row.post_acq_dt, delta_lookback_days, delta_window_days
                )
                record['n_bursts_with_baseline'] = n_baseline
                record['missing_product_expected'] = n_baseline > 0
                detail = (
                    f'{n_baseline}/{row.n_bursts} bursts have baseline imagery'
                    if n_baseline > 0
                    else 'no baseline imagery; product not expected'
                )
            else:
                record['missing_product_expected'] = True
                detail = 'baseline not checked (no lookback RTC metadata)'
            record['reason'] = f'no DIST-S1 product for RTC pass at {row.post_acq_dt} ({detail})'
        records.append(record)
    return pd.DataFrame(records)


def check_orphan_products(
    df_expected_groups: pd.DataFrame,
    df_dist_dedup: pd.DataFrame,
    tolerance_seconds: int = 3600,
) -> pd.DataFrame:
    """Find DIST-S1 products with no corresponding RTC-S1 pass in the expected groups."""
    df_dist = df_dist_dedup.copy()
    df_dist['acq_time_utc'] = _to_utc(df_dist.acq_time)
    groups_by_tile = {
        tile: group.post_acq_dt.sort_values().reset_index(drop=True)
        for tile, group in df_expected_groups.groupby('mgrs_tile_id')
    }

    records = []
    for row in df_dist.itertuples():
        passes = groups_by_tile.get(row.mgrs_tile_id)
        min_diff = None if passes is None else (passes - row.acq_time_utc).abs().dt.total_seconds().min()
        if min_diff is None or min_diff > tolerance_seconds:
            records.append(
                {
                    'opera_id': row.opera_id,
                    'mgrs_tile_id': row.mgrs_tile_id,
                    'acq_time': row.acq_time,
                    'nearest_pass_diff_seconds': min_diff,
                    'reason': 'DIST-S1 product exists but no matching RTC pass found in expected groups',
                }
            )
    return pd.DataFrame(records)
