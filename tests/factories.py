"""Synthetic DIST-S1 / RTC-S1 / tag tables for offline tests.

The frames built here mirror the caches written by `download-metadata` and `fetch-tags`
(`cmr.get_dist_s1_table`, `cmr.get_rtc_s1_table`, `tags.fetch_tags_table`): same columns, column
order and dtypes. Geometries come from the MGRS tile and burst tables bundled with
`dist_s1_enumerator`, so nothing here touches the network.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from dist_s1_enumerator.mgrs_burst_data import get_burst_table, get_lut_by_mgrs_tile_ids, get_mgrs_table

from dist_s1_prod_check import cmr, manifest
from dist_s1_prod_check.constants import (
    DIST_S1_PARQUET,
    DIST_S1_S3_PREFIX,
    DIST_S1_S3_REGION,
    LAYER_URL_MAP,
    RTC_S1_PARQUET,
    TAGS_PARQUET,
    TARGET_TILES_FILE,
    VENUE_FILE,
)
from dist_s1_prod_check.ids import get_opera_id_trunc, get_track_number
from dist_s1_prod_check.tags import TAG_KEYS, resolve_tag_urls


TimeLike = str | datetime | pd.Timestamp

DIST_COLUMNS = [
    'mgrs_tile_id',
    'geometry',
    'opera_id',
    'acq_time',
    'processing_time',
    'sat',
    'alert_url',
    'metric_url',
    'alert_acq_url',
    'opera_dedup_id',
]
RTC_COLUMNS = [
    'jpl_burst_id',
    'geometry',
    'opera_id',
    'acq_dt',
    'processing_dt',
    'track_number',
    'polarizations',
    'url_copol',
    'url_crosspol',
    'opera_dedup_id',
]
TAG_COLUMNS = ['opera_id', 'error', *TAG_KEYS]

DIST_BUCKET = 'opera-test-bucket'
# Acquisition window the seeded manifests record by default; every synthetic run requests a window inside it.
DEFAULT_WINDOW = ('2026-01-01', '2026-03-01')
RTC_URL_BASE = 'https://datapool.asf.alaska.edu/RTC/OPERA-S1/'
_POL_SUFFIXES = {'VV+VH': ('VV', 'VH'), 'HH+HV': ('HH', 'HV'), 'VV': ('VV', None), 'HH': ('HH', None)}


def _utc_naive(t: TimeLike) -> pd.Timestamp:
    """Timestamp in UTC with the timezone dropped, floored to whole seconds (id resolution)."""
    ts = pd.Timestamp(t)
    if ts.tz is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts.floor('s')


def _fmt_id_time(t: TimeLike) -> str:
    return _utc_naive(t).strftime('%Y%m%dT%H%M%SZ')


def dist_id(tile: str, acq: TimeLike, proc: TimeLike, sensor: str = 'S1A', version: str = 'v1.0') -> str:
    """Build a DIST-S1 product id.

    Parameters
    ----------
    tile : str
        MGRS tile id, with or without the leading ``T`` (``'10SGC'`` or ``'T10SGC'``).
    acq, proc : str, datetime or pd.Timestamp
        Acquisition and processing times. Naive values are taken as UTC; aware ones are converted.
    sensor : str
        Platform token, e.g. ``'S1A'``, ``'S1C'``.
    version : str
        Product version token.

    Returns
    -------
    str
        e.g. ``OPERA_L3_DIST-ALERT-S1_T10SGC_20260102T020653Z_20260507T233527Z_S1A_30_v1.0``.
    """
    tile = tile.removeprefix('T')
    return f'OPERA_L3_DIST-ALERT-S1_T{tile}_{_fmt_id_time(acq)}_{_fmt_id_time(proc)}_{sensor}_30_{version}'


def rtc_id(burst: str, acq: TimeLike, proc: TimeLike, sensor: str = 'S1A') -> str:
    """Build an RTC-S1 product id.

    Parameters
    ----------
    burst : str
        JPL burst id, e.g. ``'T035-073226-IW3'``.
    acq, proc : str, datetime or pd.Timestamp
        Acquisition and processing times. Naive values are taken as UTC; aware ones are converted.
    sensor : str
        Platform token.

    Returns
    -------
    str
        e.g. ``OPERA_L2_RTC-S1_T035-073226-IW3_20260102T020653Z_20260102T074151Z_S1A_30_v1.0``.
    """
    return f'OPERA_L2_RTC-S1_{burst}_{_fmt_id_time(acq)}_{_fmt_id_time(proc)}_{sensor}_30_v1.0'


def _dist_urls(opera_id: str, acq: pd.Timestamp) -> dict[str, str]:
    base = f'https://{DIST_BUCKET}.s3.{DIST_S1_S3_REGION}.amazonaws.com/{DIST_S1_S3_PREFIX}/{acq:%Y/%m/%d}/'
    return {col: f'{base}{opera_id}/{opera_id}_{suffix}.tif' for col, suffix in LAYER_URL_MAP.items()}


def make_dist(rows: Iterable[Mapping[str, Any]]) -> gpd.GeoDataFrame:
    """DIST-S1 product table shaped like `cmr.get_dist_s1_table` output.

    Parameters
    ----------
    rows : iterable of mapping
        Each row gives either ``opera_id`` or ``tile``, ``acq``, ``proc`` (optional ``sensor``,
        ``version``). Any of ``alert_url``, ``metric_url``, ``alert_acq_url`` may be given to override
        the generated URL (``None`` is kept as a null).

    Returns
    -------
    gpd.GeoDataFrame
        Columns `DIST_COLUMNS`, EPSG:4326, MGRS tile footprints from the enumerator's bundled table
        (``None`` for unknown tiles), sorted by tile and acquisition time. Times are tz-naive UTC.
        Rows are not deduplicated, so reprocessed versions can be represented.
    """
    records = []
    for row in rows:
        opera_id = row.get('opera_id') or dist_id(
            row['tile'], row['acq'], row['proc'], row.get('sensor', 'S1A'), row.get('version', 'v1.0')
        )
        tokens = opera_id.split('_')
        acq_time = pd.Timestamp(tokens[4].rstrip('Z'))
        urls = _dist_urls(opera_id, acq_time)
        urls.update({col: row[col] for col in LAYER_URL_MAP if col in row})
        records.append(
            {
                'opera_id': opera_id,
                'mgrs_tile_id': tokens[3].lstrip('T'),
                'acq_time': acq_time,
                'processing_time': pd.Timestamp(tokens[5].rstrip('Z')),
                'sat': tokens[6],
                **urls,
            }
        )
    df = pd.DataFrame(records, columns=[c for c in DIST_COLUMNS if c not in ('geometry', 'opera_dedup_id')])
    df['acq_time'] = df.acq_time.astype('datetime64[us]')
    df['processing_time'] = df.processing_time.astype('datetime64[us]')
    df['opera_dedup_id'] = df.opera_id.map(get_opera_id_trunc).astype(df.opera_id.dtype)
    tile_geoms = get_mgrs_table()[['mgrs_tile_id', 'geometry']]
    df = tile_geoms.merge(df, on='mgrs_tile_id', how='right')
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)[DIST_COLUMNS]


def make_rtc(rows: Iterable[Mapping[str, Any]]) -> gpd.GeoDataFrame:
    """RTC-S1 granule table shaped like `cmr.get_rtc_s1_table` output.

    Parameters
    ----------
    rows : iterable of mapping
        Each row gives either ``opera_id`` or ``burst``, ``acq``, ``proc`` (optional ``sensor``).
        Optional ``polarizations`` (default ``'VV+VH'``), ``url_copol``, ``url_crosspol``.

    Returns
    -------
    gpd.GeoDataFrame
        Columns `RTC_COLUMNS`, EPSG:4326, burst footprints from the enumerator's bundled table,
        ``acq_dt``/``processing_dt`` tz-aware UTC, sorted by burst and acquisition time. Reprocessed
        duplicates are kept (the real table drops all but the last-processed one).
    """
    records = []
    for row in rows:
        opera_id = row.get('opera_id') or rtc_id(row['burst'], row['acq'], row['proc'], row.get('sensor', 'S1A'))
        tokens = opera_id.split('_')
        burst = tokens[3]
        pols = row.get('polarizations', 'VV+VH')
        co, cross = _POL_SUFFIXES.get(pols, (None, None))
        records.append(
            {
                'opera_id': opera_id,
                'jpl_burst_id': burst,
                'acq_dt': pd.Timestamp(tokens[4].rstrip('Z')).tz_localize('UTC'),
                'processing_dt': pd.Timestamp(tokens[5].rstrip('Z')).tz_localize('UTC'),
                'track_number': get_track_number(burst),
                'polarizations': pols,
                'url_copol': row.get('url_copol', f'{RTC_URL_BASE}{opera_id}_{co}.tif' if co else ''),
                'url_crosspol': row.get('url_crosspol', f'{RTC_URL_BASE}{opera_id}_{cross}.tif' if cross else ''),
            }
        )
    cols = [c for c in RTC_COLUMNS if c not in ('geometry', 'opera_dedup_id')]
    df = pd.DataFrame(records, columns=cols)
    df['acq_dt'] = df.acq_dt.astype('datetime64[us, UTC]')
    df['processing_dt'] = df.processing_dt.astype('datetime64[us, UTC]')
    df['track_number'] = df.track_number.astype('int64')
    df['opera_dedup_id'] = df.opera_id.map(get_opera_id_trunc).astype(df.opera_id.dtype)
    burst_geoms = get_burst_table(sorted(df.jpl_burst_id.unique()))[['jpl_burst_id', 'geometry']]
    df = burst_geoms.merge(df, on='jpl_burst_id', how='right')
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return df.sort_values(['jpl_burst_id', 'acq_dt']).reset_index(drop=True)[RTC_COLUMNS]


def _join_ids(value: str | Iterable[str] | None) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return ','.join(value)


def make_tags(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Tag table shaped like `tags.fetch_tags_table` output (rows of `tags._fetch_one`).

    Parameters
    ----------
    rows : iterable of mapping
        Each row gives ``opera_id`` and optionally ``pre_rtc_opera_ids``/``post_rtc_opera_ids`` (a
        comma-joined string or a list of RTC ids), ``prior_dist_s1_product`` (default ``'None'``, the
        literal string real products carry when there is no prior), ``mgrs_tile_id`` (default: from
        ``opera_id``) and ``error`` (default ``''``). A row with a non-empty ``error`` has every tag
        key set to ``''``, as `_fetch_one` does on a read failure, unless given explicitly.

    Returns
    -------
    pd.DataFrame
        Columns `TAG_COLUMNS`, all strings.
    """
    records = []
    for row in rows:
        opera_id = row['opera_id']
        error = row.get('error', '')
        if error:
            rec = dict.fromkeys(TAG_KEYS, '')
        else:
            rec = {
                'mgrs_tile_id': opera_id.split('_')[3].lstrip('T'),
                'pre_rtc_opera_ids': '',
                'post_rtc_opera_ids': '',
                'prior_dist_s1_product': 'None',
            }
        for key in TAG_KEYS:
            if key in row:
                rec[key] = _join_ids(row[key])
        records.append({'opera_id': opera_id, 'error': error, **rec})
    return pd.DataFrame(records, columns=TAG_COLUMNS).astype('str')


def bursts_for_tile(tile: str) -> pd.DataFrame:
    """Burst-to-tile lookup rows for one MGRS tile, from the enumerator's bundled LUT.

    Parameters
    ----------
    tile : str
        MGRS tile id (leading ``T`` optional).

    Returns
    -------
    pd.DataFrame
        Columns ``jpl_burst_id``, ``mgrs_tile_id``, ``track_number``, ``acq_group_id_within_mgrs_tile``,
        ``orbit_pass`` and more; one row per burst feeding the tile. Use it to pick the bursts of
        one acquisition group when building RTC rows for coverage/inputs tests.
    """
    return get_lut_by_mgrs_tile_ids([tile.removeprefix('T')]).reset_index(drop=True)


def run_dir(out_dir: Path) -> Path:
    """Return the newest run directory a CLI command wrote under ``--out-dir`` (F34), else ``out_dir``."""
    from dist_s1_prod_check.cli import latest_run_directory

    return latest_run_directory(Path(out_dir))


def seed_data_dir(
    data_dir: Path,
    dist: gpd.GeoDataFrame | None = None,
    rtc: gpd.GeoDataFrame | None = None,
    tags: pd.DataFrame | None = None,
    venue: str = 'PROD',
    target_tiles: Iterable[str] | None = None,
    window: tuple[str, str] = DEFAULT_WINDOW,
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Write cache files into ``data_dir`` so CLI commands run offline.

    The DIST-S1 and RTC-S1 tables get ``query_receipt`` manifests, as if ``run-all`` had just queried `window`
    (main and lookback windows, all complete) over `bbox`, so they pass the cache-compatibility checks.

    Parameters
    ----------
    data_dir : Path
        Directory to populate (created if needed); pass it as ``--data-dir``.
    dist, rtc, tags : frame or None
        Tables from `make_dist`, `make_rtc`, `make_tags`; ``None`` skips that file.
    venue : str
        Written to ``venue.txt``, which `cli._check_venue` compares against ``--venue``.
    target_tiles : iterable of str or None
        If given, written to ``target_mgrs_tiles.txt`` (what ``--bbox`` would have produced); it limits
        the MGRS tiles `build_rtc_input_frame` expands RTC rows to.
    window : (start, stop)
        Acquisition window the manifests record; a run must request a window inside it.
    bbox : tuple or None
        AOI the manifests record; a run must pass the same ``--bbox``.

    Returns
    -------
    Path
        ``data_dir``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if target_tiles is not None:
        (data_dir / TARGET_TILES_FILE).write_text('\n'.join(t.removeprefix('T') for t in target_tiles))
    tiles = (data_dir / TARGET_TILES_FILE).read_text().split() if (data_dir / TARGET_TILES_FILE).exists() else None
    start, stop = window
    common = {'start': start, 'stop': stop, 'bbox': bbox, 'target_tiles': tiles}
    if dist is not None:
        dist.to_parquet(data_dir / DIST_S1_PARQUET, compression='zstd')
        manifest.write_table_manifest(
            data_dir / DIST_S1_PARQUET,
            table='dist_s1',
            venue=venue,
            source=f'{venue} (synthetic)',
            lookback_windows=manifest.windows_with_counts([(start, stop)], dist.acq_time),
            **common,
        )
    if rtc is not None:
        rtc.to_parquet(data_dir / RTC_S1_PARQUET, compression='zstd')
        manifest.write_table_manifest(
            data_dir / RTC_S1_PARQUET,
            table='rtc_s1',
            venue='PROD',
            source='PROD (synthetic)',
            lookback_windows=manifest.windows_with_counts(cmr.lookback_date_ranges(start, stop), rtc.acq_dt),
            **common,
        )
    if tags is not None:
        if 'source_url' not in tags.columns and (data_dir / DIST_S1_PARQUET).exists():
            # As if fetched from each product's alert_url, so run-all reuses the rows instead of refetching (F23).
            df_dist = pd.read_parquet(data_dir / DIST_S1_PARQUET)
            url_by_id = dict(zip(df_dist.opera_id, resolve_tag_urls(df_dist)))
            tags = tags.assign(source_url=tags.opera_id.map(url_by_id).fillna(''))
        tags.to_parquet(data_dir / TAGS_PARQUET, compression='zstd')
    (data_dir / VENUE_FILE).write_text(f'{venue}\n')
    return data_dir
