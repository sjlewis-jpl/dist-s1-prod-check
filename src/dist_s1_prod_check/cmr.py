from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from warnings import warn
from xml.etree import ElementTree

import earthaccess
import geopandas as gpd
import pandas as pd
import requests
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id, format_polarization
from dist_s1_enumerator.mgrs_burst_data import (
    get_burst_table,
    get_lut_by_mgrs_tile_ids,
    get_mgrs_burst_lut,
    get_mgrs_table,
    get_mgrs_tiles_overlapping_geometry,
)
from shapely.geometry import box
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from dist_s1_prod_check.constants import (
    CMR_GRANULES_URL,
    CMR_GRANULES_URLS,
    CMR_PAGE_SIZE,
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    DIST_S1_CONCEPT_IDS,
    DIST_S1_S3_BUCKETS,
    DIST_S1_S3_PREFIX,
    DIST_S1_S3_REGION,
    DUAL_POLARIZATIONS,
    LAYER_URL_MAP,
    RTC_S1_CONCEPT_ID,
)
from dist_s1_prod_check.ids import get_opera_id_trunc, get_track_number
from dist_s1_prod_check.network import OfflineError, network_allowed


DateLike = str | datetime | pd.Timestamp


def get_edl_token(venue: str = 'PROD') -> str:
    auth = earthaccess.login(system=earthaccess.UAT if venue == 'UAT' else earthaccess.PROD)
    assert auth.authenticated, 'Earthdata login failed; check ~/.netrc or EARTHDATA_* env vars.'
    return auth.token['access_token']


def _utc_naive(t: DateLike) -> pd.Timestamp:
    """Timestamp in UTC without an offset; a naive input is taken to be UTC already."""
    ts = pd.Timestamp(t)
    return ts if ts.tz is None else ts.tz_convert('UTC').tz_localize(None)


def _fmt_dt(t: DateLike) -> str:
    ts = pd.Timestamp(t)
    ts = ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')
    return ts.strftime('%Y-%m-%dT%H:%M:%SZ')


class IncompleteResultsError(RuntimeError):
    """A CMR search did not return every granule it reported (or its completeness cannot be established)."""


class GranuleList(list):
    """Granules of one CMR search, with the `hits` CMR reported for it."""

    hits: int | None = None


@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
def _get_page(
    params: dict, headers: dict, cmr_url: str = CMR_GRANULES_URL
) -> tuple[list[dict], str | None, int | None]:
    """One page of a CMR granule search: its items, the search-after token and the hits CMR reports.

    Raises `IncompleteResultsError` when the response has no `items` list: a malformed or truncated response must
    not read as an empty last page (F15).
    """
    resp = requests.get(cmr_url, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get('items') if isinstance(payload, dict) else None
    if not isinstance(items, list):
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise IncompleteResultsError(f'CMR response from {cmr_url} has no items list (got {keys})')
    hits = resp.headers.get('CMR-Hits', payload.get('hits'))
    return items, resp.headers.get('CMR-Search-After'), None if hits is None else int(hits)


def search_granules(
    collection_concept_id: str,
    start_time: DateLike,
    stop_time: DateLike,
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    granule_name_pattern: str | None = None,
    parse_fn: Callable[[dict], dict] | None = None,
    cmr_url: str = CMR_GRANULES_URL,
) -> list[dict]:
    """Search CMR granules; with `parse_fn`, each page is parsed as it arrives so raw UMM JSON is never accumulated.

    Returns a `GranuleList` whose `hits` is the count CMR reported. Raises `IncompleteResultsError` if CMR reported
    no count, or if fewer granules arrived than it reported (F15). More is accepted: granules ingested while paging.
    A hits match proves the query was paged to the end, not that the archive itself is complete.
    """
    network_allowed('a CMR granule search (metadata download or chunk re-query)')
    params: dict = {
        'collection_concept_id': collection_concept_id,
        'temporal': f'{_fmt_dt(start_time)},{_fmt_dt(stop_time)}',
        'page_size': CMR_PAGE_SIZE,
    }
    if bbox is not None:
        params['bounding_box'] = ','.join(map(str, bbox))
    if granule_name_pattern is not None:
        params['readable_granule_name'] = granule_name_pattern
        params['options[readable_granule_name][pattern]'] = 'true'

    headers = {'Authorization': f'Bearer {token}'} if token else {}
    items = GranuleList()
    search_after = None
    while True:
        page_headers = headers | ({'CMR-Search-After': search_after} if search_after else {})
        page_items, search_after, page_hits = _get_page(params, page_headers, cmr_url)
        items.hits = page_hits if items.hits is None else items.hits
        items.extend(map(parse_fn, page_items) if parse_fn else page_items)
        if not page_items or search_after is None:
            break
    window = f'{params["temporal"]} of {collection_concept_id}'
    if items.hits is None:
        raise IncompleteResultsError(f'CMR reported no hit count for {window}; completeness is unknown')
    if len(items) < items.hits:
        raise IncompleteResultsError(f'CMR returned {len(items)} of {items.hits} granules for {window}')
    return items


def _date_chunks(start_time: DateLike, stop_time: DateLike, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, stop = pd.Timestamp(start_time), pd.Timestamp(stop_time)
    edges = pd.date_range(start, stop, freq=f'{chunk_days}D')
    if len(edges) == 0 or edges[-1] < stop:
        edges = edges.append(pd.DatetimeIndex([stop]))
    return list(zip(edges[:-1], edges[1:]))


def search_granules_chunked(
    collection_concept_id: str,
    start_time: DateLike,
    stop_time: DateLike,
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_days: int = 3,
    max_workers: int = 8,
    desc: str = 'CMR chunks',
    parse_fn: Callable[[dict], dict] | None = None,
    cmr_url: str = CMR_GRANULES_URL,
) -> list[dict]:
    """Search CMR granules over a time range by splitting into temporal chunks queried concurrently."""
    chunks = _date_chunks(start_time, stop_time, chunk_days)

    def _one(chunk: tuple[pd.Timestamp, pd.Timestamp]) -> list[dict]:
        return search_granules(
            collection_concept_id, chunk[0], chunk[1], token=token, bbox=bbox, parse_fn=parse_fn, cmr_url=cmr_url
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        with tqdm(total=len(chunks), desc=desc, unit='chunk') as pbar:
            results = []
            for chunk_items in pool.map(_one, chunks):
                results.append(chunk_items)
                pbar.update(1)
                pbar.set_postfix(granules=sum(len(r) for r in results))
    return [item for chunk_items in results for item in chunk_items]


def _search_chunked_to_df(
    collection_concept_id: str,
    start_time: DateLike,
    stop_time: DateLike,
    parse_fn: Callable[[dict], dict],
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_days: int = 3,
    max_workers: int = 8,
    desc: str = 'CMR chunks',
    cmr_url: str = CMR_GRANULES_URL,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Chunked CMR search where each chunk is converted to a compact DataFrame as it completes.

    Peak memory is bounded by the final concatenated table plus one in-flight chunk per worker,
    never the whole corpus as python dicts. With `cache_dir`, each chunk is checkpointed to parquet
    and reused on rerun, so an interrupted download resumes instead of starting over.
    """
    chunks = _date_chunks(start_time, stop_time, chunk_days)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _one(chunk: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
        path = None
        if cache_dir is not None:
            path = cache_dir / f'{collection_concept_id}_{chunk[0]:%Y%m%dT%H%M%S}_{chunk[1]:%Y%m%dT%H%M%S}.parquet'
            if path.exists():
                return pd.read_parquet(path)
        records = search_granules(
            collection_concept_id, chunk[0], chunk[1], token=token, bbox=bbox, parse_fn=parse_fn, cmr_url=cmr_url
        )
        df_chunk = pd.DataFrame(records)
        if path is not None:
            tmp = path.with_suffix('.tmp')
            df_chunk.to_parquet(tmp, compression='zstd')
            tmp.replace(path)
        return df_chunk

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        with tqdm(total=len(chunks), desc=desc, unit='chunk') as pbar:
            frames = []
            n_granules = 0
            for df_chunk in pool.map(_one, chunks):
                if not df_chunk.empty:
                    frames.append(df_chunk)
                    n_granules += len(df_chunk)
                pbar.update(1)
                pbar.set_postfix(granules=n_granules)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def resolve_target_tiles(
    bbox: tuple[float, float, float, float], margin_deg: float = 0.25
) -> tuple[list[str], tuple[float, float, float, float]]:
    """Resolve a lon/lat bbox to the DIST-S1 MGRS tiles it overlaps plus a widened bbox for RTC-S1 queries.

    The widened bbox covers the footprints of every burst feeding those tiles (with a margin for
    per-acquisition footprint jitter) so no product input is missed at the AOI edge. Does not handle
    antimeridian-crossing AOIs.
    """
    tiles = get_mgrs_tiles_overlapping_geometry(box(*bbox))
    tile_ids = sorted(tiles.mgrs_tile_id)
    burst_ids = sorted(get_lut_by_mgrs_tile_ids(tile_ids).jpl_burst_id.unique())
    lon_min, lat_min, lon_max, lat_max = get_burst_table(burst_ids).total_bounds
    widened = (
        max(lon_min - margin_deg, -180.0),
        max(lat_min - margin_deg, -90.0),
        min(lon_max + margin_deg, 180.0),
        min(lat_max + margin_deg, 90.0),
    )
    return tile_ids, widened


def parse_dist_s1_granule(item: dict) -> dict:
    umm = item['umm']
    opera_id = umm['GranuleUR']
    tokens = opera_id.split('_')
    urls = [
        u['URL'] for u in umm.get('RelatedUrls', []) if u['URL'].startswith('https://') and u['URL'].endswith('.tif')
    ]
    url_cols = {
        col: next((u for u in urls if u.endswith(f'{suffix}.tif')), None) for col, suffix in LAYER_URL_MAP.items()
    }
    return {
        'opera_id': opera_id,
        'mgrs_tile_id': tokens[3].lstrip('T'),
        'acq_time': pd.Timestamp(tokens[4].rstrip('Z')),
        'processing_time': pd.Timestamp(tokens[5].rstrip('Z')),
        'sat': tokens[6],
        **url_cols,
    }


def parse_rtc_s1_granule(item: dict) -> dict:
    umm = item['umm']
    opera_id = umm['GranuleUR']
    tokens = opera_id.split('_')
    jpl_burst_id = tokens[3]
    attrs = {a['Name']: a['Values'] for a in umm.get('AdditionalAttributes', [])}
    urls = [
        u['URL'] for u in umm.get('RelatedUrls', []) if u['URL'].startswith('https://') and u['URL'].endswith('.tif')
    ]
    return {
        'opera_id': opera_id,
        'jpl_burst_id': jpl_burst_id,
        'acq_dt': pd.Timestamp(tokens[4].rstrip('Z')).tz_localize('UTC'),
        'processing_dt': pd.Timestamp(tokens[5].rstrip('Z')).tz_localize('UTC'),
        'track_number': get_track_number(jpl_burst_id),
        'polarizations': format_polarization(attrs.get('POLARIZATION', [])),
        'url_copol': next((u for u in urls if u.endswith(('_VV.tif', '_HH.tif'))), ''),
        'url_crosspol': next((u for u in urls if u.endswith(('_VH.tif', '_HV.tif'))), ''),
    }


def _dedup_reprocessed(df: pd.DataFrame, processing_col: str) -> pd.DataFrame:
    df = df.sort_values(['opera_dedup_id', processing_col])
    return df[~df.duplicated(subset='opera_dedup_id', keep='last')]


def get_dist_s1_table(
    start_time: DateLike,
    stop_time: DateLike,
    bbox: tuple[float, float, float, float] | None = None,
    token: str | None = None,
    chunk_days: int = 7,
    max_workers: int = 8,
    venue: str = 'PROD',
) -> gpd.GeoDataFrame:
    """Download DIST-S1 granule metadata from CMR (or a venue's S3 bucket) into a GeoDataFrame with MGRS footprints."""
    if venue in DIST_S1_S3_BUCKETS:
        df = _list_dist_s1_s3(start_time, stop_time, DIST_S1_S3_BUCKETS[venue], max_workers=max_workers)
        source = f's3://{DIST_S1_S3_BUCKETS[venue]}/{DIST_S1_S3_PREFIX}'
    else:
        token = token or get_edl_token(venue)
        df = _search_chunked_to_df(
            DIST_S1_CONCEPT_IDS[venue],
            start_time,
            stop_time,
            parse_dist_s1_granule,
            token=token,
            bbox=bbox,
            chunk_days=chunk_days,
            max_workers=max_workers,
            desc=f'DIST-S1 CMR ({venue})',
            cmr_url=CMR_GRANULES_URLS[venue],
        )
        source = 'CMR; check the Earthdata credentials'
    if df.empty:
        raise RuntimeError(f'No DIST-S1 granules returned from {source} for the time range.')
    df = df.drop_duplicates(subset='opera_id')
    df['opera_dedup_id'] = df.opera_id.map(get_opera_id_trunc)
    tile_geoms = get_mgrs_table()[['mgrs_tile_id', 'geometry']]
    df = tile_geoms.merge(df, on='mgrs_tile_id', how='right')
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    if venue in DIST_S1_S3_BUCKETS and bbox is not None:
        # CMR filters spatially server-side; a bucket listing cannot, so keep the tiles touching the bbox here.
        df = df[df.intersects(box(*bbox))]
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_not_exception_type(OfflineError),
    reraise=True,
)
def _list_s3_keys(bucket: str, prefix: str) -> list[str]:
    """Every object key under `prefix` (no delimiter), via anonymous ListObjectsV2."""
    network_allowed(f'listing s3://{bucket}/{prefix}')
    url = f'https://{bucket}.s3.{DIST_S1_S3_REGION}.amazonaws.com/'
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    params = {'list-type': '2', 'prefix': prefix}
    keys = []
    while True:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        keys += [e.text for e in root.findall('s3:Contents/s3:Key', ns)]
        token = root.findtext('s3:NextContinuationToken', namespaces=ns)
        if not token:
            return keys
        params['continuation-token'] = token


def _list_dist_s1_s3(start_time: DateLike, stop_time: DateLike, bucket: str, max_workers: int = 8) -> pd.DataFrame:
    """DIST-S1 products acquired in [start_time, stop_time), from the bucket's per-acquisition-day product folders.

    Lists every key, not just the product folders, so a product is known to have each layer only if that layer's
    key exists. A layer whose key is missing gets a null URL; a product without its STATUS layer therefore has no
    `alert_url`, its tags cannot be read (`missing_tag_url`) and coverage reports it UNKNOWN (F14).
    """
    # Product ids carry UTC times without an offset: convert the window to UTC before dropping its offset (F28).
    start, stop = (_utc_naive(t) for t in (start_time, stop_time))
    days = pd.date_range(start.normalize(), (stop - pd.Timedelta(1, 'ns')).normalize(), freq='D')
    base = f'https://{bucket}.s3.{DIST_S1_S3_REGION}.amazonaws.com/'

    def list_day(day: pd.Timestamp) -> list[dict]:
        day_prefix = f'{DIST_S1_S3_PREFIX}/{day:%Y/%m/%d}/'
        files: dict[str, set[str]] = {}
        for key in _list_s3_keys(bucket, day_prefix):
            parts = key.removeprefix(day_prefix).split('/')
            if len(parts) == 2 and parts[0]:
                files.setdefault(parts[0], set()).add(parts[1])
        records = []
        for opera_id, names in sorted(files.items()):
            tokens = opera_id.split('_')
            urls = {
                col: f'{base}{day_prefix}{opera_id}/{opera_id}_{suffix}.tif'
                if f'{opera_id}_{suffix}.tif' in names
                else None
                for col, suffix in LAYER_URL_MAP.items()
            }
            records.append(
                {
                    'opera_id': opera_id,
                    'mgrs_tile_id': tokens[3].lstrip('T'),
                    'acq_time': pd.Timestamp(tokens[4].rstrip('Z')),
                    'processing_time': pd.Timestamp(tokens[5].rstrip('Z')),
                    'sat': tokens[6],
                    **urls,
                }
            )
        return records

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        per_day = list(tqdm(pool.map(list_day, days), total=len(days), desc=f'DIST-S1 S3 ({bucket})', unit='day'))
    df = pd.DataFrame([r for day in per_day for r in day])
    if df.empty:
        return df
    in_window = (df.acq_time >= start) & (df.acq_time < stop)
    return df[in_window].reset_index(drop=True)


def get_rtc_s1_table(
    start_time: DateLike,
    stop_time: DateLike,
    bbox: tuple[float, float, float, float] | None = None,
    chunk_days: int = 2,
    max_workers: int = 8,
    cache_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Download RTC-S1 granule metadata from CMR into a GeoDataFrame with burst footprints."""
    df = _search_chunked_to_df(
        RTC_S1_CONCEPT_ID,
        start_time,
        stop_time,
        parse_rtc_s1_granule,
        bbox=bbox,
        chunk_days=chunk_days,
        max_workers=max_workers,
        desc='RTC-S1 CMR',
        cache_dir=cache_dir,
    )
    if df.empty:
        raise RuntimeError('No RTC-S1 granules returned; check the time range.')
    df = df.drop_duplicates(subset='opera_id')
    df['opera_dedup_id'] = df.opera_id.map(get_opera_id_trunc)
    df = _dedup_reprocessed(df, 'processing_dt')
    burst_geoms = get_burst_table(sorted(df.jpl_burst_id.unique()))[['jpl_burst_id', 'geometry']]
    df = burst_geoms.merge(df, on='jpl_burst_id', how='right')
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return df.sort_values(['jpl_burst_id', 'acq_dt']).reset_index(drop=True)


def lookback_date_ranges(
    start_time: DateLike,
    stop_time: DateLike,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Date ranges of RTC-S1 metadata needed to re-enumerate baselines for products in [start_time, stop_time]."""
    start, stop = pd.Timestamp(start_time), pd.Timestamp(stop_time)
    ranges = [
        (start - pd.Timedelta(days=delta + delta_window_days), stop - pd.Timedelta(days=delta))
        for delta in delta_lookback_days
    ]
    return sorted(ranges) + [(start, stop)]


def get_rtc_s1_table_with_lookback(
    start_time: DateLike,
    stop_time: DateLike,
    bbox: tuple[float, float, float, float] | None = None,
    delta_lookback_days: tuple[int, ...] = DELTA_LOOKBACK_DAYS,
    delta_window_days: int = DELTA_WINDOW_DAYS,
    chunk_days: int = 2,
    max_workers: int = 8,
    cache_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    tables = [
        get_rtc_s1_table(t0, t1, bbox=bbox, chunk_days=chunk_days, max_workers=max_workers, cache_dir=cache_dir)
        for t0, t1 in lookback_date_ranges(start_time, stop_time, delta_lookback_days, delta_window_days)
    ]
    df = pd.concat(tables).drop_duplicates(subset='opera_id')
    df = _dedup_reprocessed(df, 'processing_dt')
    return (
        gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        .sort_values(['jpl_burst_id', 'acq_dt'])
        .reset_index(drop=True)
    )


def build_rtc_input_frame(df_rtc: gpd.GeoDataFrame, mgrs_tile_ids: list[str] | None = None) -> gpd.GeoDataFrame:
    """Shape an RTC-S1 metadata table into the enumerator's rtc_s1_schema; see `build_rtc_input_frame_with_drops`."""
    return build_rtc_input_frame_with_drops(df_rtc, mgrs_tile_ids)[0]


RTC_DROP_REASONS = ('single_polarization', 'null_geometry', 'not_in_lut')


def build_rtc_input_frame_with_drops(
    df_rtc: gpd.GeoDataFrame, mgrs_tile_ids: list[str] | None = None
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Shape an RTC-S1 metadata table into the enumerator's rtc_s1_schema (LUT join adds one row per MGRS tile).

    Returns the frame and the number of RTC-S1 rows dropped per reason (F30), in this order of precedence:

    - ``single_polarization``: DIST-S1 never uses them, as a post-image or in a baseline.
    - ``null_geometry``: the burst has no footprint in the burst table.
    - ``not_in_lut``: the burst feeds none of `mgrs_tile_ids` in the MGRS-burst LUT (with an AOI, e.g. bursts of
      the widened RTC-S1 bbox that only touch neighbouring tiles).
    """
    dual_pol = df_rtc.polarizations.isin(DUAL_POLARIZATIONS)
    n_dropped = int((~dual_pol).sum())
    if n_dropped:
        warn(f'Dropping {n_dropped} RTC-S1 granules that are not dual polarization ({", ".join(DUAL_POLARIZATIONS)}).')
    has_geom = df_rtc.geometry.notna()
    df = df_rtc[dual_pol & has_geom].copy()
    df['acq_dt'] = pd.to_datetime(df.acq_dt, utc=True)
    df['pass_id'] = df.acq_dt.map(extract_pass_id)
    df['url_copol'] = df.url_copol.fillna('')
    df['url_crosspol'] = df.url_crosspol.fillna('')
    if mgrs_tile_ids is None:
        lut = get_mgrs_burst_lut()
        mgrs_tile_ids = sorted(lut[lut.jpl_burst_id.isin(df.jpl_burst_id.unique())].mgrs_tile_id.unique())
    n_in = df.opera_id.nunique()
    df = append_pass_data(df, mgrs_tile_ids)
    drops = {
        'single_polarization': n_dropped,
        'null_geometry': int((dual_pol & ~has_geom).sum()),
        'not_in_lut': n_in - int(df.opera_id.nunique()),
    }
    return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326'), drops
