from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.env import Env
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from dist_s1_prod_check.constants import DEFAULT_TAG_WORKERS
from dist_s1_prod_check.network import OfflineError, is_offline, network_allowed


COOKIE_FILE = str(Path.home() / '.dist_s1_prod_check_cookies.txt')

TAG_KEYS = ['mgrs_tile_id', 'pre_rtc_opera_ids', 'post_rtc_opera_ids', 'prior_dist_s1_product']


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_not_exception_type(OfflineError),
    reraise=True,
)
def get_tags(url: str) -> dict[str, str]:
    network_allowed(f'reading tags from {url}')
    gdal_env = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
        'GDAL_HTTP_COOKIEFILE': COOKIE_FILE,
        'GDAL_HTTP_COOKIEJAR': COOKIE_FILE,
    }
    with Env(**gdal_env):
        with rasterio.open(url) as ds:
            return ds.tags()


def _preflight(url: str) -> None:
    """Read one product without retries so an access problem fails in seconds, not after hours of backoff."""
    try:
        get_tags.retry_with(stop=stop_after_attempt(1))(url)
    except Exception as e:
        raise RuntimeError(f'Cannot read product tags from {url}: {type(e).__name__}: {e}') from e


def _fetch_one(record: dict) -> dict:
    out = {'opera_id': record['opera_id'], 'error': ''}
    try:
        tags = get_tags(record['alert_url'])
        out.update({key: tags.get(key, '') for key in TAG_KEYS})
    except Exception as e:
        out.update(dict.fromkeys(TAG_KEYS, ''))
        out['error'] = f'{type(e).__name__}: {e}'
    return out


MISSING_TAG_URL = 'missing_tag_url'
NO_CACHED_TAG_ROW = 'no cached tag row (run fetch-tags)'
TAG_COLUMNS = ['opera_id', 'error', *TAG_KEYS]


def resolve_tag_urls(df_products: pd.DataFrame, url_template: str | None = None) -> pd.Series:
    """URL each product's tags are read from: `url_template` formatted with the product row, else `alert_url`.

    Indexed like `df_products`; a product with no URL (null `alert_url`, no template) is None.
    """
    if url_template:
        return pd.Series(
            [url_template.format(**row) for row in df_products.to_dict('records')], index=df_products.index
        )
    urls = df_products.alert_url if 'alert_url' in df_products.columns else pd.Series(None, index=df_products.index)
    return urls.astype(object).where(urls.notna() & (urls.astype(str) != ''), None)


def align_to_products(
    df_products: pd.DataFrame, df_tags: pd.DataFrame, urls: pd.Series | None = None, missing: str = NO_CACHED_TAG_ROW
) -> pd.DataFrame:
    """Return exactly one tag row per product of `df_products`, in its order (F03).

    A product with no URL gets ``error='missing_tag_url'``; one with a URL but no row in `df_tags` gets
    `missing`. Rows are never dropped, so a check built on the result covers the whole product universe. Error rows
    with an empty `mgrs_tile_id` take the product's tile.
    """
    urls = resolve_tag_urls(df_products) if urls is None else urls
    rows = df_tags.drop_duplicates(subset='opera_id', keep='last')
    df = pd.DataFrame({'opera_id': df_products.opera_id.to_numpy()}).merge(rows, on='opera_id', how='left')
    df.index = df_products.index
    for col in TAG_KEYS:
        df[col] = df[col].fillna('') if col in df.columns else ''
    absent = df.error.isna()
    df['error'] = df.error.where(~absent, missing)
    no_url = urls.isna().to_numpy()
    df.loc[no_url, TAG_KEYS] = ''
    df.loc[no_url, 'error'] = MISSING_TAG_URL
    blank_tile = (df.error != '') & (df.mgrs_tile_id == '')
    if 'mgrs_tile_id' in df_products.columns:
        df.loc[blank_tile, 'mgrs_tile_id'] = df_products.mgrs_tile_id[blank_tile]
    return df[[*TAG_COLUMNS, *(c for c in df.columns if c not in TAG_COLUMNS)]].reset_index(drop=True)


def fetch_tags_table(
    df_products: pd.DataFrame,
    out_path: str | Path,
    max_workers: int = DEFAULT_TAG_WORKERS,
    batch_size: int = 2_000,
    retry_errors: bool = True,
    url_template: str | None = None,
) -> pd.DataFrame:
    """Fetch DIST-S1 GeoTIFF metadata tags for each product, resumably cached in a parquet file.

    Reads `pre_rtc_opera_ids`, `post_rtc_opera_ids`, `mgrs_tile_id`, and `prior_dist_s1_product`
    from each product's GEN-DIST-STATUS layer. Products already present in `out_path` are skipped,
    so interrupted runs resume where they left off.

    `url_template` reads the layer from somewhere other than the CMR `alert_url`, e.g. the producing
    venue's S3 bucket. It is formatted with each product row, so `{opera_id}`, `{mgrs_tile_id}` and
    `{acq_time:%Y%m%d}` are available.

    Returns exactly one row per product of `df_products`, in its order (F03): a product with no URL gets
    ``error='missing_tag_url'`` (in the returned frame only; nothing was read, so nothing is cached).

    In offline mode cached rows holding a read error are kept, not retried, with `(not retried: offline)`
    appended to their error in the returned frame (never in the parquet), and any product with no cached row
    raises `OfflineError`.
    """
    out_path = Path(out_path)
    urls = resolve_tag_urls(df_products, url_template)
    df_existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=TAG_COLUMNS)
    offline = is_offline()
    df_reuse = df_existing[df_existing.error == ''] if retry_errors and not offline else df_existing

    todo_mask = urls.notna() & ~df_products.opera_id.isin(df_reuse.opera_id)
    records = [{'opera_id': o, 'alert_url': u} for o, u in zip(df_products.opera_id[todo_mask], urls[todo_mask])]
    if offline and records:
        network_allowed(f'fetching tags for {len(records)} products with no cached tag row')
    if records:
        df_reuse = _fetch_into(records, df_reuse, out_path, max_workers, batch_size)
    df = align_to_products(df_products, df_reuse, urls)
    if offline:
        failed = (df.error != '') & (df.error != MISSING_TAG_URL)
        df['error'] = df.error.where(~failed, df.error + ' (not retried: offline)')
    return df


def _fetch_into(
    records: list[dict], df_existing: pd.DataFrame, out_path: Path, max_workers: int, batch_size: int
) -> pd.DataFrame:
    """Fetch `records` in batches, appending each batch to `df_existing` and checkpointing it to `out_path`."""
    _preflight(records[0]['alert_url'])
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    with tqdm(total=len(records), desc='Fetching tags') as pbar:
        for batch in batches:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = []
                for result in pool.map(_fetch_one, batch):
                    results.append(result)
                    pbar.update(1)
            df_existing = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
            tmp_path = out_path.with_suffix('.parquet.tmp')
            df_existing.to_parquet(tmp_path, compression='zstd')
            tmp_path.replace(out_path)
    return df_existing
