import pandas as pd

from dist_s1_prod_check.ids import get_opera_id_trunc


def deduplicate_products(df_dist: pd.DataFrame) -> pd.DataFrame:
    df = df_dist.sort_values(['opera_dedup_id', 'processing_time'])
    df = df[~df.duplicated(subset='opera_dedup_id', keep='last')]
    return df.sort_values(['mgrs_tile_id', 'acq_time']).reset_index(drop=True)


def check_duplicates(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Flag DIST-S1 granules sharing the same tile/acquisition (reprocessed or duplicated products)."""
    dup_mask = df_dist.duplicated(subset='opera_dedup_id', keep=False)
    df_dup = df_dist[dup_mask].sort_values(['opera_dedup_id', 'processing_time']).copy()
    if df_dup.empty:
        return pd.DataFrame(
            columns=['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time', 'n_versions', 'is_latest', 'reason']
        )
    df_dup['n_versions'] = df_dup.groupby('opera_dedup_id')['opera_id'].transform('size')
    df_dup['is_latest'] = ~df_dup.duplicated(subset='opera_dedup_id', keep='last')
    df_dup['reason'] = df_dup.apply(lambda r: f'{r.n_versions} granules share acquisition {r.opera_dedup_id}', axis=1)
    cols = ['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time', 'n_versions', 'is_latest', 'reason']
    extra = [c for c in ['alert_url'] if c in df_dup.columns]
    return df_dup[cols + extra].reset_index(drop=True)


def check_processing_order(df_dist_dedup: pd.DataFrame) -> pd.DataFrame:
    """Check that within each MGRS tile products were processed in acquisition order."""
    df = df_dist_dedup.sort_values(['mgrs_tile_id', 'acq_time']).copy()
    df['acq_rank'] = df.groupby('mgrs_tile_id')['acq_time'].rank(method='first').astype(int)
    df['proc_rank'] = df.groupby('mgrs_tile_id')['processing_time'].rank(method='first').astype(int)
    df['out_of_order'] = df.acq_rank != df.proc_rank
    df['reason'] = [
        f'processing rank {p} != acquisition rank {a} within tile' if bad else ''
        for bad, p, a in zip(df.out_of_order, df.proc_rank, df.acq_rank)
    ]
    cols = [
        'opera_id',
        'mgrs_tile_id',
        'acq_time',
        'processing_time',
        'acq_rank',
        'proc_rank',
        'out_of_order',
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)


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


def check_confirmation(
    df_dist_dedup: pd.DataFrame,
    df_tags: pd.DataFrame,
    expect_none_at_start: bool = False,
) -> pd.DataFrame:
    """Check each product's `prior_dist_s1_product` tag points to the previous product in its tile's time series."""
    df = df_dist_dedup.merge(df_tags[['opera_id', 'prior_dist_s1_product', 'error']], on='opera_id', how='left')
    df = df.sort_values(['mgrs_tile_id', 'acq_time'])

    df['tag_missing'] = df.error.isna() | (df.error != '') | df.prior_dist_s1_product.isna()
    df['tile_incomplete'] = df.groupby('mgrs_tile_id')['tag_missing'].transform('any')
    df['prior_used'] = df.prior_dist_s1_product.fillna('').map(lambda p: p.split('/')[-1] or 'None')
    df['prior_expected'] = df.groupby('mgrs_tile_id')['opera_id'].shift(1).fillna('None')
    df['is_first_in_window'] = df.prior_expected == 'None'

    used = df.prior_used.map(get_opera_id_trunc)
    expected = df.prior_expected.map(get_opera_id_trunc)
    df['confirmed_ok'] = used == expected
    if not expect_none_at_start:
        df.loc[df.is_first_in_window, 'confirmed_ok'] = True
    df.loc[df.tag_missing, 'confirmed_ok'] = False

    def _reason(row: pd.Series) -> str:
        if row.tag_missing:
            return 'could not read prior_dist_s1_product tag'
        if not row.confirmed_ok:
            return f'prior product used: {row.prior_used}; expected: {row.prior_expected}'
        return ''

    df['reason'] = df.apply(_reason, axis=1)

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
        'reason',
    ]
    extra = [c for c in ['alert_url'] if c in df.columns]
    return df[cols + extra].reset_index(drop=True)
