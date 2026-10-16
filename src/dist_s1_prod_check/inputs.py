import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dist_s1_enumerator import enumerate_dist_s1_products, enumerate_one_dist_s1_product
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from dist_s1_prod_check import manifest
from dist_s1_prod_check.constants import (
    DELTA_LOOKBACK_DAYS,
    DELTA_WINDOW_DAYS,
    DUAL_POLARIZATIONS,
    MAX_POST_SPAN_SECONDS,
    MAX_PRE_IMGS_PER_BURST,
    PASS_MATCH_TOLERANCE_SECONDS,
    POST_DATE_BUFFER_SECONDS,
)
from dist_s1_prod_check.ids import (
    get_acq_timestamp,
    get_burst_id,
    get_mgrs_tile_id,
    get_opera_id_trunc,
    get_track_number,
)
from dist_s1_prod_check.network import OfflineError, network_allowed
from dist_s1_prod_check.tags import MISSING_TAG_URL, TAG_COLUMNS


N_S1_TRACKS = 175


def parse_actual_inputs(pre_rtc_opera_ids: str, post_rtc_opera_ids: str) -> pd.DataFrame:
    """Split comma-joined pre/post RTC-S1 ids from a product's tags into rows with burst, track and acquisition time."""
    pre_ids = [p for p in pre_rtc_opera_ids.split(',') if p]
    post_ids = [p for p in post_rtc_opera_ids.split(',') if p]
    df = pd.DataFrame(
        {
            'opera_id': pre_ids + post_ids,
            'input_category': ['pre'] * len(pre_ids) + ['post'] * len(post_ids),
        }
    )
    df['opera_id_trunc'] = df.opera_id.map(get_opera_id_trunc)
    df['jpl_burst_id'] = df.opera_id.map(get_burst_id)
    df['track_number'] = df.jpl_burst_id.map(get_track_number)
    df['acq_dt'] = df.opera_id.map(get_acq_timestamp)
    return df


def build_expected_products(df_rtc_frame: gpd.GeoDataFrame, mgrs_tile_ids: list[str] | None = None) -> pd.DataFrame:
    """Enumerate all expected DIST-S1 products (with pre/post inputs) offline from an RTC-S1 metadata table.

    Enumerates tile by tile via groupby so a global-scale frame is scanned once, not once per tile.
    """
    keep = None if mgrs_tile_ids is None else set(mgrs_tile_ids)
    frames = []
    product_id_offset = 0
    groups = df_rtc_frame.groupby('mgrs_tile_id', sort=True)
    with Path('/dev/null').open('w') as devnull, redirect_stdout(devnull):
        for tile, df_tile in tqdm(groups, total=groups.ngroups, desc='Enumerating expected products'):
            if keep is not None and tile not in keep:
                continue
            df_exp = enumerate_dist_s1_products(
                df_tile.reset_index(drop=True), mgrs_tile_ids=[tile], tqdm_enabled=False
            )
            if df_exp.empty:
                continue
            df_exp['product_id'] = df_exp['product_id'] + product_id_offset
            product_id_offset = int(df_exp['product_id'].max()) + 1
            frames.append(pd.DataFrame(df_exp.drop(columns='geometry')))
    df_expected = pd.concat(frames, ignore_index=True)
    df_expected['opera_id_trunc'] = df_expected.opera_id.map(get_opera_id_trunc)
    return df_expected


def _tracks_adjacent(track_numbers: list[int]) -> bool:
    if len(track_numbers) <= 1:
        return True
    if len(track_numbers) > 2:
        return False
    diff = abs(track_numbers[0] - track_numbers[1])
    return diff == 1 or diff == N_S1_TRACKS - 1


def _polarization_issues(
    df_actual: pd.DataFrame, pol_by_opera_id: dict[str, str], check_single_pol: bool = False
) -> list[dict]:
    """Flag inputs whose baseline polarization differs from the post-image, and optionally single-pol inputs.

    A burst's baseline must be built from acquisitions of the same dual polarization as its post-image
    (VV+VH with VV+VH, HH+HV with HH+HV). Inputs absent from `pol_by_opera_id` (outside the RTC table)
    are skipped; the RTC ID comparisons already report those.

    `check_single_pol` is for online mode, whose lookup comes from the enumerator's metadata. Offline it could
    never fire: the RTC frame drops single-polarization granules, so such an input is absent from the lookup and
    fails the RTC ID comparison instead (F38).
    """
    pols = df_actual.opera_id_trunc.map(pol_by_opera_id)
    df = df_actual.assign(polarizations=pols)[pols.notna()]
    if df.empty:
        return []

    issues = []
    df_single = df[~df.polarizations.isin(DUAL_POLARIZATIONS)]
    if check_single_pol and not df_single.empty:
        detail = ';'.join(f'{r.opera_id_trunc}={r.polarizations}' for r in df_single.itertuples())
        issues.append(_issue('Single polarization input used', detail))

    df_post = df[df.input_category == 'post']
    post_pol_by_burst = dict(zip(df_post.jpl_burst_id, df_post.polarizations))
    df_pre = df[df.input_category == 'pre']
    mismatched = [
        (r.opera_id_trunc, r.polarizations, post_pol_by_burst[r.jpl_burst_id])
        for r in df_pre.itertuples()
        if r.jpl_burst_id in post_pol_by_burst and r.polarizations != post_pol_by_burst[r.jpl_burst_id]
    ]
    if mismatched:
        detail = ';'.join(f'{opera_id}={pol} vs post-image {post_pol}' for opera_id, pol, post_pol in mismatched)
        issues.append(_issue('Pre RTC baseline polarization mismatch', detail))
    return issues


def compare_inputs(
    df_actual: pd.DataFrame,
    df_expected_product: pd.DataFrame,
    pol_by_opera_id: dict[str, str] | None = None,
    check_single_pol: bool = False,
) -> list[dict]:
    """Issues of a product's actual inputs against its expected product's inputs (see `_polarization_issues`)."""
    issues = []
    if pol_by_opera_id:
        issues.extend(_polarization_issues(df_actual, pol_by_opera_id, check_single_pol=check_single_pol))

    track_numbers = sorted(df_actual.track_number.unique().tolist())
    if not _tracks_adjacent(track_numbers):
        issues.append(_issue('Too many track numbers present', str(track_numbers)))

    df_post = df_actual[df_actual.input_category == 'post']
    df_pre = df_actual[df_actual.input_category == 'pre']
    post_span = df_post.acq_dt.max() - df_post.acq_dt.min()
    if post_span.total_seconds() > MAX_POST_SPAN_SECONDS:
        issues.append(_issue('Post-dates span too long', str(post_span)))

    exp_post = df_expected_product[df_expected_product.input_category == 'post']
    exp_pre = df_expected_product[df_expected_product.input_category == 'pre']

    bursts_expected = sorted(df_expected_product.jpl_burst_id.unique())
    bursts_actual = sorted(df_actual.jpl_burst_id.unique())
    if bursts_expected != bursts_actual:
        issues.append(_issue('Burst ID mismatch', f'expected={bursts_expected}; found={bursts_actual}'))

    comparisons = [
        ('Pre RTC IDs expected but not found', exp_pre, df_pre),
        ('Pre RTC IDs found but not expected', df_pre, exp_pre),
        ('Post RTC IDs expected but not found', exp_post, df_post),
        ('Post RTC IDs found but not expected', df_post, exp_post),
    ]
    for issue_type, df_left, df_right in comparisons:
        # Multisets (F26): an input id listed twice but expected once appears once in "found but not expected".
        diff = sorted((Counter(df_left.opera_id_trunc) - Counter(df_right.opera_id_trunc)).elements())
        if diff:
            issues.append(_issue(issue_type, ';'.join(diff)))

    return issues


def archive_ids_by_trunc(opera_ids: pd.Series) -> dict[str, list[str]]:
    """Full RTC-S1 ids of an archive table (frame or enumerator output) keyed by truncated id."""
    ids = pd.Series(opera_ids).drop_duplicates()
    return {trunc: sorted(group) for trunc, group in ids.groupby(ids.map(get_opera_id_trunc))}


def _version_notes(df_actual: pd.DataFrame, archive_by_trunc: dict[str, list[str]] | None) -> list[dict]:
    """INFO `RTC processing version differs` for inputs whose acquisition is archived under another full id (F26).

    The tags name an RTC-S1 product whose truncated id (burst and acquisition) is in the archive table but whose
    full id (processing time, version) is not, e.g. because the input was reprocessed after the product was made.
    Pass/fail depends only on acquisition identity, so this is a note, not a failure.
    """
    if not archive_by_trunc:
        return []
    differs = [
        f'{r.opera_id} vs archive {",".join(archive_by_trunc[r.opera_id_trunc])}'
        for r in df_actual.drop_duplicates('opera_id').itertuples()
        if r.opera_id_trunc in archive_by_trunc and r.opera_id not in archive_by_trunc[r.opera_id_trunc]
    ]
    return [_issue('RTC processing version differs', ';'.join(differs), status='INFO')] if differs else []


def _expected_product_index(df_expected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    posts = df_expected[df_expected.input_category == 'post']
    idx = (
        posts.groupby(['mgrs_tile_id', 'product_id'])
        .agg(post_min_dt=('acq_dt', 'min'), track_numbers=('track_number', 'unique'))
        .reset_index()
    )
    return {tile: group for tile, group in idx.groupby('mgrs_tile_id')}


def own_tile(opera_id: str) -> str:
    """MGRS tile of a DIST-S1 product from its own opera_id, or '' if the id cannot be parsed."""
    try:
        return get_mgrs_tile_id(opera_id)
    except (IndexError, AttributeError, TypeError):
        return ''


def _match_expected_product(
    candidates: pd.DataFrame | None, acq: pd.Timestamp, track_numbers: list[int]
) -> tuple[int, float] | None:
    """Choose the expected product of a DIST-S1 product from its own tile and acquisition time (F05).

    Candidates are the tile's expected products whose earliest post acquisition is within
    `POST_DATE_BUFFER_SECONDS` of the product's opera_id acquisition time. If several qualify, those sharing a
    track with the tagged post inputs are preferred (the only use of the tags here); among the rest the closest in
    time wins (ties: smaller product_id). Returns the product_id and its time difference in seconds, or None.
    """
    if candidates is None:
        return None
    offset = (candidates.post_min_dt - acq).abs().dt.total_seconds()
    matches = candidates.assign(offset_s=offset)[offset < POST_DATE_BUFFER_SECONDS]
    if len(matches) > 1:
        on_track = matches[matches.track_numbers.map(lambda tracks: bool(set(tracks) & set(track_numbers)))]
        if not on_track.empty:
            matches = on_track
    if matches.empty:
        return None
    best = matches.sort_values(['offset_s', 'product_id']).iloc[0]
    return int(best.product_id), float(best.offset_s)


def _identity_issues(tag_tile: str, tile: str, acq: pd.Timestamp, df_actual: pd.DataFrame) -> list[dict]:
    """FAIL `Identity mismatch` if the tags describe another tile or acquisition than the product's opera_id (F05)."""
    disagreements = []
    if tag_tile != tile:
        disagreements.append(f'tag mgrs_tile_id={tag_tile} vs opera_id tile={tile}')
    post_min = df_actual[df_actual.input_category == 'post'].acq_dt.min()
    if pd.notna(post_min) and abs((post_min - acq).total_seconds()) > PASS_MATCH_TOLERANCE_SECONDS:
        disagreements.append(f'earliest post input acq={post_min} vs opera_id acq={acq}')
    return [_issue('Identity mismatch', '; '.join(disagreements))] if disagreements else []


def _check_products_against_expected(
    df_tags_tile: pd.DataFrame,
    df_expected: pd.DataFrame,
    pol_by_opera_id: dict[str, str] | None = None,
    archive_by_trunc: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Check one tile's products against the tile's expected products and return their result records.

    Each product's expected product comes from its own opera_id (tile and acquisition time), and its tags must agree
    with that identity (F05). Matching is one-to-one: if several products claim one expected product, the closest
    in acquisition time wins (ties: smaller opera_id) and the others FAIL `Expected product already claimed`
    without an input comparison.

    A product whose tags cannot be parsed or compared gets one UNKNOWN `Comparison error` row and takes no part in
    claiming, so its tile neighbours are still checked (F22). Inputs archived under another full id in
    `archive_by_trunc` add an INFO note (`_version_notes`). Errors enumerating the tile are the caller's
    (`_check_one_tile`: the whole tile is UNKNOWN `Enumeration error`).
    """
    index_by_tile = _expected_product_index(df_expected)
    expected_by_product = dict(tuple(df_expected.groupby('product_id')))

    pending = {}  # opera_id -> (base, issues, df_actual, product_id)
    claims: dict[int, list[tuple[float, str]]] = {}
    records_by_product: dict[str, list[dict]] = {}
    for row in df_tags_tile.itertuples():
        tile = own_tile(row.opera_id)
        base = {'opera_id': row.opera_id, 'mgrs_tile_id': tile}
        try:
            acq = get_acq_timestamp(row.opera_id)
            df_actual = parse_actual_inputs(row.pre_rtc_opera_ids, row.post_rtc_opera_ids)
            issues = [
                *_identity_issues(row.mgrs_tile_id, tile, acq, df_actual),
                *_version_notes(df_actual, archive_by_trunc),
            ]
            post_tracks = sorted(df_actual[df_actual.input_category == 'post'].track_number.unique().tolist())
            match = _match_expected_product(index_by_tile.get(tile), acq, post_tracks)
        except Exception as e:
            records_by_product[row.opera_id] = [_record(base, 'UNKNOWN', 'Comparison error', _error_text(e))]
            continue
        if match is None:
            no_match = _issue('No expected product found', f'tile={tile}; acq={acq}; post tracks={post_tracks}')
            records_by_product[row.opera_id] = _product_records(base, [*issues, no_match])
            continue
        product_id, dt = match
        pending[row.opera_id] = (base, issues, df_actual, product_id)
        claims.setdefault(product_id, []).append((dt, row.opera_id))

    winner_by_product = {product_id: min(claimants)[1] for product_id, claimants in claims.items()}
    for opera_id, (base, issues, df_actual, product_id) in pending.items():
        winner = winner_by_product[product_id]
        if winner != opera_id:
            issues = [*issues, _issue('Expected product already claimed', f'claimed by {winner}')]
        else:
            try:
                issues = [
                    *issues,
                    *compare_inputs(df_actual, expected_by_product[product_id], pol_by_opera_id=pol_by_opera_id),
                ]
            except Exception as e:
                records_by_product[opera_id] = [_record(base, 'UNKNOWN', 'Comparison error', _error_text(e))]
                continue
        records_by_product[opera_id] = _product_records(base, issues)
    return [record for row in df_tags_tile.itertuples() for record in records_by_product[row.opera_id]]


def _check_one_tile(tile: str, df_tags_tile: pd.DataFrame, df_tile_frame: gpd.GeoDataFrame) -> list[dict]:
    try:
        with Path('/dev/null').open('w') as devnull, redirect_stdout(devnull):
            df_expected = enumerate_dist_s1_products(
                df_tile_frame.reset_index(drop=True), mgrs_tile_ids=[tile], tqdm_enabled=False
            )
        df_expected = pd.DataFrame(df_expected.drop(columns='geometry'))
        df_expected['opera_id_trunc'] = df_expected.opera_id.map(get_opera_id_trunc)
        pol_by_opera_id = dict(zip(df_tile_frame.opera_id.map(get_opera_id_trunc), df_tile_frame.polarizations))
        return _check_products_against_expected(
            df_tags_tile,
            df_expected,
            pol_by_opera_id=pol_by_opera_id,
            archive_by_trunc=archive_ids_by_trunc(df_tile_frame.opera_id),
        )
    except Exception as e:
        return [
            _record({'opera_id': row.opera_id, 'mgrs_tile_id': tile}, 'UNKNOWN', 'Enumeration error', _error_text(e))
            for row in df_tags_tile.itertuples()
        ]


RESULT_COLUMNS = ['opera_id', 'mgrs_tile_id', 'inputs_correct', 'issue_type', 'issue_value', 'status']
# A result row's status and the legacy `inputs_correct` it implies: INFO notes never fail a product.
INPUTS_CORRECT_BY_STATUS = {'PASS': True, 'INFO': True, 'FAIL': False, 'UNKNOWN': None}


def _error_text(e: Exception) -> str:
    return f'{type(e).__name__}: {e}'


def _issue(issue_type: str, issue_value: str, status: str = 'FAIL') -> dict:
    return {'issue_type': issue_type, 'issue_value': issue_value, 'status': status}


def _record(base: dict, status: str, issue_type: str = '', issue_value: str = '') -> dict:
    """One result row with columns `RESULT_COLUMNS`."""
    return {
        **base,
        'inputs_correct': INPUTS_CORRECT_BY_STATUS[status],
        'issue_type': issue_type,
        'issue_value': issue_value,
        'status': status,
    }


def _product_records(base: dict, issues: list[dict]) -> list[dict]:
    """Return a product's rows: one FAIL row per failing issue, or one PASS row if none fails, then any INFO rows."""
    failing = [_record(base, i['status'], i['issue_type'], i['issue_value']) for i in issues if i['status'] != 'INFO']
    notes = [_record(base, 'INFO', i['issue_type'], i['issue_value']) for i in issues if i['status'] == 'INFO']
    return (failing or [_record(base, 'PASS')]) + notes


def checkpoint_context(rtc_sha256: str, options: dict | None = None) -> str:
    """Digest of everything besides a product's own tags that its inputs result depends on (F04).

    Covers the RTC-S1 table (its parquet sha256), this tool's and the enumerator's versions, the enumeration
    constants and `options` (e.g. mode and target tiles). A checkpoint row computed under another context is stale.
    """
    payload = {
        'rtc_sha256': rtc_sha256,
        'tool_version': manifest.tool_version(),
        'enumerator_version': manifest.enumerator_version(),
        'enumeration': [
            DELTA_LOOKBACK_DAYS,
            MAX_PRE_IMGS_PER_BURST,
            DELTA_WINDOW_DAYS,
            POST_DATE_BUFFER_SECONDS,
            PASS_MATCH_TOLERANCE_SECONDS,
            MAX_POST_SPAN_SECONDS,
        ],
        'options': options or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def product_fingerprints(df_tags: pd.DataFrame, context: str) -> dict[str, str]:
    """Fingerprint per product: sha256 of the checkpoint context and the product's tag row (incl. `source_url`)."""
    cols = [c for c in [*TAG_COLUMNS, 'source_url'] if c in df_tags.columns]
    rows = df_tags[cols].fillna('').astype(str)
    return {
        oid: hashlib.sha256('\x1f'.join([context, *values]).encode()).hexdigest()
        for oid, values in zip(rows.opera_id, rows.itertuples(index=False, name=None))
    }


def _results_frame(records: list[dict]) -> pd.DataFrame:
    """Return result rows as a frame that keeps its columns when there are no rows (an empty universe)."""
    return pd.DataFrame(records) if records else pd.DataFrame(columns=RESULT_COLUMNS)


def check_inputs_offline(
    df_tags: pd.DataFrame,
    df_rtc_frame: gpd.GeoDataFrame,
    checkpoint_path: str | Path | None = None,
    workers: int = 1,
    checkpoint_every: int = 100,
    context: str = '',
) -> pd.DataFrame:
    """Compare each product's recorded RTC inputs against an offline re-enumeration from the RTC-S1 table.

    Enumerates expected products one MGRS tile at a time (in `workers` processes when > 1) and
    discards them after comparison, so peak memory is bounded by a single tile's inputs. With
    `checkpoint_path`, results are persisted every `checkpoint_every` tiles and prior results are
    reused, so interrupted runs resume instead of restarting.

    Each checkpoint row stores its product's fingerprint (`product_fingerprints` under `context`, see
    `checkpoint_context`). A product is reused only if its stored fingerprint equals the current one, so a changed
    RTC-S1 table, tag row, tag source, tool or enumerator version, or option recomputes it (F04). Rows that could
    not be checked (tag read error, enumeration error) are never written to the checkpoint, so they are always
    recomputed. A checkpoint without fingerprints or a `status` column (written before those rules) is recomputed
    in full.

    Products are grouped by the tile in their own opera_id, not the tile in their tags (F05).
    """
    fingerprints = product_fingerprints(df_tags, context)
    records: list[dict] = []
    df_other = pd.DataFrame()
    done_opera: set[str] = set()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists():
        df_ck = pd.read_parquet(checkpoint_path)
        if {'opera_id', 'fingerprint', 'status'} <= set(df_ck.columns):
            current = df_ck.opera_id.map(fingerprints)
            df_other = df_ck[current.isna()]  # other products' rows, kept as they are
            df_mine = df_ck[current.notna()]
            fresh = (df_mine.fingerprint == df_mine.opera_id.map(fingerprints)) & df_mine.inputs_correct.notna()
            stale_products = set(df_mine.opera_id[~fresh])
            df_reuse = df_mine[~df_mine.opera_id.isin(stale_products)]
            records = df_reuse.drop(columns='fingerprint').to_dict('records')
            done_opera = set(df_reuse.opera_id)

    requested_opera = set(df_tags.opera_id)
    df_todo = df_tags[~df_tags.opera_id.isin(done_opera)]
    df_err = df_todo[df_todo.error != '']
    records.extend(
        _record(
            {'opera_id': row.opera_id, 'mgrs_tile_id': own_tile(row.opera_id)},
            'UNKNOWN',
            'Missing tag URL' if row.error == MISSING_TAG_URL else 'Tag read error',
            row.error,
        )
        for row in df_err.itertuples()
    )
    df_ok = df_todo[df_todo.error == '']
    tags_by_tile = dict(tuple(df_ok.groupby(df_ok.opera_id.map(own_tile))))
    frame_tiles = set(df_rtc_frame.mgrs_tile_id.unique())

    def _flush() -> None:
        if checkpoint_path is None:
            return
        df_done = _results_frame(records)
        df_done = df_done[df_done.inputs_correct.notna()]
        df_done = df_done.assign(fingerprint=df_done.opera_id.map(fingerprints))
        tmp_path = checkpoint_path.with_suffix('.parquet.tmp')
        df_out = pd.concat([df_other, df_done], ignore_index=True) if not df_other.empty else df_done
        df_out.to_parquet(tmp_path, compression='zstd')
        tmp_path.replace(checkpoint_path)

    frame_groups = df_rtc_frame.groupby('mgrs_tile_id')
    tiles_todo = []
    for tile, df_tags_tile in sorted(tags_by_tile.items()):
        if tile not in frame_tiles:
            records.extend(
                _record(
                    {'opera_id': row.opera_id, 'mgrs_tile_id': tile},
                    'FAIL',
                    'No expected product found',
                    f'tile={tile} absent from RTC-S1 table',
                )
                for row in df_tags_tile.itertuples()
            )
            continue
        tiles_todo.append(tile)

    with tqdm(total=len(tiles_todo), desc='Checking inputs by tile') as pbar:
        if workers <= 1:
            for k, tile in enumerate(tiles_todo):
                records.extend(_check_one_tile(tile, tags_by_tile[tile], frame_groups.get_group(tile)))
                pbar.update(1)
                if (k + 1) % checkpoint_every == 0:
                    _flush()
        else:
            window = workers * 4
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i in range(0, len(tiles_todo), window):
                    futures = [
                        pool.submit(_check_one_tile, tile, tags_by_tile[tile], frame_groups.get_group(tile))
                        for tile in tiles_todo[i : i + window]
                    ]
                    for future in as_completed(futures):
                        records.extend(future.result())
                        pbar.update(1)
                    _flush()
    _flush()
    df = _results_frame(records)
    return df[df.opera_id.isin(requested_opera)].reset_index(drop=True)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_not_exception_type(OfflineError),
    reraise=True,
)
def _enumerate_one(mgrs_tile_id: str, track_number: int, post_date: str) -> pd.DataFrame:
    network_allowed('online inputs mode (enumerator queries ASF)')
    df = enumerate_one_dist_s1_product(
        mgrs_tile_id=mgrs_tile_id,
        track_number=track_number,
        post_date=post_date,
        lookback_strategy='multi_window',
        delta_lookback_days=DELTA_LOOKBACK_DAYS,
        max_pre_imgs_per_burst=MAX_PRE_IMGS_PER_BURST,
        delta_window_days=DELTA_WINDOW_DAYS,
        tqdm_enabled=False,
    )
    df = pd.DataFrame(df.drop(columns='geometry'))
    df['opera_id_trunc'] = df.opera_id.map(get_opera_id_trunc)
    return df


def _enumerate_post_tracks(mgrs_tile_id: str, post_tracks: list[int], post_date: str) -> pd.DataFrame:
    """Enumerate one product's expected inputs once per post track and combine them (F24).

    An acquisition group can span two adjacent tracks; enumerating each and dropping repeated inputs covers both
    whether the enumerator returns one track's inputs or the whole product per call.
    """
    frames = [_enumerate_one(mgrs_tile_id, track, post_date) for track in post_tracks]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(['opera_id_trunc', 'input_category']).reset_index(drop=True)


def check_inputs_online(df_tags: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Compare recorded RTC inputs against a live per-product re-enumeration (slow; use for small batches).

    The tile and post date come from the product's own opera_id (F05), and the track numbers from its tagged post
    inputs, one enumeration per post track (F24). The polarization lookup and the version notes use the RTC-S1
    metadata the enumerator returned. Enumeration failures are UNKNOWN `Enumeration error`; tags that cannot be
    parsed or compared are UNKNOWN `Comparison error`.
    """

    def _one(row: tuple) -> list[dict]:
        tile = own_tile(row.opera_id)
        base = {'opera_id': row.opera_id, 'mgrs_tile_id': tile}
        if row.error:
            return [_record(base, 'UNKNOWN', 'Tag read error', row.error)]
        try:
            acq = get_acq_timestamp(row.opera_id)
            df_actual = parse_actual_inputs(row.pre_rtc_opera_ids, row.post_rtc_opera_ids)
            post_tracks = sorted(df_actual[df_actual.input_category == 'post'].track_number.unique().tolist())
        except Exception as e:
            return [_record(base, 'UNKNOWN', 'Comparison error', _error_text(e))]
        if not post_tracks:
            return [_record(base, 'UNKNOWN', 'Comparison error', 'no post inputs in tags')]
        try:
            df_expected = _enumerate_post_tracks(tile, [int(t) for t in post_tracks], str(acq.date()))
        except Exception as e:
            return [_record(base, 'UNKNOWN', 'Enumeration error', _error_text(e))]
        try:
            pol_by_opera_id = dict(zip(df_expected.opera_id_trunc, df_expected.polarizations))
            issues = [
                *_identity_issues(row.mgrs_tile_id, tile, acq, df_actual),
                *compare_inputs(df_actual, df_expected, pol_by_opera_id=pol_by_opera_id, check_single_pol=True),
                *_version_notes(df_actual, archive_ids_by_trunc(df_expected.opera_id)),
            ]
        except Exception as e:
            return [_record(base, 'UNKNOWN', 'Comparison error', _error_text(e))]
        return _product_records(base, issues)

    rows = list(df_tags.itertuples())
    with (
        Path('/dev/null').open('w') as devnull,
        redirect_stdout(devnull),
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        results = list(tqdm(pool.map(_one, rows), total=len(rows), desc='Checking inputs (online)'))
    return _results_frame([record for result in results for record in result])
