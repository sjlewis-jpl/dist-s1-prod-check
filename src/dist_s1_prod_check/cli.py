import shutil
from collections.abc import Callable
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

from dist_s1_prod_check import checks, coverage, inputs, manifest, report, status, tags
from dist_s1_prod_check.constants import (
    DIST_S1_PARQUET,
    DIST_S1_S3_BUCKETS,
    INPUTS_CHECKPOINT_PARQUET,
    RTC_CHUNKS_DIR,
    RTC_S1_PARQUET,
    TAGS_PARQUET,
    TARGET_TILES_FILE,
    VENUES,
    VENUE_FILE,
)
from dist_s1_prod_check.status import EVALUATED, SectionState, Status


def _stamp() -> str:
    return pd.Timestamp.now().strftime('%Y%m%dT%H%M%S')


def _load_dist(data_dir: Path) -> gpd.GeoDataFrame:
    path = data_dir / DIST_S1_PARQUET
    if not path.exists():
        raise click.ClickException(f'{path} not found; run `dist-s1-prod-check download-metadata` first.')
    return gpd.read_parquet(path)


def _venue_of(data_dir: Path) -> str | None:
    marker = data_dir / VENUE_FILE
    return marker.read_text().strip() if marker.exists() else None


def _load_cached(
    data_dir: Path, filename: str, run: '_Run', request: manifest.CacheRequest
) -> tuple[gpd.GeoDataFrame, manifest.TableManifest]:
    """Load a cached metadata table after checking its manifest (sha256, provenance, venue, window, AOI)."""
    path = data_dir / filename
    if not path.exists():
        raise click.ClickException(f'{path} not found; run `dist-s1-prod-check download-metadata` first.')
    m = manifest.load_table_manifest(path)
    manifest.require_compatible(m, request, path=path, regression_baseline=run.manifest.regression_baseline)
    run.manifest.tables[m.table] = m.to_dict()
    return gpd.read_parquet(path), m


def _load_dist_checked(data_dir: Path, run: '_Run') -> gpd.GeoDataFrame:
    """DIST-S1 table for a standalone check command (no window or AOI to match)."""
    df, _ = _load_cached(data_dir, DIST_S1_PARQUET, run, manifest.CacheRequest(venue=_venue_of(data_dir) or 'PROD'))
    return df


def _load_rtc_checked(
    data_dir: Path, run: '_Run', start: str | None = None, stop: str | None = None
) -> tuple[gpd.GeoDataFrame, manifest.TableManifest]:
    return _load_cached(data_dir, RTC_S1_PARQUET, run, manifest.CacheRequest('PROD', start, stop))


def _record_file(run: '_Run', path: Path) -> None:
    if path.exists():
        run.manifest.file_sha256[path.name] = manifest.sha256_file(path)


def _load_tags(data_dir: Path) -> pd.DataFrame:
    path = data_dir / TAGS_PARQUET
    if not path.exists():
        raise click.ClickException(f'{path} not found; run `dist-s1-prod-check fetch-tags` first.')
    return pd.read_parquet(path)


def _all_tile_geoms() -> gpd.GeoDataFrame:
    from dist_s1_enumerator.mgrs_burst_data import get_mgrs_table

    return get_mgrs_table()[['mgrs_tile_id', 'geometry']]


def _resolve_bboxes(bbox: tuple | None, data_dir: Path) -> tuple[tuple | None, tuple | None]:
    """Resolve an AOI bbox to (dist_bbox, rtc_bbox) and persist the target tile list for later checks."""
    from dist_s1_prod_check.cmr import resolve_target_tiles

    tiles_path = data_dir / TARGET_TILES_FILE
    if bbox is None:
        tiles_path.unlink(missing_ok=True)
        return None, None
    tile_ids, rtc_bbox = resolve_target_tiles(bbox)
    tiles_path.write_text('\n'.join(tile_ids))
    click.echo(f'AOI overlaps {len(tile_ids)} MGRS tiles; RTC-S1 bbox widened to {rtc_bbox}')
    return bbox, rtc_bbox


def _load_target_tiles(data_dir: Path) -> list[str] | None:
    tiles_path = data_dir / TARGET_TILES_FILE
    if not tiles_path.exists():
        return None
    return tiles_path.read_text().split()


def _build_rtc_frame(df_rtc: gpd.GeoDataFrame, data_dir: Path) -> gpd.GeoDataFrame:
    from dist_s1_prod_check.cmr import build_rtc_input_frame

    return build_rtc_input_frame(df_rtc, mgrs_tile_ids=_load_target_tiles(data_dir))


def _write_csvs(out_dir: Path, stamp: str, csvs: dict[str, pd.DataFrame]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, df in csvs.items():
        path = out_dir / f'{name}_{stamp}.csv'
        df.drop(columns=[c for c in ['geometry'] if c in df.columns]).to_csv(path, index=False)
        names.append(path.name)
    return names


def _write_report(
    out_dir: Path,
    stamp: str,
    tile_geoms: gpd.GeoDataFrame,
    sections: list[dict],
    title: str,
    subtitle: str,
    run_status: dict | None = None,
    provenance: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = report.write_html_report(
        sections,
        tile_geoms,
        out_dir / f'report_{stamp}.html',
        title,
        subtitle,
        run_status=run_status,
        provenance=provenance,
    )
    click.echo(f'Wrote {html_path}')
    return html_path


class _Run:
    """Track one command's run: `run_status.json`, CSVs and the HTML report, complete or partial.

    Writes RUNNING on creation. Inside ``with run:`` each check is wrapped by `stage` and recorded by `add`;
    `complete` writes the report and then COMPLETE as the very last step. Any exception leaving the block writes
    INCOMPLETE, then a partial report with an INCOMPLETE banner, and propagates unchanged; an error while writing the
    partial report is logged and never replaces the original exception.
    """

    def __init__(self, out_dir: Path, stamp: str, checks: list[str], title: str, regression_baseline: bool) -> None:
        self.out_dir, self.stamp, self.checks, self.title = out_dir, stamp, checks, title
        self.manifest = _new_run_manifest(regression_baseline)
        self.subtitle = ''
        self.current = 'setup'
        self.done: list[str] = []
        self.sections: list[dict] = []
        status.write_run_status(out_dir, status.RunState.RUNNING, checks_planned=checks, checks_done=[])

    def __enter__(self) -> '_Run':
        return self

    def stage(self, name: str) -> None:
        self.current = name

    def add(self, key: str, sections: list[dict], csvs: dict[str, pd.DataFrame]) -> None:
        for section in sections:
            click.echo(f'{section["title"]}: {section["n_failures"]} flagged of {section["n_checked"]} checked')
        self.sections.extend(sections)
        self.manifest.outputs.extend(_write_csvs(self.out_dir, self.stamp, csvs))
        self.done.append(key)

    def _status_dict(self, state: status.RunState, error: str | None = None) -> dict:
        failed = self.current if state != status.RunState.COMPLETE else None
        not_run = [c for c in self.checks if c not in self.done and c != failed]
        return {'state': str(state), 'failed_check': failed, 'error': error, 'not_run': not_run}

    def complete(self) -> None:
        self.current = 'report'
        self.manifest.outputs += [f'report_{self.stamp}.html', manifest.RUN_MANIFEST_FILE, status.RUN_STATUS_FILE]
        self.manifest.write(self.out_dir)
        _write_report(
            self.out_dir,
            self.stamp,
            _all_tile_geoms(),
            self.sections,
            self.title,
            self.subtitle,
            run_status=self._status_dict(status.RunState.COMPLETE),
            provenance=self.manifest.to_dict(),
        )
        status.write_run_status(
            self.out_dir, status.RunState.COMPLETE, checks_planned=self.checks, checks_done=self.done
        )

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: object) -> bool:
        if exc is None:
            return False
        error = f'{exc_type.__name__}: {exc}'
        status.write_run_status(
            self.out_dir,
            status.RunState.INCOMPLETE,
            checks_planned=self.checks,
            checks_done=self.done,
            failed_check=self.current,
            error=error,
        )
        try:
            self.manifest.write(self.out_dir)
            _write_report(
                self.out_dir,
                self.stamp,
                _all_tile_geoms(),
                self.sections,
                self.title,
                self.subtitle,
                run_status=self._status_dict(status.RunState.INCOMPLETE, error),
                provenance=self.manifest.to_dict(),
            )
        except Exception as report_exc:  # noqa: BLE001
            click.echo(f'Could not write the partial report: {type(report_exc).__name__}: {report_exc}', err=True)
        return False


def _json_param(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_param(v) for v in value]
    return value


def _new_run_manifest(regression_baseline: bool) -> manifest.RunManifest:
    """Return a run manifest for the click command being executed, with its parameters as given."""
    ctx = click.get_current_context(silent=True)
    command = ctx.command_path if ctx else 'dist-s1-prod-check'
    params = {k: _json_param(v) for k, v in (ctx.params if ctx else {}).items()}
    return manifest.RunManifest(
        command=command,
        params=params,
        offline=bool(params.get('offline')),
        regression_baseline=regression_baseline,
    )


def _run_single_check(
    out_dir: Path,
    key: str,
    title: str,
    build: Callable[['_Run'], tuple[list[dict], dict[str, pd.DataFrame]]],
    regression_baseline: bool = False,
) -> None:
    """Run one standalone `check-*` command under a tracked run."""
    with _Run(out_dir, _stamp(), [key], title, regression_baseline) as run:
        run.stage(key)
        sections, csvs = build(run)
        run.add(key, sections, csvs)
        run.complete()


RUN_ALL_CHECKS = ('duplicates', 'ordering', 'confirmation', 'inputs', 'coverage')

regression_option = click.option(
    '--regression-baseline',
    is_flag=True,
    help='Allow caches adopted with adopt-cache (provenance unverified). For regression runs only, never for a '
    'production-gate report; the report says so prominently.',
)
data_dir_option = click.option('--data-dir', type=click.Path(path_type=Path), default=Path('data'), show_default=True)
out_dir_option = click.option('--out-dir', type=click.Path(path_type=Path), default=Path('outputs'), show_default=True)
venue_option = click.option(
    '--venue',
    type=click.Choice(VENUES, case_sensitive=False),
    default='PROD',
    show_default=True,
    callback=lambda ctx, param, value: value.upper(),
    help="Venue of the DIST-S1 products: ASF PROD or UAT CMR, or OPS-POP1 to list them from that cluster's product "
    'bucket (RTC-S1 inputs always come from PROD).',
)


def _rtc_chunks_dir(data_dir: Path, rtc_bbox: tuple | None) -> Path:
    suffix = '_' + '_'.join(f'{v:g}' for v in rtc_bbox) if rtc_bbox else ''
    return data_dir / f'{RTC_CHUNKS_DIR}{suffix}'


def _dist_source(venue: str) -> str:
    return f'{venue} S3 product bucket' if venue in DIST_S1_S3_BUCKETS else f'{venue} CMR'


def _rtc_ranges(start: str, stop: str, lookback: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    from dist_s1_prod_check.cmr import lookback_date_ranges

    return lookback_date_ranges(start, stop) if lookback else [(pd.Timestamp(start), pd.Timestamp(stop))]


def _write_dist(
    df_dist: gpd.GeoDataFrame,
    data_dir: Path,
    venue: str,
    start: str,
    stop: str,
    bbox: tuple | None,
    dist_bbox: tuple | None,
) -> None:
    """Write the DIST-S1 table and its query receipt (the query just ran to completion)."""
    path = data_dir / DIST_S1_PARQUET
    df_dist.to_parquet(path, compression='zstd')
    manifest.write_table_manifest(
        path,
        table='dist_s1',
        venue=venue,
        source=_dist_source(venue),
        start=start,
        stop=stop,
        bbox=bbox,
        query_bbox=dist_bbox,
        target_tiles=_load_target_tiles(data_dir),
        lookback_windows=manifest.windows_with_counts([(start, stop)], df_dist.acq_time),
    )


def _write_rtc(
    df_rtc: gpd.GeoDataFrame,
    data_dir: Path,
    chunks_dir: Path,
    start: str,
    stop: str,
    bbox: tuple | None,
    rtc_bbox: tuple | None,
    lookback: bool = True,
) -> None:
    """Write the RTC-S1 table and its query receipt, recording every window queried (main and lookback)."""
    path = data_dir / RTC_S1_PARQUET
    df_rtc.to_parquet(path, compression='zstd')
    manifest.write_table_manifest(
        path,
        table='rtc_s1',
        venue='PROD',
        source='PROD CMR',
        start=start,
        stop=stop,
        bbox=bbox,
        query_bbox=rtc_bbox,
        target_tiles=_load_target_tiles(data_dir),
        lookback_windows=manifest.windows_with_counts(_rtc_ranges(start, stop, lookback), df_rtc.acq_dt),
    )
    shutil.rmtree(chunks_dir, ignore_errors=True)


def _filter_processing_time(
    df_dist: gpd.GeoDataFrame, processing_start: str | None, processing_stop: str | None
) -> gpd.GeoDataFrame:
    """Keep products processed in [processing_start, processing_stop), e.g. one campaign in a venue holding several."""
    keep = pd.Series(True, index=df_dist.index)
    if processing_start:
        keep &= df_dist.processing_time >= pd.Timestamp(processing_start)
    if processing_stop:
        keep &= df_dist.processing_time < pd.Timestamp(processing_stop)
    click.echo(f'Processing-time filter kept {int(keep.sum())} of {len(df_dist)} DIST-S1 products')
    return df_dist[keep].reset_index(drop=True)


def _check_venue(data_dir: Path, venue: str) -> None:
    """Refuse to reuse a data dir whose cached tables came from another venue (PROD/UAT product ids can collide)."""
    marker = data_dir / VENUE_FILE
    has_cache = any((data_dir / name).exists() for name in (DIST_S1_PARQUET, TAGS_PARQUET, INPUTS_CHECKPOINT_PARQUET))
    cached = marker.read_text().strip() if marker.exists() else ('PROD' if has_cache else venue)
    if cached != venue:
        raise click.ClickException(f'{data_dir} holds {cached} data; use a separate --data-dir for {venue}.')
    data_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f'{venue}\n')


@click.group()
def cli() -> None:
    """Validate the DIST-S1 production system."""


@cli.command()
@click.option('--start', required=True, help='Start of acquisition period, e.g. 2026-01-01.')
@click.option('--stop', required=True, help='End of acquisition period, e.g. 2026-03-01.')
@click.option('--bbox', type=float, nargs=4, default=None, help='AOI as lon_min lat_min lon_max lat_max.')
@click.option(
    '--lookback/--no-lookback', default=True, show_default=True, help='Also fetch RTC-S1 metadata for baseline windows.'
)
@click.option('--skip-rtc', is_flag=True, help='Only fetch DIST-S1 metadata.')
@click.option('--chunk-days', type=int, default=2, show_default=True)
@click.option('--workers', type=int, default=8, show_default=True)
@venue_option
@data_dir_option
def download_metadata(
    start: str,
    stop: str,
    bbox: tuple,
    lookback: bool,
    skip_rtc: bool,
    chunk_days: int,
    workers: int,
    venue: str,
    data_dir: Path,
) -> None:
    """Download DIST-S1 and RTC-S1 granule metadata from CMR into geoparquet tables (check 1)."""
    from dist_s1_prod_check import cmr

    _check_venue(data_dir, venue)
    bbox = tuple(bbox) if bbox else None
    dist_bbox, rtc_bbox = _resolve_bboxes(bbox, data_dir)

    manifest.manifest_path(data_dir / DIST_S1_PARQUET).unlink(missing_ok=True)
    df_dist = cmr.get_dist_s1_table(
        start, stop, bbox=dist_bbox, chunk_days=max(chunk_days, 7), max_workers=workers, venue=venue
    )
    _write_dist(df_dist, data_dir, venue, start, stop, bbox, dist_bbox)
    click.echo(f'DIST-S1: {len(df_dist)} granules -> {data_dir / DIST_S1_PARQUET}')

    if skip_rtc:
        return
    chunks_dir = _rtc_chunks_dir(data_dir, rtc_bbox)
    manifest.manifest_path(data_dir / RTC_S1_PARQUET).unlink(missing_ok=True)
    if lookback:
        df_rtc = cmr.get_rtc_s1_table_with_lookback(
            start, stop, bbox=rtc_bbox, chunk_days=chunk_days, max_workers=workers, cache_dir=chunks_dir
        )
    else:
        df_rtc = cmr.get_rtc_s1_table(
            start, stop, bbox=rtc_bbox, chunk_days=chunk_days, max_workers=workers, cache_dir=chunks_dir
        )
    _write_rtc(df_rtc, data_dir, chunks_dir, start, stop, bbox, rtc_bbox, lookback)
    click.echo(f'RTC-S1: {len(df_rtc)} granules -> {data_dir / RTC_S1_PARQUET}')


@cli.command()
@click.option('--start', required=True, help='--start of the run that built the cache.')
@click.option('--stop', required=True, help='--stop of the run that built the cache.')
@click.option('--bbox', type=float, nargs=4, default=None, help='--bbox of the run that built the cache, if any.')
@click.option(
    '--lookback/--no-lookback',
    default=True,
    show_default=True,
    help='Whether the RTC-S1 table was built with the baseline lookback windows (run-all always does).',
)
@click.option(
    '--rtc-source', default='PROD CMR', show_default=True, help='Where the RTC-S1 table came from, as recorded.'
)
@venue_option
@data_dir_option
def adopt_cache(
    start: str, stop: str, bbox: tuple, lookback: bool, rtc_source: str, venue: str, data_dir: Path
) -> None:
    """Write manifests for a cache built before manifests existed, from the arguments that built it.

    The result is `adopted_unverified`: it records what you claim the cache holds, not what was queried or that
    the query finished. Adopted caches are accepted only with --regression-baseline and never back a
    production-gate report.
    """
    bbox = tuple(bbox) if bbox else None
    cached_venue = _venue_of(data_dir)
    if cached_venue and cached_venue != venue:
        raise click.ClickException(f'{data_dir} holds {cached_venue} data, not {venue}.')
    dist_path, rtc_path = data_dir / DIST_S1_PARQUET, data_dir / RTC_S1_PARQUET
    if not dist_path.exists():
        raise click.ClickException(f'{dist_path} not found; nothing to adopt.')
    tiles = _load_target_tiles(data_dir)
    common = {'start': start, 'stop': stop, 'bbox': bbox, 'target_tiles': tiles}
    common |= {'provenance': manifest.ADOPTED_UNVERIFIED}
    df_dist = pd.read_parquet(dist_path, columns=['acq_time'])
    m = manifest.write_table_manifest(
        dist_path,
        table='dist_s1',
        venue=venue,
        source=_dist_source(venue),
        lookback_windows=manifest.windows_with_counts([(start, stop)], df_dist.acq_time),
        **common,
    )
    click.echo(f'Adopted {dist_path} ({m.n_rows} rows, provenance {m.provenance})')
    if rtc_path.exists():
        df_rtc = pd.read_parquet(rtc_path, columns=['acq_dt'])
        m = manifest.write_table_manifest(
            rtc_path,
            table='rtc_s1',
            venue='PROD',
            source=rtc_source,
            lookback_windows=manifest.windows_with_counts(_rtc_ranges(start, stop, lookback), df_rtc.acq_dt),
            **common,
        )
        click.echo(f'Adopted {rtc_path} ({m.n_rows} rows, {len(m.lookback_windows)} query windows claimed complete)')
    click.echo('Adopted caches are usable only with --regression-baseline.')


tag_url_template_option = click.option(
    '--tag-url-template',
    default=None,
    help='Read product tags from this location instead of the DAAC URL, formatted per product, e.g. '
    "'s3://BUCKET/PREFIX/{opera_id}/{opera_id}_GEN-DIST-STATUS.tif' (fields: opera_id, mgrs_tile_id, "
    'acq_time, processing_time).',
)


@cli.command()
@click.option('--workers', type=int, default=16, show_default=True)
@click.option('--sample', type=int, default=0, show_default=True, help='Randomly sample N products (0 = all).')
@click.option('--seed', type=int, default=42, show_default=True)
@tag_url_template_option
@data_dir_option
def fetch_tags(workers: int, sample: int, seed: int, tag_url_template: str | None, data_dir: Path) -> None:
    """Fetch product GeoTIFF tags (inputs, prior product) from the DAAC; resumable."""
    df_dist = checks.deduplicate_products(_load_dist(data_dir))
    if sample:
        df_dist = df_dist.sample(n=min(sample, len(df_dist)), random_state=seed)
    df_tags = tags.fetch_tags_table(
        df_dist, data_dir / TAGS_PARQUET, max_workers=workers, url_template=tag_url_template
    )
    n_err = int((df_tags.error != '').sum())
    click.echo(f'Tags: {len(df_tags)} fetched, {n_err} errors -> {data_dir / TAGS_PARQUET}')


def _wkt_by_tile(tile_geoms: gpd.GeoDataFrame) -> dict[str, str]:
    from shapely import to_wkt

    return {row.mgrs_tile_id: to_wkt(row.geometry, rounding_precision=4) for row in tile_geoms.itertuples()}


def _append_tile_wkt(df: pd.DataFrame, wkt_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df['mgrs_tile_wkt'] = df.mgrs_tile_id.map(wkt_map)
    return df


def _vertex_search_url(wkt: str | None, post_acq_dt: pd.Timestamp) -> str:
    from urllib.parse import quote

    from shapely import from_wkt, to_wkt

    if not isinstance(wkt, str) or not wkt:
        return ''
    geom = from_wkt(wkt)
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda g: g.area)
    start = (post_acq_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end = (post_acq_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    vertex_wkt = to_wkt(geom, rounding_precision=4).replace(', ', ',').replace('POLYGON (', 'POLYGON(')
    return (
        f'https://search.asf.alaska.edu/#/?dataset=OPERA-S1&productTypes=RTC'
        f'&polygon={quote(vertex_wkt)}&start={start}&end={end}&resultsLoaded=true'
    )


def _issue_summary(df_bad: pd.DataFrame) -> dict | None:
    if df_bad.empty:
        return None
    counts = df_bad.issue_type.value_counts().reset_index()
    counts.columns = ['issue_type', 'n_rows']
    return {
        'title': 'Issue types (click a row to filter the table above)',
        'columns': ['issue_type', 'n_rows'],
        'records': counts.to_dict('records'),
    }


def _duplicates_section(
    df_dist: gpd.GeoDataFrame, wkt_map: dict[str, str], state: SectionState = SectionState.COMPLETE
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_dup = _append_tile_wkt(checks.check_duplicates(df_dist), wkt_map)
    section = report.build_section(
        'duplicates',
        'Duplicates',
        'DIST-S1 granules sharing the same tile and acquisition time. The last-processed granule of '
        'each group is taken as the correct one; every other check runs on that granule only.',
        df_dup,
        len(df_dist),
        state=state,
    )
    return [section], {'duplicates': df_dup}


def _ordering_section(
    df_dedup: pd.DataFrame, wkt_map: dict[str, str], state: SectionState = SectionState.COMPLETE
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_order = checks.check_processing_order(df_dedup)
    df_bad = _append_tile_wkt(df_order[df_order.out_of_order], wkt_map)
    n_reprocessed = int(df_order.is_reprocessed.sum())
    n_evaluated = int(df_order.evaluated.sum())
    section = report.build_section(
        'ordering',
        'Processing order',
        'Products processed before the previous acquisition in their MGRS tile. Excluded from this '
        f'check: {n_reprocessed} reprocessed acquisitions (see Duplicates - their retained version is '
        'processed late by construction) and the first evaluated product of each tile.',
        df_bad,
        n_evaluated,
        state=state,
        n_universe=len(df_order),
        n_skipped=len(df_order) - n_evaluated,
    )
    return [section], {'ordering_failures': df_bad, 'ordering': df_order}


def _confirmation_section(
    df_dedup: pd.DataFrame,
    df_tags: pd.DataFrame,
    expect_none_at_start: bool,
    wkt_map: dict[str, str],
    state: SectionState = SectionState.COMPLETE,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_conf = checks.check_confirmation(df_dedup, df_tags, expect_none_at_start=expect_none_at_start)
    df_conf = df_conf[df_conf.opera_id.isin(df_tags.opera_id)]
    df_bad = _append_tile_wkt(df_conf[~df_conf.confirmed_ok], wkt_map)
    # A first-in-window product has no predecessor to compare against unless the window starts the mission.
    skipped = df_conf.is_first_in_window & ~df_conf.tag_missing & (not expect_none_at_start)
    n_skipped = int(skipped.sum())
    n_fail = int((~df_conf.confirmed_ok).sum())
    section = report.build_section(
        'confirmation',
        'Confirmation chain',
        'Products whose prior_dist_s1_product tag does not point to the previous product in the tile time series. '
        'The first product of each tile in the window has no predecessor to check and is skipped unless '
        '--expect-none-at-start is set.',
        df_bad,
        len(df_conf) - n_skipped,
        state=state,
        n_universe=len(df_dedup),
        n_skipped=n_skipped,
        counts={Status.PASS: len(df_conf) - n_skipped - n_fail, Status.FAIL: n_fail},
    )
    tiles = checks.summarize_by_tile(df_conf.assign(fail=~df_conf.confirmed_ok), 'fail')
    return [section], {'confirmation_failures': df_bad, 'confirmation_tiles': tiles}


def _inputs_section(
    df_tags: pd.DataFrame,
    df_frame: gpd.GeoDataFrame | None,
    mode: str,
    workers: int,
    wkt_map: dict[str, str],
    checkpoint_path: Path | None = None,
    state: SectionState = SectionState.COMPLETE,
    n_universe: int | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    if mode == 'offline':
        df_results = inputs.check_inputs_offline(df_tags, df_frame, checkpoint_path=checkpoint_path, workers=workers)
    else:
        df_results = inputs.check_inputs_online(df_tags, max_workers=workers)
    if df_results.empty:
        df_results = pd.DataFrame(columns=inputs.RESULT_COLUMNS)
    df_bad = _append_tile_wkt(df_results[df_results.inputs_correct != True], wkt_map)  # noqa: E712
    product_status = _inputs_product_status(df_results)
    counts = product_status.value_counts().to_dict()
    n_checked = int(product_status.isin(EVALUATED).sum())
    n_universe = len(product_status) if n_universe is None else n_universe

    def _counts(df_subset: pd.DataFrame) -> dict[str, int]:
        n_bad = df_subset.opera_id.nunique()
        return {Status.FAIL: n_bad, Status.PASS: max(n_checked - n_bad, 0)}

    common = {'state': state, 'n_universe': n_universe}

    post_mask = df_bad.issue_type.str.contains('Post', case=False)
    pre_mask = df_bad.issue_type.str.contains('Pre', case=False) & ~post_mask
    sections = [
        report.build_section(
            'inputs',
            'Product inputs',
            'Products whose recorded pre/post RTC-S1 inputs differ from the re-enumerated expected inputs.',
            df_bad,
            n_checked,
            summary=_issue_summary(df_bad),
            counts=counts,
            **common,
        ),
        report.build_section(
            'inputs_post',
            'Inputs: post issues',
            'Subset of product-input issues involving post RTC-S1 inputs (wrong or missing post images).',
            df_bad[post_mask],
            n_checked,
            summary=_issue_summary(df_bad[post_mask]),
            counts=_counts(df_bad[post_mask]),
            **common,
        ),
        report.build_section(
            'inputs_pre',
            'Inputs: pre issues',
            'Subset of product-input issues involving pre (baseline) RTC-S1 inputs.',
            df_bad[pre_mask],
            n_checked,
            summary=_issue_summary(df_bad[pre_mask]),
            counts=_counts(df_bad[pre_mask]),
            **common,
        ),
    ]
    csvs = {
        'inputs_failures': df_bad,
        'inputs_failures_post': df_bad[post_mask],
        'inputs_failures_pre': df_bad[pre_mask],
        'inputs': df_results,
    }
    return sections, csvs


def _inputs_product_status(df_results: pd.DataFrame) -> pd.Series:
    """One status per product: UNKNOWN if any row could not be checked, else FAIL if any row failed, else PASS."""
    if df_results.empty:
        return pd.Series(dtype=str)
    value = df_results.inputs_correct.map({True: Status.PASS, False: Status.FAIL}).fillna(Status.UNKNOWN)
    rank = value.map({Status.UNKNOWN: 0, Status.FAIL: 1, Status.PASS: 2})
    return value.groupby(df_results.opera_id).agg(lambda v: v.loc[rank[v.index].idxmin()])


def _coverage_section(
    df_dedup: pd.DataFrame,
    df_frame: gpd.GeoDataFrame,
    start: str,
    stop: str,
    wkt_map: dict[str, str],
    state: SectionState = SectionState.COMPLETE,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_groups = coverage.expected_pass_groups(df_frame, start, stop)
    df_cov = coverage.check_coverage(df_groups, df_dedup, df_rtc_frame=df_frame)
    df_missing = df_cov[~df_cov.product_found & df_cov.missing_product_expected]
    df_missing = _append_tile_wkt(df_missing.drop(columns=['burst_ids', 'burst_pols'], errors='ignore'), wkt_map)
    df_missing['asf_search_url'] = [
        _vertex_search_url(row.mgrs_tile_wkt, row.post_acq_dt) for row in df_missing.itertuples()
    ]
    df_orphans = coverage.check_orphan_products(df_groups, df_dedup)
    n_found = int(df_cov.product_found.sum()) if len(df_cov) else 0

    summary = None
    df_by_tile = pd.DataFrame(columns=['mgrs_tile_id', 'n_missing', 'missing_post_dates'])
    if not df_missing.empty:
        df_by_tile = (
            df_missing.groupby('mgrs_tile_id')
            .agg(
                n_missing=('post_acq_dt', 'size'),
                missing_post_dates=('post_acq_dt', lambda s: ';'.join(pd.to_datetime(s).dt.strftime('%Y-%m-%d'))),
            )
            .reset_index()
            .sort_values('n_missing', ascending=False)
        )
        summary = {
            'title': 'Missing products by tile (click a row to filter the table above)',
            'columns': ['mgrs_tile_id', 'n_missing', 'missing_post_dates'],
            'records': df_by_tile.to_dict('records'),
        }
    section = report.build_section(
        'coverage',
        'Coverage',
        'RTC-S1 passes with baseline imagery that should have triggered a DIST-S1 product but have none.',
        df_missing,
        n_found + len(df_missing),
        summary=summary,
        state=state,
        unit='passes',
        n_universe=len(df_cov),
        n_skipped=len(df_cov) - n_found - len(df_missing),
        counts={Status.PASS: n_found, Status.FAIL: len(df_missing)},
    )
    csvs = {
        'coverage_missing': df_missing,
        'coverage_missing_by_tile': df_by_tile,
        'coverage': df_cov,
        'coverage_orphans': df_orphans,
    }
    return [section], csvs


@cli.command()
@regression_option
@data_dir_option
@out_dir_option
def check_duplicates(data_dir: Path, out_dir: Path, regression_baseline: bool) -> None:
    """Check for duplicated DIST-S1 products (check 2)."""
    _run_single_check(
        out_dir,
        'duplicates',
        'DIST-S1 duplicates',
        lambda run: _duplicates_section(_load_dist_checked(data_dir, run), _wkt_by_tile(_all_tile_geoms())),
        regression_baseline,
    )


@cli.command()
@regression_option
@data_dir_option
@out_dir_option
def check_ordering(data_dir: Path, out_dir: Path, regression_baseline: bool) -> None:
    """Check processing order matches acquisition order (check 3)."""
    _run_single_check(
        out_dir,
        'ordering',
        'DIST-S1 processing order',
        lambda run: _ordering_section(
            checks.deduplicate_products(_load_dist_checked(data_dir, run)), _wkt_by_tile(_all_tile_geoms())
        ),
        regression_baseline,
    )


@cli.command()
@click.option(
    '--expect-none-at-start', is_flag=True, help='Require the first product per tile to have no prior (mission start).'
)
@regression_option
@data_dir_option
@out_dir_option
def check_confirmation(expect_none_at_start: bool, data_dir: Path, out_dir: Path, regression_baseline: bool) -> None:
    """Check the confirmation chain via prior product tags (check 4)."""
    _run_single_check(
        out_dir,
        'confirmation',
        'DIST-S1 confirmation chain',
        lambda run: _confirmation_section(
            checks.deduplicate_products(_load_dist_checked(data_dir, run)),
            _load_tags(data_dir),
            expect_none_at_start,
            _wkt_by_tile(_all_tile_geoms()),
        ),
        regression_baseline,
    )


@cli.command()
@click.option('--mode', type=click.Choice(['offline', 'online']), default='offline', show_default=True)
@click.option('--sample', type=int, default=0, show_default=True, help='Randomly sample N products (0 = all).')
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--workers', type=int, default=8, show_default=True)
@regression_option
@data_dir_option
@out_dir_option
def check_inputs(
    mode: str, sample: int, seed: int, workers: int, data_dir: Path, out_dir: Path, regression_baseline: bool
) -> None:
    """Check product inputs against re-enumerated expected inputs (check 5)."""

    def build(run: _Run) -> tuple[list[dict], dict[str, pd.DataFrame]]:
        df_dist = _load_dist_checked(data_dir, run)
        df_tags = _load_tags(data_dir)
        df_tags = df_tags[df_tags.opera_id.isin(checks.deduplicate_products(df_dist).opera_id)]
        if sample:
            df_tags = df_tags.sample(n=min(sample, len(df_tags)), random_state=seed)
        df_frame = _build_rtc_frame(_load_rtc_checked(data_dir, run)[0], data_dir) if mode == 'offline' else None
        return _inputs_section(
            df_tags,
            df_frame,
            mode,
            workers,
            _wkt_by_tile(_all_tile_geoms()),
            checkpoint_path=data_dir / INPUTS_CHECKPOINT_PARQUET,
        )

    _run_single_check(out_dir, 'inputs', 'DIST-S1 product inputs', build, regression_baseline)


@cli.command()
@click.option('--start', required=True)
@click.option('--stop', required=True)
@regression_option
@data_dir_option
@out_dir_option
def check_coverage(start: str, stop: str, data_dir: Path, out_dir: Path, regression_baseline: bool) -> None:
    """Check all triggerable DIST-S1 products exist given available RTC-S1 inputs (check 6)."""
    _run_single_check(
        out_dir,
        'coverage',
        'DIST-S1 coverage',
        lambda run: _coverage_section(
            checks.deduplicate_products(_load_dist_checked(data_dir, run)),
            _build_rtc_frame(_load_rtc_checked(data_dir, run, start, stop)[0], data_dir),
            start,
            stop,
            _wkt_by_tile(_all_tile_geoms()),
        ),
        regression_baseline,
    )


@cli.command()
@click.option('--start', required=True, help='Start of acquisition period, e.g. 2026-01-01.')
@click.option('--stop', required=True, help='End of acquisition period, e.g. 2026-03-01.')
@click.option('--bbox', type=float, nargs=4, default=None)
@click.option('--workers', type=int, default=16, show_default=True)
@click.option('--cmr-workers', type=int, default=8, show_default=True)
@click.option(
    '--inputs-workers',
    type=int,
    default=8,
    show_default=True,
    help='Processes for the offline inputs check (the long stage).',
)
@click.option(
    '--sample-inputs', type=int, default=0, show_default=True, help='Sample N products for the inputs check (0 = all).'
)
@click.option(
    '--sample-tiles',
    type=int,
    default=0,
    show_default=True,
    help='Restrict every check to N random MGRS tiles (full time series per tile) for fast iteration (0 = all).',
)
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--expect-none-at-start', is_flag=True)
@click.option('--refresh', is_flag=True, help='Re-download metadata tables even if present.')
@click.option(
    '--processing-start',
    default=None,
    help='Only check DIST-S1 products processed at or after this time, e.g. 2026-09-24 (isolates one campaign).',
)
@click.option('--processing-stop', default=None, help='Only check DIST-S1 products processed before this time.')
@tag_url_template_option
@venue_option
@regression_option
@data_dir_option
@out_dir_option
def run_all(
    start: str,
    stop: str,
    bbox: tuple,
    workers: int,
    cmr_workers: int,
    inputs_workers: int,
    sample_inputs: int,
    sample_tiles: int,
    seed: int,
    expect_none_at_start: bool,
    refresh: bool,
    processing_start: str | None,
    processing_stop: str | None,
    tag_url_template: str | None,
    venue: str,
    regression_baseline: bool,
    data_dir: Path,
    out_dir: Path,
) -> None:
    """Run checks 1-6 end to end: download metadata, fetch tags, run every check, write CSVs and an HTML report."""
    from dist_s1_prod_check import cmr

    with _Run(out_dir, _stamp(), list(RUN_ALL_CHECKS), 'DIST-S1 Production Check', regression_baseline) as run:
        _check_venue(data_dir, venue)
        bbox = tuple(bbox) if bbox else None

        need_dist = refresh or not (data_dir / DIST_S1_PARQUET).exists()
        need_rtc = refresh or not (data_dir / RTC_S1_PARQUET).exists()
        if need_dist or need_rtc:
            (data_dir / INPUTS_CHECKPOINT_PARQUET).unlink(missing_ok=True)
            for name, needed in ((DIST_S1_PARQUET, need_dist), (RTC_S1_PARQUET, need_rtc)):
                if needed:
                    manifest.manifest_path(data_dir / name).unlink(missing_ok=True)
            dist_bbox, rtc_bbox = _resolve_bboxes(bbox, data_dir)
            chunks_dir = _rtc_chunks_dir(data_dir, rtc_bbox)
            if refresh:
                shutil.rmtree(chunks_dir, ignore_errors=True)
            if need_dist:
                df_dist = cmr.get_dist_s1_table(start, stop, bbox=dist_bbox, max_workers=cmr_workers, venue=venue)
                _write_dist(df_dist, data_dir, venue, start, stop, bbox, dist_bbox)
            if need_rtc:
                df_rtc = cmr.get_rtc_s1_table_with_lookback(
                    start, stop, bbox=rtc_bbox, max_workers=cmr_workers, cache_dir=chunks_dir
                )
                _write_rtc(df_rtc, data_dir, chunks_dir, start, stop, bbox, rtc_bbox)
                del df_rtc
        request = manifest.CacheRequest(venue, start, stop, bbox, check_bbox=True)
        df_dist, m_dist = _load_cached(data_dir, DIST_S1_PARQUET, run, request)
        if manifest.strictly_inside(m_dist, start, stop):
            acq = pd.to_datetime(df_dist.acq_time, utc=True)
            in_window = (acq >= manifest._utc(start)) & (acq < manifest._utc(stop))
            click.echo(f'Window filter kept {int(in_window.sum())} of {len(df_dist)} cached DIST-S1 products')
            df_dist = df_dist[in_window].reset_index(drop=True)
        if processing_start or processing_stop:
            df_dist = _filter_processing_time(df_dist, processing_start, processing_stop)
        if sample_tiles:
            all_tiles = pd.Series(sorted(df_dist.mgrs_tile_id.unique()))
            keep = set(all_tiles.sample(n=min(sample_tiles, len(all_tiles)), random_state=seed))
            df_dist = df_dist[df_dist.mgrs_tile_id.isin(keep)].reset_index(drop=True)
            click.echo(f'Sampled {len(keep)} of {len(all_tiles)} tiles')
        click.echo(f'DIST-S1 products: {len(df_dist)}')
        df_rtc, _ = _load_cached(
            data_dir, RTC_S1_PARQUET, run, manifest.CacheRequest('PROD', start, stop, bbox, check_bbox=True)
        )
        click.echo(f'RTC-S1 granules: {len(df_rtc)}')
        from dist_s1_prod_check.cmr import build_rtc_input_frame

        frame_tiles = sorted(keep) if sample_tiles else _load_target_tiles(data_dir)
        df_frame = build_rtc_input_frame(df_rtc, mgrs_tile_ids=frame_tiles)

        df_dedup = checks.deduplicate_products(df_dist)
        df_tags_products = (
            df_dedup if not sample_inputs else df_dedup.sample(n=min(sample_inputs, len(df_dedup)), random_state=42)
        )
        run.subtitle = (
            f'{start} to {stop}'
            + (f' over {bbox}' if bbox else ' (global)')
            + (f' - sample of {sample_tiles} tiles' if sample_tiles else '')
            + (f' - {venue} venue' if venue != 'PROD' else '')
            + (
                f' - processed {processing_start or "start"} to {processing_stop or "now"}'
                if processing_start or processing_stop
                else ''
            )
        )
        run.stage('tags')
        df_tags = tags.fetch_tags_table(
            df_tags_products, data_dir / TAGS_PARQUET, max_workers=workers, url_template=tag_url_template
        )
        df_tags = df_tags[df_tags.opera_id.isin(df_tags_products.opera_id)]
        _record_file(run, data_dir / TAGS_PARQUET)

        wkt_map = _wkt_by_tile(_all_tile_geoms())
        builds = {
            'duplicates': lambda: _duplicates_section(df_dist, wkt_map),
            'ordering': lambda: _ordering_section(df_dedup, wkt_map),
            'confirmation': lambda: _confirmation_section(df_dedup, df_tags, expect_none_at_start, wkt_map),
            'inputs': lambda: _inputs_section(
                df_tags,
                df_frame,
                'offline',
                inputs_workers,
                wkt_map,
                checkpoint_path=data_dir / INPUTS_CHECKPOINT_PARQUET,
            ),
            'coverage': lambda: _coverage_section(df_dedup, df_frame, start, stop, wkt_map),
        }
        for key in RUN_ALL_CHECKS:
            run.stage(key)
            run.add(key, *builds[key]())
        _record_file(run, data_dir / INPUTS_CHECKPOINT_PARQUET)
        run.complete()
