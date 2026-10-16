import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

from dist_s1_prod_check import checks, coverage, inputs, manifest, matching, network, report, schema, status, tags
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
    return schema.validate(gpd.read_parquet(path), schema.DIST_SCHEMA, path)


SCHEMAS = {DIST_S1_PARQUET: schema.DIST_SCHEMA, RTC_S1_PARQUET: schema.RTC_SCHEMA}


def _venue_of(data_dir: Path) -> str | None:
    marker = data_dir / VENUE_FILE
    return marker.read_text().strip() if marker.exists() else None


def _has_cache(data_dir: Path) -> bool:
    return any((data_dir / name).exists() for name in (DIST_S1_PARQUET, TAGS_PARQUET, INPUTS_CHECKPOINT_PARQUET))


def _require_venue(data_dir: Path) -> str | None:
    """Venue recorded for a data dir; refuses a cache with no `venue.txt` instead of assuming PROD (F28)."""
    venue = _venue_of(data_dir)
    if venue is None and _has_cache(data_dir):
        raise click.ClickException(
            f'{data_dir} holds cached tables but no {VENUE_FILE}, so their venue is unknown (PROD and UAT product '
            f'ids can collide). Re-download with --refresh, or run `adopt-cache --venue VENUE` to record it.'
        )
    return venue


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
    return schema.validate(gpd.read_parquet(path), SCHEMAS[filename], path), m


def _filter_versions(df_dist: gpd.GeoDataFrame, product_versions: str, run: '_Run | None') -> gpd.GeoDataFrame:
    """Keep the allowed product versions (decision 6) and record what was excluded in the run manifest."""
    allowed = [v.strip() for v in product_versions.split(',') if v.strip()]
    df_kept, excluded = checks.filter_product_versions(df_dist, allowed)
    n_excluded = sum(excluded.values())
    if run is not None:
        run.manifest.version_filter = {'allowed': allowed, 'n_excluded': n_excluded, 'excluded_by_version': excluded}
    click.echo(f'Product-version filter ({", ".join(allowed)}) kept {len(df_kept)} of {len(df_dist)} DIST-S1 products')
    return df_kept


def _version_note(run: '_Run') -> str:
    vf = run.manifest.version_filter
    if not vf.get('n_excluded'):
        return ''
    detail = ', '.join(f'{v}: {n}' for v, n in vf['excluded_by_version'].items())
    return f' - excluded {vf["n_excluded"]} products of other versions ({detail})'


def _load_dist_checked(data_dir: Path, run: '_Run', product_versions: str) -> gpd.GeoDataFrame:
    """DIST-S1 table for a standalone check command (no window or AOI to match), version-filtered."""
    df, _ = _load_cached(
        data_dir, DIST_S1_PARQUET, run, manifest.CacheRequest(venue=_require_venue(data_dir) or 'PROD')
    )
    df = _filter_versions(df, product_versions, run)
    run.subtitle = _version_note(run).removeprefix(' - ')
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


def _build_rtc_frame(df_rtc: gpd.GeoDataFrame, run: '_Run', mgrs_tile_ids: list[str] | None) -> gpd.GeoDataFrame:
    """Build the enumerator input frame for `mgrs_tile_ids`; count the RTC-S1 rows it drops in the run manifest."""
    from dist_s1_prod_check.cmr import build_rtc_input_frame_with_drops

    df_frame, drops = build_rtc_input_frame_with_drops(df_rtc, mgrs_tile_ids=mgrs_tile_ids)
    run.manifest.rtc_frame_drops = drops
    click.echo('RTC-S1 rows dropped building the input frame: ' + ', '.join(f'{k} {v}' for k, v in drops.items()))
    return df_frame


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
        self.manifest.outputs.extend(_write_csvs(self.out_dir, self.stamp, csvs) or [])
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
            self._status_dict(
                status.RunState.COMPLETE
            ),  # positional: tests capture _write_report(out_dir, stamp, tile_geoms, sections, *a)
            self.manifest.to_dict(),
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
                self._status_dict(
                    status.RunState.INCOMPLETE, error
                ),  # positional: tests capture _write_report(out_dir, stamp, tile_geoms, sections, *a)
                self.manifest.to_dict(),
            )
        except Exception as report_exc:  # noqa: BLE001
            click.echo(f'Could not write the partial report: {type(report_exc).__name__}: {report_exc}', err=True)
        return False


def _sampling_record(n_universe: int, selected: 'pd.Series | list[str]', seed: int, unit: str) -> dict:
    """Describe a sample for the run manifest: sizes, seed and a digest of the selected ids."""
    ids = sorted(map(str, selected))
    digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    return {
        'n_universe': int(n_universe),
        'n_selected': len(ids),
        'seed': seed,
        'unit': unit,
        'selected_sha256': digest,
    }


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


def _set_offline(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    network.set_offline(value)
    return value


offline_option = click.option(
    '--offline',
    is_flag=True,
    callback=_set_offline,
    help='Use cached data only. Any step that needs the network (metadata download, chunk re-query, tag fetch for '
    'a product with no cached row, online inputs mode) fails with an error naming it; cached tag-read errors are '
    'kept as UNKNOWN, not retried.',
)
product_versions_option = click.option(
    '--product-versions',
    default='v1.0',
    show_default=True,
    help='Comma-separated DIST-S1 product versions to check; other versions are excluded and counted in the report.',
)
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
    chunks: list[dict] | None = None,
) -> None:
    """Write the DIST-S1 table and its query receipt (the query just ran to completion), with its query chunks."""
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
        chunks=chunks,
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
    chunks: list[dict] | None = None,
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
        chunks=chunks,
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
    cached = _require_venue(data_dir) or venue
    if cached != venue:
        raise click.ClickException(f'{data_dir} holds {cached} data; use a separate --data-dir for {venue}.')
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / VENUE_FILE).write_text(f'{venue}\n')


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
@offline_option
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
    offline: bool,
    data_dir: Path,
) -> None:
    """Download DIST-S1 and RTC-S1 granule metadata from CMR into geoparquet tables (check 1)."""
    from dist_s1_prod_check import cmr

    network.network_allowed('download-metadata')
    _check_venue(data_dir, venue)
    # New metadata makes every checkpointed inputs result stale, as in run-all (F04).
    (data_dir / INPUTS_CHECKPOINT_PARQUET).unlink(missing_ok=True)
    bbox = tuple(bbox) if bbox else None
    dist_bbox, rtc_bbox = _resolve_bboxes(bbox, data_dir)

    manifest.manifest_path(data_dir / DIST_S1_PARQUET).unlink(missing_ok=True)
    dist_chunks: list[dict] = []
    df_dist = cmr.get_dist_s1_table(
        start,
        stop,
        bbox=dist_bbox,
        chunk_days=max(chunk_days, 7),
        max_workers=workers,
        venue=venue,
        chunk_log=dist_chunks,
    )
    _write_dist(df_dist, data_dir, venue, start, stop, bbox, dist_bbox, dist_chunks)
    click.echo(f'DIST-S1: {len(df_dist)} granules -> {data_dir / DIST_S1_PARQUET}')

    if skip_rtc:
        return
    chunks_dir = _rtc_chunks_dir(data_dir, rtc_bbox)
    manifest.manifest_path(data_dir / RTC_S1_PARQUET).unlink(missing_ok=True)
    rtc_chunks: list[dict] = []
    fetch = cmr.get_rtc_s1_table_with_lookback if lookback else cmr.get_rtc_s1_table
    df_rtc = fetch(
        start,
        stop,
        bbox=rtc_bbox,
        chunk_days=chunk_days,
        max_workers=workers,
        cache_dir=chunks_dir,
        chunk_log=rtc_chunks,
    )
    _write_rtc(df_rtc, data_dir, chunks_dir, start, stop, bbox, rtc_bbox, lookback, rtc_chunks)
    click.echo(f'RTC-S1: {len(df_rtc)} granules -> {data_dir / RTC_S1_PARQUET}')


def _adopt_tag_sources(data_dir: Path, url_template: str | None) -> None:
    """Fill a missing `source_url` in the cached tag table with the URL each product resolves to (asserted)."""
    tags_path = data_dir / TAGS_PARQUET
    if not tags_path.exists():
        return
    df_tags = pd.read_parquet(tags_path)
    current = tags.cached_source_urls(df_tags)
    df_dist = pd.read_parquet(data_dir / DIST_S1_PARQUET)
    resolved = tags.resolve_tag_urls(df_dist, url_template)
    url_by_id = pd.Series(resolved.to_numpy(), index=df_dist.opera_id.to_numpy())
    fill = (current == '') & df_tags.opera_id.isin(url_by_id.dropna().index)
    df_tags['source_url'] = current.where(~fill, df_tags.opera_id.map(url_by_id))
    tmp = tags_path.with_suffix('.parquet.tmp')
    df_tags.to_parquet(tmp, compression='zstd')
    tmp.replace(tags_path)
    source = f'template {url_template}' if url_template else 'alert_url'
    click.echo(
        f'Recorded source_url ({source}) for {int(fill.sum())} of {len(df_tags)} cached tag rows (asserted, not '
        f'verified); {int((df_tags.source_url == "").sum())} rows have none and will be refetched.'
    )


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
@click.option(
    '--tag-url-template',
    default=None,
    help="--tag-url-template of the run that fetched the cached tags, if any; recorded as each row's source_url.",
)
@venue_option
@data_dir_option
def adopt_cache(
    start: str,
    stop: str,
    bbox: tuple,
    lookback: bool,
    rtc_source: str,
    tag_url_template: str | None,
    venue: str,
    data_dir: Path,
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
    if cached_venue is None:
        # Adopting asserts the venue, as it asserts the window and AOI.
        (data_dir / VENUE_FILE).write_text(f'{venue}\n')
        click.echo(f'Recorded venue {venue} in {data_dir / VENUE_FILE} (asserted, not verified)')
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
    _adopt_tag_sources(data_dir, tag_url_template)
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
@product_versions_option
@offline_option
@data_dir_option
def fetch_tags(
    workers: int,
    sample: int,
    seed: int,
    tag_url_template: str | None,
    product_versions: str,
    offline: bool,
    data_dir: Path,
) -> None:
    """Fetch product GeoTIFF tags (inputs, prior product) from the DAAC; resumable."""
    df_dist = checks.deduplicate_products(_filter_versions(_load_dist(data_dir), product_versions, None))
    if sample:
        df_dist = df_dist.sample(n=min(sample, len(df_dist)), random_state=seed)
    df_tags = tags.fetch_tags_table(
        df_dist, data_dir / TAGS_PARQUET, max_workers=workers, url_template=tag_url_template
    )
    n_err = int((df_tags.error != '').sum())
    n_no_url = int((df_tags.error == tags.MISSING_TAG_URL).sum())
    click.echo(
        f'Tags: {len(df_tags)} products, {n_err} errors ({n_no_url} with no tag URL) -> {data_dir / TAGS_PARQUET}'
    )


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
    # Units (F12): every acquisition is one group; a group with more than one version is a duplicate group.
    n_acquisitions = int(df_dist.opera_dedup_id.nunique()) if len(df_dist) else 0
    n_groups = int(df_dup.latest_opera_id.nunique()) if len(df_dup) else 0
    section = report.build_section(
        'duplicates',
        'Duplicates',
        'DIST-S1 granules sharing the same tile and acquisition time. The last-processed granule of '
        'each group is taken as the correct one; every other check runs on that granule only. Counted in '
        'duplicate groups (acquisitions with more than one version); the table lists every version.',
        df_dup,
        n_acquisitions,
        state=state,
        unit='duplicate groups',
        counts={Status.PASS: n_acquisitions - n_groups, Status.FAIL: n_groups},
        extra={
            'n_products': len(df_dist),
            'n_groups': n_groups,
            'n_versions': len(df_dup),
            'n_superseded': int((~df_dup.is_latest.astype(bool)).sum()) if len(df_dup) else 0,
        },
    )
    return [section], {'duplicates': df_dup}


def _ordering_section(
    df_dedup: pd.DataFrame,
    wkt_map: dict[str, str],
    state: SectionState = SectionState.COMPLETE,
    df_versions: pd.DataFrame | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Two views (decision 2, F08): first-production order over every version, and retained-version order."""
    df_versions = df_dedup if df_versions is None else df_versions
    df_first = checks.check_first_production_order(df_versions, df_dedup.opera_dedup_id)
    df_first_bad = _append_tile_wkt(df_first[df_first.out_of_order], wkt_map)
    n_first = int(df_first.evaluated.sum())
    first = report.build_section(
        'ordering_first_production',
        'Processing order: first production',
        'Acquisitions whose first version was processed before the first version of the previous acquisition in '
        'their MGRS tile. Every version counts, so a later reprocessing never hides an inversion; the first '
        'acquisition of each tile has no predecessor and is not evaluated.',
        df_first_bad,
        n_first,
        state=state,
        n_universe=len(df_first),
        n_skipped=len(df_first) - n_first,
        counts=df_first.status.value_counts().to_dict(),
        note=f'{int(df_first.tie.sum())} ties (equal processing times) counted as in order',
    )
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
        counts=df_order.status.value_counts().to_dict(),
        note=f'{int(df_order.tie.sum())} ties (equal processing times) counted as in order; '
        f'{int((~df_order.evaluated).sum())} acquisitions not evaluated (per tile in ordering_tiles)',
    )
    csvs = {
        'ordering_failures': df_bad,
        'ordering': df_order,
        'ordering_tiles': checks.ordering_by_tile(df_order),
        'ordering_first_production_failures': df_first_bad,
        'ordering_first_production': df_first,
    }
    return [first, section], csvs


def _confirmation_section(
    df_dedup: pd.DataFrame,
    df_tags: pd.DataFrame,
    expect_none_at_start: bool,
    wkt_map: dict[str, str],
    state: SectionState = SectionState.COMPLETE,
    df_versions: pd.DataFrame | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_conf = checks.check_confirmation(
        df_dedup, df_tags, expect_none_at_start=expect_none_at_start, df_versions=df_versions
    )
    # The predecessor comes from the full deduplicated table; the universe is the products whose tags were read
    # for this run (all of them, or the --sample-inputs sample), each with exactly one status (F03).
    df_conf = df_conf[df_conf.opera_id.isin(df_tags.opera_id)].reset_index(drop=True)
    _require_one_row_per_product(df_conf, df_tags.opera_id, 'confirmation', unique=True)
    df_bad = _append_tile_wkt(df_conf[~df_conf.confirmed_ok], wkt_map)
    counts = df_conf.status.value_counts().to_dict()
    n_skipped = counts.get(Status.SKIPPED, 0)
    section = report.build_section(
        'confirmation',
        'Confirmation chain',
        'Products whose prior_dist_s1_product tag is not the latest version of the previous acquisition in the tile '
        'that existed when the product was processed (full ids, all versions; processing time is the proxy for '
        'availability). A prior reprocessed only after the product is INFO (stale chain). The first product of each '
        'tile in the window has no predecessor to check and is skipped unless --expect-none-at-start is set.',
        df_bad,
        int(df_conf.status.isin(EVALUATED).sum()),
        state=state,
        unit='links',
        n_universe=len(df_dedup),
        n_skipped=n_skipped,
        counts=counts,
    )
    tiles = checks.summarize_by_tile(df_conf.assign(fail=~df_conf.confirmed_ok), 'fail')
    return [section], {'confirmation_failures': df_bad, 'confirmation_tiles': tiles, 'confirmation': df_conf}


def _inputs_section(
    df_tags: pd.DataFrame,
    df_frame: gpd.GeoDataFrame | None,
    mode: str,
    workers: int,
    wkt_map: dict[str, str],
    checkpoint_path: Path | None = None,
    state: SectionState = SectionState.COMPLETE,
    n_universe: int | None = None,
    checkpoint_context: str = '',
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    if mode == 'offline':
        df_results = inputs.check_inputs_offline(
            df_tags, df_frame, checkpoint_path=checkpoint_path, workers=workers, context=checkpoint_context
        )
    else:
        df_results = inputs.check_inputs_online(df_tags, max_workers=workers)
    if df_results.empty:
        df_results = pd.DataFrame(columns=inputs.RESULT_COLUMNS)
    _require_one_row_per_product(df_results, df_tags.opera_id, 'inputs')
    df_results = df_results.assign(status=_row_status(df_results))
    df_bad = _append_tile_wkt(df_results[df_results.inputs_correct != True], wkt_map)  # noqa: E712
    product_status = _inputs_product_status(df_results)
    counts = product_status.value_counts().to_dict()
    n_checked = int(product_status.isin(EVALUATED).sum())
    n_universe = len(product_status) if n_universe is None else n_universe

    def _counts(df_subset: pd.DataFrame) -> dict[str, int]:
        """Count products with an issue in the subset by status; the other checked products PASS for it."""
        status = _inputs_product_status(df_subset)
        n_fail, n_unknown = int((status == Status.FAIL).sum()), int((status == Status.UNKNOWN).sum())
        return {Status.FAIL: n_fail, Status.UNKNOWN: n_unknown, Status.PASS: max(n_checked - n_fail, 0)}

    common = {'state': state, 'n_universe': n_universe}

    post_mask = df_bad.issue_type.str.contains('Post', case=False)
    pre_mask = df_bad.issue_type.str.contains('Pre', case=False) & ~post_mask
    other_mask = ~post_mask & ~pre_mask  # the three subsections partition every issue (F12)
    sections = [
        report.build_section(
            'inputs',
            'Product inputs',
            'Products whose recorded pre/post RTC-S1 inputs differ from the re-enumerated expected inputs. The '
            'expected product is chosen from the product id (tile, acquisition), and tags that disagree are an '
            'identity mismatch.',
            df_bad,
            n_checked,
            summary=_issue_summary(df_bad),
            counts=counts,
            note=f'{int((df_results.status == Status.INFO).sum())} INFO notes (RTC processing version differs), '
            'which never fail a product (inputs CSV)',
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
        report.build_section(
            'inputs_other',
            'Inputs: other issues',
            'Product-input issues that are neither pre nor post specific: burst or track mismatches, identity '
            'mismatches, claims, tag read and enumeration errors.',
            df_bad[other_mask],
            n_checked,
            summary=_issue_summary(df_bad[other_mask]),
            counts=_counts(df_bad[other_mask]),
            **common,
        ),
    ]
    csvs = {
        'inputs_failures': df_bad,
        'inputs_failures_post': df_bad[post_mask],
        'inputs_failures_pre': df_bad[pre_mask],
        'inputs_failures_other': df_bad[other_mask],
        'inputs': df_results,
    }
    return sections, csvs


def _inputs_checkpoint_context(m_rtc: manifest.TableManifest, frame_tiles: list[str] | None) -> str:
    """Checkpoint context of the offline inputs check: RTC-S1 table, versions, mode and frame tiles (F04)."""
    return inputs.checkpoint_context(
        m_rtc.parquet_sha256, {'mode': 'offline', 'frame_tiles_sha256': manifest.target_tiles_sha(frame_tiles)}
    )


def _require_one_row_per_product(df: pd.DataFrame, universe: pd.Series, check: str, unique: bool = False) -> None:
    """Raise if a check's results do not cover its product universe exactly (F03): nothing may drop out."""
    ids, expected = set(df.opera_id), set(universe)
    duplicated = unique and df.opera_id.duplicated().any()
    if ids != expected or duplicated:
        raise RuntimeError(
            f'{check} results do not match its universe of {len(expected)} products: {len(expected - ids)} missing, '
            f'{len(ids - expected)} unexpected' + (', duplicated rows' if duplicated else '')
        )


def _row_status(df_results: pd.DataFrame) -> pd.Series:
    """Status of each inputs result row: its `status` (PASS, FAIL, UNKNOWN, INFO), else from `inputs_correct`."""
    derived = df_results.inputs_correct.map({True: Status.PASS, False: Status.FAIL}).fillna(Status.UNKNOWN)
    if 'status' in df_results.columns:
        derived = df_results.status.where(df_results.status.notna() & (df_results.status != ''), derived)
    return derived.astype(str)


def _inputs_product_status(df_results: pd.DataFrame) -> pd.Series:
    """One status per product: UNKNOWN if any row could not be checked, else FAIL if any row failed, else PASS."""
    if df_results.empty:
        return pd.Series(dtype=str)
    # INFO notes (e.g. RTC processing version differs) never decide a product's status (decision 4).
    value = _row_status(df_results).replace({Status.INFO: Status.PASS})
    rank = value.map({Status.UNKNOWN: 0, Status.FAIL: 1, Status.PASS: 2})
    return value.groupby(df_results.opera_id).agg(lambda v: v.loc[rank[v.index].idxmin()])


def _eligible(df_cov: pd.DataFrame) -> pd.Series:
    """Return which passes have baseline imagery (`eligible`; older frames: found or missing-and-expected)."""
    if 'eligible' in df_cov.columns:
        return df_cov.eligible.astype(bool)
    return df_cov.product_found.astype(bool) | df_cov.missing_product_expected.astype(bool)


def _coverage_section(
    df_dedup: pd.DataFrame,
    df_frame: gpd.GeoDataFrame,
    start: str,
    stop: str,
    wkt_map: dict[str, str],
    state: SectionState = SectionState.COMPLETE,
    query_windows: list[manifest.QueryWindow] | None = None,
    df_tags: pd.DataFrame | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:

    df_groups = coverage.expected_pass_groups(df_frame, start, stop)
    # One assignment feeds both checks, so every product is either matched to a pass or an orphan (F06).
    assignment = matching.assign_passes(df_groups, df_dedup)
    df_cov = coverage.check_coverage(
        df_groups, df_dedup, df_rtc_frame=df_frame, query_windows=query_windows, df_tags=df_tags, assignment=assignment
    )
    # FAIL (missing, baseline available) and UNKNOWN (lookback incomplete, latest product unreadable, ambiguous
    # match, group mismatch) passes all need attention.
    df_missing = df_cov[df_cov.status.isin([Status.FAIL, Status.UNKNOWN])]
    df_missing = _append_tile_wkt(df_missing.drop(columns=['burst_ids', 'burst_pols'], errors='ignore'), wkt_map)
    df_missing['asf_search_url'] = [
        _vertex_search_url(row.mgrs_tile_wkt, row.post_acq_dt) for row in df_missing.itertuples()
    ]
    df_orphans = coverage.check_orphan_products(df_groups, df_dedup, assignment=assignment)

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
        'The check expects every eligible pass in the acquisition window and AOI: an RTC-S1 pass with baseline '
        'imagery must have triggered a DIST-S1 product. Listed: eligible passes with no product (FAIL), missing '
        'passes whose baseline windows were not all queried to completion (UNKNOWN), passes whose latest product '
        'could not be read (UNKNOWN: latest product unreadable), passes that match two products equally well '
        '(UNKNOWN: ambiguous match), and passes whose product was made from another acquisition group '
        '(UNKNOWN: group mismatch). Passes and products are matched one-to-one per tile.',
        df_missing,
        int(df_cov.status.isin(EVALUATED).sum()),
        summary=summary,
        state=state,
        unit='passes',
        n_universe=len(df_cov),
        n_skipped=int((df_cov.status == Status.SKIPPED).sum()),
        counts=df_cov.status.value_counts().to_dict(),
        extra={
            'n_eligible': int(_eligible(df_cov).sum()),
            'n_found': int(df_cov.product_found.astype(bool).sum()),
            'n_missing': int((df_cov.status == Status.FAIL).sum()),
        },
    )
    if df_orphans.columns.empty:
        df_orphans = pd.DataFrame(columns=coverage.ORPHAN_COLUMNS)  # keep the header in an empty CSV (F20)
    df_orphans = _append_tile_wkt(df_orphans, wkt_map)
    orphans = report.build_section(
        'orphans',
        'Orphan products',
        'DIST-S1 products that the one-to-one assignment could not match to any expected RTC-S1 pass in the '
        'window (UNKNOWN: the product cannot be accounted for by the RTC-S1 record), including a second product '
        'within tolerance of a pass already matched to another product.',
        df_orphans,
        len(df_dedup) - len(df_orphans),
        state=state,
        n_universe=len(df_dedup),
        counts={Status.PASS: len(df_dedup) - len(df_orphans), Status.UNKNOWN: len(df_orphans)},
    )
    csvs = {
        'coverage_missing': df_missing,
        'coverage_missing_by_tile': df_by_tile,
        'coverage': df_cov,
        'coverage_orphans': df_orphans,
    }
    return [section, orphans], csvs


@cli.command()
@offline_option
@product_versions_option
@regression_option
@data_dir_option
@out_dir_option
def check_duplicates(
    data_dir: Path, out_dir: Path, regression_baseline: bool, product_versions: str, offline: bool
) -> None:
    """Check for duplicated DIST-S1 products (check 2)."""
    _run_single_check(
        out_dir,
        'duplicates',
        'DIST-S1 duplicates',
        lambda run: _duplicates_section(
            _load_dist_checked(data_dir, run, product_versions), _wkt_by_tile(_all_tile_geoms())
        ),
        regression_baseline,
    )


@cli.command()
@offline_option
@product_versions_option
@regression_option
@data_dir_option
@out_dir_option
def check_ordering(
    data_dir: Path, out_dir: Path, regression_baseline: bool, product_versions: str, offline: bool
) -> None:
    """Check processing order matches acquisition order (check 3)."""
    _run_single_check(
        out_dir,
        'ordering',
        'DIST-S1 processing order',
        lambda run: _ordering_section(
            checks.deduplicate_products(df_all := _load_dist_checked(data_dir, run, product_versions)),
            _wkt_by_tile(_all_tile_geoms()),
            df_versions=df_all,
        ),
        regression_baseline,
    )


@cli.command()
@click.option(
    '--expect-none-at-start', is_flag=True, help='Require the first product per tile to have no prior (mission start).'
)
@offline_option
@product_versions_option
@regression_option
@data_dir_option
@out_dir_option
def check_confirmation(
    expect_none_at_start: bool,
    data_dir: Path,
    out_dir: Path,
    regression_baseline: bool,
    product_versions: str,
    offline: bool,
) -> None:
    """Check the confirmation chain via prior product tags (check 4)."""
    _run_single_check(
        out_dir,
        'confirmation',
        'DIST-S1 confirmation chain',
        lambda run: _confirmation_section(
            df_dedup := checks.deduplicate_products(df_all := _load_dist_checked(data_dir, run, product_versions)),
            tags.align_to_products(df_dedup, _load_tags(data_dir)),
            expect_none_at_start,
            _wkt_by_tile(_all_tile_geoms()),
            df_versions=df_all,
        ),
        regression_baseline,
    )


@cli.command()
@click.option('--mode', type=click.Choice(['offline', 'online']), default='offline', show_default=True)
@click.option('--sample', type=int, default=0, show_default=True, help='Randomly sample N products (0 = all).')
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--workers', type=int, default=8, show_default=True)
@offline_option
@product_versions_option
@regression_option
@data_dir_option
@out_dir_option
def check_inputs(
    mode: str,
    sample: int,
    seed: int,
    workers: int,
    data_dir: Path,
    out_dir: Path,
    regression_baseline: bool,
    product_versions: str,
    offline: bool,
) -> None:
    """Check product inputs against re-enumerated expected inputs (check 5)."""
    if mode == 'online':
        network.network_allowed('online inputs mode')

    def build(run: _Run) -> tuple[list[dict], dict[str, pd.DataFrame]]:
        df_dist = _load_dist_checked(data_dir, run, product_versions)
        df_tags = tags.align_to_products(checks.deduplicate_products(df_dist), _load_tags(data_dir))
        n_universe = len(df_tags)
        state = SectionState.COMPLETE
        if sample:
            df_tags = df_tags.sample(n=min(sample, len(df_tags)), random_state=seed)
            run.manifest.sampling['inputs'] = _sampling_record(n_universe, df_tags.opera_id, seed, 'products')
            state = SectionState.SAMPLED
        df_frame, context = None, ''
        if mode == 'offline':
            df_rtc, m_rtc = _load_rtc_checked(data_dir, run)
            tiles = _load_target_tiles(data_dir)
            df_frame = _build_rtc_frame(df_rtc, run, tiles)
            context = _inputs_checkpoint_context(m_rtc, tiles)
        return _inputs_section(
            df_tags,
            df_frame,
            mode,
            workers,
            _wkt_by_tile(_all_tile_geoms()),
            checkpoint_path=data_dir / INPUTS_CHECKPOINT_PARQUET,
            state=state,
            n_universe=n_universe,
            checkpoint_context=context,
        )

    _run_single_check(out_dir, 'inputs', 'DIST-S1 product inputs', build, regression_baseline)


def _check_coverage_standalone(
    run: _Run, data_dir: Path, start: str, stop: str, product_versions: str
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df_dedup = checks.deduplicate_products(_load_dist_checked(data_dir, run, product_versions))
    df_rtc, m_rtc = _load_rtc_checked(data_dir, run, start, stop)
    # Readability of the latest versions comes from the cached tags, if any; a product with no cached row is
    # not known to be unreadable (missing='').
    df_tags = (
        tags.align_to_products(df_dedup, _load_tags(data_dir), missing='')
        if (data_dir / TAGS_PARQUET).exists()
        else None
    )
    return _coverage_section(
        df_dedup,
        _build_rtc_frame(df_rtc, run, _load_target_tiles(data_dir)),
        start,
        stop,
        _wkt_by_tile(_all_tile_geoms()),
        query_windows=m_rtc.lookback_windows,
        df_tags=df_tags,
    )


@cli.command()
@click.option('--start', required=True)
@click.option('--stop', required=True)
@offline_option
@product_versions_option
@regression_option
@data_dir_option
@out_dir_option
def check_coverage(
    start: str,
    stop: str,
    data_dir: Path,
    out_dir: Path,
    regression_baseline: bool,
    product_versions: str,
    offline: bool,
) -> None:
    """Check all triggerable DIST-S1 products exist given available RTC-S1 inputs (check 6)."""
    _run_single_check(
        out_dir,
        'coverage',
        'DIST-S1 coverage',
        lambda run: _check_coverage_standalone(run, data_dir, start, stop, product_versions),
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
@offline_option
@product_versions_option
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
    product_versions: str,
    offline: bool,
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
            missing = [n for n, needed in ((DIST_S1_PARQUET, need_dist), (RTC_S1_PARQUET, need_rtc)) if needed]
            network.network_allowed(
                f'metadata download ({"--refresh" if refresh else "missing " + ", ".join(missing)})'
            )
            (data_dir / INPUTS_CHECKPOINT_PARQUET).unlink(missing_ok=True)
            for name, needed in ((DIST_S1_PARQUET, need_dist), (RTC_S1_PARQUET, need_rtc)):
                if needed:
                    manifest.manifest_path(data_dir / name).unlink(missing_ok=True)
            dist_bbox, rtc_bbox = _resolve_bboxes(bbox, data_dir)
            chunks_dir = _rtc_chunks_dir(data_dir, rtc_bbox)
            if refresh:
                shutil.rmtree(chunks_dir, ignore_errors=True)
            if need_dist:
                dist_chunks: list[dict] = []
                df_dist = cmr.get_dist_s1_table(
                    start, stop, bbox=dist_bbox, max_workers=cmr_workers, venue=venue, chunk_log=dist_chunks
                )
                _write_dist(df_dist, data_dir, venue, start, stop, bbox, dist_bbox, dist_chunks)
            if need_rtc:
                rtc_chunks: list[dict] = []
                df_rtc = cmr.get_rtc_s1_table_with_lookback(
                    start, stop, bbox=rtc_bbox, max_workers=cmr_workers, cache_dir=chunks_dir, chunk_log=rtc_chunks
                )
                _write_rtc(df_rtc, data_dir, chunks_dir, start, stop, bbox, rtc_bbox, chunks=rtc_chunks)
                del df_rtc
        request = manifest.CacheRequest(venue, start, stop, bbox, check_bbox=True)
        df_dist, m_dist = _load_cached(data_dir, DIST_S1_PARQUET, run, request)
        df_dist = _filter_versions(df_dist, product_versions, run)
        # Always apply the exact [start, stop) acquisition filter (F31): CMR matches temporal extents inclusively
        # and a cache may span a wider window. Rows outside it stay in the cache as context and are counted.
        acq = pd.to_datetime(df_dist.acq_time, utc=True)
        in_window = (acq >= manifest._utc(start)) & (acq < manifest._utc(stop))
        click.echo(f'Window filter kept {int(in_window.sum())} of {len(df_dist)} cached DIST-S1 products')
        run.manifest.window_filter = {
            'start': start,
            'stop': stop,
            'n_before': len(df_dist),
            'n_kept': int(in_window.sum()),
        }
        df_dist = df_dist[in_window].reset_index(drop=True)
        # Version context comes from every version in the window, before any processing filter (F18).
        df_dist = checks.attach_version_context(df_dist)
        df_versions = df_dist
        if processing_start or processing_stop:
            n_before = len(df_dist)
            df_dist = _filter_processing_time(df_dist, processing_start, processing_stop)
            run.manifest.processing_filter = {
                'start': processing_start,
                'stop': processing_stop,
                'n_before': n_before,
                'n_kept': len(df_dist),
            }
        tile_note = ''
        if sample_tiles:
            all_tiles = pd.Series(sorted(df_dist.mgrs_tile_id.unique()))
            keep = set(all_tiles.sample(n=min(sample_tiles, len(all_tiles)), random_state=seed))
            df_dist = df_dist[df_dist.mgrs_tile_id.isin(keep)].reset_index(drop=True)
            run.manifest.sampling['tiles'] = _sampling_record(len(all_tiles), sorted(keep), seed, 'tiles')
            tile_note = f' - sample of {len(keep)} of {len(all_tiles)} tiles (seed {seed})'
            click.echo(f'Sampled {len(keep)} of {len(all_tiles)} tiles')
        click.echo(f'DIST-S1 products: {len(df_dist)}')
        df_rtc, m_rtc = _load_cached(
            data_dir, RTC_S1_PARQUET, run, manifest.CacheRequest('PROD', start, stop, bbox, check_bbox=True)
        )
        click.echo(f'RTC-S1 granules: {len(df_rtc)}')
        frame_tiles = sorted(keep) if sample_tiles else _load_target_tiles(data_dir)
        df_frame = _build_rtc_frame(df_rtc, run, frame_tiles)

        df_dedup = checks.deduplicate_products(df_dist)
        # Sampling selects the products whose tags are fetched (the expensive step), so it applies to both the
        # confirmation and the inputs sections; the expected predecessor still comes from the full DIST table.
        df_tags_products = df_dedup
        product_note = ''
        if sample_inputs:
            df_tags_products = df_dedup.sample(n=min(sample_inputs, len(df_dedup)), random_state=seed)
            record = _sampling_record(len(df_dedup), df_tags_products.opera_id, seed, 'products')
            run.manifest.sampling |= {'confirmation': record, 'inputs': record}
            product_note = (
                f' - confirmation and inputs sampled: {len(df_tags_products)} of {len(df_dedup)} products (seed {seed})'
            )
        all_state = SectionState.SAMPLED if sample_tiles else SectionState.COMPLETE
        tag_state = SectionState.SAMPLED if sample_tiles or sample_inputs else SectionState.COMPLETE
        run.subtitle = (
            f'{start} to {stop}'
            + (f' over {bbox}' if bbox else ' (global)')
            + tile_note
            + product_note
            + _version_note(run)
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
        _record_file(run, data_dir / TAGS_PARQUET)

        wkt_map = _wkt_by_tile(_all_tile_geoms())
        builds = {
            'duplicates': lambda: _duplicates_section(df_dist, wkt_map, all_state),
            'ordering': lambda: _ordering_section(df_dedup, wkt_map, all_state, df_versions=df_versions),
            'confirmation': lambda: _confirmation_section(
                df_dedup, df_tags, expect_none_at_start, wkt_map, tag_state, df_versions=df_versions
            ),
            'inputs': lambda: _inputs_section(
                df_tags,
                df_frame,
                'offline',
                inputs_workers,
                wkt_map,
                checkpoint_path=data_dir / INPUTS_CHECKPOINT_PARQUET,
                state=tag_state,
                n_universe=len(df_dedup),
                checkpoint_context=_inputs_checkpoint_context(m_rtc, frame_tiles),
            ),
            'coverage': lambda: _coverage_section(
                df_dedup,
                df_frame,
                start,
                stop,
                wkt_map,
                all_state,
                query_windows=m_rtc.lookback_windows,
                df_tags=df_tags,
            ),
        }
        for key in RUN_ALL_CHECKS:
            run.stage(key)
            run.add(key, *builds[key]())
        _record_file(run, data_dir / INPUTS_CHECKPOINT_PARQUET)
        run.complete()
