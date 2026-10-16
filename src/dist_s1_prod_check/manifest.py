"""Provenance for cached metadata tables (`TableManifest`) and for each run (`RunManifest`).

A table manifest is written as ``<table>.manifest.json`` next to its parquet. It records what was queried (venue,
acquisition window, AOI, every query window with its completeness) and how the record came to be:

- ``query_receipt``: this tool ran the query to completion and wrote the parquet itself.
- ``adopted_unverified``: a user asserted the contents with ``adopt-cache`` for a cache built before manifests
  existed. Such a cache only backs runs marked ``--regression-baseline``.

Every load recomputes the parquet's sha256 and rejects the cache if it differs from the manifest.
"""

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
import pandas as pd
import pyarrow.parquet as pq


QUERY_RECEIPT = 'query_receipt'
ADOPTED_UNVERIFIED = 'adopted_unverified'
RUN_MANIFEST_FILE = 'run_manifest.json'
LAYER_COMPLETENESS_DISCLAIMER = 'Layer completeness (GEN-METRIC, GEN-DIST-STATUS-ACQ) is not checked.'


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _iso(t: str | pd.Timestamp | None) -> str | None:
    return None if t is None else _utc(t).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_utc() -> str:
    """Return the current UTC time as an ISO string (``YYYY-MM-DDTHH:MM:SSZ``)."""
    return _iso(pd.Timestamp.now(tz='UTC'))


@dataclass(frozen=True)
class QueryWindow:
    """One `[start, stop)` acquisition range that was queried; `complete` only if the query ran to the end."""

    start: pd.Timestamp
    stop: pd.Timestamp
    complete: bool
    n_rows: int

    def to_dict(self) -> dict:
        return {'start': _iso(self.start), 'stop': _iso(self.stop), 'complete': self.complete, 'n_rows': self.n_rows}

    @classmethod
    def from_dict(cls, d: dict) -> 'QueryWindow':
        return cls(_utc(d['start']), _utc(d['stop']), bool(d['complete']), int(d['n_rows']))


def _covered(windows: list[QueryWindow], start: pd.Timestamp, stop: pd.Timestamp) -> bool:
    """Whether `[start, stop)` (or the instant `start` when equal) lies inside one complete window."""
    return any(
        w.complete and w.start <= start and (stop < w.stop if stop == start else stop <= w.stop) for w in windows
    )


def lookback_complete(
    windows: list[QueryWindow] | None,
    post_acq_dt: pd.Timestamp,
    delta_lookback_days: tuple[int, ...],
    delta_window_days: int,
) -> bool:
    """Return whether every baseline window a pass needs, and the pass itself, was queried to completion.

    The baseline is the separate windows `[post - delta - window, post - delta)` for each `delta`, not continuous
    history, so the gaps between them are never required. A complete window that returned 0 rows counts as
    complete: "no baseline" is then a real finding.
    """
    if not windows:
        return False
    post = _utc(post_acq_dt)
    required = [
        (post - pd.Timedelta(days=d + delta_window_days), post - pd.Timedelta(days=d)) for d in delta_lookback_days
    ]
    return all(_covered(windows, a, b) for a, b in required) and _covered(windows, post, post)


def sha256_file(path: str | Path) -> str:
    """Return the hex sha256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tool_version() -> str:
    """Git sha of the checkout this module runs from (``+dirty`` with local changes), else the package version."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ['git', '-C', str(here), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', '-C', str(here), 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ('+dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return importlib.metadata.version('dist-s1-prod-check')


def enumerator_version() -> str:
    """Return the installed dist-s1-enumerator version."""
    return importlib.metadata.version('dist-s1-enumerator')


def target_tiles_sha(tiles: list[str] | None) -> str | None:
    """Return the sha256 of a sorted MGRS tile list (None for no list), as recorded in manifests."""
    if tiles is None:
        return None
    return hashlib.sha256('\n'.join(sorted(tiles)).encode()).hexdigest()


@dataclass
class TableManifest:
    """Provenance of one cached metadata table (written as ``<table>.manifest.json`` next to its parquet)."""

    table: str
    venue: str
    source: str
    start: str
    stop: str
    bbox: list[float] | None
    query_bbox: list[float] | None
    target_tiles_sha: str | None
    lookback_windows: list[QueryWindow]
    query_time_utc: str | None
    file_mtime_utc: str | None
    n_rows: int
    parquet_sha256: str
    tool_version: str
    enumerator_version: str
    complete: bool
    provenance: str
    # One record per CMR query chunk (query window, hits, rows, query time, reused); empty for adopted caches.
    chunks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['lookback_windows'] = [w.to_dict() for w in self.lookback_windows]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TableManifest':
        d = dict(d)
        d['lookback_windows'] = [QueryWindow.from_dict(w) for w in d.get('lookback_windows', [])]
        return cls(**d)


def manifest_path(parquet_path: str | Path) -> Path:
    """Return the manifest path of a cached parquet: ``<table>.manifest.json`` beside it."""
    parquet_path = Path(parquet_path)
    return parquet_path.with_name(parquet_path.name.removesuffix('.parquet') + '.manifest.json')


def write_table_manifest(
    parquet_path: str | Path,
    *,
    table: str,
    venue: str,
    source: str,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    bbox: tuple | list | None = None,
    query_bbox: tuple | list | None = None,
    target_tiles: list[str] | None = None,
    lookback_windows: list[QueryWindow] | None = None,
    provenance: str = QUERY_RECEIPT,
    query_time_utc: str | None = None,
    complete: bool = True,
    chunks: list[dict] | None = None,
) -> TableManifest:
    """Write the manifest of a parquet that has just been written (receipt) or is being adopted.

    A receipt records `query_time_utc` (default now). An adopted cache never does, because the query time is
    unknown; it records the parquet's modification time as `file_mtime_utc` instead.
    """
    parquet_path = Path(parquet_path)
    adopted = provenance == ADOPTED_UNVERIFIED
    mtime = pd.Timestamp(parquet_path.stat().st_mtime, unit='s', tz='UTC')
    m = TableManifest(
        table=table,
        venue=venue,
        source=source,
        start=_iso(start),
        stop=_iso(stop),
        bbox=[float(v) for v in bbox] if bbox else None,
        query_bbox=[float(v) for v in query_bbox] if query_bbox else None,
        target_tiles_sha=target_tiles_sha(target_tiles),
        lookback_windows=list(lookback_windows or [QueryWindow(_utc(start), _utc(stop), complete, -1)]),
        query_time_utc=None if adopted else (query_time_utc or now_utc()),
        file_mtime_utc=_iso(mtime) if adopted else None,
        n_rows=pq.read_metadata(parquet_path).num_rows,
        parquet_sha256=sha256_file(parquet_path),
        tool_version=tool_version(),
        enumerator_version=enumerator_version(),
        complete=complete,
        provenance=provenance,
        chunks=list(chunks or []),
    )
    path = manifest_path(parquet_path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(m.to_dict(), indent=2) + '\n')
    tmp.replace(path)
    return m


def windows_with_counts(
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]], acq_times: pd.Series, complete: bool = True
) -> list[QueryWindow]:
    """Query windows for `ranges` with the number of rows whose acquisition time falls in each."""
    acq = pd.to_datetime(acq_times, utc=True)
    return [QueryWindow(_utc(a), _utc(b), complete, int(((acq >= _utc(a)) & (acq < _utc(b))).sum())) for a, b in ranges]


def load_table_manifest(parquet_path: str | Path) -> TableManifest | None:
    """Read a parquet's manifest (None if absent) and verify the parquet still has the recorded sha256."""
    path = manifest_path(parquet_path)
    if not path.exists():
        return None
    m = TableManifest.from_dict(json.loads(path.read_text()))
    sha = sha256_file(parquet_path)
    if sha != m.parquet_sha256:
        raise click.ClickException(
            f'{parquet_path} changed after its manifest was written (sha256 {sha[:12]}... != recorded '
            f'{m.parquet_sha256[:12]}...); refresh it with --refresh, or re-adopt it with adopt-cache.'
        )
    return m


@dataclass(frozen=True)
class CacheRequest:
    """What a run needs from a cached table. `start`/`stop`/`bbox` None means "any" (standalone check commands)."""

    venue: str
    start: str | None = None
    stop: str | None = None
    bbox: tuple | None = None
    check_bbox: bool = False


def _same_bbox(a: list | tuple | None, b: list | tuple | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b))


def require_compatible(
    m: TableManifest | None, request: CacheRequest, *, path: str | Path, regression_baseline: bool = False
) -> None:
    """Raise a `ClickException` listing every reason the cached table cannot serve `request`."""
    if m is None:
        raise click.ClickException(
            f'{path} has no manifest ({manifest_path(path).name}), so what it holds is unknown. Re-download it with '
            '`run-all --refresh` (or `download-metadata`); for a regression run on an existing cache, run '
            '`adopt-cache` with the arguments that built it and pass --regression-baseline.'
        )
    problems = []
    if m.provenance == ADOPTED_UNVERIFIED and not regression_baseline:
        problems.append(
            'it is an adopted cache (provenance unverified: nothing proves what was queried or that the query '
            'finished), allowed only with --regression-baseline and never for a production-gate report'
        )
    elif m.provenance not in (QUERY_RECEIPT, ADOPTED_UNVERIFIED):
        problems.append(f'unknown provenance {m.provenance!r}')
    if not m.complete:
        problems.append('its query did not complete')
    if m.venue != request.venue:
        problems.append(f'venue {m.venue} != requested {request.venue}')
    if request.start is not None and request.stop is not None:
        start, stop = _utc(request.start), _utc(request.stop)
        if not (_utc(m.start) <= start and stop <= _utc(m.stop)):
            problems.append(
                f'requested window --start {_iso(start)} --stop {_iso(stop)} is not inside the cached window '
                f'{m.start} to {m.stop}'
            )
    if request.check_bbox and not _same_bbox(m.bbox, request.bbox):
        problems.append(f'cached --bbox {m.bbox or "none (global)"} != requested --bbox {request.bbox or "none"}')
    if problems:
        raise click.ClickException(
            f'{path} cannot be used for this run: ' + '; '.join(problems) + '. Use --refresh or a separate --data-dir.'
        )


def strictly_inside(m: TableManifest, start: str, stop: str) -> bool:
    """Whether the requested window is a strict subset of the cached one (the table then needs filtering)."""
    return (_utc(m.start), _utc(m.stop)) != (_utc(start), _utc(stop))


@dataclass
class RunManifest:
    """Everything a report claims about its run; the report's provenance block is generated from this only."""

    command: str
    params: dict
    tool_version: str = field(default_factory=tool_version)
    enumerator_version: str = field(default_factory=enumerator_version)
    created_utc: str = field(default_factory=now_utc)
    tables: dict[str, dict] = field(default_factory=dict)
    file_sha256: dict[str, str] = field(default_factory=dict)
    sampling: dict[str, dict] = field(default_factory=dict)
    version_filter: dict = field(default_factory=dict)
    processing_filter: dict = field(default_factory=dict)
    window_filter: dict = field(default_factory=dict)
    rtc_frame_drops: dict = field(default_factory=dict)
    offline: bool = False
    regression_baseline: bool = False
    outputs: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=lambda: [LAYER_COMPLETENESS_DISCLAIMER])

    @property
    def adopted(self) -> bool:
        return any(t.get('provenance') == ADOPTED_UNVERIFIED for t in self.tables.values())

    def to_dict(self) -> dict:
        return {**asdict(self), 'adopted_cache': self.adopted}

    def write(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / RUN_MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str) + '\n')
        tmp.replace(path)
        return path
