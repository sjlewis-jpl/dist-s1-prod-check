"""End-to-end regression run of ``run-all`` on the 20-tile real-data subset (hardening plan, Task 0.4).

See ``tests/data/README.md`` for the subset's provenance and for the tag-retry workaround used here.
"""

import json
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

from dist_s1_prod_check import report
from dist_s1_prod_check.cli import cli


TESTS_DATA = Path(__file__).parent / 'data'
DATA_ROOT = Path('~/data/opera/dist-s1-prod-check').expanduser()
SUBSET_ROOT = DATA_ROOT / 'hardening_review' / 'subset'
COUNTS_PATH = SUBSET_ROOT / 'baseline_counts.json'
COUNTS_MIRROR = TESTS_DATA / 'baseline_counts.json'
BASELINES_JSON = Path(__file__).resolve().parents[2] / 'docs' / 'prod-gate-hardening' / 'baselines.json'

# The original baseline arguments (baselines.json), minus --data-dir/--out-dir/--workers/--inputs-workers.
# --inputs-workers 1 keeps the inputs check in this process, so the network blocks below cover it too.
VENUE_ARGS = {
    'uat': [
        '--venue', 'UAT',
        '--start', '2026-01-01',
        '--stop', '2026-03-01',
        '--processing-start', '2026-09-24',
        '--tag-url-template',
        'http://opera-int-rs-fwd.s3-website-us-west-2.amazonaws.com/products/DIST_S1/{acq_time:%Y/%m/%d}/{opera_id}/'
        '{opera_id}_GEN-DIST-STATUS.tif',
    ],
    'pop1': ['--venue', 'OPS-POP1', '--start', '2026-01-01', '--stop', '2026-03-01'],
}  # fmt: skip
DROPPED_ARGS = {'--data-dir', '--out-dir', '--workers', '--inputs-workers'}
LOOPBACK = {None, '', 'localhost', '127.0.0.1', '::1'}


def _subset_dir(venue: str) -> Path | None:
    for root in (TESTS_DATA, SUBSET_ROOT):
        if (root / venue / 'dist_s1_products.parquet').exists():
            return root / venue
    return None


def _strip(args: list[str]) -> list[str]:
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg in DROPPED_ARGS:
            skip = True
        elif arg != 'run-all':
            out.append(arg)
    return out


def _install_network_blocks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make requests, boto3, rasterio and remote sockets raise; return the list that records every attempt."""
    import boto3
    import rasterio
    import requests

    attempts: list[str] = []

    def blocker(name: str) -> Callable[..., Any]:
        def _raise(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            attempts.append(f'{name} {args[:2]!r}')
            raise ConnectionError(f'network access blocked: {name}')

        return _raise

    def remote_only(name: str, real: Callable[..., Any], host_of: Callable[..., Any]) -> Callable[..., Any]:
        def _guard(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            host = host_of(*args)
            host = host[0] if isinstance(host, tuple) and host else host
            if host not in LOOPBACK:
                return blocker(name)(*args)
            return real(*args, **kwargs)

        return _guard

    monkeypatch.setattr(requests.Session, 'request', blocker('requests.Session.request'))
    monkeypatch.setattr(boto3, 'client', blocker('boto3.client'))
    monkeypatch.setattr(boto3, 'resource', blocker('boto3.resource'))
    monkeypatch.setattr(rasterio, 'open', blocker('rasterio.open'))
    monkeypatch.setattr(
        socket.socket, 'connect', remote_only('socket.connect', socket.socket.connect, lambda self, addr: addr)
    )
    monkeypatch.setattr(
        socket, 'getaddrinfo', remote_only('socket.getaddrinfo', socket.getaddrinfo, lambda host, *a: host)
    )
    return attempts


def _csv_rows(out_dir: Path) -> dict[str, int]:
    counts = {}
    for path in sorted(out_dir.glob('*.csv')):
        name = path.stem.rsplit('_', 1)[0]
        try:
            counts[name] = len(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            counts[name] = 0
    return counts


def test_venue_args_match_baselines() -> None:
    """The subset runs use the same check arguments as the recorded baselines."""
    if not BASELINES_JSON.exists():
        pytest.skip(f'{BASELINES_JSON} not found')
    baselines = json.loads(BASELINES_JSON.read_text())['baselines']
    for venue, args in VENUE_ARGS.items():
        assert _strip(baselines[venue]['original_args']) == args


@pytest.mark.data
@pytest.mark.parametrize('venue', ['uat', 'pop1'])
def test_subset_baseline(venue: str, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``run-all`` offline on a venue subset and compare section and CSV counts with the recorded reference.

    The reference is ``hardening_review/subset/baseline_counts.json``; it is written when absent (and mirrored to
    ``tests/data/baseline_counts.json``), otherwise the counts must be equal.
    """
    subset = _subset_dir(venue)
    if subset is None:
        pytest.skip(f'no {venue} subset in {TESTS_DATA} or {SUBSET_ROOT}; run scripts/make_test_subset.py')
    data_dir, out_dir = tmp_path / 'data', tmp_path / 'out'
    shutil.copytree(subset, data_dir)
    # The subset was cut from caches built before manifests existed: adopt the copy with the baseline's arguments.
    adopt_args = [a for a in VENUE_ARGS[venue] if a not in ('--processing-start', '2026-09-24')]
    adopted = cli_runner.invoke(cli, ['adopt-cache', *adopt_args, '--data-dir', str(data_dir)])
    assert adopted.exit_code == 0, adopted.output

    attempts = _install_network_blocks(monkeypatch)
    sections: list[dict] = []
    write_html = report.write_html_report

    def _capture(sections_arg: list[dict], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        sections.extend(sections_arg)
        return write_html(sections_arg, *args, **kwargs)

    monkeypatch.setattr(report, 'write_html_report', _capture)

    args = [
        'run-all',
        *VENUE_ARGS[venue],
        '--offline',
        '--regression-baseline',
        '--workers',
        '1',
        '--inputs-workers',
        '1',
    ]
    result = cli_runner.invoke(cli, [*args, '--data-dir', str(data_dir), '--out-dir', str(out_dir)])
    assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
    assert attempts == []

    counts = {
        'args': args,
        'sections': {s['key']: {'n_checked': s['n_checked'], 'n_failures': s['n_failures']} for s in sections},
        'csv_rows': _csv_rows(out_dir),
    }
    assert set(counts['sections']) == {'duplicates', 'ordering_first_production', 'ordering', 'confirmation',
                                       'inputs', 'inputs_post', 'inputs_pre', 'inputs_other', 'coverage'}  # fmt: skip

    reference_path = COUNTS_PATH if COUNTS_PATH.exists() else COUNTS_MIRROR
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    if venue in reference:
        assert counts == reference[venue]
        return
    if not SUBSET_ROOT.exists():
        pytest.fail(f'no reference counts for {venue} and {SUBSET_ROOT} does not exist to record them')
    reference[venue] = counts
    text = json.dumps(reference, indent=2, sort_keys=True) + '\n'
    COUNTS_PATH.write_text(text)
    if subset.parent == TESTS_DATA:
        COUNTS_MIRROR.write_text(text)
