"""Cache manifests, adopt-cache and the provenance block (hardening plan, Task 1.3)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner, Result

from dist_s1_prod_check import manifest
from dist_s1_prod_check.cli import cli
from dist_s1_prod_check.constants import DIST_S1_PARQUET, RTC_S1_PARQUET
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, make_tags, seed_data_dir


TILE = '10SGC'
ACQ = pd.Timestamp('2026-01-02T02:06:53')
RUN_ARGS = ['--start', '2026-01-01', '--stop', '2026-03-01']


def _seed(data_dir: Path) -> list[str]:
    ids = [dist_id(TILE, ACQ + pd.Timedelta(days=12 * k), ACQ + pd.Timedelta(days=12 * k + 1)) for k in range(2)]
    burst = bursts_for_tile(TILE).jpl_burst_id.iloc[0]
    rtc = make_rtc([{'burst': burst, 'acq': ACQ, 'proc': ACQ + pd.Timedelta(hours=5)}])
    prior = ['None', ids[0]]
    tags = make_tags([{'opera_id': i, 'prior_dist_s1_product': p} for i, p in zip(ids, prior)])
    seed_data_dir(data_dir, make_dist([{'opera_id': i} for i in ids]), rtc, tags, target_tiles=[TILE])
    return ids


def _run_all(data_dir: Path, out_dir: Path, *extra: str) -> Result:
    args = [
        'run-all',
        *RUN_ARGS,
        *extra,
        '--inputs-workers',
        '1',
        '--data-dir',
        str(data_dir),
        '--out-dir',
        str(out_dir),
    ]
    return CliRunner().invoke(cli, args)


def _adopt(data_dir: Path) -> Result:
    return CliRunner().invoke(cli, ['adopt-cache', *RUN_ARGS, '--data-dir', str(data_dir)])


def _strip_manifests(data_dir: Path) -> None:
    for name in (DIST_S1_PARQUET, RTC_S1_PARQUET):
        manifest.manifest_path(data_dir / name).unlink()


def test_receipt_records_query(tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    m = manifest.load_table_manifest(tmp_data_dir / RTC_S1_PARQUET)
    assert m.provenance == manifest.QUERY_RECEIPT
    assert m.query_time_utc and m.file_mtime_utc is None
    assert len(m.lookback_windows) == 4 and all(w.complete for w in m.lookback_windows)
    assert m.parquet_sha256 == manifest.sha256_file(tmp_data_dir / RTC_S1_PARQUET)


def test_tampered_parquet_rejected(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    path = tmp_data_dir / DIST_S1_PARQUET
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    result = _run_all(tmp_data_dir, tmp_path / 'out')
    assert result.exit_code != 0
    assert 'changed after its manifest was written' in result.output


def test_cache_without_manifest_rejected(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    _strip_manifests(tmp_data_dir)
    result = _run_all(tmp_data_dir, tmp_path / 'out')
    assert result.exit_code != 0
    assert 'no manifest' in result.output and '--refresh' in result.output and 'adopt-cache' in result.output
    status = json.loads((tmp_path / 'out' / 'run_status.json').read_text())
    assert (status['state'], status['failed_check']) == ('INCOMPLETE', 'setup')


def test_adopted_cache_needs_regression_flag(tmp_path: Path, tmp_data_dir: Path) -> None:
    ids = _seed(tmp_data_dir)
    _strip_manifests(tmp_data_dir)
    assert _adopt(tmp_data_dir).exit_code == 0
    m = manifest.load_table_manifest(tmp_data_dir / RTC_S1_PARQUET)
    assert m.provenance == manifest.ADOPTED_UNVERIFIED
    assert m.query_time_utc is None and m.file_mtime_utc

    refused = _run_all(tmp_data_dir, tmp_path / 'refused')
    assert refused.exit_code != 0
    assert 'adopted cache' in refused.output and '--regression-baseline' in refused.output
    assert not any(i in p.read_text() for p in (tmp_path / 'refused').iterdir() for i in ids)

    out_dir = tmp_path / 'out'
    result = _run_all(tmp_data_dir, out_dir, '--regression-baseline')
    assert result.exit_code == 0, result.output + repr(result.exception)
    run_manifest = json.loads((out_dir / manifest.RUN_MANIFEST_FILE).read_text())
    assert run_manifest['adopted_cache'] is True
    assert {t['provenance'] for t in run_manifest['tables'].values()} == {manifest.ADOPTED_UNVERIFIED}
    html = next(out_dir.glob('report_*.html')).read_text()
    assert 'ADOPTED CACHE: provenance unverified (regression run)' in html
    assert 'adopted_unverified' in html


@pytest.mark.parametrize('adopted', [False, True])
def test_provenance_time_labels_never_shared(tmp_path: Path, tmp_data_dir: Path, adopted: bool) -> None:
    _seed(tmp_data_dir)
    extra = []
    if adopted:
        _strip_manifests(tmp_data_dir)
        _adopt(tmp_data_dir)
        extra = ['--regression-baseline']
    out_dir = tmp_path / 'out'
    assert _run_all(tmp_data_dir, out_dir, *extra).exit_code == 0
    html = next(out_dir.glob('report_*.html')).read_text()
    provenance = html[html.index('<details class="prov"') : html.index('</details>')]
    run_manifest = json.loads((out_dir / manifest.RUN_MANIFEST_FILE).read_text())
    if adopted:
        assert 'file_mtime_utc (inferred)' in provenance and 'query_time_utc' not in provenance
    else:
        assert 'query_time_utc' in provenance and 'file_mtime_utc' not in provenance
        assert run_manifest['tables']['rtc_s1']['query_time_utc'] in provenance
        assert 'ADOPTED CACHE' not in html
    # The block is generated from run_manifest.json: the command line and tool version come from it.
    assert run_manifest['tool_version'] in provenance
    assert 'run-all --start 2026-01-01 --stop 2026-03-01' in provenance


def test_run_manifest_lists_outputs(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    out_dir = tmp_path / 'out'
    assert _run_all(tmp_data_dir, out_dir).exit_code == 0
    run_manifest = json.loads((out_dir / manifest.RUN_MANIFEST_FILE).read_text())
    assert set(run_manifest['outputs']) == {p.name for p in out_dir.iterdir()}
    assert set(run_manifest['tables']) == {'dist_s1', 'rtc_s1'}


def test_standalone_check_embeds_provenance(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    out_dir = tmp_path / 'out'
    args = ['check-duplicates', '--data-dir', str(tmp_data_dir), '--out-dir', str(out_dir)]
    assert CliRunner().invoke(cli, args).exit_code == 0
    html = next(out_dir.glob('report_*.html')).read_text()
    assert '<details class="prov"' in html and 'dist_s1 table' in html
    assert json.loads((out_dir / 'run_status.json').read_text())['state'] == 'COMPLETE'
