"""--offline: cached data only, every network step refused by name (hardening plan, Task 1.7)."""

import json
from pathlib import Path
from typing import NoReturn

import boto3
import pandas as pd
import pytest
import requests
from click.testing import CliRunner, Result

from dist_s1_prod_check import inputs, network, tags
from dist_s1_prod_check.cli import cli
from dist_s1_prod_check.constants import RTC_S1_PARQUET
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, make_tags, run_dir, seed_data_dir


TILE = '10SGC'
ACQ = pd.Timestamp('2026-01-02T02:06:53')


def _network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('network')


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests.Session, 'request', _network)
    monkeypatch.setattr(requests, 'get', _network)
    monkeypatch.setattr(boto3, 'client', _network)
    monkeypatch.setattr(tags.rasterio, 'open', _network)
    yield
    network.set_offline(False)


def _ids(n: int = 2) -> list[str]:
    return [dist_id(TILE, ACQ + pd.Timedelta(days=12 * k), ACQ + pd.Timedelta(days=12 * k + 1)) for k in range(n)]


def _seed(data_dir: Path, tag_rows: list[dict] | None = None) -> list[str]:
    ids = _ids()
    burst = bursts_for_tile(TILE).jpl_burst_id.iloc[0]
    rtc = make_rtc([{'burst': burst, 'acq': ACQ, 'proc': ACQ + pd.Timedelta(hours=5)}])
    if tag_rows is None:
        tag_rows = [{'opera_id': i, 'prior_dist_s1_product': p} for i, p in zip(ids, ['None', ids[0]])]
    seed_data_dir(data_dir, make_dist([{'opera_id': i} for i in ids]), rtc, make_tags(tag_rows), target_tiles=[TILE])
    return ids


def _run_all(data_dir: Path, out_dir: Path, *extra: str) -> Result:
    args = ['run-all', '--start', '2026-01-01', '--stop', '2026-03-01', '--offline', '--inputs-workers', '1', *extra]
    return CliRunner().invoke(cli, [*args, '--data-dir', str(data_dir), '--out-dir', str(out_dir)])


def test_offline_run_all_completes_without_network(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    out_dir = tmp_path / 'out'
    result = _run_all(tmp_data_dir, out_dir)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert json.loads((run_dir(out_dir) / 'run_status.json').read_text())['state'] == 'COMPLETE'
    assert json.loads((run_dir(out_dir) / 'run_manifest.json').read_text())['offline'] is True
    assert 'offline run' in next(run_dir(out_dir).glob('report_*.html')).read_text()


def test_offline_missing_tag_row_names_the_step(tmp_path: Path, tmp_data_dir: Path) -> None:
    ids = _seed(tmp_data_dir, tag_rows=[{'opera_id': _ids()[0], 'prior_dist_s1_product': 'None'}])
    result = _run_all(tmp_data_dir, tmp_path / 'out')
    assert result.exit_code != 0
    assert '--offline is set but fetching tags for 1 products with no cached tag row' in result.output
    status = json.loads((run_dir(tmp_path / 'out') / 'run_status.json').read_text())
    assert (status['state'], status['failed_check']) == ('INCOMPLETE', 'tags')
    assert ids


def test_offline_missing_cache_names_metadata_download(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    (tmp_data_dir / RTC_S1_PARQUET).unlink()
    result = _run_all(tmp_data_dir, tmp_path / 'out')
    assert result.exit_code != 0
    assert 'metadata download (missing rtc_s1_products.parquet)' in result.output
    assert (tmp_data_dir / 'dist_s1_products.parquet').exists(), 'offline refusal must not touch the cache'


def test_offline_refresh_refused(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    result = _run_all(tmp_data_dir, tmp_path / 'out', '--refresh')
    assert result.exit_code != 0 and 'metadata download (--refresh)' in result.output


def test_offline_tag_error_rows_kept_not_retried(tmp_path: Path, tmp_data_dir: Path) -> None:
    ids = _ids()
    _seed(tmp_data_dir, tag_rows=[{'opera_id': ids[0], 'error': 'RasterioIOError: 404'}, {'opera_id': ids[1]}])
    out_dir = tmp_path / 'out'
    result = _run_all(tmp_data_dir, out_dir)
    assert result.exit_code == 0, result.output + repr(result.exception)
    df = pd.read_csv(next(run_dir(out_dir).glob('inputs_[0-9]*.csv')))
    row = df[df.opera_id == ids[0]].iloc[0]
    assert row.issue_type == 'Tag read error' and pd.isna(row.inputs_correct)
    assert row.issue_value == 'RasterioIOError: 404 (not retried: offline)'
    cached = pd.read_parquet(tmp_data_dir / 'dist_s1_tags.parquet')
    assert cached.loc[cached.opera_id == ids[0], 'error'].iloc[0] == 'RasterioIOError: 404', 'parquet rewritten'


def test_offline_online_inputs_mode_refused(tmp_path: Path, tmp_data_dir: Path) -> None:
    _seed(tmp_data_dir)
    args = ['check-inputs', '--mode', 'online', '--offline', '--data-dir', str(tmp_data_dir)]
    result = CliRunner().invoke(cli, [*args, '--out-dir', str(tmp_path / 'out')])
    assert result.exit_code != 0 and 'online inputs mode' in result.output


def test_offline_guard_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    network.set_offline(True)
    with pytest.raises(network.OfflineError, match='online inputs mode'):
        inputs._enumerate_one(TILE, 35, '2026-01-02')
    with pytest.raises(network.OfflineError, match='reading tags'):
        tags.get_tags('https://example.com/x.tif')
