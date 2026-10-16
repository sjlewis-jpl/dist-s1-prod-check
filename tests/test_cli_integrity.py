"""Defect tests (Task 0.3) for end-to-end output integrity: F01, F03, F04, F09, F10, F17.

Every test drives the CLI offline against a seeded data dir; all network paths raise.
"""

import json
from pathlib import Path
from typing import Any, NoReturn

import boto3
import geopandas as gpd
import pandas as pd
import pytest
import requests
from click.testing import CliRunner, Result

from dist_s1_prod_check import cmr, coverage, tags
from dist_s1_prod_check.cli import cli
from dist_s1_prod_check.constants import DELTA_WINDOW_DAYS, INPUTS_CHECKPOINT_PARQUET
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, make_tags, seed_data_dir


TILE = '10SGC'
FIRST_ACQ = pd.Timestamp('2026-01-02T02:06:53')
REVISIT = pd.Timedelta(days=12)
PRE_OFFSET = pd.Timedelta(days=375)  # inside the first baseline window [post - 425 d, post - 365 d)
JANUARY = ['--start', '2026-01-01', '--stop', '2026-02-01']


def _no_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('network access attempted')


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests.Session, 'request', _no_network)
    monkeypatch.setattr(boto3, 'client', _no_network)
    monkeypatch.setattr(tags.rasterio, 'open', _no_network)
    for name in ('get_dist_s1_table', 'get_rtc_s1_table', 'get_rtc_s1_table_with_lookback'):
        monkeypatch.setattr(cmr, name, _no_network)


def _group_bursts() -> list[str]:
    bursts = bursts_for_tile(TILE)
    return bursts[bursts.track_number == 35].jpl_burst_id.tolist()


def _acq(k: int) -> pd.Timestamp:
    return FIRST_ACQ + k * REVISIT


def _pre_ids(post_acq: pd.Timestamp, rtc: gpd.GeoDataFrame) -> list[str]:
    """Pre RTC ids the enumerator selects for a post date: every pre granule in the first baseline window."""
    lo = (post_acq - pd.Timedelta(days=365 + DELTA_WINDOW_DAYS)).tz_localize('UTC')
    hi = (post_acq - pd.Timedelta(days=365)).tz_localize('UTC')
    return sorted(rtc[(rtc.acq_dt >= lo) & (rtc.acq_dt < hi)].opera_id)


def _world(n: int = 3, extra_pre: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """One tile, one acquisition group, `n` January passes 12 days apart, each with a DIST product and correct tags.

    The tags carry the inputs the enumerator expects and a correct prior chain (literal 'None' for the first).
    With `extra_pre`, one more baseline granule per burst is added to the RTC table for the first pass only, so
    the expected inputs of product 0 change while its tags do not.
    """
    group = _group_bursts()
    rtc_rows = []
    for k in range(n):
        post, pre = _acq(k), _acq(k) - PRE_OFFSET
        rtc_rows += [{'burst': b, 'acq': post, 'proc': post + pd.Timedelta(hours=5)} for b in group]
        rtc_rows += [{'burst': b, 'acq': pre, 'proc': pre + pd.Timedelta(hours=5)} for b in group]
    rtc = make_rtc(rtc_rows)
    dist_ids = [dist_id(TILE, _acq(k), _acq(k) + pd.Timedelta(days=1)) for k in range(n)]
    tag_rows = []
    for k, oid in enumerate(dist_ids):
        post_ids = sorted(rtc[rtc.acq_dt == _acq(k).tz_localize('UTC')].opera_id)
        prior = dist_ids[k - 1] if k else 'None'
        tag_rows.append(
            {
                'opera_id': oid,
                'pre_rtc_opera_ids': _pre_ids(_acq(k), rtc),
                'post_rtc_opera_ids': post_ids,
                'prior_dist_s1_product': prior,
            }
        )
    if extra_pre:
        # 5 days before the existing baseline of the first pass, so its expected pre inputs grow.
        extra = _acq(0) - PRE_OFFSET - pd.Timedelta(days=5)
        rtc = make_rtc([*rtc_rows, *({'burst': b, 'acq': extra, 'proc': extra} for b in group)])
    return make_dist([{'opera_id': o} for o in dist_ids]), rtc, make_tags(tag_rows)


def _invoke(runner: CliRunner, *args: str) -> Result:
    return runner.invoke(cli, [*args], catch_exceptions=True)


def _run_all(runner: CliRunner, data_dir: Path, out_dir: Path, *extra: str) -> Result:
    args = ['run-all', *extra, '--data-dir', str(data_dir), '--out-dir', str(out_dir), '--inputs-workers', '1']
    if '--start' not in extra:
        args[1:1] = JANUARY
    return _invoke(runner, *args)


def _csv(out_dir: Path, name: str) -> pd.DataFrame:
    paths = sorted(out_dir.glob(f'{name}_[0-9]*T[0-9]*.csv'))
    assert paths, f'no {name} CSV in {out_dir}: {sorted(p.name for p in out_dir.iterdir())}'
    return pd.read_csv(paths[-1]) if paths[-1].stat().st_size > 1 else pd.DataFrame()


def _report(out_dir: Path) -> tuple[str, dict[str, Any]]:
    paths = sorted(out_dir.glob('report_*.html'))
    assert paths, f'no HTML report in {out_dir}'
    html = paths[-1].read_text()
    marker = 'const DATA = '
    data, _ = json.JSONDecoder().raw_decode(html, html.index(marker) + len(marker))
    return html, data


def _all_output_text(out_dir: Path) -> str:
    if not out_dir.exists():
        return ''
    return '\n'.join(p.read_text() for p in out_dir.iterdir() if p.suffix in ('.csv', '.html', '.json'))


# ---------------------------------------------------------------------------------------------------------------
# F01: cached tables are reused regardless of the requested window / AOI.


# aa2014f: report titled '2026-02-01 to 2026-03-01' written over the January products, then exit 1 (empty coverage).
def test_f01_cache_window_mismatch_refused(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    dist, rtc, tag_df = _world()
    seed_data_dir(tmp_data_dir, dist, rtc, tag_df, window=('2026-01-01', '2026-02-01'))
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir, '--start', '2026-02-01', '--stop', '2026-03-01')
    leaked = [oid for oid in dist.opera_id if oid in _all_output_text(out_dir)]
    assert not leaked, f'February run (exit {result.exit_code}) reported January products {leaked}'
    # Refusing must be a ClickException that names the mismatch, not a crash.
    if result.exit_code != 0 and 'network access attempted' not in repr(result.exception):
        assert 'Error:' in result.output and ('window' in result.output or 'start' in result.output)


# aa2014f: exit 0; report subtitle 'over (10.0, 45.0, 11.0, 46.0)' while every row is tile 10SGC from the cache.
def test_f01_cache_bbox_mismatch_refused(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    dist, rtc, tag_df = _world()
    seed_data_dir(tmp_data_dir, dist, rtc, tag_df, target_tiles=[TILE])
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir, '--bbox', '10', '45', '11', '46')
    leaked = [oid for oid in dist.opera_id if oid in _all_output_text(out_dir)]
    assert not leaked, f'run over a European bbox (exit {result.exit_code}) reported cached {TILE} products'
    if result.exit_code != 0 and 'network access attempted' not in repr(result.exception):
        assert 'Error:' in result.output and 'bbox' in result.output.lower()


# ---------------------------------------------------------------------------------------------------------------
# F03: products without a tag URL silently drop out of confirmation and inputs.


# aa2014f: inputs CSV holds 1 product, not 2; the null-URL product is absent (confirmation tiles n_products=1 too).
@pytest.mark.xfail(strict=True, reason='F03: product with alert_url=None dropped from confirmation and inputs')
def test_f03_null_tag_url_is_unknown_not_dropped(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    dist, rtc, tag_df = _world(n=2)
    null_id = dist.opera_id.iloc[1]
    dist.loc[dist.opera_id == null_id, 'alert_url'] = None
    seed_data_dir(tmp_data_dir, dist, rtc, tag_df[tag_df.opera_id != null_id])
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir)
    assert result.exit_code == 0, result.output + repr(result.exception)

    df_inputs = _csv(out_dir, 'inputs')
    assert set(df_inputs.opera_id) == set(dist.opera_id), 'inputs universe is not the deduplicated DIST set'
    assert int(_csv(out_dir, 'confirmation_tiles').n_products.sum()) == 2

    # Future per-product outputs carry one status per product per check.
    for name, df in (('confirmation', _csv(out_dir, 'confirmation')), ('inputs', df_inputs)):
        assert df.opera_id.nunique() == 2, name
        row = df[df.opera_id == null_id]
        assert len(row) == 1, name
        assert row.status.iloc[0] == 'UNKNOWN', name
        assert 'missing_tag_url' in ' '.join(row.astype(str).iloc[0]), name


# ---------------------------------------------------------------------------------------------------------------
# F04: the inputs checkpoint is keyed only on opera_id.


def _check_inputs(runner: CliRunner, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    result = _invoke(runner, 'check-inputs', '--workers', '1', '--data-dir', str(data_dir), '--out-dir', str(out_dir))
    assert result.exit_code == 0, result.output + repr(result.exception)
    return _csv(out_dir, 'inputs')


# aa2014f: rerun returns the checkpointed inputs_correct=True for P; the new RTC table expects 3 more pre ids.
@pytest.mark.xfail(strict=True, reason='F04: checkpointed PASS survives an RTC re-download that changes inputs')
def test_f04_checkpoint_pass_reevaluated_after_rtc_change(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner
) -> None:
    dist, rtc, tag_df = _world()
    target = dist.opera_id.iloc[0]
    seed_data_dir(tmp_data_dir, dist, rtc, tag_df, target_tiles=[TILE])
    first = _check_inputs(cli_runner, tmp_data_dir, tmp_path / 'out1')
    assert first[first.opera_id == target].inputs_correct.tolist() == [True]
    assert (tmp_data_dir / INPUTS_CHECKPOINT_PARQUET).exists()

    # Sanity: without the checkpoint the new RTC table makes P fail.
    _, rtc_new, _ = _world(extra_pre=True)
    fresh_dir = tmp_path / 'fresh'
    seed_data_dir(fresh_dir, dist, rtc_new, tag_df, target_tiles=[TILE])
    fresh = _check_inputs(cli_runner, fresh_dir, tmp_path / 'out_fresh')
    assert not fresh[fresh.opera_id == target].inputs_correct.any()

    seed_data_dir(tmp_data_dir, rtc=rtc_new)
    second = _check_inputs(cli_runner, tmp_data_dir, tmp_path / 'out2')
    rows = second[second.opera_id == target]
    assert not rows.inputs_correct.any(), f'P still PASS from the checkpoint: {rows.to_dict("records")}'
    assert 'Pre RTC IDs expected but not found' in set(rows.issue_type)


# aa2014f: the checkpointed 'Tag read error' row is returned again (inputs_correct None) after the tag became readable.
@pytest.mark.xfail(strict=True, reason='F04: checkpointed Tag read error rows are never retried')
def test_f04_checkpoint_tag_error_retried(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    dist, rtc, tag_df = _world()
    target = dist.opera_id.iloc[0]
    broken = pd.concat(
        [make_tags([{'opera_id': target, 'error': 'RasterioIOError: 404'}]), tag_df[tag_df.opera_id != target]],
        ignore_index=True,
    )
    seed_data_dir(tmp_data_dir, dist, rtc, broken, target_tiles=[TILE])
    first = _check_inputs(cli_runner, tmp_data_dir, tmp_path / 'out1')
    assert first[first.opera_id == target].issue_type.tolist() == ['Tag read error']

    seed_data_dir(tmp_data_dir, tags=tag_df)
    second = _check_inputs(cli_runner, tmp_data_dir, tmp_path / 'out2')
    rows = second[second.opera_id == target]
    assert rows.inputs_correct.tolist() == [True], f'tag error not retried: {rows.to_dict("records")}'


# ---------------------------------------------------------------------------------------------------------------
# F09: a crash leaves partial outputs with no marker.


# aa2014f: exit 1 (RuntimeError) but the report has 4 sections, no INCOMPLETE banner and no run_status.json.
def test_f09_crash_marks_run_incomplete(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError('coverage exploded')

    monkeypatch.setattr(coverage, 'check_coverage', _boom)
    dist, rtc, tag_df = _world()
    seed_data_dir(tmp_data_dir, dist, rtc, tag_df, target_tiles=[TILE])
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir)
    assert result.exit_code != 0

    html, data = _report(out_dir)
    assert 'coverage' not in {s['key'] for s in data['sections']}
    assert 'INCOMPLETE' in html, 'partial HTML report carries no INCOMPLETE banner'
    assert 'coverage' in html.lower()

    status_path = out_dir / 'run_status.json'
    assert status_path.exists(), 'no run_status.json'
    status = json.loads(status_path.read_text())
    assert status.get('state', status.get('status')) == 'INCOMPLETE'
    assert status['failed_check'] == 'coverage'


def test_f09_complete_run_marked_complete(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    seed_data_dir(tmp_data_dir, *_world(), target_tiles=[TILE])
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir)
    assert result.exit_code == 0, result.output + repr(result.exception)
    html, data = _report(out_dir)
    assert 'INCOMPLETE' not in html
    assert data['run_status']['state'] == 'COMPLETE'
    status = json.loads((out_dir / 'run_status.json').read_text())
    assert status['state'] == 'COMPLETE'
    assert status['checks_done'] == ['duplicates', 'ordering', 'confirmation', 'inputs', 'coverage']


# aa2014f: ModuleNotFoundError: no dist_s1_prod_check.status (no run status is ever written or read).
def test_f09_stale_running_status_reads_incomplete(tmp_path: Path) -> None:
    from dist_s1_prod_check import status

    path = tmp_path / 'run_status.json'
    path.write_text(json.dumps({'state': 'RUNNING', 'started_utc': '2026-10-15T00:00:00Z'}))
    assert status.read_run_status(path).state == status.RunState.INCOMPLETE


# ---------------------------------------------------------------------------------------------------------------
# F10: sampling is not disclosed and ignores --seed.


def _sampled_run(runner: CliRunner, tmp_path: Path, seed: int) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    data_dir, out_dir = tmp_path / f'data_{seed}', tmp_path / f'out_{seed}'
    seed_data_dir(data_dir, *_world(), target_tiles=[TILE])
    result = _run_all(runner, data_dir, out_dir, '--sample-inputs', '1', '--seed', str(seed), '--expect-none-at-start')
    assert result.exit_code == 0, result.output + repr(result.exception)
    html, data = _report(out_dir)
    return _csv(out_dir, 'inputs'), html, data


def test_f10_fixture_seeds_choose_different_products() -> None:
    """Guard for the F10 test: with seeds 7 and 9, `DataFrame.sample(n=1)` picks different products of 3."""
    dedup = make_dist([{'tile': TILE, 'acq': _acq(k), 'proc': _acq(k) + pd.Timedelta(days=1)} for k in range(3)])
    assert dedup.sample(n=1, random_state=7).opera_id.iloc[0] != dedup.sample(n=1, random_state=9).opera_id.iloc[0]


# aa2014f: seeds 7 and 9 sample the same product (random_state hard-coded 42); sections carry no state/n_universe.
def test_f10_sampling_disclosed_and_seeded(tmp_path: Path, cli_runner: CliRunner) -> None:
    inputs7, html7, data7 = _sampled_run(cli_runner, tmp_path, 7)
    inputs9, _, _ = _sampled_run(cli_runner, tmp_path, 9)
    assert set(inputs7.opera_id) != set(inputs9.opera_id), 'seed 7 and seed 9 sampled the same product'

    sections = {s['key']: s for s in data7['sections']}
    for key in ('confirmation', 'inputs'):
        assert sections[key].get('state') == 'SAMPLED', key
        assert (sections[key].get('n_universe'), sections[key]['n_checked']) == (3, 1), key
    for key in ('duplicates', 'ordering', 'coverage'):
        assert sections[key].get('state') == 'COMPLETE', key
        assert sections[key].get('n_universe') == 3, key

    subtitle = data7['subtitle'].lower()
    assert 'sample' in subtitle and 'seed 7' in subtitle, data7['subtitle']
    assert 'expect-none-at-start' in html7.lower()


def test_f10_sample_tiles_disclosed_with_actual_count(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner
) -> None:
    seed_data_dir(tmp_data_dir, *_world(), target_tiles=[TILE])
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir, '--sample-tiles', '5', '--seed', '3')
    assert result.exit_code == 0, result.output + repr(result.exception)
    _, data = _report(out_dir)
    assert 'sample of 1 of 1 tiles (seed 3)' in data['subtitle']
    assert {s['state'] for s in data['sections']} == {'SAMPLED'}
    sampling = json.loads((out_dir / 'run_manifest.json').read_text())['sampling']
    assert (sampling['tiles']['n_selected'], sampling['tiles']['seed']) == (1, 3)


# ---------------------------------------------------------------------------------------------------------------
# F17: an absent prior_dist_s1_product key reads as "None".


class _FakeDataset:
    def __init__(self, tag_values: dict[str, str]) -> None:
        self._tags = tag_values

    def __enter__(self) -> '_FakeDataset':
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def tags(self) -> dict[str, str]:
        return self._tags


# aa2014f: confirmation_failures CSV is empty; the first product passes with prior_used='None' (key stored as '').
@pytest.mark.xfail(strict=True, reason='F17: absent prior_dist_s1_product key passes under --expect-none-at-start')
def test_f17_absent_prior_key_fails(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist, _, tag_df = _world(n=1)
    row = tag_df.iloc[0]
    readable = {k: row[k] for k in ('mgrs_tile_id', 'pre_rtc_opera_ids', 'post_rtc_opera_ids')}  # no prior key
    monkeypatch.setattr(tags.rasterio, 'open', lambda url, *a, **kw: _FakeDataset(readable))
    seed_data_dir(tmp_data_dir, dist)

    result = _invoke(cli_runner, 'fetch-tags', '--workers', '1', '--data-dir', str(tmp_data_dir))
    assert result.exit_code == 0, result.output + repr(result.exception)
    out_dir = tmp_path / 'out'
    result = _invoke(
        cli_runner,
        'check-confirmation',
        '--expect-none-at-start',
        '--data-dir',
        str(tmp_data_dir),
        '--out-dir',
        str(out_dir),
    )
    assert result.exit_code == 0, result.output + repr(result.exception)

    failures = _csv(out_dir, 'confirmation_failures')
    assert list(failures.get('opera_id', [])) == [row.opera_id], 'missing prior tag passed the chain check'
    assert 'prior tag absent' in failures.reason.iloc[0]
