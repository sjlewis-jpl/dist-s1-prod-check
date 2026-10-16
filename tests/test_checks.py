from pathlib import Path
from typing import NoReturn

import boto3
import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from dist_s1_prod_check import checks
from dist_s1_prod_check import cli as cli_module
from dist_s1_prod_check.cli import cli
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, make_tags, run_dir, seed_data_dir


TILE = '10SGC'
ACQ_A = pd.Timestamp('2026-01-10T02:06:53')
ACQ_B = ACQ_A + pd.Timedelta(days=12)
ACQ_C = ACQ_B + pd.Timedelta(days=12)
T1 = pd.Timestamp('2026-03-01T00:00:00')
T1_5 = pd.Timestamp('2026-03-05T00:00:00')
T2 = pd.Timestamp('2026-03-10T00:00:00')
T3 = pd.Timestamp('2026-03-20T00:00:00')


def _no_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('network access attempted')


def _run_capturing(
    monkeypatch: pytest.MonkeyPatch, cli_runner: CliRunner, args: list[str]
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Invoke the CLI offline and return the report sections and CSV frames it would write."""
    monkeypatch.setattr(requests.Session, 'request', _no_network)
    monkeypatch.setattr(boto3, 'client', _no_network)
    captured: dict = {'sections': [], 'csvs': {}}

    def _csvs(out_dir: Path, stamp: str, csvs: dict) -> None:
        captured['csvs'].update(csvs)

    def _report(out_dir: Path, stamp: str, tile_geoms: object, sections: list[dict], *a: object, **k: object) -> None:
        captured['sections'].extend(sections)

    monkeypatch.setattr(cli_module, '_write_csvs', _csvs)
    monkeypatch.setattr(cli_module, '_write_report', _report)
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    return captured['sections'], captured['csvs']


def _confirmation_status(csvs: dict[str, pd.DataFrame], opera_id: str) -> tuple[str, str]:
    """(status, reason) for one product; falls back to today's failures-only CSV when no status column exists."""
    df_all = csvs.get('confirmation')
    if df_all is not None and 'status' in df_all.columns:
        row = df_all[df_all.opera_id == opera_id].iloc[0]
        return str(row.status), str(row.reason)
    df_bad = csvs['confirmation_failures']
    bad = df_bad[df_bad.opera_id == opera_id]
    return ('FAIL', str(bad.reason.iloc[0])) if len(bad) else ('PASS', '')


A_V1 = dist_id(TILE, ACQ_A, T1)
A_V1_5 = dist_id(TILE, ACQ_A, T1_5)
A_V3 = dist_id(TILE, ACQ_A, T3)
A_GHOST = dist_id(TILE, ACQ_A, '2026-03-02T12:34:56')
B = dist_id(TILE, ACQ_B, T2)

F07_CASES = {
    # name: (A versions in the DIST table, B's prior, expected status, expected reason fragment)
    'stale_chain': ([A_V1, A_V3], A_V1, 'INFO', 'stale chain'),
    'outdated_prior': ([A_V1, A_V1_5], A_V1, 'FAIL', ''),
    'prior_processed_after_product': ([A_V1, A_V3], A_V3, 'FAIL', 'prior processed after product'),
    'prior_does_not_exist': ([A_V1], A_GHOST, 'FAIL', ''),
}


# aa2014f: confirmation compares truncated ids (tile + acquisition), so B is PASS in every case
@pytest.mark.parametrize('case', list(F07_CASES))
def test_f07_confirmation_compares_full_ids_by_processing_time(
    case: str, tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_versions, prior, want_status, want_reason = F07_CASES[case]
    dist = make_dist([{'opera_id': oid} for oid in [*a_versions, B]])
    tags = make_tags([{'opera_id': oid} for oid in a_versions] + [{'opera_id': B, 'prior_dist_s1_product': prior}])
    seed_data_dir(tmp_data_dir, dist=dist, tags=tags)

    _, csvs = _run_capturing(
        monkeypatch,
        cli_runner,
        ['check-confirmation', '--data-dir', str(tmp_data_dir), '--out-dir', str(tmp_path / 'out')],
    )

    status, reason = _confirmation_status(csvs, B)
    assert status == want_status, f'{case}: B is {status} ({reason!r})'
    assert want_reason in reason.lower()


def _ordering_inputs() -> list[dict]:
    """Build A, B, C in one tile: B-v1 is processed after C, then B is reprocessed much later."""
    return [
        {'tile': TILE, 'acq': ACQ_A, 'proc': T1},
        {'tile': TILE, 'acq': ACQ_B, 'proc': T2},
        {'tile': TILE, 'acq': ACQ_C, 'proc': T1_5},
        {'tile': TILE, 'acq': ACQ_B, 'proc': T3},
    ]


def _flagged(csvs: dict[str, pd.DataFrame]) -> set[str]:
    flagged = set()
    for df in csvs.values():
        if 'opera_id' not in df.columns:
            continue
        if 'out_of_order' in df.columns:
            flagged |= set(df[df.out_of_order.astype(bool)].opera_id)
        if 'status' in df.columns:
            flagged |= set(df[df.status == 'FAIL'].opera_id)
    return flagged


# aa2014f: B is dropped as reprocessed, C is compared only with A and is in order; nothing is flagged
def test_f08_first_production_view_flags_inversion(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_data_dir(tmp_data_dir, dist=make_dist(_ordering_inputs()))

    _, csvs = _run_capturing(
        monkeypatch, cli_runner, ['check-ordering', '--data-dir', str(tmp_data_dir), '--out-dir', str(tmp_path / 'out')]
    )

    assert dist_id(TILE, ACQ_C, T1_5) in _flagged(csvs), 'C was produced before B was first produced'


# aa2014f: the ordering section has no n_skipped; only 1 of 3 acquisitions is evaluated and it reads "0 flagged"
def test_f08_retained_view_reports_not_evaluated(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_data_dir(tmp_data_dir, dist=make_dist(_ordering_inputs()))

    sections, _ = _run_capturing(
        monkeypatch, cli_runner, ['check-ordering', '--data-dir', str(tmp_data_dir), '--out-dir', str(tmp_path / 'out')]
    )

    ordering = [s for s in sections if s['key'].startswith('ordering')]
    counts = [(s['key'], s.get('n_checked'), s.get('n_skipped')) for s in ordering]
    # Retained view: A is first in the tile and B is reprocessed, so only C is evaluated.
    assert ('ordering', 1, 2) in counts or ('ordering_retained', 1, 2) in counts, counts


# --- Task 1.6: product-version filter and deterministic deduplication -----------------------------------------------


def test_dedup_tie_broken_by_opera_id() -> None:
    acq, proc = pd.Timestamp('2026-01-02T02:06:53'), pd.Timestamp('2026-01-03T00:00:00')
    rows = [{'opera_id': dist_id('10SGC', acq, proc, sensor=s)} for s in ('S1C', 'S1A')]
    for order in (rows, rows[::-1]):
        df = make_dist(order)
        assert checks.deduplicate_products(df).opera_id.tolist() == [rows[0]['opera_id']]
        dup = checks.check_duplicates(df)
        assert set(dup.latest_opera_id) == {rows[0]['opera_id']}


def test_filter_product_versions_counts_exclusions() -> None:
    acq = pd.Timestamp('2026-01-02T02:06:53')
    ids = [dist_id('10SGC', acq + pd.Timedelta(days=12 * k), acq + pd.Timedelta(days=12 * k + 1)) for k in range(2)]
    ids.append(dist_id('10SGC', acq, acq + pd.Timedelta(days=90), version='v0.1'))
    kept, excluded = checks.filter_product_versions(make_dist([{'opera_id': i} for i in ids]), ['v1.0'])
    assert kept.opera_id.tolist() == ids[:2]
    assert excluded == {'v0.1': 1}
    kept, excluded = checks.filter_product_versions(make_dist([{'opera_id': i} for i in ids]), ['v1.0', 'v0.1'])
    assert len(kept) == 3 and excluded == {}


def test_run_all_excludes_other_versions(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    acq = pd.Timestamp('2026-01-02T02:06:53')
    v1 = dist_id('10SGC', acq, acq + pd.Timedelta(days=1))
    v01 = dist_id('10SGC', acq, acq + pd.Timedelta(days=2), version='v0.1')
    tags = make_tags([{'opera_id': v1, 'prior_dist_s1_product': 'None'}])
    rtc = make_rtc([{'burst': bursts_for_tile('10SGC').jpl_burst_id.iloc[0], 'acq': acq, 'proc': acq}])
    seed_data_dir(tmp_data_dir, dist=make_dist([{'opera_id': v1}, {'opera_id': v01}]), rtc=rtc, tags=tags)
    out_dir = tmp_path / 'out'
    args = ['run-all', '--start', '2026-01-01', '--stop', '2026-03-01', '--inputs-workers', '1']
    result = cli_runner.invoke(cli, [*args, '--data-dir', str(tmp_data_dir), '--out-dir', str(out_dir)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    html = next(run_dir(out_dir).glob('report_*.html')).read_text()
    assert 'excluded 1 products of other versions (v0.1: 1)' in html
    assert v01 not in html
    duplicates = pd.read_csv(next(run_dir(out_dir).glob('duplicates_*.csv')))
    assert duplicates.empty, 'the v0.1 product must not form a duplicate group with the v1.0 one'


# --- Task 3.2 (F17): prior tag presence and first-in-window priors -------------------------------------------------


@pytest.mark.parametrize(
    ('prior', 'present', 'want_status', 'want_reason'),
    [
        ('None', True, 'SKIPPED', 'first acquisition'),
        (dist_id(TILE, ACQ_A - pd.Timedelta(days=12), T1), True, 'SKIPPED', 'first acquisition'),
        (dist_id('11SLT', ACQ_A - pd.Timedelta(days=12), T1), True, 'FAIL', 'impossible prior'),
        (dist_id(TILE, ACQ_B, T1), True, 'FAIL', 'not before the product'),
        ('', False, 'FAIL', 'prior tag absent'),
    ],
)
def test_f17_first_in_window_prior(prior: str, present: bool, want_status: str, want_reason: str) -> None:
    df = checks.deduplicate_products(make_dist([{'opera_id': A_V1}]))
    tags = make_tags([{'opera_id': A_V1, 'prior_dist_s1_product': prior}])
    if not present:
        tags = tags.assign(prior_tag_present=False)
    row = checks.check_confirmation(df, tags).iloc[0]
    assert (row.status, want_reason in row.reason) == (want_status, True), row.reason


def test_f17_legacy_tag_rows_derive_prior_presence() -> None:
    tags = make_tags(
        [{'opera_id': A_V1, 'prior_dist_s1_product': ''}, {'opera_id': B, 'prior_dist_s1_product': 'None'}]
    )
    from dist_s1_prod_check.tags import prior_tag_present

    assert prior_tag_present(tags).tolist() == [False, True]


# --- Task 3.3 / 3.10 (F08, F36): first-production view, not-evaluated counts and ties ------------------------------


def test_f36_equal_processing_times_are_ties_not_failures() -> None:
    df = make_dist([{'tile': TILE, 'acq': ACQ_A, 'proc': T1}, {'tile': TILE, 'acq': ACQ_B, 'proc': T1}])
    for df_order in (
        checks.check_processing_order(checks.deduplicate_products(df)),
        checks.check_first_production_order(df),
    ):
        row = df_order.iloc[1]
        assert (row.status, bool(row.tie), bool(row.out_of_order)) == ('PASS', True, False)


def test_f08_ordering_tiles_count_not_evaluated() -> None:
    df_order = checks.check_processing_order(checks.deduplicate_products(make_dist(_ordering_inputs())))
    tiles = checks.ordering_by_tile(df_order).iloc[0]
    assert (tiles.n_products, tiles.n_evaluated, tiles.n_not_evaluated, tiles.n_reprocessed) == (3, 1, 2, 1)
