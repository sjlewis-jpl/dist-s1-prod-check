from pathlib import Path
from typing import NoReturn

import boto3
import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from dist_s1_prod_check import cli as cli_module
from dist_s1_prod_check.cli import cli
from factories import dist_id, make_dist, make_tags, seed_data_dir


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

    def _capture(out_dir: Path, stamp: str, tile_geoms: object, sections: list[dict], csvs: dict, *a: object) -> None:
        captured['sections'].extend(sections)
        captured['csvs'].update(csvs)

    monkeypatch.setattr(cli_module, '_write_outputs', _capture)
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
@pytest.mark.xfail(strict=True, reason='F07: confirmation compares truncated ids, ignoring version and processing time')
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
@pytest.mark.xfail(strict=True, reason='F08: ordering drops reprocessed acquisitions, hiding inversions')
def test_f08_first_production_view_flags_inversion(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_data_dir(tmp_data_dir, dist=make_dist(_ordering_inputs()))

    _, csvs = _run_capturing(
        monkeypatch, cli_runner, ['check-ordering', '--data-dir', str(tmp_data_dir), '--out-dir', str(tmp_path / 'out')]
    )

    assert dist_id(TILE, ACQ_C, T1_5) in _flagged(csvs), 'C was produced before B was first produced'


# aa2014f: the ordering section has no n_skipped; only 1 of 3 acquisitions is evaluated and it reads "0 flagged"
@pytest.mark.xfail(strict=True, reason='F08: retained-version ordering does not report acquisitions not evaluated')
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
