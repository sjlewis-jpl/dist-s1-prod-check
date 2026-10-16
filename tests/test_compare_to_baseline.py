"""Synthetic tests for scripts/compare_to_baseline.py (hardening plan, Task 0.6)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_to_baseline.py'
STAMP = '20260101T000000'
RTC = 'OPERA_L2_RTC-S1_T144-308019-IW{}_20241209T140048Z'


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('compare_to_baseline', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctb = _load()


def _inputs_rows() -> list[dict]:
    return [
        {'opera_id': 'P1', 'mgrs_tile_id': '11SKU', 'inputs_correct': True, 'issue_type': '', 'issue_value': ''},
        {
            'opera_id': 'P2',
            'mgrs_tile_id': '11SKU',
            'inputs_correct': False,
            'issue_type': 'Pre RTC IDs expected but not found',
            'issue_value': ';'.join(RTC.format(i) for i in (1, 2, 3)),
        },
        {
            'opera_id': 'P2',
            'mgrs_tile_id': '11SKU',
            'inputs_correct': False,
            'issue_type': 'Pre RTC IDs found but not expected',
            'issue_value': RTC.format(4),
        },
        {
            'opera_id': 'P3',
            'mgrs_tile_id': '11SKU',
            'inputs_correct': None,
            'issue_type': 'Tag read error',
            'issue_value': 'RasterioIOError: not found',
        },
    ]


def _write_run(out_dir: Path, inputs_rows: list[dict] | None = None, ordering_ids: list[str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_ids = ordering_ids or ['P1', 'P2', 'P3']
    frames = {
        'duplicates': pd.DataFrame(
            {'opera_id': ['D1', 'D2'], 'mgrs_tile_id': ['11SKU'] * 2, 'is_latest': [False, True], 'reason': ['a', 'b']}
        ),
        'ordering': pd.DataFrame(
            {
                'opera_id': ordering_ids,
                'mgrs_tile_id': '11SKU',
                'evaluated': [i > 0 for i in range(len(ordering_ids))],
                'out_of_order': False,
                'reason': '',
            }
        ),
        'confirmation_failures': pd.DataFrame({'opera_id': ['P2'], 'confirmed_ok': [False], 'reason': ['prior None']}),
        'coverage': pd.DataFrame(
            {
                'mgrs_tile_id': ['11SKU', '11SKU'],
                'track_token': ['71', '71'],
                'post_acq_dt': ['2026-01-02 14:08:47+00:00', '2026-01-14 14:08:47+00:00'],
                'pass_id': ['730', '731'],
                'product_found': [True, False],
                'missing_product_expected': [False, True],
                'reason': ['', 'missing'],
            }
        ),
        'inputs': pd.DataFrame(inputs_rows if inputs_rows is not None else _inputs_rows()),
        'inputs_failures': pd.DataFrame({'opera_id': ['ignored']}),
    }
    for name, df in frames.items():
        df.to_csv(out_dir / f'{name}_{STAMP}.csv', index=False)
    return out_dir


def _compare(tmp_path: Path, old_dir: Path, new_dir: Path, renames: dict | None = None) -> pd.DataFrame:
    return ctb.compare_dirs(old_dir, STAMP, new_dir, STAMP, tmp_path / 'deltas', renames).set_index('check')


def _changed(summary: pd.DataFrame, check: str) -> int:
    return int(summary.loc[check, ctb.CHANGE_CLASSES[1:]].sum())


def test_self_comparison_is_all_unchanged(tmp_path: Path) -> None:
    run = _write_run(tmp_path / 'run')
    summary = _compare(tmp_path, run, run)
    assert (summary.ambiguous == 0).all()
    assert (summary.n_rows == summary.unchanged).all()
    assert summary.identical_csv.all()
    assert summary.loc['inputs_product', 'n_rows'] == 3
    assert summary.loc['inputs_issue', 'n_rows'] == 5  # 3 + 1 RTC ids for P2, 1 free-text issue for P3
    deltas = pd.read_csv(tmp_path / 'deltas' / 'inputs_product.csv', keep_default_na=False).set_index('key')
    assert deltas.loc[['P1', 'P2', 'P3'], 'old_status'].tolist() == ['PASS', 'FAIL', 'UNKNOWN']
    assert {p.name for p in (tmp_path / 'deltas').iterdir()} == {
        'duplicates.csv',
        'ordering.csv',
        'confirmation.csv',
        'coverage.csv',
        'inputs_product.csv',
        'inputs_issue.csv',
    }


def test_removing_one_input_id_is_one_issue_change(tmp_path: Path) -> None:
    rows = _inputs_rows()
    rows[1]['issue_value'] = ';'.join(RTC.format(i) for i in (1, 3))
    summary = _compare(tmp_path, _write_run(tmp_path / 'old'), _write_run(tmp_path / 'new', inputs_rows=rows))
    assert _changed(summary, 'inputs_issue') == 1
    assert summary.loc['inputs_issue', 'removed_row'] == 1
    assert _changed(summary, 'inputs_product') <= 1
    for check in ('duplicates', 'ordering', 'confirmation', 'coverage'):
        assert _changed(summary, check) == 0


def test_repeated_input_id_is_compared_as_multiset(tmp_path: Path) -> None:
    rows = _inputs_rows()
    rows[2]['issue_value'] = f'{RTC.format(4)};{RTC.format(4)}'
    summary = _compare(tmp_path, _write_run(tmp_path / 'old'), _write_run(tmp_path / 'new', inputs_rows=rows))
    assert summary.loc['inputs_issue', 'added_row'] == 1
    assert summary.loc['inputs_issue', 'ambiguous'] == 0


def test_duplicated_key_is_ambiguous(tmp_path: Path) -> None:
    old = _write_run(tmp_path / 'old')
    new = _write_run(tmp_path / 'new', ordering_ids=['P1', 'P2', 'P2'])
    summary = _compare(tmp_path, old, new)
    assert summary.loc['ordering', 'ambiguous'] == 1
    assert summary.loc['ordering', 'unchanged'] == 1
    assert _changed(summary, 'ordering') == 1  # P3 removed; the ambiguous P2 rows are not turned into deltas
    deltas = pd.read_csv(tmp_path / 'deltas' / 'ordering.csv', keep_default_na=False)
    assert (deltas[deltas.key == 'P2'].change_class == 'ambiguous').all()


def test_key_falls_back_to_post_acq_dt() -> None:
    df = pd.DataFrame({'mgrs_tile_id': ['T', 'T'], 'track_token': ['1', '1'], 'pass_id': ['5', '5']})
    df['post_acq_dt'] = ['2026-01-01', '2026-01-13']
    cols = ctb.resolve_key_columns(df, df, ('mgrs_tile_id', 'track_token', 'pass_id'))
    assert cols[-1] == 'post_acq_dt'
    unique = ctb.resolve_key_columns(df.iloc[:1], df.iloc[:1], ('mgrs_tile_id', 'track_token', 'pass_id'))
    assert unique == ['mgrs_tile_id', 'track_token', 'pass_id']


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{'key': k, 'status': s, 'reason': ''} for k, s in rows], columns=['key', 'status', 'reason'])


@pytest.mark.parametrize(
    ('old', 'new', 'expected'),
    [
        ([('a', 'PASS')], [('a', 'PASS')], 'unchanged'),
        ([('a', 'PASS')], [('a', 'FAIL')], 'newly_failing'),
        ([('a', 'FAIL')], [('a', 'PASS')], 'newly_passing'),
        ([('a', 'FAIL')], [('a', 'UNKNOWN')], 'status_refined'),
        ([('a', 'PASS')], [('a', 'INFO')], 'status_refined'),
        ([], [('a', 'PASS')], 'added_row'),
        ([('a', 'PASS')], [], 'removed_row'),
    ],
)
def test_each_change_class(old: list, new: list, expected: str) -> None:
    deltas = ctb.compare_frames(_frame(old), _frame(new))
    assert deltas.change_class.tolist() == [expected]


def test_failures_only_side_reads_absent_key_as_pass() -> None:
    deltas = ctb.compare_frames(_frame([('a', 'FAIL')]), _frame([]), old_failures_only=True, new_failures_only=True)
    assert deltas.change_class.tolist() == ['newly_passing']
    deltas = ctb.compare_frames(_frame([]), _frame([('a', 'FAIL')]), old_failures_only=True)
    assert deltas.change_class.tolist() == ['newly_failing']


def test_status_column_overrides_legacy_booleans() -> None:
    df = pd.DataFrame({'opera_id': ['a', 'b'], 'confirmed_ok': ['False', 'False'], 'status': ['UNKNOWN', 'FAIL']})
    assert ctb.row_status('confirmation', df).tolist() == ['UNKNOWN', 'FAIL']
    assert ctb.row_status('confirmation', df.drop(columns='status')).tolist() == ['FAIL', 'FAIL']


def test_issue_type_rename_is_applied_to_old_side(tmp_path: Path) -> None:
    rows = _inputs_rows()
    rows[3]['issue_type'] = 'Tag unreadable'
    old, new = _write_run(tmp_path / 'old'), _write_run(tmp_path / 'new', inputs_rows=rows)
    assert _changed(_compare(tmp_path, old, new), 'inputs_issue') == 2
    summary = _compare(tmp_path, old, new, renames={'Tag read error': 'Tag unreadable'})
    assert _changed(summary, 'inputs_issue') == 0
    assert summary.loc['inputs_product', 'reason_diff'] == 0


def test_find_csv_matches_exact_prefix(tmp_path: Path) -> None:
    run = _write_run(tmp_path / 'run')
    assert ctb.find_csv(run, 'inputs', STAMP).name == f'inputs_{STAMP}.csv'
    assert ctb.find_csv(run, 'confirmation', STAMP) is None
    assert ctb.find_stamps(run) == [STAMP]
