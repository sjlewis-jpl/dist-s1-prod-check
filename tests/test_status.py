"""Run status file and result states (hardening plan, Task 1.1)."""

import json
from pathlib import Path

import pytest

from dist_s1_prod_check import status


def test_write_read_round_trip(tmp_path: Path) -> None:
    path = status.write_run_status(
        tmp_path, status.RunState.INCOMPLETE, checks_planned=['a', 'b'], checks_done=['a'], failed_check='b', error='E'
    )
    assert path == tmp_path / status.RUN_STATUS_FILE
    run = status.read_run_status(path)
    assert run.state == status.RunState.INCOMPLETE
    assert (run.checks_planned, run.checks_done, run.failed_check, run.error) == (['a', 'b'], ['a'], 'b', 'E')
    assert run.not_run == []


def test_write_is_atomic(tmp_path: Path) -> None:
    status.write_run_status(tmp_path, status.RunState.RUNNING, checks_planned=['a'], checks_done=[])
    status.write_run_status(tmp_path, status.RunState.COMPLETE, checks_planned=['a'], checks_done=['a'])
    assert sorted(p.name for p in tmp_path.iterdir()) == [status.RUN_STATUS_FILE]
    data = json.loads((tmp_path / status.RUN_STATUS_FILE).read_text())
    assert data['state'] == 'COMPLETE'
    assert data['started_utc'] and data['updated_utc']


def test_running_reads_incomplete_but_keeps_raw_state(tmp_path: Path) -> None:
    status.write_run_status(tmp_path, status.RunState.RUNNING, checks_planned=['a', 'b'], checks_done=['a'])
    run = status.read_run_status(tmp_path / status.RUN_STATUS_FILE)
    assert run.state == status.RunState.INCOMPLETE
    assert run.raw_state == 'RUNNING'
    assert run.not_run == ['b']


def test_started_utc_survives_updates(tmp_path: Path) -> None:
    status.write_run_status(tmp_path, status.RunState.RUNNING, checks_planned=['a'], checks_done=[])
    first = json.loads((tmp_path / status.RUN_STATUS_FILE).read_text())['started_utc']
    status.write_run_status(tmp_path, status.RunState.COMPLETE, checks_planned=['a'], checks_done=['a'])
    assert json.loads((tmp_path / status.RUN_STATUS_FILE).read_text())['started_utc'] == first


@pytest.mark.parametrize(
    ('state', 'counts', 'n_checked', 'green'),
    [
        ('COMPLETE', {}, 3, True),
        ('COMPLETE', {'FAIL': 1}, 3, False),
        ('COMPLETE', {'UNKNOWN': 1}, 3, False),
        ('COMPLETE', {}, 0, False),
        ('SAMPLED', {}, 3, False),
        ('EMPTY', {}, 0, False),
        ('ERROR', {}, 3, False),
    ],
)
def test_is_green(state: str, counts: dict, n_checked: int, green: bool) -> None:
    assert status.is_green(status.SectionState(state), counts, n_checked) is green


# F34: runs shared one directory, so a crashed run's files and a later run's could be read together.
def test_f34_each_run_gets_its_own_directory(tmp_path: Path) -> None:
    from dist_s1_prod_check.cli import latest_run_directory, run_directory

    first = run_directory(tmp_path, '20261016T000000')
    second = run_directory(tmp_path, '20261016T000000')
    assert first != second and first.parent == second.parent == tmp_path
    assert latest_run_directory(tmp_path) == second
    assert not list(tmp_path.glob('*.tmp')) and not list(tmp_path.glob('.*.tmp'))
