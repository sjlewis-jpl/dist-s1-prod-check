"""Result states shared by every check and section, and the run status file written by each command."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd


RUN_STATUS_FILE = 'run_status.json'


class Status(StrEnum):
    """Status of one checked unit (product, pass, link or group)."""

    PASS = 'PASS'  # checked, correct
    FAIL = 'FAIL'  # checked, wrong
    INFO = 'INFO'  # checked, noteworthy but not a failure
    UNKNOWN = 'UNKNOWN'  # could not be checked
    SKIPPED = 'SKIPPED'  # deliberately not checked


class SectionState(StrEnum):
    """State of one report section."""

    COMPLETE = 'COMPLETE'  # ran to completion over its whole universe
    SAMPLED = 'SAMPLED'  # ran over a sample; universe and sample sizes are recorded
    EMPTY = 'EMPTY'  # universe was empty: 0 items checked
    ERROR = 'ERROR'  # raised; partial or no results
    NOT_RUN = 'NOT_RUN'  # never started because an earlier check crashed


class RunState(StrEnum):
    """State of a whole run, as written to ``run_status.json``."""

    RUNNING = 'RUNNING'
    COMPLETE = 'COMPLETE'
    INCOMPLETE = 'INCOMPLETE'


EVALUATED = (Status.PASS, Status.FAIL, Status.INFO)


def is_green(state: SectionState, counts: dict[str, int], n_checked: int) -> bool:
    """Return whether a section is green.

    Green needs a COMPLETE state, at least one evaluated item, and no FAIL or UNKNOWN items.
    """
    return (
        state == SectionState.COMPLETE
        and n_checked > 0
        and not counts.get(Status.FAIL, 0)
        and not counts.get(Status.UNKNOWN, 0)
    )


def _now_utc() -> str:
    return pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


@dataclass
class RunStatus:
    """A run status read back from ``run_status.json`` (a persisted RUNNING reads as INCOMPLETE)."""

    state: RunState
    raw_state: str
    checks_planned: list[str] = field(default_factory=list)
    checks_done: list[str] = field(default_factory=list)
    failed_check: str | None = None
    error: str | None = None
    started_utc: str | None = None
    updated_utc: str | None = None

    @property
    def not_run(self) -> list[str]:
        skip = {*self.checks_done, self.failed_check}
        return [c for c in self.checks_planned if c not in skip]


def write_run_status(
    out_dir: str | Path,
    state: RunState,
    *,
    checks_planned: list[str],
    checks_done: list[str],
    failed_check: str | None = None,
    error: str | None = None,
) -> Path:
    """Write `out_dir/run_status.json` atomically (temporary file, then `os.replace`).

    The first write of a run sets `started_utc`; later writes keep it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RUN_STATUS_FILE
    started = None
    if path.exists() and state != RunState.RUNNING:
        try:
            started = json.loads(path.read_text()).get('started_utc')
        except (OSError, ValueError):
            started = None
    now = _now_utc()
    payload = {
        'state': str(state),
        'started_utc': started or now,
        'updated_utc': now,
        'checks_planned': list(checks_planned),
        'checks_done': list(checks_done),
        'failed_check': failed_check,
        'error': error,
    }
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)
    return path


def read_run_status(path: str | Path) -> RunStatus:
    """Read a run status file.

    A file still at RUNNING belongs to a run that is in progress or was killed, so it reads as INCOMPLETE;
    `raw_state` keeps the stored value.
    """
    data = json.loads(Path(path).read_text())
    raw = data.get('state', data.get('status', RunState.INCOMPLETE))
    state = RunState.COMPLETE if raw == RunState.COMPLETE else RunState.INCOMPLETE
    return RunStatus(
        state=state,
        raw_state=raw,
        checks_planned=data.get('checks_planned', []),
        checks_done=data.get('checks_done', []),
        failed_check=data.get('failed_check'),
        error=data.get('error'),
        started_utc=data.get('started_utc'),
        updated_utc=data.get('updated_utc'),
    )
