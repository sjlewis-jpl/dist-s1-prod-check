"""Reconcile a run's HTML report with the CSVs written beside it (Task 4.6).

The report and the CSVs are generated from the same result objects, so every count the page shows must be
recomputable from the CSVs: failure-table rows, per-status unit counts, section-specific counts, and the map (tile
failure counts plus unmappable rows equal the failure rows). `reconcile_run` returns every mismatch it finds, so a
test or a validation run can assert the list is empty.
"""

import json
from pathlib import Path

import pandas as pd

from dist_s1_prod_check.report import FAILURE_CSVS
from dist_s1_prod_check.status import EVALUATED, Status


def report_data(run_dir: str | Path) -> tuple[str, dict]:
    """Return the run's stamp and the `DATA` object embedded in its HTML report."""
    (html_path,) = sorted(Path(run_dir).glob('report_*.html'))[-1:]
    html = html_path.read_text()
    marker = 'const DATA = '
    data, _ = json.JSONDecoder().raw_decode(html, html.index(marker) + len(marker))
    return html_path.stem.removeprefix('report_'), data


def _csv(run_dir: Path, name: str, stamp: str) -> pd.DataFrame | None:
    path = run_dir / f'{name}_{stamp}.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False, low_memory=False) if path.stat().st_size > 1 else pd.DataFrame()


def _counts(statuses: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in statuses.value_counts().items() if v}


def _inputs_product_counts(df: pd.DataFrame) -> dict[str, int]:
    """Per-product status as the report computes it: UNKNOWN > FAIL > PASS, INFO counting as PASS."""
    status = df.status.replace({Status.INFO: Status.PASS})
    rank = status.map({Status.UNKNOWN: 0, Status.FAIL: 1, Status.PASS: 2})
    worst = rank.groupby(df.opera_id).min().map({0: Status.UNKNOWN, 1: Status.FAIL, 2: Status.PASS})
    return _counts(worst)


# Section key -> (per-unit CSV, how to get unit statuses from it). Subsections and sampled views are checked through
# their failure tables only.
def _unit_counts(key: str, run_dir: Path, stamp: str) -> dict[str, int] | None:
    if key in ('ordering', 'ordering_first_production', 'confirmation', 'coverage'):
        df = _csv(run_dir, key, stamp)
        return None if df is None else _counts(df.status) if len(df) else {}
    if key == 'inputs':
        df = _csv(run_dir, 'inputs', stamp)
        return None if df is None else _inputs_product_counts(df) if len(df) else {}
    return None


def reconcile_run(run_dir: str | Path) -> list[str]:
    """Return every disagreement between the report's embedded section counts and the run's CSVs."""
    run_dir = Path(run_dir)
    stamp, data = report_data(run_dir)
    problems: list[str] = []

    def check(key: str, what: str, shown: object, recomputed: object) -> None:
        if shown != recomputed:
            problems.append(f'{key}: {what} shown {shown!r}, recomputed {recomputed!r}')

    for sec in data['sections']:
        key = sec['key']
        failures = _csv(run_dir, FAILURE_CSVS.get(key, key), stamp)
        if failures is None:
            problems.append(f'{key}: no failure CSV {FAILURE_CSVS.get(key, key)}_{stamp}.csv')
        else:
            check(key, 'failure rows', sec['n_failures'], len(failures))
        mapped = sum(f['properties']['n_failures'] for f in sec['geojson']['features'])
        check(key, 'map failures + unmapped', mapped + sec['n_unmapped'], sec['n_failures'])
        check(key, 'n_checked', sec['n_checked'], sum(sec['counts'].get(s, 0) for s in EVALUATED))
        check(key, 'n_failed', sec['n_failed'], sec['counts'].get(Status.FAIL, 0))
        unit_counts = _unit_counts(key, run_dir, stamp)
        if unit_counts is not None and sec['state'] != 'SAMPLED':
            check(key, 'status counts', sec['counts'], unit_counts)
        if key == 'duplicates' and failures is not None and len(failures):
            check(key, 'n_groups', sec['n_groups'], failures.latest_opera_id.nunique())
            check(key, 'n_versions', sec['n_versions'], len(failures))
            check(key, 'n_superseded', sec['n_superseded'], int((failures.is_latest.astype(str) != 'True').sum()))
        if key == 'coverage':
            df_cov = _csv(run_dir, 'coverage', stamp)
            if df_cov is not None and len(df_cov):
                check(key, 'n_missing', sec['n_missing'], int((df_cov.status == Status.FAIL).sum()))
                check(key, 'n_eligible', sec['n_eligible'], int((df_cov.eligible.astype(str) == 'True').sum()))
                check(key, 'n_universe', sec['n_universe'], len(df_cov))
        if key == 'orphans' and failures is not None:
            check(key, 'UNKNOWN orphans', sec['counts'].get(Status.UNKNOWN, 0), len(failures))
    by_key = {s['key']: s for s in data['sections']}
    if {'inputs', 'inputs_post', 'inputs_pre', 'inputs_other'} <= set(by_key):
        parts = sum(by_key[k]['n_failures'] for k in ('inputs_post', 'inputs_pre', 'inputs_other'))
        check('inputs', 'subsection issue rows', parts, by_key['inputs']['n_failures'])
    return problems
