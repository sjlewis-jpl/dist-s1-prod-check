"""Row-level comparison of a dist-s1-prod-check output directory against a recorded baseline run.

Usage::

    compare_to_baseline.py <baseline> <new_out_dir> [--phase NAME] [--new-stamp STAMP] [--expect-unchanged]

``<baseline>`` names an entry of ``docs/prod-gate-hardening/baselines.json`` (``uat``, ``pop1``, ``pop1_aoi``).
For every check the old and new CSVs are joined row by row and each row gets one change class:

- ``unchanged``: same status on both sides.
- ``newly_failing``: the new status is FAIL and the old one was not.
- ``newly_passing``: the old status was FAIL and the new one is PASS.
- ``status_refined``: any other status change, e.g. FAIL -> UNKNOWN or PASS -> INFO.
- ``added_row`` / ``removed_row``: the key exists on one side only.
- ``ambiguous``: the key matches more than one row on either side. Such rows are counted and listed, never turned
  into one of the classes above.

Results go to ``<data_root>/hardening_review/deltas/<baseline>/<phase>/<check>.csv`` with the columns
``key, old_status, new_status, old_reason, new_reason, change_class``, and a summary table is printed. The summary
also counts ``reason_diff`` (rows classed ``unchanged`` whose reason text differs) and whether the source CSVs are
byte-identical.

Per-row source CSV and key for each check (files are ``<prefix>_<YYYYmmddTHHMMSS>.csv``, found by pattern):

=============  ==========================================================  ===================================
check          source CSV (first one present wins)                         key
=============  ==========================================================  ===================================
duplicates     ``duplicates`` (failures only: lists duplicate-group rows)  ``opera_id``
ordering       ``ordering`` (every deduplicated product)                   ``opera_id``
confirmation   ``confirmation`` (future full table), else                  ``opera_id``
               ``confirmation_failures`` (failures only; aa2014f)
coverage       ``coverage`` (every expected pass)                          ``mgrs_tile_id, track_token, pass_id``
inputs         ``inputs`` (every product, one row per issue)               product: ``opera_id``;
                                                                           issue: ``opera_id, issue_type,
                                                                           input_id, ordinal``
=============  ==========================================================  ===================================

A *failures only* CSV lists only flagged rows, so a key absent from it is read as an implicit PASS (with an empty
reason) rather than a removed row. If a key is not unique within a CSV, ``post_acq_dt`` is added to the key when both
sides have that column; keys that are still duplicated are ``ambiguous``.

The inputs check is compared at two levels. *Product level* aggregates each ``opera_id`` into one status (FAIL over
UNKNOWN over INFO over SKIPPED over PASS) with a reason listing its issue count and issue types. *Issue level* splits
``issue_value`` into one record per RTC-S1 input id (when every ``;``-separated token is an OPERA id; otherwise the
whole value is one record), adds a duplicate ordinal for repeats, and compares the two sides as multisets. Old issue
types are first renamed through ``issue_type_renames`` in ``baselines.json``.

Statuses: when a CSV has a ``status`` column (the schema planned for Phase 1) it is used as is. Otherwise the status
is derived from the aa2014f booleans by the functions in ``LEGACY_STATUS``; that table is the extension point for
mapping later schemas back onto the old ones.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASELINES = Path(__file__).resolve().parents[2] / 'docs' / 'prod-gate-hardening' / 'baselines.json'
STATUS_COLUMN = 'status'
FALLBACK_KEY_COLUMN = 'post_acq_dt'
CHANGE_CLASSES = ['unchanged', 'newly_failing', 'newly_passing', 'status_refined', 'added_row', 'removed_row']
DELTA_COLUMNS = ['key', 'old_status', 'new_status', 'old_reason', 'new_reason', 'change_class']
STATUS_PRECEDENCE = ['FAIL', 'UNKNOWN', 'INFO', 'SKIPPED', 'PASS']
STAMP_RE = re.compile(r'_(\d{8}T\d{6})\.(?:csv|html)$')


@dataclass(frozen=True)
class CheckSpec:
    """How to read one check's per-row CSV.

    Parameters
    ----------
    name : str
        Check name, also the name of the delta CSV.
    sources : tuple of (str, bool)
        Candidate CSV prefixes in priority order, each with a flag that is True if the CSV lists failures only.
    key : tuple of str
        Columns that identify a row.
    reason : str
        Column holding the human-readable reason.
    """

    name: str
    sources: tuple[tuple[str, bool], ...]
    key: tuple[str, ...]
    reason: str = 'reason'


ROW_CHECKS = [
    CheckSpec('duplicates', (('duplicates', True),), ('opera_id',)),
    CheckSpec('ordering', (('ordering', False),), ('opera_id',)),
    CheckSpec('confirmation', (('confirmation', False), ('confirmation_failures', True)), ('opera_id',)),
    CheckSpec('coverage', (('coverage', False),), ('mgrs_tile_id', 'track_token', 'pass_id')),
]
INPUTS_SOURCE = 'inputs'


def _is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq('true')


def _legacy_duplicates(df: pd.DataFrame) -> pd.Series:
    # aa2014f lists every member of a duplicate group and counts each listed row as flagged.
    return pd.Series('FAIL', index=df.index)


def _legacy_ordering(df: pd.DataFrame) -> pd.Series:
    status = np.select([_is_true(df.out_of_order), ~_is_true(df.evaluated)], ['FAIL', 'SKIPPED'], 'PASS')
    return pd.Series(status, index=df.index)


def _legacy_confirmation(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(_is_true(df.confirmed_ok), 'PASS', 'FAIL'), index=df.index)


def _legacy_coverage(df: pd.DataFrame) -> pd.Series:
    found = _is_true(df.product_found)
    expected = _is_true(df.missing_product_expected)
    return pd.Series(np.select([found, expected], ['PASS', 'FAIL'], 'SKIPPED'), index=df.index)


def _legacy_inputs(df: pd.DataFrame) -> pd.Series:
    value = df.inputs_correct.astype(str).str.strip().str.lower()
    return pd.Series(np.select([value.eq('true'), value.eq('false')], ['PASS', 'FAIL'], 'UNKNOWN'), index=df.index)


# Maps each check to the function deriving a status from the aa2014f boolean columns. A CSV with a STATUS_COLUMN
# bypasses this table. If a later schema renames or drops a boolean without adding STATUS_COLUMN, add the
# translation here so baseline and new runs still land on the same status vocabulary.
LEGACY_STATUS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    'duplicates': _legacy_duplicates,
    'ordering': _legacy_ordering,
    'confirmation': _legacy_confirmation,
    'coverage': _legacy_coverage,
    'inputs': _legacy_inputs,
}


def row_status(check: str, df: pd.DataFrame) -> pd.Series:
    """Return one status per row: the ``status`` column if present, else the legacy boolean mapping."""
    if STATUS_COLUMN in df.columns:
        return df[STATUS_COLUMN].astype(str)
    if df.empty:
        return pd.Series(dtype=str, index=df.index)
    return LEGACY_STATUS[check](df)


def find_stamps(out_dir: Path) -> list[str]:
    """Return the sorted run stamps of the CSV/HTML outputs in ``out_dir``."""
    return sorted({m.group(1) for p in out_dir.iterdir() if (m := STAMP_RE.search(p.name))})


def find_csv(out_dir: Path, prefix: str, stamp: str) -> Path | None:
    """Return ``<out_dir>/<prefix>_<stamp>.csv`` if it exists (exact prefix, so ``inputs`` is not ``inputs_pre``)."""
    pattern = re.compile(rf'^{re.escape(prefix)}_{re.escape(stamp)}\.csv$')
    matches = [p for p in out_dir.iterdir() if pattern.match(p.name)]
    return matches[0] if matches else None


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as text, so values compare exactly as written."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sha256(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def _join_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str, index=df.index)
    key = df[cols[0]].astype(str)
    return key.str.cat([df[c].astype(str) for c in cols[1:]], sep='|') if len(cols) > 1 else key


def resolve_key_columns(old: pd.DataFrame, new: pd.DataFrame, key: tuple[str, ...]) -> list[str]:
    """Return ``key``, extended with ``post_acq_dt`` if the key is not unique on a side that has that column."""
    cols = list(key)
    not_unique = any(not df.empty and df.duplicated(subset=cols).any() for df in (old, new))
    have_fallback = all(df.empty or FALLBACK_KEY_COLUMN in df.columns for df in (old, new))
    if not_unique and have_fallback and FALLBACK_KEY_COLUMN not in cols:
        cols.append(FALLBACK_KEY_COLUMN)
    return cols


def classify(old_status: str, new_status: str) -> str:
    """Return the change class for a key present on both sides."""
    if old_status == new_status:
        return 'unchanged'
    if new_status == 'FAIL':
        return 'newly_failing'
    if old_status == 'FAIL' and new_status == 'PASS':
        return 'newly_passing'
    return 'status_refined'


def compare_frames(
    old: pd.DataFrame, new: pd.DataFrame, old_failures_only: bool = False, new_failures_only: bool = False
) -> pd.DataFrame:
    """Join two ``key, status, reason`` frames and classify every key.

    Parameters
    ----------
    old, new : pandas.DataFrame
        Frames with columns ``key``, ``status`` and ``reason``.
    old_failures_only, new_failures_only : bool
        True if that side lists failures only, so an absent key there is an implicit PASS.

    Returns
    -------
    pandas.DataFrame
        One row per key with ``DELTA_COLUMNS``. Keys that occur more than once on either side get one row per
        occurrence with ``change_class`` ``ambiguous``.
    """
    ambiguous_keys = set(old.key[old.key.duplicated()]) | set(new.key[new.key.duplicated()])
    amb_old = old[old.key.isin(ambiguous_keys)]
    amb_new = new[new.key.isin(ambiguous_keys)]
    ambiguous = pd.concat(
        [
            pd.DataFrame({'key': amb_old.key, 'old_status': amb_old.status, 'old_reason': amb_old.reason}),
            pd.DataFrame({'key': amb_new.key, 'new_status': amb_new.status, 'new_reason': amb_new.reason}),
        ],
        ignore_index=True,
    )
    ambiguous['change_class'] = 'ambiguous'

    merged = pd.merge(
        old[~old.key.isin(ambiguous_keys)].rename(columns={'status': 'old_status', 'reason': 'old_reason'}),
        new[~new.key.isin(ambiguous_keys)].rename(columns={'status': 'new_status', 'reason': 'new_reason'}),
        on='key',
        how='outer',
        indicator=True,
    )
    only_old = merged._merge.eq('left_only')
    only_new = merged._merge.eq('right_only')
    if old_failures_only:
        merged.loc[only_new, ['old_status', 'old_reason']] = ['PASS', '']
    if new_failures_only:
        merged.loc[only_old, ['new_status', 'new_reason']] = ['PASS', '']
    classes = [classify(o, n) for o, n in zip(merged.old_status.fillna(''), merged.new_status.fillna(''))]
    merged['change_class'] = classes
    if not old_failures_only:
        merged.loc[only_new, 'change_class'] = 'added_row'
    if not new_failures_only:
        merged.loc[only_old, 'change_class'] = 'removed_row'
    out = pd.concat([merged.drop(columns='_merge'), ambiguous], ignore_index=True)
    return out[DELTA_COLUMNS].fillna('').sort_values(['change_class', 'key'], kind='stable').reset_index(drop=True)


def _row_frame(check: str, df: pd.DataFrame, key_cols: list[str], reason: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {'key': pd.Series(dtype=str), 'status': pd.Series(dtype=str), 'reason': pd.Series(dtype=str)}
        )
    reasons = df[reason].astype(str) if reason in df.columns else pd.Series('', index=df.index)
    return pd.DataFrame({'key': _join_key(df, key_cols), 'status': row_status(check, df), 'reason': reasons})


def _load_source(out_dir: Path, stamp: str, spec: CheckSpec) -> tuple[pd.DataFrame, bool, Path | None]:
    for prefix, failures_only in spec.sources:
        path = find_csv(out_dir, prefix, stamp)
        if path is not None:
            return read_csv(path), failures_only, path
    return pd.DataFrame(), False, None


def _summary_row(check: str, deltas: pd.DataFrame, old_path: Path | None, new_path: Path | None) -> dict:
    counts = deltas.change_class.value_counts()
    unchanged = deltas[deltas.change_class == 'unchanged']
    row = {'check': check, 'n_rows': len(deltas)}
    row.update({c: int(counts.get(c, 0)) for c in CHANGE_CLASSES})
    row['ambiguous'] = int(deltas[deltas.change_class == 'ambiguous'].key.nunique())
    row['reason_diff'] = int((unchanged.old_reason != unchanged.new_reason).sum())
    row['identical_csv'] = old_path is not None and _sha256(old_path) == _sha256(new_path)
    row['old_csv'] = old_path.name if old_path else ''
    row['new_csv'] = new_path.name if new_path else ''
    return row


def product_level(df: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    """Aggregate an inputs CSV to one ``key, status, reason`` row per ``opera_id``."""
    if df.empty:
        return _row_frame('inputs', df, ['opera_id'], 'issue_type')
    rank = {s: i for i, s in enumerate(STATUS_PRECEDENCE)}
    work = pd.DataFrame(
        {
            'opera_id': df.opera_id.astype(str),
            'status': row_status('inputs', df).to_numpy(),
            'issue_type': df.issue_type.astype(str).replace(renames).to_numpy(),
        }
    )
    # A status outside the known vocabulary ranks first, so it surfaces instead of hiding behind PASS.
    work['rank'] = work.status.map(rank).fillna(-1)
    work['has_issue'] = work.issue_type != ''
    grouped = work.groupby('opera_id', sort=True)
    status = work.loc[grouped['rank'].idxmin(), ['opera_id', 'status']].set_index('opera_id').status
    n_issues = grouped.has_issue.sum()
    types = work[work.has_issue].groupby('opera_id').issue_type.agg(lambda s: '; '.join(sorted(set(s))))
    types = types.reindex(n_issues.index, fill_value='')
    reason = n_issues.astype(int).astype(str) + ' issue(s): ' + types
    return pd.DataFrame({'key': status.index, 'status': status.to_numpy(), 'reason': reason.loc[status.index]})


def _split_inputs(value: str) -> list[str]:
    tokens = [t for t in value.split(';') if t]
    return tokens if tokens and all(t.startswith('OPERA_') for t in tokens) else [value]


def issue_level(df: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    """Normalise an inputs CSV to one ``key, status, reason`` row per (opera_id, issue_type, input_id, ordinal)."""
    if df.empty:
        return _row_frame('inputs', df, ['opera_id'], 'issue_type')
    work = pd.DataFrame(
        {
            'opera_id': df.opera_id.astype(str),
            'status': row_status('inputs', df),
            'issue_type': df.issue_type.astype(str).replace(renames),
            'input_id': [_split_inputs(str(v)) for v in df.issue_value],
        }
    )
    work = work[work.issue_type != ''].explode('input_id')
    if work.empty:
        return _row_frame('inputs', pd.DataFrame(), ['opera_id'], 'issue_type')
    work['ordinal'] = work.groupby(['opera_id', 'issue_type', 'input_id']).cumcount().astype(str)
    key = _join_key(work, ['opera_id', 'issue_type', 'input_id', 'ordinal'])
    return pd.DataFrame({'key': key, 'status': work.status, 'reason': work.issue_type}).reset_index(drop=True)


def compare_dirs(
    old_dir: Path,
    old_stamp: str,
    new_dir: Path,
    new_stamp: str,
    deltas_dir: Path,
    renames: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compare every check between two output directories and write one delta CSV per check.

    Parameters
    ----------
    old_dir, new_dir : pathlib.Path
        Baseline and new output directories.
    old_stamp, new_stamp : str
        Run stamps (``YYYYmmddTHHMMSS``) of the CSVs to compare.
    deltas_dir : pathlib.Path
        Directory that receives ``<check>.csv`` for each check.
    renames : dict, optional
        Old -> new issue-type names applied to the old inputs CSV.

    Returns
    -------
    pandas.DataFrame
        Summary with one row per check: counts per change class, ``ambiguous``, ``reason_diff`` and source files.
    """
    renames = renames or {}
    deltas_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in ROW_CHECKS:
        old_df, old_fo, old_path = _load_source(old_dir, old_stamp, spec)
        new_df, new_fo, new_path = _load_source(new_dir, new_stamp, spec)
        key_cols = resolve_key_columns(old_df, new_df, spec.key)
        deltas = compare_frames(
            _row_frame(spec.name, old_df, key_cols, spec.reason),
            _row_frame(spec.name, new_df, key_cols, spec.reason),
            old_failures_only=old_fo,
            new_failures_only=new_fo,
        )
        deltas.to_csv(deltas_dir / f'{spec.name}.csv', index=False)
        summary.append(_summary_row(spec.name, deltas, old_path, new_path))

    old_path, new_path = find_csv(old_dir, INPUTS_SOURCE, old_stamp), find_csv(new_dir, INPUTS_SOURCE, new_stamp)
    old_df = read_csv(old_path) if old_path else pd.DataFrame()
    new_df = read_csv(new_path) if new_path else pd.DataFrame()
    for name, level in (('inputs_product', product_level), ('inputs_issue', issue_level)):
        deltas = compare_frames(level(old_df, renames), level(new_df, {}))
        deltas.to_csv(deltas_dir / f'{name}.csv', index=False)
        summary.append(_summary_row(name, deltas, old_path, new_path))
    return pd.DataFrame(summary)


def load_baselines(path: Path) -> dict:
    """Read ``baselines.json``."""
    return json.loads(path.read_text())


def data_path(baselines: dict, value: str) -> Path:
    """Resolve a path from ``baselines.json`` against its ``data_root``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(baselines['data_root']).expanduser() / path


def _pick_stamp(out_dir: Path, stamp: str | None) -> str:
    if stamp:
        return stamp
    stamps = find_stamps(out_dir)
    if not stamps:
        raise SystemExit(f'No stamped outputs in {out_dir}')
    if len(stamps) > 1:
        print(f'warning: {out_dir} holds stamps {stamps}; using the latest, {stamps[-1]}', file=sys.stderr)
    return stamps[-1]


def main(argv: list[str] | None = None) -> int:
    """Run the comparison from the command line; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('baseline', help='Baseline name in baselines.json, e.g. pop1_aoi.')
    parser.add_argument(
        'new_out_dir', type=Path, help='Run directory to compare, or an --out-dir (its `latest` run is used).'
    )
    parser.add_argument('--phase', default='adhoc', help='Name of the delta subdirectory (default: adhoc).')
    parser.add_argument('--new-stamp', default=None, help='Stamp of the new run (default: the latest in the dir).')
    parser.add_argument('--baselines', type=Path, default=DEFAULT_BASELINES)
    parser.add_argument(
        '--expect-unchanged', action='store_true', help='Exit 1 unless every row is unchanged and none is ambiguous.'
    )
    args = parser.parse_args(argv)

    baselines = load_baselines(args.baselines)
    if args.baseline not in baselines['baselines']:
        parser.error(f'unknown baseline {args.baseline!r}; choose from {sorted(baselines["baselines"])}')
    entry = baselines['baselines'][args.baseline]
    old_dir = data_path(baselines, entry['outputs'])
    new_dir = args.new_out_dir.expanduser().resolve()
    if (new_dir / 'latest').exists():  # an --out-dir holding per-run directories (F34): use the newest run
        new_dir = new_dir / (new_dir / 'latest').read_text().strip()
    new_stamp = _pick_stamp(new_dir, args.new_stamp)
    deltas_dir = data_path(baselines, baselines['deltas_root']) / args.baseline / args.phase

    summary = compare_dirs(
        old_dir, entry['stamp'], new_dir, new_stamp, deltas_dir, baselines.get('issue_type_renames', {})
    )
    print(f'baseline {args.baseline}: {old_dir} ({entry["stamp"]}) vs {new_dir} ({new_stamp})')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(summary.to_string(index=False))
    print(f'deltas written to {deltas_dir}')
    changed = summary[CHANGE_CLASSES[1:]].to_numpy().sum() + summary.ambiguous.sum()
    return 1 if args.expect_unchanged and changed else 0


if __name__ == '__main__':
    sys.exit(main())
