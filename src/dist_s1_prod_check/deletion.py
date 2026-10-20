"""Deletion-candidate lists from a finished run: which DIST-S1 granules to delete and regenerate, and why.

Reads the per-check CSVs of one complete, unsampled run directory and writes reports only; nothing is deleted.

DIST-S1 is a time series per MGRS tile: each product names the previous product of its tile as its prior, and the
confirmation state carries forward. A granule that failed a check is a **root**; every later product of its tile
(by acquisition time) inherits the error and is listed as **downstream** of the earliest root in the tile. A
coverage gap (an RTC-S1 pass with no product) is a root without a granule: the products after it chained over it.

Each root class has a level:

- ``candidate``: a validated failure; it and its downstream are deletion candidates;
- ``under_review``: a failure class not yet validated (pre-image inputs); listed with the downstream it would add,
  separately from the candidates, until it is triaged;
- ``unverified``: could not be checked (UNKNOWN); listed on its own with no downstream, unless
  `unknown_as_failure`, which makes it a candidate root like any failure.

Superseded duplicates are candidate roots without downstream: a successor that used a superseded version as its
prior is itself a confirmation failure (outdated prior), and one whose prior was reprocessed only later is a
stale-chain root.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd

from dist_s1_prod_check import manifest, status
from dist_s1_prod_check.status import Status


DELETION_DIR = 'deletion_lists'
DELETION_DIR_UNKNOWN_AS_FAILURE = 'deletion_lists_unknown_as_failure'
DELETION_MANIFEST_FILE = 'deletion_lists_manifest.json'
REQUIRED_CHECKS = ['duplicates', 'ordering', 'confirmation', 'inputs', 'coverage']

CANDIDATE, UNDER_REVIEW, UNVERIFIED = 'candidate', 'under_review', 'unverified'
TIERS = [CANDIDATE, UNDER_REVIEW, UNVERIFIED]
ROOT, DOWNSTREAM = 'root', 'downstream'

SUPERSEDED_DUPLICATE = 'superseded_duplicate'
CONFIRMATION_FAILURE = 'confirmation_failure'
STALE_CHAIN = 'stale_chain'
PROCESSING_ORDER = 'processing_order'
FIRST_PRODUCTION_ORDER = 'first_production_order'
INPUTS_POST = 'inputs_post'
INPUTS_PRE = 'inputs_pre'
INPUTS_BURST_MISMATCH = 'inputs_burst_mismatch'
INPUTS_NO_EXPECTED_PRODUCT = 'inputs_no_expected_product'
INPUTS_OTHER = 'inputs_other'
COVERAGE_GAP = 'coverage_gap'
COVERAGE_ORPHAN = 'coverage_orphan'
UNKNOWN_CONFIRMATION = 'unknown_confirmation'
UNKNOWN_INPUTS = 'unknown_inputs'
UNKNOWN_COVERAGE = 'unknown_coverage'
UNKNOWN_CLASSES = [UNKNOWN_CONFIRMATION, UNKNOWN_INPUTS, UNKNOWN_COVERAGE]

# Report order of the root classes.
CLASSES = [
    SUPERSEDED_DUPLICATE,
    CONFIRMATION_FAILURE,
    STALE_CHAIN,
    PROCESSING_ORDER,
    FIRST_PRODUCTION_ORDER,
    INPUTS_POST,
    INPUTS_BURST_MISMATCH,
    INPUTS_NO_EXPECTED_PRODUCT,
    INPUTS_OTHER,
    COVERAGE_GAP,
    COVERAGE_ORPHAN,
    INPUTS_PRE,
    *UNKNOWN_CLASSES,
]
INPUT_ISSUE_CLASSES = {
    'Post RTC IDs expected but not found': INPUTS_POST,
    'Post RTC IDs found but not expected': INPUTS_POST,
    'Pre RTC IDs expected but not found': INPUTS_PRE,
    'Pre RTC IDs found but not expected': INPUTS_PRE,
    'Burst ID mismatch': INPUTS_BURST_MISMATCH,
    'No expected product found': INPUTS_NO_EXPECTED_PRODUCT,
}

ROOT_COLUMNS = ['unit_id', 'opera_id', 'mgrs_tile_id', 'acq_time', 'root_class', 'level', 'cascades', 'detail']
GRANULE_COLUMNS = [
    'opera_id',
    'mgrs_tile_id',
    'acq_time',
    'processing_time',
    'tier',
    'role',
    'own_failures',
    'cause_id',
    'cause_class',
    'cause_acq_time',
]
PASS_PREFIX = 'pass:'


@dataclass
class RunTables:
    """The check results of one run that the deletion lists are built from."""

    products: pd.DataFrame  # every checked (deduplicated) product: the confirmation CSV's universe
    duplicates: pd.DataFrame
    confirmation: pd.DataFrame
    ordering: pd.DataFrame
    first_production: pd.DataFrame
    inputs: pd.DataFrame
    coverage: pd.DataFrame
    orphans: pd.DataFrame


@dataclass
class DeletionLists:
    """Every output table (see `write_deletion_lists` for the files)."""

    roots: pd.DataFrame
    granules: pd.DataFrame
    tiles: pd.DataFrame
    generate: pd.DataFrame
    summary: pd.DataFrame


def pass_unit_id(mgrs_tile_id: str, post_acq_time: pd.Timestamp) -> str:
    """Return the id of a root that is an RTC-S1 pass, not a granule (coverage gaps and unmatched UNKNOWN passes)."""
    return f'{PASS_PREFIX}{mgrs_tile_id}:{post_acq_time:%Y%m%dT%H%M%SZ}'


def _naive_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(None)


def _roots(rows: pd.DataFrame, root_class: str, level: str, cascades: bool, detail: pd.Series) -> pd.DataFrame:
    """Root rows for granules (`rows` has `opera_id`, `mgrs_tile_id`, `acq_time`)."""
    return pd.DataFrame(
        {
            'unit_id': rows.opera_id.to_numpy(),
            'opera_id': rows.opera_id.to_numpy(),
            'mgrs_tile_id': rows.mgrs_tile_id.astype(str).to_numpy(),
            'acq_time': rows.acq_time.to_numpy(),
            'root_class': root_class,
            'level': level,
            'cascades': cascades,
            'detail': detail.fillna('').astype(str).to_numpy(),
        },
        columns=ROOT_COLUMNS,
    )


def _inputs_row_status(df: pd.DataFrame) -> pd.Series:
    """Status of each inputs result row: its `status`, else derived from the legacy `inputs_correct`."""
    derived = df.inputs_correct.map({True: Status.PASS, False: Status.FAIL}).fillna(Status.UNKNOWN)
    if 'status' in df.columns:
        derived = df.status.where(df.status.notna() & (df.status != ''), derived)
    return derived.astype(str)


def collect_roots(tables: RunTables, unknown_as_failure: bool = False) -> pd.DataFrame:
    """One row per (root, class): every failed or unverified granule or pass of the run, with its level.

    A granule failing several checks has several rows. Inputs rows are reduced to one root per product and class.
    """
    unknown_level = CANDIDATE if unknown_as_failure else UNVERIFIED
    acq = dict(zip(tables.products.opera_id, tables.products.acq_time))
    kept = set(tables.products.opera_id)
    parts = []

    dup = tables.duplicates
    superseded = dup[~dup.is_latest.astype(bool)]
    parts.append(_roots(superseded, SUPERSEDED_DUPLICATE, CANDIDATE, False, superseded.reason))

    conf = tables.confirmation
    fail = conf[conf.status == Status.FAIL]
    parts.append(_roots(fail, CONFIRMATION_FAILURE, CANDIDATE, True, fail.reason))
    stale = conf[(conf.status == Status.INFO) & conf.reason.fillna('').str.startswith('stale chain')]
    parts.append(_roots(stale, STALE_CHAIN, CANDIDATE, True, stale.reason))
    unknown = conf[conf.status == Status.UNKNOWN]
    parts.append(_roots(unknown, UNKNOWN_CONFIRMATION, unknown_level, unknown_as_failure, unknown.reason))

    order = tables.ordering
    late = order[order.status == Status.FAIL]
    parts.append(_roots(late, PROCESSING_ORDER, CANDIDATE, True, late.reason))
    # A first-production inversion of a version that has since been superseded is covered by the duplicates list.
    first = tables.first_production
    first = first[(first.status == Status.FAIL) & first.opera_id.isin(kept)]
    parts.append(_roots(first, FIRST_PRODUCTION_ORDER, CANDIDATE, True, first.reason))

    inp = tables.inputs.assign(row_status=_inputs_row_status(tables.inputs))
    inp = inp[inp.row_status.isin([Status.FAIL, Status.UNKNOWN])].copy()
    inp['root_class'] = [
        (UNKNOWN_INPUTS if s == Status.UNKNOWN else INPUT_ISSUE_CLASSES.get(t, INPUTS_OTHER))
        for s, t in zip(inp.row_status, inp.issue_type.fillna(''))
    ]
    inp['acq_time'] = inp.opera_id.map(acq)
    for root_class, group in inp.groupby('root_class', sort=False):
        detail = group.groupby('opera_id', sort=False).issue_type.agg(lambda v: '; '.join(sorted(set(v.fillna('')))))
        rows = group.drop_duplicates('opera_id').set_index('opera_id').loc[detail.index].reset_index()
        level = {UNKNOWN_INPUTS: unknown_level, INPUTS_PRE: UNDER_REVIEW}.get(root_class, CANDIDATE)
        cascades = unknown_as_failure if root_class == UNKNOWN_INPUTS else True
        parts.append(_roots(rows, root_class, level, cascades, detail.reset_index(drop=True)))

    cov = tables.coverage
    gaps = cov[cov.status == Status.FAIL]
    parts.append(_pass_roots(gaps, COVERAGE_GAP, CANDIDATE, True))
    cov_unknown = cov[cov.status == Status.UNKNOWN]
    matched = cov_unknown.matched_opera_id.fillna('').astype(str).isin(kept)
    with_product = cov_unknown[matched].assign(opera_id=lambda d: d.matched_opera_id.astype(str))
    with_product = with_product.assign(acq_time=with_product.opera_id.map(acq))
    parts.append(_roots(with_product, UNKNOWN_COVERAGE, unknown_level, unknown_as_failure, with_product.reason))
    parts.append(_pass_roots(cov_unknown[~matched], UNKNOWN_COVERAGE, unknown_level, unknown_as_failure))
    orphans = tables.orphans
    parts.append(_roots(orphans, COVERAGE_ORPHAN, CANDIDATE, True, orphans.reason))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=ROOT_COLUMNS)
    roots = pd.concat(parts, ignore_index=True)
    roots['class_order'] = roots.root_class.map(CLASSES.index)
    roots = roots.sort_values(['class_order', 'mgrs_tile_id', 'acq_time', 'unit_id'], kind='stable')
    return roots.drop(columns='class_order').reset_index(drop=True)


def _pass_roots(passes: pd.DataFrame, root_class: str, level: str, cascades: bool) -> pd.DataFrame:
    """Root rows for RTC-S1 passes (`passes` from the coverage CSV), keyed by `pass_unit_id`."""
    if passes.empty:
        return pd.DataFrame(columns=ROOT_COLUMNS)
    times = _naive_utc(passes.post_acq_dt)
    tiles = passes.mgrs_tile_id.astype(str)
    return pd.DataFrame(
        {
            'unit_id': [pass_unit_id(t, at) for t, at in zip(tiles, times)],
            'opera_id': '',
            'mgrs_tile_id': tiles.to_numpy(),
            'acq_time': times.to_numpy(),
            'root_class': root_class,
            'level': level,
            'cascades': cascades,
            'detail': passes.reason.fillna('').astype(str).to_numpy(),
        },
        columns=ROOT_COLUMNS,
    )


def _first_cascading_root(roots: pd.DataFrame) -> pd.DataFrame:
    """Per tile, the earliest cascading root (ties: class order, then unit id): its id, class and time."""
    cascading = roots[roots.cascades.astype(bool)].copy()
    cascading['class_order'] = cascading.root_class.map(CLASSES.index)
    cascading = cascading.sort_values(['mgrs_tile_id', 'acq_time', 'class_order', 'unit_id'], kind='stable')
    first = cascading.drop_duplicates('mgrs_tile_id', keep='first')
    return first.rename(columns={'unit_id': 'cause_id', 'root_class': 'cause_class', 'acq_time': 'cause_acq_time'})[
        ['mgrs_tile_id', 'cause_id', 'cause_class', 'cause_acq_time']
    ]


def _downstream(products: pd.DataFrame, roots: pd.DataFrame) -> pd.DataFrame:
    """Products acquired strictly after the earliest cascading root of their tile, with that root as the cause."""
    first = _first_cascading_root(roots)
    df = products.merge(first, on='mgrs_tile_id', how='inner')
    return df[df.acq_time > df.cause_acq_time]


def build_deletion_lists(tables: RunTables, unknown_as_failure: bool = False) -> DeletionLists:
    """Build every deletion list from one run's check results (see the module docstring for the rules)."""
    roots = collect_roots(tables, unknown_as_failure)
    products = tables.products[['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time']]
    superseded = tables.duplicates.loc[
        ~tables.duplicates.is_latest.astype(bool), ['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time']
    ]
    all_granules = pd.concat([products, superseded], ignore_index=True).drop_duplicates('opera_id')
    granule_roots = roots[roots.opera_id != '']
    own = granule_roots.groupby('opera_id').root_class.agg(lambda v: ';'.join(dict.fromkeys(v)))

    levels_by_tier = {CANDIDATE: [CANDIDATE], UNDER_REVIEW: [CANDIDATE, UNDER_REVIEW], UNVERIFIED: TIERS}
    assigned: set[str] = set()
    parts = []
    for tier in TIERS:
        tier_roots = roots[roots.level.isin(levels_by_tier[tier])]
        new_roots = granule_roots[(granule_roots.level == tier) & ~granule_roots.opera_id.isin(assigned)]
        root_rows = all_granules[all_granules.opera_id.isin(set(new_roots.opera_id))].assign(
            tier=tier, role=ROOT, cause_id=lambda d: d.opera_id
        )
        root_rows['cause_class'] = root_rows.opera_id.map(
            new_roots[new_roots.level == tier].drop_duplicates('opera_id').set_index('opera_id').root_class
        )
        root_rows['cause_acq_time'] = root_rows.acq_time
        assigned |= set(root_rows.opera_id)
        down = _downstream(products, tier_roots)
        down = down[~down.opera_id.isin(assigned)].assign(tier=tier, role=DOWNSTREAM)
        assigned |= set(down.opera_id)
        parts += [root_rows, down]

    granules = pd.concat([p for p in parts if not p.empty] or [pd.DataFrame(columns=GRANULE_COLUMNS)])
    granules['own_failures'] = granules.opera_id.map(own).fillna('')
    granules = granules[GRANULE_COLUMNS].copy()
    granules['tier_order'] = granules.tier.map(TIERS.index)
    granules = granules.sort_values(['tier_order', 'mgrs_tile_id', 'acq_time', 'opera_id'], kind='stable')
    granules = granules.drop(columns='tier_order').reset_index(drop=True)

    gaps = tables.coverage[tables.coverage.status == Status.FAIL]
    generate = pd.DataFrame(
        {
            'mgrs_tile_id': gaps.mgrs_tile_id.astype(str).to_numpy(),
            'post_acq_time': _naive_utc(gaps.post_acq_dt).to_numpy(),
            'reason': gaps.reason.to_numpy(),
        }
    ).sort_values(['mgrs_tile_id', 'post_acq_time'], kind='stable', ignore_index=True)
    return DeletionLists(
        roots=roots,
        granules=granules,
        tiles=_tiles_table(granules, roots, products),
        generate=generate,
        summary=_summary(granules, roots),
    )


def _tiles_table(granules: pd.DataFrame, roots: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One row per tile and tier: products in the tile, roots and downstream listed, and the first cascading root."""
    columns = ['mgrs_tile_id', 'tier', 'n_products_in_tile', 'n_roots', 'n_downstream', 'cause_id', 'cause_class']
    if granules.empty:
        return pd.DataFrame(columns=[*columns, 'cause_acq_time'])
    counts = granules.groupby(['mgrs_tile_id', 'tier', 'role']).size().unstack('role', fill_value=0)
    counts = counts.reindex(columns=[ROOT, DOWNSTREAM], fill_value=0).reset_index()
    counts = counts.rename(columns={ROOT: 'n_roots', DOWNSTREAM: 'n_downstream'})
    counts['n_products_in_tile'] = counts.mgrs_tile_id.map(products.groupby('mgrs_tile_id').size()).fillna(0)
    counts['n_products_in_tile'] = counts.n_products_in_tile.astype(int)
    firsts = [_first_cascading_root(roots[roots.level.isin(TIERS[: TIERS.index(t) + 1])]).assign(tier=t) for t in TIERS]
    df = counts.merge(pd.concat(firsts, ignore_index=True), on=['mgrs_tile_id', 'tier'], how='left')
    df['tier_order'] = df.tier.map(TIERS.index)
    df = df.sort_values(['tier_order', 'mgrs_tile_id'], kind='stable').drop(columns='tier_order')
    return df[[*columns, 'cause_acq_time']].reset_index(drop=True)


def _summary(granules: pd.DataFrame, roots: pd.DataFrame) -> pd.DataFrame:
    """Count roots, passes and the downstream they cause, per tier and root class, plus per-tier totals."""
    rows = []
    for tier in TIERS:
        g = granules[granules.tier == tier]
        tier_roots = roots[roots.level == tier]
        for root_class in CLASSES:
            r = tier_roots[tier_roots.root_class == root_class]
            if r.empty:
                continue
            down = g[(g.role == DOWNSTREAM) & (g.cause_class == root_class)]
            rows.append(
                {
                    'tier': tier,
                    'root_class': root_class,
                    'n_root_granules': int(r[r.opera_id != ''].opera_id.nunique()),
                    'n_root_passes': int((r.opera_id == '').sum()),
                    'n_downstream_caused': len(down),
                    'n_tiles': int(r.mgrs_tile_id.nunique()),
                }
            )
        rows.append(
            {
                'tier': tier,
                'root_class': 'TOTAL (unique granules)',
                'n_root_granules': int((g.role == ROOT).sum()),
                'n_root_passes': int((tier_roots.opera_id == '').sum()),
                'n_downstream_caused': int((g.role == DOWNSTREAM).sum()),
                'n_tiles': int(g.mgrs_tile_id.nunique()),
            }
        )
    return pd.DataFrame(rows)


SAME, CHANGED, ONLY_THIS, ONLY_OTHER = 'same', 'changed', 'only_this_run', 'only_other_run'
COMPARE_KEY = ['tier', 'role', 'cause_class']


@dataclass
class Comparison:
    """How this run's lists differ from another run's (see `compare_lists`)."""

    granules: pd.DataFrame  # one row per granule listed in either run whose listing differs
    roots: pd.DataFrame  # one row per (root, class) present in only one run
    summary: pd.DataFrame
    status: pd.Series  # `in_other_run` for every granule of this run, indexed by opera_id


def compare_lists(this: DeletionLists, other: DeletionLists) -> Comparison:
    """Compare two runs' lists granule by granule and root by root.

    A granule listed in both runs is ``same`` if its tier, role and cause class agree, else ``changed``; the
    others are ``only_this_run`` or ``only_other_run``. The summary counts granules per (tier, role, cause class)
    of the run that lists them: in this run, in the other, the same in both, changed, and in one run only.
    """
    cols = ['opera_id', *COMPARE_KEY]
    df = this.granules[cols].merge(other.granules[cols], on='opera_id', how='outer', suffixes=('', '_other'))
    in_this, in_other = df.tier.notna(), df.tier_other.notna()
    agree = (df[COMPARE_KEY].to_numpy() == df[[f'{c}_other' for c in COMPARE_KEY]].to_numpy()).all(axis=1)
    df['in_other_run'] = ''
    df.loc[in_this & in_other & agree, 'in_other_run'] = SAME
    df.loc[in_this & in_other & ~agree, 'in_other_run'] = CHANGED
    df.loc[in_this & ~in_other, 'in_other_run'] = ONLY_THIS
    df.loc[~in_this & in_other, 'in_other_run'] = ONLY_OTHER
    status = df[in_this].set_index('opera_id').in_other_run

    rows = []
    keys = sorted(
        set(map(tuple, this.granules[COMPARE_KEY].to_numpy()))
        | set(map(tuple, other.granules[COMPARE_KEY].to_numpy())),
        key=lambda k: (TIERS.index(k[0]), k[1] != ROOT, CLASSES.index(k[2]) if k[2] in CLASSES else len(CLASSES)),
    )
    a_key = df[COMPARE_KEY].apply(tuple, axis=1)
    b_key = df[[f'{c}_other' for c in COMPARE_KEY]].apply(tuple, axis=1)
    for key in keys:
        a, b = a_key == key, b_key == key
        rows.append(
            {
                **dict(zip(COMPARE_KEY, key)),
                'n_this_run': int(a.sum()),
                'n_other_run': int(b.sum()),
                'n_same': int((a & (df.in_other_run == SAME)).sum()),
                'n_changed': int((a & (df.in_other_run == CHANGED)).sum()),
                'n_only_this_run': int((a & (df.in_other_run == ONLY_THIS)).sum()),
                'n_only_other_run': int((b & (df.in_other_run == ONLY_OTHER)).sum()),
            }
        )
    summary = pd.DataFrame(
        rows,
        columns=[
            *COMPARE_KEY,
            'n_this_run',
            'n_other_run',
            'n_same',
            'n_changed',
            'n_only_this_run',
            'n_only_other_run',
        ],
    )

    root_cols = ['unit_id', 'root_class', 'mgrs_tile_id', 'acq_time']
    r = this.roots[root_cols].merge(
        other.roots[root_cols], on=['unit_id', 'root_class'], how='outer', suffixes=('', '_other'), indicator=True
    )
    r = r[r._merge != 'both'].copy()
    r['in_run'] = r._merge.map({'left_only': ONLY_THIS, 'right_only': ONLY_OTHER}).astype(str)
    r['mgrs_tile_id'] = r.mgrs_tile_id.fillna(r.mgrs_tile_id_other)
    r['acq_time'] = r.acq_time.fillna(r.acq_time_other)
    roots = r[['unit_id', 'root_class', 'mgrs_tile_id', 'acq_time', 'in_run']]
    roots = roots.sort_values(['root_class', 'mgrs_tile_id', 'acq_time', 'unit_id'], kind='stable', ignore_index=True)

    changed = df[df.in_other_run != SAME].sort_values(['in_other_run', 'opera_id'], kind='stable', ignore_index=True)
    return Comparison(granules=changed, roots=roots, summary=summary, status=status)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.stat().st_size <= 1:
        return pd.DataFrame(columns=columns)
    present = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in present]
    if missing:
        raise click.ClickException(f'{path} lacks columns {missing}; it was not written by this tool version.')
    return pd.read_csv(path, usecols=columns, dtype={c: str for c in columns if c not in ('is_latest',)})


def load_run(run_dir: Path) -> tuple[RunTables, dict]:
    """Load a run directory's check CSVs; refuse an incomplete, partial or sampled run.

    Returns the tables and the run manifest. Every required check must be done, because a missing root anywhere
    in a tile changes what follows it.
    """
    run_dir = Path(run_dir)
    status_path, manifest_path = run_dir / status.RUN_STATUS_FILE, run_dir / manifest.RUN_MANIFEST_FILE
    for path in (status_path, manifest_path):
        if not path.exists():
            raise click.ClickException(f'{run_dir} is not a run directory: no {path.name}.')
    run = status.read_run_status(status_path)
    missing_checks = [c for c in REQUIRED_CHECKS if c not in run.checks_done]
    if run.state != status.RunState.COMPLETE or missing_checks:
        raise click.ClickException(
            f'{run_dir} is not a complete run-all (state {run.raw_state}, checks not done: {missing_checks}). '
            'Deletion lists need every check over the whole window; rerun run-all first.'
        )
    run_manifest = json.loads(manifest_path.read_text())
    if run_manifest.get('sampling'):
        raise click.ClickException(
            f'{run_dir} was sampled ({sorted(run_manifest["sampling"])}); deletion lists need an unsampled run.'
        )
    stamp = run_dir.name.split('_')[0]
    outputs = set(run_manifest.get('outputs', []))

    def csv(name: str, columns: list[str]) -> pd.DataFrame:
        file = f'{name}_{stamp}.csv'
        if file not in outputs or not (run_dir / file).exists():
            raise click.ClickException(f'{run_dir} has no {file}; was it written by run-all?')
        return _read_csv(run_dir / file, columns)

    confirmation = csv('confirmation', ['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time', 'status', 'reason'])
    for col in ('acq_time', 'processing_time'):
        confirmation[col] = pd.to_datetime(confirmation[col])
    duplicates = csv('duplicates', ['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time', 'is_latest', 'reason'])
    duplicates['is_latest'] = duplicates.is_latest.map({True: True, False: False, 'True': True, 'False': False})
    for col in ('acq_time', 'processing_time'):
        duplicates[col] = pd.to_datetime(duplicates[col])
    order_cols = ['opera_id', 'mgrs_tile_id', 'acq_time', 'status', 'reason']
    ordering = csv('ordering', order_cols)
    first_production = csv('ordering_first_production', order_cols)
    for df in (ordering, first_production):
        df['acq_time'] = pd.to_datetime(df.acq_time)
    inputs_cols = ['opera_id', 'mgrs_tile_id', 'inputs_correct', 'issue_type', 'status']
    inp = csv('inputs', inputs_cols)
    inp['inputs_correct'] = inp.inputs_correct.map({'True': True, 'False': False})
    coverage = csv('coverage', ['mgrs_tile_id', 'post_acq_dt', 'status', 'reason', 'matched_opera_id'])
    orphans = csv('coverage_orphans', ['opera_id', 'mgrs_tile_id', 'acq_time', 'reason'])
    orphans['acq_time'] = pd.to_datetime(orphans.acq_time)
    tables = RunTables(
        products=confirmation,
        duplicates=duplicates,
        confirmation=confirmation,
        ordering=ordering,
        first_production=first_production,
        inputs=inp,
        coverage=coverage,
        orphans=orphans,
    )
    _require_consistent(tables, run_dir)
    return tables, run_manifest


def _require_consistent(tables: RunTables, run_dir: Path) -> None:
    """Refuse results that do not describe one product universe (one confirmation row per product, etc.)."""
    products = tables.products
    problems = []
    if products.opera_id.duplicated().any():
        problems.append('confirmation has more than one row for a product')
    if products[['acq_time', 'processing_time']].isna().any().any():
        problems.append('confirmation has products without acquisition or processing time')
    kept = set(products.opera_id)
    for name, df in (('ordering', tables.ordering), ('inputs', tables.inputs)):
        stray = set(df.opera_id) - kept
        if stray:
            problems.append(f'{name} has {len(stray)} products not in confirmation, e.g. {sorted(stray)[0]}')
    if set(tables.inputs.opera_id) != kept:
        problems.append('inputs does not cover every product')
    superseded = tables.duplicates[~tables.duplicates.is_latest.astype(bool)]
    if superseded.opera_id.isin(kept).any():
        problems.append('a superseded duplicate is also a checked product')
    if problems:
        raise click.ClickException(f'{run_dir}: inconsistent check results: ' + '; '.join(problems) + '.')


def _write_ids(path: Path, ids: pd.Series) -> None:
    path.write_text(''.join(f'{i}\n' for i in ids))


def write_deletion_lists(
    run_dir: Path,
    lists: DeletionLists,
    run_manifest: dict,
    unknown_as_failure: bool,
    comparison: Comparison | None = None,
    compare_run_dir: Path | None = None,
) -> Path:
    """Write the lists into `run_dir/deletion_lists[_unknown_as_failure]/`, replacing any earlier copy at once.

    Files: ``granules.csv`` (one row per listed granule: tier, role, own failures, cause), ``roots.csv`` (one row
    per root and class, passes included), ``tiles.csv``, ``generate.csv`` (missing passes to produce),
    ``summary.csv``, ``ids/`` (one opera_id per line: per tier, per tier and root class, and downstream per tier)
    and ``deletion_lists_manifest.json``. With a `comparison` against `compare_run_dir`, ``granules.csv`` gains
    ``in_other_run``, and ``comparison_summary.csv``, ``comparison_granules.csv`` (every granule listed differently)
    and ``comparison_roots.csv`` (roots found in one run only) are added.
    """
    run_dir = Path(run_dir)
    target = run_dir / (DELETION_DIR_UNKNOWN_AS_FAILURE if unknown_as_failure else DELETION_DIR)
    tmp = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=run_dir))
    try:
        granules = lists.granules
        if comparison is not None:
            granules = granules.assign(in_other_run=granules.opera_id.map(comparison.status))
            comparison.summary.to_csv(tmp / 'comparison_summary.csv', index=False)
            comparison.granules.to_csv(tmp / 'comparison_granules.csv', index=False)
            comparison.roots.to_csv(tmp / 'comparison_roots.csv', index=False)
        granules.to_csv(tmp / 'granules.csv', index=False)
        lists.roots.to_csv(tmp / 'roots.csv', index=False)
        lists.tiles.to_csv(tmp / 'tiles.csv', index=False)
        lists.generate.to_csv(tmp / 'generate.csv', index=False)
        lists.summary.to_csv(tmp / 'summary.csv', index=False)
        ids_dir = tmp / 'ids'
        ids_dir.mkdir()
        files = {}
        for tier in TIERS:
            g = lists.granules[lists.granules.tier == tier]
            if g.empty:
                continue
            files[f'{tier}.txt'] = g.opera_id
            files[f'{tier}_downstream.txt'] = g[g.role == DOWNSTREAM].opera_id
            tier_roots = lists.roots[(lists.roots.level == tier) & (lists.roots.opera_id != '')]
            for root_class, r in tier_roots.groupby('root_class', sort=False):
                files[f'{tier}_root_{root_class}.txt'] = r.opera_id.drop_duplicates()
        for name, ids in files.items():
            _write_ids(ids_dir / name, ids)
        meta = {
            'source_run_dir': str(run_dir.resolve()),
            'source_tool_version': run_manifest.get('tool_version'),
            'source_params': run_manifest.get('params', {}),
            'tool_version': manifest.tool_version(),
            'created_utc': manifest.now_utc(),
            'unknown_as_failure': unknown_as_failure,
            'compare_run_dir': str(Path(compare_run_dir).resolve()) if compare_run_dir is not None else None,
            'comparison': (
                {k: int(v) for k, v in comparison.granules.in_other_run.value_counts().sort_index().items()}
                | {SAME: int((comparison.status == SAME).sum())}
                if comparison is not None
                else None
            ),
            'counts': {
                tier: {
                    'granules': int((lists.granules.tier == tier).sum()),
                    'roots': int(((lists.granules.tier == tier) & (lists.granules.role == ROOT)).sum()),
                    'downstream': int(((lists.granules.tier == tier) & (lists.granules.role == DOWNSTREAM)).sum()),
                }
                for tier in TIERS
            },
            'id_files': {name: len(ids) for name, ids in files.items()},
        }
        (tmp / DELETION_MANIFEST_FILE).write_text(json.dumps(meta, indent=2, default=str) + '\n')
        if target.exists():
            shutil.rmtree(target)
        tmp.replace(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target
