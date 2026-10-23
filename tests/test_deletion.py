"""Deletion-candidate lists built from a run's check CSVs (`deletion.py`, ``deletion-lists`` command)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from click import ClickException
from click.testing import CliRunner

from dist_s1_prod_check import deletion, manifest, status
from dist_s1_prod_check.cli import cli
from dist_s1_prod_check.deletion import (
    CANDIDATE,
    DOWNSTREAM,
    ROOT,
    UNDER_REVIEW,
    UNVERIFIED,
    RunTables,
    build_deletion_lists,
    pass_unit_id,
)
from factories import dist_id, run_dir, seed_data_dir
from test_cli_integrity import TILE as WORLD_TILE
from test_cli_integrity import _block_network, _run_all, _world  # noqa: F401  (_block_network: autouse fixture)


TILE, OTHER = '10SGC', '11SKA'
T0 = pd.Timestamp('2026-01-02T02:06:53')
REVISIT = pd.Timedelta(days=6)
STAMP = '20261020T101402'


def _acq(k: int) -> pd.Timestamp:
    return T0 + k * REVISIT


def _id(k: int, tile: str = TILE, reprocessed: bool = False) -> str:
    return dist_id(tile, _acq(k), _acq(k) + pd.Timedelta(days=30 if reprocessed else 1))


def _products(n: int = 5, tiles: tuple[str, ...] = (TILE, OTHER)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'opera_id': _id(k, tile),
                'mgrs_tile_id': tile,
                'acq_time': _acq(k),
                'processing_time': _acq(k) + pd.Timedelta(days=1),
            }
            for tile in tiles
            for k in range(n)
        ]
    )


def _tables(products: pd.DataFrame | None = None, **overrides: pd.DataFrame) -> RunTables:
    """Every check PASS over `products`; `overrides` replace whole tables."""
    products = _products() if products is None else products
    confirmation = products.assign(status='PASS', reason='')
    order = products[['opera_id', 'mgrs_tile_id', 'acq_time']].assign(status='PASS', reason='')
    tables = {
        'products': products,
        'duplicates': pd.DataFrame(
            columns=['opera_id', 'mgrs_tile_id', 'acq_time', 'processing_time', 'is_latest', 'reason']
        ),
        'confirmation': confirmation,
        'ordering': order,
        'first_production': order.iloc[0:0],
        'inputs': products[['opera_id', 'mgrs_tile_id']].assign(inputs_correct=True, issue_type='', status='PASS'),
        'coverage': pd.DataFrame(columns=['mgrs_tile_id', 'post_acq_dt', 'status', 'reason', 'matched_opera_id']),
        'orphans': pd.DataFrame(columns=['opera_id', 'mgrs_tile_id', 'acq_time', 'reason']),
    }
    tables.update(overrides)
    if 'confirmation' in overrides and 'products' not in overrides:
        tables['products'] = products
    return RunTables(**tables)


def _with_status(df: pd.DataFrame, ids: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Set (status, reason) of the rows whose opera_id is a key of `ids`."""
    df = df.copy()
    for opera_id, (st, reason) in ids.items():
        df.loc[df.opera_id == opera_id, ['status', 'reason']] = [st, reason]
    return df


def _confirmation(ids: dict[str, tuple[str, str]], products: pd.DataFrame | None = None) -> pd.DataFrame:
    products = _products() if products is None else products
    return _with_status(products.assign(status='PASS', reason=''), ids)


def _inputs(rows: dict[str, list[tuple[str, str]]], products: pd.DataFrame | None = None) -> pd.DataFrame:
    """PASS rows for every product, replaced by (status, issue_type) rows for the products in `rows`."""
    products = _products() if products is None else products
    out = []
    for p in products.itertuples():
        for st, issue in rows.get(p.opera_id, [('PASS', '')]):
            correct = {'PASS': True, 'FAIL': False}.get(st)
            out.append(
                {
                    'opera_id': p.opera_id,
                    'mgrs_tile_id': p.mgrs_tile_id,
                    'inputs_correct': correct,
                    'issue_type': issue,
                    'status': st,
                }
            )
    return pd.DataFrame(out)


def _coverage(rows: list[dict]) -> pd.DataFrame:
    cols = ['mgrs_tile_id', 'post_acq_dt', 'status', 'reason', 'matched_opera_id']
    return pd.DataFrame([{'reason': '', 'matched_opera_id': '', **r} for r in rows], columns=cols)


def _listed(lists: deletion.DeletionLists, tier: str | None = None, role: str | None = None) -> list[str]:
    g = lists.granules
    if tier:
        g = g[g.tier == tier]
    if role:
        g = g[g.role == role]
    return list(g.opera_id)


# ---------------------------------------------------------------------------------------------------------------
# Cascade rules


def test_confirmation_failure_lists_itself_and_every_later_product_of_its_tile() -> None:
    lists = build_deletion_lists(_tables(confirmation=_confirmation({_id(2): ('FAIL', 'no prior')})))
    assert _listed(lists, CANDIDATE, ROOT) == [_id(2)]
    assert _listed(lists, CANDIDATE, DOWNSTREAM) == [_id(3), _id(4)]
    assert not any(o.startswith('OPERA') and OTHER in o for o in _listed(lists)), 'another tile was listed'
    down = lists.granules.set_index('opera_id').loc[_id(4)]
    assert (down.cause_id, down.cause_class, down.cause_acq_time) == (_id(2), 'confirmation_failure', _acq(2))
    assert down.own_failures == ''


def test_earliest_root_of_the_tile_is_the_cause_and_later_roots_stay_roots() -> None:
    lists = build_deletion_lists(_tables(confirmation=_confirmation({_id(1): ('FAIL', 'a'), _id(3): ('FAIL', 'b')})))
    g = lists.granules.set_index('opera_id')
    assert g.loc[_id(2), 'cause_id'] == _id(1)
    assert g.loc[_id(3), 'role'] == ROOT
    assert g.loc[_id(4), 'cause_id'] == _id(1)
    assert _id(0) not in g.index


def test_superseded_duplicate_is_a_root_without_downstream() -> None:
    superseded = _id(1, reprocessed=False)
    products = _products()
    products.loc[products.opera_id == _id(1), 'opera_id'] = latest = _id(1, reprocessed=True)
    dup = pd.DataFrame(
        [
            {'opera_id': superseded, 'is_latest': False, 'reason': 'superseded'},
            {'opera_id': latest, 'is_latest': True, 'reason': 'latest, kept'},
        ]
    ).assign(mgrs_tile_id=TILE, acq_time=_acq(1), processing_time=[_acq(1), _acq(1) + pd.Timedelta(days=30)])
    lists = build_deletion_lists(_tables(products, duplicates=dup, confirmation=_confirmation({}, products)))
    assert _listed(lists) == [superseded]
    assert lists.granules.iloc[0].role == ROOT
    assert lists.granules.iloc[0].cause_class == 'superseded_duplicate'


def test_stale_chain_info_cascades_but_other_info_does_not() -> None:
    conf = _confirmation(
        {_id(2): ('INFO', 'stale chain: X was reprocessed after the product'), _id(1, OTHER): ('INFO', 'note')}
    )
    lists = build_deletion_lists(_tables(confirmation=conf))
    assert _listed(lists, CANDIDATE, ROOT) == [_id(2)]
    assert _listed(lists, CANDIDATE, DOWNSTREAM) == [_id(3), _id(4)]


def test_coverage_gap_is_a_pass_root_whose_later_products_are_downstream() -> None:
    gap_time = _acq(1) + pd.Timedelta(days=3)
    cov = _coverage([{'mgrs_tile_id': TILE, 'post_acq_dt': f'{gap_time}+00:00', 'status': 'FAIL', 'reason': 'gap'}])
    lists = build_deletion_lists(_tables(coverage=cov))
    assert _listed(lists, CANDIDATE, ROOT) == []
    assert _listed(lists, CANDIDATE, DOWNSTREAM) == [_id(2), _id(3), _id(4)]
    assert set(lists.granules.cause_id) == {pass_unit_id(TILE, gap_time)}
    assert list(lists.generate.post_acq_time) == [gap_time]
    roots = lists.roots[lists.roots.root_class == 'coverage_gap']
    assert list(roots.opera_id) == [''] and list(roots.unit_id) == [pass_unit_id(TILE, gap_time)]


def test_coverage_gap_after_the_last_product_lists_nothing_but_marks_the_tile() -> None:
    late = _acq(4) + pd.Timedelta(days=3)
    cov = _coverage([{'mgrs_tile_id': TILE, 'post_acq_dt': f'{late}+00:00', 'status': 'FAIL'}])
    lists = build_deletion_lists(_tables(coverage=cov))
    assert lists.granules.empty
    summary = lists.summary.set_index(['tier', 'root_class'])
    assert summary.loc[(CANDIDATE, 'coverage_gap'), 'n_root_passes'] == 1


def test_orphan_product_cascades() -> None:
    orphans = pd.DataFrame([{'opera_id': _id(3), 'mgrs_tile_id': TILE, 'acq_time': _acq(3), 'reason': 'no pass'}])
    lists = build_deletion_lists(_tables(orphans=orphans))
    assert _listed(lists, CANDIDATE) == [_id(3), _id(4)]


def test_inputs_pre_is_under_review_and_never_reaches_the_candidates() -> None:
    inp = _inputs({_id(1): [('FAIL', 'Pre RTC IDs found but not expected')]})
    lists = build_deletion_lists(_tables(inputs=inp))
    assert _listed(lists, CANDIDATE) == []
    assert _listed(lists, UNDER_REVIEW, ROOT) == [_id(1)]
    assert _listed(lists, UNDER_REVIEW, DOWNSTREAM) == [_id(2), _id(3), _id(4)]


def test_candidate_tier_wins_over_under_review() -> None:
    inp = _inputs({_id(1): [('FAIL', 'Pre RTC IDs expected but not found')]})
    conf = _confirmation({_id(3): ('FAIL', 'no prior')})
    lists = build_deletion_lists(_tables(inputs=inp, confirmation=conf))
    assert _listed(lists, CANDIDATE) == [_id(3), _id(4)]
    assert _listed(lists, UNDER_REVIEW) == [_id(1), _id(2)]
    g = lists.granules.set_index('opera_id')
    assert g.loc[_id(2), 'cause_id'] == _id(1)


def test_under_review_root_downstream_of_a_candidate_keeps_its_own_failure() -> None:
    inp = _inputs({_id(3): [('FAIL', 'Pre RTC IDs found but not expected')]})
    conf = _confirmation({_id(1): ('FAIL', 'no prior')})
    g = build_deletion_lists(_tables(inputs=inp, confirmation=conf)).granules.set_index('opera_id')
    assert (g.loc[_id(3), 'tier'], g.loc[_id(3), 'role']) == (CANDIDATE, DOWNSTREAM)
    assert g.loc[_id(3), 'own_failures'] == 'inputs_pre_extra'


@pytest.mark.parametrize(
    ('issue', 'root_class'),
    [
        ('Post RTC IDs expected but not found', 'inputs_post'),
        ('Post RTC IDs found but not expected', 'inputs_post'),
        ('Burst ID mismatch', 'inputs_burst_mismatch'),
        ('No expected product found', 'inputs_no_expected_product'),
        ('Some future issue type', 'inputs_other'),
    ],
)
def test_input_failures_are_candidate_roots_by_issue(issue: str, root_class: str) -> None:
    lists = build_deletion_lists(_tables(inputs=_inputs({_id(2): [('FAIL', issue)]})))
    assert _listed(lists, CANDIDATE) == [_id(2), _id(3), _id(4)]
    assert lists.granules.iloc[0].cause_class == root_class


def test_product_with_several_input_issues_is_one_granule_with_every_class() -> None:
    inp = _inputs({_id(2): [('FAIL', 'Burst ID mismatch'), ('FAIL', 'Post RTC IDs expected but not found')]})
    lists = build_deletion_lists(_tables(inputs=inp))
    assert lists.granules.opera_id.is_unique
    assert lists.granules.set_index('opera_id').loc[_id(2), 'own_failures'] == 'inputs_post;inputs_burst_mismatch'
    assert sorted(lists.roots.root_class) == ['inputs_burst_mismatch', 'inputs_post']


def test_inputs_status_falls_back_to_inputs_correct() -> None:
    inp = _inputs({_id(2): [('FAIL', 'Post RTC IDs expected but not found')]})
    inp['status'] = ''
    assert _listed(build_deletion_lists(_tables(inputs=inp)), CANDIDATE, ROOT) == [_id(2)]


def test_first_production_inversion_counts_only_for_kept_products() -> None:
    """An inversion of a version since superseded is the duplicates list's business, not a cascading root."""
    superseded, latest = _id(1), _id(1, reprocessed=True)
    products = _products()
    products.loc[products.opera_id == superseded, 'opera_id'] = latest
    dup = pd.DataFrame(
        [
            {'opera_id': superseded, 'is_latest': False, 'reason': 's'},
            {'opera_id': latest, 'is_latest': True, 'reason': 'k'},
        ]
    ).assign(mgrs_tile_id=TILE, acq_time=_acq(1), processing_time=_acq(1))
    first = pd.DataFrame(
        [
            {'opera_id': superseded, 'status': 'FAIL', 'reason': 'early'},
            {'opera_id': _id(3), 'status': 'FAIL', 'reason': 'early'},
        ]
    ).assign(mgrs_tile_id=TILE, acq_time=[_acq(1), _acq(3)])
    lists = build_deletion_lists(
        _tables(products, duplicates=dup, confirmation=_confirmation({}, products), first_production=first)
    )
    assert list(lists.roots[lists.roots.root_class == 'first_production_order'].opera_id) == [_id(3)]
    assert _listed(lists, CANDIDATE) == [superseded, _id(3), _id(4)]
    assert lists.granules.set_index('opera_id').loc[superseded, 'own_failures'] == 'superseded_duplicate'


def test_downstream_is_strictly_after_the_root() -> None:
    """A pass root at exactly a product's acquisition time does not make that product downstream."""
    cov = _coverage([{'mgrs_tile_id': TILE, 'post_acq_dt': f'{_acq(2)}+00:00', 'status': 'FAIL'}])
    assert _listed(build_deletion_lists(_tables(coverage=cov)), CANDIDATE) == [_id(3), _id(4)]


def test_processing_order_failure_cascades() -> None:
    order = _with_status(
        _products()[['opera_id', 'mgrs_tile_id', 'acq_time']].assign(status='PASS', reason=''),
        {_id(3): ('FAIL', 'late')},
    )
    assert _listed(build_deletion_lists(_tables(ordering=order)), CANDIDATE) == [_id(3), _id(4)]


def test_roots_at_the_same_time_pick_the_cause_by_class_order() -> None:
    conf = _confirmation({_id(2): ('FAIL', 'no prior')})
    gap = _coverage([{'mgrs_tile_id': TILE, 'post_acq_dt': f'{_acq(2)}+00:00', 'status': 'FAIL'}])
    g = build_deletion_lists(_tables(confirmation=conf, coverage=gap)).granules.set_index('opera_id')
    assert g.loc[_id(3), 'cause_class'] == 'confirmation_failure'


# ---------------------------------------------------------------------------------------------------------------
# UNKNOWN results


def _unknown_tables() -> RunTables:
    conf = _confirmation({_id(1): ('UNKNOWN', 'could not read tag')})
    inp = _inputs({_id(2, OTHER): [('UNKNOWN', 'Tag read error')]})
    cov = _coverage(
        [
            {'mgrs_tile_id': OTHER, 'post_acq_dt': f'{_acq(3)}+00:00', 'status': 'UNKNOWN', 'reason': 'ambiguous'},
            {
                'mgrs_tile_id': TILE,
                'post_acq_dt': f'{_acq(3)}+00:00',
                'status': 'UNKNOWN',
                'reason': 'latest product unreadable',
                'matched_opera_id': _id(3),
            },
        ]
    )
    return _tables(confirmation=conf, inputs=inp, coverage=cov)


def test_unknown_is_unverified_without_downstream_by_default() -> None:
    lists = build_deletion_lists(_unknown_tables())
    assert sorted(_listed(lists, UNVERIFIED, ROOT)) == sorted([_id(1), _id(3), _id(2, OTHER)])
    assert _listed(lists, UNVERIFIED, DOWNSTREAM) == [] and _listed(lists, CANDIDATE) == []
    assert set(lists.roots.root_class) == {'unknown_confirmation', 'unknown_inputs', 'unknown_coverage'}
    assert (lists.roots.unit_id == pass_unit_id(OTHER, _acq(3))).sum() == 1


def test_unknown_as_failure_makes_unknowns_cascading_candidates() -> None:
    lists = build_deletion_lists(_unknown_tables(), unknown_as_failure=True)
    assert _listed(lists, UNVERIFIED) == []
    assert _listed(lists, CANDIDATE) == [_id(1), _id(2), _id(3), _id(4), _id(2, OTHER), _id(3, OTHER), _id(4, OTHER)]
    g = lists.granules.set_index('opera_id')
    assert g.loc[_id(3, OTHER), 'cause_id'] == _id(2, OTHER)


# ---------------------------------------------------------------------------------------------------------------
# Summary and tiles


def test_summary_and_tiles_count_what_is_listed() -> None:
    conf = _confirmation({_id(2): ('FAIL', 'no prior'), _id(1, OTHER): ('FAIL', 'no prior')})
    inp = _inputs({_id(0): [('FAIL', 'Pre RTC IDs found but not expected')]})
    lists = build_deletion_lists(_tables(confirmation=conf, inputs=inp))
    s = lists.summary.set_index(['tier', 'root_class'])
    # n_failing, n_root_granules, n_root_passes, n_downstream_caused, n_tiles
    assert s.loc[(CANDIDATE, 'confirmation_failure')].tolist() == [2, 2, 0, 5, 2]
    assert s.loc[(CANDIDATE, 'TOTAL (unique granules)')].tolist() == [2, 2, 0, 5, 2]
    assert s.loc[(UNDER_REVIEW, 'TOTAL (unique granules)')].tolist() == [1, 1, 0, 1, 1]
    tiles = lists.tiles.set_index(['mgrs_tile_id', 'tier'])
    assert tiles.loc[(TILE, CANDIDATE), ['n_products_in_tile', 'n_roots', 'n_downstream']].tolist() == [5, 1, 2]
    assert tiles.loc[(TILE, CANDIDATE), 'cause_id'] == _id(2)
    assert tiles.loc[(TILE, UNDER_REVIEW), 'cause_id'] == _id(0)


def test_summary_separates_failing_from_listed_roots() -> None:
    """A pre-input failure already downstream of a candidate root fails its class but adds no under-review root."""
    inp = _inputs({_id(3): [('FAIL', 'Pre RTC IDs expected but not found')]})
    conf = _confirmation({_id(1): ('FAIL', 'no prior')})
    s = build_deletion_lists(_tables(inputs=inp, confirmation=conf)).summary.set_index(['tier', 'root_class'])
    assert s.loc[(UNDER_REVIEW, 'inputs_pre_missing')].tolist() == [1, 0, 0, 0, 1]
    assert s.loc[(UNDER_REVIEW, 'TOTAL (unique granules)')].tolist() == [1, 0, 0, 0, 0]
    assert s.loc[(CANDIDATE, 'TOTAL (unique granules)')].tolist() == [1, 1, 0, 3, 1]


@pytest.mark.parametrize(
    ('issues', 'classes'),
    [
        (['Pre RTC IDs expected but not found'], ['inputs_pre_missing']),
        (['Pre RTC IDs found but not expected'], ['inputs_pre_extra']),
        (
            ['Pre RTC IDs expected but not found', 'Pre RTC IDs found but not expected'],
            ['inputs_pre_missing', 'inputs_pre_extra'],
        ),
    ],
)
def test_pre_input_failures_split_by_direction(issues: list[str], classes: list[str]) -> None:
    lists = build_deletion_lists(_tables(inputs=_inputs({_id(2): [('FAIL', i) for i in issues]})))
    assert list(lists.roots.root_class) == classes
    assert set(lists.roots.level) == {UNDER_REVIEW}
    assert _listed(lists, UNDER_REVIEW) == [_id(2), _id(3), _id(4)]
    assert lists.granules.set_index('opera_id').loc[_id(2), 'own_failures'] == ';'.join(classes)


def test_all_pass_lists_nothing() -> None:
    lists = build_deletion_lists(_tables())
    assert lists.granules.empty and lists.roots.empty and lists.generate.empty and lists.tiles.empty
    assert lists.summary.n_root_granules.sum() == 0


def test_output_is_deterministic_under_input_order() -> None:
    conf = _confirmation({_id(2): ('FAIL', 'a'), _id(1, OTHER): ('FAIL', 'b')})
    inp = _inputs({_id(0): [('FAIL', 'Pre RTC IDs found but not expected')]})
    a = build_deletion_lists(_tables(confirmation=conf, inputs=inp))
    shuffled = conf.sample(frac=1, random_state=1)
    b = build_deletion_lists(
        _tables(
            products=_products().sample(frac=1, random_state=2),
            confirmation=shuffled,
            inputs=inp.sample(frac=1, random_state=3),
        )
    )
    for name in ('granules', 'roots', 'tiles', 'summary'):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_dtype=False)


# ---------------------------------------------------------------------------------------------------------------
# Loading a run directory and writing the lists


def _write_run(
    root: Path, tables: RunTables, state: status.RunState = status.RunState.COMPLETE, **manifest_extra: object
) -> Path:
    """Write `tables` as a run directory the way `run-all` names its files."""
    d = root / f'{STAMP}_abc123'
    d.mkdir(parents=True)
    csvs = {
        'duplicates': tables.duplicates,
        'confirmation': tables.confirmation,
        'ordering': tables.ordering,
        'ordering_first_production': tables.first_production,
        'inputs': tables.inputs,
        'coverage': tables.coverage,
        'coverage_orphans': tables.orphans,
    }
    for name, df in csvs.items():
        df.to_csv(d / f'{name}_{STAMP}.csv', index=False)
    done = deletion.REQUIRED_CHECKS if state == status.RunState.COMPLETE else deletion.REQUIRED_CHECKS[:2]
    status.write_run_status(d, state, checks_planned=deletion.REQUIRED_CHECKS, checks_done=done)
    meta = {
        'tool_version': 'test',
        'params': {'start': '2026-01-01'},
        'sampling': {},
        'outputs': [f'{n}_{STAMP}.csv' for n in csvs],
        **manifest_extra,
    }
    (d / manifest.RUN_MANIFEST_FILE).write_text(json.dumps(meta))
    return d


def test_load_round_trip_matches_in_memory(tmp_path: Path) -> None:
    tables = _tables(confirmation=_confirmation({_id(2): ('FAIL', 'no prior')}))
    loaded, meta = deletion.load_run(_write_run(tmp_path, tables))
    assert meta['tool_version'] == 'test'
    a, b = build_deletion_lists(tables), build_deletion_lists(loaded)
    pd.testing.assert_frame_equal(a.granules, b.granules, check_dtype=False)


def test_load_reads_real_run_all_dtypes(tmp_path: Path) -> None:
    """Tile ids with leading zeros and boolean columns survive the CSV round trip."""
    products = _products(tiles=('01UCT',))
    tables = _tables(products, confirmation=_confirmation({}, products), inputs=_inputs({}, products))
    loaded, _ = deletion.load_run(_write_run(tmp_path, tables))
    assert set(loaded.products.mgrs_tile_id) == {'01UCT'}
    assert loaded.inputs.inputs_correct.tolist() == [True] * len(products)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda d: (d / status.RUN_STATUS_FILE).unlink(), 'not a run directory'),
        (lambda d: (d / f'inputs_{STAMP}.csv').unlink(), f'no inputs_{STAMP}.csv'),
        (
            lambda d: (
                pd.read_csv(d / f'confirmation_{STAMP}.csv')
                .drop(columns='reason')
                .to_csv(d / f'confirmation_{STAMP}.csv', index=False)
            ),
            'lacks columns',
        ),
        (
            lambda d: pd.read_csv(d / f'inputs_{STAMP}.csv').iloc[1:].to_csv(d / f'inputs_{STAMP}.csv', index=False),
            'inputs does not cover every product',
        ),
    ],
)
def test_load_refuses_broken_runs(tmp_path: Path, mutate: object, message: str) -> None:
    d = _write_run(tmp_path, _tables())
    mutate(d)
    with pytest.raises(ClickException, match=message):
        deletion.load_run(d)


def test_load_refuses_incomplete_run(tmp_path: Path) -> None:
    d = _write_run(tmp_path, _tables(), state=status.RunState.INCOMPLETE)
    with pytest.raises(ClickException, match='not a complete run-all'):
        deletion.load_run(d)


def test_load_refuses_sampled_run(tmp_path: Path) -> None:
    d = _write_run(tmp_path, _tables(), sampling={'inputs': {'n': 10}})
    with pytest.raises(ClickException, match='sampled'):
        deletion.load_run(d)


def _write(tmp_path: Path, tables: RunTables, unknown_as_failure: bool = False) -> Path:
    d = _write_run(tmp_path, tables)
    loaded, meta = deletion.load_run(d)
    lists = build_deletion_lists(loaded, unknown_as_failure)
    return deletion.write_deletion_lists(d, lists, meta, unknown_as_failure)


def test_written_files_and_id_lists(tmp_path: Path) -> None:
    conf = _confirmation({_id(2): ('FAIL', 'no prior')})
    inp = _inputs({_id(0, OTHER): [('FAIL', 'Pre RTC IDs found but not expected')]})
    target = _write(tmp_path, _tables(confirmation=conf, inputs=inp))
    assert target.name == deletion.DELETION_DIR
    expected = {'granules.csv', 'roots.csv', 'tiles.csv', 'generate.csv', 'summary.csv', 'ids'}
    assert {p.name for p in target.iterdir()} == expected | {deletion.DELETION_MANIFEST_FILE}
    ids = {p.name: p.read_text().split() for p in (target / 'ids').iterdir()}
    assert ids['candidate.txt'] == [_id(2), _id(3), _id(4)]
    assert ids['candidate_downstream.txt'] == [_id(3), _id(4)]
    assert ids['candidate_root_confirmation_failure.txt'] == [_id(2)]
    assert ids['under_review_root_inputs_pre_extra.txt'] == [_id(0, OTHER)]
    meta = json.loads((target / deletion.DELETION_MANIFEST_FILE).read_text())
    assert meta['counts'][CANDIDATE] == {'granules': 3, 'roots': 1, 'downstream': 2}
    assert meta['id_files'] == {k: len(v) for k, v in ids.items()}
    assert meta['source_tool_version'] == 'test' and meta['unknown_as_failure'] is False


def test_rewrite_replaces_the_previous_lists_and_leaves_no_temp_dirs(tmp_path: Path) -> None:
    d = _write_run(tmp_path, _tables(confirmation=_confirmation({_id(2): ('FAIL', 'no prior')})))
    loaded, meta = deletion.load_run(d)
    target = deletion.write_deletion_lists(d, build_deletion_lists(loaded), meta, False)
    (target / 'stale.txt').write_text('old')
    deletion.write_deletion_lists(d, build_deletion_lists(loaded), meta, False)
    assert not (target / 'stale.txt').exists()
    assert not [p for p in d.iterdir() if p.name.startswith('.')]


def test_unknown_as_failure_writes_its_own_directory(tmp_path: Path) -> None:
    d = _write_run(tmp_path, _unknown_tables())
    loaded, meta = deletion.load_run(d)
    deletion.write_deletion_lists(d, build_deletion_lists(loaded), meta, False)
    target = deletion.write_deletion_lists(d, build_deletion_lists(loaded, True), meta, True)
    assert target.name == deletion.DELETION_DIR_UNKNOWN_AS_FAILURE
    assert (d / deletion.DELETION_DIR / 'ids' / 'unverified.txt').exists()
    assert not (target / 'ids' / 'unverified.txt').exists()


# ---------------------------------------------------------------------------------------------------------------
# CLI, end to end from run-all


def test_cli_on_a_run_all_output(tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner) -> None:
    dist, rtc, tag_df = _world(n=3)
    tag_df.loc[1, 'prior_dist_s1_product'] = 'None'  # product 1 claims no prior: a confirmation failure
    seed_data_dir(tmp_data_dir, dist, rtc, tag_df, target_tiles=[WORLD_TILE])
    out_dir = tmp_path / 'out'
    result = _run_all(cli_runner, tmp_data_dir, out_dir)
    assert result.exit_code == 0, result.output + repr(result.exception)

    result = cli_runner.invoke(cli, ['deletion-lists', '--out-dir', str(out_dir)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    target = run_dir(out_dir) / deletion.DELETION_DIR
    granules = pd.read_csv(target / 'granules.csv')
    ids = list(dist.sort_values('acq_time').opera_id)
    assert list(granules.opera_id) == ids[1:]
    assert list(granules.role) == [ROOT, DOWNSTREAM]
    assert set(granules.tier) == {CANDIDATE}
    assert list(granules.cause_class) == ['confirmation_failure'] * 2


def test_cli_refuses_incomplete_run(tmp_path: Path, cli_runner: CliRunner) -> None:
    d = _write_run(tmp_path, _tables(), state=status.RunState.INCOMPLETE)
    result = cli_runner.invoke(cli, ['deletion-lists', '--run-dir', str(d)])
    assert result.exit_code != 0
    assert 'not a complete run-all' in result.output
    assert not (d / deletion.DELETION_DIR).exists()


# ---------------------------------------------------------------------------------------------------------------
# Comparing two runs


def _two_runs() -> tuple[deletion.DeletionLists, deletion.DeletionLists]:
    """Run A: confirmation failure at TILE k=2. Run B: the failure at k=3 instead, plus a gap in OTHER after k=3."""
    a = build_deletion_lists(_tables(confirmation=_confirmation({_id(2): ('FAIL', 'no prior')})))
    gap = _coverage(
        [{'mgrs_tile_id': OTHER, 'post_acq_dt': f'{_acq(3) + pd.Timedelta(days=1)}+00:00', 'status': 'FAIL'}]
    )
    b = build_deletion_lists(_tables(confirmation=_confirmation({_id(3): ('FAIL', 'no prior')}), coverage=gap))
    return a, b


def test_compare_classifies_every_granule() -> None:
    a, b = _two_runs()
    comp = deletion.compare_lists(a, b)
    assert comp.status.to_dict() == {_id(2): deletion.ONLY_THIS, _id(3): deletion.CHANGED, _id(4): deletion.SAME}
    differing = comp.granules.set_index('opera_id').in_other_run.to_dict()
    assert differing == {_id(2): deletion.ONLY_THIS, _id(3): deletion.CHANGED, _id(4, OTHER): deletion.ONLY_OTHER}
    row = comp.granules.set_index('opera_id').loc[_id(3)]
    assert (row.role, row.role_other) == (DOWNSTREAM, ROOT)


def test_compare_roots_and_summary() -> None:
    a, b = _two_runs()
    comp = deletion.compare_lists(a, b)
    roots = comp.roots.set_index('unit_id').in_run.to_dict()
    assert roots == {
        _id(2): deletion.ONLY_THIS,
        _id(3): deletion.ONLY_OTHER,
        pass_unit_id(OTHER, _acq(3) + pd.Timedelta(days=1)): deletion.ONLY_OTHER,
    }
    s = comp.summary.set_index(deletion.COMPARE_KEY)
    cols = ['n_this_run', 'n_other_run', 'n_same', 'n_changed', 'n_only_this_run', 'n_only_other_run']
    assert s.loc[(CANDIDATE, ROOT, 'confirmation_failure'), cols].tolist() == [1, 1, 0, 0, 1, 0]
    assert s.loc[(CANDIDATE, DOWNSTREAM, 'confirmation_failure'), cols].tolist() == [2, 1, 1, 1, 0, 0]
    assert s.loc[(CANDIDATE, DOWNSTREAM, 'coverage_gap'), cols].tolist() == [0, 1, 0, 0, 0, 1]


def test_compare_identical_runs_reports_nothing_different() -> None:
    a, _ = _two_runs()
    comp = deletion.compare_lists(a, a)
    assert comp.granules.empty and comp.roots.empty
    assert set(comp.status) == {deletion.SAME}
    assert (comp.summary.n_same == comp.summary.n_this_run).all()


def test_cli_compare_writes_comparison_files(tmp_path: Path, cli_runner: CliRunner) -> None:
    run_a = _write_run(tmp_path / 'a', _tables(confirmation=_confirmation({_id(2): ('FAIL', 'no prior')})))
    run_b = _write_run(tmp_path / 'b', _tables(confirmation=_confirmation({_id(3): ('FAIL', 'no prior')})))

    result = cli_runner.invoke(cli, ['deletion-lists', '--run-dir', str(run_a), '--compare-run-dir', str(run_b)])

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert '1 granules listed the same, 1 changed, 1 only_this_run, 0 only_other_run; 2 roots' in result.output
    target = run_a / deletion.DELETION_DIR
    granules = pd.read_csv(target / 'granules.csv').set_index('opera_id')
    assert granules.in_other_run.to_dict() == {_id(2): 'only_this_run', _id(3): 'changed', _id(4): 'same'}
    assert {'comparison_summary.csv', 'comparison_granules.csv', 'comparison_roots.csv'} <= {
        p.name for p in target.iterdir()
    }
    meta = json.loads((target / deletion.DELETION_MANIFEST_FILE).read_text())
    assert meta['compare_run_dir'] == str(run_b.resolve())
    assert meta['comparison'] == {'changed': 1, 'only_this_run': 1, 'same': 1}


def test_cli_compare_refuses_incomplete_other_run(tmp_path: Path, cli_runner: CliRunner) -> None:
    run_a = _write_run(tmp_path / 'a', _tables())
    run_b = _write_run(tmp_path / 'b', _tables(), state=status.RunState.INCOMPLETE)
    result = cli_runner.invoke(cli, ['deletion-lists', '--run-dir', str(run_a), '--compare-run-dir', str(run_b)])
    assert result.exit_code != 0 and 'not a complete run-all' in result.output
    assert not (run_a / deletion.DELETION_DIR).exists()


def test_without_compare_there_are_no_comparison_files(tmp_path: Path) -> None:
    target = _write(tmp_path, _tables(confirmation=_confirmation({_id(2): ('FAIL', 'no prior')})))
    assert not list(target.glob('comparison_*'))
    assert 'in_other_run' not in pd.read_csv(target / 'granules.csv').columns
