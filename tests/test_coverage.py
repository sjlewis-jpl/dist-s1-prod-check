import importlib.metadata
import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import boto3
import geopandas as gpd
import pandas as pd
import pytest
import requests
from click.testing import CliRunner
from dist_s1_enumerator.dist_enum import enumerate_dist_s1_products, enumerate_one_dist_s1_product

from dist_s1_prod_check import checks, cmr, coverage
from dist_s1_prod_check import cli as cli_module
from dist_s1_prod_check.constants import DELTA_LOOKBACK_DAYS, DELTA_WINDOW_DAYS, MAX_PRE_IMGS_PER_BURST
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, make_tags, rtc_id, seed_data_dir


PINNED_ENUMERATOR_VERSION = '1.0.13'
ENUMERATOR_FUNCS = (enumerate_one_dist_s1_product, enumerate_dist_s1_products)


def _default(func: Callable[..., object], name: str) -> object:
    return inspect.signature(func).parameters[name].default


@pytest.mark.parametrize('enum_func', ENUMERATOR_FUNCS, ids=lambda f: f.__name__)
def test_baseline_constants_match_enumerator(enum_func: Callable[..., object]) -> None:
    assert importlib.metadata.version('dist-s1-enumerator') == PINNED_ENUMERATOR_VERSION

    # Our coverage check models the enumerator's multi-window lookback strategy.
    assert _default(enum_func, 'lookback_strategy') == 'multi_window'

    assert tuple(_default(enum_func, 'delta_lookback_days')) == DELTA_LOOKBACK_DAYS
    assert _default(enum_func, 'delta_window_days') == DELTA_WINDOW_DAYS
    assert tuple(_default(enum_func, 'max_pre_imgs_per_burst')) == MAX_PRE_IMGS_PER_BURST

    # Our code hardcodes the minimum as `int(n_pre > 0)`, i.e. >= 1 pre-image
    # (src/dist_s1_prod_check/coverage.py:_baseline_burst_counts).
    assert _default(enum_func, 'min_pre_imgs_per_burst') == 1

    # Function defaults in our modules must use the same constants.
    assert _default(cmr.lookback_date_ranges, 'delta_lookback_days') == DELTA_LOOKBACK_DAYS
    assert _default(cmr.lookback_date_ranges, 'delta_window_days') == DELTA_WINDOW_DAYS
    assert _default(cmr.get_rtc_s1_table_with_lookback, 'delta_lookback_days') == DELTA_LOOKBACK_DAYS
    assert _default(cmr.get_rtc_s1_table_with_lookback, 'delta_window_days') == DELTA_WINDOW_DAYS
    assert _default(coverage.check_coverage, 'delta_lookback_days') == DELTA_LOOKBACK_DAYS
    assert _default(coverage.check_coverage, 'delta_window_days') == DELTA_WINDOW_DAYS


# ---------------------------------------------------------------------------------------------------------------------
# Defect tests (Task 0.3): F02, F06, F18
# ---------------------------------------------------------------------------------------------------------------------

TILE = '10SGC'
START, STOP = '2026-01-01', '2026-02-01'
T0 = pd.Timestamp('2026-01-10T02:06:53')
PROC = '2026-01-20T00:00:00'


def _group_bursts(group: int) -> list[str]:
    lut = bursts_for_tile(TILE)
    return lut[lut.acq_group_id_within_mgrs_tile == group].jpl_burst_id.tolist()


def _rtc_rows(group: int, acq: pd.Timestamp) -> list[dict]:
    """RTC rows for every burst of one acquisition group; the pass time is `acq` (the earliest burst)."""
    return [
        {'burst': burst, 'acq': acq + pd.Timedelta(seconds=i), 'proc': acq + pd.Timedelta(hours=3)}
        for i, burst in enumerate(_group_bursts(group))
    ]


def _frame(rows: list[dict]) -> gpd.GeoDataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cmr.build_rtc_input_frame(make_rtc(rows), [TILE])


def _passes(*group_times: tuple[int, pd.Timestamp]) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    frame = _frame([row for group, acq in group_times for row in _rtc_rows(group, acq)])
    return frame, coverage.expected_pass_groups(frame, START, STOP)


def _products(*acqs: pd.Timestamp, tile: str = TILE) -> pd.DataFrame:
    return checks.deduplicate_products(make_dist([{'tile': tile, 'acq': acq, 'proc': PROC} for acq in acqs]))


def _pid(acq: pd.Timestamp) -> str:
    return dist_id(TILE, acq, PROC)


def _no_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('network access attempted')


# aa2014f: the post pass's own RTC rows make the burst map non-empty, so the pass reads
# "no baseline imagery; product not expected" with missing_product_expected=False
def test_f02_missing_lookback_is_unknown_not_unexpected() -> None:
    frame, groups = _passes((0, T0))
    df_dedup = _products(T0, tile='10SGD')  # no product in TILE

    row = coverage.check_coverage(groups, df_dedup, df_rtc_frame=frame).iloc[0]

    assert not row.product_found
    assert 'not expected' not in row.reason, row.reason
    assert row.get('status') == 'UNKNOWN'
    assert 'lookback incomplete' in row.reason


def _query_windows(complete: list[bool] | None = None, n_rows: int = 10) -> list:
    """Manifest query windows exactly as `cmr.lookback_date_ranges` requests them (3 lookback + main)."""
    from dist_s1_prod_check.manifest import QueryWindow

    ranges = cmr.lookback_date_ranges(START, STOP)
    complete = complete or [True] * len(ranges)
    return [
        QueryWindow(
            start=start.tz_localize('UTC'),
            stop=stop.tz_localize('UTC'),
            complete=ok,
            n_rows=n_rows if (start, stop) != ranges[-1] else 100,
        )
        for (start, stop), ok in zip(ranges, complete)
    ]


# aa2014f: ModuleNotFoundError: No module named 'dist_s1_prod_check.manifest' (Phase 1, Task 1.4)
@pytest.mark.parametrize(
    ('case', 'expected'),
    [('all_windows_complete', True), ('one_window_missing', False), ('one_window_interrupted', False)],
)
def test_f02_manifest_lookback_completeness(case: str, expected: bool) -> None:
    from dist_s1_prod_check.manifest import lookback_complete

    windows = _query_windows()
    # The three lookback windows are disjoint: the gaps between them are never queried and never required.
    assert all(a.stop < b.start for a, b in zip(windows[:-2], windows[1:-1]))
    if case == 'one_window_missing':
        windows = windows[1:]
    elif case == 'one_window_interrupted':
        windows = _query_windows(complete=[True, False, True, True])

    post_acq_dt = T0.tz_localize('UTC')
    assert lookback_complete(windows, post_acq_dt, DELTA_LOOKBACK_DAYS, DELTA_WINDOW_DAYS) is expected


# aa2014f: ModuleNotFoundError: No module named 'dist_s1_prod_check.manifest' (Phase 1, Task 1.4)
def test_f02_complete_empty_lookback_means_no_baseline() -> None:
    frame, groups = _passes((0, T0))  # no lookback RTC rows at all
    windows = _query_windows(n_rows=0)

    df = coverage.check_coverage(groups, _products(T0, tile='10SGD'), df_rtc_frame=frame, query_windows=windows)

    row = df.iloc[0]
    assert row.get('status') != 'UNKNOWN'
    assert not row.missing_product_expected
    assert 'no baseline' in row.reason


T40 = T0 + pd.Timedelta(minutes=40)


# aa2014f: the single product is nearest to both passes (0 s and 2400 s), so both passes are product_found
@pytest.mark.xfail(strict=True, reason='F06: one product can cover two passes (independent nearest-time matching)')
def test_f06a_one_product_covers_only_one_of_two_passes() -> None:
    frame, groups = _passes((0, T0), (2, T40))
    assert len(groups) == 2

    df = coverage.check_coverage(groups, _products(T0), df_rtc_frame=frame)

    assert df.product_found.sum() == 1, df[['post_acq_dt', 'product_found', 'matched_opera_id']]
    assert set(df[df.product_found].matched_opera_id) == {_pid(T0)}


# aa2014f: both products are within 3600 s of the one pass, so check_orphan_products returns no orphan
@pytest.mark.xfail(strict=True, reason='F06: two products near one pass hide an orphan')
def test_f06b_second_product_near_one_pass_is_orphan() -> None:
    _, groups = _passes((0, T0))

    df_orphans = coverage.check_orphan_products(groups, _products(T0, T40))

    assert len(df_orphans) == 1
    assert df_orphans.opera_id.tolist() == [_pid(T40)]


# aa2014f: nearest-time matching maps both passes (0 s, 100 s) to the product at 90 s; the one at 120 s is unused
@pytest.mark.xfail(strict=True, reason='F06: greedy nearest-time matching is not a one-to-one assignment')
def test_f06c_optimal_assignment_beats_greedy() -> None:
    t100 = T0 + pd.Timedelta(seconds=100)
    p90, p120 = T0 + pd.Timedelta(seconds=90), T0 + pd.Timedelta(seconds=120)
    frame, groups = _passes((0, T0), (2, t100))
    df_dedup = _products(p90, p120)

    df = coverage.check_coverage(groups, df_dedup, df_rtc_frame=frame, tolerance_seconds=100)
    df_orphans = coverage.check_orphan_products(groups, df_dedup, tolerance_seconds=100)

    matched = dict(zip(df.post_acq_dt.dt.tz_convert(None), df.matched_opera_id))
    assert matched == {T0: _pid(p90), t100: _pid(p120)}
    assert df_orphans.empty


# aa2014f: today's per-direction checks already give MISSING + orphan here; the test fails with
# ModuleNotFoundError: No module named 'dist_s1_prod_check.matching' (shared assignment, Phase 2)
@pytest.mark.xfail(strict=True, reason='F06: no shared assignment; unmatched passes and products must come from one')
def test_f06d_no_candidate_within_tolerance() -> None:
    far = T0 + pd.Timedelta(hours=2)
    frame, groups = _passes((0, T0))
    df_dedup = _products(far)

    df = coverage.check_coverage(groups, df_dedup, df_rtc_frame=frame)
    df_orphans = coverage.check_orphan_products(groups, df_dedup)
    assert not df.product_found.any() and df_orphans.opera_id.tolist() == [_pid(far)]

    from dist_s1_prod_check import matching

    assignment = matching.assign_passes(groups, df_dedup, tolerance_seconds=3600)
    assert assignment.pass_to_product == {} and assignment.product_to_pass == {}


# aa2014f: TypeError: check_coverage() got an unexpected keyword argument 'df_tags' (the tagged post bursts are
# never consulted, so the product of acquisition group 2 is accepted for the group-0 pass)
@pytest.mark.xfail(strict=True, reason='F06: acquisition group is not checked through the tagged post bursts')
def test_f06e_tagged_group_mismatch_is_unknown() -> None:
    frame, groups = _passes((0, T0))
    df_dedup = _products(T0)
    other_group_posts = [rtc_id(r['burst'], r['acq'], r['proc']) for r in _rtc_rows(2, T0)]
    df_tags = make_tags([{'opera_id': _pid(T0), 'post_rtc_opera_ids': other_group_posts}])

    df = coverage.check_coverage(groups, df_dedup, df_rtc_frame=frame, df_tags=df_tags)

    row = df.iloc[0]
    assert row.status == 'UNKNOWN'
    assert 'group mismatch' in row.reason


# aa2014f: coverage matches 1 of 2 products and the orphan check reports 0, so 1 product is in neither set
@pytest.mark.xfail(strict=True, reason='F06: coverage and orphan checks do not share one assignment')
def test_f06f_coverage_and_orphans_share_one_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, groups = _passes((0, T0))
    df_dedup = _products(T0, T40)

    df = coverage.check_coverage(groups, df_dedup, df_rtc_frame=frame)
    df_orphans = coverage.check_orphan_products(groups, df_dedup)
    matched = set(df[df.product_found].matched_opera_id)
    orphans = set(df_orphans.opera_id) if not df_orphans.empty else set()
    assert matched.isdisjoint(orphans) and matched | orphans == set(df_dedup.opera_id), (matched, orphans)

    from dist_s1_prod_check import matching

    assignment = matching.assign_passes(groups, df_dedup, tolerance_seconds=3600)
    assert {v: k for k, v in assignment.pass_to_product.items()} == assignment.product_to_pass

    calls = []
    real_assign = matching.assign_passes

    def _spy(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real_assign(*args, **kwargs)

    monkeypatch.setattr(matching, 'assign_passes', _spy)
    cli_module._coverage_section(df_dedup, frame, START, STOP, {})
    assert len(calls) == 1, 'coverage and orphans must consume the same assignment object'


F18_A_V1 = dist_id(TILE, T0, '2026-09-01T00:00:00')
F18_A_V2 = dist_id(TILE, T0, '2026-09-30T00:00:00')


def _run_all_processing_window(
    tmp_path: Path, data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """`run-all --processing-start` that puts A-v1 before the window and A-v2 inside it; capture report and CSVs."""
    monkeypatch.setattr(requests.Session, 'request', _no_network)
    monkeypatch.setattr(boto3, 'client', _no_network)
    captured: dict = {'sections': [], 'csvs': {}}

    def _csvs(out_dir: Path, stamp: str, csvs: dict) -> None:
        captured['csvs'].update(csvs)

    def _report(out_dir: Path, stamp: str, tile_geoms: object, sections: list[dict], *a: object) -> None:
        captured['sections'].extend(sections)

    monkeypatch.setattr(cli_module, '_write_csvs', _csvs)
    monkeypatch.setattr(cli_module, '_write_report', _report)

    dist = make_dist([{'opera_id': F18_A_V1}, {'opera_id': F18_A_V2}])
    rtc = make_rtc(_rtc_rows(0, T0))
    tags = make_tags([{'opera_id': F18_A_V1}, {'opera_id': F18_A_V2}])
    seed_data_dir(data_dir, dist=dist, rtc=rtc, tags=tags, target_tiles=[TILE])
    args = ['run-all', '--start', START, '--stop', STOP, '--processing-start', '2026-09-24', '--inputs-workers', '1']
    args += ['--data-dir', str(data_dir), '--out-dir', str(tmp_path / 'out')]
    result = cli_runner.invoke(cli_module.cli, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    return captured['sections'], captured['csvs']


# aa2014f: the processing filter runs before deduplication, so A-v2 gets n_versions=1 and is_reprocessed=False
@pytest.mark.xfail(strict=True, reason='F18: n_versions is recomputed inside the processing window')
def test_f18_n_versions_from_unfiltered_table(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, csvs = _run_all_processing_window(tmp_path, tmp_data_dir, cli_runner, monkeypatch)

    seen = []
    for name, df in csvs.items():
        if 'opera_id' not in df.columns or F18_A_V2 not in set(df.opera_id):
            continue
        row = df[df.opera_id == F18_A_V2].iloc[0]
        if 'n_versions' in df.columns:
            seen.append((name, 'n_versions', int(row.n_versions) == 2))
        if 'is_reprocessed' in df.columns:
            seen.append((name, 'is_reprocessed', bool(row.is_reprocessed)))
    assert seen and all(ok for *_, ok in seen), seen


# aa2014f: the coverage description only says passes "should have triggered a DIST-S1 product"; it never states
# that every pass in the whole acquisition window is expected in a processing-window run
@pytest.mark.xfail(strict=True, reason='F18: coverage text does not state the whole acquisition window is expected')
def test_f18_coverage_text_states_whole_acquisition_window(
    tmp_path: Path, tmp_data_dir: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    sections, _ = _run_all_processing_window(tmp_path, tmp_data_dir, cli_runner, monkeypatch)

    (cov,) = [s for s in sections if s['key'] == 'coverage']
    assert 'acquisition window' in cov['description'].lower(), cov['description']
