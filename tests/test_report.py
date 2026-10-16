"""Defect tests for section states, counts/units and truncation in the HTML report (F11, F12, F13).

A section's "green" status is decided in the report template today: the stat tile gets class ``ok``
when ``sec.n_failures === 0``. `_is_green` mirrors that rule unless the section carries an explicit
``green`` flag (the interface assumed for Phase 1, Task 1.2), so each test reads today's verdict.
"""

import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from click.testing import CliRunner
from dist_s1_enumerator.mgrs_burst_data import get_mgrs_table

from dist_s1_prod_check import cli as cli_mod
from dist_s1_prod_check import report
from dist_s1_prod_check.cli import cli
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, make_tags, seed_data_dir


TILE = '11SLT'
ACQ0 = pd.Timestamp('2026-01-01T13:50:00')


def _payload(html_path: Path) -> dict:
    """Return the ``DATA`` object embedded in a generated report."""
    text = html_path.read_text()
    match = re.search(r'const DATA = (\{.*?\});\n', text, flags=re.S)
    assert match, 'no DATA payload in report'
    return json.loads(match.group(1).replace('<\\/', '</'))


def _is_green(section: dict) -> bool:
    """Whether the report shows this section green: explicit ``green`` flag, else today's template rule."""
    if 'green' in section:
        return bool(section['green'])
    return section['n_failures'] == 0


def _tile_geoms(tiles: list[str]) -> gpd.GeoDataFrame:
    df = get_mgrs_table()[['mgrs_tile_id', 'geometry']]
    return df[df.mgrs_tile_id.isin(tiles)].reset_index(drop=True)


def _chain(tile: str, n: int, start: pd.Timestamp = ACQ0) -> list[str]:
    """Opera ids of ``n`` consecutive 12-day acquisitions in ``tile``, processed in order."""
    return [
        dist_id(tile, start + pd.Timedelta(days=12 * k), start + pd.Timedelta(days=12 * k, hours=6)) for k in range(n)
    ]


def _chain_tags(ids: list[str]) -> list[dict]:
    """Tags whose prior points at the previous id (first gets the literal 'None')."""
    return [{'opera_id': oid, 'prior_dist_s1_product': prev} for oid, prev in zip(ids, ['None', *ids[:-1]])]


# --- F11 -------------------------------------------------------------------------------------------------------------


# aa2014f: duplicates/ordering/confirmation get n_checked=0, n_failures=0, no state -> green; run-all then exits 1
# (AttributeError in the inputs section on the empty universe), so the report lacks inputs/coverage.
def test_f11_processing_start_after_all_products_gives_empty_sections(
    tmp_data_dir: Path, tmp_path: Path, cli_runner: CliRunner
) -> None:
    ids = _chain(TILE, 3)
    burst = bursts_for_tile(TILE).jpl_burst_id.iloc[0]
    # RTC far outside --start/--stop so coverage has no passes either.
    rtc = make_rtc([{'burst': burst, 'acq': '2020-01-01T13:50:00', 'proc': '2020-01-01T20:00:00'}])
    seed_data_dir(
        tmp_data_dir, dist=make_dist([{'opera_id': i} for i in ids]), rtc=rtc, tags=make_tags(_chain_tags(ids))
    )
    out_dir = tmp_path / 'out'
    result = cli_runner.invoke(
        cli,
        [
            'run-all',
            '--start', '2026-01-01',
            '--stop', '2026-03-01',
            '--processing-start', '2030-01-01',
            '--data-dir', str(tmp_data_dir),
            '--out-dir', str(out_dir),
        ],
    )  # fmt: skip
    sections = _payload(next(out_dir.glob('report_*.html')))['sections']
    assert sections
    for sec in sections:
        assert sec['n_checked'] == 0
        assert not _is_green(sec), f'{sec["key"]} shown green with 0 checked'
        assert sec.get('state') == 'EMPTY', sec['key']
    assert result.exit_code == 0, result.output
    assert {s['key'] for s in sections} >= {'duplicates', 'ordering', 'confirmation', 'inputs', 'coverage'}


def _confirmation_html(df_dist: gpd.GeoDataFrame, df_tags: pd.DataFrame, tmp_path: Path) -> tuple[dict, str]:
    sections, _ = cli_mod._confirmation_section(df_dist.assign(n_versions=1), df_tags, False, {})
    html_path = report.write_html_report(
        sections, _tile_geoms(sorted(df_dist.mgrs_tile_id.unique())), tmp_path / 'r.html'
    )
    return sections[0], html_path.read_text()


# aa2014f: n_checked=100 (all first-in-window), n_failures=0 -> green, no 'not evaluated' anywhere in the HTML.
def test_f11_all_skipped_confirmation_is_not_evaluated(tmp_path: Path) -> None:
    tiles = sorted(get_mgrs_table().mgrs_tile_id)[:100]
    ids = [dist_id(t, ACQ0, ACQ0 + pd.Timedelta(hours=6)) for t in tiles]
    sec, html = _confirmation_html(
        make_dist([{'opera_id': i} for i in ids]), make_tags([{'opera_id': i} for i in ids]), tmp_path
    )
    assert sec['n_checked'] == 0, f'first-in-window products counted as checked: {sec["n_checked"]}'
    assert sec.get('n_skipped') == 100
    assert sec.get('state') == 'COMPLETE'
    assert not _is_green(sec)
    assert 'not evaluated' in html


# aa2014f: n_checked=13 (skips counted as checked) and the section has no n_skipped.
def test_f11_evaluated_plus_skipped_is_green_with_split_counts(tmp_path: Path) -> None:
    ids = _chain('11SLT', 5) + _chain('11SLU', 4) + _chain('11SKT', 4)
    tags = make_tags(_chain_tags(ids[:5]) + _chain_tags(ids[5:9]) + _chain_tags(ids[9:]))
    sec, _ = _confirmation_html(make_dist([{'opera_id': i} for i in ids]), tags, tmp_path)
    assert sec['n_failures'] == 0
    assert _is_green(sec)
    assert sec['n_checked'] == 10
    assert sec.get('n_skipped') == 3


# --- F21 -------------------------------------------------------------------------------------------------------------


def test_f21_map_initialised_last_and_only_with_leaflet(tmp_path: Path) -> None:
    df = pd.DataFrame({'opera_id': ['p0'], 'mgrs_tile_id': [TILE], 'reason': ['bad']})
    section = report.build_section('one', 'One', 'One failure.', df, 1)
    html = report.write_html_report([section], _tile_geoms([TILE]), tmp_path / 'r.html').read_text()
    script = html[html.index('const DATA = ') :]
    map_at = script.index('L.map(')
    assert script.count('L.map(') == 1
    guard_at = script.index('if (window.L) {')
    assert guard_at < map_at < script.index('} else {', guard_at)
    assert 'showMapUnavailable()' in script[script.index('} else {', guard_at) :]
    # Stat tiles and section tables are rendered before the map is touched.
    for marker in ("tilesDiv.insertAdjacentHTML('beforeend'", 'sectionsDiv.appendChild(el)'):
        assert script.index(marker) < guard_at, marker


# --- F12 -------------------------------------------------------------------------------------------------------------


def _inputs_sections(monkeypatch: pytest.MonkeyPatch, issues: list[tuple[str, str]]) -> dict[str, dict]:
    """Run `cli._inputs_section` on synthetic per-issue result rows ``(opera_id, issue_type)``."""
    df_results = pd.DataFrame(
        [
            {
                'opera_id': oid,
                'mgrs_tile_id': oid.split('_')[3].lstrip('T'),
                'inputs_correct': not issue,
                'issue_type': issue,
                'issue_value': 'x' if issue else '',
            }
            for oid, issue in issues
        ]
    )
    monkeypatch.setattr(cli_mod.inputs, 'check_inputs_offline', lambda *a, **k: df_results)
    df_tags = pd.DataFrame({'opera_id': df_results.opera_id.unique()})  # the universe (F03 assertion)
    sections, _ = cli_mod._inputs_section(df_tags, None, 'offline', 1, {})
    return {s['key']: s for s in sections}


# aa2014f: n_failures=4 (issue rows) against n_checked=3 products; no n_failed/n_issues/unit keys.
def test_f12_inputs_counts_products_and_issues_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, c = _chain(TILE, 3)
    secs = _inputs_sections(
        monkeypatch,
        [
            (a, 'Post RTC input mismatch'),
            (a, 'Pre RTC input mismatch'),
            (a, 'Burst ID mismatch'),
            (b, 'Pre RTC input mismatch'),
            (c, ''),
        ],
    )
    sec = secs['inputs']
    assert sec['n_checked'] == 3
    assert sec.get('n_failed') == 2, f'n_failures={sec["n_failures"]} counts issue rows, not products'
    assert sec.get('n_issues') == 4
    assert sec.get('unit') == 'products'


# aa2014f: inputs_post + inputs_pre hold 2 of 4 issues; 'Burst ID mismatch' and track issues fall in neither.
def test_f12_inputs_issue_subsections_partition_all_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, _ = _chain(TILE, 3)
    secs = _inputs_sections(
        monkeypatch,
        [
            (a, 'Post RTC input mismatch'),
            (a, 'Burst ID mismatch'),
            (b, 'Pre RTC input mismatch'),
            (b, 'Too many track numbers present'),
        ],
    )
    total = secs['inputs']['n_failures']
    subsections = [k for k in secs if k != 'inputs']
    assert sum(secs[k]['n_failures'] for k in subsections) == total, f'{subsections} miss issues'
    assert 'inputs_other' in secs


# aa2014f: n_checked=3 includes the ineligible pass; no n_universe/n_eligible/n_missing keys.
def test_f12_coverage_counts_universe_eligible_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    post = [ACQ0 + pd.Timedelta(days=12 * k) for k in range(3)]
    df_cov = pd.DataFrame(
        {
            'mgrs_tile_id': [TILE] * 3,
            'post_acq_dt': pd.to_datetime(post).tz_localize('UTC'),
            'product_found': [True, False, False],
            'missing_product_expected': [True, True, False],  # eligible, eligible, ineligible (no baseline)
            'eligible': [True, True, False],
            'status': ['PASS', 'FAIL', 'SKIPPED'],
            'burst_ids': [''] * 3,
            'burst_pols': [''] * 3,
        }
    )
    monkeypatch.setattr(cli_mod.coverage, 'expected_pass_groups', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cli_mod.matching, 'assign_passes', lambda *a, **k: None)
    monkeypatch.setattr(cli_mod.coverage, 'check_coverage', lambda *a, **k: df_cov)
    monkeypatch.setattr(cli_mod.coverage, 'check_orphan_products', lambda *a, **k: pd.DataFrame())
    sections, _ = cli_mod._coverage_section(pd.DataFrame(), None, '2026-01-01', '2026-03-01', {})
    sec = sections[0]
    assert sec['n_failures'] == 1
    assert sec['n_checked'] == 2, f'denominator {sec["n_checked"]} includes the ineligible pass'
    assert sec.get('n_universe') == 3
    assert sec.get('n_eligible') == 2
    assert sec.get('n_missing') == 1
    assert sec.get('unit') == 'passes'


# aa2014f: n_failures=5 (every version row); no n_groups/n_versions/n_superseded keys.
def test_f12_duplicates_counts_groups_versions_superseded() -> None:
    acq1, acq2 = ACQ0, ACQ0 + pd.Timedelta(days=12)
    rows = [{'tile': TILE, 'acq': acq1, 'proc': acq1 + pd.Timedelta(hours=h)} for h in (6, 30, 54)]
    rows += [{'tile': TILE, 'acq': acq2, 'proc': acq2 + pd.Timedelta(hours=h)} for h in (6, 30)]
    rows += [{'tile': TILE, 'acq': acq2 + pd.Timedelta(days=12), 'proc': acq2 + pd.Timedelta(days=12, hours=6)}]
    sections, _ = cli_mod._duplicates_section(make_dist(rows), {})
    sec = sections[0]
    assert sec.get('n_groups') == 2, f'n_failures={sec["n_failures"]} counts every version row'
    assert sec.get('n_versions') == 5
    assert sec.get('n_superseded') == 3
    assert sec.get('unit') == 'duplicate groups'


# --- F13 -------------------------------------------------------------------------------------------------------------

N_ROWS = report.MAX_EMBEDDED_ROWS + 1
TILE_Z = '11SLU'


@pytest.fixture(scope='module')
def big_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, str]:
    """Build a report whose single section has 50,001 failure rows; only the last is in `TILE_Z`."""
    tiles = [TILE] * (N_ROWS - 1) + [TILE_Z]
    df = pd.DataFrame({'opera_id': [f'p{i}' for i in range(N_ROWS)], 'mgrs_tile_id': tiles, 'reason': 'bad'})
    section = report.build_section('big', 'Big', 'Many failures.', df, N_ROWS)
    out = report.write_html_report([section], _tile_geoms([TILE, TILE_Z]), tmp_path_factory.mktemp('f13') / 'r.html')
    return _payload(out)['sections'][0], out.read_text()


# aa2014f: map features are built from the 50,000 embedded records, so only tile 11SLT is drawn.
@pytest.mark.xfail(strict=True, reason='F13: map is built from the truncated table and drops failing tiles')
def test_f13_map_includes_tile_beyond_truncation(big_report: tuple[dict, str]) -> None:
    sec, _ = big_report
    assert sec['truncated']
    drawn = {f['properties']['mgrs_tile_id'] for f in sec['geojson']['features']}
    assert TILE_Z in drawn, f'map shows {drawn}'


# aa2014f: only the generic '(table truncated; full results in the CSV ...)' note; no shown-of-total count.
@pytest.mark.xfail(strict=True, reason='F13: truncated table/export is not labelled with how much was dropped')
def test_f13_truncated_table_labelled_with_counts(big_report: tuple[dict, str]) -> None:
    _, html = big_report
    assert 'showing first 50,000 of 50,001' in html


# --- Phase 4, task 0: the script must render with and without Leaflet ----------------------------------------------


# c27b56b..4fcf2a8: esc() and isUrl() were defined inside `if (window.L) {...}`, so the stat-tile loop threw
# "esc is not a function" before rendering anything, with or without Leaflet.
@pytest.mark.parametrize('leaflet', [False, True])
def test_report_script_renders_tiles_and_sections(tmp_path: Path, leaflet: bool) -> None:
    from jsrender import render

    df = pd.DataFrame({'opera_id': ['p0'], 'mgrs_tile_id': [TILE], 'reason': ['bad']})
    section = report.build_section('one', 'One', 'One failure.', df, 1)
    out = report.write_html_report([section], _tile_geoms([TILE]), tmp_path / 'r.html')
    page = render(out, leaflet=leaflet)
    assert page['error'] is None, page['error']
    assert 'One' in page['elements']['tiles']['innerHTML']
    assert len(page['sections']) == 1 and 'One failure.' in page['sections'][0]
    if leaflet:
        assert ['geoJSON', 1] in page['mapCalls']
    else:
        assert 'Map unavailable' in page['elements']['map']['innerHTML']
