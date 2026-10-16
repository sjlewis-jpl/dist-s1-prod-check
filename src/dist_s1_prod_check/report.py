import json
from html import escape
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dist_s1_prod_check.status import SectionState, Status, is_green


CHECK_COLORS = [
    ('#2a78d6', '#3987e5'),
    ('#eb6834', '#d95926'),
    ('#1baf7a', '#199e70'),
    ('#eda100', '#c98500'),
    ('#e87ba4', '#d55181'),
    ('#008300', '#008300'),
    ('#4a3aa7', '#9085e9'),
    ('#e34948', '#e66767'),
]

MAX_EMBEDDED_ROWS = 50_000
MAX_POPUP_PRODUCTS = 12


def _verdict(state: SectionState, counts: dict[str, int], n_checked: int) -> str:
    """One-line verdict shown on a section's stat tile and header."""
    n_fail, n_unknown = counts.get(Status.FAIL, 0), counts.get(Status.UNKNOWN, 0)
    if state == SectionState.EMPTY:
        return 'empty: nothing to check'
    if n_fail or n_unknown:
        return f'{n_fail:,} failed, {n_unknown:,} unknown'
    if n_checked == 0:
        return 'not evaluated: insufficient evidence'
    if state == SectionState.SAMPLED:
        return 'no failures in the sample'
    return 'no failures'


def build_section(
    key: str,
    title: str,
    description: str,
    df_failures: pd.DataFrame,
    n_checked: int,
    summary: dict | None = None,
    *,
    state: SectionState = SectionState.COMPLETE,
    unit: str = 'products',
    n_universe: int | None = None,
    n_skipped: int = 0,
    counts: dict[str, int] | None = None,
    note: str = '',
) -> dict:
    """Package one check's failures and counts for the HTML report.

    `n_checked` counts the units actually evaluated (PASS + FAIL + INFO), `n_skipped` those deliberately not
    evaluated, and `n_universe` every unit the section should cover (default `n_checked + n_skipped`). `counts`
    maps each `Status` to a unit count; by default every failure row is one FAIL and the other checked units PASS.
    A COMPLETE section with an empty universe becomes EMPTY. The section is green only under `status.is_green`.

    `summary` is an optional aggregate table `{'title', 'columns', 'records'}` rendered below the
    main table; clicking one of its rows filters the main table by the row's first cell.
    """
    df = df_failures.drop(columns=[c for c in ['geometry'] if c in df_failures.columns])
    truncated = len(df) > MAX_EMBEDDED_ROWS
    df = df.head(MAX_EMBEDDED_ROWS)
    records = json.loads(df.to_json(orient='records', date_format='iso', default_handler=str))
    n_universe = n_checked + n_skipped if n_universe is None else n_universe
    if counts is None:
        n_fail = len(df_failures)
        counts = {Status.PASS: max(n_checked - n_fail, 0), Status.FAIL: n_fail}
    counts = {str(Status(k)): int(v) for k, v in counts.items() if v}
    if state == SectionState.COMPLETE and n_universe == 0:
        state = SectionState.EMPTY
    return {
        'key': key,
        'title': title,
        'description': description,
        'state': str(state),
        'unit': unit,
        'n_universe': int(n_universe),
        'n_checked': int(n_checked),
        'n_skipped': int(n_skipped),
        'counts': counts,
        'green': is_green(state, counts, n_checked),
        'verdict': _verdict(state, counts, n_checked),
        'note': note,
        'n_failures': int(len(df_failures)),
        'truncated': truncated,
        'columns': df.columns.tolist(),
        'records': records,
        'summary': summary,
    }


def _tile_features(section: dict, tile_geoms: gpd.GeoDataFrame) -> list[dict]:
    df = pd.DataFrame(section['records'])
    if df.empty or 'mgrs_tile_id' not in df.columns:
        return []
    geom_by_tile = tile_geoms.set_index('mgrs_tile_id').geometry
    features = []
    for tile, group in df.groupby('mgrs_tile_id'):
        if tile not in geom_by_tile.index:
            continue
        products = group.head(MAX_POPUP_PRODUCTS).to_dict('records')
        features.append(
            {
                'type': 'Feature',
                'geometry': geom_by_tile[tile].__geo_interface__,
                'properties': {'mgrs_tile_id': tile, 'n_failures': int(len(group)), 'products': products},
            }
        )
    return features


def _banner_html(run_status: dict) -> str:
    """Return the banner for a run that did not complete, rendered server-side so it shows without JavaScript."""
    if run_status['state'] == 'COMPLETE':
        return ''
    failed = run_status.get('failed_check')
    what = (
        f'<code>{escape(failed)}</code> raised {escape(str(run_status.get("error")))}'
        if failed
        else 'run did not finish'
    )
    not_run = ', '.join(run_status.get('not_run') or []) or 'none'
    return (
        f'<div class="banner" id="banner"><b>INCOMPLETE:</b> {what}; not run: {escape(not_run)}. '
        'Results below are partial and must not be read as a pass.</div>'
    )


def command_line(command: str, params: dict) -> str:
    """Rebuild the command line from a run manifest's parameters (flags that are off and empty values omitted)."""
    parts = [command]
    for name, value in params.items():
        flag = '--' + name.replace('_', '-')
        if value is None or value is False or value == '' or value == []:
            continue
        if value is True:
            parts.append(flag)
        elif isinstance(value, list):
            parts += [flag, *(str(v) for v in value)]
        else:
            parts += [flag, str(value)]
    return ' '.join(parts)


def _table_provenance(name: str, t: dict) -> tuple[str, str]:
    """One provenance row per cached table. Receipts and adopted caches never share a time label."""
    if t.get('provenance') == 'query_receipt':
        kind, when = 'query receipt', f'query_time_utc {t.get("query_time_utc")}'
    else:
        kind, when = 'adopted cache (adopted_unverified)', f'file_mtime_utc (inferred) {t.get("file_mtime_utc")}'
    windows = t.get('lookback_windows') or []
    n_incomplete = sum(not w['complete'] for w in windows)
    text = (
        f'{kind}; {t.get("source")}; {t.get("start")} to {t.get("stop")}; bbox {t.get("bbox") or "none (global)"}; '
        f'{t.get("n_rows"):,} rows; {len(windows)} query windows ({n_incomplete} incomplete); {when}; '
        f'sha256 {str(t.get("parquet_sha256"))[:12]}...'
    )
    return f'{name} table', text


def provenance_rows(prov: dict) -> list[tuple[str, str]]:
    """Label/value rows of the provenance block, generated from a `RunManifest` dict only."""
    rows = [
        ('Command', command_line(prov['command'], prov.get('params', {}))),
        ('Tool version', f'dist-s1-prod-check {prov.get("tool_version")}'),
        ('Enumerator version', f'dist-s1-enumerator {prov.get("enumerator_version")}'),
        ('Network', 'offline run (no network access; cached data only)' if prov.get('offline') else 'online run'),
        ('Run type', 'regression baseline run' if prov.get('regression_baseline') else 'gate run'),
    ]
    vf = prov.get('version_filter') or {}
    if vf:
        excluded = ', '.join(f'{v}: {n:,}' for v, n in sorted(vf.get('excluded_by_version', {}).items()))
        rows.append(
            (
                'Product versions',
                f'{", ".join(vf["allowed"])}; excluded {vf.get("n_excluded", 0):,} products of other versions'
                + (f' ({excluded})' if excluded else ''),
            )
        )
    pf = prov.get('processing_filter') or {}
    if pf:
        rows.append(('Processing filter', f'processed {pf.get("start") or "start"} to {pf.get("stop") or "now"}; '
                     f'kept {pf.get("n_kept", 0):,} of {pf.get("n_before", 0):,} products'))  # fmt: skip
    wf = prov.get('window_filter') or {}
    if wf:
        n_out = wf.get('n_before', 0) - wf.get('n_kept', 0)
        rows.append(('Acquisition window', f'[{wf["start"]}, {wf["stop"]}): kept {wf.get("n_kept", 0):,} of '
                     f'{wf.get("n_before", 0):,} cached products; {n_out:,} outside the acquisition window '
                     'excluded (kept in the cache as context)'))  # fmt: skip
    drops = prov.get('rtc_frame_drops') or {}
    if drops:
        detail = ', '.join(f'{k.replace("_", " ")} {v:,}' for k, v in drops.items())
        rows.append(('RTC-S1 rows dropped', f'building the enumerator input frame: {detail}'))
    for key, smp in (prov.get('sampling') or {}).items():
        rows.append(
            (
                f'Sample: {key}',
                f'{smp["n_selected"]:,} of {smp["n_universe"]:,} {smp.get("unit", "products")} (seed {smp["seed"]}, '
                f'selected ids sha256 {smp["selected_sha256"][:12]}...)',
            )
        )
    rows += [_table_provenance(name, t) for name, t in (prov.get('tables') or {}).items()]
    for name, sha in (prov.get('file_sha256') or {}).items():
        rows.append((name, f'sha256 {sha[:12]}... (no manifest; per-product cache)'))
    rows += [('Not checked', text) for text in prov.get('disclaimers', [])]
    return rows


def _provenance_html(prov: dict | None) -> str:
    if not prov:
        return ''
    banner = (
        '<div class="banner adopted"><b>ADOPTED CACHE: provenance unverified (regression run).</b> The cached tables '
        'were adopted with adopt-cache; nothing proves what was queried. This report cannot serve as a gate.</div>'
        if prov.get('adopted_cache')
        else ''
    )
    body = ''.join(f'<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>' for k, v in provenance_rows(prov))
    return f'{banner}<details class="prov" open><summary>Provenance</summary><table>{body}</table></details>'


def write_html_report(
    sections: list[dict],
    tile_geoms: gpd.GeoDataFrame,
    out_path: str | Path,
    title: str = 'DIST-S1 Production Check',
    subtitle: str = '',
    run_status: dict | None = None,
    provenance: dict | None = None,
) -> Path:
    """Write a self-contained HTML report: map of failing MGRS tiles, stat tiles, and per-check tables with CSV export.

    `run_status` is ``{'state', 'failed_check', 'error', 'not_run'}``. Unless its state is COMPLETE the page opens
    with an INCOMPLETE banner, and the failed and never-run checks are drawn as grey tiles (they are not sections).
    `provenance` is a `RunManifest` dict; the provenance block is generated from it alone.
    """
    payload = {
        'title': title,
        'subtitle': subtitle,
        'run_status': run_status or {'state': 'COMPLETE', 'failed_check': None, 'error': None, 'not_run': []},
        'sections': [
            {
                **section,
                'colorLight': CHECK_COLORS[i % len(CHECK_COLORS)][0],
                'colorDark': CHECK_COLORS[i % len(CHECK_COLORS)][1],
                'geojson': {'type': 'FeatureCollection', 'features': _tile_features(section, tile_geoms)},
            }
            for i, section in enumerate(sections)
        ],
    }
    data_js = json.dumps(payload, default=str).replace('</', '<\\/')
    html = (
        _TEMPLATE.replace('__TITLE__', title)
        .replace('__BANNER__', _banner_html(payload['run_status']) + _provenance_html(provenance))
        .replace('__DATA__', data_js)
    )
    out_path = Path(out_path)
    out_path.write_text(html)
    return out_path


_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
    --good: #0ca30c; --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
      --good: #0ca30c; --critical: #d03b3b;
    }
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--text-primary);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1320px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 1.35rem; }
  .subtitle { color: var(--text-secondary); margin-top: 4px; font-size: 0.9rem; }
  .tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
          padding: 12px 16px; min-width: 168px; }
  .tile .label { font-size: 0.78rem; color: var(--text-secondary); display: flex; align-items: center; gap: 6px; }
  .tile .swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  .tile .value { font-size: 1.55rem; margin-top: 2px; }
  .tile .sub { font-size: 0.75rem; color: var(--muted); }
  .tile .value.ok { color: var(--good); } .tile .value.bad { color: var(--critical); }
  #map { height: 520px; border-radius: 12px; border: 1px solid var(--border); }
  .section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
             padding: 16px; margin-top: 20px; }
  .section h2 { font-size: 1.05rem; display: flex; align-items: center; gap: 8px; }
  .section .desc { color: var(--text-secondary); font-size: 0.85rem; margin: 6px 0 12px; }
  .controls { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
  .controls input { background: var(--page); color: var(--text-primary); border: 1px solid var(--grid);
                    border-radius: 7px; padding: 7px 10px; font-size: 0.85rem; width: 300px; }
  .controls button { background: var(--page); color: var(--text-primary); border: 1px solid var(--grid);
                     border-radius: 7px; padding: 7px 12px; font-size: 0.85rem; cursor: pointer; }
  .controls button:hover { border-color: var(--muted); }
  .controls .count { font-size: 0.8rem; color: var(--muted); }
  .tbl-wrap { overflow-x: auto; max-height: 480px; overflow-y: auto; border: 1px solid var(--grid); border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.78rem; }
  th { position: sticky; top: 0; background: var(--surface-1); text-align: left; color: var(--text-secondary);
       border-bottom: 1px solid var(--grid); padding: 7px 10px; white-space: nowrap; cursor: pointer;
       user-select: none; }
  th:hover { color: var(--text-primary); }
  td { border-bottom: 1px solid var(--grid); padding: 6px 10px; white-space: nowrap;
       font-variant-numeric: tabular-nums; max-width: 480px; overflow: hidden; text-overflow: ellipsis; }
  .summary-tbl { margin-top: 14px; }
  .summary-tbl h3 { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 6px; }
  .summary-tbl td { cursor: pointer; }
  .summary-tbl tbody tr:hover td { background: var(--page); }
  td a { color: inherit; text-decoration: underline; text-decoration-color: var(--muted); }
  .banner { background: var(--critical); color: #fff; border-radius: 10px; padding: 12px 16px; margin-top: 14px;
            font-size: 0.95rem; }
  .banner.adopted { background: #b26b00; }
  .prov { margin-top: 12px; font-size: 0.8rem; color: var(--text-secondary); }
  .prov summary { cursor: pointer; }
  .prov table { width: auto; margin-top: 6px; }
  .prov th { position: static; cursor: default; padding: 3px 10px 3px 0; vertical-align: top; }
  .prov td { white-space: normal; max-width: 1000px; padding: 3px 0; }
  .tile.grey { opacity: 0.75; } .tile .value.na { color: var(--muted); }
  .empty.na { color: var(--muted); }
  .empty { color: var(--good); font-size: 0.9rem; padding: 8px 0; }
  .leaflet-popup-content { font: 0.78rem system-ui, sans-serif; max-height: 260px; overflow-y: auto; }
  .leaflet-popup-content b { font-size: 0.85rem; }
  .leaflet-popup-content .prod { margin-top: 6px; border-top: 1px solid #ddd; padding-top: 4px; }
</style>
</head>
<body>
<div class="wrap">
  <h1 id="title"></h1>
  <div class="subtitle" id="subtitle"></div>
  __BANNER__
  <div class="tiles" id="tiles"></div>
  <div id="map"></div>
  <div id="sections"></div>
</div>
<script>
const DATA = __DATA__;
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent = DATA.subtitle;

const RS = DATA.run_status;
// Helpers are defined at top level: a function declared inside the Leaflet guard is not defined before that block
// runs, which broke the whole page (stat tiles first) with or without Leaflet.
function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function isUrl(v) { return typeof v === 'string' && v.startsWith('https://'); }


const tilesDiv = document.getElementById('tiles');
DATA.sections.forEach(sec => {
  const color = dark ? sec.colorDark : sec.colorLight;
  const bad = (sec.counts.FAIL || 0) + (sec.counts.UNKNOWN || 0) > 0;
  const cls = sec.green ? 'ok' : (bad ? 'bad' : 'na');
  const skipped = sec.n_skipped ? `, ${sec.n_skipped.toLocaleString()} skipped` : '';
  tilesDiv.insertAdjacentHTML('beforeend', `
    <div class="tile${cls === 'na' ? ' grey' : ''}">
      <div class="label"><span class="swatch" style="background:${color}"></span>${esc(sec.title)}</div>
      <div class="value ${cls}">${sec.n_failures.toLocaleString()}</div>
      <div class="sub">flagged of ${sec.n_checked.toLocaleString()} ${esc(sec.unit)} checked${skipped}</div>
      <div class="sub">${esc(sec.state)} &middot; universe ${sec.n_universe.toLocaleString()} &middot; ${esc(sec.verdict)}</div>
    </div>`);
});

const greyTiles = (RS.failed_check && !DATA.sections.some(s => s.key === RS.failed_check) ? [[RS.failed_check, 'ERROR']] : [])
  .concat(RS.not_run.map(k => [k, 'NOT_RUN']));
greyTiles.forEach(([key, state]) => {
  tilesDiv.insertAdjacentHTML('beforeend', `
    <div class="tile grey">
      <div class="label">${esc(key)}</div>
      <div class="value na">${state === 'ERROR' ? 'error' : 'not run'}</div>
      <div class="sub">${state}</div>
    </div>`);
});

function toCsv(columns, records) {
  const q = v => { const s = String(v ?? ''); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [columns.map(q).join(',')].concat(records.map(r => columns.map(c => q(r[c])).join(','))).join('\n');
}

const RENDER_LIMIT = 1000;
const sectionsDiv = document.getElementById('sections');
DATA.sections.forEach((sec, i) => {
  const color = dark ? sec.colorDark : sec.colorLight;
  const el = document.createElement('div');
  el.className = 'section';
  const summaryHtml = sec.summary ? `
    <div class="summary-tbl">
      <h3>${esc(sec.summary.title)}</h3>
      <div class="tbl-wrap"><table>
        <thead><tr>${sec.summary.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
        <tbody>${sec.summary.records.map(r =>
          `<tr data-filter="${esc(r[sec.summary.columns[0]])}">${sec.summary.columns.map(c => `<td>${esc(r[c])}</td>`).join('')}</tr>`
        ).join('')}</tbody>
      </table></div>
    </div>` : '';
  el.innerHTML = `
    <h2><span class="swatch" style="background:${color};width:11px;height:11px;border-radius:3px;display:inline-block"></span>${esc(sec.title)}</h2>
    <div class="desc">${esc(sec.description)}${sec.truncated ? ' (table truncated; full results in the CSV written alongside this report)' : ''}</div>
    <div class="desc"><b>${esc(sec.state)}</b> &middot; ${sec.n_checked.toLocaleString()} of ${sec.n_universe.toLocaleString()} ${esc(sec.unit)} checked, ${sec.n_skipped.toLocaleString()} skipped &middot; ${esc(sec.verdict)}${sec.note ? ' &middot; ' + esc(sec.note) : ''}</div>
    ${sec.records.length === 0 ? `<div class="${sec.green ? 'empty' : 'empty na'}">${esc(sec.verdict)}</div>` : `
    <div class="controls">
      <input type="search" placeholder="Filter rows&hellip;" id="filter-${i}">
      <button id="csv-${i}">Download CSV</button>
      <span class="count" id="count-${i}"></span>
    </div>
    <div class="tbl-wrap"><table id="tbl-${i}"></table></div>
    ${summaryHtml}`}
  `;
  sectionsDiv.appendChild(el);
  if (!sec.records.length) return;

  const sort = {col: null, dir: 1};
  const cmp = (a, b) => {
    const va = a[sort.col], vb = b[sort.col];
    if (va === null || va === undefined || va === '') return 1;
    if (vb === null || vb === undefined || vb === '') return -1;
    const na = Number(va), nb = Number(vb);
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return (na - nb) * sort.dir;
    return String(va).localeCompare(String(vb)) * sort.dir;
  };
  let filterText = '';
  const render = () => {
    const f = filterText.toLowerCase();
    let rows = f ? sec.records.filter(r => sec.columns.some(c => String(r[c] ?? '').toLowerCase().includes(f)))
                 : sec.records.slice();
    if (sort.col !== null) rows = rows.slice().sort(cmp);
    const shown = rows.slice(0, RENDER_LIMIT);
    const arrow = c => sort.col === c ? (sort.dir === 1 ? ' ▲' : ' ▼') : '';
    let html = '<thead><tr>' + sec.columns.map(c => `<th data-col="${esc(c)}">${esc(c)}${arrow(c)}</th>`).join('') + '</tr></thead><tbody>';
    html += shown.map(r => '<tr>' + sec.columns.map(c => {
      const v = r[c];
      return isUrl(v) ? `<td><a href="${esc(v)}" target="_blank">link</a></td>` : `<td title="${esc(v)}">${esc(v)}</td>`;
    }).join('') + '</tr>').join('');
    const tbl = document.getElementById(`tbl-${i}`);
    tbl.innerHTML = html + '</tbody>';
    tbl.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
      const col = th.dataset.col;
      sort.dir = sort.col === col ? -sort.dir : 1;
      sort.col = col;
      current = render();
    }));
    document.getElementById(`count-${i}`).textContent =
      `showing ${shown.length.toLocaleString()} of ${rows.length.toLocaleString()} rows (click a header to sort)`;
    return rows;
  };
  let current = render();
  const filterInput = document.getElementById(`filter-${i}`);
  filterInput.addEventListener('input', e => { filterText = e.target.value; current = render(); });
  el.querySelectorAll('.summary-tbl tbody tr').forEach(tr => tr.addEventListener('click', () => {
    filterText = tr.dataset.filter;
    filterInput.value = filterText;
    current = render();
  }));
  document.getElementById(`csv-${i}`).addEventListener('click', () => {
    const blob = new Blob([toCsv(sec.columns, current)], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${sec.key}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
});

function showMapUnavailable() {
  document.getElementById('map').innerHTML =
    '<div class="empty" style="padding:16px;color:var(--muted)">Map unavailable (Leaflet could not be loaded). ' +
    'Counts and tables above and below are unaffected.</div>';
}

// The map is drawn last and only if Leaflet loaded, so a blocked CDN never hides counts or tables.
if (window.L) {
  const map = L.map('map', {worldCopyJump: true}).setView([15, 0], 2);
  L.tileLayer(
    // Esri gray canvas: keyless (CARTO basemaps now require an API key).
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_' + (dark ? 'Dark' : 'Light') + '_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    {attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ', maxZoom: 16}
  ).addTo(map);

  const overlays = {};
  let allBounds = null;
  DATA.sections.forEach(sec => {
    const color = dark ? sec.colorDark : sec.colorLight;
    const layer = L.geoJSON(sec.geojson, {
      style: {color: color, weight: 1.5, fillColor: color, fillOpacity: 0.35},
      onEachFeature: (feat, lyr) => {
        const p = feat.properties;
        let html = `<b>${esc(p.mgrs_tile_id)}</b> &mdash; ${esc(sec.title)}<br>${p.n_failures} flagged`;
        p.products.forEach(prod => {
          html += `<div class="prod">`;
          Object.entries(prod).forEach(([k, v]) => {
            if (v === null || v === '' || k === 'mgrs_tile_id') return;
            html += isUrl(v) ? `<div><a href="${esc(v)}" target="_blank">${esc(k)}</a></div>`
                             : `<div><b>${esc(k)}:</b> ${esc(v)}</div>`;
          });
          html += `</div>`;
        });
        if (p.n_failures > p.products.length) html += `<div class="prod">&hellip; ${p.n_failures - p.products.length} more (see table)</div>`;
        lyr.bindPopup(html, {maxWidth: 420});
      },
    });
    if (sec.geojson.features.length) {
      layer.addTo(map);
      const b = layer.getBounds();
      allBounds = allBounds ? allBounds.extend(b) : b;
    }
    overlays[`<span style="color:${color}">&#9632;</span> ${esc(sec.title)} (${sec.n_failures})`] = layer;
  });
  if (allBounds) map.fitBounds(allBounds, {padding: [20, 20]});
  L.control.layers(null, overlays, {collapsed: false}).addTo(map);
} else {
  showMapUnavailable();
}
</script>
</body>
</html>
"""
