// Run a report's inline script under Node with a minimal DOM stub and print what it rendered as JSON.
// Usage: node js_harness.js <script.js> [--leaflet]. Used by tests/jsrender.py.
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const withLeaflet = process.argv.includes('--leaflet');
const elements = {};
function makeEl(id) {
  return {
    id, innerHTML: '', textContent: '', value: '', className: '', dataset: {}, children: [],
    insertAdjacentHTML(pos, html) { this.innerHTML += html; },
    appendChild(child) { this.children.push(child); },
    addEventListener() {},
    querySelectorAll() { return []; },
    click() {},
  };
}
const document = {
  getElementById(id) { return elements[id] || (elements[id] = makeEl(id)); },
  createElement() { return makeEl(null); },
};
const window = { matchMedia: () => ({ matches: false }) };
const mapCalls = [];
if (withLeaflet) {
  const layer = () => ({ addTo() { return this; }, getBounds() { return { extend() { return this; } }; } });
  window.L = {
    map: () => { mapCalls.push('map'); return { setView() { return this; }, fitBounds() {} }; },
    tileLayer: () => layer(),
    geoJSON: (data) => { mapCalls.push(['geoJSON', data.features.length]); return layer(); },
    control: { layers: () => layer() },
  };
}
global.document = document;
global.window = window;
global.L = window.L;
global.Blob = function Blob() {};
global.URL = { createObjectURL() { return ''; }, revokeObjectURL() {} };
let error = null;
try { eval(src); } catch (e) { error = `${e.name}: ${e.message}`; }
const sections = (elements.sections ? elements.sections.children : []).map(el => el.innerHTML);
const out = { error, sections, mapCalls, elements: {} };
for (const [id, el] of Object.entries(elements)) out.elements[id] = { innerHTML: el.innerHTML, textContent: el.textContent };
process.stdout.write(JSON.stringify(out));
