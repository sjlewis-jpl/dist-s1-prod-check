"""Run a written HTML report's inline script under Node (tests/js_harness.js) and return what it rendered."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


HARNESS = Path(__file__).with_name('js_harness.js')


def render(html_path: Path, leaflet: bool = False) -> dict:
    """Return ``{'error', 'sections', 'mapCalls', 'elements'}``; skips the test when node is not installed."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not on PATH; cannot execute the report script')
    html = Path(html_path).read_text()
    start = html.index('<script>\nconst DATA = ') + len('<script>\n')
    script = html[start : html.index('</script>', start)]
    script_path = Path(html_path).with_suffix('.js')
    script_path.write_text(script)
    args = [node, str(HARNESS), str(script_path), *(['--leaflet'] if leaflet else [])]
    result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout)
