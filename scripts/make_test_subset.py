"""Cut a 20-tile real-data regression subset out of the frozen UAT and pop1 baseline caches.

Usage::

    make_test_subset.py [--data-root DIR] [--tests-data DIR] [--max-tests-bytes N]

For each venue (``uat``, ``pop1``) the script reads the read-only frozen cache at
``<data-root>/hardening_review/frozen/<venue>/`` (after verifying it against ``SHA256SUMS``) and writes
``<data-root>/hardening_review/subset/<venue>/`` holding, for the tiles in ``TILES``:

- ``dist_s1_products.parquet``: every DIST-S1 product (all versions, all processing times) in the tiles;
- ``dist_s1_tags.parquet``: every cached tag row of those products, read errors included;
- ``rtc_s1_products.parquet``: every RTC-S1 row of every burst that the enumerator's MGRS/burst LUT maps to the
  tiles, over the full cached period (post window and the 1-3 year lookback), so coverage and inputs see exactly
  the bursts and acquisitions the full run saw for these tiles;
- ``venue.txt`` copied from the cache, and ``target_mgrs_tiles.txt`` listing the tiles. The frozen caches have no
  ``target_mgrs_tiles.txt`` (the baselines were global runs); writing one restricts the RTC frame to the subset
  tiles, which is what the global run computed for them, since every check is per tile;
- ``subset_manifest.json``: tile choices and reasons, source sha256 sums, row counts and output sha256 sums.

``inputs_check_results.parquet`` is deliberately not copied, so a run on the subset recomputes the inputs check.
If both subsets together are smaller than ``--max-tests-bytes`` (5 MB), they are also copied to
``<tests-data>/<venue>/``.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids


DEFAULT_DATA_ROOT = Path('~/data/opera/dist-s1-prod-check').expanduser()
DEFAULT_TESTS_DATA = Path(__file__).resolve().parents[1] / 'tests' / 'data'
VENUES = ('uat', 'pop1')
MAX_TESTS_BYTES = 5 * 1024 * 1024

DIST_S1_PARQUET = 'dist_s1_products.parquet'
RTC_S1_PARQUET = 'rtc_s1_products.parquet'
TAGS_PARQUET = 'dist_s1_tags.parquet'
VENUE_FILE = 'venue.txt'
TARGET_TILES_FILE = 'target_mgrs_tiles.txt'
MANIFEST_FILE = 'subset_manifest.json'

# Tile -> why it is in the subset. Counts are from the baseline outputs (outputs_uat 20261005T202425,
# outputs_pop1 20261013T152653); "both" means both venues.
TILES: dict[str, str] = {
    '13TEG': 'Review R2 (pop1): confirmed-OK product whose prior names a superseded version; pop1 duplicates.',
    '18STD': 'Review R3 (pop1): May products processed before their predecessors, hidden by the reprocessed '
    'exclusion in ordering; 7 pop1 duplicate rows.',
    '56WMT': 'Review R3 (pop1): 3 hidden out-of-order cases; acquisitions with 3 versions (13 pop1 duplicate rows).',
    '27XVK': 'Polar (Greenland) tile with double matches: 12 coverage rows share a matched product (both); F06.',
    '24XWL': 'Second polar double-match tile: 10 coverage rows share a matched product (both); F06.',
    '11SNA': 'UAT tag read error (HTTP 404) on one product; also coverage missing, no-baseline and pre-input issues.',
    '34UDV': 'UAT tag read error (HTTP 404); UAT duplicates; 25 UAT coverage-missing passes.',
    '01UCT': 'pop1 confirmation failure (prior names the wrong product); UTM zone 1 next to the antimeridian.',
    '12SXF': 'Burst ID mismatch, post RTC expected-but-not-found and pre issues in both venues.',
    '35WPN': 'Pre RTC IDs found but not expected (both venues); no-baseline coverage rows.',
    '34WDA': 'Largest UAT coverage-missing tile (52 passes).',
    '34WDC': 'Largest pop1 coverage-missing tile (29 passes).',
    '37MBV': 'UAT acquisition with 3 versions (duplicates); no-baseline coverage rows.',
    '22WEA': 'pop1 acquisition with 3 versions; pre-input issues in both venues.',
    '21HUU': 'Southern hemisphere; burst ID mismatch and post/pre input issues in both venues.',
    '60GTA': 'UTM zone 60, southern hemisphere, next to the antimeridian; 13 pop1 coverage-missing passes.',
    '01WCQ': 'No DIST-S1 product in either venue: coverage rows only, all "no baseline imagery".',
    '11SLT': 'Inside the pop1 AOI baseline bbox (-120 34 -118 36), so the AOI and subset runs overlap.',
    '33VWH': 'Control tile: no flagged row in any check in either venue (high latitude, many products).',
    '47QKB': 'Control tile: no flagged row in any check in either venue (tropics).',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen(frozen_root: Path, venue: str, names: list[str]) -> dict[str, str]:
    """Check the frozen files against ``SHA256SUMS`` and return their digests.

    Parameters
    ----------
    frozen_root : Path
        Directory holding ``SHA256SUMS`` and one subdirectory per frozen cache.
    venue : str
        Frozen cache name, e.g. ``'uat'``.
    names : list[str]
        File names inside the cache to verify.

    Returns
    -------
    dict[str, str]
        ``{file name: sha256}`` for each verified file.
    """
    expected = {}
    for line in (frozen_root / 'SHA256SUMS').read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            expected[rel.strip()] = digest
    digests = {}
    for name in names:
        rel = f'{venue}/{name}'
        if rel not in expected:
            raise SystemExit(f'{rel} is not listed in SHA256SUMS')
        digest = _sha256(frozen_root / rel)
        if digest != expected[rel]:
            raise SystemExit(f'Frozen cache changed: {rel} no longer matches SHA256SUMS')
        digests[name] = digest
    return digests


def burst_ids_for_tiles(tiles: list[str]) -> list[str]:
    """Return every JPL burst id that the enumerator's LUT maps to any of ``tiles``.

    Parameters
    ----------
    tiles : list[str]
        MGRS tile ids without the ``T`` prefix.

    Returns
    -------
    list[str]
        Sorted unique burst ids.
    """
    return sorted(get_lut_by_mgrs_tile_ids(tiles).jpl_burst_id.unique())


def make_subset(frozen_dir: Path, out_dir: Path, tiles: list[str], source_sha256: dict[str, str]) -> dict:
    """Write the subset of one frozen cache and return its manifest.

    Parameters
    ----------
    frozen_dir : Path
        Frozen cache directory (read only).
    out_dir : Path
        Subset directory; replaced if it exists.
    tiles : list[str]
        MGRS tile ids to keep.
    source_sha256 : dict[str, str]
        Verified digests of the source files, recorded in the manifest.

    Returns
    -------
    dict
        The manifest written to ``subset_manifest.json``.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    df_dist = gpd.read_parquet(frozen_dir / DIST_S1_PARQUET)
    df_dist = df_dist[df_dist.mgrs_tile_id.isin(tiles)].reset_index(drop=True)
    df_dist.to_parquet(out_dir / DIST_S1_PARQUET, compression='zstd')

    df_tags = pd.read_parquet(frozen_dir / TAGS_PARQUET)
    df_tags = df_tags[df_tags.opera_id.isin(df_dist.opera_id)].reset_index(drop=True)
    df_tags.to_parquet(out_dir / TAGS_PARQUET, compression='zstd')

    burst_ids = burst_ids_for_tiles(tiles)
    df_rtc = gpd.read_parquet(frozen_dir / RTC_S1_PARQUET, filters=[('jpl_burst_id', 'in', burst_ids)])
    df_rtc = df_rtc.reset_index(drop=True)
    df_rtc.to_parquet(out_dir / RTC_S1_PARQUET, compression='zstd')

    shutil.copyfile(frozen_dir / VENUE_FILE, out_dir / VENUE_FILE)
    (out_dir / TARGET_TILES_FILE).write_text('\n'.join(tiles))

    outputs = [DIST_S1_PARQUET, TAGS_PARQUET, RTC_S1_PARQUET, VENUE_FILE, TARGET_TILES_FILE]
    manifest = {
        'source_cache': f'hardening_review/frozen/{frozen_dir.name}',
        'source_sha256': source_sha256,
        'tiles': {tile: TILES[tile] for tile in tiles},
        'n_burst_ids': len(burst_ids),
        'rows': {
            DIST_S1_PARQUET: len(df_dist),
            TAGS_PARQUET: len(df_tags),
            'tags_with_error': int((df_tags.error != '').sum()),
            RTC_S1_PARQUET: len(df_rtc),
        },
        'rtc_acq_dt_range': [str(df_rtc.acq_dt.min()), str(df_rtc.acq_dt.max())],
        'output_sha256': {name: _sha256(out_dir / name) for name in outputs},
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def main(argv: list[str] | None = None) -> int:
    """Build both subsets and copy them into the test tree when small enough.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--tests-data', type=Path, default=DEFAULT_TESTS_DATA)
    parser.add_argument('--max-tests-bytes', type=int, default=MAX_TESTS_BYTES)
    opts = parser.parse_args(argv)

    frozen_root = opts.data_root / 'hardening_review' / 'frozen'
    subset_root = opts.data_root / 'hardening_review' / 'subset'
    tiles = sorted(TILES)
    for venue in VENUES:
        digests = verify_frozen(frozen_root, venue, [DIST_S1_PARQUET, TAGS_PARQUET, RTC_S1_PARQUET, VENUE_FILE])
        manifest = make_subset(frozen_root / venue, subset_root / venue, tiles, digests)
        print(f'{venue}: {json.dumps(manifest["rows"])} -> {subset_root / venue}')

    total = sum(_dir_bytes(subset_root / venue) for venue in VENUES)
    print(f'subset size: {total / 1e6:.2f} MB')
    if total < opts.max_tests_bytes:
        for venue in VENUES:
            dest = opts.tests_data / venue
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(subset_root / venue, dest)
        print(f'copied to {opts.tests_data}')
    else:
        print(f'over {opts.max_tests_bytes / 1e6:.1f} MB: tests read {subset_root} and skip when it is absent')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
