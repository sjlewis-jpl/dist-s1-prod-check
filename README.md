# dist-s1-prod-check

Validation checks for the OPERA DIST-S1 production system, used as its production gate. Cross-references product
metadata (GeoTIFF tags), NASA CMR granule metadata (or a producing cluster's bucket), and the expected enumeration
from [dist-s1-enumerator](https://github.com/opera-adt/dist-s1-enumerator) at global scale.

**New here?** Follow the [quick start](docs/QUICKSTART.md): install, test, a first small run and how to read the
report, step by step.

## Checks

`run-all` runs every check and writes one combined report; each also exists as a standalone command.

1. **Duplicates** (`check-duplicates`): one DIST-S1 version per tile and acquisition.
2. **Processing order** (`check-ordering`): two views, first production over every version, and retained versions.
3. **Confirmation chain** (`check-confirmation`): each product's `prior_dist_s1_product` is the latest version of
   the previous acquisition that existed when the product was processed.
4. **Product inputs** (`check-inputs`): recorded pre/post RTC-S1 inputs equal a re-enumeration
   (`--mode offline` from the cached RTC table; `--mode online` queries ASF per product).
5. **Coverage and orphans** (`check-coverage`): every eligible RTC-S1 pass has a product, and every product a pass.

Contracts, units, every status and reason string, and the known limits are in [docs/CHECKS.md](docs/CHECKS.md).

## Setup

The [quick start](docs/QUICKSTART.md) walks through setup and a first run end to end; this section is the short
version.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if pixi is missing
pixi install
```

Earthdata credentials are required (the DIST-S1 CMR collection and the DAAC GeoTIFFs are
gated): put them in `~/.netrc` (`machine urs.earthdata.nasa.gov login ... password ...`) or
let `earthaccess.login()` prompt once.

## Gate workflow

For a guided first run, including how to read the report and fix common errors, see the
[quick start](docs/QUICKSTART.md).

A gate run downloads fresh metadata, fetches tags, and runs every check over the whole window and AOI:

```bash
pixi shell
nohup dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 \
    --data-dir data --out-dir outputs > run_all.log 2>&1 &
```

Then, before reading any count:

1. Open `outputs/latest`: it names the run directory, `outputs/<stamp>_<6 hex>/`. Each run writes only there, so
   runs never mix their files.
2. Read `run_status.json` in it. Only `"state": "COMPLETE"` means every check ran and every file was written (files
   are written to temporary names and renamed; COMPLETE is written last). `RUNNING` read later means the process
   died; `INCOMPLETE` names `failed_check`, `error` and the checks `not_run`. The HTML report of an incomplete run
   opens with a red INCOMPLETE banner and grey tiles for the checks that did not finish.
3. Read the report's **provenance block** (generated only from `run_manifest.json`): command line, tool and
   enumerator versions, venue, window and AOI, product versions and exclusions, processing and window filters,
   sampling (if any), RTC-S1 rows dropped building the input frame, every cached table with its provenance, and what
   is not checked (layer completeness). An orange **ADOPTED CACHE** banner means the run used `adopt-cache`d tables
   and cannot serve as a gate.
4. Read each section: "N failed of M checked (unit)" with the per-status counts. A section is green only when it is
   COMPLETE, has no FAIL and no UNKNOWN, and evaluated something. UNKNOWN is never green.

**Never use sampling for the gate.** `--sample-tiles N` (every check on N random tiles, full time series) and
`--sample-inputs N` (confirmation and inputs on N products) are for fast iteration. Sampled sections are labelled
SAMPLED with the sample size, seed and a digest of the selected ids in the provenance. `--sample-tiles` can never
select a tile with no DIST-S1 products, so it cannot show a tile the system skipped entirely.

## Caches and manifests

Every cached metadata table has a manifest next to it (`<table>.manifest.json`): venue, source, acquisition window,
AOI, every query window with its completeness, per-chunk CMR hits and rows, query time, row count and the parquet's
sha256 (checked on every load). A run refuses a cache that does not match its request (venue, window not inside the
cached one, AOI, incomplete query, changed parquet, no manifest) and says how to fix it. A missing `venue.txt` next
to an existing cache is refused, never taken as PROD.

- **`--refresh`** re-downloads both metadata tables and deletes the inputs checkpoint and the RTC chunk
  checkpoints. `download-metadata` also deletes the inputs checkpoint.
- **RTC chunk checkpoints** (`rtc_s1_chunks*/`, removed once the table is written) are reused on resume only if
  their manifest matches the query and the chunk's acquisitions ended at least 30 days before it was queried.
- **Tags** (`dist_s1_tags.parquet`) record the URL each row was read from; a row is reused only if that URL is still
  the one the product resolves to (`alert_url` or `--tag-url-template`). Rows with a read error are retried, except
  with `--offline`.
- **Inputs checkpoint** (`inputs_check_results.parquet`): each row stores a fingerprint of the RTC table, the
  product's tag row, tool and enumerator versions and options; anything that changes recomputes it. Rows that could
  not be checked are never checkpointed.
- **Schema validation** on load refuses null acquisition or processing times, duplicate ids and wrong dtypes.
- **`--offline`** uses cached data only: any step needing the network fails naming the step; cached tag errors stay
  UNKNOWN "(not retried: offline)".
- **`adopt-cache`** writes `adopted_unverified` manifests (and asserts each tag row's `source_url` and the venue) for
  a cache built before manifests existed. Such caches are accepted only with `--regression-baseline`, never for a
  gate report.

## Product versions and campaigns

Only DIST-S1 v1.0 products are checked by default (`--product-versions v1.0`; a comma list widens it). Excluded
products are counted per version in the report header. Within a version, deduplication keeps the last-processed
product of each tile and acquisition (ties: greatest id).

**UAT venue** (DIST-S1 products delivered to ASF UAT, e.g. from a test-venue replay):

```bash
dist-s1-prod-check run-all --venue UAT --start 2026-01-01 --stop 2026-03-01 \
    --processing-start 2026-09-24 --data-dir data_uat --out-dir outputs_uat
```

A venue can hold several processing campaigns of the same acquisitions (older ones are not deleted).
`--processing-start` / `--processing-stop` restrict every check to products processed in that range;
otherwise de-duplication keeps the latest product per tile/acquisition across campaigns.

Only DIST-S1 metadata and GeoTIFF tags come from UAT; RTC-S1 inputs are still read from PROD CMR.
Requires a `machine uat.urs.earthdata.nasa.gov` entry in `~/.netrc`. Each data dir is tied to one venue
(`venue.txt`), since PROD and UAT can hold products with the same ids. If the UAT run did not cover the
whole globe, pass `--bbox` so the coverage check does not flag every unprocessed tile.

**OPS-POP1 venue** (DIST-S1 products still in the producing cluster's bucket, e.g. before the ASF
collection is visible to your Earthdata account):

```bash
dist-s1-prod-check run-all --venue OPS-POP1 --start 2026-01-01 --stop 2026-03-01 \
    --data-dir data_pop1 --out-dir outputs_pop1
```

DIST-S1 metadata comes from listing `s3://opera-ops-rs-pop1/products/DIST_S1/{acq YYYY/MM/DD}/`
anonymously (every key, so a product whose STATUS layer is missing is seen and reported UNKNOWN; no CMR or
Earthdata login), and GeoTIFF tags are read from the same bucket over HTTPS. The bucket holds what the cluster
produced, which may differ from what was delivered to ASF. RTC-S1 inputs still come from PROD CMR.

## Fast iteration

```bash
dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 \
    --data-dir data_global --out-dir outputs_sample --sample-tiles 100
```

Same `--seed` (default 42) means the same tiles every run. Tags fetched for a sample land in the shared cache.

Over an AOI: `--bbox lon_min lat_min lon_max lat_max` (resolved to the overlapping MGRS tiles; the RTC-S1 query is
widened to their full burst footprints; antimeridian-crossing AOIs are not handled).

Individual checks after `download-metadata` / `fetch-tags`:

```bash
dist-s1-prod-check download-metadata --start 2026-01-01 --stop 2026-03-01
dist-s1-prod-check fetch-tags --workers 16
dist-s1-prod-check check-duplicates
dist-s1-prod-check check-ordering
dist-s1-prod-check check-confirmation
dist-s1-prod-check check-inputs            # offline, uses the RTC table
dist-s1-prod-check check-inputs --mode online --sample 100   # live ASF
dist-s1-prod-check check-coverage --start 2026-01-01 --stop 2026-03-01
```

## Outputs

Per run directory: one CSV per result table (`<name>_<stamp>.csv`, e.g. `confirmation`, `inputs`, `coverage`, their
`_failures`/`_missing` subsets, `ordering_tiles`, `coverage_orphans`), `report_<stamp>.html`, `run_manifest.json`
and `run_status.json`. The report embeds its data (tables up to 50,000 rows each, labelled when truncated; the map
counts every failure, with rows that have no tile counted in the legend) and loads Leaflet and basemap tiles from
CDNs; without them the map says so and everything else still renders. `dist_s1_prod_check.reconcile.reconcile_run`
checks that a run's report and CSVs agree.

## Memory at global scale

CMR pages are parsed and discarded as they stream in (never accumulated as raw JSON), the
inputs check enumerates and compares one MGRS tile at a time, and coverage groups are
prefiltered to the check window. Measured: 680 B/row for the RTC table, so the global Jan-Mar
2026 run peaks around 12-16 GB (RTC table ~5 GB + LUT-joined frame ~7 GB + bounded transients).

## Development

Tests: `pytest -q` (offline; the `data` tests use the real-data subset in `tests/data/`; the report-rendering tests
need `node`). Lint: `ruff check . && ruff format --check .`.
