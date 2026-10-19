# Quick start

From a fresh clone to a report you can read, in six steps. Background, every option and the gate rules are in
[README.md](../README.md). What each check means is in [CHECKS.md](CHECKS.md).

## 1. Prerequisites

- An **Earthdata login** in `~/.netrc`, for the PROD and UAT venues only. It is not needed for the OPS-POP1 first
  run in step 4. For a UAT run, add the `uat.urs.earthdata.nasa.gov` entry as well:
  ```
  machine urs.earthdata.nasa.gov login <user> password <password>
  ```
- **Memory**: 16 GB for a global run. A small AOI run needs very little.
- **Disk**: ~1 GB per global data directory. Keep `--data-dir` and `--out-dir` off synced folders (OneDrive,
  Dropbox): the caches are large and rewritten often.

## 2. Install

```bash
git clone https://github.com/OPERA-Cal-Val/dist-s1-prod-check.git
cd dist-s1-prod-check
curl -fsSL https://pixi.sh/install.sh | bash   # only if pixi is missing
pixi install
pixi shell                                     # every command below runs inside this shell
```

## 3. Check the install (offline, ~2 min)

```bash
pytest -q
```

All tests should pass; the report-rendering tests skip if `node` is not installed. They need no network and no
credentials.

## 4. First run: a small AOI (about 10 minutes)

**Without Earthdata credentials (OPS-POP1 bucket):**

```bash
dist-s1-prod-check run-all --venue OPS-POP1 --start 2026-01-01 --stop 2026-03-01 \
    --bbox -120 34 -118 36 --data-dir ~/dsp/data_aoi --out-dir ~/dsp/outputs_aoi
```

This reads the DIST-S1 metadata by listing `s3://opera-ops-rs-pop1/products/DIST_S1/` anonymously, one acquisition
day at a time, and reads the product tags from the same bucket. RTC-S1 metadata comes from PROD CMR, which needs no
login for a search. Checked with no `~/.netrc`: COMPLETE in about 10 minutes, 388 products over 12 MGRS tiles. The
bucket holds what the producing cluster made, which may differ from what was delivered to ASF.

**With Earthdata credentials (ASF PROD, the default venue):**

```bash
dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 \
    --bbox -120 34 -118 36 --data-dir ~/dsp/data_aoi_prod --out-dir ~/dsp/outputs_aoi
```

This logs in to Earthdata to query the DIST-S1 collection in CMR. Without a `~/.netrc` entry it stops with an
Earthdata login error.

Either way, the run downloads metadata for the 12 MGRS tiles under the box, reads the product tags, runs every
check and writes one run directory. Each venue needs its own `--data-dir`, because a data dir is tied to the venue
that built it. Re-running the same command reuses the cache. Add `--refresh` to download again.

## 5. Read the result

```bash
cat ~/dsp/outputs_aoi/latest                         # names the run directory
cat ~/dsp/outputs_aoi/<run dir>/run_status.json      # must say "state": "COMPLETE"
open ~/dsp/outputs_aoi/<run dir>/report_*.html       # xdg-open on Linux
```

Read the report in this order:

1. **Banners.** A red INCOMPLETE banner means the run did not finish, so don't trust any count. An orange ADOPTED
   CACHE banner means the run used an adopted cache and the report can't serve as a gate.
2. **Provenance block.** Check the venue, window, AOI and product versions it lists against what you meant to run.
3. **Sections.** A section is green only if it finished, has no FAIL and no UNKNOWN, and checked at least one item.
   UNKNOWN means "couldn't check", never "fine".

The per-row results are in the CSVs in the same directory: `confirmation_*.csv`, `inputs_*.csv`,
`coverage_*.csv`, and so on.

## 6. A gate run (global, hours)

```bash
nohup dist-s1-prod-check run-all --start 2026-01-01 --stop 2026-03-01 \
    --data-dir ~/dsp/data --out-dir ~/dsp/outputs > ~/dsp/run_all.log 2>&1 &
tail -f ~/dsp/run_all.log
```

For a gate run, use no `--sample-*` options and no adopted caches, and only accept it if `run_status.json` says
COMPLETE.

For other venues:

- **ASF UAT:** add `--venue UAT --processing-start <campaign start>` and a `--bbox` if the campaign wasn't global.
- **Producing cluster's bucket:** add `--venue OPS-POP1`; no Earthdata login is needed for the DIST-S1 side.

Each venue needs its own `--data-dir`.

## Common problems

| Symptom | Cause and fix |
|---|---|
| Error saying the cache doesn't match the request (window, AOI, venue, manifest) | The data dir was built for a different run. Use a new `--data-dir`, or add `--refresh`. |
| `venue.txt` missing or doesn't match `--venue` | Each data dir belongs to one venue. Use a separate `--data-dir` per venue. |
| 401/403, a login prompt, or "Earthdata login failed" | `~/.netrc` entry missing or wrong (UAT needs its own `uat.` entry). Without credentials, use `--venue OPS-POP1` (step 4). |
| Run killed, no report, or `RUNNING` in `run_status.json` | Usually out of memory on a global run. Use a machine with 16 GB+, or test with `--bbox` / `--sample-tiles 100` first. |
| Many UNKNOWN "Tag read error" rows | Product GeoTIFFs couldn't be read (auth, network, or missing). Re-running retries them, unless you pass `--offline`. |
| `--offline` fails naming a step | That step needs data that isn't cached yet. Run once without `--offline`. |
