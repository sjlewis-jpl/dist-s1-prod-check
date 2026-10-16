# Real-data regression subset

A 20-MGRS-tile cut of the UAT and pop1 baseline caches (hardening plan, Task 0.4). The test that uses it is
`tests/test_end_to_end.py::test_subset_baseline` (marker `data`).

```
tests/data/
  uat/   pop1/                 one data dir per venue; never mix them (product ids can collide across venues)
    dist_s1_products.parquet   every DIST-S1 product in the 20 tiles: all versions, all processing times
    dist_s1_tags.parquet       every cached tag row for those products, read-error rows included
    rtc_s1_products.parquet    every RTC-S1 row of every burst the enumerator LUT maps to the 20 tiles,
                               over the whole cached period (2022-11-03 onwards, so the 1-3 year lookback is in)
    venue.txt                  copied from the cache (UAT / OPS-POP1)
    target_mgrs_tiles.txt      the 20 tiles (written by the script; see below)
    subset_manifest.json       tiles and reasons, source sha256, row counts, output sha256
  baseline_counts.json         mirror of the reference counts (see "Reference counts")
```

`inputs_check_results.parquet` is not included, so every run recomputes the inputs check.

## Provenance

Built by `scripts/make_test_subset.py` from the read-only frozen copies of the Task 0.6 baseline caches,
`~/data/opera/dist-s1-prod-check/hardening_review/frozen/{uat,pop1}/`. The script verifies each source file against
`hardening_review/frozen/SHA256SUMS` before reading it:

| File | uat sha256 | pop1 sha256 |
|---|---|---|
| `dist_s1_products.parquet` | `fb7d93db2886ff54d5a3cb5d68bdeb4f8d6826403080e3bd600de95a22a3ff68` | `52068a7d927da787ef10bfd11cd5a2f79c6913004271459bbe8b264da9cddd9f` |
| `dist_s1_tags.parquet` | `718c68f87ead0aa6fd409c624fc9f6cd54abfd681057d47959dd2ced9823cb68` | `2f06b9c0c242190eb9c581eea0ef50c0ee906f91a273477e14aea9f9dd082a0b` |
| `rtc_s1_products.parquet` | `b53285199f383876faa2600079365b28bc56dc9e18f78c718a0ed10d6399a542` | same file (PROD CMR, fetched 2026-10-05) |

The baselines these caches produced are `outputs_uat/` (stamp 20261005T202425) and `outputs_pop1/`
(20261013T152653); see `../docs/prod-gate-hardening/baselines.json`.

Rows: uat 1,084 products, 582 tag rows (2 with an HTTP 404 read error), 34,459 RTC rows (1,682 bursts); pop1 632
products, 618 tag rows (no errors), the same 34,459 RTC rows. Total 3.0 MB, under the 5 MB limit, so the subset lives
here; a copy is also at `hardening_review/subset/{uat,pop1}/`.

The frozen caches have no `target_mgrs_tiles.txt`: both baselines were global runs. Writing one with the 20 tiles
restricts the RTC frame to them. Every check is per tile (duplicates per acquisition, ordering/confirmation/inputs/
coverage per tile), and the RTC table holds every acquisition of every burst in those tiles, so the subset run
computes the same rows the global run computed for these tiles.

### Tiles

The same 20 tiles are used for both venues. Counts come from the baseline outputs.

| Tile | Why |
|---|---|
| 13TEG | Review R2 (pop1): a confirmed-OK product whose prior names a superseded version; pop1 duplicates |
| 18STD | Review R3 (pop1): May products processed before their predecessors, hidden by the reprocessed exclusion |
| 56WMT | Review R3 (pop1): 3 hidden out-of-order cases; acquisitions with 3 versions |
| 27XVK | Polar (Greenland), double matches: 12 coverage rows share a matched product (both venues; F06) |
| 24XWL | Second polar double-match tile: 10 coverage rows share a matched product (both venues) |
| 11SNA | UAT tag read error (HTTP 404); coverage missing, no-baseline and pre-input issues |
| 34UDV | UAT tag read error (HTTP 404); UAT duplicates; 25 UAT coverage-missing passes |
| 01UCT | pop1 confirmation failure (prior names the wrong product); UTM zone 1, next to the antimeridian |
| 12SXF | Burst ID mismatch, post expected-but-not-found and pre issues (both venues) |
| 35WPN | Pre RTC IDs found but not expected (both venues) |
| 34WDA | Largest UAT coverage-missing tile (52 passes) |
| 34WDC | Largest pop1 coverage-missing tile (29 passes) |
| 37MBV | UAT acquisition with 3 versions |
| 22WEA | pop1 acquisition with 3 versions; pre-input issues |
| 21HUU | Southern hemisphere; burst ID mismatch, post and pre input issues (both venues) |
| 60GTA | UTM zone 60, southern hemisphere, next to the antimeridian; 13 pop1 coverage-missing passes |
| 01WCQ | No DIST-S1 product in either venue: coverage rows only, all "no baseline imagery" |
| 11SLT | Inside the pop1 AOI baseline bbox (-120 34 -118 36) |
| 33VWH | Control: nothing flagged in any check in either venue (high latitude) |
| 47QKB | Control: nothing flagged in any check in either venue (tropics) |

The reasons are also in `TILES` in `scripts/make_test_subset.py` and in each `subset_manifest.json`.

### Regenerating

```
~/miniforge3/envs/dist-s1-prod-check/bin/python scripts/make_test_subset.py
```

It rewrites `hardening_review/subset/{uat,pop1}/` and, if the total is under 5 MB, `tests/data/{uat,pop1}/`. It never
touches the frozen copies or the original caches. Changing `TILES` changes the reference counts: delete
`hardening_review/subset/baseline_counts.json` and `tests/data/baseline_counts.json`, then rerun the test to record new
ones.

## Anonymisation

The data is not anonymised, and does not need to be. It holds only public product metadata: OPERA product and burst
ids, acquisition and processing times, footprints, public DAAC/bucket URLs and GeoTIFF tag values. It holds no user
names, credentials, cookies or tokens. `subset_manifest.json` records source paths relative to the data root.

## Reference counts

`test_subset_baseline` copies each venue's subset to `tmp_path` (run-all writes into its data dir), then runs
`run-all` with the venue's original baseline arguments from `baselines.json`: `--venue`, `--start 2026-01-01`,
`--stop 2026-03-01`, and for UAT also `--processing-start 2026-09-24` and the INT-FWD `--tag-url-template`.
`--workers 1 --inputs-workers 1` keep the inputs check in the test process, so the network blocks apply to it.
`test_venue_args_match_baselines` checks that the embedded arguments still match `baselines.json`.

It records the per-section `n_checked`/`n_failures` and the row count of every CSV in
`hardening_review/subset/baseline_counts.json` (mirrored here when the subset is read from `tests/data`). If the
venue is already recorded, the counts must be equal, so the test is a regression check. It also asserts exit code 0
and that no network access was attempted.

At aa2014f the subset counts equal the full baselines' rows for these tiles in every CSV. The only apparent gap is in
UAT `inputs` and `inputs_failures` (2 rows fewer in the baseline when filtered on `mgrs_tile_id`): the two
`Tag read error` rows have an empty `mgrs_tile_id`, because it comes from the unreadable tag. Filtered on `opera_id`
the counts match (600 and 55).

## Tag-retry workaround (until `--offline`, Task 1.7)

At aa2014f, `tags.fetch_tags_table(retry_errors=True)` drops cached rows that hold a read error and fetches them
again on every run. The UAT subset has two such rows (HTTP 404), so an unmodified run makes a network attempt, and
`scripts/run_network_blocked.py` exits 2. The test therefore:

- replaces `tags.fetch_tags_table` with the same function bound to `retry_errors=False`, which replays the cached rows
  unchanged (any product without a cached row would still try to fetch and fail);
- patches `requests.Session.request`, `boto3.client`/`resource`, `rasterio.open` and non-loopback
  `socket.connect`/`getaddrinfo` to raise and record the attempt, and asserts that none was recorded.

Task 1.7 adds `--offline`, which keeps error rows as UNKNOWN `(not retried: offline)`. Once it lands, drop the
`fetch_tags_table` patch and pass `--offline`. Any count change at that point must come from how the two error rows are reported,
not from a fetch, since the rows are already in the cache.
