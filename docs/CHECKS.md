# Checks

What each check asserts, over which universe and in which unit, every status and reason it can report, and its
known limits. The result model is shared:

| Status | Meaning |
|---|---|
| PASS | checked, correct |
| FAIL | checked, wrong |
| INFO | checked, noteworthy but not a failure (stale chain, RTC processing version differs); never fails a unit |
| UNKNOWN | could not be checked (missing or unreadable tag, incomplete lookback, ambiguous match, orphan) |
| SKIPPED | deliberately not checked (first acquisition in the window, not evaluated in a view, no baseline) |

A section is **green** only if its state is COMPLETE, it has 0 FAIL and 0 UNKNOWN units, and it evaluated at least
one unit (`n_checked` = PASS + FAIL + INFO > 0). A COMPLETE section with nothing evaluated reads "not evaluated:
insufficient evidence" (grey). Section states: COMPLETE, SAMPLED (ran on a sample; never use sampling for the gate),
EMPTY (empty universe), ERROR / NOT_RUN (the run crashed; see `run_status.json`). Every section reports `unit`,
`n_universe`, `n_checked`, `n_failed`, `n_unknown`, `n_skipped`, per-status `counts`, and `n_issues` (failure-table
rows; a unit can have several). The HTML counts, map and CSVs reconcile (`dist_s1_prod_check.reconcile`).

**Product universe.** Every check runs on DIST-S1 products of the allowed versions (`--product-versions`, default
`v1.0`; excluded counts are in the report header; decision 6) acquired in `[--start, --stop)` (an exact filter, so
products CMR matched at the window edges are excluded and counted), optionally restricted to a processing-time
window (`--processing-start/--processing-stop`) and an AOI (`--bbox`). Version context (how many versions an
acquisition has) is computed before the processing filter (decision 1, F18). Deduplication keeps the last-processed
version of each tile and acquisition (ties: greatest `opera_id`) as the product under test; it never falls back to
an older, readable version.

**Limits shared by every check.**

- Processing time (from the product id) is used as the proxy for when a product became available (decision 3).
- Layer completeness (GEN-METRIC, GEN-DIST-STATUS-ACQ) is not checked (decision 5); only the STATUS layer's tags
  are read.
- A CMR query that returned as many rows as `CMR-Hits` was paged to the end; that does not prove the archive itself
  is complete.
- Sampling is not for the gate: `--sample-tiles` draws only from tiles that have DIST-S1 products, so a tile with
  no products at all can never be sampled and its missing passes are never seen (F29).
- AOI runs (`--bbox`) query DIST-S1 CMR with the raw bbox, not the MGRS tile geometry (F32, speculative); the
  RTC-S1 query is widened to the tiles' full burst footprints.
- Network paths (CMR paging and response validation, chunk manifests, the OPS-POP1 S3 listing, live tag reads and
  retries, online inputs mode) are validated with mocked responses only, not against a live venue.

## 1. Duplicates (`duplicates`)

- **Asserts:** each tile and acquisition time has exactly one DIST-S1 version.
- **Universe and unit:** every acquisition in the universe (all versions, before deduplication); unit **duplicate
  groups**. `n_checked` = acquisitions, FAIL = acquisitions with more than one version; `n_groups`, `n_versions`
  (rows listed), `n_superseded`, `n_products`.
- **Statuses:** FAIL per duplicate group. The CSV lists every version with `is_latest`, `latest_opera_id`,
  `reprocessing_lag_hours` and reason "N granules share acquisition …; latest, kept" or "…; superseded by <id>".
- **Limits:** grouping is by `opera_dedup_id`, which contains the acquisition time to the second. Versions of the
  same pass whose recorded acquisition time differs by a few seconds are not grouped (127 such products exist in UAT
  outside the September campaign; decision D11). They surface as orphans in coverage.

## 2. Processing order (`ordering_first_production`, `ordering`)

Two views (decision 2):

- **First production** (`ordering_first_production`): each acquisition's *earliest* version, over every version in
  the window (before the processing filter), compared with the earliest version of the previous acquisition in the
  tile. Every acquisition but the first of its tile is evaluated.
- **Retained** (`ordering`): the retained (latest) version of each acquisition compared with the previous evaluated
  acquisition in the tile. Reprocessed acquisitions (more than one version, counted over the unfiltered table) and
  the first acquisition of a tile are not evaluated; per-tile counts are in `ordering_tiles_<stamp>.csv`
  (`n_evaluated`, `n_not_evaluated`, `n_reprocessed`, `n_failures`, `n_ties`).
- **Unit:** products (acquisitions).
- **Statuses:** FAIL "processed X h before <the previous acquisition / its first production> (<id>)"; PASS; SKIPPED
  (no predecessor, or reprocessed in the retained view). Equal processing times are in order and counted as `tie`
  (F36).

## 3. Confirmation chain (`confirmation`)

- **Asserts:** a product's `prior_dist_s1_product` tag names the latest version of the previous acquisition in its
  tile that existed when the product was processed (decision 3), compared by **full** id against every version.
- **Universe and unit:** the deduplicated products whose tags were read (all of them, or the `--sample-inputs`
  sample); unit **links** (product → prior). Predecessors come from every version in the window (before the
  processing filter).
- **Statuses and reasons:**
  - PASS.
  - FAIL "no prior, but acquisition <A> precedes the product"; "prior <P> is not a version of the previous
    acquisition <A>"; "prior processed after product: …"; "outdated prior: newer version <V> was processed before the
    product"; "prior tag absent: the STATUS layer has no prior_dist_s1_product key" (F17); for the first product in
    the window, "impossible prior <P>: tile …" or "…: acquired at …, not before the product"; with
    `--expect-none-at-start`, "prior <P> given, but the window starts the mission".
  - INFO "stale chain: <V> was reprocessed after the product" (the prior was right when the product was made).
  - UNKNOWN "could not read prior_dist_s1_product tag (<error>)" (read error, `missing_tag_url`, no cached row).
  - SKIPPED "first acquisition of the tile in the window" unless `--expect-none-at-start`.
- **Outputs:** `confirmation_<stamp>.csv` (every product: `prior_used`, `prior_expected` = the ideal prior,
  `status`, `reason`), `confirmation_failures` (FAIL and UNKNOWN), `confirmation_tiles`.
- **Limits:** the previous acquisition is the previous one in time even if it had not been produced yet when the
  product was made (D9); such a product fails here and the inversion shows in the first-production view.

## 4. Product inputs (`inputs`, `inputs_post`, `inputs_pre`, `inputs_other`)

- **Asserts:** the pre and post RTC-S1 ids recorded in a product's tags equal those a re-enumeration with
  dist-s1-enumerator (multi-window lookback 365/730/1095 days, 60-day windows, max 4/3/3 pre-images per burst)
  expects for that product, compared as multisets on acquisition identity (truncated ids; decision 4).
- **Identity (F05):** the expected product is chosen from the product's own id (tile, acquisition time), never from
  its tags; among several candidates the tagged post track breaks ties. Each expected product is claimed by one
  product (the closest; ties by id).
- **Universe and unit:** the deduplicated (or sampled) products; unit **products**; `n_issues` counts issue rows.
  Subsections `inputs_post`, `inputs_pre` and `inputs_other` partition every issue.
- **Statuses and issue types:**
  - FAIL: "Identity mismatch" (tag tile or post acquisition disagrees with the id, beyond
    `PASS_MATCH_TOLERANCE_SECONDS`), "Expected product already claimed", "No expected product found",
    "Burst ID mismatch", "Too many track numbers present", "Post-dates span too long" (post inputs span more than
    `MAX_POST_SPAN_SECONDS` = 86,400 s), "Pre/Post RTC IDs expected but not found", "Pre/Post RTC IDs found but not
    expected" (a repeated id appears here), "Pre RTC baseline polarization mismatch"; online mode only, "Single
    polarization input used".
  - UNKNOWN: "Tag read error", "Missing tag URL", "Enumeration error" (whole tile), "Comparison error" (one product).
  - INFO: "RTC processing version differs" (a recorded RTC id's processing version is not the archive's; never
    fails the product).
  - A product's status is UNKNOWN if any row is, else FAIL if any row is, else PASS.
- **Modes:** `--mode offline` (default) enumerates from the cached RTC-S1 table per tile, checkpointed per product
  with a fingerprint of the RTC table, the tag row (including its source URL), tool and enumerator versions and
  options; error rows are never checkpointed (F04). `--mode online` queries ASF per product (enumerating once per
  post track); validated with mocks only.
- **Limits:** acquisition identity decides pass/fail; RTC processing versions are informational only. The offline
  re-enumeration uses today's RTC-S1 table, not the archive snapshot production saw when it ran (F27): an RTC
  granule ingested after the product was made can appear as "expected but not found" (7 of 133 pop1 post failures
  at review time). Do not "fix" this by filtering on processing time, which is not ingestion time.

## 5. Coverage (`coverage`) and orphans (`orphans`)

- **Asserts:** every eligible RTC-S1 pass in the acquisition window and AOI has a DIST-S1 product (decision 1): the
  check expects every pass with baseline imagery, so in a campaign run (`--processing-start`) passes the campaign did
  not process are reported missing.
- **Matching:** passes (tile, acquisition group, pass) and products are assigned one-to-one per tile
  (`matching.assign_passes`, `linear_sum_assignment` with dummy rows and columns) within
  `PASS_MATCH_TOLERANCE_SECONDS` (below). The same assignment yields missing passes and orphans.
- **Universe and unit:** every expected pass in the window; unit **passes**; `n_eligible` (passes with
  same-polarization baseline imagery in the lookback windows), `n_found`, `n_missing`.
- **Statuses and reasons:**
  - PASS (matched, readable product).
  - FAIL "no DIST-S1 product for RTC pass at … (k/n bursts have baseline imagery)".
  - SKIPPED "… (no baseline imagery; product not expected)".
  - UNKNOWN "… (lookback incomplete: a required RTC-S1 baseline window was not queried to completion)";
    "ambiguous match: …"; "latest product unreadable: <id> (<error>)" (F14); "group mismatch: …" (the product's
    tagged post bursts belong to another acquisition group). Precedence: ambiguous, unreadable, group mismatch, then
    the missing-pass rules.
- **Orphans:** products matched to no pass are UNKNOWN (decision D12): "DIST-S1 product exists but no matching RTC
  pass found in expected groups", or "… within tolerance of an RTC pass that was matched to another product".
- **Limits:** lookback completeness is judged from the RTC table's manifest; an adopted cache only *claims* its
  windows are complete. The acquisition group is a consistency check through tags, not an independent identity.

## Coverage matching tolerance

The coverage check (check 6) and the orphan check match each expected RTC-S1 pass (an MGRS tile, acquisition
group and pass) to at most one DIST-S1 product, and each product to at most one pass, with a one-to-one assignment
per tile (`matching.assign_passes`). A pass and a product can only be matched when the product's `acq_time` is
within `constants.PASS_MATCH_TOLERANCE_SECONDS` of the pass's `post_acq_dt` (the earliest post-image burst
acquisition). The constant is shared with the inputs check's identity test (post inputs belong to the product's own acquisition).

**Value: 120 s.**

Measured on the frozen caches in `hardening_review/frozen/` (acquisitions 2026-01-01 to 2026-03-01, latest product
versions, assignment computed at a deliberately wide 3,600 s). A match is *verified* when every tagged post-image
burst of the product (`post_rtc_opera_ids`) is one of the pass's bursts.

| cache | products | verified matches | median | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| pop1_aoi (12 tiles) | 388 | 388 | 0 s | 0 s | 0 s | 0 s |
| pop1 (14,930 tiles) | 238,091 | 238,091 | 0 s | 8 s | 19 s | 23 s |
| uat (14,929 tiles) | 238,002 | 236,894 | 0 s | 8 s | 19 s | 23 s |

pop1 verified |difference| histogram: 0-5 s 234,760; 5-10 s 1,145; 10-15 s 1,179; 15-20 s 835; 20-25 s 172; none
above 25 s. uat is the same within a few products. No tagged match failed verification on any cache. A nonzero
difference occurs when a product's earliest post-image burst is not the pass's earliest burst.

The upper bound comes from the passes themselves: in pop1 and uat the smallest gap between two distinct passes of
the same tile is 2,872 s, and 264 gaps are below 3,600 s. A 3,600 s tolerance therefore lets a product be a
candidate for a neighbouring pass. 120 s is about 5x the largest true difference and about 1/24 of the smallest
pass gap.

uat also holds 127 products that the one-to-one assignment leaves as orphans 1-21 s from a matched pass. Each is an
older processing of the same pass whose `acq_time` differs by a few seconds from the latest version, so
deduplication by `opera_dedup_id` (which contains the acquisition time) keeps both. Nearest-time matching used to
hide them by matching both products to the same pass.

Measurement script and outputs: `hardening_review/p3_matching/` (`measure_tolerance.py`, `*_pairs.parquet`,
`*_orphans.parquet`, `*.log`). Re-measure and revisit the value if the enumerator's pass definition or the DIST-S1
`acq_time` convention changes.
