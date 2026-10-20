# Deletion lists

`dist-s1-prod-check deletion-lists` turns the results of one finished `run-all` into lists of DIST-S1 granules that
should be deleted and regenerated, with the reason for each. **It only writes reports. It deletes nothing and sends
nothing anywhere.** The lists are input to a human decision, not to ASF directly.

```bash
dist-s1-prod-check deletion-lists --out-dir outputs                    # newest run under outputs/
dist-s1-prod-check deletion-lists --run-dir outputs/20261020T101402_871821
dist-s1-prod-check deletion-lists --run-dir ... --unknown-as-failure   # see "Unverified" below
dist-s1-prod-check deletion-lists --run-dir RUN_B --compare-run-dir RUN_A   # see "Comparing two runs"
```

The run must be complete (`run_status.json` COMPLETE, every check done) and unsampled. Otherwise the command refuses
to run, because a missing result anywhere in a tile changes what follows it.

## Why one bad granule condemns the rest of its tile

This is the part most easily missed. DIST-S1 is not a set of independent products. Within one MGRS tile it is a
**chain**:

- Each product names the previous product of its tile (by acquisition time, across all tracks) as its
  `prior_dist_s1_product`.
- It carries that product's alert state forward: provisional and confirmed disturbances, counts and dates.
- So product *N* is only as right as products *1 … N-1*.

If product *N* is wrong, regenerating *N* alone does not fix *N+1*, *N+2* and so on: they were all computed from the
wrong *N*. If *N* is missing, every later product chained over the gap and skipped a step it should have taken.

So every listed granule has a `role`:

| role | meaning |
|---|---|
| `root` | the granule itself failed a check (`own_failures` lists every check it failed) |
| `downstream` | it passed, or failed only a lower-tier check, but comes after a root in its tile and inherited the error. `cause_id` / `cause_class` / `cause_acq_time` name the **earliest** root of the tile, which is what condemns it. |

A root can itself come after an earlier root. It stays `root`, since it failed in its own right, and is deleted
either way. A coverage gap is a root without a granule: its `cause_id` is a pass id such as
`pass:15QXE:20260202T001637Z`.

## Tiers

| tier | what it holds | downstream? |
|---|---|---|
| `candidate` | validated failure classes (all classes below except `inputs_pre` and UNKNOWN) | yes |
| `under_review` | `inputs_pre` roots, and the downstream they would add on top of the candidates | yes, but kept apart |
| `unverified` | UNKNOWN results: the check could not be done | no (unless `--unknown-as-failure`) |

A granule is listed once, in the highest tier that reaches it (candidate > under_review > unverified). A granule
that is an `inputs_pre` root but already downstream of a candidate root is listed as candidate downstream, with
`inputs_pre` in `own_failures`.

## Classes

Examples are from the OPS-POP1 run of 2026-10-20 13:03 local (`test-run2`), acquisitions 2026-01-01 to
2026-03-01 (239,059 products, 14,930 tiles). Ids are shortened: `T09UVB_20260101T021959Z_20260507T233232Z_S1C` stands for
`OPERA_L3_DIST-ALERT-S1_T09UVB_20260101T021959Z_20260507T233232Z_S1C_30_v1.0`. Fields are tile, acquisition time,
processing time and sensor.

### `superseded_duplicate` (candidate, no downstream)

Two or more granules exist for the same tile and acquisition. The one processed last is kept; the earlier ones are
listed. Deleting a superseded copy does not by itself condemn the products after it:

- a later product whose prior *was* the superseded copy fails the confirmation check (`outdated prior`, below);
- a later product whose prior was reprocessed only *after* it ran is a `stale_chain` root.

*Example.* `T09UVB_20260101T021959Z_20260507T233232Z_S1C` was superseded by
`T09UVB_20260101T021959Z_20260624T004004Z_S1C`, the same acquisition reprocessed seven weeks later. The first is
listed; the second stays.

### `confirmation_failure` (candidate)

The product's `prior_dist_s1_product` is not the latest version of the previous acquisition of its tile that existed
when the product was processed:

- the prior is missing;
- the prior is the wrong acquisition;
- the prior is a version that was already outdated;
- the prior tag is absent.

The product's alert state was computed from the wrong history.

*Example (wrong acquisition).* `T01UCT_20260210T055351Z_20260611T185108Z_S1A` names
`T01UCT_20260206T181911Z_…` as prior, but the previous acquisition of 01UCT is `T01UCT_20260208T180249Z`. It
skipped a product. Its 2 later products (`T01UCT_20260218T181911Z_…`, `T01UCT_20260220T180248Z_…`) are downstream.

*Example (outdated prior).* `T15QWV_20260213T002227Z_20260716T034818Z_S1C` used a version of the 2026-02-11
acquisition that had already been replaced by `T15QWV_20260211T120153Z_20260715T221918Z_S1A`. That replacement was
processed about 5.5 hours before the product. 7 later products are downstream.

### `stale_chain` (candidate)

The prior was correct when the product ran, but has since been reprocessed. The superseded prior is on the
`superseded_duplicate` list, so once it is deleted this product points at a granule that no longer exists, and its
state was built on the old version. The check reports these as INFO; for deletion they are roots.

*Example.* `T13TEG_20260211T004544Z_20260715T081003Z_S1C` was built on a version of the 2026-02-08 acquisition. That
acquisition was reprocessed afterwards as `T13TEG_20260208T130910Z_20260717T160943Z_S1A`. 6 later products are
downstream.

### `processing_order`, `first_production_order` (candidate)

A product was processed before the previous acquisition of its tile existed, so its prior cannot have been right.
`first_production_order` only counts when the inverted version is the one still kept. An inversion of a version
since superseded is already covered by `superseded_duplicate`. None in the reference run: all 5 first-production
inversions were of superseded versions.

### `inputs_post` (candidate)

The RTC-S1 post-image granules recorded in the product differ from those a re-enumeration selects. The product was
made from the wrong acquisition data.

*Example.* `T04WDC_20260126T171631Z_20260510T143917Z_S1A` lacks the expected post granules
`OPERA_L2_RTC-S1_T044-093017-IW3_20260126T171640Z` and two others. 9 later products are downstream.

### `inputs_burst_mismatch`, `inputs_no_expected_product`, `inputs_other` (candidate)

- `inputs_burst_mismatch`: the product's bursts are not the burst set expected for its tile and track.
- `inputs_no_expected_product`: the enumerator expects no product for that tile, acquisition and track at all.
- `inputs_other`: any input FAIL of a type this tool does not yet name.

*Examples.*
- `T04WDC_20260126T171631Z_…` above also has a burst mismatch. One granule can be a root for several classes.
- There is no `inputs_no_expected_product` in the reference run. An earlier run had 7, but its RTC-S1 table was
  missing four days of baseline granules (see "Comparing two runs"). With the complete table those products have no
  input issue at all.

### `coverage_gap` (candidate, a pass, not a granule)

An RTC-S1 pass with baseline imagery has no DIST-S1 product. There is nothing to delete for the pass itself; it is
listed in `generate.csv` to be produced. But every later product of the tile chained over it, so they are downstream.

*Example.* No product exists for the 15QXE pass of 2026-02-02 00:16:37. The next product of the tile,
`T15QXE_20260221T000825Z_20260714T075222Z_S1A`, is downstream.

### `coverage_orphan` (candidate)

A product with no RTC-S1 pass behind it. None in the reference run.

### `inputs_pre` (under_review)

The pre-image (baseline) RTC-S1 granules recorded in the product differ from a re-enumeration: "expected but not
found" or "found but not expected". This class is held apart because it is large:
- 6,580 products in 1,874 of the 14,930 tiles;
- with its downstream, 25,716 granules.

In the reference run every one of them has pre granules "expected but not found"; 15 also have some "found but not
expected". That pattern may reflect a difference between this tool and the pipeline rather than bad products: for
example, a baseline RTC granule published only after the product was made, or a different enumerator version.
Triage it before acting on it. This class is also the most sensitive to an incomplete RTC-S1 table. The earlier run
with four missing baseline days reported 39,155 products, nearly all of them artifacts.

*Example.* `T01KAA_20260104T063221Z_20260509T070939Z_S1A`: the re-enumeration expects baseline granules the product
did not use. 3 later products are downstream.

### Unverified: `unknown_confirmation`, `unknown_inputs`, `unknown_coverage`

The check could not be done for that product or pass:

| Check | Cause | Does a rerun fix it? |
|---|---|---|
| confirmation, inputs | the STATUS GeoTIFF could not be read after 10 retries | yes if transient; no if the file is missing or corrupt |
| inputs | `Missing tag URL` | no |
| inputs | `Enumeration error` / `Comparison error` | only if transient |
| coverage | `lookback incomplete` (an RTC baseline window was not fully queried) | yes |
| coverage | `latest product unreadable` | same as a tag read error |
| coverage | `ambiguous match` / `group mismatch` | no; needs a person |

Workflow:
1. Rerun `run-all` on the same `--data-dir`. Failed reads are retried and a new run directory is written.
2. Run `deletion-lists` on that run.
3. When the remaining UNKNOWNs have been looked at, `--unknown-as-failure` makes them candidate roots, with
   downstream, written to `deletion_lists_unknown_as_failure/` beside the default lists.

## Limits

- **Downstream stops at the run window.** Only products acquired in `[--start, --stop)` are listed. Products of the
  same tile acquired after `--stop` descend from the same root and are just as affected.
  - `tiles.csv` gives, per tile, the earliest root and its time (`cause_id`, `cause_acq_time`).
  - "Delete tile X from time T onward" is the open-ended form of the request.
  - In the reference run, tile 29TQE has gaps from 2026-02-03 to the end of the window. Nothing is downstream inside
    the window, but every later 29TQE product is.
- **Roots before `--start` are not seen.** The first product of each tile in the window is SKIPPED by the
  confirmation check, so an error inherited from before the window does not appear.
- With `--processing-start/--processing-stop`, "later products of the tile" means later products inside that
  processing window.

## Output files (`<run dir>/deletion_lists/`)

| file | content |
|---|---|
| `granules.csv` | one row per listed granule: `tier`, `role`, `own_failures`, `cause_id`, `cause_class`, `cause_acq_time` |
| `roots.csv` | one row per root and class, coverage passes included, with the check's reason in `detail` |
| `tiles.csv` | one row per tile and tier: products in the tile, roots, downstream, and the earliest root |
| `generate.csv` | coverage gaps: passes with no product, to be produced |
| `summary.csv` | per tier and class: root granules, root passes, downstream caused, tiles; plus a per-tier unique total |
| `ids/*.txt` | one granule id per line: `<tier>.txt` (all), `<tier>_downstream.txt`, `<tier>_root_<class>.txt` |
| `deletion_lists_manifest.json` | source run, tool versions, options, counts, and the line count of every id file |

In `summary.csv` the class rows overlap: one granule can fail several classes. The `TOTAL (unique granules)` rows
count each granule once. `n_downstream_caused` credits each downstream granule to the class of the earliest root of
its tile.

Counts for the reference run:

| tier | roots | passes | downstream | granules | tiles |
|---|---|---|---|---|---|
| candidate | 1,571 | 786 | 1,638 | 3,209 | 329 |
| under_review | 6,221 | 0 | 19,495 | 25,716 | 1,812 |
| unverified | 0 | 0 | 0 | 0 | 0 |

## Comparing two runs

`--compare-run-dir` builds the lists of a second complete run with the same rules and compares the two:

- `granules.csv` gains `in_other_run`: `same` (same tier, role and cause class), `changed`, `only_this_run` or
  `only_other_run`;
- `comparison_summary.csv`: per tier, role and cause class, the granules in each run, the same, changed and in one
  run only;
- `comparison_granules.csv`: every granule listed differently, with both runs' tier, role and cause class;
- `comparison_roots.csv`: every root, granule or pass, found in one run only.

Two runs over the same window should agree. A difference has one of two causes: the sources changed between the
runs (CMR or the product bucket), or one run's input tables were incomplete. The reference run against the run
before it (same code, enumerator and parameters) showed both:

- one product, `T35UMA_20260213T161145Z`, was reprocessed on 2026-10-19 after the first run had listed the bucket.
  The second run has one more `superseded_duplicate`, and its successor is a new `stale_chain`;
- the first run's RTC-S1 table was missing every granule acquired 2024-01-28, 01-29, 02-13 and 02-14. During a CMR
  outage, CMR under-reported the hits for two 2-day chunks and the chunks were accepted (fixed by F39). This alone
  produced 32,575 extra `inputs_pre` roots, 19 extra input roots in the candidate tier, and 64,231 extra listed
  granules.
