# Checks

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
