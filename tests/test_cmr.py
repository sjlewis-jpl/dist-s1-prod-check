"""Defect tests for CMR paging/validation (F15) and RTC chunk checkpoints (F16) in `cmr.py`."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from tenacity import stop_after_attempt, wait_none

from dist_s1_prod_check import cmr
from dist_s1_prod_check.constants import CMR_PAGE_SIZE, RTC_S1_CONCEPT_ID
from factories import rtc_id


BURST = 'T035-073226-IW3'


class FakeResponse:
    """Minimal stand-in for `requests.Response` as `cmr._get_page` uses it."""

    def __init__(self, payload: dict, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError('network access attempted in test')

    monkeypatch.setattr(cmr.requests, 'get', _blocked)
    # A future validating `_get_page` may raise inside its tenacity retry loop; keep that instant.
    monkeypatch.setattr(cmr._get_page.retry, 'wait', wait_none())
    monkeypatch.setattr(cmr._get_page.retry, 'stop', stop_after_attempt(2))


def _serve(monkeypatch: pytest.MonkeyPatch, pages: list[FakeResponse]) -> list[dict]:
    """Serve `pages` in order from `requests.get`; returns the list of request kwargs seen."""
    calls: list[dict] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append({'url': url, **kwargs})
        return pages[min(len(calls), len(pages)) - 1]

    monkeypatch.setattr(cmr.requests, 'get', fake_get)
    return calls


def _search_outcome(**kwargs: object) -> tuple[list[dict] | None, Exception | None]:
    try:
        return cmr.search_granules(RTC_S1_CONCEPT_ID, '2026-01-01', '2026-01-03', **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, e


def _assert_incomplete_raised(items: list[dict] | None, error: Exception | None) -> None:
    assert error is not None, f'search_granules returned {len(items)} rows instead of raising IncompleteResultsError'
    from dist_s1_prod_check.cmr import IncompleteResultsError  # future interface (Task 2.4)

    assert isinstance(error, IncompleteResultsError), f'raised {type(error).__name__}: {error}'


# aa2014f: search_granules returned 0 rows instead of raising (missing `items` read as an empty last page)
def test_f15_missing_items_key_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _serve(monkeypatch, [FakeResponse({'hits': 10, 'took': 5}, headers={'CMR-Hits': '10'})])

    items, error = _search_outcome()

    assert calls, 'mocked CMR was never queried'
    _assert_incomplete_raised(items, error)


# aa2014f: search_granules returned 2000 rows instead of raising (CMR-Hits 4000 never compared)
def test_f15_hits_exceed_rows_without_search_after_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    page = {'hits': 4000, 'items': [{'meta': {'concept-id': f'G{i}-ASF'}} for i in range(CMR_PAGE_SIZE)]}
    calls = _serve(monkeypatch, [FakeResponse(page, headers={'CMR-Hits': '4000'})])

    items, error = _search_outcome()

    assert len(calls) == 1
    _assert_incomplete_raised(items, error)


def test_f15_paged_search_returns_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [{'meta': {'concept-id': f'G{i}-ASF'}} for i in range(CMR_PAGE_SIZE + 5)]
    pages = [
        FakeResponse({'items': items[:CMR_PAGE_SIZE]}, headers={'CMR-Hits': '2005', 'CMR-Search-After': 'k1'}),
        FakeResponse({'items': items[CMR_PAGE_SIZE:]}, headers={'CMR-Hits': '2005', 'CMR-Search-After': 'k2'}),
        FakeResponse({'items': []}, headers={'CMR-Hits': '2005'}),
    ]
    calls = _serve(monkeypatch, pages)

    result, error = _search_outcome()

    assert error is None and len(result) == 2005 and result.hits == 2005
    assert calls[1]['headers']['CMR-Search-After'] == 'k1'


def test_f15_unknown_hits_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [FakeResponse({'items': [{'meta': {}}]})])

    items, error = _search_outcome()

    _assert_incomplete_raised(items, error)
    assert 'no hit count' in str(error)


def _umm_item(opera_id: str) -> dict:
    return {
        'umm': {
            'GranuleUR': opera_id,
            'AdditionalAttributes': [{'Name': 'POLARIZATION', 'Values': ['VV', 'VH']}],
            'RelatedUrls': [],
        }
    }


# aa2014f: the interrupted run's chunk parquet was reused; CMR was queried 0 times and the fresh granule is absent
def test_f16_chunk_checkpoint_without_manifest_is_not_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    start, stop = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-03')
    stale_id = rtc_id(BURST, '2026-01-02T02:06:53', '2026-01-02T07:00:00')
    fresh_ids = [stale_id, rtc_id(BURST, '2026-01-02T14:06:53', '2026-01-02T20:00:00')]

    # An interrupted run wrote the chunk parquet (same path `_search_chunked_to_df` uses) but no manifest.
    cache_dir = tmp_path / 'rtc_s1_chunks'
    cache_dir.mkdir()
    chunk_path = cache_dir / f'{RTC_S1_CONCEPT_ID}_{start:%Y%m%dT%H%M%S}_{stop:%Y%m%dT%H%M%S}.parquet'
    pd.DataFrame([cmr.parse_rtc_s1_granule(_umm_item(stale_id))]).to_parquet(chunk_path)
    assert not list(cache_dir.glob('*manifest*'))

    queries: list[tuple] = []

    def fake_search(
        concept_id: str, t0: pd.Timestamp, t1: pd.Timestamp, parse_fn: Callable[[dict], dict], **kwargs: object
    ) -> list[dict]:
        queries.append((concept_id, t0, t1))
        return [parse_fn(_umm_item(i)) for i in fresh_ids]

    monkeypatch.setattr(cmr, 'search_granules', fake_search)

    df = cmr.get_rtc_s1_table(start, stop, chunk_days=2, max_workers=1, cache_dir=cache_dir)

    assert len(queries) == 1, f'CMR queried {len(queries)} times; the manifest-less checkpoint was reused'
    assert sorted(df.opera_id) == sorted(fresh_ids)


# c27b56b: the +02:00 offset was dropped without conversion, so the window started 2 h late and the 23:30 UTC
# product acquired inside it was lost (and day 2026-01-01 was never listed).
def test_f28_s3_listing_window_converted_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    from factories import dist_id

    acqs = {'2026/01/01': ['2026-01-01T21:30:00', '2026-01-01T23:30:00'], '2026/01/02': ['2026-01-02T05:00:00']}
    listed: list[str] = []

    def fake_list(bucket: str, prefix: str) -> list[str]:
        day = prefix.removeprefix('products/DIST_S1/').rstrip('/')
        listed.append(day)
        ids = [dist_id('10SGC', a, '2026-01-03T00:00:00') for a in acqs.get(day, [])]
        return [f'{prefix}{i}/{i}_GEN-DIST-STATUS.tif' for i in ids]

    monkeypatch.setattr(cmr, '_list_s3_keys', fake_list)
    df = cmr._list_dist_s1_s3('2026-01-02T00:00:00+02:00', '2026-01-03', 'bucket', max_workers=1)

    assert '2026/01/01' in listed
    assert sorted(df.acq_time.dt.strftime('%H:%M')) == ['05:00', '23:30']


# c27b56b: the listing saw only product folders, so a product whose STATUS layer was never written had an alert_url
# and counted as delivered.
def test_f14_s3_product_without_status_key_has_no_alert_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from factories import dist_id

    complete = dist_id('10SGC', '2026-01-02T02:06:53', '2026-01-03T00:00:00')
    partial = dist_id('10SGC', '2026-01-02T14:06:53', '2026-01-03T00:00:00')

    def fake_list(bucket: str, prefix: str) -> list[str]:
        if not prefix.endswith('2026/01/02/'):
            return []
        layers = ['GEN-DIST-STATUS', 'GEN-METRIC', 'GEN-DIST-STATUS-ACQ']
        keys = [f'{prefix}{complete}/{complete}_{layer}.tif' for layer in layers]
        return keys + [f'{prefix}{partial}/{partial}_GEN-METRIC.tif', f'{prefix}{partial}/{partial}.png']

    monkeypatch.setattr(cmr, '_list_s3_keys', fake_list)
    df = cmr._list_dist_s1_s3('2026-01-01', '2026-01-03', 'bucket', max_workers=1).set_index('opera_id')

    assert df.alert_url[complete].endswith(f'{complete}/{complete}_GEN-DIST-STATUS.tif')
    assert df.alert_url.isna()[partial] and df.alert_acq_url.isna()[partial]
    assert df.metric_url[partial].endswith('_GEN-METRIC.tif')


def _fake_chunk_search(queries: list[tuple], ids: list[str]) -> Callable[..., list[dict]]:
    def fake_search(
        concept_id: str, t0: pd.Timestamp, t1: pd.Timestamp, parse_fn: Callable[[dict], dict], **kwargs: object
    ) -> list[dict]:
        queries.append((concept_id, t0, t1, kwargs.get('bbox')))
        result = cmr.GranuleList(parse_fn(_umm_item(i)) for i in ids)
        result.hits = len(ids)
        return result

    return fake_search


@pytest.mark.parametrize(
    ('case', 'n_queries'),
    [('same_query', 1), ('recent_chunk', 2), ('other_bbox', 2), ('rows_differ', 2), ('other_schema', 2)],
)
def test_f16_chunk_reuse_needs_matching_manifest(
    case: str, n_queries: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    start = pd.Timestamp('2026-01-01')
    if case == 'recent_chunk':
        start = pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
    stop = start + pd.Timedelta(days=2)
    ids = [rtc_id(BURST, start + pd.Timedelta(hours=2), start + pd.Timedelta(hours=7))]
    queries: list[tuple] = []
    monkeypatch.setattr(cmr, 'search_granules', _fake_chunk_search(queries, ids))
    cache_dir = tmp_path / 'chunks'
    bbox = (-120.0, 34.0, -118.0, 36.0)

    log: list[dict] = []
    cmr.get_rtc_s1_table(start, stop, bbox=bbox, chunk_days=2, max_workers=1, cache_dir=cache_dir, chunk_log=log)
    (mpath,) = cache_dir.glob('*.manifest.json')
    record = json.loads(mpath.read_text())
    assert (record['hits'], record['rows'], record['complete']) == (1, 1, True)
    assert log == [{**record, 'reused': False}]
    if case == 'rows_differ':
        mpath.write_text(json.dumps({**record, 'rows': 2}))
    if case == 'other_schema':
        mpath.write_text(json.dumps({**record, 'schema_version': record['schema_version'] - 1}))
    rerun_bbox = (-121.0, 34.0, -118.0, 36.0) if case == 'other_bbox' else bbox

    log2: list[dict] = []
    df = cmr.get_rtc_s1_table(
        start, stop, bbox=rerun_bbox, chunk_days=2, max_workers=1, cache_dir=cache_dir, chunk_log=log2
    )

    assert len(queries) == n_queries, f'{case}: CMR queried {len(queries)} times'
    assert log2[0]['reused'] is (n_queries == 1)
    assert df.opera_id.tolist() == ids


# ---------------------------------------------------------------------------------------------------------------
# F39: CMR under-reported hits during the 2026-10-20 outage. Two RTC-S1 chunks answered 970 and 797 hits (true
# 45,017 and 34,557), returned 1,039 and 800 granules, and were cached as complete; every resume reused them.


def _pages_with_hits(n_items: int, hits: list[int]) -> list[FakeResponse]:
    items = [{'meta': {'concept-id': f'G{i}-ASF'}} for i in range(n_items)]
    pages = []
    for k, h in enumerate(hits):
        chunk = items[k * CMR_PAGE_SIZE : (k + 1) * CMR_PAGE_SIZE]
        headers = {'CMR-Hits': str(h)} | ({'CMR-Search-After': f'k{k}'} if k < len(hits) - 1 else {})
        pages.append(FakeResponse({'items': chunk}, headers=headers))
    return pages


# b23b0b6: returned 2,005 rows; the first page's 2,005 hits were kept and the second page's 9,000 ignored.
def test_f39_hits_changing_between_pages_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _pages_with_hits(CMR_PAGE_SIZE + 5, [2005, 9000]))

    items, error = _search_outcome()

    _assert_incomplete_raised(items, error)
    assert 'changed while paging' in str(error)


# b23b0b6: returned 1,039 rows for an old window that reported 970 hits (accepted as "ingested while paging").
def test_f39_more_rows_than_hits_for_an_old_window_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _pages_with_hits(1039, [970]))

    items, error = _search_outcome()

    _assert_incomplete_raised(items, error)
    assert 'unreliable' in str(error)


def test_f39_more_rows_than_hits_for_a_recent_window_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _pages_with_hits(12, [10]))
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)

    result = cmr.search_granules(RTC_S1_CONCEPT_ID, start, start + pd.Timedelta(days=2))

    assert len(result) == 12 and result.hits == 10


OLD_START = pd.Timestamp('2025-01-01')
N_PER_CHUNK = 150


def _chunk_ids(t0: pd.Timestamp, n: int) -> list[str]:
    return [rtc_id(BURST, t0 + pd.Timedelta(minutes=i), t0 + pd.Timedelta(hours=5)) for i in range(n)]


def _scripted_search(queries: list[pd.Timestamp], short: dict[pd.Timestamp, list[int]]) -> Callable[..., list[dict]]:
    """Return a CMR stand-in serving `N_PER_CHUNK` granules per chunk, with hits equal to what it returns.

    The k-th query of a chunk in `short` returns `short[t0][k]` granules instead (the last entry repeats).
    """

    def fake_search(
        concept_id: str, t0: pd.Timestamp, t1: pd.Timestamp, parse_fn: Callable[[dict], dict], **kwargs: object
    ) -> list[dict]:
        k = sum(q == t0 for q in queries)
        queries.append(t0)
        script = short.get(t0)
        n = script[min(k, len(script) - 1)] if script else N_PER_CHUNK
        result = cmr.GranuleList(parse_fn(_umm_item(i)) for i in _chunk_ids(t0, n))
        result.hits = n
        return result

    return fake_search


def _rtc_table(tmp_path: Path, log: list[dict] | None = None) -> pd.DataFrame:
    stop = OLD_START + pd.Timedelta(days=10)  # five 2-day chunks
    return cmr.get_rtc_s1_table(
        OLD_START, stop, chunk_days=2, max_workers=1, cache_dir=tmp_path / 'chunks', chunk_log=log
    )


def _manifest_of(tmp_path: Path, t0: pd.Timestamp) -> dict:
    import json

    (path,) = (tmp_path / 'chunks').glob(f'*_{t0:%Y%m%dT%H%M%S}_*.manifest.json')
    return json.loads(path.read_text())


# b23b0b6: the 4-granule answer was accepted and cached; the table had 4 granules for the chunk instead of 150.
def test_f39_under_reported_chunk_is_requeried_and_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = OLD_START + pd.Timedelta(days=4)
    queries: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(queries, {bad: [4, N_PER_CHUNK]}))

    log: list[dict] = []
    df = _rtc_table(tmp_path, log)

    assert len(df) == 5 * N_PER_CHUNK
    assert queries.count(bad) == 3  # the short answer, then two agreeing full answers
    record = _manifest_of(tmp_path, bad)
    assert (record['rows'], record['plausibility_confirmed'], record['implausible_rows']) == (N_PER_CHUNK, True, 4)
    assert record['confirm_queries'] == 2
    assert [r['rows'] for r in log] == [N_PER_CHUNK] * 5


# b23b0b6: the short chunk cached by an earlier (outage) run was reused on every rerun; CMR was never asked again.
def test_f39_short_chunk_cached_by_an_earlier_run_is_caught_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = OLD_START + pd.Timedelta(days=2)
    outage: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(outage, {bad: [4]}))
    monkeypatch.setattr(cmr, 'CHUNK_CONFIRM_ATTEMPTS', 0)  # the outage run could not confirm and stopped
    with pytest.raises(cmr.IncompleteResultsError):
        _rtc_table(tmp_path)
    assert _manifest_of(tmp_path, bad)['rows'] == 4, 'the outage answer was not cached; the test sets up nothing'

    monkeypatch.setattr(cmr, 'CHUNK_CONFIRM_ATTEMPTS', 4)
    resume: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(resume, {}))
    df = _rtc_table(tmp_path)

    assert resume == [bad, bad], 'only the short chunk should be re-queried, until two answers agree'
    assert len(df) == 5 * N_PER_CHUNK


def test_f39_real_gap_is_confirmed_kept_and_not_requeried_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gap = OLD_START + pd.Timedelta(days=6)
    queries: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(queries, {gap: [3]}))

    df = _rtc_table(tmp_path)

    assert len(df) == 4 * N_PER_CHUNK + 3
    assert queries.count(gap) == 2
    assert _manifest_of(tmp_path, gap)['plausibility_confirmed'] is True

    rerun: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(rerun, {gap: [3]}))
    assert len(_rtc_table(tmp_path)) == 4 * N_PER_CHUNK + 3
    assert rerun == [], 'a confirmed gap was re-queried on a rerun'


def test_f39_chunk_never_agreeing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = OLD_START + pd.Timedelta(days=4)
    queries: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(queries, {bad: [4, 5, 6, 7, 8, 9]}))

    with pytest.raises(cmr.IncompleteResultsError, match='never returned the same granules twice'):
        _rtc_table(tmp_path)
    assert queries.count(bad) == 1 + cmr.CHUNK_CONFIRM_ATTEMPTS


def test_f39_small_queries_are_not_judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = {OLD_START + pd.Timedelta(days=2 * k): [n] for k, n in enumerate([20, 0, 20, 0, 20])}
    queries: list[pd.Timestamp] = []
    monkeypatch.setattr(cmr, 'search_granules', _scripted_search(queries, sparse))

    assert len(_rtc_table(tmp_path)) == 60
    assert len(queries) == 5, 'an AOI-sized query was re-queried'


def test_f39_incomplete_chunk_answer_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr._search_chunk.retry, 'wait', wait_none())
    answers = iter([cmr.IncompleteResultsError('flaky'), ['ok']])

    def flaky(*args: object, **kwargs: object) -> list[str]:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(cmr, 'search_granules', flaky)
    assert cmr._search_chunk(RTC_S1_CONCEPT_ID, OLD_START, OLD_START) == ['ok']
