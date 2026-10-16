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
@pytest.mark.xfail(strict=True, reason='F15: a 200 response with no `items` key is read as an empty page')
def test_f15_missing_items_key_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _serve(monkeypatch, [FakeResponse({'hits': 10, 'took': 5}, headers={'CMR-Hits': '10'})])

    items, error = _search_outcome()

    assert calls, 'mocked CMR was never queried'
    _assert_incomplete_raised(items, error)


# aa2014f: search_granules returned 2000 rows instead of raising (CMR-Hits 4000 never compared)
@pytest.mark.xfail(strict=True, reason='F15: CMR-Hits is never compared with the rows received')
def test_f15_hits_exceed_rows_without_search_after_raises_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    page = {'hits': 4000, 'items': [{'meta': {'concept-id': f'G{i}-ASF'}} for i in range(CMR_PAGE_SIZE)]}
    calls = _serve(monkeypatch, [FakeResponse(page, headers={'CMR-Hits': '4000'})])

    items, error = _search_outcome()

    assert len(calls) == 1
    _assert_incomplete_raised(items, error)


def _umm_item(opera_id: str) -> dict:
    return {
        'umm': {
            'GranuleUR': opera_id,
            'AdditionalAttributes': [{'Name': 'POLARIZATION', 'Values': ['VV', 'VH']}],
            'RelatedUrls': [],
        }
    }


# aa2014f: the interrupted run's chunk parquet was reused; CMR was queried 0 times and the fresh granule is absent
@pytest.mark.xfail(strict=True, reason='F16: chunk checkpoints without a manifest are reused on resume')
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
