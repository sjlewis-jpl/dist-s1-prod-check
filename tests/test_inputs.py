import warnings

import geopandas as gpd
import pandas as pd
import pytest

from dist_s1_prod_check import inputs
from dist_s1_prod_check.cmr import build_rtc_input_frame
from dist_s1_prod_check.constants import DELTA_LOOKBACK_DAYS
from dist_s1_prod_check.ids import get_opera_id_trunc
from factories import bursts_for_tile, dist_id, make_rtc, make_tags, rtc_id


TILE = '10SGC'
T_B = pd.Timestamp('2026-01-22T02:06:53')
T_A = T_B - pd.Timedelta(days=12)
PROC = '2026-05-01T00:00:00'


def _group_bursts(tile: str = TILE, group: int = 0) -> list[str]:
    lut = bursts_for_tile(tile)
    return lut[lut.acq_group_id_within_mgrs_tile == group].jpl_burst_id.tolist()


def _rtc_frame(extra_rows: tuple[dict, ...] = ()) -> gpd.GeoDataFrame:
    """RTC frame for one acquisition group holding posts A and B and a baseline for both, plus `extra_rows`."""
    posts = [T_A, T_B]
    pres = {
        post - pd.Timedelta(days=delta + 12 * k) for post in posts for delta in DELTA_LOOKBACK_DAYS for k in range(1, 5)
    }
    acqs = sorted(set(posts) | pres)
    rows = [
        {'burst': burst, 'acq': acq + pd.Timedelta(seconds=i), 'proc': acq + pd.Timedelta(hours=3)}
        for acq in acqs
        for i, burst in enumerate(_group_bursts())
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build_rtc_input_frame(make_rtc([*rows, *extra_rows]), [TILE])


def _expected_inputs(frame: gpd.GeoDataFrame, post: pd.Timestamp) -> dict[str, list[str]]:
    """Pre/post RTC ids the enumerator expects for the product whose post acquisition is `post`."""
    df_exp = inputs.build_expected_products(frame, [TILE])
    post_min = df_exp[df_exp.input_category == 'post'].groupby('product_id').acq_dt.min()
    product_id = post_min[post_min == post.tz_localize('UTC')].index[0]
    df_prod = df_exp[df_exp.product_id == product_id]
    return {
        'pre_rtc_opera_ids': df_prod[df_prod.input_category == 'pre'].opera_id.tolist(),
        'post_rtc_opera_ids': df_prod[df_prod.input_category == 'post'].opera_id.tolist(),
    }


def _status_by_product(df_results: pd.DataFrame) -> dict[str, bool]:
    return {oid: bool((grp.inputs_correct == True).all()) for oid, grp in df_results.groupby('opera_id')}  # noqa: E712


# F05: the expected product is chosen from the product's own opera_id (acquisition A), not from its tags (B).
def test_f05_mislabelled_product_fails_identity_mismatch() -> None:
    frame = _rtc_frame()
    tags_b = _expected_inputs(frame, T_B)
    id_a, id_b = dist_id(TILE, T_A, PROC), dist_id(TILE, T_B, PROC)
    df_tags = make_tags([{'opera_id': id_a, **tags_b}, {'opera_id': id_b, **tags_b}])

    df = inputs.check_inputs_offline(df_tags, frame)

    assert _status_by_product(df)[id_b], 'B carries its own expected inputs and must pass'
    rows_a = df[df.opera_id == id_a]
    assert not _status_by_product(df)[id_a], 'A carries the inputs of acquisition B'
    assert rows_a.issue_type.str.lower().str.contains('identity mismatch').any()


# F05: B and B2 (acq 30 s later, same tags) both map to expected product B; the closer one (B) claims it.
def test_f05_two_products_one_expected_product_second_fails() -> None:
    frame = _rtc_frame()
    tags_b = _expected_inputs(frame, T_B)
    id_b = dist_id(TILE, T_B, PROC)
    id_b2 = dist_id(TILE, T_B + pd.Timedelta(seconds=30), PROC)
    df_tags = make_tags([{'opera_id': id_b, **tags_b}, {'opera_id': id_b2, **tags_b}])

    status = _status_by_product(inputs.check_inputs_offline(df_tags, frame))

    assert sorted(status.values()) == [False, True], f'actual-to-expected must be one-to-one: {status}'
    assert status[id_b] and not status[id_b2]


def test_f05_tag_tile_disagreeing_with_opera_id_fails_identity_mismatch() -> None:
    frame = _rtc_frame()
    id_b = dist_id(TILE, T_B, PROC)
    df_tags = make_tags([{'opera_id': id_b, 'mgrs_tile_id': '10SGD', **_expected_inputs(frame, T_B)}])

    df = inputs.check_inputs_offline(df_tags, frame)

    rows = df[df.issue_type == 'Identity mismatch']
    assert len(rows) == 1 and '10SGD' in rows.issue_value.iloc[0], df.to_dict('records')
    assert set(df.mgrs_tile_id) == {TILE}, 'rows carry the tile of the opera_id'
    assert set(df.status) == {'FAIL'} and not df.inputs_correct.any()


def test_f05_claim_tie_goes_to_smaller_opera_id_and_status_column_set() -> None:
    frame = _rtc_frame()
    tags_b = _expected_inputs(frame, T_B)
    id_s1a, id_s1c = dist_id(TILE, T_B, PROC, sensor='S1A'), dist_id(TILE, T_B, PROC, sensor='S1C')
    df_tags = make_tags([{'opera_id': id_s1c, **tags_b}, {'opera_id': id_s1a, **tags_b}])

    df = inputs.check_inputs_offline(df_tags, frame)

    assert list(df.columns) == inputs.RESULT_COLUMNS
    by_id = {oid: grp for oid, grp in df.groupby('opera_id')}
    assert by_id[id_s1a].status.tolist() == ['PASS'] and by_id[id_s1a].inputs_correct.tolist() == [True]
    loser = by_id[id_s1c]
    assert loser.issue_type.tolist() == ['Expected product already claimed']
    assert id_s1a in loser.issue_value.iloc[0] and loser.status.tolist() == ['FAIL']


# F22: a malformed tag row must not turn its tile neighbours into an `Enumeration error`.
def test_f22_malformed_tags_are_a_per_product_comparison_error() -> None:
    frame = _rtc_frame()
    id_a, id_b = dist_id(TILE, T_A, PROC), dist_id(TILE, T_B, PROC)
    df_tags = make_tags(
        [
            {'opera_id': id_a, **_expected_inputs(frame, T_A)},
            {'opera_id': id_b, 'pre_rtc_opera_ids': 'not-an-rtc-id', 'post_rtc_opera_ids': 'garbage'},
        ]
    )

    df = inputs.check_inputs_offline(df_tags, frame)

    assert df[df.opera_id == id_a].status.tolist() == ['PASS']
    rows_b = df[df.opera_id == id_b]
    assert rows_b.issue_type.tolist() == ['Comparison error'] and rows_b.status.tolist() == ['UNKNOWN']
    assert rows_b.inputs_correct.isna().all()


def test_f22_enumeration_error_stays_per_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _rtc_frame()
    id_a, id_b = dist_id(TILE, T_A, PROC), dist_id(TILE, T_B, PROC)
    df_tags = make_tags([{'opera_id': id_a, **_expected_inputs(frame, T_A)}, {'opera_id': id_b}])

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError('enumerator failed')

    monkeypatch.setattr(inputs, 'enumerate_dist_s1_products', _boom)
    df = inputs.check_inputs_offline(df_tags, frame)

    assert df.issue_type.tolist() == ['Enumeration error'] * 2 and set(df.status) == {'UNKNOWN'}


def _reprocessed(opera_id: str) -> str:
    """Return the same RTC acquisition (truncated id) with another processing time."""
    tokens = opera_id.split('_')
    tokens[5] = '20991231T000000Z'
    return '_'.join(tokens)


# F26: inputs are compared as multisets, so a repeated input id is reported as found but not expected.
def test_f26_duplicate_rtc_input_fails() -> None:
    frame = _rtc_frame()
    tags_b = _expected_inputs(frame, T_B)
    dup = tags_b['post_rtc_opera_ids'][0]
    id_b = dist_id(TILE, T_B, PROC)
    df_tags = make_tags([{'opera_id': id_b, **tags_b, 'post_rtc_opera_ids': [*tags_b['post_rtc_opera_ids'], dup]}])

    df = inputs.check_inputs_offline(df_tags, frame)

    rows = df[df.issue_type == 'Post RTC IDs found but not expected']
    assert len(rows) == 1 and rows.issue_value.iloc[0] == get_opera_id_trunc(dup), df.to_dict('records')
    assert set(df.status) == {'FAIL'}


# F26: an input reprocessed since the archive snapshot is an INFO note; it does not fail the product.
def test_f26_rtc_processing_version_difference_is_info() -> None:
    frame = _rtc_frame()
    tags_b = _expected_inputs(frame, T_B)
    archived = tags_b['pre_rtc_opera_ids'][0]
    newer = _reprocessed(archived)
    id_b = dist_id(TILE, T_B, PROC)
    df_tags = make_tags([{'opera_id': id_b, **tags_b, 'pre_rtc_opera_ids': [newer, *tags_b['pre_rtc_opera_ids'][1:]]}])

    df = inputs.check_inputs_offline(df_tags, frame)

    assert df.status.tolist() == ['PASS', 'INFO'] and df.inputs_correct.tolist() == [True, True]
    note = df.iloc[1]
    assert note.issue_type == 'RTC processing version differs'
    assert newer in note.issue_value and archived in note.issue_value


ONLINE_BURSTS = {35: 'T035-073226-IW3', 36: 'T036-073300-IW1'}


def _two_track_inputs() -> dict[int, dict[str, list[str]]]:
    """Pre/post RTC ids per track of a product whose acquisition group spans adjacent tracks 35 and 36."""
    post = {35: T_B, 36: T_B + pd.Timedelta(seconds=40)}
    return {
        track: {
            'pre': [rtc_id(burst, post[track] - pd.Timedelta(days=365 + 12 * k), PROC) for k in range(1, 3)],
            'post': [rtc_id(burst, post[track], PROC)],
        }
        for track, burst in ONLINE_BURSTS.items()
    }


def _fake_enumerator(by_track: dict, pols: dict[str, str], calls: list) -> object:
    """Stand-in for `inputs._enumerate_one` that returns only the given track's inputs, with polarizations."""

    def _enumerate_one(mgrs_tile_id: str, track_number: int, post_date: str) -> pd.DataFrame:
        calls.append((mgrs_tile_id, track_number, post_date))
        ids = by_track[track_number]
        df = inputs.parse_actual_inputs(','.join(ids['pre']), ','.join(ids['post']))
        return df.assign(polarizations=df.opera_id.map(lambda oid: pols.get(oid, 'VV+VH')))

    return _enumerate_one


def _online_tags(by_track: dict) -> pd.DataFrame:
    # pre inputs of track 36 first, so the first input is not a post input of the first post track
    return make_tags(
        [
            {
                'opera_id': dist_id(TILE, T_B, PROC),
                'pre_rtc_opera_ids': by_track[36]['pre'] + by_track[35]['pre'],
                'post_rtc_opera_ids': by_track[35]['post'] + by_track[36]['post'],
            }
        ]
    )


# F24: online mode enumerates once per post track and takes its tracks from the post inputs.
def test_f24_online_enumerates_each_post_track(monkeypatch: pytest.MonkeyPatch) -> None:
    by_track, calls = _two_track_inputs(), []
    monkeypatch.setattr(inputs, '_enumerate_one', _fake_enumerator(by_track, {}, calls))

    df = inputs.check_inputs_online(_online_tags(by_track), max_workers=1)

    assert sorted(calls) == [(TILE, 35, '2026-01-22'), (TILE, 36, '2026-01-22')]
    assert df.status.tolist() == ['PASS'] and df.issue_type.tolist() == [''], df.to_dict('records')


# F24: the polarization lookup covers the inputs of every enumerated track.
def test_f24_online_polarization_lookup_covers_both_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    by_track, calls = _two_track_inputs(), []
    hh_pre = by_track[35]['pre'][0]  # not on the track of the first input (36)
    monkeypatch.setattr(inputs, '_enumerate_one', _fake_enumerator(by_track, {hh_pre: 'HH+HV'}, calls))

    df = inputs.check_inputs_online(_online_tags(by_track), max_workers=1)

    rows = df[df.issue_type == 'Pre RTC baseline polarization mismatch']
    assert len(rows) == 1 and get_opera_id_trunc(hh_pre) in rows.issue_value.iloc[0], df.to_dict('records')
    assert set(df.status) == {'FAIL'}


# F37: the post-span limit is MAX_POST_SPAN_SECONDS of elapsed time, not `.days > 1` (which let 47 h through).
@pytest.mark.parametrize(('span_hours', 'flagged'), [(23, False), (25, True)])
def test_f37_post_span_limit_in_seconds(span_hours: int, flagged: bool) -> None:
    post_ids = [
        rtc_id(ONLINE_BURSTS[35], T_B, PROC),
        rtc_id('T035-073227-IW1', T_B + pd.Timedelta(hours=span_hours), PROC),
    ]
    df_actual = inputs.parse_actual_inputs('', ','.join(post_ids))

    issues = inputs.compare_inputs(df_actual, df_actual)

    assert [i['issue_type'] for i in issues] == (['Post-dates span too long'] if flagged else [])


# F38: offline, single-polarization granules are dropped from the frame, so such an input fails the set comparison
# and there is no separate single-polarization diagnostic.
def test_f38_offline_single_pol_input_fails_set_comparison_only() -> None:
    burst = _group_bursts()[0]
    single_acq = T_B - pd.Timedelta(days=400)
    single = rtc_id(burst, single_acq, single_acq + pd.Timedelta(hours=3))
    frame = _rtc_frame(({'opera_id': single, 'polarizations': 'VV'},))
    tags_b = _expected_inputs(frame, T_B)
    id_b = dist_id(TILE, T_B, PROC)
    df_tags = make_tags([{'opera_id': id_b, **tags_b, 'pre_rtc_opera_ids': [*tags_b['pre_rtc_opera_ids'], single]}])

    df = inputs.check_inputs_offline(df_tags, frame)

    assert df.issue_type.tolist() == ['Pre RTC IDs found but not expected'], df.to_dict('records')
    assert df.issue_value.iloc[0] == get_opera_id_trunc(single)


# F38: online, the enumerator's metadata still names single-polarization inputs.
def test_f38_online_single_pol_diagnostic_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    by_track, calls = _two_track_inputs(), []
    single = by_track[35]['pre'][0]
    monkeypatch.setattr(inputs, '_enumerate_one', _fake_enumerator(by_track, {single: 'VV'}, calls))

    df = inputs.check_inputs_online(_online_tags(by_track), max_workers=1)

    rows = df[df.issue_type == 'Single polarization input used']
    assert len(rows) == 1 and get_opera_id_trunc(single) in rows.issue_value.iloc[0], df.to_dict('records')
