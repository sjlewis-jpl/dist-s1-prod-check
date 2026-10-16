import warnings

import geopandas as gpd
import pandas as pd

from dist_s1_prod_check import inputs
from dist_s1_prod_check.cmr import build_rtc_input_frame
from dist_s1_prod_check.constants import DELTA_LOOKBACK_DAYS
from factories import bursts_for_tile, dist_id, make_rtc, make_tags


TILE = '10SGC'
T_B = pd.Timestamp('2026-01-22T02:06:53')
T_A = T_B - pd.Timedelta(days=12)
PROC = '2026-05-01T00:00:00'


def _group_bursts(tile: str = TILE, group: int = 0) -> list[str]:
    lut = bursts_for_tile(tile)
    return lut[lut.acq_group_id_within_mgrs_tile == group].jpl_burst_id.tolist()


def _rtc_frame() -> gpd.GeoDataFrame:
    """RTC frame for one acquisition group holding posts A and B and a baseline for both."""
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
        return build_rtc_input_frame(make_rtc(rows), [TILE])


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
