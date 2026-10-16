import pandas as pd

from dist_s1_prod_check import matching
from dist_s1_prod_check.constants import PASS_MATCH_TOLERANCE_SECONDS


TILE = '10SGC'
T0 = pd.Timestamp('2026-01-10T02:06:53', tz='UTC')
PROC = '20260120T000000Z'


def _passes(*specs: tuple[int, pd.Timestamp], tile: str = TILE) -> pd.DataFrame:
    """Build expected pass groups shaped like `coverage.expected_pass_groups` (only the columns matching reads)."""
    return pd.DataFrame(
        [
            {'mgrs_tile_id': tile, 'acq_group_id_within_mgrs_tile': group, 'pass_id': i, 'post_acq_dt': acq}
            for i, (group, acq) in enumerate(specs)
        ]
    )


def _pid(acq: pd.Timestamp, tile: str = TILE) -> str:
    return f'OPERA_L3_DIST-ALERT-S1_T{tile}_{acq:%Y%m%dT%H%M%S}Z_{PROC}_S1A_30_v1'


def _products(*acqs: pd.Timestamp, tile: str = TILE) -> pd.DataFrame:
    return pd.DataFrame(
        [{'opera_id': _pid(acq, tile), 'mgrs_tile_id': tile, 'acq_time': acq.tz_convert(None)} for acq in acqs]
    )


def _key(i: int, group: int = 0, tile: str = TILE) -> tuple[str, int, int]:
    return (tile, group, i)


def _s(seconds: float) -> pd.Timedelta:
    return pd.Timedelta(seconds=seconds)


def test_default_tolerance_is_the_shared_constant() -> None:
    import inspect

    from dist_s1_prod_check import coverage

    for func in (matching.assign_passes, coverage.check_coverage, coverage.check_orphan_products):
        assert inspect.signature(func).parameters['tolerance_seconds'].default == PASS_MATCH_TOLERANCE_SECONDS


def test_no_candidates_everything_goes_to_dummies() -> None:
    passes = _passes((0, T0), (1, T0 + _s(600)))

    result = matching.assign_passes(passes, _products(), tolerance_seconds=100)

    assert result.pass_to_product == {} and result.product_to_pass == {} and result.ambiguous == {}
    assert result.unmatched_passes == [_key(0), _key(1, group=1)]
    assert result.orphan_products == []


def test_products_in_other_tile_are_orphans_and_passes_unmatched() -> None:
    result = matching.assign_passes(_passes((0, T0)), _products(T0, tile='10SGD'), tolerance_seconds=100)

    assert result.pass_to_product == {}
    assert result.unmatched_passes == [_key(0)]
    assert result.orphan_products == [_pid(T0, '10SGD')]


def test_no_passes_all_products_orphans() -> None:
    result = matching.assign_passes(_passes().reindex(columns=['mgrs_tile_id']), _products(T0), tolerance_seconds=100)

    assert result.orphan_products == [_pid(T0)] and result.unmatched_passes == []


def test_disallowed_far_pair_is_never_matched() -> None:
    # Matching the far pair would let both passes be "found"; it must not happen at any total cost.
    passes = _passes((0, T0), (1, T0 + _s(500)))
    products = _products(T0 + _s(10), T0 + _s(5000))

    result = matching.assign_passes(passes, products, tolerance_seconds=100)

    assert result.pass_to_product == {_key(0): _pid(T0 + _s(10))}
    assert result.unmatched_passes == [_key(1, group=1)]
    assert result.orphan_products == [_pid(T0 + _s(5000))]
    assert result.time_diff_seconds[_key(0)] == 10


def test_pair_exactly_at_tolerance_is_matched() -> None:
    result = matching.assign_passes(_passes((0, T0)), _products(T0 + _s(100)), tolerance_seconds=100)

    assert result.pass_to_product == {_key(0): _pid(T0 + _s(100))}


def test_more_products_than_passes_extras_are_orphans() -> None:
    passes = _passes((0, T0))
    products = _products(T0 - _s(30), T0 + _s(5), T0 + _s(60))

    result = matching.assign_passes(passes, products, tolerance_seconds=100)

    assert result.pass_to_product == {_key(0): _pid(T0 + _s(5))}
    assert result.product_to_pass == {_pid(T0 + _s(5)): _key(0)}
    assert sorted(result.orphan_products) == sorted([_pid(T0 - _s(30)), _pid(T0 + _s(60))])
    assert result.ambiguous == {}


def test_one_to_one_is_maximal_then_minimal_time() -> None:
    passes = _passes((0, T0), (2, T0 + _s(100)))
    products = _products(T0 + _s(90), T0 + _s(120))

    result = matching.assign_passes(passes, products, tolerance_seconds=100)

    assert result.pass_to_product == {_key(0): _pid(T0 + _s(90)), _key(1, group=2): _pid(T0 + _s(120))}
    assert result.orphan_products == [] and result.unmatched_passes == []


def test_equally_good_products_make_pass_ambiguous() -> None:
    passes = _passes((0, T0))
    products = _products(T0 - _s(20), T0 + _s(20))

    result = matching.assign_passes(passes, products, tolerance_seconds=100)

    assert set(result.ambiguous) == {_key(0)}
    assert sorted(result.ambiguous[_key(0)]) == sorted(products.opera_id)
    # The tentative match is still one-to-one and every product is either matched or an orphan.
    assert len(result.pass_to_product) == 1 and len(result.orphan_products) == 1


def test_product_equidistant_from_two_passes_makes_both_ambiguous() -> None:
    passes = _passes((0, T0), (1, T0 + _s(40)))

    result = matching.assign_passes(passes, _products(T0 + _s(20)), tolerance_seconds=100)

    assert set(result.ambiguous) == {_key(0), _key(1, group=1)}
    assert len(result.pass_to_product) == 1


def test_tiles_are_matched_independently() -> None:
    passes = pd.concat([_passes((0, T0)), _passes((0, T0), tile='10SGD')], ignore_index=True)
    products = pd.concat([_products(T0 + _s(3)), _products(T0 + _s(4), tile='10SGD')], ignore_index=True)

    result = matching.assign_passes(passes, products, tolerance_seconds=100)

    assert result.pass_to_product == {_key(0): _pid(T0 + _s(3)), _key(0, tile='10SGD'): _pid(T0 + _s(4), '10SGD')}
