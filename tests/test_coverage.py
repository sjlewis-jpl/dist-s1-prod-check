import importlib.metadata
import inspect
from collections.abc import Callable

import pytest
from dist_s1_enumerator.dist_enum import enumerate_dist_s1_products, enumerate_one_dist_s1_product

from dist_s1_prod_check import cmr, coverage
from dist_s1_prod_check.constants import DELTA_LOOKBACK_DAYS, DELTA_WINDOW_DAYS, MAX_PRE_IMGS_PER_BURST


PINNED_ENUMERATOR_VERSION = '1.0.13'
ENUMERATOR_FUNCS = (enumerate_one_dist_s1_product, enumerate_dist_s1_products)


def _default(func: Callable[..., object], name: str) -> object:
    return inspect.signature(func).parameters[name].default


@pytest.mark.parametrize('enum_func', ENUMERATOR_FUNCS, ids=lambda f: f.__name__)
def test_baseline_constants_match_enumerator(enum_func: Callable[..., object]) -> None:
    assert importlib.metadata.version('dist-s1-enumerator') == PINNED_ENUMERATOR_VERSION

    # Our coverage check models the enumerator's multi-window lookback strategy.
    assert _default(enum_func, 'lookback_strategy') == 'multi_window'

    assert tuple(_default(enum_func, 'delta_lookback_days')) == DELTA_LOOKBACK_DAYS
    assert _default(enum_func, 'delta_window_days') == DELTA_WINDOW_DAYS
    assert tuple(_default(enum_func, 'max_pre_imgs_per_burst')) == MAX_PRE_IMGS_PER_BURST

    # Our code hardcodes the minimum as `int(n_pre > 0)`, i.e. >= 1 pre-image
    # (src/dist_s1_prod_check/coverage.py:_baseline_burst_counts).
    assert _default(enum_func, 'min_pre_imgs_per_burst') == 1

    # Function defaults in our modules must use the same constants.
    assert _default(cmr.lookback_date_ranges, 'delta_lookback_days') == DELTA_LOOKBACK_DAYS
    assert _default(cmr.lookback_date_ranges, 'delta_window_days') == DELTA_WINDOW_DAYS
    assert _default(cmr.get_rtc_s1_table_with_lookback, 'delta_lookback_days') == DELTA_LOOKBACK_DAYS
    assert _default(cmr.get_rtc_s1_table_with_lookback, 'delta_window_days') == DELTA_WINDOW_DAYS
    assert _default(coverage.check_coverage, 'delta_lookback_days') == DELTA_LOOKBACK_DAYS
    assert _default(coverage.check_coverage, 'delta_window_days') == DELTA_WINDOW_DAYS
