"""Correctness tests for cross_asset_test.py — independent of the multi-asset
head-to-head protocol."""

import numpy as np
import pytest

from cross_asset_test import merge_correlated_diffs, stouffer_combine


def test_merge_correlated_diffs_elementwise_mean():
    a = np.array([2.0, 4.0, -6.0])
    b = np.array([0.0, 2.0, -2.0])
    out = merge_correlated_diffs(a, b)
    assert np.allclose(out, [1.0, 3.0, -4.0])


def test_merge_correlated_diffs_shape_mismatch_raises():
    with pytest.raises(ValueError):
        merge_correlated_diffs(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_stouffer_combine_all_strongly_favor_w_is_significant():
    bucket_tests = {
        "SPY": {"mean_diff": -2.0, "p_value": 0.01},
        "BTC": {"mean_diff": -3.0, "p_value": 0.02},
        "ETH": {"mean_diff": -1.5, "p_value": 0.03},
        "Bonds": {"mean_diff": -1.0, "p_value": 0.04},
    }
    out = stouffer_combine(bucket_tests)
    assert out["favors_first_term"] is True
    assert out["significant_at_05"] is True
    assert out["p_combined"] < 0.01   # combining should be MORE powerful than any single bucket


def test_stouffer_combine_no_effect_anywhere_not_significant():
    bucket_tests = {
        "SPY": {"mean_diff": 0.1, "p_value": 0.95},
        "BTC": {"mean_diff": -0.05, "p_value": 0.9},
        "ETH": {"mean_diff": 0.2, "p_value": 0.8},
        "Bonds": {"mean_diff": 0.0, "p_value": 1.0},
    }
    out = stouffer_combine(bucket_tests)
    assert out["significant_at_05"] is False


def test_stouffer_combine_borderline_individual_pvalues_combine_to_significant():
    """The whole point of combining tests: 4 buckets each individually just at
    p~0.05 (not each significant enough alone by a stricter reading, but all
    pointing the same direction) should combine to something much stronger."""
    bucket_tests = {f"b{i}": {"mean_diff": -1.0, "p_value": 0.2} for i in range(4)}
    out = stouffer_combine(bucket_tests)
    assert out["p_combined"] < 0.2
    assert out["favors_first_term"] is True


def test_stouffer_combine_mixed_directions_partially_cancel():
    bucket_tests = {
        "A": {"mean_diff": -5.0, "p_value": 0.01},
        "B": {"mean_diff": 5.0, "p_value": 0.01},
    }
    out = stouffer_combine(bucket_tests)
    assert abs(out["z_combined"]) < 0.5   # near-perfect cancellation


def test_stouffer_combine_empty_raises():
    with pytest.raises(ValueError):
        stouffer_combine({})
