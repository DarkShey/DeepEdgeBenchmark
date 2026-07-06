"""
Unit tests for compute_metrics() — duplicated near-identically across the 4 model
modules, parametrized here over all 4 to lock down shared behaviour and catch drift.
"""

import numpy as np
import pytest


def test_compute_metrics_perfect_prediction_gives_zero_error(model_module):
    actual = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    metrics = model_module.compute_metrics(actual, actual.copy())

    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["Dir. Acc (%)"] == 100.0


def test_compute_metrics_directional_accuracy_known_case(model_module):
    # actual diffs signs:    [+1, -1, +1]
    # predicted diffs signs: [-1, +1, +1]  -> only the last one agrees -> 1/3
    actual    = np.array([100.0, 101.0, 99.0, 102.0])
    predicted = np.array([100.0, 99.0, 100.0, 103.0])

    metrics = model_module.compute_metrics(actual, predicted)

    assert metrics["Dir. Acc (%)"] == pytest.approx(33.33, abs=0.01)


def test_compute_metrics_pi_coverage_full_when_bounds_enclose_actual(model_module):
    actual    = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    predicted = actual.copy()
    lower = actual - 10.0
    upper = actual + 10.0

    metrics = model_module.compute_metrics(actual, predicted, pi_lower=lower, pi_upper=upper)

    assert metrics["PI Cov 95% (%)"] == 100.0


def test_compute_metrics_pi_coverage_partial_when_one_point_outside(model_module):
    actual    = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    predicted = actual.copy()
    lower = actual - 10.0
    upper = actual + 10.0
    upper[0] = actual[0] - 1.0  # first point now strictly outside [lower, upper]

    metrics = model_module.compute_metrics(actual, predicted, pi_lower=lower, pi_upper=upper)

    assert metrics["PI Cov 95% (%)"] == pytest.approx(80.0, abs=0.01)


def test_compute_metrics_inconsistent_lengths_raises_clear_error(model_module):
    actual    = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    predicted = np.array([100.0, 101.0, 99.0])

    with pytest.raises(ValueError):
        model_module.compute_metrics(actual, predicted)
