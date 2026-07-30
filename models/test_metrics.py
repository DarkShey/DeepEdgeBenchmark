"""
Unit tests for compute_metrics() — duplicated near-identically across the 4 model
modules, parametrized here over all 4 to lock down shared behaviour and catch drift.

arima_model's compute_metrics diverged on purpose (PI-calibration work): it takes
`pi_bounds={level: (lower, upper)}` instead of `pi_lower=`/`pi_upper=` and reports
one Coverage/Width pair per level. `_metrics_with_pi` adapts the call so the shared
coverage semantics stay locked down across all 4 modules.
"""

import numpy as np
import pytest


def _metrics_with_pi(model_module, actual, predicted, lower, upper):
    """Call compute_metrics with a 95% PI, whichever signature the module has."""
    if model_module.__name__ == "arima_model":
        return model_module.compute_metrics(actual, predicted,
                                            pi_bounds={0.95: (lower, upper)})
    return model_module.compute_metrics(actual, predicted,
                                        pi_lower=lower, pi_upper=upper)


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

    metrics = _metrics_with_pi(model_module, actual, predicted, lower, upper)

    assert metrics["PI Cov 95% (%)"] == 100.0


def test_compute_metrics_pi_coverage_partial_when_one_point_outside(model_module):
    actual    = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    predicted = actual.copy()
    lower = actual - 10.0
    upper = actual + 10.0
    upper[0] = actual[0] - 1.0  # first point now strictly outside [lower, upper]

    metrics = _metrics_with_pi(model_module, actual, predicted, lower, upper)

    assert metrics["PI Cov 95% (%)"] == pytest.approx(80.0, abs=0.01)


def test_compute_metrics_inconsistent_lengths_raises_clear_error(model_module):
    actual    = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    predicted = np.array([100.0, 101.0, 99.0])

    with pytest.raises(ValueError):
        model_module.compute_metrics(actual, predicted)
