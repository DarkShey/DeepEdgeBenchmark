"""Unit tests for model_artifacts/crps_kpis.py::crps_from_step_ensembles."""

import numpy as np
import pytest

from model_artifacts import crps_kpis


def test_crps_from_step_ensembles_none_when_no_ensembles():
    assert crps_kpis.crps_from_step_ensembles(None, [1.0, 2.0]) is None
    assert crps_kpis.crps_from_step_ensembles([], [1.0, 2.0]) is None


def test_crps_from_step_ensembles_degenerate_equals_mae():
    """Nuage dégénéré (un seul point répété) : CRPS empirique se réduit au MAE
    (cf. docstring de experiments/crps_metrics.py::crps_empirical, term2=0)."""
    ensembles = [np.full(10, 100.0), np.full(10, 102.0)]
    actual = [101.0, 100.0]

    result = crps_kpis.crps_from_step_ensembles(ensembles, actual)

    expected = np.mean([abs(100.0 - 101.0), abs(102.0 - 100.0)])
    assert result == pytest.approx(expected)


def test_crps_from_step_ensembles_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        crps_kpis.crps_from_step_ensembles([np.array([1.0, 2.0])], [1.0, 2.0])
