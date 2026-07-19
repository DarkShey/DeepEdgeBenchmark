"""
Correctness tests for experiments/crps_metrics.crps_empirical, independent of
the weekly head-to-head protocol. A wrong dispersion coefficient (e.g. missing
the 1/2 factor) or a sample-count bias would silently corrupt every CRPS number
in the head-to-head run, so this is checked in isolation first.
"""

import numpy as np
import pytest

from crps_metrics import crps_empirical
from honest_eval.metrics import crps_gaussian

try:
    import properscoring as ps
    HAVE_PROPERSCORING = True
except ImportError:
    HAVE_PROPERSCORING = False


def test_degenerate_ensemble_equals_absolute_error():
    """All samples collapsed on one point c: F is a point mass, so
    CRPS(F, y) = |c - y| exactly (the dispersion term is exactly zero)."""
    samples = np.full(50, 5.0)
    assert crps_empirical(samples, 8.0) == pytest.approx(3.0, abs=1e-12)
    assert crps_empirical(samples, 5.0) == pytest.approx(0.0, abs=1e-12)


def test_two_point_ensemble_matches_closed_form():
    """Ensemble uniform on {a, b} (a<b), evaluated at y<=a: a known closed form
    is CRPS = (b - a) / 4 + (a - y)  (half the point-mass spread penalises the
    ensemble even when centred correctly, plus the miss beyond a)."""
    a, b = 10.0, 20.0
    samples = np.array([a, b])
    y = 5.0
    expected = (b - a) / 4.0 + (a - y)
    assert crps_empirical(samples, y) == pytest.approx(expected, rel=1e-9)


def test_crps_is_nonnegative():
    rng = np.random.default_rng(0)
    samples = rng.normal(100, 10, size=40)
    for y in (80.0, 100.0, 130.0):
        assert crps_empirical(samples, y) >= 0.0


def test_converges_to_closed_form_gaussian_crps():
    """A large sample from N(mu, sigma) should give an empirical CRPS close to
    the closed-form Gaussian CRPS (honest_eval.metrics.crps_gaussian) — this is
    the check that would catch a wrong 1/2 factor or an O(1) bias: get either
    wrong and the two diverge by far more than sampling noise."""
    rng = np.random.default_rng(42)
    mu, sigma = 100.0, 8.0
    samples = rng.normal(mu, sigma, size=5000)
    for y in (95.0, 100.0, 115.0):
        got = crps_empirical(samples, y)
        want = crps_gaussian(mu, sigma, y)
        assert got == pytest.approx(want, rel=0.05)


@pytest.mark.skipif(not HAVE_PROPERSCORING, reason="properscoring not installed")
def test_matches_properscoring_crps_ensemble():
    rng = np.random.default_rng(1)
    for y in (95.0, 105.0, 130.0):
        samples = rng.normal(100, 12, size=64)
        got = crps_empirical(samples, y)
        want = float(ps.crps_ensemble(y, samples))
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9)
