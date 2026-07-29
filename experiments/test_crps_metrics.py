"""
Correctness tests for experiments/crps_metrics.crps_empirical, independent of
the weekly head-to-head protocol. A wrong dispersion coefficient (e.g. missing
the 1/2 factor) or a sample-count bias would silently corrupt every CRPS number
in the head-to-head run, so this is checked in isolation first.
"""

import numpy as np
import pytest

from crps_metrics import crps_empirical, crps_fair
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


# ── crps_fair (Ferro, Richardson & Sherlock 2014) ────────────────────────────

def test_fair_equals_plain_minus_half_term2_over_m_minus_1():
    """fair = plain - 0.5 * term2_plain / (m - 1) — exact algebraic relation,
    checked directly against the two independent implementations. (The
    correction is a DOWNWARD adjustment: term2_fair = term2_plain * m/(m-1)
    is LARGER than term2_plain, and CRPS subtracts 0.5*term2, so the plain
    estimator is biased HIGH and crps_fair corrects it back down.)"""
    rng = np.random.default_rng(3)
    samples = rng.normal(50, 5, size=17)
    y = 48.0
    m = samples.size
    term2_plain = np.mean(np.abs(samples[:, None] - samples[None, :]))
    expected_fair = crps_empirical(samples, y) - 0.5 * term2_plain / (m - 1)
    assert crps_fair(samples, y) == pytest.approx(expected_fair, rel=1e-12)


def test_fair_is_always_less_or_equal_than_plain():
    """The fair correction only ever inflates the subtracted spread term
    (m/(m-1) >= 1), so crps_fair <= crps_empirical for any non-degenerate
    ensemble -- the plain estimator overestimates CRPS at finite m."""
    rng = np.random.default_rng(4)
    for m in (2, 5, 50):
        samples = rng.normal(0, 1, size=m)
        y = 0.3
        assert crps_fair(samples, y) <= crps_empirical(samples, y) + 1e-12


def test_fair_degenerate_ensemble_equals_absolute_error():
    samples = np.full(30, 5.0)
    assert crps_fair(samples, 8.0) == pytest.approx(3.0, abs=1e-12)


def test_fair_single_sample_falls_back_to_absolute_error():
    assert crps_fair(np.array([7.0]), 10.0) == pytest.approx(3.0, abs=1e-12)


def test_fair_converges_to_plain_as_m_grows():
    """The bias correction is O(1/m) — at large m, fair and plain must agree
    much more closely than at small m."""
    rng = np.random.default_rng(5)
    y = 100.0
    small = rng.normal(100, 10, size=5)
    large = rng.normal(100, 10, size=5000)
    gap_small = crps_fair(small, y) - crps_empirical(small, y)
    gap_large = crps_fair(large, y) - crps_empirical(large, y)
    assert abs(gap_large) < abs(gap_small)
    assert abs(gap_large) < 1e-2


def test_fair_empty_raises():
    with pytest.raises(ValueError):
        crps_fair(np.array([]), 1.0)
