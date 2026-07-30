"""
Tests for the distribution-aware CRPS/PIT helpers added to honest_eval/metrics.py
(the reporting-side fix of the 'CRPS gaussien force une loi symétrique'
incoherence) and for prob_kpi_common's two-piece parametric reconstruction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "experiments"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from honest_eval import metrics as hm  # noqa: E402
import prob_kpi_common as pkc  # noqa: E402

MU = np.array([100.0, 50.0])
SIG = np.array([2.0, 1.0])
Y = np.array([101.5, 49.0])


def test_crps_student_t_high_dof_converges_to_gaussian():
    g = hm.crps_gaussian(MU, SIG, Y)
    t = hm.crps_parametric(MU, SIG, Y, dist="student_t", shape=200)
    assert t == pytest.approx(g, rel=1e-3)


def test_crps_lognormal_small_sigma_converges_to_gaussian():
    # lognormal with sigma_log*level == sigma converges to N(level, sigma)
    g = hm.crps_gaussian(np.array([100.0]), np.array([2.0]), np.array([101.5]))
    ln = hm.crps_lognormal(np.log([100.0]), np.array([0.02]), np.array([101.5]))
    assert ln == pytest.approx(g, rel=5e-3)


def test_crps_ged_beta2_is_gaussian():
    g = hm.crps_gaussian(MU, SIG, Y)
    e = hm.crps_parametric(MU, SIG, Y, dist="ged", shape=2.0,
                           n_samples=50000, seed=3)
    assert e == pytest.approx(g, rel=2e-2)  # Monte Carlo tolerance


def test_pit_parametric_normal_matches_pit_values():
    np.testing.assert_allclose(hm.pit_parametric(MU, SIG, Y, dist="normal"),
                               hm.pit_values(MU, SIG, Y))


def test_pit_parametric_student_t_unit_variance_convention():
    # dof large -> equals the normal PIT (unit-variance scaling correct)
    np.testing.assert_allclose(
        hm.pit_parametric(MU, SIG, Y, dist="student_t", shape=500),
        hm.pit_values(MU, SIG, Y), atol=1e-3)


def test_two_piece_symmetric_bounds_reproduce_gaussian():
    rng = np.random.default_rng(0)
    mu, sig = 100.0, 2.0
    lo, hi = mu - pkc.Z95 * sig, mu + pkc.Z95 * sig
    s = pkc.sample_parametric("Naive", mu, lo, hi, 99.0, 100000, rng)
    assert s.mean() == pytest.approx(mu, abs=0.05)
    assert s.std() == pytest.approx(sig, abs=0.05)


def test_two_piece_asymmetric_bounds_shift_mass():
    rng = np.random.default_rng(1)
    mu = 100.0
    lo, hi = mu - pkc.Z95 * 1.0, mu + pkc.Z95 * 3.0   # right-skewed
    s = pkc.sample_parametric("SARIMA", mu, lo, hi, 99.0, 100000, rng)
    assert np.mean(s < mu) == pytest.approx(0.25, abs=0.01)  # sigma_lo/(lo+hi)
    assert np.quantile(s, 0.975) > mu + pkc.Z95 * 2.0        # wide right tail


def test_arima_and_prophet_reconstruct_in_log_space():
    rng = np.random.default_rng(2)
    for model in ("ARIMA-GARCH", "Prophet"):
        s = pkc.sample_parametric(model, 100.0, 90.0, 112.0, 99.0, 200000, rng)
        assert (s > 0).all()
        q = np.quantile(s, [0.025, 0.975])
        assert q[0] == pytest.approx(90.0, rel=5e-3)
        assert q[1] == pytest.approx(112.0, rel=5e-3)
