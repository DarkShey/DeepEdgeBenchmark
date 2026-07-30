"""
Tests for the 2026-07-30 sigma-calibration adoptions in models/*.py
(HANDOFF_dist_options_comparison.md follow-up):

  - ARIMA-GARCH : dist='skewt' default (native asymmetric innovation quantiles)
  - SARIMA      : calibrate_sigma='ewma' (variance-targeting on the Kalman PI)
  - Naive/LSTM  : sigma_mode='ewma' (causal EWMA residual sigma path)
  - Prophet     : log_space=True + calibrate_sigma='ewma'

Each adoption keeps a switch reproducing the historical behaviour -- locked
down here alongside the new defaults.
"""

import numpy as np
import pytest

from conftest import make_train_test, reset_seeds

import arima_model
import lstm_model
import naive_model
import prophet_model
import sarima_model


# ── ARIMA-GARCH ──────────────────────────────────────────────────────────────

def test_arima_default_dist_is_skewt():
    assert arima_model.GARCH_DIST == "skewt"


def test_arima_next_step_normal_reproduces_symmetric_band():
    train, test = make_train_test()
    series = train
    reset_seeds(0)
    pred, lo, hi = arima_model.next_step_arima_garch(series, dist="normal")
    # normal innovations: the log-space band is symmetric around log(pred)
    np.testing.assert_allclose(np.log(pred) - np.log(lo),
                               np.log(hi) - np.log(pred), rtol=1e-6)


# ── SARIMA ───────────────────────────────────────────────────────────────────

def test_sarima_ewma_first_step_equals_raw_band():
    """The EWMA state starts at 1, so step 0 must equal the raw conf_int band;
    afterwards the corrected band diverges from the raw one."""
    train, test = make_train_test(n_test=6)
    raw = sarima_model.run_sarima(train, test, calibrate_sigma="off")
    cal = sarima_model.run_sarima(train, test, calibrate_sigma="ewma")

    np.testing.assert_allclose(raw["predictions"], cal["predictions"])
    np.testing.assert_allclose(raw["lower"][0], cal["lower"][0])
    np.testing.assert_allclose(raw["upper"][0], cal["upper"][0])
    assert not np.allclose(raw["lower"][1:], cal["lower"][1:])


def test_sarima_next_step_ewma_returns_finite_ordered_bounds():
    train, _ = make_train_test()
    pred, lo, hi = sarima_model.next_step_sarima(train)
    assert np.isfinite([pred, lo, hi]).all()
    assert lo < pred < hi


# ── Naive / LSTM shared EWMA path ────────────────────────────────────────────

def test_lstm_ewma_sigma_path_is_causal_and_positive():
    rng = np.random.default_rng(0)
    warm = rng.normal(0, 1.0, 100)
    resid = rng.normal(0, 2.0, 30)
    base = lstm_model.ewma_sigma_path(warm, resid)

    assert (base > 0).all()
    resid2 = resid.copy()
    resid2[5] += 50.0
    pert = lstm_model.ewma_sigma_path(warm, resid2)
    np.testing.assert_allclose(pert[:6], base[:6])
    assert (pert[6:] > base[6:]).all()   # a shock can only widen later sigmas


def test_naive_frozen_switch_reproduces_historical_band():
    train, test = make_train_test()
    result = naive_model.run_naive(train, test, sigma_mode="frozen")
    sigma = naive_model.train_sigma(train)
    prev = np.asarray(result["predictions"], float)
    np.testing.assert_allclose(result["upper"], prev + naive_model.Z_95 * sigma)


# ── Prophet ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_prophet_log_space_bounds_are_positive_and_ordered():
    train, test = make_train_test(n_test=3)
    result = prophet_model.run_prophet(train, test)
    lo = np.asarray(result["lower"], float)
    hi = np.asarray(result["upper"], float)
    pred = np.asarray(result["predictions"], float)
    assert (lo > 0).all()
    assert (lo < pred).all() and (pred < hi).all()


@pytest.mark.slow
def test_prophet_price_space_switch_still_works():
    train, test = make_train_test(n_test=2)
    result = prophet_model.run_prophet(train, test, log_space=False,
                                       calibrate_sigma="off")
    assert np.isfinite(result["predictions"]).all()


def test_prophet_next_step_sigma_scale_widens_band():
    train, _ = make_train_test()
    p1, lo1, hi1 = prophet_model.next_step_prophet(train)
    p2, lo2, hi2 = prophet_model.next_step_prophet(train, sigma_scale=2.0)
    assert p2 == pytest.approx(p1, rel=1e-9)
    assert (hi2 - lo2) > (hi1 - lo1)
