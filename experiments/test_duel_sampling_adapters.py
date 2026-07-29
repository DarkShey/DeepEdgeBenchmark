"""Correctness + guardrail tests for duel_sampling_adapters.py — each classic
model must produce genuine multi-step trajectory clouds (not a reconstructed
Gaussian from stored CI bounds, audit reserve N1), and the frozen-fit contract
(fit once, never mutate/refit across origins) must hold structurally."""

import numpy as np
import pandas as pd
import pytest

import duel_sampling_adapters as dsa


def _synthetic_prices(n=300, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.Series(prices, index=idx, name="Close")


# ── ARIMA-GARCH ───────────────────────────────────────────────────────────

def test_garch_trajectory_samples_shape_and_finiteness():
    train = _synthetic_prices(250, seed=1)
    state = dsa.fit_garch_state(train, order=(1, 0, 1))
    samples = dsa.garch_trajectory_samples(state, np.array([]), last_price=float(train.iloc[-1]),
                                           horizons=[1, 5, 10], m=64, seed=0)
    assert set(samples) == {1, 5, 10}
    for h, arr in samples.items():
        assert arr.shape == (64,)
        assert np.all(np.isfinite(arr))
        assert np.all(arr > 0)


def test_garch_paths_diverge_across_horizons_and_are_not_degenerate():
    train = _synthetic_prices(250, seed=1)
    state = dsa.fit_garch_state(train, order=(1, 0, 1))
    samples = dsa.garch_trajectory_samples(state, np.array([]), last_price=float(train.iloc[-1]),
                                           horizons=[1, 10], m=200, seed=0)
    assert np.std(samples[1]) > 0
    assert np.std(samples[10]) > 0
    # longer-horizon uncertainty compounds -> should not collapse to a point
    assert np.std(samples[10]) >= np.std(samples[1]) * 0.5


def test_garch_frozen_fit_not_mutated_by_advancing_origins():
    train = _synthetic_prices(250, seed=1)
    state = dsa.fit_garch_state(train, order=(1, 0, 1))
    params_before = state["arima_res"].params.copy()
    new_returns = np.diff(np.log(_synthetic_prices(260, seed=1).values))[-10:] * 100.0
    dsa.garch_trajectory_samples(state, new_returns, last_price=100.0, horizons=[1, 3], m=32, seed=0)
    assert np.allclose(params_before, state["arima_res"].params)


def test_garch_same_seed_reproducible():
    train = _synthetic_prices(250, seed=1)
    state = dsa.fit_garch_state(train, order=(1, 0, 1))
    s1 = dsa.garch_trajectory_samples(state, np.array([]), 100.0, [5], m=32, seed=7)
    s2 = dsa.garch_trajectory_samples(state, np.array([]), 100.0, [5], m=32, seed=7)
    assert np.array_equal(s1[5], s2[5])


# ── SARIMA ────────────────────────────────────────────────────────────────

def test_sarima_trajectory_samples_shape_and_finiteness():
    train = _synthetic_prices(200, seed=2)
    state = dsa.fit_sarima_state(train, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
    samples = dsa.sarima_trajectory_samples(state, np.array([]), horizons=[1, 5], m=48, seed=0)
    assert set(samples) == {1, 5}
    for arr in samples.values():
        assert arr.shape == (48,)
        assert np.all(np.isfinite(arr))


def test_sarima_advancing_state_with_realized_prices_changes_forecast():
    train = _synthetic_prices(200, seed=2)
    full = _synthetic_prices(215, seed=2)
    state = dsa.fit_sarima_state(train, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
    realized = full.values[200:210]
    s_frozen = dsa.sarima_trajectory_samples(state, np.array([]), horizons=[1], m=100, seed=0)
    s_advanced = dsa.sarima_trajectory_samples(state, realized, horizons=[1], m=100, seed=0)
    assert not np.allclose(np.mean(s_frozen[1]), np.mean(s_advanced[1]), atol=1e-6)


# ── Prophet ───────────────────────────────────────────────────────────────

def test_prophet_trajectory_samples_shape_and_dates():
    train = _synthetic_prices(150, seed=3)
    state = dsa.fit_prophet_state(train, m=40)
    future_dates = pd.bdate_range(train.index[-1] + pd.Timedelta(days=1), periods=5)
    target_dates = [future_dates[0], future_dates[2], future_dates[4]]
    samples = dsa.prophet_trajectory_samples(state, target_dates, horizons=[1, 3, 5], seed=0)
    assert set(samples) == {1, 3, 5}
    for arr in samples.values():
        assert arr.shape == (40,)
        assert np.all(np.isfinite(arr))


# ── LSTM ──────────────────────────────────────────────────────────────────

def test_lstm_trajectory_samples_paths_diverge():
    train = _synthetic_prices(120, seed=4)
    state = dsa.fit_lstm_state(train, seq_len=10, units=4, epochs=2, batch_size=16, seed=0)
    samples = dsa.lstm_trajectory_samples(state, train.values, horizons=[1, 3], m=16, seed=0)
    assert set(samples) == {1, 3}
    for arr in samples.values():
        assert arr.shape == (16,)
        assert np.all(np.isfinite(arr))
    # independent dropout masks per path -> not all paths identical
    assert np.std(samples[3]) > 0


def test_lstm_raises_when_tail_too_short():
    train = _synthetic_prices(120, seed=4)
    state = dsa.fit_lstm_state(train, seq_len=10, units=4, epochs=2, batch_size=16, seed=0)
    with pytest.raises(ValueError):
        dsa.lstm_trajectory_samples(state, train.values[-5:], horizons=[1], m=8, seed=0)


# ── Naive (reused random_walk_samples) ───────────────────────────────────

def test_naive_trajectory_samples_draws_from_empirical_pool():
    from weekly_headtohead_v2 import random_walk_samples
    rng = np.random.default_rng(0)
    hist_returns = rng.normal(0, 0.02, 60)
    samples = dsa.naive_trajectory_samples(hist_returns, last_price=100.0, horizons=[1, 2], m=500, seed=0)
    assert set(samples) == {1, 2}
    for h, arr in samples.items():
        assert arr.shape == (500,)
        pool = 100.0 * np.exp(random_walk_samples(hist_returns, h))
        assert set(np.round(arr, 10)).issubset(set(np.round(pool, 10)))


def test_naive_m_is_exact_regardless_of_pool_size():
    rng = np.random.default_rng(0)
    hist_returns = rng.normal(0, 0.02, 10)   # small pool
    samples = dsa.naive_trajectory_samples(hist_returns, last_price=100.0, horizons=[3], m=500, seed=0)
    assert samples[3].shape == (500,)
