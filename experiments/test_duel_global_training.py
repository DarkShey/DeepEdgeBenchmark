"""Correctness tests for duel_global_training.py — TSDiff trained on the
POOLED windows of several assets at once (BRIEF_multigraines.md §2.3). Uses
synthetic in-memory series only (no network): the pure pooling/selection
functions never call td.fetch_data themselves, only evaluate_tsdiff_on_test
does (exercised in the real multi-seed run, not here — same convention as
duel_backtest.run_asset_duel not being network-unit-tested directly)."""

import numpy as np
import pandas as pd
import pytest

import duel_global_training as dgt
import tsdiff_model as td


def _synthetic_weekly(n=150, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-08", periods=n, freq="W-FRI")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.Series(prices, index=idx, name="Close")


def _pooled_weekly(assets=("A", "B", "C"), n=150):
    return {a: _synthetic_weekly(n=n, seed=i) for i, a in enumerate(assets)}


# ── fit_tsdiff_checkpoints_pooled ───────────────────────────────────────────

def test_pooled_checkpoints_yield_shared_model_and_per_asset_mu_sd():
    weekly_by_asset = _pooled_weekly()
    gen = dgt.fit_tsdiff_checkpoints_pooled(weekly_by_asset, horizon=3, candidates=[2, 4], seed=0)
    epochs_seen = []
    for epochs, model, mu_sd_by_asset in gen:
        epochs_seen.append(epochs)
        assert set(mu_sd_by_asset) == {"A", "B", "C"}
        for mu, sd in mu_sd_by_asset.values():
            assert np.isfinite(mu) and np.isfinite(sd) and sd > 0
        assert isinstance(model, td.TSDiff)
    assert epochs_seen == [2, 4]


def test_pooled_checkpoints_raises_on_insufficient_history():
    weekly_by_asset = _pooled_weekly()
    weekly_by_asset["short"] = _synthetic_weekly(n=10, seed=99)  # too short for seq_len=30
    with pytest.raises(ValueError):
        list(dgt.fit_tsdiff_checkpoints_pooled(weekly_by_asset, horizon=3, candidates=[2], seed=0))


def test_pooled_model_can_forecast_for_any_pooled_asset():
    weekly_by_asset = _pooled_weekly()
    gen = dgt.fit_tsdiff_checkpoints_pooled(weekly_by_asset, horizon=3, candidates=[2], seed=0)
    _, model, mu_sd_by_asset = next(gen)
    for asset_code, weekly in weekly_by_asset.items():
        mu, sd = mu_sd_by_asset[asset_code]
        r = td._log_returns(weekly.values.astype(float))
        z = (r - mu) / sd
        td.set_seed(0)
        samples = td.forecast_from_fitted(model, z, mu, sd, last_price=float(weekly.iloc[-1]),
                                          horizons=[1, 2, 3], n_samples=16, k_denoise=3)
        assert set(samples) == {1, 2, 3}
        for arr in samples.values():
            assert arr.shape == (16,)
            assert np.all(np.isfinite(arr))
            assert np.all(arr > 0)


# ── select_global_tsdiff_epochs ─────────────────────────────────────────────

def test_select_global_epochs_picks_one_candidate_with_scores_for_all():
    from epoch_sweep import three_way_split
    from weekly_headtohead import build_weekly

    weekly_by_asset = _pooled_weekly(n=200)
    daily_by_asset, weekly2_by_asset, weekly_dates_by_asset, val_pos_by_asset, train_weekly_by_asset = ({}, {}, {}, {}, {})
    for asset_code, weekly in weekly_by_asset.items():
        # fabricate a daily series consistent with the weekly one (business days)
        idx_daily = pd.bdate_range(weekly.index[0], weekly.index[-1])
        rng = np.random.default_rng(hash(asset_code) % 2**32)
        daily_prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx_daily))))
        daily = pd.Series(daily_prices, index=idx_daily, name="Close")
        w, wd = build_weekly(daily)
        train_end, val_pos, _ = three_way_split(w, n_val=4, n_test=4)
        daily_by_asset[asset_code] = daily
        weekly2_by_asset[asset_code] = w
        weekly_dates_by_asset[asset_code] = wd
        val_pos_by_asset[asset_code] = val_pos
        train_weekly_by_asset[asset_code] = w.iloc[:train_end + 1]

    best_epochs, scores = dgt.select_global_tsdiff_epochs(
        train_weekly_by_asset, weekly2_by_asset, weekly_dates_by_asset, daily_by_asset,
        val_pos_by_asset, candidates=[2, 4], seed=0, n_samples=8, k_denoise=3)
    assert best_epochs in (2, 4)
    assert set(scores) == {2, 4}
    assert all(np.isfinite(v) for v in scores.values())


# ── fit_tsdiff_global ────────────────────────────────────────────────────────

def test_fit_tsdiff_global_is_independent_fresh_fit():
    """A fresh fit_tsdiff_global call at the SAME epoch count as one already
    reached by the incremental sweep must not just be an alias of the sweep's
    (further-trained) model object -- it's a distinct TSDiff instance."""
    weekly_by_asset = _pooled_weekly()
    gen = dgt.fit_tsdiff_checkpoints_pooled(weekly_by_asset, horizon=3, candidates=[2, 4], seed=0)
    _, model_sweep_at_2, _ = next(gen)   # sweep continues past epoch 2 internally after this
    model_fresh, mu_sd_by_asset = dgt.fit_tsdiff_global(weekly_by_asset, horizon=3, epochs=2, seed=0)
    assert model_fresh is not model_sweep_at_2
    assert set(mu_sd_by_asset) == {"A", "B", "C"}
