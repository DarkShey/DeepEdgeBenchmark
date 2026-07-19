"""
Plumbing + guardrail tests for epoch_sweep.py — no network access (yfinance is
monkeypatched with a synthetic series), tiny hyperparameters. Checks the v2
guardrails: validation strictly before test (no leakage), epoch selection reads
only validation scores, and incremental checkpointing trains monotonically.
"""

import numpy as np
import pandas as pd
import pytest

import epoch_sweep as es
import tsdiff_model as td


def _synthetic_daily(n_days=900, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.Series(prices, index=idx, name="Close")


# ── three_way_split ──────────────────────────────────────────────────────────

def test_three_way_split_disjoint_and_ordered():
    daily = _synthetic_daily()
    weekly, _ = es.build_weekly(daily)
    train_end, val_pos, test_pos = es.three_way_split(weekly, n_val=5, n_test=8)

    assert len(val_pos) == 5
    assert len(test_pos) == 8
    # strictly increasing, no overlap, validation entirely before test
    assert train_end < val_pos[0]
    assert val_pos == list(range(val_pos[0], val_pos[0] + 5))
    assert val_pos[-1] < test_pos[0]
    assert test_pos == list(range(test_pos[0], test_pos[0] + 8))
    # last test origin leaves exactly WEEK_MARGIN points for its W1-W3 targets
    assert test_pos[-1] + es.WEEK_MARGIN == len(weekly) - 1


def test_three_way_split_raises_when_not_enough_history():
    short_weekly = pd.Series(np.arange(20.0),
                             index=pd.date_range("2024-01-05", periods=20, freq="W-FRI"))
    with pytest.raises(ValueError):
        es.three_way_split(short_weekly, n_val=12, n_test=30)


# ── fit_checkpoints: incremental training ────────────────────────────────────

def test_fit_checkpoints_yields_cumulative_epochs_same_mu_sd():
    daily = _synthetic_daily()
    train = daily.iloc[:400]
    td.set_seed(0)
    seen_epochs = []
    mus, sds = [], []
    for epochs, model, mu, sd in es.fit_checkpoints(train, horizon=5, candidates=[2, 4, 6],
                                                     hidden=8, depth=1, batch_size=32):
        seen_epochs.append(epochs)
        mus.append(mu)
        sds.append(sd)
        assert isinstance(model, td.TSDiff)
    assert seen_epochs == [2, 4, 6]
    # mu/sd are derived from `train` alone -- identical at every checkpoint
    assert len(set(mus)) == 1
    assert len(set(sds)) == 1


def test_fit_checkpoints_trains_incrementally_not_from_scratch():
    """The model object yielded at each checkpoint must be the SAME object
    (weights carried over), not a fresh TSDiff — that's the whole point of the
    checkpointing optimization (120 epoch-units instead of 400 for the sweep)."""
    daily = _synthetic_daily()
    train = daily.iloc[:400]
    td.set_seed(0)
    models = [model for _, model, _, _ in
             es.fit_checkpoints(train, horizon=5, candidates=[2, 4], hidden=8, depth=1)]
    assert models[0] is models[1]


# ── select_epochs ─────────────────────────────────────────────────────────────

def test_select_epochs_picks_argmin_crps_per_asset_model():
    records = [
        {"asset": "SPY", "model": "TSDiff-W", "epochs": 40, "crps_val": 5.0,
         "cov95_val": 0.5, "rel_std_pct_val": 1.0},
        {"asset": "SPY", "model": "TSDiff-W", "epochs": 60, "crps_val": 3.0,
         "cov95_val": 0.6, "rel_std_pct_val": 0.8},
        {"asset": "SPY", "model": "TSDiff-D", "epochs": 40, "crps_val": 2.0,
         "cov95_val": 0.4, "rel_std_pct_val": 1.5},
        {"asset": "SPY", "model": "TSDiff-D", "epochs": 60, "crps_val": 4.0,
         "cov95_val": 0.3, "rel_std_pct_val": 0.5},
    ]
    selected = es.select_epochs(records)
    assert selected["SPY|TSDiff-W"]["epochs"] == 60
    assert selected["SPY|TSDiff-D"]["epochs"] == 40


# ── sweep_asset: guardrails end-to-end, offline ──────────────────────────────

@pytest.fixture
def patched_fetch(monkeypatch):
    daily = _synthetic_daily()
    monkeypatch.setattr(td, "fetch_data", lambda ticker, start, end: daily)
    return daily


def test_sweep_asset_validation_strictly_before_test(patched_fetch):
    td.set_seed(0)
    records, meta = es.sweep_asset("TEST", "TEST", candidates=[2, 4], n_val=3, n_test=5,
                                   seed=0, n_samples=6, k_denoise=3,
                                   start="2021-01-01", end="2025-01-01")
    val_dates = pd.to_datetime(meta["val_origins"])
    test_dates = pd.to_datetime(meta["test_origins"])
    train_end = pd.Timestamp(meta["train_end"])
    assert train_end < val_dates.min()
    assert val_dates.max() < test_dates.min()


def test_sweep_asset_records_cover_every_candidate_and_model(patched_fetch):
    td.set_seed(0)
    candidates = [2, 4, 6]
    records, meta = es.sweep_asset("TEST", "TEST", candidates=candidates, n_val=3, n_test=5,
                                   seed=0, n_samples=6, k_denoise=3,
                                   start="2021-01-01", end="2025-01-01")
    assert len(records) == len(candidates) * 2   # W + D
    seen = {(r["model"], r["epochs"]) for r in records}
    assert seen == {(m, e) for m in ("TSDiff-W", "TSDiff-D") for e in candidates}
    for r in records:
        assert 0.0 <= r["cov95_val"] <= 1.0
        assert r["crps_val"] >= 0.0
        assert r["rel_std_pct_val"] >= 0.0
