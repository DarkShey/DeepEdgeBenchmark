"""
Plumbing + guardrail tests for weekly_headtohead.py — no network access (yfinance
is monkeypatched with a synthetic series), tiny hyperparameters. These check the
protocol's structural guarantees (plan step 7): no lookahead, W/D target-date
alignment, and mu/sd frozen to the first origin — not forecast quality.
"""

import numpy as np
import pandas as pd
import pytest

import weekly_headtohead as wh
import tsdiff_model as td


def _synthetic_daily(n_days=600, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.Series(prices, index=idx, name="Close")


def _tiny_kwargs():
    return dict(n_origins=4, epochs=1, seed=3, n_samples=6, k_denoise=3)


# ── build_weekly / pick_origins ──────────────────────────────────────────────

def test_build_weekly_dates_are_actual_trading_days():
    daily = _synthetic_daily()
    weekly, weekly_dates = wh.build_weekly(daily)
    assert len(weekly) == len(weekly_dates)
    # every "actual date" must be a real entry of the daily index (unlike the
    # W-FRI bin label, which can be a holiday with no daily observation)
    assert weekly_dates.isin(daily.index).all()
    # and the weekly close must equal the daily close on that actual date
    assert np.allclose(weekly.values, daily.loc[weekly_dates.values].values)


def test_pick_origins_leaves_exactly_margin_points_after_last_origin():
    daily = _synthetic_daily()
    weekly, _ = wh.build_weekly(daily)
    origins = wh.pick_origins(weekly, n_origins=5)
    assert len(origins) == 5
    assert origins[-1] + wh.WEEK_MARGIN == len(weekly) - 1
    assert origins == list(range(origins[0], origins[0] + 5))


def test_pick_origins_raises_when_not_enough_weekly_history():
    short_weekly = pd.Series(np.arange(10.0),
                             index=pd.date_range("2024-01-05", periods=10, freq="W-FRI"))
    with pytest.raises(ValueError):
        wh.pick_origins(short_weekly, n_origins=30)


# ── run_pair: guardrails, using a monkeypatched (offline) fetch ─────────────

@pytest.fixture
def patched_fetch(monkeypatch):
    daily = _synthetic_daily()
    monkeypatch.setattr(td, "fetch_data", lambda ticker, start, end: daily)
    return daily


def test_no_lookahead_target_dates_strictly_after_origin(patched_fetch):
    td.set_seed(0)
    result = wh.run_pair("TEST", "TEST", **_tiny_kwargs(), start="2022-01-01", end="2029-01-01")
    for r in result["records"]:
        assert pd.Timestamp(r["target_date"]) > pd.Timestamp(r["origin_date"])
    # and daily-model horizons are always a strictly positive step count
    for r in result["records"]:
        if r["model"] == "TSDiff-D":
            assert r["daily_steps"] > 0


def test_same_target_dates_for_both_models(patched_fetch):
    """Equity guardrail: TSDiff-W and TSDiff-D must be scored on identical
    target dates at every (origin, horizon) — otherwise the comparison is void."""
    td.set_seed(0)
    result = wh.run_pair("TEST", "TEST", **_tiny_kwargs(), start="2022-01-01", end="2029-01-01")
    by_key = {}
    for r in result["records"]:
        key = (r["origin"], r["horizon"])
        by_key.setdefault(key, {})[r["model"]] = r["target_date"]
    for key, by_model in by_key.items():
        assert by_model["TSDiff-W"] == by_model["TSDiff-D"], key
        assert by_model["TSDiff-W"] == by_model["TSDiff-D"] == \
            [r["target_date"] for r in result["records"]
             if r["origin"] == key[0] and r["horizon"] == key[1]][0]


def test_mu_sd_frozen_across_all_origins(patched_fetch, monkeypatch):
    """Guardrail: mu/sd must come from the FIRST origin's train stats only —
    fit_tsdiff must be called exactly once per model (never inside the loop)."""
    calls = []
    real_fit = td.fit_tsdiff

    def counting_fit(*args, **kwargs):
        calls.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(td, "fit_tsdiff", counting_fit)
    td.set_seed(0)
    wh.run_pair("TEST", "TEST", **_tiny_kwargs(), start="2022-01-01", end="2029-01-01")
    assert len(calls) == 2, "fit_tsdiff must be called exactly once for W and once for D"


def test_records_count_matches_origins_horizons_models(patched_fetch):
    td.set_seed(0)
    kw = _tiny_kwargs()
    result = wh.run_pair("TEST", "TEST", **kw, start="2022-01-01", end="2029-01-01")
    assert len(result["records"]) == kw["n_origins"] * 3 * 2  # 3 horizons x 2 models


# ── summarize / crps_empirical wiring ────────────────────────────────────────

def test_summarize_cov95_matches_manual_fraction(patched_fetch):
    td.set_seed(0)
    kw = _tiny_kwargs()
    result = wh.run_pair("TEST", "TEST", **kw, start="2022-01-01", end="2029-01-01")
    summary = wh.summarize(result["records"])
    for r in result["records"]:
        pass  # sanity: summarize must not crash and must key by (asset, horizon, model)
    assert set(summary["TEST"].keys()) == {"W1", "W2", "W3"}
    for horizon in ("W1", "W2", "W3"):
        for model in ("TSDiff-W", "TSDiff-D"):
            m = summary["TEST"][horizon][model]
            manual = np.mean([r["in_interval"] for r in result["records"]
                              if r["horizon"] == horizon and r["model"] == model])
            assert m["Cov95"] == pytest.approx(manual)
            assert m["n_origins"] == kw["n_origins"]
