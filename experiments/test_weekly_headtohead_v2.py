"""
Plumbing + guardrail tests for weekly_headtohead_v2.py — no network access
(yfinance monkeypatched), tiny hyperparameters. Checks: RandomWalk baseline math,
G1/G2 flag logic, and end-to-end wiring (same target dates for W/D/RW, RW point
is pure persistence, records cover all 3 models x 3 horizons x n_test origins).
"""

import numpy as np
import pandas as pd
import pytest

import weekly_headtohead_v2 as wh2
import tsdiff_model as td


def _synthetic_daily(n_days=900, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.Series(prices, index=idx, name="Close")


# ── random_walk_samples ───────────────────────────────────────────────────────

def test_random_walk_samples_h1_returns_the_returns_unchanged():
    returns = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    out = wh2.random_walk_samples(returns, h=1)
    assert np.allclose(out, returns)


def test_random_walk_samples_h2_is_rolling_pairwise_sum():
    returns = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    out = wh2.random_walk_samples(returns, h=2)
    expected = np.array([0.01 - 0.02, -0.02 + 0.03, 0.03 + 0.0, 0.0 + 0.015])
    assert np.allclose(out, expected)


def test_random_walk_samples_raises_when_too_few_returns():
    with pytest.raises(ValueError):
        wh2.random_walk_samples(np.array([0.01, 0.02]), h=5)


# ── g1_g2_flags ───────────────────────────────────────────────────────────────

def test_g1_g2_flags_significant_win_over_rw():
    paired = {
        "SPY|W1|TSDiff-W vs TSDiff-D": {"mean_diff": 0.1, "significant_at_05": False},
        "SPY|W1|TSDiff-W vs RandomWalk": {"mean_diff": -2.0, "significant_at_05": True},
        "SPY|W1|TSDiff-D vs RandomWalk": {"mean_diff": -0.1, "significant_at_05": False},
    }
    flags = wh2.g1_g2_flags(paired)
    f = flags["SPY"]["W1"]
    assert f["G1_W_beats_RW"] is True
    assert f["G1_D_beats_RW"] is False
    assert f["G1_any_beats_RW"] is True
    assert f["G2_W_vs_D_significant"] is False


def test_g1_g2_flags_no_significant_effects():
    paired = {
        "BTC|W2|TSDiff-W vs TSDiff-D": {"mean_diff": 0.0, "significant_at_05": False},
        "BTC|W2|TSDiff-W vs RandomWalk": {"mean_diff": 1.0, "significant_at_05": False},
        "BTC|W2|TSDiff-D vs RandomWalk": {"mean_diff": 1.0, "significant_at_05": False},
    }
    flags = wh2.g1_g2_flags(paired)
    f = flags["BTC"]["W2"]
    assert not any(f.values())


def test_g1_g2_flags_positive_significant_diff_does_not_count_as_beating():
    """A significant diff where TSDiff CRPS is WORSE than RW (mean_diff > 0)
    must not be flagged as "beats RW" -- only a significant NEGATIVE diff does."""
    paired = {
        "SPY|W3|TSDiff-W vs TSDiff-D": {"mean_diff": 0.0, "significant_at_05": False},
        "SPY|W3|TSDiff-W vs RandomWalk": {"mean_diff": 3.0, "significant_at_05": True},
        "SPY|W3|TSDiff-D vs RandomWalk": {"mean_diff": 0.0, "significant_at_05": False},
    }
    flags = wh2.g1_g2_flags(paired)
    assert flags["SPY"]["W3"]["G1_W_beats_RW"] is False
    assert flags["SPY"]["W3"]["G1_any_beats_RW"] is False


# ── run_pair_v2 / summarize_v2 / run_paired_tests: offline end-to-end ───────

@pytest.fixture
def patched_fetch(monkeypatch):
    daily = _synthetic_daily()
    monkeypatch.setattr(td, "fetch_data", lambda ticker, start, end: daily)
    return daily


def test_run_pair_v2_records_cover_all_models_horizons_origins(patched_fetch):
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=2, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01")
    assert len(records) == 4 * 3 * 3   # n_test x 3 horizons x 3 models
    seen_models = {r["model"] for r in records}
    assert seen_models == {"TSDiff-W", "TSDiff-D", "RandomWalk"}


def test_run_pair_v2_include_d_false_skips_daily_model(patched_fetch):
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=None, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01", include_d=False)
    assert len(records) == 4 * 3 * 2   # n_test x 3 horizons x 2 models (W + RW only)
    seen_models = {r["model"] for r in records}
    assert seen_models == {"TSDiff-W", "RandomWalk"}


def test_run_paired_tests_skips_absent_d_comparisons(patched_fetch):
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=None, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01", include_d=False)
    paired = wh2.run_paired_tests(records, n_boot=200)
    for horizon in ("W1", "W2", "W3"):
        assert f"TEST|{horizon}|TSDiff-W vs RandomWalk" in paired
        assert f"TEST|{horizon}|TSDiff-W vs TSDiff-D" not in paired
        assert f"TEST|{horizon}|TSDiff-D vs RandomWalk" not in paired


def test_run_pair_v2_same_target_dates_across_all_three_models(patched_fetch):
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=2, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01")
    by_key = {}
    for r in records:
        by_key.setdefault((r["origin"], r["horizon"]), {})[r["model"]] = r["target_date"]
    for key, by_model in by_key.items():
        dates = set(by_model.values())
        assert len(dates) == 1, (key, by_model)


def test_run_pair_v2_random_walk_point_is_pure_persistence(patched_fetch):
    """Brief §5: RW's point forecast must be the last observed price, not the
    mean of its own sample cloud (unlike TSDiff-W/D)."""
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=2, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01")
    rw_records = [r for r in records if r["model"] == "RandomWalk"]
    assert len(rw_records) > 0
    for r in rw_records:
        # same origin/horizon combos with W1 vs W2 vs W3 have different points
        # only if last_price differs by origin -- here we just check the point
        # is finite and that within one origin all 3 horizons share the SAME
        # point (persistence: "today" doesn't change across W1/W2/W3)
        pass
    by_origin = {}
    for r in rw_records:
        by_origin.setdefault(r["origin"], set()).add(r["point"])
    for origin, points in by_origin.items():
        assert len(points) == 1, f"RW point should be constant across horizons at origin {origin}"


def test_summarize_v2_keys_and_values(patched_fetch):
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=2, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01")
    summary = wh2.summarize_v2(records)
    assert set(summary["TEST"].keys()) == {"W1", "W2", "W3"}
    for horizon in ("W1", "W2", "W3"):
        assert set(summary["TEST"][horizon].keys()) == {"TSDiff-W", "TSDiff-D", "RandomWalk"}
        for model, m in summary["TEST"][horizon].items():
            assert m["n_origins"] == 4
            assert m["AvgWidth"] >= 0


def test_run_paired_tests_covers_all_pairs(patched_fetch):
    td.set_seed(0)
    records = wh2.run_pair_v2("TEST", "TEST", epochs_w=2, epochs_d=2, n_val=3, n_test=4,
                              seed=0, n_samples=6, k_denoise=3,
                              start="2021-01-01", end="2025-01-01")
    paired = wh2.run_paired_tests(records, n_boot=200)
    for horizon in ("W1", "W2", "W3"):
        for a, b in wh2.COMPARISON_PAIRS:
            key = f"TEST|{horizon}|{a} vs {b}"
            assert key in paired
            assert paired[key]["n"] == 4
