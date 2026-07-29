"""Correctness tests for duel_pairwise_tests.py — clark_west_test is new code
(nothing else in the repo tests a nested-model comparison), and the grid
orchestration (holm_correct_grid, pooled_pair_verdict) is checked against the
already-tested pooled_analysis/paired_test/honest_eval primitives it wraps."""

import numpy as np
import pandas as pd
import pytest

import duel_pairwise_tests as dpt
from pooled_analysis import holm_correction


# ── clark_west_test ──────────────────────────────────────────────────────

def test_clark_west_candidate_with_perfect_signal_beats_naive():
    rng = np.random.default_rng(0)
    n = 200
    actual = rng.normal(100, 5, n)
    naive = np.roll(actual, 1)
    naive[0] = actual[0]
    candidate = actual + rng.normal(0, 0.1, n)   # near-perfect candidate
    out = dpt.clark_west_test(actual, naive, candidate, h=1)
    assert out["candidate_beats_naive"] is True
    assert out["p_value"] < 0.01
    assert out["cw_stat"] > 0


def test_clark_west_candidate_identical_to_naive_is_not_significant():
    rng = np.random.default_rng(1)
    n = 100
    actual = rng.normal(100, 5, n)
    naive = np.roll(actual, 1)
    naive[0] = actual[0]
    out = dpt.clark_west_test(actual, naive, naive, h=1)   # candidate == restricted
    assert out["mean_f"] == pytest.approx(0.0, abs=1e-9)
    assert out["candidate_beats_naive"] is False


def test_clark_west_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dpt.clark_west_test(np.zeros(5), np.zeros(5), np.zeros(4), h=1)


def test_clark_west_p_value_is_one_sided_half_of_two_sided():
    """When cw_stat > 0, the returned p_value must be exactly half of the
    underlying two-sided dm_hac_test p-value (the CW one-sided convention)."""
    from honest_eval.metrics import dm_hac_test
    rng = np.random.default_rng(2)
    n = 60
    actual = rng.normal(0, 1, n)
    restricted = np.zeros(n)
    candidate = actual * 0.3 + rng.normal(0, 0.5, n)
    f = (actual - restricted) ** 2 - ((actual - candidate) ** 2 - (restricted - candidate) ** 2)
    ref = dm_hac_test(f, h=1)
    out = dpt.clark_west_test(actual, restricted, candidate, h=1)
    if ref["dm_stat"] > 0:
        assert out["p_value"] == pytest.approx(ref["p_value"] / 2.0)
    else:
        assert out["p_value"] == pytest.approx(1.0 - ref["p_value"] / 2.0)


# ── pairwise_crps_tests / holm_correct_grid ─────────────────────────────

def _fake_crps_df(assets, horizons, models, n_origins=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for asset in assets:
        for horizon in horizons:
            for model in models:
                bias = 0.0 if model == "TSDiff-W" else 0.5   # TSDiff systematically better
                for origin in range(n_origins):
                    rows.append({"asset": asset, "horizon": horizon, "model": model,
                                "origin": origin, "crps": bias + rng.normal(5.0, 0.3)})
    return pd.DataFrame(rows)


def test_pairwise_crps_tests_detects_systematic_advantage():
    df = _fake_crps_df(["SPY"], ["W1"], ["TSDiff-W", "Naive"], n_origins=30, seed=0)
    out = dpt.pairwise_crps_tests(df, pairs=[("TSDiff-W", "Naive")], h_by_horizon={"W1": 1})
    res = out["SPY|W1|TSDiff-W vs Naive"]
    assert res["status"] == "tested"
    assert res["mean_diff"] < 0    # TSDiff-W has lower CRPS
    assert res["significant_dm"] is True
    assert res["significant_bootstrap"] is True


def test_pairwise_crps_tests_skips_missing_models():
    df = _fake_crps_df(["SPY"], ["W1"], ["TSDiff-W"], n_origins=10, seed=0)
    out = dpt.pairwise_crps_tests(df, pairs=[("TSDiff-W", "Naive")], h_by_horizon={"W1": 1})
    assert out == {}


def test_holm_correct_grid_matches_holm_correction_reference():
    assets, horizons = ["SPY", "BTC"], ["W1", "W2"]
    df = _fake_crps_df(assets, horizons, ["TSDiff-W", "Naive"], n_origins=20, seed=3)
    cell_results = dpt.pairwise_crps_tests(df, pairs=[("TSDiff-W", "Naive")],
                                           h_by_horizon={"W1": 1, "W2": 2})
    adjusted = dpt.holm_correct_grid(cell_results, assets, horizons, "TSDiff-W vs Naive")
    keys = [f"{a}|{h}|TSDiff-W vs Naive" for a in assets for h in horizons]
    raw_pvals = [cell_results[k]["p_value_bootstrap"] for k in keys]
    expected = dict(zip(keys, holm_correction(raw_pvals)))
    assert adjusted == pytest.approx(expected)


def test_holm_correct_grid_ignores_untested_cells():
    df = _fake_crps_df(["SPY"], ["W1"], ["TSDiff-W", "Naive"], n_origins=3, seed=0)  # < MIN_POINTS
    cell_results = dpt.pairwise_crps_tests(df, pairs=[("TSDiff-W", "Naive")], h_by_horizon={"W1": 1})
    adjusted = dpt.holm_correct_grid(cell_results, ["SPY"], ["W1"], "TSDiff-W vs Naive")
    assert adjusted == {}


# ── pooled_pair_verdict ──────────────────────────────────────────────────

def test_pooled_pair_verdict_fuses_correlated_pairs_and_detects_advantage():
    # BTC-USD/ETH-USD are the crypto pair; SPY is standalone -- 2 independent
    # class series after fusion, not 3 raw assets.
    df = _fake_crps_df(["SPY", "BTC-USD", "ETH-USD"], ["W1"], ["TSDiff-W", "Naive"],
                       n_origins=25, seed=4)
    scales = {"SPY": 1.0, "BTC-USD": 1.0, "ETH-USD": 1.0}
    out = dpt.pooled_pair_verdict(df, ("TSDiff-W", "Naive"), scales, h=1)
    assert out["status"] == "tested"
    assert out["mean_diff"] < 0
    assert out["significant_bootstrap"] is True


def test_pooled_pair_verdict_scale_invariant_ranking():
    """Doubling one asset's MASE scale halves its contribution to the pooled
    differential but must not flip the sign of the pooled verdict."""
    df = _fake_crps_df(["SPY", "ZN=F", "TLT"], ["W1"], ["TSDiff-W", "Naive"], n_origins=25, seed=5)
    out1 = dpt.pooled_pair_verdict(df, ("TSDiff-W", "Naive"), {"SPY": 1.0, "ZN=F": 1.0, "TLT": 1.0}, h=1)
    out2 = dpt.pooled_pair_verdict(df, ("TSDiff-W", "Naive"), {"SPY": 2.0, "ZN=F": 1.0, "TLT": 1.0}, h=1)
    assert np.sign(out1["mean_diff"]) == np.sign(out2["mean_diff"])
