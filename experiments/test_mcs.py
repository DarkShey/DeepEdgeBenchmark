"""Correctness tests for mcs.py — model_confidence_set (Hansen-Lunde-Nason
2011) and spa_test (Hansen 2005), both new code (BRIEF_unification_protocole_
duel.md §2.5, nothing else in the repo implements either)."""

import numpy as np
import pandas as pd
import pytest

import mcs


def _synthetic_loss(T=300, means=(0.0, 0.0, 0.0), noise_std=1.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"M{i}": means[i] + rng.normal(0, noise_std, T) for i in range(len(means))}
    return pd.DataFrame(cols)


# ── model_confidence_set ─────────────────────────────────────────────────

def test_mcs_all_models_equal_survives_intact():
    loss = _synthetic_loss(T=300, means=(0.0, 0.0, 0.0), noise_std=1.0, seed=1)
    out = mcs.model_confidence_set(loss, alpha=0.05, block_length=1, n_boot=2000, seed=0)
    assert set(out["mcs"]) == {"M0", "M1", "M2"}
    assert out["steps"] == []


def test_mcs_eliminates_clearly_worse_models():
    # M0 much lower loss (better); M1/M2 both much worse, similar to each other.
    loss = _synthetic_loss(T=400, means=(0.0, 3.0, 3.0), noise_std=1.0, seed=2)
    out = mcs.model_confidence_set(loss, alpha=0.05, block_length=1, n_boot=2000, seed=0)
    assert out["mcs"] == ["M0"]
    eliminated_names = {s["eliminated"] for s in out["steps"]}
    assert eliminated_names == {"M1", "M2"}


def test_mcs_elimination_order_is_worst_first():
    # M2 is the worst by a wide margin -> must be eliminated before M1.
    loss = _synthetic_loss(T=400, means=(0.0, 1.5, 5.0), noise_std=1.0, seed=3)
    out = mcs.model_confidence_set(loss, alpha=0.05, block_length=1, n_boot=2000, seed=0)
    assert out["steps"][0]["eliminated"] == "M2"


def test_mcs_single_model_input_is_trivial_mcs():
    loss = _synthetic_loss(T=50, means=(0.0,), noise_std=1.0, seed=4)
    out = mcs.model_confidence_set(loss, block_length=1, n_boot=500, seed=0)
    assert out["mcs"] == ["M0"]
    assert out["steps"] == []


def test_mcs_accepts_bare_array_with_integer_names():
    loss = _synthetic_loss(T=200, means=(0.0, 2.0), noise_std=1.0, seed=5).values
    out = mcs.model_confidence_set(loss, block_length=1, n_boot=1000, seed=0)
    assert out["mcs"] == [0]


def test_mcs_rejects_nan_loss():
    loss = _synthetic_loss(T=20, means=(0.0, 0.0), seed=6)
    loss.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        mcs.model_confidence_set(loss)


def test_mcs_block_length_out_of_range_raises():
    loss = _synthetic_loss(T=10, means=(0.0, 0.0), seed=7)
    with pytest.raises(ValueError):
        mcs.model_confidence_set(loss, block_length=11)


# ── spa_test ──────────────────────────────────────────────────────────────

def test_spa_rejects_when_challenger_clearly_beats_benchmark():
    rng = np.random.default_rng(8)
    T = 400
    bench = 3.0 + rng.normal(0, 1.0, T)
    challenger = 0.0 + rng.normal(0, 1.0, T)
    out = mcs.spa_test(bench, {"challenger": challenger}, block_length=1, n_boot=2000, seed=0)
    assert out["reject_no_model_beats_benchmark"] is True
    assert out["p_value"] < 0.05
    assert out["per_model_mean_gain_vs_benchmark"]["challenger"] > 0


def test_spa_does_not_reject_when_models_equal_benchmark():
    rng = np.random.default_rng(9)
    T = 300
    bench = rng.normal(0, 1.0, T)
    challenger = rng.normal(0, 1.0, T)
    out = mcs.spa_test(bench, {"challenger": challenger}, block_length=1, n_boot=2000, seed=0)
    assert out["reject_no_model_beats_benchmark"] is False


def test_spa_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mcs.spa_test(np.zeros(5), {"m": np.zeros(4)})


def test_spa_empty_models_raises():
    with pytest.raises(ValueError):
        mcs.spa_test(np.zeros(5), {})
