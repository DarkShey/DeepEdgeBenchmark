"""Correctness tests for paired_test.paired_bootstrap_test, independent of the
head-to-head protocol — a wrong significance call would silently invalidate G1/G2."""

import numpy as np
import pytest

from paired_test import paired_bootstrap_test, paired_block_bootstrap_test


def test_all_positive_diffs_are_significant():
    diffs = np.full(30, 2.0)   # every origin: model A beats model B by 2.0
    out = paired_bootstrap_test(diffs, n_boot=5000, seed=0)
    assert out["mean_diff"] == pytest.approx(2.0)
    assert out["p_value"] < 0.001
    assert out["significant_at_05"] is True
    assert out["ci95_lo"] > 0.0   # CI excludes zero


def test_all_zero_diffs_are_not_significant():
    diffs = np.zeros(30)
    out = paired_bootstrap_test(diffs, n_boot=5000, seed=0)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["significant_at_05"] is False


def test_balanced_alternating_diffs_are_not_significant():
    diffs = np.array([1.0, -1.0] * 15)   # mean exactly zero, real dispersion
    out = paired_bootstrap_test(diffs, n_boot=5000, seed=0)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["ci95_lo"] <= 0.0 <= out["ci95_hi"]   # CI brackets zero
    assert out["significant_at_05"] is False


def test_noisy_but_clearly_positive_diffs_are_significant():
    rng = np.random.default_rng(1)
    diffs = rng.normal(loc=5.0, scale=1.0, size=30)   # strong, noisy positive effect
    out = paired_bootstrap_test(diffs, n_boot=5000, seed=0)
    assert out["mean_diff"] > 0
    assert out["significant_at_05"] is True


def test_empty_diffs_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_test([])


def test_result_is_reproducible_given_seed():
    rng = np.random.default_rng(2)
    diffs = rng.normal(0, 1, size=20)
    out1 = paired_bootstrap_test(diffs, n_boot=2000, seed=7)
    out2 = paired_bootstrap_test(diffs, n_boot=2000, seed=7)
    assert out1 == out2


# ── paired_block_bootstrap_test (BRIEF_soir_D7_tests_apparies.md §2) ────────────

def test_block_bootstrap_all_positive_diffs_are_significant():
    diffs = np.full(30, 2.0)
    out = paired_block_bootstrap_test(diffs, block_length=3, n_boot=3000, seed=0)
    assert out["mean_diff"] == pytest.approx(2.0)
    assert out["significant_at_05"] is True
    assert out["effective_n"] == 10   # 30 // 3


def test_block_bootstrap_all_zero_diffs_are_not_significant():
    diffs = np.zeros(30)
    out = paired_block_bootstrap_test(diffs, block_length=3, n_boot=3000, seed=0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["significant_at_05"] is False


def test_block_bootstrap_reports_effective_n_not_n():
    diffs = np.arange(30, dtype=float)
    out = paired_block_bootstrap_test(diffs, block_length=3, n_boot=100, seed=0)
    assert out["n"] == 30
    assert out["effective_n"] == 10
    assert out["effective_n"] < out["n"]


def test_block_bootstrap_rejects_block_length_larger_than_n():
    with pytest.raises(ValueError):
        paired_block_bootstrap_test(np.zeros(5), block_length=10)


def test_block_bootstrap_rejects_empty_diffs():
    with pytest.raises(ValueError):
        paired_block_bootstrap_test([])


def test_block_bootstrap_is_reproducible_given_seed():
    rng = np.random.default_rng(3)
    diffs = rng.normal(0, 1, size=30)
    out1 = paired_block_bootstrap_test(diffs, block_length=3, n_boot=500, seed=5)
    out2 = paired_block_bootstrap_test(diffs, block_length=3, n_boot=500, seed=5)
    assert out1 == out2


def test_block_bootstrap_wider_ci_than_iid_on_autocorrelated_series():
    """Coeur de la justification (BRIEF §2) : sur une serie fortement
    autocorrelee (marche aleatoire lente, pas du bruit blanc), le bootstrap par
    blocs doit donner un CI plus large (donc moins souvent "significatif" a tort)
    que le bootstrap i.i.d. naif -- sinon block bootstrap n'apporte rien."""
    rng = np.random.default_rng(11)
    # serie autocorrelee : marche aleatoire lente autour d'une moyenne nulle
    innovations = rng.normal(0, 1, size=60)
    diffs = np.cumsum(innovations) * 0.3
    diffs = diffs[:30] - diffs[:30].mean()   # centree, mais fortement autocorrelee

    iid = paired_bootstrap_test(diffs, n_boot=5000, seed=0)
    block = paired_block_bootstrap_test(diffs, block_length=5, n_boot=5000, seed=0)

    width_iid = iid["ci95_hi"] - iid["ci95_lo"]
    width_block = block["ci95_hi"] - block["ci95_lo"]
    assert width_block > width_iid
