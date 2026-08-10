"""Correctness tests for calibration_tests.py -- Kupiec / Christoffersen /
TOST are NEW code (nothing in the repo tested coverage or equivalence
formally before), so each is checked against a hand-computed case, its
degenerate inputs, and the invariants it claims. The opt-in
`return_boot_means` added to paired_test.paired_block_bootstrap_test is
checked to leave the existing output untouched."""

import numpy as np
import pytest

import calibration_tests as ct
from paired_test import paired_block_bootstrap_test


# ── Kupiec (unconditional coverage) ──────────────────────────────────────

def test_kupiec_observed_rate_equal_to_target_gives_zero_statistic():
    hits = np.array([1] * 5 + [0] * 95)
    out = ct.kupiec_lr_uc(hits, alpha_target=0.05)
    assert out["status"] == "tested"
    assert out["violation_rate"] == pytest.approx(0.05)
    assert out["lr_uc"] == pytest.approx(0.0, abs=1e-12)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["significant_at_05"] is False


def test_kupiec_matches_hand_computed_statistic():
    """n=100, 20 violations, target 5%:
    LR = -2 [ (80 ln.95 + 20 ln.05) - (80 ln.8 + 20 ln.2) ] = 27.9557..."""
    hits = np.array([1] * 20 + [0] * 80)
    out = ct.kupiec_lr_uc(hits, alpha_target=0.05)
    assert out["lr_uc"] == pytest.approx(27.955724, rel=1e-6)
    assert out["p_value"] < 1e-6
    assert out["significant_at_05"] is True


def test_kupiec_handles_zero_violations_without_log_zero():
    """Over-coverage (0 violations) must give a finite statistic, not nan:
    the unrestricted likelihood is exactly 0, so LR = -2 n ln(1-alpha)."""
    out = ct.kupiec_lr_uc(np.zeros(100), alpha_target=0.05)
    assert np.isfinite(out["lr_uc"])
    assert out["lr_uc"] == pytest.approx(-2.0 * 100 * np.log(0.95), rel=1e-9)
    assert out["violation_rate"] == 0.0


def test_kupiec_empty_input_is_insufficient_data():
    assert ct.kupiec_lr_uc([])["status"] == "insufficient_data"


# ── Christoffersen (independence / conditional coverage) ─────────────────

def test_christoffersen_ind_is_zero_when_transition_rates_match():
    """n00=n01=n10=n11=2 -> pi_01 = pi_11 = 0.5: no clustering at all."""
    hits = [0, 0, 0, 1, 1, 1, 0, 1, 0]
    out = ct.christoffersen_lr_ind(hits)
    assert out["status"] == "tested"
    assert out["counts"] == {"n00": 2, "n01": 2, "n10": 2, "n11": 2}
    assert out["pi_01"] == pytest.approx(out["pi_11"])
    assert out["lr_ind"] == pytest.approx(0.0, abs=1e-12)
    assert out["significant_at_05"] is False


def test_christoffersen_ind_detects_clustered_violations():
    """n00=6, n01=1, n10=1, n11=3 -> pi_01=1/7, pi_11=3/4, pi=4/11 :
    LR = -2 [ (7 ln(7/11) + 4 ln(4/11)) - (6 ln(6/7) + ln(1/7) + ln(1/4)
    + 3 ln(3/4)) ] = 4.1802888..."""
    hits = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    out = ct.christoffersen_lr_ind(hits)
    assert out["counts"] == {"n00": 6, "n01": 1, "n10": 1, "n11": 3}
    assert out["pi_11"] > out["pi_01"]           # une violation en appelle une autre
    assert out["lr_ind"] == pytest.approx(4.1802888, rel=1e-6)
    assert out["significant_at_05"] is True


def test_christoffersen_ind_not_identified_without_both_states():
    """Aucune violation -> pi_11 n'existe pas ; on refuse de chiffrer."""
    out = ct.christoffersen_lr_ind(np.zeros(50, dtype=int))
    assert out["status"] == "not_identified"
    assert "p_value" not in out


def test_christoffersen_cc_is_the_sum_of_its_two_components():
    hits = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
    cc = ct.christoffersen_lr_cc(hits, alpha_target=0.05)
    assert cc["status"] == "tested"
    assert cc["lr_cc"] == pytest.approx(cc["uc"]["lr_uc"] + cc["ind"]["lr_ind"])
    # chi2(2) sur la somme, pas chi2(1)
    from scipy import stats
    assert cc["p_value"] == pytest.approx(stats.chi2.sf(cc["lr_cc"], df=2))


def test_christoffersen_cc_propagates_unidentified_independence_part():
    cc = ct.christoffersen_lr_cc(np.zeros(50, dtype=int))
    assert cc["status"] == "not_identified"
    assert cc["uc"]["status"] == "tested"        # la partie couverture reste lisible


# ── coverage gap, bootstrap par blocs (le test de reference) ─────────────

def test_coverage_gap_block_test_reports_the_signed_gap():
    in_interval = np.array([1.0] * 90 + [0.0] * 10)   # 90% observe vs 95% cible
    out = ct.coverage_gap_block_test(in_interval, target=0.95)
    assert out["status"] == "tested"
    assert out["coverage"] == pytest.approx(0.90)
    assert out["coverage_gap"] == pytest.approx(-0.05)
    assert out["effective_n"] == 100 // 3


def test_coverage_gap_block_test_accepts_fractional_rates():
    """Un taux de couverture moyenne par origine (moyenne sur les graines)
    n'est pas binaire -- le test doit l'accepter tel quel."""
    out = ct.coverage_gap_block_test(np.full(60, 0.95), target=0.95)
    assert out["coverage_gap"] == pytest.approx(0.0)


# ── TOST (equivalence) ───────────────────────────────────────────────────

def test_tost_identical_series_are_equivalent():
    rng = np.random.default_rng(0)
    sq = rng.gamma(2.0, 1.0, 90)
    out = ct.tost_relative_rmse(sq, sq, margin_rel=0.05)
    assert out["rmse_ratio"] == pytest.approx(1.0)
    assert out["verdict"] == "equivalent"
    assert out["p_tost"] < 0.05


def test_tost_large_true_difference_is_inconclusive_not_equivalent():
    """b deux fois pire que a : ratio RMSE ~0.71, tres au-dela de +/-5% --
    l'equivalence ne doit PAS etre conclue."""
    rng = np.random.default_rng(1)
    sq_a = rng.gamma(2.0, 1.0, 90)
    sq_b = 2.0 * sq_a
    out = ct.tost_relative_rmse(sq_a, sq_b, margin_rel=0.05)
    assert out["rmse_ratio"] == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-9)
    assert out["verdict"] == "inconclusive"
    assert out["p_lower"] > 0.05


def test_tost_noisy_but_centred_difference_can_stay_inconclusive():
    """Meme moyenne, beaucoup de bruit et peu de points : ni difference ni
    equivalence etablie -- c'est le cas que le brief veut distinguer de
    'indistinguable donc interchangeable'."""
    rng = np.random.default_rng(2)
    sq_a = rng.gamma(1.0, 1.0, 20)
    sq_b = rng.gamma(1.0, 1.0, 20)
    out = ct.tost_relative_rmse(sq_a, sq_b, margin_rel=0.05)
    assert out["verdict"] == "inconclusive"
    assert 0.0 <= out["p_tost"] <= 1.0


def test_tost_p_tost_is_the_max_of_the_two_one_sided_p_values():
    rng = np.random.default_rng(3)
    sq_a = rng.gamma(2.0, 1.0, 60)
    sq_b = sq_a * 1.02
    out = ct.tost_relative_rmse(sq_a, sq_b, margin_rel=0.05)
    assert out["p_tost"] == pytest.approx(max(out["p_upper"], out["p_lower"]))


def test_tost_wider_margin_never_makes_equivalence_harder():
    rng = np.random.default_rng(4)
    sq_a = rng.gamma(2.0, 1.0, 60)
    sq_b = sq_a * 1.08
    tight = ct.tost_relative_rmse(sq_a, sq_b, margin_rel=0.05)
    wide = ct.tost_relative_rmse(sq_a, sq_b, margin_rel=0.20)
    assert wide["p_tost"] <= tight["p_tost"]


def test_tost_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        ct.tost_relative_rmse(np.ones(10), np.ones(11))


# ── l'opt-in ajoute a paired_test ne change rien au reste ────────────────

def test_return_boot_means_is_purely_additive():
    rng = np.random.default_rng(5)
    diffs = rng.normal(0.2, 1.0, 45)
    base = paired_block_bootstrap_test(diffs, block_length=3, seed=0)
    with_means = paired_block_bootstrap_test(diffs, block_length=3, seed=0, return_boot_means=True)
    assert "boot_means" not in base
    assert with_means["boot_means"].shape == (10000,)
    assert {k: v for k, v in with_means.items() if k != "boot_means"} == base


def test_one_sided_p_values_are_complementary_tails():
    rng = np.random.default_rng(6)
    diffs = rng.normal(0.5, 1.0, 60)
    boot = paired_block_bootstrap_test(diffs, block_length=3, seed=0,
                                       return_boot_means=True)["boot_means"]
    p_less = ct._one_sided_p(boot, "less")
    p_greater = ct._one_sided_p(boot, "greater")
    assert p_less + p_greater == pytest.approx(1.0, abs=1e-9)   # aucun replicat exactement nul
    assert p_greater < 0.05                                       # moyenne clairement > 0
