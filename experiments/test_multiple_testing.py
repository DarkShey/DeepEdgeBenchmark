"""Tests de `multiple_testing.py` -- Holm-Bonferroni.

Cas calcules a la main (le point du test : verifier la procedure step-down, pas
la reimplementer), invariants theoriques, et entrees degenerees.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiple_testing import holm_bonferroni, correct_family, family_summary   # noqa: E402


# ── cas calcules a la main ──────────────────────────────────────────────────

def test_holm_textbook_case():
    """m=4, alpha=0.05. Seuils par rang : 0.0125, 0.0167, 0.025, 0.05.
    p triees = [0.005, 0.011, 0.02, 0.30].
      rang 0 : 0.005 < 0.0125   -> rejete
      rang 1 : 0.011 < 0.01667  -> rejete
      rang 2 : 0.020 < 0.025    -> rejete
      rang 3 : 0.300 >= 0.05    -> stop
    """
    res = holm_bonferroni([0.005, 0.011, 0.02, 0.30])
    assert res["m"] == 4
    assert res["reject"] == [True, True, True, False]
    assert res["n_rejected"] == 3
    assert res["smallest_threshold"] == pytest.approx(0.0125)


def test_holm_step_down_kills_a_p_below_its_own_threshold():
    """Le coeur de Holm : une p qui passerait SON seuil est quand meme rejetee
    (au sens: non rejetee) si un rang anterieur a echoue. p = [0.04, 0.045],
    m=2, seuils 0.025 puis 0.05. 0.04 >= 0.025 -> stop des le rang 0, donc
    0.045 ne survit pas malgre 0.045 < 0.05."""
    res = holm_bonferroni([0.04, 0.045])
    assert res["reject"] == [False, False]
    assert res["thresholds"][0] == pytest.approx(0.025)
    assert res["thresholds"][1] == pytest.approx(0.05)


def test_holm_adjusted_p_values_are_monotone_and_match_hand_calc():
    """p triees [0.01, 0.02, 0.30], m=3.
    p_adj brutes = 3*0.01=0.03 ; 2*0.02=0.04 ; 1*0.30=0.30, deja croissantes.
    """
    res = holm_bonferroni([0.01, 0.02, 0.30])
    assert res["p_adjusted"] == pytest.approx([0.03, 0.04, 0.30])
    assert res["p_adjusted"] == sorted(res["p_adjusted"])


def test_holm_adjusted_p_values_are_monotonised():
    """p triees [0.02, 0.021], m=2 : 2*0.02=0.04 puis 1*0.021=0.021 < 0.04.
    La monotonisation force le second a 0.04 (sinon un test moins significatif
    ressortirait avec une p corrigee plus petite que le precedent)."""
    res = holm_bonferroni([0.02, 0.021])
    assert res["p_adjusted"] == pytest.approx([0.04, 0.04])


def test_holm_caps_adjusted_p_at_one():
    res = holm_bonferroni([0.6, 0.7, 0.8])
    assert all(p <= 1.0 for p in res["p_adjusted"])
    assert res["n_rejected"] == 0


def test_holm_single_test_is_uncorrected():
    """m=1 : Holm doit se reduire exactement au test brut."""
    assert holm_bonferroni([0.04])["reject"] == [True]
    assert holm_bonferroni([0.06])["reject"] == [False]
    assert holm_bonferroni([0.04])["p_adjusted"] == pytest.approx([0.04])


def test_holm_dominates_bonferroni():
    """Invariant theorique : tout ce que Bonferroni rejette, Holm le rejette."""
    ps = [0.001, 0.009, 0.02, 0.04, 0.5]
    m = len(ps)
    holm = holm_bonferroni(ps)
    for p, rejected in zip(ps, holm["reject"]):
        if p < 0.05 / m:                    # rejete par Bonferroni
            assert rejected, f"Holm doit rejeter p={p} que Bonferroni rejette"


def test_holm_never_rejects_more_than_raw_alpha():
    """Invariant : Holm est conservateur -- il ne peut pas rejeter une p >= alpha."""
    ps = [0.001, 0.049, 0.051, 0.9]
    holm = holm_bonferroni(ps, alpha=0.05)
    for p, rejected in zip(ps, holm["reject"]):
        if p >= 0.05:
            assert not rejected


def test_holm_order_independent():
    """Le resultat ne doit pas dependre de l'ordre de presentation."""
    ps = [0.30, 0.005, 0.02, 0.011]
    res = holm_bonferroni(ps)
    assert res["reject"] == [False, True, True, True]


# ── entrees degenerees ──────────────────────────────────────────────────────

def test_holm_ignores_none_and_does_not_inflate_m():
    """3 p reelles + 2 tests non rendus -> m=3, pas 5. Si m valait 5, le seuil
    du rang 0 serait 0.01 et p=0.012 ne serait pas rejetee."""
    res = holm_bonferroni([0.012, None, 0.013, None, 0.014])
    assert res["m"] == 3
    assert res["smallest_threshold"] == pytest.approx(0.05 / 3)
    assert res["reject"] == [True, None, True, None, True]


def test_holm_all_none():
    res = holm_bonferroni([None, None])
    assert res["m"] == 0 and res["n_rejected"] == 0
    assert res["reject"] == [None, None]


def test_holm_empty():
    res = holm_bonferroni([])
    assert res["m"] == 0 and res["reject"] == []


def test_holm_rejects_invalid_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            holm_bonferroni([0.01], alpha=bad)


def test_holm_zero_p_value():
    """p=0 (aucun replicat bootstrap du mauvais cote) doit passer partout."""
    res = holm_bonferroni([0.0, 0.9])
    assert res["reject"] == [True, False]
    assert res["p_adjusted"][0] == pytest.approx(0.0)


# ── correct_family / family_summary ─────────────────────────────────────────

def _fam():
    return {
        "W+1/global": {"p_value": 0.001, "verdict": "A_significantly_better"},
        "W+1/crypto": {"p_value": 0.030, "verdict": "A_significantly_better"},
        "W+2/global": {"p_value": 0.400, "verdict": "indistinguishable"},
        "W+3/crypto": {"p_value": 0.038, "verdict": "B_significantly_better"},
    }


def test_correct_family_adds_without_overwriting():
    out = correct_family(_fam())
    fam = out["family"]
    # verdicts bruts intacts
    assert fam["W+1/crypto"]["verdict"] == "A_significantly_better"
    assert fam["W+1/crypto"]["p_value"] == 0.030
    # m=4, seuils 0.0125/0.0167/0.025/0.05 sur p triees 0.001/0.030/0.038/0.400
    #   rang 0 : 0.001 < 0.0125 -> rejete
    #   rang 1 : 0.030 >= 0.01667 -> stop
    assert fam["W+1/global"]["holm_reject"] is True
    assert fam["W+1/crypto"]["holm_reject"] is False
    assert fam["W+3/crypto"]["holm_reject"] is False
    assert fam["W+1/crypto"]["holm_verdict"] == "indistinguishable"
    assert fam["W+1/global"]["holm_verdict"] == "A_significantly_better"


def test_correct_family_skips_untested_entries():
    fam = {**_fam(), "W+3/bond": {"status": "insufficient_data"}}
    out = correct_family(fam)
    assert out["holm"]["m"] == 4                      # l'entree sans p_value n'entre pas dans m
    assert out["family"]["W+3/bond"]["holm_reject"] is None
    assert out["family"]["W+3/bond"]["holm_verdict"] is None


def test_family_summary_reports_the_cost():
    summary = family_summary(correct_family(_fam()))
    assert summary["m"] == 4
    assert summary["n_significant_raw"] == 3
    assert summary["n_significant_holm"] == 1
    assert summary["lost_to_correction"] == ["W+1/crypto", "W+3/crypto"]
    assert summary["survivors"] == ["W+1/global"]
    assert summary["smallest_threshold"] == pytest.approx(0.0125)


def test_family_summary_when_nothing_significant():
    fam = {"a": {"p_value": 0.5, "verdict": "indistinguishable"},
           "b": {"p_value": 0.7, "verdict": "indistinguishable"}}
    summary = family_summary(correct_family(fam))
    assert summary["n_significant_raw"] == 0
    assert summary["n_significant_holm"] == 0
    assert summary["lost_to_correction"] == []
