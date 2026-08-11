"""Tests unitaires du méta-labeling CTA x taxonomie TC
(BRIEF_metalabeling_cta_filtre_tc_nsdiff.md).

Les deux tests exigés nommément par le brief sont ici : **la matrice de décision sur
cas construits** (chaque case des deux filtres, y compris « aucun état ») et **le
placebo à couverture exactement égale**. Aucune base n'est ouverte : tout est calculé
sur des géométries fabriquées à la main, dont on connaît l'état TC attendu.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metalabel_cta_tc as m                                           # noqa: E402


# ── états TC : cas construits, un par état ─────────────────────────────────

def test_tc_state_bull_stress_quand_tout_l_intervalle_est_au_dessus():
    # pi_low > ref -> hausse quasi certaine même au pire bas de l'IC
    assert m.tc_state(ref=100.0, predicted=110.0, pi_low=105.0, pi_high=115.0) == "Bull-Stress"


def test_tc_state_bear_stress_quand_tout_l_intervalle_est_au_dessous():
    assert m.tc_state(ref=100.0, predicted=90.0, pi_low=85.0, pi_high=95.0) == "Bear-Stress"


def test_tc_state_sideways_quand_la_derive_est_negligeable_devant_la_largeur():
    # W = 20, k = 0.10 -> eps = 2 ; |predicted - ref| = 1 <= 2, et ref dans la bande
    assert m.tc_state(ref=100.0, predicted=101.0, pi_low=90.0, pi_high=110.0) == "Sideways"


def test_tc_state_bull_calm_quand_la_derive_depasse_le_seuil_sideways():
    # |predicted - ref| = 5 > eps = 2 -> plus sideways, mais predicted > ref et ref >= pi_low
    assert m.tc_state(ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0) == "Bull-Calm"


def test_tc_state_bear_calm_symetrique():
    assert m.tc_state(ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0) == "Bear-Calm"


def test_precedence_stress_prime_sur_sideways():
    """Une géométrie qui satisfait à la fois Bull-Stress et le critère de dérive
    faible doit ressortir Stress -- la précédence déclarée est Stress > Sideways."""
    # pi_low > ref (stress) ET |predicted - ref| petit devant W
    state = m.tc_state(ref=100.0, predicted=100.5, pi_low=100.2, pi_high=120.0)
    assert state == "Bull-Stress"


def test_precedence_sideways_prime_sur_calm():
    """Le point qui décide de tout le brief : à dérive négligeable, l'état est
    Sideways (« pas de direction »), pas Bull-Calm."""
    assert m.tc_state(ref=100.0, predicted=100.5, pi_low=90.0, pi_high=110.0) == "Sideways"


# ── la matrice de décision, case par case ──────────────────────────────────

CONCORDANTS = [("Bull-Calm", +1), ("Bull-Stress", +1), ("Bear-Calm", -1), ("Bear-Stress", -1)]
CONTRADICTOIRES = [("Bull-Calm", -1), ("Bull-Stress", -1), ("Bear-Calm", +1), ("Bear-Stress", +1)]


@pytest.mark.parametrize("state,cta", CONCORDANTS)
def test_concordance_prise_par_les_deux_filtres(state, cta):
    assert m.decide(state, cta, "F1") == cta
    assert m.decide(state, cta, "F2") == cta


@pytest.mark.parametrize("state,cta", CONTRADICTOIRES)
def test_contradiction_vetoee_par_les_deux_filtres(state, cta):
    assert m.decide(state, cta, "F1") == 0.0
    assert m.decide(state, cta, "F2") == 0.0


@pytest.mark.parametrize("cta", [+1, -1])
def test_sideways_pris_par_F1_vetoe_par_F2(cta):
    """La ligne qui sépare le filtre faible du filtre strict."""
    assert m.decide("Sideways", cta, "F1") == cta
    assert m.decide("Sideways", cta, "F2") == 0.0


@pytest.mark.parametrize("cta", [+1, -1])
def test_aucun_etat_le_filtre_ne_sait_pas_donc_ne_bloque_pas_en_F1(cta):
    assert m.decide("aucun", cta, "F1") == cta
    assert m.decide("aucun", cta, "F2") == 0.0     # F2 n'agit QUE sur concordance


def test_signal_cta_nul_ne_produit_jamais_de_trade():
    for f in ("C0", "F1", "F2"):
        assert m.decide("Bull-Calm", 0.0, f) == 0.0


def test_C0_ne_filtre_rien():
    for state in ("Bull-Calm", "Bear-Stress", "Sideways", "aucun"):
        assert m.decide(state, +1, "C0") == +1


def test_filtre_inconnu_leve():
    with pytest.raises(ValueError, match="filtre inconnu"):
        m.decide("Sideways", 1.0, "F3")


# ── le placebo, à couverture EXACTEMENT égale ──────────────────────────────

def test_placebo_veto_exactement_le_nombre_demande():
    cta = np.array([1.0, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    rng = np.random.default_rng(0)
    for n_veto in (0, 1, 5, 10):
        w = m.placebo_weights(cta, n_veto, rng)
        assert int((w == 0).sum()) == n_veto, "couverture approchée, pas exacte"
        # les trades non vetoés gardent le signe du CTA, jamais un autre
        kept = w != 0
        assert np.array_equal(w[kept], cta[kept])


def test_placebo_ne_vetoe_que_des_cellules_ou_le_cta_a_un_signal():
    """Les cellules sans signal CTA ne sont pas « éligibles » : les compter dans le
    veto gonflerait artificiellement la couverture du placebo."""
    cta = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    rng = np.random.default_rng(1)
    w = m.placebo_weights(cta, 3, rng)
    assert int((w == 0).sum()) == 3 + 2      # les 3 vetoés + les 2 déjà nuls
    # les 3 cellules PORTEUSES de signal ont bien été vetoées, et elles seules
    porteuses = np.flatnonzero(cta != 0)
    assert int((w[porteuses] == 0).sum()) == 3
    assert np.all(w[cta == 0] == 0)


def test_placebo_refuse_de_vetoer_plus_que_les_cellules_eligibles():
    cta = np.array([1.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="veto demandé"):
        m.placebo_weights(cta, 3, np.random.default_rng(0))


def test_placebo_est_reproductible_a_graine_fixee():
    cta = np.ones(20)
    a = m.placebo_weights(cta, 7, np.random.default_rng(123))
    b = m.placebo_weights(cta, 7, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_placebo_varie_entre_tirages_successifs():
    """Deux tirages du même générateur doivent différer -- sinon les 100 tirages du
    contrôle ne mesureraient qu'une seule configuration."""
    cta = np.ones(50)
    rng = np.random.default_rng(7)
    a = m.placebo_weights(cta, 20, rng)
    b = m.placebo_weights(cta, 20, rng)
    assert not np.array_equal(a, b)


# ── cohérence de bout en bout sur une géométrie fabriquée ──────────────────

def test_F2_est_un_sous_ensemble_strict_de_F1():
    """Invariant structurel : tout trade pris par F2 est pris par F1 (F2 est le
    filtre strict). L'inverse est faux dès qu'il existe un jour Sideways."""
    rng = np.random.default_rng(3)
    states = rng.choice(["Bull-Calm", "Bear-Calm", "Sideways", "aucun",
                         "Bull-Stress", "Bear-Stress"], size=500)
    ctas = rng.choice([-1.0, 1.0], size=500)
    f1 = np.array([m.decide(s, c, "F1") for s, c in zip(states, ctas)])
    f2 = np.array([m.decide(s, c, "F2") for s, c in zip(states, ctas)])
    assert np.all((f2 == 0) | (f2 == f1))
    assert (f2 != 0).sum() < (f1 != 0).sum()
