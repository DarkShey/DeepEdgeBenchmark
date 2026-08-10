"""Tests unitaires du modele de roulement ajoute a `real_fees` (chantier H2)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_fees as rf                                                # noqa: E402


def test_seuls_les_futures_roulent():
    """L'invariant, pas l'enumeration : tout instrument qui roule doit etre un
    future, et aucun ETF ou crypto au comptant ne doit rouler. Ecrit ainsi, le
    test survit a l'ajout d'instruments (l'univers 0-bis en a ajoute trois) et
    continue d'interdire la vraie erreur -- facturer un roulement a un ETF."""
    for inst in rf.ROLLED_INSTRUMENTS:
        assert "future" in rf.INSTRUMENTS[inst]["vehicle"].lower(), inst
    for inst, spec in rf.INSTRUMENTS.items():
        if "future" not in spec["vehicle"].lower():
            assert rf.roll_cost_bps(inst, 3) == 0.0, inst


def test_tous_les_futures_de_l_univers_roulent():
    """Contre-epreuve : un future ajoute sans etre declare roulant serait facture
    trop peu, silencieusement."""
    futures = [k for k, v in rf.INSTRUMENTS.items() if "future" in v["vehicle"].lower()]
    assert set(futures) == set(rf.ROLLED_INSTRUMENTS), set(futures) ^ set(rf.ROLLED_INSTRUMENTS)


def test_chaque_actif_de_l_univers_0bis_a_un_instrument():
    """Sans instrument, un actif n'a pas de frais et serait trade gratuitement."""
    import prices_v4
    for asset in prices_v4.PANEL:
        assert rf.instruments_for_asset(asset), asset


def test_cout_de_roulement_proportionnel_a_la_duree():
    a, b = rf.roll_cost_bps("SPY-ES", 1), rf.roll_cost_bps("SPY-ES", 3)
    assert b == pytest.approx(3 * a)


def test_un_trimestre_de_detention_coute_un_aller_retour_complet():
    """13 semaines = une echeance traversee en esperance, donc un roulement plein."""
    assert rf.roll_cost_bps("ZN-FUT", 13) == pytest.approx(rf.round_trip_bps("ZN-FUT"))


def test_total_est_bien_outright_plus_roulement():
    for inst in ("SPY-ES", "ZN-FUT", "SPY-ETF"):
        assert rf.total_round_trip_bps(inst, 2) == pytest.approx(
            rf.round_trip_bps(inst) + rf.roll_cost_bps(inst, 2))


def test_conversion_unidirectionnelle_est_la_moitie():
    """Le piege documente en tete de module : `econ_backtest` double lui-meme."""
    assert rf.one_way_total_bps("SPY-ES", 3) == pytest.approx(
        rf.total_round_trip_bps("SPY-ES", 3) / 2.0)


def test_le_roulement_ne_touche_pas_le_chemin_historique():
    """`one_way_bps` reste l'aller-retour outright : aucun appelant existant ne
    doit voir son cout changer parce que H2 a ete ajoute."""
    assert rf.one_way_bps("SPY-ES") == pytest.approx(rf.round_trip_bps("SPY-ES") / 2.0)


@pytest.mark.parametrize("level", list(rf.LEVELS))
def test_le_roulement_suit_le_niveau_de_frais(level):
    assert rf.roll_cost_bps("SPY-ES", 13, level) == pytest.approx(
        rf.round_trip_bps("SPY-ES", level))


def test_duree_nulle_ne_coute_pas_de_roulement():
    assert rf.roll_cost_bps("SPY-ES", 0) == 0.0
