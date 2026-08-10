"""Tests de `real_fees.py`.

Le risque reel de ce module n'est pas mathematique, il est SEMANTIQUE : ses
chiffres sont des frais ALLER-RETOUR alors que le moteur de backtest attend un
cout UNIDIRECTIONNEL qu'il double lui-meme. Confondre les deux doublerait toute
la grille sans qu'aucun calcul ne tombe -- et inverserait la conclusion du
chantier. C'est ce que ces tests verrouillent en premier.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import econ_backtest as eb                                            # noqa: E402
import real_fees as rf                                                # noqa: E402


# ── la conversion aller-retour / unidirectionnel ────────────────────────────

def test_one_way_is_half_the_round_trip():
    for key in rf.INSTRUMENTS:
        for lvl in rf.LEVELS:
            assert rf.one_way_bps(key, lvl) == pytest.approx(rf.round_trip_bps(key, lvl) / 2.0)


def test_the_engine_reconstructs_exactly_the_declared_round_trip():
    """Le test qui compte : une position pleine, rendement nul, doit coûter
    EXACTEMENT le frais aller-retour declare -- ni la moitie, ni le double."""
    for key, spec in rf.INSTRUMENTS.items():
        for lvl in rf.LEVELS:
            rt = rf.round_trip_bps(key, lvl)
            pnl = eb.sleeve_pnl([1.0], [0.0], cost_bps=rf.one_way_bps(key, lvl))[0]
            assert pnl == pytest.approx(-rt * 1e-4), f"{key}/{lvl}: {pnl} vs {-rt * 1e-4}"


def test_half_position_costs_half_the_round_trip():
    rt = rf.round_trip_bps("SPY-ES", "central")
    pnl = eb.sleeve_pnl([0.5], [0.0], cost_bps=rf.one_way_bps("SPY-ES", "central"))[0]
    assert pnl == pytest.approx(-0.5 * rt * 1e-4)


# ── coherence de la grille ──────────────────────────────────────────────────

def test_levels_are_ordered_for_every_instrument():
    for key in rf.INSTRUMENTS:
        vals = [rf.round_trip_bps(key, lvl) for lvl in rf.LEVELS]
        assert vals == sorted(vals), key
        assert vals[0] < vals[-1], f"{key} : niveaux bas et haut identiques"


def test_every_instrument_declares_all_three_levels():
    for key, spec in rf.INSTRUMENTS.items():
        assert set(spec["round_trip_bps"]) == set(rf.LEVELS), key


def test_futures_are_cheaper_than_etfs_which_are_cheaper_than_crypto():
    """L'ordonnancement qui fonde tout le chantier : si un ETF ressortait moins
    cher qu'un future, la grille serait fausse."""
    for lvl in rf.LEVELS:
        fut = max(rf.round_trip_bps(k, lvl) for k in ("SPY-ES", "ZN-FUT"))
        etf = min(rf.round_trip_bps(k, lvl) for k in ("SPY-ETF", "TLT-ETF", "GLD-ETF", "USO-ETF"))
        crypto = min(rf.round_trip_bps(k, lvl) for k in ("BTC-SPOT", "ETH-SPOT"))
        assert fut <= etf < crypto, lvl


def test_commodity_etfs_are_priced_between_equity_etfs_and_crypto():
    """GLD/USO : plus chers qu'un ETF actions, tres loin de la crypto."""
    for lvl in rf.LEVELS:
        for k in ("GLD-ETF", "USO-ETF"):
            assert rf.round_trip_bps("SPY-ETF", lvl) <= rf.round_trip_bps(k, lvl)
            assert rf.round_trip_bps(k, lvl) < rf.round_trip_bps("BTC-SPOT", lvl)


def test_uso_contango_is_declared_not_hidden():
    """Le cout de roulement d'USO n'est pas un frais de transaction : il doit
    etre declare comme non modelise, pas passe sous silence."""
    assert "contango" in rf.INSTRUMENTS["USO-ETF"]["caveat"].lower()


def test_two_vehicles_share_the_spy_forecasts():
    assert sorted(rf.instruments_for_asset("SPY")) == ["SPY-ES", "SPY-ETF"]
    assert rf.INSTRUMENTS["SPY-ES"]["asset"] == rf.INSTRUMENTS["SPY-ETF"]["asset"]
    # et le future doit bien etre le moins cher des deux, sinon la comparaison
    # de vehicules n'a pas d'objet
    assert rf.round_trip_bps("SPY-ES") < rf.round_trip_bps("SPY-ETF")


def test_every_asset_of_the_panel_has_at_least_one_vehicle():
    for asset in ("SPY", "TLT", "ZN=F", "BTC-USD", "ETH-USD", "GLD", "USO"):
        assert rf.instruments_for_asset(asset), asset


def test_the_spy_es_shortcut_is_declared_as_a_caveat():
    """La base ES/SPY et le roulement ne sont pas modelises : cette simplification
    est FAVORABLE au future, donc elle doit etre declaree, pas tue."""
    assert "caveat" in rf.INSTRUMENTS["SPY-ES"]
    assert "base" in rf.INSTRUMENTS["SPY-ES"]["caveat"].lower()


# ── le filtre de viabilite ──────────────────────────────────────────────────

def test_viable_instruments_at_the_decision_level():
    """Face a la borne haute de l'edge mesure (5 bps aller-retour) : les deux
    futures ET les deux ETF passent ; seule la crypto est exclue. C'est le
    resultat structurant du chantier 1 -- l'edge n'est pas hors de portee
    partout, il l'est sur la crypto."""
    viables = rf.viable_instruments()
    # Le panel d'origine, verifie nommement : c'est le resultat structurant du
    # chantier 1. Les instruments ajoutes depuis (univers 0-bis) sont couverts par
    # les invariants ci-dessous plutot que par une enumeration, qui casserait a
    # chaque ajout sans rien proteger de plus.
    for k in ("GLD-ETF", "SPY-ES", "SPY-ETF", "TLT-ETF", "ZN-FUT"):
        assert k in viables, k
    assert not any(k.endswith("SPOT") for k in viables), "la crypto ne doit jamais passer"
    # USO (5,0 bps) est EXACTEMENT a la borne : le filtre est strict (<), il l'exclut.
    assert "USO-ETF" not in viables
    # Invariant du filtre : viable <=> aller-retour strictement sous l'edge.
    for k in rf.INSTRUMENTS:
        assert (k in viables) == (rf.round_trip_bps(k) < rf.EDGE_REFERENCE_BPS), k


def test_viable_instruments_shrinks_as_the_edge_shrinks():
    assert set(rf.viable_instruments(edge_bps=5.0)) == {
        k for k in rf.INSTRUMENTS if rf.round_trip_bps(k) < 5.0}
    # La monotonie est l'invariant : abaisser l'edge ne peut que retirer des
    # instruments, jamais en ajouter.
    prev = set(rf.INSTRUMENTS)
    for edge in (100.0, 10.0, 5.1, 5.0, 3.0, 2.0, 1.2):
        cur = set(rf.viable_instruments(edge_bps=edge))
        assert cur <= prev, edge
        prev = cur
    assert "USO-ETF" in rf.viable_instruments(edge_bps=5.1)     # 5,0 < 5,1
    # A 2 bps il ne reste que les futures les plus liquides ; a 1,2 plus rien.
    assert set(rf.viable_instruments(edge_bps=2.0)) == {"SPY-ES", "ZN-FUT"}
    assert rf.viable_instruments(edge_bps=1.2) == []
    assert set(rf.viable_instruments(edge_bps=100.0)) == set(rf.INSTRUMENTS)


def test_viable_instruments_is_computed_not_hardcoded():
    """Si la grille bouge, la liste doit bouger avec elle."""
    saved = rf.INSTRUMENTS["TLT-ETF"]["round_trip_bps"]["central"]
    try:
        rf.INSTRUMENTS["TLT-ETF"]["round_trip_bps"]["central"] = 0.1
        assert "TLT-ETF" in rf.viable_instruments()
    finally:
        rf.INSTRUMENTS["TLT-ETF"]["round_trip_bps"]["central"] = saved
    assert "TLT-ETF" not in rf.viable_instruments(edge_bps=2.0)


def test_summary_table_is_complete():
    rows = rf.summary_table()
    assert len(rows) == len(rf.INSTRUMENTS)
    for r in rows:
        assert set(r["aller_retour_bps"]) == set(rf.LEVELS)
        assert r["justification"]
    assert ({r["instrument"] for r in rows if r["sous_edge_5bps_au_central"]}
            == set(rf.viable_instruments()))


def test_unknown_instrument_raises():
    with pytest.raises(KeyError):
        rf.round_trip_bps("GOLD-ETF")
