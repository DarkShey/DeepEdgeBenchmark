"""Tests du marquage des cellules a defaut permanent de couverture
(option 2 de DECISION_derive_couverture_daily.md).

Deux familles distinctes, volontairement :
  * des cas CONSTRUITS (purs, aucune base ouverte) pour la regle elle-meme --
    notamment la symetrie sous/sur-couverture et le refus de marquer une derive ;
  * deux cas REELS lus dans tracking.db en lecture seule (skip si la base est
    absente) -- parce qu'une brique de protection du lecteur qui ne detecterait
    pas le cas qui l'a fait ecrire (Prophet/BTC daily, 28,9 % de couverture) ne
    protegerait personne.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage_monitor import (BAND, DB_PATH, monitor_series,             # noqa: E402
                              permanent_defect, permanent_defect_cells)

DB_MISSING = not Path(DB_PATH).exists()


# ── la regle, sur cas construits ────────────────────────────────────────────

def test_symetrie_sous_et_sur_couverture():
    """Les deux cotes sont marques, et le `side` est lu sur le plein echantillon."""
    sous = permanent_defect(monitor_series([0] * 30, window=26, band=BAND), BAND)
    sur = permanent_defect(monitor_series([1] * 30, window=26, band=BAND), BAND)
    assert sous["side"] == "sous_couverture" and sous["coverage_full_sample"] == 0.0
    assert sur["side"] == "sur_couverture" and sur["coverage_full_sample"] == 1.0
    assert sous["n_origins"] == sur["n_origins"] == 30


def test_une_cellule_dans_la_bande_n_est_pas_marquee():
    """Couverture ~95 % partout : ni alerte, ni defaut."""
    serie = [1] * 19 + [0] + [1] * 19 + [0]
    assert permanent_defect(monitor_series(serie, window=26, band=BAND), BAND) is None


def test_une_derive_n_est_jamais_marquee():
    """Plein echantillon DANS la bande, fenetre glissante hors bande : c'est une
    derive (probleme B de la note), pas un defaut permanent (probleme A). La
    marquer melangerait les deux problemes que la note a separes."""
    r = monitor_series([1] * 100 + [0] * 13 + [1] * 13, window=26, band=BAND)
    assert r["status"] == "sous_couverture"          # bien en alerte...
    assert permanent_defect(r, BAND) is None         # ...mais pas un defaut permanent


def test_une_fenetre_incomplete_ne_marque_rien():
    assert permanent_defect(monitor_series([0] * 10, window=26, band=BAND), BAND) is None


# ── la bande vit en UN seul endroit ─────────────────────────────────────────

def test_la_bande_du_marquage_est_celle_du_suivi_h3():
    """Si le dashboard redeclarait sa propre bande, une cellule pourrait etre en
    alerte dans le job quotidien et « fiable » sur la page, sans que rien ne le
    signale. Ce test l'interdit."""
    import coverage_monitor
    import dashboard_d7_w1

    assert dashboard_d7_w1.COVERAGE_BAND is coverage_monitor.BAND
    assert dashboard_d7_w1.COVERAGE_WINDOW is coverage_monitor.WINDOW
    # et la valeur publiee dans le payload est bien celle-la, pas une copie derivee
    src = Path(dashboard_d7_w1.__file__).read_text()
    assert "0.88" not in src and "0.99" not in src, \
        "la bande de couverture est recopiee en dur dans dashboard_d7_w1.py"


# ── les deux cas reels ──────────────────────────────────────────────────────

@pytest.mark.skipif(DB_MISSING, reason=f"tracking.db absent ({DB_PATH})")
def test_prophet_btc_daily_est_detectee():
    """Le cas qui justifie la brique : ~29-33 % de couverture pour une cible de
    95 %, sur les trois horizons, sur toute la fenetre disponible."""
    cells = {c["key"]: c for c in permanent_defect_cells(source="oos")}
    for hu in ("W+1", "W+2", "W+3"):
        c = cells.get(f"Prophet|BTC-USD|daily|{hu}")
        assert c is not None, f"Prophet/BTC-USD/daily/{hu} devrait etre marquee"
        assert c["side"] == "sous_couverture"
        assert c["coverage_full_sample"] < 0.5


@pytest.mark.skipif(DB_MISSING, reason=f"tracking.db absent ({DB_PATH})")
def test_une_cellule_saine_n_est_pas_marquee():
    """ARIMA-GARCH est le seul modele a ZERO defaut permanent en daily (note de
    decision, §1 : Naive 10, Prophet 9, SARIMA 9, LSTM 6, NsDiff 1, ARIMA-GARCH 0)
    -- c'est le contre-exemple qui montre que le marquage ne marque pas tout.
    (Le regime weekly n'est pas concerne par ce compte : ARIMA-GARCH y porte une
    cellule marquee, ETH-USD/W+3.)"""
    marked = {c["key"] for c in permanent_defect_cells(source="oos")}
    daily_arima = [k for k in marked if k.startswith("ARIMA-GARCH|") and "|daily|" in k]
    assert daily_arima == [], f"cellules ARIMA-GARCH daily marquees a tort : {daily_arima}"
    assert "NsDiff|SPY|daily|W+1" not in marked
