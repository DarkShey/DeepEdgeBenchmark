"""Tests unitaires de `coverage_monitor` (chantier H3). Aucune entree-sortie :
les fonctions mesurees sont pures, la base n'est jamais ouverte ici."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage_monitor import (alerts, episodes, monitor_series,      # noqa: E402
                              rolling_coverage)


# ── rolling_coverage ────────────────────────────────────────────────────────

def test_rolling_coverage_alignee_a_droite_et_amorcage_nan():
    cov = rolling_coverage([1, 1, 0, 1], window=2)
    assert np.isnan(cov[0])
    assert cov[1:].tolist() == [1.0, 0.5, 0.5]


def test_rolling_coverage_serie_plus_courte_que_la_fenetre_ne_declenche_rien():
    cov = rolling_coverage([1, 0], window=26)
    assert cov.size == 2 and np.isnan(cov).all()
    assert alerts(cov) == []


def test_rolling_coverage_couverture_parfaite_et_nulle():
    assert rolling_coverage([1] * 5, window=5)[-1] == 1.0
    assert rolling_coverage([0] * 5, window=5)[-1] == 0.0


def test_rolling_coverage_fenetre_invalide():
    with pytest.raises(ValueError):
        rolling_coverage([1, 0, 1], window=0)


def test_rolling_coverage_ne_regarde_pas_devant():
    """La valeur a t ne doit dependre que de x[:t+1] : on modifie le futur et la
    fenetre courante ne doit pas bouger."""
    base = [1, 1, 1, 0, 1, 1]
    a = rolling_coverage(base, window=3)
    b = rolling_coverage(base[:4] + [0, 0], window=3)
    assert a[3] == b[3]


# ── alerts / episodes ───────────────────────────────────────────────────────

def test_alerts_les_deux_cotes_de_la_bande():
    a = alerts([0.80, 0.95, 0.995], band=(0.88, 0.99))
    assert [x["side"] for x in a] == ["sous_couverture", "sur_couverture"]
    assert [x["index"] for x in a] == [0, 2]


def test_alerts_bornes_incluses_dans_la_bande():
    assert alerts([0.88, 0.99], band=(0.88, 0.99)) == []


def test_alerts_ignore_les_nan():
    assert alerts([np.nan, np.nan, 0.95], band=(0.88, 0.99)) == []


def test_episodes_regroupe_les_alertes_consecutives():
    eps = episodes(alerts([0.80, 0.81, 0.95, 0.70], band=(0.88, 0.99)))
    assert len(eps) == 2
    assert (eps[0]["start"], eps[0]["end"], eps[0]["length"]) == (0, 1, 2)
    assert eps[0]["worst_coverage"] == pytest.approx(0.80)
    assert eps[1]["length"] == 1


def test_episodes_ne_fusionne_pas_deux_cotes_opposes():
    eps = episodes(alerts([0.80, 0.995], band=(0.88, 0.99)))
    assert len(eps) == 2
    assert {e["side"] for e in eps} == {"sous_couverture", "sur_couverture"}


# ── monitor_series ──────────────────────────────────────────────────────────

def test_monitor_series_statut_ok():
    r = monitor_series([1] * 25 + [0], window=26, band=(0.88, 0.99))
    assert r["status"] == "OK"
    assert r["coverage_current_window"] == pytest.approx(25 / 26)
    assert r["episodes"] == []


def test_monitor_series_detecte_la_sous_couverture():
    r = monitor_series([1] * 13 + [0] * 13, window=26, band=(0.88, 0.99))
    assert r["status"] == "sous_couverture"
    assert r["coverage_current_window"] == pytest.approx(0.5)


def test_monitor_series_detecte_la_sur_couverture():
    r = monitor_series([1] * 26, window=26, band=(0.88, 0.99))
    assert r["status"] == "sur_couverture"


def test_monitor_series_fenetre_incomplete_ne_juge_pas():
    r = monitor_series([0] * 10, window=26)
    assert r["status"] == "fenetre_incomplete"
    assert r["coverage_current_window"] is None
    assert r["n_alert_origins"] == 0


def test_monitor_series_couverture_plein_echantillon_distincte_de_la_fenetre():
    """Une derive recente doit se voir dans la fenetre alors que le plein
    echantillon reste dans la bande -- c'est toute la raison d'etre du suivi."""
    serie = [1] * 100 + [0] * 13 + [1] * 13
    r = monitor_series(serie, window=26, band=(0.88, 0.99))
    assert r["coverage_full_sample"] > 0.88
    assert r["coverage_current_window"] == pytest.approx(0.5)
    assert r["status"] == "sous_couverture"


# ── la brique tourne en routine : elle doit rester legere et alignee ─────────

def test_db_path_reste_aligne_sur_la_convention_du_depot():
    """`coverage_monitor` resout DB_PATH lui-meme pour ne pas importer la pile de
    modeles dans le job quotidien. Ce test interdit que les deux divergent."""
    from backtest_rolling_tsdiffw import DB_PATH as CANONICAL
    from coverage_monitor import DB_PATH
    assert DB_PATH == CANONICAL


def test_le_monitor_n_importe_pas_la_pile_de_modeles():
    """Garde-fou de routine : importer le monitor ne doit charger ni torch, ni
    tensorflow, ni statsmodels. Un job quotidien qui installe la pile complete
    pour lire une couverture est un job qui tombera pour une raison sans rapport."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import coverage_monitor; "
            "print(','.join(m for m in ('torch','tensorflow','statsmodels','yfinance') "
            "if m in sys.modules))" % str(Path(__file__).resolve().parent))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"modules lourds charges : {out.stdout.strip()}"


# ── le resume de routine ────────────────────────────────────────────────────

def test_episode_en_cours_ignore_un_episode_clos():
    """Un episode qui s'est termine avant la derniere origine appartient au
    passe : il n'appelle aucune action aujourd'hui et ne doit pas etre compte."""
    from coverage_monitor_summary import current_episode_length
    clos = {"n": 10, "episodes": [{"start": 2, "end": 4, "length": 3,
                                   "side": "sous_couverture", "worst_coverage": 0.5}]}
    assert current_episode_length(clos) == 0


def test_episode_en_cours_compte_celui_qui_touche_la_derniere_origine():
    from coverage_monitor_summary import current_episode_length
    en_cours = {"n": 10, "episodes": [{"start": 7, "end": 9, "length": 3,
                                       "side": "sous_couverture", "worst_coverage": 0.5}]}
    assert current_episode_length(en_cours) == 3


def test_episode_en_cours_sans_episode():
    from coverage_monitor_summary import current_episode_length
    assert current_episode_length({"n": 10, "episodes": []}) == 0
