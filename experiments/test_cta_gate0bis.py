"""
Tests du chantier 0-bis (BRIEF_re_porte_cta_corrige_univers_deita.md) : la
declaration d'univers, la grille de marche construite sur les prix, et l'IC
bootstrap par blocs du Sharpe.

Ce que ces tests protegent : les DECLARATIONS. Un brief qui fige un univers, des
substitutions et quatre classes avant les runs ne vaut que si rien ne peut les
deplacer ensuite sans qu'un test tombe.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cta_gate0bis as g0b                                            # noqa: E402
import prices_v4 as p4                                                # noqa: E402
import real_fees as rf                                                # noqa: E402
from prices_v3 import PANEL as PANEL_V3                               # noqa: E402


# ── la declaration d'univers ────────────────────────────────────────────────

def test_le_panel_du_benchmark_est_un_sous_ensemble_strict():
    """C'est le motif declare de la regle de substitution : le chantier 1
    conditionnel ne peut trader que les instruments du panel (les nuages NsDiff
    n'existent que la). Si le panel n'etait pas inclus, il faudrait re-geler."""
    assert set(PANEL_V3) <= set(p4.PANEL)


def test_une_seule_convention_par_exposition():
    """« pas les deux » : les futures substitues ne doivent pas coexister avec
    leurs ETF de remplacement dans l'univers."""
    for remplace, spec in p4.SUBSTITUTIONS.items():
        assert remplace not in p4.PANEL, remplace
        assert spec["remplace_par"] in p4.PANEL


def test_chaque_actif_a_une_classe_ou_est_declare_hors_classes():
    for a, v in p4.PANEL.items():
        assert v["classe"] in p4.CLASSES or v["classe"] is None, a
        if v["classe"] is None:
            assert "note" in v, f"{a} hors classes doit porter sa justification"


def test_les_quatre_classes_sont_toutes_peuplees():
    """Une branche 2 qui exige 3 classes sur 4 n'a de sens que si les quatre
    existent reellement."""
    peuplees = {v["classe"] for v in p4.PANEL.values() if v["classe"]}
    assert peuplees == set(p4.CLASSES)


def test_vxx_est_hors_classes_et_reste_dans_l_univers():
    """Declare avant tout calcul : VXX participe a la conviction (moteur DEITA
    tel quel) mais ne contamine pas le test de classes."""
    assert p4.PANEL["VXX"]["classe"] is None
    assert "VXX" in p4.ASSET_MAP


def test_la_carte_de_conviction_couvre_tout_l_univers():
    assert set(p4.ASSET_MAP) == set(p4.PANEL)
    for v in p4.ASSET_MAP.values():
        assert v["sector"] and v["subsector"]


# ── la grille de marche ─────────────────────────────────────────────────────

def _fake_prices(tmp_path: Path, assets, n: int = 500) -> Path:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2019-06-03", periods=n)
    for a in assets:
        s = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        s.to_frame("close").to_parquet(tmp_path / f"{p4.slug(a)}.parquet")
    return tmp_path


def test_build_market_ne_produit_que_des_origines_apres_le_depart(tmp_path):
    d = _fake_prices(tmp_path, ["SPY", "TLT"])
    m = g0b.build_market(["SPY", "TLT"], d)
    assert (pd.to_datetime(m["cutoff_date"]) >= pd.Timestamp("2020-01-01")).all()


def test_build_market_cible_posterieure_a_l_origine(tmp_path):
    d = _fake_prices(tmp_path, ["SPY"])
    m = g0b.build_market(["SPY"], d)
    assert (pd.to_datetime(m["target_date"]) > pd.to_datetime(m["cutoff_date"])).all()


def test_build_market_laisse_la_marge_pour_l_horizon(tmp_path):
    """Les trois dernieres origines sont ecartees : sans marge, la cible W+3
    n'existerait pas -- meme convention que la grille oos2020."""
    d = _fake_prices(tmp_path, ["SPY"])
    daily = pd.read_parquet(d / "SPY.parquet")["close"]
    from weekly_headtohead import build_weekly
    weekly, wd = build_weekly(daily)
    n_after = sum(1 for x in wd if x >= pd.Timestamp("2020-01-01"))
    assert g0b.build_market(["SPY"], d)["cutoff_date"].nunique() == n_after - 3


def test_build_market_ne_lit_aucun_modele(tmp_path):
    """Une porte de CTA n'a besoin que du prix courant et du prix a echeance :
    la grille doit se construire sans aucune bande de modele."""
    d = _fake_prices(tmp_path, ["SPY"])
    m = g0b.build_market(["SPY"], d)
    assert set(m.columns) == {"asset", "frequence", "horizon_unit", "cutoff_date",
                              "target_date", "last_close", "y_true"}


# ── l'IC bootstrap par blocs ────────────────────────────────────────────────

def test_ic_bootstrap_encadre_le_point():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 300)
    r = g0b.block_bootstrap_sharpe_ci(x, 1, n_boot=400)
    assert r["ci"][0] <= r["point"] <= r["ci"][1]


def test_ic_bootstrap_exclut_zero_sur_un_signal_franc():
    rng = np.random.default_rng(2)
    x = rng.normal(0.004, 0.004, 400)          # Sharpe elevé, sans ambiguite
    r = g0b.block_bootstrap_sharpe_ci(x, 1, n_boot=600)
    assert r["ci"][0] > 0 and r["share_positive"] == 1.0


def test_ic_bootstrap_traverse_zero_sur_du_bruit():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.01, 400)
    r = g0b.block_bootstrap_sharpe_ci(x, 1, n_boot=600)
    assert r["ci"][0] < 0 < r["ci"][1]


def test_ic_bootstrap_serie_trop_courte():
    r = g0b.block_bootstrap_sharpe_ci(np.array([0.01, 0.02]), 1)
    assert r["n_boot"] == 0 and np.isnan(r["point"])


def test_ic_bootstrap_est_reproductible():
    rng = np.random.default_rng(4)
    x = rng.normal(0.001, 0.01, 200)
    a = g0b.block_bootstrap_sharpe_ci(x, 1, n_boot=300)
    b = g0b.block_bootstrap_sharpe_ci(x, 1, n_boot=300)
    assert a["ci"] == b["ci"]


# ── les frais couvrent l'univers ────────────────────────────────────────────

def test_un_seul_instrument_par_actif_sauf_spy():
    """Deux vehicules du meme actif portent le MEME signal ; le portefeuille n'en
    retient qu'un pour ne pas ponderer cet actif double. SPY est le seul cas."""
    from collections import Counter
    c = Counter(rf.INSTRUMENTS[i]["asset"] for i in rf.INSTRUMENTS
                if rf.INSTRUMENTS[i]["asset"] in p4.PANEL)
    assert [a for a, n in c.items() if n > 1] == ["SPY"]
