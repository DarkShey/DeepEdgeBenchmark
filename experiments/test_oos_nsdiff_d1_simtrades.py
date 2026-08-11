"""Tests unitaires du générateur NsDiff D+1 sur la grille sim_trades
(BRIEF_retest_nsdiff_tc_sim_trades.md).

Aucun fit réel : le modèle NsDiff est remplacé par un stub déterministe. Ce qui
est testé ici, c'est le CÂBLAGE -- alignement des clés, héritage de ref/realized,
agrégation de l'ensemble, et surtout l'absence de look-ahead -- pas la qualité
prédictive du modèle, qui ne se teste pas en unitaire.

Le test central est `test_look_ahead_*` : il vérifie que tronquer la série juste
après d_date ne change pas la prévision, ET (contre-épreuve) qu'une fuite
délibérément injectée EST bien détectée -- sans quoi le premier test pourrait
passer pour de mauvaises raisons.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oos_nsdiff_d1_simtrades as g                                    # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

def _daily(n=60, start="2026-01-01", base=100.0):
    """Série quotidienne strictement croissante -- chaque prix est unique, donc
    toute confusion d'indice se voit immédiatement."""
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series([base + i for i in range(n)], index=idx, dtype=float)


def _grid(daily, positions):
    """Grille (d_date, target_date, ref, realized) construite depuis la série --
    ref/realized sont ici volontairement DÉCALÉS de la série (+1000) pour prouver
    qu'ils sont hérités et jamais recalculés depuis les prix."""
    rows = []
    for pos in positions:
        rows.append({
            "asset": "TEST", "d_date": daily.index[pos].strftime("%Y-%m-%d"),
            "target_date": daily.index[pos + 1].strftime("%Y-%m-%d"),
            "reference_price": float(daily.iloc[pos]) + 1000.0,
            "realized_price": float(daily.iloc[pos + 1]) + 1000.0,
        })
    return pd.DataFrame(rows)


class _StubModel:
    seq_len = 5
    horizon = 15


@pytest.fixture
def stub_forecast(monkeypatch):
    """Remplace fit_nsdiff et forecast_from_fitted par des stubs déterministes.

    `forecast_from_fitted` devient une fonction PURE de (fenêtre, ancrage) : la
    sortie encode la longueur de la fenêtre, sa somme et le last_price -- toute
    modification de l'information vue par le modèle change donc la sortie, ce
    qui est précisément ce que les tests de look-ahead doivent pouvoir détecter.

    Le fit est stubé aussi : ces tests portent sur le câblage, pas sur
    l'entraînement (et un vrai fit exigerait > seq_len + horizon points). La
    fenêtre d'entraînement réellement transmise est enregistrée dans
    `fits`, pour être vérifiée par test_fit_asset_*."""
    fits = []

    def fake_fit(train, horizon=None, epochs=None, k_denoise=None, **_):
        fits.append(train)
        return _StubModel(), 0.0, 1.0

    def fake(model, hist_window, mu, sd, last_price, horizons=None, n_samples=4, **_):
        w = np.asarray(hist_window, dtype=float)
        sig = float(w.size) * 1e6 + float(np.nansum(w)) + float(last_price)
        return {1: np.full(n_samples, sig)}

    monkeypatch.setattr(g.nm, "fit_nsdiff", fake_fit)
    monkeypatch.setattr(g.nm, "forecast_from_fitted", fake)
    monkeypatch.setattr(g.nm, "set_seed", lambda *_a, **_k: None)
    return fits


# ── grille et héritage ──────────────────────────────────────────────────────

def test_load_grid_refuse_une_grille_vide(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, model TEXT, source TEXT, "
                 "horizon INT, horizon_type TEXT, daily_duplicate INT, cutoff_date TEXT, "
                 "target_date TEXT, last_close REAL, y_pred REAL, y_lower REAL, y_upper REAL, "
                 "y_true REAL, asset TEXT, run_id TEXT, regime TEXT)")
    conn.execute("CREATE VIEW all_predictions AS SELECT id, run_id, model, asset, horizon, regime, "
                 "cutoff_date AS d_date, target_date, last_close AS reference_price, "
                 "y_pred AS predicted, y_lower AS pi_lower, y_upper AS pi_upper, "
                 "y_true AS realized_price, source FROM predictions WHERE horizon=1 "
                 "AND horizon_type='daily' AND daily_duplicate=0")
    conn.commit()
    conn.close()
    with pytest.raises(g.GridMismatchError, match="aucune ligne oos"):
        g.load_grid(str(db))


def test_build_ensemble_rows_herite_ref_realized_et_ne_les_recalcule_pas(stub_forecast):
    daily = _daily()
    grid = _grid(daily, [40, 41])
    clouds = {s: {d: [1.0, 2.0, 3.0] for d in grid["d_date"]} for s in g.SEEDS}
    rows = g.build_ensemble_rows("TEST", grid, clouds)
    for row, (_, src) in zip(rows, grid.iterrows()):
        assert row["last_close"] == src["reference_price"]     # hérité tel quel
        assert row["y_true"] == src["realized_price"]
        assert row["target_date"] == src["target_date"]
        assert row["model"] == "NsDiff" and row["source"] == "oos"
        assert row["horizon"] == 1 and row["horizon_type"] == "daily"


def test_build_ensemble_rows_concatene_les_nuages_et_lit_mediane_et_quantiles():
    """L'ensemble doit CONCATÉNER les 5 nuages puis lire les quantiles dessus --
    jamais moyenner 5 bornes déjà quantilées (convention actée par
    repoint_oos_to_ensemble.py)."""
    daily = _daily()
    grid = _grid(daily, [40])
    d = grid["d_date"].iloc[0]
    clouds = {s: {d: [float(s) * 10 + i for i in range(200)]} for s in g.SEEDS}
    row = g.build_ensemble_rows("TEST", grid, clouds)[0]

    expected = np.concatenate([np.asarray(clouds[s][d]) for s in g.SEEDS])
    assert row["n_samples_total"] == 1000 == expected.size
    assert row["y_pred"] == pytest.approx(float(np.median(expected)))
    lo, hi = np.quantile(expected, [0.025, 0.975])
    assert (row["y_lower"], row["y_upper"]) == pytest.approx((lo, hi))
    # la moyenne des bornes par graine donnerait autre chose : on vérifie qu'on
    # n'a pas fait ça par accident
    moyenne_des_bornes = np.mean([np.quantile(clouds[s][d], 0.025) for s in g.SEEDS])
    assert row["y_lower"] != pytest.approx(moyenne_des_bornes)


# ── audit des prix ──────────────────────────────────────────────────────────

def test_audit_signale_une_date_absente_de_la_serie():
    daily = _daily()
    grid = _grid(daily, [40])
    grid.loc[0, "d_date"] = "2030-01-01"
    audit = g.audit_grid_prices(daily, grid)
    assert audit["n_missing"] == 1 and audit["missing"] == ["2030-01-01"]


def test_audit_signale_un_niveau_divergent_et_chiffre_l_ecart():
    daily = _daily()
    grid = _grid(daily, [40])
    grid.loc[0, "reference_price"] = float(daily.iloc[40]) * 1.01   # +1 %
    audit = g.audit_grid_prices(daily, grid)
    assert audit["n_mismatched"] == 1
    assert audit["max_abs_rel_pct"] == pytest.approx(1.0, abs=0.02)


def test_verify_grid_prices_est_bloquant_quand_une_cle_manque():
    daily = _daily()
    grid = _grid(daily, [40])
    grid.loc[0, "d_date"] = "2030-01-01"
    with pytest.raises(g.GridMismatchError, match="absente"):
        g.verify_grid_prices(daily, grid, "TEST")


# ── look-ahead : le test central, et sa contre-épreuve ──────────────────────

def test_look_ahead_la_troncature_apres_d_date_ne_change_rien(stub_forecast):
    """Une prévision qui ne regarde pas le futur doit être IDENTIQUE, bit à bit,
    qu'on lui donne la série complète ou tronquée juste après d_date."""
    daily = _daily(n=60)
    grid = _grid(daily, [40, 45, 50])
    picked = g.verify_no_lookahead("TEST", daily, grid, seed=42, n_dates=3,
                                   n_samples=4, epochs=1)
    assert picked[0] == grid["d_date"].iloc[0]      # première origine incluse
    assert picked[-1] == grid["d_date"].iloc[-1]    # dernière aussi


def test_look_ahead_contre_epreuve_une_fuite_injectee_EST_detectee(stub_forecast, monkeypatch):
    """Contre-épreuve exigée par le brief : si la fenêtre de conditionnement
    contenait du futur, `verify_no_lookahead` doit lever. Sans ce test, le
    précédent pourrait passer parce qu'il ne discrimine rien."""
    daily = _daily(n=60)
    grid = _grid(daily, [40, 45, 50])

    vrai = g._standardized_returns

    def fuite(prices, mu, sd):
        # renverse la série : la tranche [:pos] ramasse alors la FIN de
        # l'historique, c'est-à-dire des rendements postérieurs à d_date.
        return vrai(prices, mu, sd)[::-1]

    monkeypatch.setattr(g, "_standardized_returns", fuite)
    with pytest.raises(g.GridMismatchError, match="LOOK-AHEAD"):
        g.verify_no_lookahead("TEST", daily, grid, seed=42, n_dates=3,
                              n_samples=4, epochs=1)


def test_fit_asset_entraine_strictement_avant_la_premiere_origine(stub_forecast):
    """Train-once-forward : le fit ne doit voir AUCUN prix daté de la première
    origine ou après -- c'est là que se joue la distance train->test déclarée."""
    daily = _daily(n=60)
    grid = _grid(daily, [40, 45, 50])
    g.fit_asset(daily, grid, seed=42, epochs=1)

    train = stub_forecast[-1]
    premiere_origine = pd.Timestamp(grid["d_date"].iloc[0])
    assert len(train) == 40
    assert train.index.max() < premiere_origine


def test_la_fenetre_s_arrete_a_d_date_incluse(stub_forecast, monkeypatch):
    """Contrôle direct de la tranche : à l'origine d'indice `pos`, le dernier
    rendement vu doit être log(p[pos]/p[pos-1]) -- ni moins, ni plus."""
    daily = _daily(n=60)
    grid = _grid(daily, [40])
    vues = {}

    def capture(model, hist_window, mu, sd, last_price, horizons=None, n_samples=4, **_):
        vues["n"] = len(np.asarray(hist_window))
        return {1: np.zeros(n_samples)}

    monkeypatch.setattr(g.nm, "forecast_from_fitted", capture)
    g.forecast_grid(_StubModel(), 0.0, 1.0, daily, grid, seed=42, n_samples=4)
    assert vues["n"] == 40      # rendements d'indices 0..39 == prix 0..40 inclus


# ── empreinte de non-régression ─────────────────────────────────────────────

def _mini_db(path):
    """Base minimale contenant les trois voisinages à surveiller : un autre
    modèle, les lignes NsDiff PRÉEXISTANTES (hebdo/live), et le run courant."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, run_id TEXT, model TEXT, y_pred REAL)")
    conn.execute("CREATE TABLE sim_trades (id INTEGER PRIMARY KEY, run_id TEXT, model TEXT, roi REAL)")
    conn.executemany("INSERT INTO predictions (run_id, model, y_pred) VALUES (?,?,?)",
                     [("20260804-arima", "ARIMA-GARCH", 1.0),
                      ("20260808-oos-repoint-ensemble", "NsDiff", 2.0)])
    conn.executemany("INSERT INTO sim_trades (run_id, model, roi) VALUES (?,?,?)",
                     [("20260804-arima", "ARIMA-GARCH", 0.1),
                      ("20260808-oos-repoint-ensemble", "NsDiff", 0.2)])
    conn.commit()
    conn.close()


def test_empreinte_ignore_le_run_courant_mais_capte_tout_le_reste(tmp_path):
    db = tmp_path / "t.db"
    _mini_db(db)
    avant = g.fingerprint_untouched(str(db))

    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO predictions (run_id, model, y_pred) VALUES (?, 'NsDiff', 99.0)",
                 (g.RUN_ID,))
    conn.commit(); conn.close()
    assert g.fingerprint_untouched(str(db)) == avant, \
        "insérer les lignes DU RUN ne doit pas faire bouger l'empreinte"

    conn = sqlite3.connect(db)
    conn.execute("UPDATE predictions SET y_pred = 42.0 WHERE model = 'ARIMA-GARCH'")
    conn.commit(); conn.close()
    assert g.fingerprint_untouched(str(db)) != avant, "toucher un autre modèle DOIT se voir"


def test_empreinte_capte_une_atteinte_aux_lignes_nsdiff_preexistantes(tmp_path):
    """Le point que la version « tout sauf NsDiff » aurait manqué : les lignes
    oos HEBDO de NsDiff (ensemble 5x200 repointé) sont le voisinage le plus
    exposé à un upsert trop large -- elles doivent être surveillées."""
    db = tmp_path / "t.db"
    _mini_db(db)
    avant = g.fingerprint_untouched(str(db))

    conn = sqlite3.connect(db)
    conn.execute("UPDATE predictions SET y_pred = 7.0 "
                 "WHERE run_id = '20260808-oos-repoint-ensemble'")
    conn.commit(); conn.close()
    assert g.fingerprint_untouched(str(db)) != avant
