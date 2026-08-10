"""
Tests des parametres OPT-IN ajoutes au chantier R2 (BRIEF « regeneration oos et
famille 3 ») sur des briques partagees, plus la logique propre a
`grid2020_refs`.

Ce que ces tests protegent, et c'est le point : chaque parametre ajoute a un
fichier partage doit laisser le chemin historique STRICTEMENT inchange. Un test
par parametre verifie le comportement neuf ; un test par parametre verifie que
sans lui, rien ne bouge.
"""

import sys
from pathlib import Path

# `weekly_multimodel` importe tensorflow avant yfinance/statsmodels et doit donc
# venir en premier, comme partout ailleurs dans le depot (deadlock sinon).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weekly_multimodel as wmm                                       # noqa: E402,I001

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import pytest                                                         # noqa: E402

import grid2020_refs as refs                                          # noqa: E402
import grid2020_tests as g2t                                          # noqa: E402


def _synthetic_daily(n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx, name="close")


def _flat_forecast(train: pd.Series, horizons: list) -> dict:
    """Bras de prevision imposable : point = dernier prix, bande +/- 1 %."""
    last = float(train.iloc[-1])
    return {h: (last, last * 0.99, last * 1.01) for h in horizons}


# ── validation_origins : le bloc de selection ne touche pas la grille de test ──

def test_validation_origins_finit_avant_la_grille_de_test_marge_w3_comprise():
    test_pos = list(range(100, 140))
    val = refs.validation_origins(test_pos, n_val=12)
    assert len(val) == 12
    assert val[-1] + refs.WEEK_MARGIN < test_pos[0]


def test_validation_origins_est_contigu_et_croissant():
    val = refs.validation_origins(list(range(50, 90)), n_val=5)
    assert val == list(range(val[0], val[0] + 5))


def test_validation_origins_refuse_un_historique_trop_court():
    with pytest.raises(SystemExit):
        refs.validation_origins([5, 6, 7], n_val=12)


# ── to_bands : la conversion vers le format de la grille ─────────────────────

def test_to_bands_mappe_les_colonnes_et_l_horizon():
    records = [{"asset": "SPY", "horizon": "W+2", "model": "SARIMA", "regime": "C",
                "origin": 0, "origin_date": "2020-01-03", "target_date": "2020-01-17",
                "daily_steps": None, "last_close": 100.0, "actual": 101.0,
                "point": 100.5, "lower": 99.0, "upper": 102.0, "sigma_scale": 1.0}]
    df = refs.to_bands(records, "weekly")
    assert list(df.columns) == refs.BAND_COLS
    r = df.iloc[0]
    assert (r["horizon"], r["horizon_unit"], r["frequence"]) == (2, "W+2", "weekly")
    assert (r["y_pred"], r["y_lower"], r["y_upper"], r["y_true"]) == (100.5, 99.0, 102.0, 101.0)
    assert r["cutoff_date"] == "2020-01-03"


# ── run_model_asset : les trois parametres opt-in ────────────────────────────

def test_serie_gelee_et_origines_imposees_ne_touchent_pas_au_reseau():
    daily = _synthetic_daily()
    weekly, _ = wmm.build_weekly(daily)
    test_pos = [len(weekly) - 8, len(weekly) - 7]
    res = wmm.run_model_asset("Naive", "SYN", "SYN", "C", n_val=0, n_test=0,
                              start=None, end=None, daily=daily, test_pos=test_pos,
                              forecast_fn=_flat_forecast)
    assert res["n_failed"] == 0
    assert len(res["records"]) == 3 * len(test_pos)
    assert {r["origin_date"] for r in res["records"]} == {
        str(weekly.index[m].date()) for m in test_pos}


def test_forecast_fn_impose_prend_le_pas_sur_le_registre_de_regime():
    daily = _synthetic_daily()
    weekly, _ = wmm.build_weekly(daily)
    res = wmm.run_model_asset("SARIMA", "SYN", "SYN", "C", 0, 0, None, None,
                              daily=daily, test_pos=[len(weekly) - 6],
                              forecast_fn=_flat_forecast, calibrate_sigma="off")
    r = res["records"][0]
    assert r["point"] == pytest.approx(r["last_close"])
    assert r["lower"] == pytest.approx(r["last_close"] * 0.99)


def test_t0_est_l_origine_qui_precede_la_grille_imposee():
    daily = _synthetic_daily()
    weekly, weekly_dates = wmm.build_weekly(daily)
    first = len(weekly) - 8
    res = wmm.run_model_asset("Naive", "SYN", "SYN", "C", 0, 0, None, None,
                              daily=daily, test_pos=[first, first + 1],
                              forecast_fn=_flat_forecast)
    assert res["T0"] == str(weekly_dates.iloc[first - 1].date())


def test_calibration_sigma_off_laisse_la_bande_brute():
    daily = _synthetic_daily()
    weekly, _ = wmm.build_weekly(daily)
    res = wmm.run_model_asset("Naive", "SYN", "SYN", "C", 0, 0, None, None,
                              daily=daily, test_pos=[len(weekly) - 8, len(weekly) - 7],
                              forecast_fn=_flat_forecast, calibrate_sigma="off")
    assert all(r["sigma_scale"] == 1.0 for r in res["records"])


def test_la_premiere_origine_sort_toujours_la_bande_brute_meme_calibree():
    """Lag de resolution : a la premiere origine, l'etat EWMA est encore neutre."""
    daily = _synthetic_daily()
    weekly, _ = wmm.build_weekly(daily)
    res = wmm.run_model_asset("Naive", "SYN", "SYN", "C", 0, 0, None, None,
                              daily=daily, test_pos=[len(weekly) - 10, len(weekly) - 9],
                              forecast_fn=_flat_forecast, calibrate_sigma="on")
    assert res["records"][0]["sigma_scale"] == 1.0


# ── nsdiff_bands(point=...) : mediane vs moyenne ─────────────────────────────

def _rows_and_samples():
    rows = pd.DataFrame({
        "asset": ["SPY"] * 2, "frequence": ["weekly"] * 2, "horizon_unit": ["W+1"] * 2,
        "cutoff_date": ["2020-01-03"] * 2, "target_date": ["2020-01-10"] * 2,
        "y_true": [101.0] * 2, "last_close": [100.0] * 2, "seed": [42, 43],
    })
    # nuage franchement asymetrique : mediane et moyenne doivent differer
    samples = np.array([[1.0, 1.0, 1.0, 100.0], [1.0, 1.0, 1.0, 100.0]], dtype=float)
    return rows, samples


def test_nsdiff_bands_defaut_reste_la_moyenne():
    rows, samples = _rows_and_samples()
    out = g2t.nsdiff_bands(rows, samples, 0.95)
    assert out["y_pred"].iloc[0] == pytest.approx(samples.mean())


def test_nsdiff_bands_point_median_lit_bien_la_mediane():
    rows, samples = _rows_and_samples()
    out = g2t.nsdiff_bands(rows, samples, 0.95, point="median")
    assert out["y_pred"].iloc[0] == pytest.approx(np.median(samples))
    assert out["y_pred"].iloc[0] != pytest.approx(samples.mean())


def test_nsdiff_bands_les_bornes_ne_dependent_pas_du_point():
    rows, samples = _rows_and_samples()
    a = g2t.nsdiff_bands(rows, samples, 0.95)
    b = g2t.nsdiff_bands(rows, samples, 0.95, point="median")
    assert (a["y_lower"].iloc[0], a["y_upper"].iloc[0]) == (b["y_lower"].iloc[0],
                                                            b["y_upper"].iloc[0])


# ── normalise_horizon : le pont entre les deux conventions d'etiquetage ──────

def _band_frame(labels):
    return pd.DataFrame({
        "asset": ["SPY"] * len(labels), "frequence": ["weekly"] * len(labels),
        "horizon": [0] * len(labels), "horizon_unit": labels,
        "cutoff_date": ["2020-01-03"] * len(labels),
        "target_date": ["2020-01-10"] * len(labels),
        "last_close": [100.0] * len(labels), "y_pred": [100.0] * len(labels),
        "y_lower": [99.0] * len(labels), "y_upper": [101.0] * len(labels),
        "y_true": [100.5] * len(labels), "sigma_scale": [1.0] * len(labels),
    })


def test_normalise_horizon_convertit_w1_en_wplus1_et_deduit_l_entier():
    out = refs.normalise_horizon(_band_frame(["W1", "W2", "W3"]))
    assert out["horizon_unit"].tolist() == ["W+1", "W+2", "W+3"]
    assert out["horizon"].tolist() == [1, 2, 3]


def test_normalise_horizon_est_idempotent():
    once = refs.normalise_horizon(_band_frame(["W1", "W2", "W3"]))
    twice = refs.normalise_horizon(once)
    pd.testing.assert_frame_equal(once, twice)


def test_normalise_horizon_refuse_un_etiquetage_inconnu():
    with pytest.raises(SystemExit):
        refs.normalise_horizon(_band_frame(["D+1"]))


def test_to_bands_sort_la_convention_de_la_base():
    """Le bug attrape par la verification 1:1 : les modeles de reference
    etiquettent « W1 », la grille et tracking.db attendent « W+1 »."""
    records = [{"asset": "SPY", "horizon": "W1", "model": "SARIMA", "regime": "C",
                "origin": 0, "origin_date": "2020-01-03", "target_date": "2020-01-10",
                "daily_steps": None, "last_close": 100.0, "actual": 101.0,
                "point": 100.5, "lower": 99.0, "upper": 102.0, "sigma_scale": 1.0}]
    df = refs.to_bands(records, "weekly")
    assert df["horizon_unit"].iloc[0] == "W+1"
    assert df["horizon"].iloc[0] == 1
