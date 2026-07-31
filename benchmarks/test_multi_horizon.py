"""
benchmarks/test_multi_horizon.py — Chantier 2 de BRIEF_branchement_prod_calibration_sigma.md.

Couvre le branchement des adoptions sigma de models/*.py sur le chemin live/backtest D+7
(benchmarks/multi_horizon.py, cf. sa docstring de module) :
  - ARIMA-GARCH : dist="normal" reproduit EXACTEMENT (bit-for-bit) l'ancien calcul
    symétrique +/- Z_95*sigma ; le défaut (skew-t) donne des bornes asymétriques.
  - SARIMA / Naive / LSTM : sigma_scale absent -> bornes brutes inchangées (bit-for-bit) ;
    présent -> corrige multiplicativement UNIQUEMENT l'horizon ciblé.
  - Prophet : log_space=False + sigma_scale=None reproduit l'ancien calcul (bornes
    directement en espace prix) ; log_space=True (défaut) donne des bornes positives.

Pas d'accès réseau : séries synthétiques déterministes (même convention que
models/conftest.py : marche aléatoire + léger drift, toujours positive).
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks import multi_horizon as mh
import arima_model
import sarima_model
import naive_model


def synthetic_series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.Series(prices, index=idx, name="Close")


# ── _scaled_bounds (helper pur, partagé SARIMA/Naive/LSTM) ──────────────────────

def test_scaled_bounds_none_returns_raw_unchanged():
    lo, hi = mh._scaled_bounds(100.0, 95.0, 107.0, h=1, sigma_scale=None)
    assert (lo, hi) == (95.0, 107.0)


def test_scaled_bounds_missing_horizon_returns_raw_unchanged():
    lo, hi = mh._scaled_bounds(100.0, 95.0, 107.0, h=7, sigma_scale={1: 2.0})
    assert (lo, hi) == (95.0, 107.0)


def test_scaled_bounds_applies_multiplicative_correction():
    # point=100, lo_raw=95 (half-width 5), hi_raw=107 (half-width 7), corr=2.0
    lo, hi = mh._scaled_bounds(100.0, 95.0, 107.0, h=1, sigma_scale={1: 2.0})
    assert lo == pytest.approx(100.0 - 5.0 * 2.0)
    assert hi == pytest.approx(100.0 + 7.0 * 2.0)


# ── ARIMA-GARCH ───────────────────────────────────────────────────────────────

def test_arima_dist_normal_reproduces_legacy_symmetric_formula():
    train = synthetic_series(n=90, seed=1)
    arima_res, garch_res = mh.fit_arima(train, dist="normal")
    last_price = train.astype(float).values[-1]
    horizons = [1, 3, 5]

    got = mh.forecast_from_fitted_arima(arima_res, garch_res, last_price, horizons, dist="normal")

    # Reproduction manuelle de l'ancien calcul (avant migration) : bande symétrique
    # +/- Z_95 * sigma cumulée, à partir des MEMES objets déjà fittés.
    max_h = max(horizons)
    mean_fc = np.asarray(arima_res.forecast(steps=max_h), dtype=float) / 100.0
    garch_fc = garch_res.forecast(horizon=max_h, reindex=False)
    var_per_step = garch_fc.variance.values[-1, :] / (100.0 ** 2)
    cum_return = np.cumsum(mean_fc)
    cum_sigma = np.sqrt(np.cumsum(var_per_step))

    for h in horizons:
        i = h - 1
        exp_point = last_price * np.exp(cum_return[i])
        exp_lo = last_price * np.exp(cum_return[i] - arima_model.Z_95 * cum_sigma[i])
        exp_hi = last_price * np.exp(cum_return[i] + arima_model.Z_95 * cum_sigma[i])
        point, lo, hi = got[h]
        assert point == pytest.approx(exp_point, rel=1e-12)
        assert lo == pytest.approx(exp_lo, rel=1e-12)
        assert hi == pytest.approx(exp_hi, rel=1e-12)


def test_arima_default_dist_is_skewt():
    assert arima_model.GARCH_DIST == "skewt"


def test_forecast_horizons_arima_default_bounds_are_asymmetric():
    train = synthetic_series(n=90, seed=2)
    got = mh.forecast_horizons_arima(train, [1, 5])
    for h, (point, lo, hi) in got.items():
        assert lo < point < hi
        # Asymétrie en LOG-espace (skew-t ADOPTED, cf. arima_model.GARCH_DIST) :
        # |log(point) - log(lo)| doit différer de |log(hi) - log(point)| -- une bande
        # symétrique en prix (comme l'ancienne formule normale figée) donnerait deux
        # écarts log strictement égaux au niveau de précision flottant.
        down = np.log(point) - np.log(lo)
        up = np.log(hi) - np.log(point)
        assert down != pytest.approx(up, rel=1e-9)


def test_forecast_horizons_arima_uses_same_dist_for_fit_and_forecast():
    """forecast_horizons_arima résout `dist` UNE FOIS et l'utilise pour fit_arima ET
    forecast_from_fitted_arima (cf. docstring) -- ici vérifié en repassant dist="normal"
    explicitement : la bande doit rester symétrique (jamais accidentellement skew-t)."""
    train = synthetic_series(n=90, seed=3)
    got = mh.forecast_horizons_arima(train, [1], dist="normal")
    point, lo, hi = got[1]
    down = np.log(point) - np.log(lo)
    up = np.log(hi) - np.log(point)
    assert down == pytest.approx(up, rel=1e-9)


# ── SARIMA ────────────────────────────────────────────────────────────────────

def test_sarima_sigma_scale_none_is_bit_for_bit_with_no_correction():
    train = synthetic_series(n=60, seed=4)
    raw = mh.forecast_horizons_sarima(train, [1, 5])
    also_raw = mh.forecast_horizons_sarima(train, [1, 5], sigma_scale=None)
    assert raw == also_raw


def test_sarima_sigma_scale_widens_only_targeted_horizon():
    train = synthetic_series(n=60, seed=4)
    raw = mh.forecast_horizons_sarima(train, [1, 5])
    corrected = mh.forecast_horizons_sarima(train, [1, 5], sigma_scale={1: 3.0})

    # h=1 (dans le dict) : point inchangé, bande élargie d'un facteur 3.
    p_raw, lo_raw, hi_raw = raw[1]
    p_c, lo_c, hi_c = corrected[1]
    assert p_c == pytest.approx(p_raw)
    assert (p_c - lo_c) == pytest.approx((p_raw - lo_raw) * 3.0, rel=1e-9)
    assert (hi_c - p_c) == pytest.approx((hi_raw - p_raw) * 3.0, rel=1e-9)

    # h=5 (absent du dict) : bornes brutes inchangées, bit-for-bit.
    assert corrected[5] == raw[5]


# ── Naive ─────────────────────────────────────────────────────────────────────

def test_naive_sigma_scale_none_is_bit_for_bit_with_no_correction():
    train = synthetic_series(n=40, seed=5)
    raw = mh.forecast_horizons_naive(train, [1, 7])
    also_raw = mh.forecast_horizons_naive(train, [1, 7], sigma_scale=None)
    assert raw == also_raw


def test_naive_sigma_scale_widens_only_targeted_horizon():
    train = synthetic_series(n=40, seed=5)
    raw = mh.forecast_horizons_naive(train, [1, 7])
    corrected = mh.forecast_horizons_naive(train, [1, 7], sigma_scale={1: 2.5})

    p_raw, lo_raw, hi_raw = raw[1]
    p_c, lo_c, hi_c = corrected[1]
    assert p_c == pytest.approx(p_raw)
    assert (p_c - lo_c) == pytest.approx((p_raw - lo_raw) * 2.5, rel=1e-9)
    assert corrected[7] == raw[7]


# ── LSTM (duck-typed model/scaler -- pas d'entraînement réel) ───────────────────

class _FakeLstmModel:
    """predict() renvoie une valeur scalée constante, quel que soit le buffer --
    suffisant pour exercer la logique de bande (sigma_scale) sans entraîner de réseau."""
    def predict(self, x, verbose=0):
        return np.array([[0.5]])


class _FakeScaler:
    def inverse_transform(self, arr):
        return np.asarray(arr, dtype=float) * 100.0


def test_lstm_sigma_scale_widens_only_targeted_horizon():
    import lstm_model
    scaled = np.zeros(lstm_model.SEQ_LEN)
    std = 2.0
    model, scaler = _FakeLstmModel(), _FakeScaler()

    raw = mh.forecast_from_fitted_lstm(model, scaler, std, scaled, [1, 3])
    corrected = mh.forecast_from_fitted_lstm(model, scaler, std, scaled, [1, 3],
                                             sigma_scale={1: 3.0})

    p_raw, lo_raw, hi_raw = raw[1]
    p_c, lo_c, hi_c = corrected[1]
    assert p_c == pytest.approx(p_raw)
    assert (p_c - lo_c) == pytest.approx((p_raw - lo_raw) * 3.0, rel=1e-9)
    assert (hi_c - p_c) == pytest.approx((hi_raw - p_raw) * 3.0, rel=1e-9)
    # h=3 absent du dict -> bit-for-bit inchangé.
    assert corrected[3] == raw[3]


# ── Prophet ───────────────────────────────────────────────────────────────────
# Note méthodologique : Prophet.predict() tire ses bornes yhat_lower/yhat_upper par
# échantillonnage Monte-Carlo interne (uncertainty_samples), RE-tiré (RNG non fixée par
# ce module) à CHAQUE appel de .predict() -- même modèle déjà fitté, deux appels
# .predict() consécutifs donnent donc des bornes légèrement différentes (constaté :
# les deux premières versions de ces tests rejouaient un second .predict() et
# échouaient sporadiquement sur ce bruit, qui n'a rien à voir avec la migration).
# Les tests de logique de transformation (espace log/prix, sigma_scale) ci-dessous
# utilisent donc un FAUX modèle dont .predict() est déterministe -- ils vérifient le
# code de forecast_from_fitted_prophet, pas le générateur Monte-Carlo de Prophet
# lui-même. Les tests qui ont vraiment besoin d'un fit réel (fit_prophet, positivité
# des bornes) vérifient des propriétés STRUCTURELLES (model.history["y"], lo>0), pas
# une égalité bit-for-bit entre deux .predict() séparés.

class _FakeProphetModel:
    """.predict(df) déterministe : `n = len(df)` lignes, yhat croissant, bornes
    asymétriques fixes autour de yhat -- assez pour exercer log_space/sigma_scale
    sans dépendre du tirage Monte-Carlo interne de Prophet."""
    def predict(self, df):
        n = len(df)
        yhat = np.log(100.0) + 0.001 * np.arange(n)
        return pd.DataFrame({"yhat": yhat, "yhat_lower": yhat - 0.05, "yhat_upper": yhat + 0.08})


def test_forecast_from_fitted_prophet_price_space_returns_raw_values_unexponentiated():
    model = _FakeProphetModel()
    got = mh.forecast_from_fitted_prophet(model, pd.Timestamp("2024-01-01"), [1, 2], log_space=False)
    yhat = np.log(100.0) + 0.001 * np.arange(2)
    for h in (1, 2):
        point, lo, hi = got[h]
        i = h - 1
        assert point == pytest.approx(yhat[i], rel=1e-12)
        assert lo == pytest.approx(yhat[i] - 0.05, rel=1e-12)
        assert hi == pytest.approx(yhat[i] + 0.08, rel=1e-12)


def test_forecast_from_fitted_prophet_log_space_exponentiates_and_stays_positive():
    model = _FakeProphetModel()
    got = mh.forecast_from_fitted_prophet(model, pd.Timestamp("2024-01-01"), [1, 2], log_space=True)
    yhat = np.log(100.0) + 0.001 * np.arange(2)
    for h in (1, 2):
        point, lo, hi = got[h]
        i = h - 1
        assert point == pytest.approx(np.exp(yhat[i]), rel=1e-12)
        assert lo == pytest.approx(np.exp(yhat[i] - 0.05), rel=1e-12)
        assert hi == pytest.approx(np.exp(yhat[i] + 0.08), rel=1e-12)
        assert lo > 0


def test_forecast_from_fitted_prophet_sigma_scale_applied_before_exponentiating():
    """sigma_scale doit être appliqué EN ESPACE LOG (avant exp), pas après -- vérifié en
    comparant à la formule attendue explicitement, pas seulement "la bande s'élargit"."""
    model = _FakeProphetModel()
    yhat = np.log(100.0) + 0.001 * np.arange(2)   # h=1 -> index 0

    got = mh.forecast_from_fitted_prophet(model, pd.Timestamp("2024-01-01"), [1, 2],
                                          log_space=True, sigma_scale={1: 2.0})
    point, lo, hi = got[1]
    assert point == pytest.approx(np.exp(yhat[0]), rel=1e-12)
    assert lo == pytest.approx(np.exp(yhat[0] - 0.05 * 2.0), rel=1e-12)
    assert hi == pytest.approx(np.exp(yhat[0] + 0.08 * 2.0), rel=1e-12)

    # h=2 (absent du dict) : bit-for-bit identique à l'appel sans sigma_scale.
    raw = mh.forecast_from_fitted_prophet(model, pd.Timestamp("2024-01-01"), [1, 2], log_space=True)
    assert got[2] == raw[2]


@pytest.mark.slow
def test_fit_prophet_log_space_fits_on_log_price():
    train = synthetic_series(n=60, seed=9)
    model = mh.fit_prophet(train, log_space=True)
    np.testing.assert_allclose(model.history["y"].values, np.log(train.astype(float).values), rtol=1e-9)


@pytest.mark.slow
def test_fit_prophet_price_space_fits_on_raw_price():
    train = synthetic_series(n=60, seed=9)
    model = mh.fit_prophet(train, log_space=False)
    np.testing.assert_allclose(model.history["y"].values, train.astype(float).values, rtol=1e-9)


@pytest.mark.slow
def test_prophet_log_space_default_gives_strictly_positive_bounds():
    train = synthetic_series(n=70, seed=7)
    got = mh.forecast_horizons_prophet(train, [1, 2])
    for point, lo, hi in got.values():
        assert lo > 0
        assert lo < point < hi
