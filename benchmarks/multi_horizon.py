"""
multi_horizon.py — Adaptateurs multi-horizon pour les modèles de models/
=========================================================================
Les 5 fichiers de models/ (arima, sarima, prophet, lstm, naive) ne savent faire que du
walk-forward 1-step (ou next_step_*, un seul pas au-delà du dernier point). Ce module les
étend à un forecast multi-horizon **sans les modifier** : chaque fonction ci-dessous fitte
le modèle **une seule fois** sur la série d'entraînement, puis produit en un seul passage
les prévisions (point, IC95 bas, IC95 haut) pour tous les horizons demandés.

Contrat commun : `forecast_horizons_<model>(train: pd.Series, horizons: list[int]) ->
dict[int, tuple[float, float, float]]` où les clés sont des horizons en JOURS DE TRADING
(1-indexé, 1 = le jour suivant train.index[-1]).

Extensibilité : pour ajouter un nouveau modèle au benchmark, écrire une fonction
`forecast_horizons_<nom>` suivant ce contrat et l'ajouter à MODEL_ADAPTERS.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
import arima_model
import sarima_model
import naive_model
# prophet_model / lstm_model pull in heavy optional deps (prophet/cmdstanpy, tensorflow) —
# imported lazily inside their adapter so a missing dependency only disables that one
# model (caught per-model in run_benchmark.py) instead of crashing this whole module.


# ── ARIMA-GARCH ───────────────────────────────────────────────────────────────
def fit_arima(train: pd.Series):
    """Fit ARIMA(order) puis GARCH(1,1) sur ses résidus, une seule fois.
    Extrait de forecast_horizons_arima (même calcul, exposé pour la sérialisation
    des artefacts modèles — cf. model_artifacts/pipeline.py)."""
    prices = train.astype(float).values
    returns = np.diff(np.log(prices)) * 100.0

    arima_res = arima_model.ARIMA(
        returns, order=arima_model.ARIMA_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit()
    resid = np.asarray(arima_res.resid, dtype=float)
    garch_res = arima_model.arch_model(
        resid, vol="Garch", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")
    return arima_res, garch_res


def forecast_from_fitted_arima(arima_res, garch_res, last_price: float, horizons: list) -> dict:
    """Multi-step via ARIMA.forecast(steps=h) (retours cumulés) + variance GARCH
    cumulée (somme des variances par pas, hypothèse d'indépendance approx.),
    à partir d'objets déjà fittés (aucun nouveau fit ici)."""
    max_h = max(horizons)
    mean_fc = np.asarray(arima_res.forecast(steps=max_h), dtype=float) / 100.0
    garch_fc = garch_res.forecast(horizon=max_h, reindex=False)
    var_per_step = garch_fc.variance.values[-1, :] / (100.0 ** 2)

    cum_return = np.cumsum(mean_fc)
    cum_sigma = np.sqrt(np.cumsum(var_per_step))

    results = {}
    for h in horizons:
        i = h - 1
        point = last_price * np.exp(cum_return[i])
        lo = last_price * np.exp(cum_return[i] - arima_model.Z_95 * cum_sigma[i])
        hi = last_price * np.exp(cum_return[i] + arima_model.Z_95 * cum_sigma[i])
        results[h] = (float(point), float(lo), float(hi))
    return results


def forecast_horizons_arima(train: pd.Series, horizons: list) -> dict:
    """Fit once (fit_arima) puis forecast (forecast_from_fitted_arima) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables."""
    arima_res, garch_res = fit_arima(train)
    last_price = train.astype(float).values[-1]
    return forecast_from_fitted_arima(arima_res, garch_res, last_price, horizons)


# ── SARIMA ────────────────────────────────────────────────────────────────────
def fit_sarima(train: pd.Series):
    """Fit SARIMAX une seule fois. Extrait de forecast_horizons_sarima."""
    history = train.astype(float).values.tolist()
    return sarima_model.SARIMAX(
        history, order=sarima_model.ORDER, seasonal_order=sarima_model.SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)


def forecast_from_fitted_sarima(result, horizons: list) -> dict:
    """Multi-step natif : SARIMAX.get_forecast(steps=h) donne predicted_mean et
    conf_int() pour chaque pas 1..h en un seul appel, à partir d'un résultat déjà fitté."""
    max_h = max(horizons)
    fc = result.get_forecast(steps=max_h)
    pred_mean = np.asarray(fc.predicted_mean, dtype=float)
    ci = np.asarray(fc.conf_int(alpha=sarima_model.PI_ALPHA), dtype=float)

    results = {}
    for h in horizons:
        i = h - 1
        results[h] = (float(pred_mean[i]), float(ci[i, 0]), float(ci[i, 1]))
    return results


def forecast_horizons_sarima(train: pd.Series, horizons: list) -> dict:
    """Fit once (fit_sarima) puis forecast (forecast_from_fitted_sarima) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables."""
    result = fit_sarima(train)
    return forecast_from_fitted_sarima(result, horizons)


# ── Prophet ───────────────────────────────────────────────────────────────────
def fit_prophet(train: pd.Series):
    """Fit Prophet une seule fois. Extrait de forecast_horizons_prophet."""
    import prophet_model
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(train.index),
        "y": train.astype(float).values.flatten(),
    })
    model = prophet_model.Prophet(
        interval_width=1 - prophet_model.PI_ALPHA,
        daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
    )
    model.fit(df_train)
    return model


def forecast_from_fitted_prophet(model, last_date, horizons: list) -> dict:
    """Interroge directement les dates futures (jours ouvrés au-delà de last_date)
    sur un modèle déjà fitté — Prophet élargit nativement l'IC avec la distance
    dans le futur, sans dépendre d'un état interne à mettre à jour."""
    max_h = max(horizons)
    last_date = pd.to_datetime(last_date)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max_h)
    forecast = model.predict(pd.DataFrame({"ds": future_dates}))

    results = {}
    for h in horizons:
        i = h - 1
        row = forecast.iloc[i]
        results[h] = (float(row["yhat"]), float(row["yhat_lower"]), float(row["yhat_upper"]))
    return results


def forecast_horizons_prophet(train: pd.Series, horizons: list) -> dict:
    """Fit once (fit_prophet) puis forecast (forecast_from_fitted_prophet) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables."""
    model = fit_prophet(train)
    return forecast_from_fitted_prophet(model, train.index[-1], horizons)


# ── LSTM ──────────────────────────────────────────────────────────────────────
def fit_lstm(train: pd.Series, epochs: int = None, seed: int = None):
    """Fit le réseau une seule fois. Extrait de forecast_horizons_lstm.
    Retourne (model, scaler, std_résidus, série_scalée_complète) : tout ce qu'il
    faut pour forecaster (forecast_from_fitted_lstm) ou sérialiser l'artefact."""
    import lstm_model
    seq_len = lstm_model.SEQ_LEN
    epochs = lstm_model.EPOCHS if epochs is None else epochs
    seed = lstm_model.DEFAULT_SEED if seed is None else seed
    lstm_model.set_seed(seed)

    if len(train) <= seq_len:
        raise ValueError(
            f"train series has {len(train)} points, but seq_len={seq_len} requires "
            f"more than {seq_len} points to build at least one training sequence."
        )

    scaler = lstm_model.MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    X, y = lstm_model.make_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = lstm_model.build_lstm(seq_len)
    es = lstm_model.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=lstm_model.BATCH_SIZE,
             validation_split=0.1, callbacks=[es], verbose=0)

    train_preds = scaler.inverse_transform(
        model.predict(X, verbose=0).reshape(-1, 1)).flatten()
    std = float(np.std(train.values[seq_len:] - train_preds))
    return model, scaler, std, scaled


def forecast_from_fitted_lstm(model, scaler, std: float, scaled, horizons: list) -> dict:
    """Un seul rollout récursif de max(horizons) pas à partir d'objets déjà fittés :
    le réseau se ré-alimente de ses propres prédictions (jamais du vrai futur,
    contrainte point-in-time). L'IC s'élargit en sqrt(h) à partir de l'écart-type
    des résidus d'entraînement (même convention que next_step_lstm)."""
    import lstm_model
    seq_len = lstm_model.SEQ_LEN
    max_h = max(horizons)

    buffer = list(scaled[-seq_len:])
    rollout_scaled = []
    for _ in range(max_h):
        x = np.array(buffer[-seq_len:]).reshape(1, seq_len, 1)
        p_scaled = model.predict(x, verbose=0)[0, 0]
        rollout_scaled.append(p_scaled)
        buffer.append(p_scaled)   # recursif : jamais le vrai futur

    rollout_prices = scaler.inverse_transform(
        np.array(rollout_scaled).reshape(-1, 1)).flatten()

    results = {}
    for h in horizons:
        i = h - 1
        point = float(rollout_prices[i])
        sigma_h = std * np.sqrt(h)
        results[h] = (point, point - 1.96 * sigma_h, point + 1.96 * sigma_h)
    return results


def forecast_horizons_lstm(train: pd.Series, horizons: list, epochs: int = None,
                           seed: int = None) -> dict:
    """Fit once (fit_lstm) puis forecast (forecast_from_fitted_lstm) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables."""
    model, scaler, std, scaled = fit_lstm(train, epochs=epochs, seed=seed)
    return forecast_from_fitted_lstm(model, scaler, std, scaled, horizons)


# ── Naive ─────────────────────────────────────────────────────────────────────
def forecast_horizons_naive(train: pd.Series, horizons: list, band: float = None) -> dict:
    """point_h = dernier_prix * (1 + U(-band,band)) ; IC = dernier_prix * [1 -+ band*sqrt(h)]."""
    band = naive_model.BAND if band is None else band
    last_price = float(train.iloc[-1])
    results = {}
    for h in horizons:
        drift = np.random.uniform(-band, band)
        point = last_price * (1.0 + drift)
        half = band * np.sqrt(h)
        results[h] = (point, last_price * (1.0 - half), last_price * (1.0 + half))
    return results


# ── Registre des modèles (point d'extension) ─────────────────────────────────
MODEL_ADAPTERS = {
    "ARIMA-GARCH": forecast_horizons_arima,
    "SARIMA":      forecast_horizons_sarima,
    "Prophet":     forecast_horizons_prophet,
    "LSTM":        forecast_horizons_lstm,
    "Naive":       forecast_horizons_naive,
}
