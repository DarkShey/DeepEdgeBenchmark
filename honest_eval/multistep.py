"""
multistep.py — dense multi-horizon evaluation without re-anchoring (Point 3)
============================================================================
At horizon > 1 the "copy the last price" trick decays, so this is where models
can actually differentiate.  The old D+7 used ~10 origins spaced ~17 days apart
(n=10 → 100% coverage is meaningless).  Here we use a **dense daily rolling
origin**: one h-step forecast emitted every day ⇒ ~150+ evaluations per asset,
for D+1 / D+7 / D+30.

Because the forecasts overlap, the per-origin errors are autocorrelated up to
lag h−1 → Diebold-Mariano uses the Newey-West variance with that truncation
(metrics.diebold_mariano(..., h=h)).

A forecaster is any callable ``f(history: pd.Series, h: int) -> dict`` returning
``{"mean": (h,), "lower": (h,), "upper": (h,)}`` (sigma optional).  Adapters for
the repo's models are provided; they refit at each origin, which is expensive —
use ``step`` to thin origins for the heavy models (LSTM/SARIMA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .naive import naive_drift_path


# ── generic rolling-origin engine ────────────────────────────────────────────

def rolling_origin(series: pd.Series, forecaster, horizons, test_start,
                   step=1, max_h=None, verbose=False):
    """Emit an h-step forecast at every ``step``-th origin from ``test_start``.

    Returns ``{h: {origin_i, prev, actual, pred, lower, upper, index}}`` with one
    row per origin for which ``origin + h`` exists.  A single call to
    ``forecaster(history, max_h)`` per origin serves all horizons.
    """
    series = series.astype(float)
    values = series.values
    idx = series.index
    n = len(values)
    horizons = sorted(int(h) for h in horizons)
    max_h = int(max_h or max(horizons))

    store = {h: {"prev": [], "actual": [], "pred": [],
                 "lower": [], "upper": [], "index": []} for h in horizons}

    origins = range(test_start, n - min(horizons) + 1, step)
    for oi, origin in enumerate(origins):
        history = series.iloc[:origin]
        remaining = n - origin
        h_call = min(max_h, remaining)
        if h_call < min(horizons):
            break
        fc = forecaster(history, h_call)
        mean = np.asarray(fc["mean"], float).ravel()
        lo = np.asarray(fc.get("lower", np.full(h_call, np.nan)), float).ravel()
        hi = np.asarray(fc.get("upper", np.full(h_call, np.nan)), float).ravel()
        prev = values[origin - 1]
        for h in horizons:
            if origin + h - 1 >= n or h > h_call:
                continue
            k = h - 1
            store[h]["prev"].append(prev)
            store[h]["actual"].append(values[origin + h - 1])
            store[h]["pred"].append(mean[k])
            store[h]["lower"].append(lo[k])
            store[h]["upper"].append(hi[k])
            store[h]["index"].append(idx[origin + h - 1])
        if verbose and oi % 25 == 0:
            print(f"  [multistep] origin {oi+1} ({idx[origin].date()})")

    for h in horizons:
        for key in ("prev", "actual", "pred", "lower", "upper"):
            store[h][key] = np.asarray(store[h][key], float)
        store[h]["index"] = pd.DatetimeIndex(store[h]["index"])
    return store


def naive_forecaster(history: pd.Series, h: int) -> dict:
    """Random-walk-with-drift benchmark forecaster (the honest baseline)."""
    mean, sigma = naive_drift_path(history.values, h)
    return {"mean": mean, "lower": mean - 1.96 * sigma,
            "upper": mean + 1.96 * sigma, "sigma": sigma}


# ── per-horizon scoring vs the naive benchmark ───────────────────────────────

def score_horizon(model_rows, naive_rows, h, alpha=0.05):
    """Score one horizon: RMSE, MASE, Theil U, DirAcc±CI, coverage, DM(NW)."""
    actual = model_rows["actual"]
    prev = model_rows["prev"]
    pred = model_rows["pred"]
    npred = naive_rows["pred"]
    n = len(actual)
    if n == 0:
        return {"h": h, "n": 0}

    err_m = actual - pred
    err_n = actual - npred
    dm, p, lag = metrics.diebold_mariano(err_m, err_n, h=h)
    theil = metrics.theil_u(actual, pred, npred)
    da = metrics.directional_accuracy(pred, prev, actual)
    cov = (metrics.coverage(actual, model_rows["lower"], model_rows["upper"])
           if np.all(np.isfinite(model_rows["lower"])) else float("nan"))
    return {
        "h": h, "n": n,
        "rmse": metrics.rmse(actual, pred),
        "mase": metrics.mase(actual, pred, npred),
        "theil_u": theil,
        "change_corr": metrics.change_correlation(pred, prev, actual),
        "dir_acc": da["acc"], "dir_ci95": da["ci95"], "dir_p": da["p_vs_coin"],
        "coverage": cov,
        "dm": dm, "dm_p": p, "dm_lag": lag,
        "verdict": metrics.skill_verdict(theil, p, alpha),
    }


def evaluate_model_multih(series, model_forecaster, horizons=(1, 7, 30),
                          test_start=None, step=1, test_ratio=0.15, verbose=False):
    """Full Point-3 evaluation of one model across horizons vs the naive.

    Returns ``{h: score_dict}`` plus ``_rows`` with the raw per-origin arrays so
    report.py can draw the error-vs-horizon curve.
    """
    series = series.astype(float)
    n = len(series)
    if test_start is None:
        test_start = int(n * (1 - test_ratio))
    max_h = max(horizons)

    model_store = rolling_origin(series, model_forecaster, horizons,
                                 test_start, step=step, max_h=max_h, verbose=verbose)
    naive_store = rolling_origin(series, naive_forecaster, horizons,
                                 test_start, step=step, max_h=max_h)

    scores = {h: score_horizon(model_store[h], naive_store[h], h) for h in horizons}
    scores["_rows"] = model_store
    scores["_naive_rows"] = naive_store
    return scores


def error_vs_horizon(series, model_forecaster, horizons=range(1, 31),
                     test_start=None, step=1, test_ratio=0.15):
    """RMSE and MASE as a function of horizon h — the degradation curve.

    Returns a DataFrame indexed by h with model RMSE, naive RMSE, MASE, Theil U.
    """
    horizons = list(horizons)
    res = evaluate_model_multih(series, model_forecaster, horizons=horizons,
                                test_start=test_start, step=step, test_ratio=test_ratio)
    rows = []
    for h in horizons:
        s = res[h]
        if s.get("n", 0) == 0:
            continue
        rows.append({"h": h, "n": s["n"], "rmse": s["rmse"],
                     "mase": s["mase"], "theil_u": s["theil_u"],
                     "dir_acc": s["dir_acc"]})
    return pd.DataFrame(rows).set_index("h")


# ── model adapters (refit at each origin) ────────────────────────────────────
# Each returns a forecaster closure f(history, h) -> {mean, lower, upper}.

def make_arima_forecaster(order=(2, 0, 2), z=1.96):
    """ARIMA(order)-GARCH(1,1) multi-step: forecast h returns, compound to prices.

    Mean path from ARIMA on 100·log-returns; variance path from GARCH(1,1).
    The h-step log-return is the sum of steps → mean = Σμ_k, var = Σσ²_k.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model

    def f(history: pd.Series, h: int) -> dict:
        prices = history.values.astype(float)
        last = prices[-1]
        rets = np.diff(np.log(prices)) * 100.0
        ar = ARIMA(rets, order=order, enforce_stationarity=False,
                   enforce_invertibility=False).fit()
        mu = np.asarray(ar.forecast(steps=h), float) / 100.0        # per-step means
        try:
            g = arch_model(np.asarray(ar.resid, float), vol="Garch",
                           p=1, q=1, dist="normal", rescale=False).fit(disp="off")
            var = g.forecast(horizon=h, reindex=False).variance.values[-1, :] / (100.0**2)
        except Exception:
            var = np.full(h, np.var(rets / 100.0))
        cum_mu = np.cumsum(mu)
        cum_sig = np.sqrt(np.cumsum(var))
        mean = last * np.exp(cum_mu)
        lower = last * np.exp(cum_mu - z * cum_sig)
        upper = last * np.exp(cum_mu + z * cum_sig)
        return {"mean": mean, "lower": lower, "upper": upper}
    return f


def make_sarima_forecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 5), alpha=0.05):
    """SARIMA h-step: get_forecast(steps=h) gives the path and PI directly."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def f(history: pd.Series, h: int) -> dict:
        hist = history.values.astype(float)
        res = SARIMAX(hist, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = res.get_forecast(steps=h)
        mean = np.asarray(fc.predicted_mean, float).ravel()
        ci = fc.conf_int(alpha=alpha)
        ci = np.asarray(ci, float)
        return {"mean": mean, "lower": ci[:, 0], "upper": ci[:, 1]}
    return f


def make_prophet_forecaster(alpha=0.05):
    """Prophet h-step: extend the index by h steps and batch-predict."""
    from prophet import Prophet

    def f(history: pd.Series, h: int) -> dict:
        idx = pd.DatetimeIndex(history.index)
        df = pd.DataFrame({"ds": idx, "y": history.values.astype(float)})
        m = Prophet(interval_width=1 - alpha, daily_seasonality=False,
                    weekly_seasonality=True, yearly_seasonality=True)
        m.fit(df)
        freq = pd.infer_freq(idx) or "D"
        future = pd.date_range(idx[-1], periods=h + 1, freq=freq)[1:]
        fc = m.predict(pd.DataFrame({"ds": future}))
        return {"mean": fc["yhat"].values,
                "lower": fc["yhat_lower"].values, "upper": fc["yhat_upper"].values}
    return f


def make_lstm_forecaster(seq_len=30, epochs=20, batch_size=32, z=1.96):
    """LSTM recursive h-step: feed predictions back into the look-back buffer.

    No native multi-step PI → widen the 1-step residual sigma by √k (RW scaling).
    Expensive (trains a net per origin) — thin origins with ``step`` for real runs.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    def f(history: pd.Series, h: int) -> dict:
        vals = history.values.astype(float).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(vals).flatten()
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i - seq_len:i]); y.append(scaled[i])
        X = np.asarray(X).reshape(-1, seq_len, 1); y = np.asarray(y)
        model = Sequential([LSTM(64, input_shape=(seq_len, 1)), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
        model.fit(X, y, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, callbacks=[es], verbose=0)
        buf = list(scaled[-seq_len:])
        path_scaled = []
        for _ in range(h):
            x = np.asarray(buf[-seq_len:]).reshape(1, seq_len, 1)
            p = float(model.predict(x, verbose=0)[0, 0])
            path_scaled.append(p); buf.append(p)
        mean = scaler.inverse_transform(np.asarray(path_scaled).reshape(-1, 1)).flatten()
        fitted = scaler.inverse_transform(model.predict(X, verbose=0).reshape(-1, 1)).flatten()
        std = float(np.std(vals.flatten()[seq_len:] - fitted))
        ks = np.sqrt(np.arange(1, h + 1))
        return {"mean": mean, "lower": mean - z * std * ks, "upper": mean + z * std * ks}
    return f


MODEL_FORECASTERS = {
    "arima": make_arima_forecaster,
    "sarima": make_sarima_forecaster,
    "prophet": make_prophet_forecaster,
    "lstm": make_lstm_forecaster,
}
