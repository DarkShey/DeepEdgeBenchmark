"""
duel_sampling_adapters.py — real trajectory samplers for the 5 classical
models (BRIEF_unification_protocole_duel.md §2.2, resolves N1).

Each classical model in models/*.py only ever produced a point + an analytic
95% interval; the multi-model matrices that DO compare diffusion vs classiques
(production pipelines A & B) never sample the actual fitted model — they
reconstruct a Gaussian/log-normal cloud from the stored (y_lower, y_upper)
bounds. That destroys the distributional information a CRPS duel is supposed
to measure (audit reserve N1) for 5 of 6 models. This module draws genuine
`m`-path trajectories from each model's OWN predictive law instead:

  - ARIMA-GARCH : simulate the GARCH(1,1) variance process forward with its
    OWN fixed (frozen) recursion, fed the realised residual path up to the
    origin — not a bootstrap of stored residuals, an actual multi-step
    simulation of the volatility process, added to ARIMA's own deterministic
    multi-step mean forecast.
  - SARIMA       : `SARIMAXResults.simulate(..., anchor="end")` — the
    state-space model's own conditional simulation, frozen parameters.
  - Prophet      : `Prophet.predictive_samples()` — Prophet's own generative
    trend/seasonality/noise model, sampled directly for future dates (no
    refit needed: Prophet's model is defined for any future `ds`).
  - LSTM         : MC-Dropout multi-step rollout — `m` independent paths,
    each one's own sampled output fed back as that path's own next input
    (dropout mask varies per path per step), not a single-step residual/
    dropout cloud replicated identically at every step.
  - Naive        : `weekly_headtohead_v2.random_walk_samples` — reused
    unchanged (brief §1: "réutiliser le tirage random_walk_samples du naïf").

Re-estimation rule (duel_origins.py's module docstring, brief §2.1): every
adapter here is fit EXACTLY ONCE on data <= T0 (`fit_*_state`); every later
origin only advances the model's *filter/conditioning state* with realised
data already known at that origin (`*_trajectory_samples`) — parameters are
never re-estimated. `fit_*_state` is always called fresh from the frozen
result object (statsmodels' `.append(..., refit=False)` composes: appending
the realised-to-date slice in one call from the ORIGINAL fit gives the exact
same filtered state as appending sequentially, verified empirically), so
calling `*_trajectory_samples` at different origins never mutates shared
state — each call is independent given (frozen fit, realised-data-to-date).

Horizon convention: ARIMA-GARCH/SARIMA/Prophet/LSTM are daily models (no
weekly variant exists in models/*.py for them, unlike TSDiff-W) — they
forecast at the exact TRADING-DAY distance to each W1/W2/W3 target (mirrors
TSDiff-D's handling in weekly_headtohead(_v2).py). Naive uses the weekly
random-walk empirical distribution directly, per brief §1.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "models", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import arima_model                                             # noqa: E402
import sarima_model                                             # noqa: E402
from statsmodels.tsa.arima.model import ARIMA                   # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX          # noqa: E402
from weekly_headtohead_v2 import random_walk_samples            # noqa: E402 (reused, not reimplemented)


# ══════════════════════════════════════════════════════════════════════════
#  ARIMA-GARCH — GARCH(1,1) variance process, simulated forward
# ══════════════════════════════════════════════════════════════════════════

def fit_garch_state(train_prices: pd.Series, order=arima_model.ARIMA_ORDER) -> dict:
    """Fit ARIMA(order) on log-returns + GARCH(1,1) on its residuals, once,
    on `train_prices` — frozen forever after this call."""
    prices = train_prices.astype(float).values
    returns = np.diff(np.log(prices)) * 100.0   # percent log-returns, matches arima_model.py
    arima_res = ARIMA(returns, order=order, enforce_stationarity=False,
                      enforce_invertibility=False).fit()
    resid = np.asarray(arima_res.resid, dtype=float)
    from arch import arch_model
    garch_res = arch_model(resid, vol="Garch", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    p = garch_res.params
    return {
        "arima_res": arima_res, "garch_res": garch_res,
        "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
        "h_last": float(garch_res.conditional_volatility[-1]) ** 2,
        "n_fit_resid": len(resid),
    }


def _garch_variance_after(omega: float, alpha: float, beta: float, eps: np.ndarray, h0: float) -> float:
    """Roll the FIXED GARCH(1,1) recursion h_t = omega + alpha*eps_{t-1}^2 +
    beta*h_{t-1} forward through the REALISED shocks `eps` (never
    re-estimating omega/alpha/beta), starting from variance state `h0`.
    Returns the variance state right after the last element of `eps` — the
    one-step-ahead variance to start simulating from."""
    h = h0
    for e in eps:
        h = omega + alpha * e * e + beta * h
    return h


def garch_trajectory_samples(state: dict, realized_returns_pct_since_fit: np.ndarray,
                             last_price: float, horizons: list, m: int, seed: int) -> dict:
    """`realized_returns_pct_since_fit`: percent log-returns realised strictly
    between the fit's T0 and the current origin (possibly empty at the very
    first origin). Returns {h: price_samples[m]} for h in `horizons` (steps
    in the SAME unit as `realized_returns_pct_since_fit`, i.e. trading days).

    Simplification declared (not hidden, brief §5): the GARCH model's own
    small constant-mean term ("mu" in `garch_res.params`) is ignored in the
    simulated shocks, exactly as arima_model.py's existing 1-step interval
    already does (`sigma` is read from the GARCH forecast, but its `mu` is
    never added to the point forecast) — kept identical here for consistency
    rather than silently introducing a different convention for the duel.
    """
    realized = np.asarray(realized_returns_pct_since_fit, dtype=float)
    arima_ext = state["arima_res"].append(realized, refit=False) if realized.size else state["arima_res"]
    h_max = max(horizons)
    mean_fc = np.asarray(arima_ext.forecast(steps=h_max), dtype=float)   # percent, deterministic

    full_resid = np.asarray(arima_ext.resid, dtype=float)
    new_eps = full_resid[state["n_fit_resid"]:]                          # residuals realised since fit
    h_state = _garch_variance_after(state["omega"], state["alpha"], state["beta"], new_eps, state["h_last"])

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((m, h_max))
    shocks = np.empty((m, h_max))
    h = np.full(m, h_state)
    for s in range(h_max):
        e = np.sqrt(np.maximum(h, 1e-12)) * z[:, s]
        shocks[:, s] = e
        h = state["omega"] + state["alpha"] * e * e + state["beta"] * h

    cum_pct = np.cumsum(mean_fc[None, :] + shocks, axis=1)               # [m, h_max], percent
    return {h: last_price * np.exp(cum_pct[:, h - 1] / 100.0) for h in horizons}


# ══════════════════════════════════════════════════════════════════════════
#  SARIMA — state-space conditional simulation
# ══════════════════════════════════════════════════════════════════════════

def fit_sarima_state(train_prices: pd.Series, order=sarima_model.ORDER,
                     seasonal_order=sarima_model.SEASONAL_ORDER) -> dict:
    prices = train_prices.astype(float).values
    res = SARIMAX(prices, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return {"res": res}


def sarima_trajectory_samples(state: dict, realized_prices_since_fit: np.ndarray,
                              horizons: list, m: int, seed: int) -> dict:
    """`realized_prices_since_fit`: raw prices realised strictly between T0
    and the current origin. Returns {h: price_samples[m]}."""
    realized = np.asarray(realized_prices_since_fit, dtype=float)
    res_ext = state["res"].append(realized, refit=False) if realized.size else state["res"]
    h_max = max(horizons)
    sims = res_ext.simulate(nsimulations=h_max, repetitions=m, anchor="end", random_state=seed)
    sims = np.asarray(sims)[:, 0, :].T                                    # [m, h_max], price levels
    return {h: sims[:, h - 1] for h in horizons}


# ══════════════════════════════════════════════════════════════════════════
#  Prophet — native predictive_samples()
# ══════════════════════════════════════════════════════════════════════════

def fit_prophet_state(train_prices: pd.Series, m: int) -> dict:
    """`uncertainty_samples=m` fixed at fit time: `m` is a duel-wide constant
    (brief §2.3: identical across all 6 models), never varied per origin, so
    there is no need to refit Prophet for a different sample count later."""
    from prophet import Prophet
    df = pd.DataFrame({"ds": pd.to_datetime(train_prices.index),
                       "y": train_prices.astype(float).values})
    model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, uncertainty_samples=m)
    model.fit(df)
    return {"model": model, "m": m}


def prophet_trajectory_samples(state: dict, target_dates: list, horizons: list, seed: int) -> dict:
    """No state to advance (Prophet's generative trend/seasonality/noise model
    is defined for any future `ds` directly from the original T0 fit) — just
    sample the requested future dates. `seed` reseeds numpy's global RNG
    (Prophet's own sampling uses `np.random` internally, no dedicated seed
    argument) immediately before drawing, for reproducibility."""
    np.random.seed(seed)
    future = pd.DataFrame({"ds": pd.to_datetime(list(target_dates))})
    samples = state["model"].predictive_samples(future)["yhat"]           # [n_dates, m]
    m = samples.shape[1]
    if m != state["m"]:
        raise RuntimeError(f"Prophet predictive_samples returned m={m}, expected {state['m']}")
    return {h: samples[i, :] for i, h in enumerate(horizons)}


# ══════════════════════════════════════════════════════════════════════════
#  LSTM — MC-Dropout multi-step rollout, one independent path per sample
# ══════════════════════════════════════════════════════════════════════════

def fit_lstm_state(train_prices: pd.Series, seq_len: int = None, units: int = None,
                   epochs: int = None, batch_size: int = None, seed: int = 42) -> dict:
    import lstm_model as lm
    seq_len = seq_len or lm.SEQ_LEN
    units = units or lm.UNITS
    epochs = epochs or lm.EPOCHS
    batch_size = batch_size or lm.BATCH_SIZE
    lm.set_seed(seed)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_prices.astype(float).values.reshape(-1, 1)).flatten()
    X, y = lm.make_sequences(scaled, seq_len)
    model = lm.build_lstm(seq_len, units)
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X.reshape(-1, seq_len, 1), y, epochs=epochs, batch_size=batch_size,
             validation_split=0.1, callbacks=[es], verbose=0)
    return {"model": model, "scaler": scaler, "seq_len": seq_len}


def lstm_trajectory_samples(state: dict, realized_prices_tail: np.ndarray,
                            horizons: list, m: int, seed: int) -> dict:
    """`realized_prices_tail`: at least `state["seq_len"]` most recent REAL
    (never predicted) prices, realised up to and including the origin. Each
    of the `m` paths independently rolls `h_max` steps forward: at every
    step, ONE batched forward pass with dropout active (`training=True`)
    gives each of the `m` rows its OWN independent dropout mask, and each
    row's OWN sampled output is appended to ITS OWN buffer before the next
    step — paths diverge from each other, unlike a single-step MC-Dropout
    cloud replicated identically at every horizon."""
    import tensorflow as tf
    model, scaler, seq_len = state["model"], state["scaler"], state["seq_len"]
    tail = np.asarray(realized_prices_tail, dtype=float)[-seq_len:]
    if len(tail) < seq_len:
        raise ValueError(f"need >= {seq_len} realised prices, got {len(tail)}")
    buf_scaled = scaler.transform(tail.reshape(-1, 1)).flatten()

    tf.random.set_seed(seed)
    h_max = max(horizons)
    paths = np.tile(buf_scaled, (m, 1))                                   # [m, seq_len]
    out_scaled = np.empty((m, h_max))
    for s in range(h_max):
        x = paths[:, -seq_len:].reshape(m, seq_len, 1)
        pred = model(x, training=True).numpy().reshape(-1)                # [m], independent dropout/path
        out_scaled[:, s] = pred
        paths = np.concatenate([paths, pred[:, None]], axis=1)

    prices = scaler.inverse_transform(out_scaled.reshape(-1, 1)).reshape(m, h_max)
    return {h: prices[:, h - 1] for h in horizons}


# ══════════════════════════════════════════════════════════════════════════
#  Naive — reused unchanged from weekly_headtohead_v2.random_walk_samples
# ══════════════════════════════════════════════════════════════════════════

def naive_trajectory_samples(historical_weekly_returns: np.ndarray, last_price: float,
                             horizons: list, m: int, seed: int) -> dict:
    """Draws exactly `m` samples (with replacement) from the empirical
    distribution random_walk_samples() builds — the pool it returns has
    `len(historical_returns) - h + 1` elements, not necessarily `m`, so this
    resamples it to the duel-wide constant `m` (brief §2.3) rather than
    leaving Naive's effective sample count model-specific."""
    rng = np.random.default_rng(seed)
    out = {}
    for h in horizons:
        pool = random_walk_samples(historical_weekly_returns, h)
        draw = rng.choice(pool, size=m, replace=True)
        out[h] = last_price * np.exp(draw)
    return out
