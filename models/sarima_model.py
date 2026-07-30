"""
SARIMA Forecasting Model
========================
Standalone port of the SARIMA model from the DEITA time-series benchmark.

What it does
------------
Seasonal ARIMA — **SARIMA(1,1,1)(1,0,1)[5]** fitted directly on prices.
Forecasting is **walk-forward** (rolling 1-step-ahead): the model is re-fitted
each step on the growing history, then forecasts one point with a 95%
prediction interval from the SARIMAX confidence interval.

This file is fully self-contained — no dependency on any other DEITA module.

Quick start
-----------
    pip install numpy pandas yfinance statsmodels scikit-learn matplotlib

    python sarima_model.py                              # BTC-USD backtest
    python sarima_model.py --ticker SPY --plot out.png  # + save forecast plot
    python sarima_model.py --ticker GC=F --next-step    # single next-step forecast
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Config (defaults; override via CLI) ──────────────────────────────────────
ORDER          = (1, 1, 1)        # (p, d, q)
SEASONAL_ORDER = (1, 0, 1, 5)     # (P, D, Q, s) — weekly-ish 5-day cycle
PI_ALPHA       = 0.05             # 95% prediction interval

# EWMA sigma-level calibration ADOPTED 2026-07-30 (calibrate_sigma="ewma"
# default): SARIMA's Kalman conf_int sigma_t is already time-varying, but its
# LEVEL drifts against the realized error scale across volatility regimes --
# the three-window comparison (experiments/dynamic_sigma_variants.py: W1
# 2020-2024, W2 2018-2022, W3 2022-2026) shows no static fix survives a regime
# change (best static, CQR, cross-window MACE 8.28 vs normal 9.94), while a
# causal multiplicative correction sigma'_t = sigma_t*sqrt(EWMA(z_t^2)) with
# z_t = (actual_t - pred_t)/sigma_t gets MACE 3.96 (normal quantiles). The
# correction preserves SARIMA's own dynamics (it only retargets the variance
# level) and costs nothing. Set calibrate_sigma="off" for the historical bands.
EWMA_LAMBDA    = 0.94             # RiskMetrics daily decay
CALIBRATE_SIGMA = "ewma"
_Z975          = 1.959963984540054   # norm.ppf(0.975); conf_int's own multiplier


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily Close prices and return a clean, tz-naive Series."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker} between {start} and {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = pd.to_numeric(raw["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close.astype(float)


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(actual, predicted, pi_lower=None, pi_upper=None,
                    train_time=0.0) -> dict:
    actual    = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    mae   = mean_absolute_error(actual, predicted)
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mape  = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    smape = np.mean(2 * np.abs(actual - predicted) /
                    (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
    dir_acc = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
    try:
        lb_p = acorr_ljungbox(actual - predicted, lags=[10],
                              return_df=True)["lb_pvalue"].values[0]
    except Exception:
        lb_p = np.nan
    pi_cov = np.nan
    if pi_lower is not None and pi_upper is not None:
        pi_cov = np.mean((actual >= pi_lower) & (actual <= pi_upper)) * 100
    return {
        "RMSE":           round(rmse,  4),
        "MAE":            round(mae,   4),
        "MAPE (%)":       round(mape,  2),
        "SMAPE (%)":      round(smape, 2),
        "Dir. Acc (%)":   round(dir_acc, 2),
        "PI Cov 95% (%)": round(pi_cov, 2) if not np.isnan(pi_cov) else "N/A",
        "Ljung-Box p":    round(lb_p,  4) if not np.isnan(lb_p) else "N/A",
        "Train Time (s)": round(train_time, 2),
    }


# ── SARIMA walk-forward backtest ─────────────────────────────────────────────
def run_sarima(train: pd.Series, test: pd.Series,
               order=ORDER, seasonal_order=SEASONAL_ORDER,
               n_ensemble: int = 0, ensemble_seed=None,
               calibrate_sigma: str = CALIBRATE_SIGMA) -> dict:
    """Rolling 1-step-ahead SARIMA forecast over the test window.

    `calibrate_sigma` : "ewma" (default, adopted 2026-07-30 -- see
    CALIBRATE_SIGMA note) applies the causal variance-targeting correction
    sigma'_t = sigma_t * sqrt(EWMA(z^2)) to the Kalman interval at each step
    (z uses only steps < t; the EWMA state starts at 1 = 'trust the model's
    own sigma until told otherwise'). "off" keeps the raw conf_int bands.

    `n_ensemble` (0 = off, default -- no cost for existing callers): at each step,
    additionally draws `n_ensemble` bootstrap samples of the next-step price by
    resampling (with replacement) the in-sample residuals of that step's own fit
    (`result.resid`) around its point forecast. Populates `result["ensemble"]`
    (list of length len(test), one [n_ensemble] price array per step) for
    empirical CRPS (cf. model_artifacts/crps_kpis.py). Kept on the RAW sigma on
    purpose (production Gate 2 KPI, distribution-agnostic semantics)."""
    t0 = time.time()
    history = list(train.astype(float).values)
    preds, lower, upper = [], [], []
    ensembles = [] if n_ensemble > 0 else None
    rng = np.random.default_rng(ensemble_seed) if n_ensemble > 0 else None
    s2_ewma = 1.0                       # EWMA of z^2 (calibrate_sigma="ewma")

    for i in range(len(test)):
        model  = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        fc     = result.get_forecast(steps=1)

        pm = fc.predicted_mean
        point = float(pm.iloc[0] if hasattr(pm, "iloc") else pm[0])
        preds.append(point)

        ci = fc.conf_int(alpha=PI_ALPHA)
        if isinstance(ci, pd.DataFrame):
            lo_raw, hi_raw = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
        else:
            lo_raw, hi_raw = float(ci[0][0]), float(ci[0][1])

        sigma_own = max((hi_raw - lo_raw) / (2.0 * _Z975), 1e-12)
        if calibrate_sigma == "ewma":
            corr = np.sqrt(s2_ewma)
            lower.append(point - (point - lo_raw) * corr)
            upper.append(point + (hi_raw - point) * corr)
        else:
            lower.append(lo_raw)
            upper.append(hi_raw)

        if n_ensemble > 0:
            residuals = np.asarray(result.resid, dtype=float)
            ensembles.append(point + rng.choice(residuals, size=n_ensemble, replace=True))

        actual = float(test.iloc[i])
        if calibrate_sigma == "ewma":   # update AFTER the step's bounds are set
            z = (actual - point) / sigma_own
            s2_ewma = EWMA_LAMBDA * s2_ewma + (1 - EWMA_LAMBDA) * z * z

        history.append(actual)   # walk forward

    train_time = time.time() - t0
    preds, lower, upper = map(np.array, (preds, lower, upper))
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    result = {**metrics, "predictions": preds, "lower": lower, "upper": upper,
              "index": test.index, "actual": test.values}
    if ensembles is not None:
        result["ensemble"] = ensembles
    return result


def next_step_sarima(series: pd.Series, order=ORDER, seasonal_order=SEASONAL_ORDER,
                     calibrate_sigma: str = CALIBRATE_SIGMA):
    """Single 1-step forecast beyond the last observation. Returns (pred, lo, hi).

    With calibrate_sigma="ewma" the same variance-targeting correction as
    run_sarima is applied, using the fit's own one-step-ahead standardized
    forecast errors (Kalman filter innovations -- exactly z_t = e_t/sigma_t of
    the walk-forward, available in-sample at no extra cost)."""
    history = series.astype(float).values.tolist()
    result  = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False)
    fc   = result.get_forecast(steps=1)
    pm   = fc.predicted_mean
    pred = float(pm.iloc[0] if hasattr(pm, "iloc") else pm[0])
    ci   = fc.conf_int(alpha=PI_ALPHA)
    if isinstance(ci, pd.DataFrame):
        pi_low, pi_hi = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
    else:
        pi_low, pi_hi = float(ci[0][0]), float(ci[0][1])

    if calibrate_sigma == "ewma":
        z = np.asarray(result.filter_results.standardized_forecasts_error[0],
                       dtype=float)
        z = z[np.isfinite(z)]
        # skip the filter burn-in (diffuse initialization inflates early z)
        z = z[max(len(z) // 10, 20):] if len(z) > 40 else z
        s2 = 1.0
        for zi in z[-250:]:
            s2 = EWMA_LAMBDA * s2 + (1 - EWMA_LAMBDA) * zi * zi
        corr = float(np.sqrt(s2))
        pi_low = pred - (pred - pi_low) * corr
        pi_hi  = pred + (pi_hi - pred) * corr
    return pred, pi_low, pi_hi


# ── Plot (optional) ──────────────────────────────────────────────────────────
def save_plot(result: dict, ticker: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = result["index"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, result["actual"], label="Actual", color="black", lw=1.3)
    ax.plot(idx, result["predictions"], label="SARIMA forecast", color="tab:green", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:green",
                    alpha=0.20, label="95% PI")
    ax.set_title(f"SARIMA{ORDER}{SEASONAL_ORDER} — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="SARIMA forecasting (DEITA port)")
    p.add_argument("--ticker", default="BTC-USD", help="yfinance ticker (BTC-USD, SPY, GC=F)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--next-step", action="store_true", help="only forecast the next step")
    p.add_argument("--plot", metavar="PATH", default=None, help="save a forecast plot")
    args = p.parse_args()

    print(f"Downloading {args.ticker} [{args.start} -> {args.end}] ...")
    prices = fetch_data(args.ticker, args.start, args.end)
    print(f"  {len(prices)} daily observations.\n")

    if args.next_step:
        pred, lo, hi = next_step_sarima(prices)
        print(f"Last close      : {prices.iloc[-1]:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  "
          f"SARIMA{ORDER}{SEASONAL_ORDER}\n")
    print("Note: SARIMA re-fits every step — the backtest can take a few minutes.\n")

    result = run_sarima(train, test)
    print(f"=== SARIMA{ORDER}{SEASONAL_ORDER} — {args.ticker} ===")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
