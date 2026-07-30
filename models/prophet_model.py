"""
Prophet Forecasting Model
=========================
Standalone port of the Prophet model from the DEITA time-series benchmark.

What it does
------------
Facebook/Meta **Prophet** additive regression with weekly + yearly seasonality.
Forecasting is **walk-forward** (rolling 1-step-ahead, mirrors ``run_sarima``):
the model is re-fitted each step on the growing history, then forecasts one
point. Prediction intervals come from Prophet's own ``yhat_lower`` / ``yhat_upper``
(95% interval width).

This file is fully self-contained — no dependency on any other DEITA module.

Quick start
-----------
    pip install numpy pandas yfinance prophet scikit-learn matplotlib

    python prophet_model.py                              # BTC-USD backtest
    python prophet_model.py --ticker SPY --plot out.png  # + save forecast plot
    python prophet_model.py --ticker GC=F --next-step    # single next-step forecast

Note: `prophet` pulls in `cmdstanpy`; first install may compile a Stan backend.
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Config (defaults; override via CLI) ──────────────────────────────────────
PI_ALPHA = 0.05    # 95% prediction interval -> interval_width = 0.95

# log-space fit + EWMA sigma calibration ADOPTED 2026-07-30, following the
# dedicated investigation (experiments/prophet_sigma_investigation.py) of the
# long-standing "Prophet under-covers massively" issue (24/46/70% observed for
# 50/80/95% targets on 2020-2024). Root cause, quantified: (1) Prophet's
# interval at h=1 is essentially +/- z*sigma_obs, and sigma_obs is the std of
# IN-SAMPLE residuals around a flexible piecewise-linear trend -- flexibility
# (changepoint_prior_scale) mechanically shrinks in-sample residuals while
# out-of-sample 1-step errors do NOT shrink (sigma_ratio 0.43-0.72 at the
# default cps, >1 at cps=0.001 but with 4x worse RMSE); (2) fitting PRICES
# gives one homoskedastic sigma while errors scale with the price level.
# Fixes, validated on 3 windows x SPY/BTC: fitting log(price) improves every
# case (MACE 13-40 -> 4-19, RMSE also improves) and the same causal
# variance-targeting EWMA correction adopted for SARIMA closes the rest
# (log+EWMA: MACE 1.1-7.1 everywhere). Set log_space=False /
# calibrate_sigma="off" to reproduce the historical behaviour.
LOG_SPACE       = True
CALIBRATE_SIGMA = "ewma"
EWMA_LAMBDA     = 0.94              # RiskMetrics daily decay
_Z975           = 1.959963984540054


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


def infer_next_date(series: pd.Series):
    """Return the timestamp of the step immediately following the last observation."""
    idx = pd.DatetimeIndex(series.index)
    freq = pd.infer_freq(idx)
    if freq is not None:
        return idx[-1] + pd.tseries.frequencies.to_offset(freq)
    med = float(np.median(np.diff(idx.asi8) / 1e9 / 86400))   # median gap, days
    if 0.9 <= med <= 1.5:
        return idx[-1] + pd.tseries.offsets.BDay(1)
    if 6.5 <= med <= 7.5:
        return idx[-1] + pd.Timedelta(weeks=1)
    if 0.03 <= med <= 0.06:
        return idx[-1] + pd.Timedelta(hours=1)
    return idx[-1] + pd.Timedelta(days=med)


# ── Prophet walk-forward backtest ────────────────────────────────────────────
def run_prophet(train: pd.Series, test: pd.Series, refit_freq: int = 1,
                n_ensemble: int = 0, ensemble_seed=None,
                log_space: bool = LOG_SPACE,
                calibrate_sigma: str = CALIBRATE_SIGMA) -> dict:
    """Rolling 1-step-ahead Prophet forecast over the test window (walk-forward),
    aligned with run_sarima: at each step, (re)fit Prophet on the history known up
    to t-1 and take ONLY the next date (yhat/yhat_lower/yhat_upper).

    `refit_freq` : re-fit every N steps (1 = every step, the most correct and the
    default). Values >1 speed things up but reintroduce drift proportional to the
    gap -- between two refits the forecast degrades back into a k-step-ahead
    extrapolation, exactly the bias this walk-forward rewrite exists to remove.
    Keep it at 1 unless the cost is genuinely prohibitive.

    `n_ensemble` (0 = off, default -- no cost for existing callers): at each step,
    additionally draws `n_ensemble` bootstrap samples of the next-step price by
    resampling (with replacement) the in-sample residuals of the current fit
    (yhat - y over history_ds, recomputed on each refit -- one extra
    `model.predict(df_train)` call per refit, so roughly doubles the cost of the
    default refit_freq=1 backtest) around the point forecast. Populates
    `result["ensemble"]` (list of length len(test), one [n_ensemble] price array
    per step) for empirical CRPS (cf. model_artifacts/crps_kpis.py).

    `log_space` / `calibrate_sigma` (defaults on, adopted 2026-07-30 -- see the
    LOG_SPACE note): the model is fitted on log(price) (all Prophet mechanics
    unchanged, bounds exponentiated back, so the price-space law is lognormal)
    and the interval width is retargeted each step by the causal EWMA of the
    squared standardized walk-forward residuals (z uses only steps < i)."""
    t0 = time.time()
    history_ds = list(pd.to_datetime(train.index))
    history_y  = list(train.astype(float).values.flatten())
    preds, lower, upper = [], [], []
    ensembles = [] if n_ensemble > 0 else None
    rng = np.random.default_rng(ensemble_seed) if n_ensemble > 0 else None
    residual_pool = None
    s2_ewma = 1.0

    def to_fit(v):
        return np.log(v) if log_space else np.asarray(v, dtype=float)

    model = None
    for i in range(len(test)):
        if model is None or (i % refit_freq == 0):
            df_train = pd.DataFrame({"ds": history_ds,
                                     "y": to_fit(np.asarray(history_y, dtype=float))})
            model = Prophet(interval_width=1 - PI_ALPHA, daily_seasonality=False,
                            weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df_train)
            if n_ensemble > 0:
                in_sample = model.predict(df_train)["yhat"].values
                residual_pool = in_sample - df_train["y"].values  # fit space

        next_ds = pd.to_datetime(test.index[i])
        fc = model.predict(pd.DataFrame({"ds": [next_ds]}))
        yhat = float(fc["yhat"].iloc[0])          # fit space (log if log_space)
        lo   = float(fc["yhat_lower"].iloc[0])
        hi   = float(fc["yhat_upper"].iloc[0])

        sigma_own = max((hi - lo) / (2.0 * _Z975), 1e-12)
        if calibrate_sigma == "ewma":
            corr = np.sqrt(s2_ewma)
            lo = yhat - (yhat - lo) * corr
            hi = yhat + (hi - yhat) * corr

        if log_space:
            preds.append(float(np.exp(yhat)))
            lower.append(float(np.exp(lo)))
            upper.append(float(np.exp(hi)))
        else:
            preds.append(yhat); lower.append(lo); upper.append(hi)

        if n_ensemble > 0:
            draws = yhat + rng.choice(residual_pool, size=n_ensemble, replace=True)
            ensembles.append(np.exp(draws) if log_space else draws)

        # walk forward: reveal the realised value for step i
        actual = float(test.iloc[i])
        if calibrate_sigma == "ewma":
            z = (to_fit(actual) - yhat) / sigma_own
            s2_ewma = EWMA_LAMBDA * s2_ewma + (1 - EWMA_LAMBDA) * z * z
        history_ds.append(next_ds)
        history_y.append(actual)

    train_time = time.time() - t0
    preds, lower, upper = map(np.array, (preds, lower, upper))
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    result = {**metrics, "predictions": preds, "lower": lower, "upper": upper,
              "index": test.index, "actual": test.values}
    if ensembles is not None:
        result["ensemble"] = ensembles
    return result


def next_step_prophet(series: pd.Series, next_date=None,
                      log_space: bool = LOG_SPACE, sigma_scale: float = None):
    """Single 1-step forecast for the next date. Returns (pred, lo, hi).

    `log_space` (default on, adopted 2026-07-30): fit on log(price), bounds
    exponentiated back -- same specification as run_prophet.

    `sigma_scale` (optional): multiplicative width correction around the point
    forecast, to be supplied by the caller from OUT-OF-SAMPLE standardized
    residuals (e.g. sqrt(EWMA(z^2)) over recent tracked predictions, exactly
    run_prophet's calibrate_sigma state). It cannot be derived here from the
    fit itself: the whole root cause is that IN-SAMPLE residuals understate
    the out-of-sample error (see the LOG_SPACE note), so a fit-internal
    correction would systematically under-correct. None = no correction."""
    if next_date is None:
        next_date = infer_next_date(series)
    y = series.astype(float).values.flatten()
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(series.index),
        "y":  np.log(y) if log_space else y,
    })
    model = Prophet(interval_width=1 - PI_ALPHA, daily_seasonality=False,
                    weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_train)
    forecast = model.predict(pd.DataFrame({"ds": [next_date]}))
    yhat = float(forecast["yhat"].iloc[0])
    lo   = float(forecast["yhat_lower"].iloc[0])
    hi   = float(forecast["yhat_upper"].iloc[0])
    if sigma_scale is not None:
        lo = yhat - (yhat - lo) * float(sigma_scale)
        hi = yhat + (hi - yhat) * float(sigma_scale)
    if log_space:
        return float(np.exp(yhat)), float(np.exp(lo)), float(np.exp(hi))
    return yhat, lo, hi


# ── Plot (optional) ──────────────────────────────────────────────────────────
def save_plot(result: dict, ticker: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = result["index"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, result["actual"], label="Actual", color="black", lw=1.3)
    ax.plot(idx, result["predictions"], label="Prophet forecast", color="tab:purple", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:purple",
                    alpha=0.20, label="95% PI")
    ax.set_title(f"Prophet — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Prophet forecasting (DEITA port)")
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
        pred, lo, hi = next_step_prophet(prices)
        print(f"Last close      : {prices.iloc[-1]:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  Prophet (weekly+yearly seasonality)\n")
    print("Note: Prophet re-fits every step (refit_freq=1, Stan backend) -- "
          "the backtest can take several minutes.\n")

    result = run_prophet(train, test)
    print(f"=== Prophet — {args.ticker} ===")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
