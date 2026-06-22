"""
Prophet Forecasting Model
=========================
Standalone port of the Prophet model from the DEITA time-series benchmark.

What it does
------------
Facebook/Meta **Prophet** additive regression with weekly + yearly seasonality,
fitted once on the training prices, then predicting the test dates in one batch.
Prediction intervals come from Prophet's own ``yhat_lower`` / ``yhat_upper``
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


# ── Prophet backtest ─────────────────────────────────────────────────────────
def run_prophet(train: pd.Series, test: pd.Series) -> dict:
    """Fit Prophet on train, predict the test dates in one batch."""
    t0 = time.time()
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(train.index),
        "y":  train.astype(float).values.flatten(),
    })
    model = Prophet(interval_width=1 - PI_ALPHA, daily_seasonality=False,
                    weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_train)

    df_future = pd.DataFrame({"ds": pd.to_datetime(test.index)})
    forecast  = model.predict(df_future)
    preds = forecast["yhat"].values
    lower = forecast["yhat_lower"].values
    upper = forecast["yhat_upper"].values

    train_time = time.time() - t0
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    return {**metrics, "predictions": preds, "lower": lower, "upper": upper,
            "index": test.index, "actual": test.values}


def next_step_prophet(series: pd.Series, next_date=None):
    """Single 1-step forecast for the next date. Returns (pred, lo, hi)."""
    if next_date is None:
        next_date = infer_next_date(series)
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(series.index),
        "y":  series.astype(float).values.flatten(),
    })
    model = Prophet(interval_width=1 - PI_ALPHA, daily_seasonality=False,
                    weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_train)
    forecast = model.predict(pd.DataFrame({"ds": [next_date]}))
    return (float(forecast["yhat"].iloc[0]),
            float(forecast["yhat_lower"].iloc[0]),
            float(forecast["yhat_upper"].iloc[0]))


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
    ax.set_title(f"Prophet — {ticker} (fit-once, batch forecast)")
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
