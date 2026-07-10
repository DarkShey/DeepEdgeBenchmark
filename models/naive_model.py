"""
Naive (persistence) Forecasting Model
======================================
Baseline model for the DeepEdgeBenchmark model comparison.

What it does
------------
The simplest possible baseline: **tomorrow = today's closing price, exactly**. No
drift, no randomness, no fitting — this is the bar every other model in models/ must
clear to be worth using. Forecasting is walk-forward 1-step-ahead: each step uses the
realised previous price (never its own past prediction), which is the standard
definition of a persistence/naive baseline.

95% prediction interval: gaussian, `prediction +/- 1.96 * sigma`, where sigma is the
standard deviation of the train series' day-over-day price changes (Close_t -
Close_{t-1}) — estimated once from train, then held fixed across the whole backtest
(cf. Hyndman & Athanasopoulos, "naive method with normal errors").

This file is fully self-contained — it does not depend on any other DEITA module, and
mirrors the interface of arima_model.py / sarima_model.py / prophet_model.py / lstm_model.py
(fetch_data, compute_metrics, run_naive, next_step_naive, save_plot, main) so it plugs into
the same benchmark tooling.

Quick start
-----------
    pip install numpy pandas yfinance scikit-learn statsmodels matplotlib

    python naive_model.py                              # BTC-USD backtest
    python naive_model.py --ticker SPY --plot out.png  # + save forecast plot
    python naive_model.py --ticker GC=F --next-step    # single next-step forecast
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Config (defaults; override via CLI) ──────────────────────────────────────
Z_95          = 1.959963984540054   # scipy.stats.norm.ppf(0.975), inlined to skip a scipy dep
DEFAULT_SEED  = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Kept for interface parity with arima_model.py / sarima_model.py / etc — naive
    is now fully deterministic (no RNG draw left in run_naive/next_step_naive)."""
    np.random.seed(seed)


def train_sigma(train: pd.Series) -> float:
    """Std dev of train's day-over-day price changes — the single sigma used to build
    every 95% PI (+/- Z_95 * sigma * sqrt(h)) for this baseline."""
    diffs = np.diff(np.asarray(train, dtype=float))
    return float(np.std(diffs, ddof=1))


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


# ── Naive walk-forward backtest ──────────────────────────────────────────────
def run_naive(train: pd.Series, test: pd.Series) -> dict:
    """Rolling 1-step-ahead naive forecast: pred_t = actual_{t-1} exactly (strict
    persistence, no drift, no randomness).

    Walk-forward: uses the realised previous price at every step (train's last price
    for the first test point, then test's own realised prices), never its own prediction.
    95% PI: prediction +/- Z_95 * sigma, sigma = train_sigma(train) (fixed, not
    re-estimated per step).
    """
    t0 = time.time()
    prev_prices = np.concatenate([[train.iloc[-1]], test.values[:-1].astype(float)])
    preds = prev_prices.astype(float)

    half = Z_95 * train_sigma(train)
    lower = preds - half
    upper = preds + half

    train_time = time.time() - t0
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    return {**metrics, "predictions": preds, "lower": lower, "upper": upper,
            "index": test.index, "actual": test.values}


def next_step_naive(series: pd.Series):
    """Single 1-step forecast beyond the last observation: exact persistence, gaussian
    95% PI. Returns (pred, lo, hi)."""
    last_price = float(series.iloc[-1])
    half = Z_95 * train_sigma(series)
    return last_price, last_price - half, last_price + half


# ── Plot (optional) ──────────────────────────────────────────────────────────
def save_plot(result: dict, ticker: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = result["index"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, result["actual"], label="Actual", color="black", lw=1.3)
    ax.plot(idx, result["predictions"], label="Naive forecast", color="tab:gray", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:gray",
                    alpha=0.20, label="95% PI (gaussian, +/-1.96*sigma)")
    ax.set_title(f"Naive (persistence) — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Naive persistence forecasting (DeepEdgeBenchmark baseline)")
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
        pred, lo, hi = next_step_naive(prices)
        print(f"Last close      : {prices.iloc[-1]:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  Naive (persistence)\n")

    result = run_naive(train, test)
    print(f"=== Naive — {args.ticker} ===")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
