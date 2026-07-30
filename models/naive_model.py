"""
Naive (persistence) Forecasting Model
======================================
Baseline model for the DeepEdgeBenchmark model comparison.

What it does
------------
The canonical persistence baseline: **tomorrow = yesterday's price, exactly**.
No fitting, no statistics, no perturbation — this is the bar every other model
in models/ must clear to be worth using. Forecasting is walk-forward
1-step-ahead: each step uses the realised previous price (never its own past
prediction), which is the standard definition of a persistence/naive baseline.

    IMPORTANT (Point 0 of IMPROVEMENTS_BRIEF.md): an earlier version perturbed
    the prediction by a uniform ±5% drift "by design". That inflated the naive
    RMSE ×1.5–×10 (std of U(-5%,5%) ≈ 2.9% of price vs realised daily moves of
    0.3–3%), which made every model look like it beat the baseline when none
    did. A handicapped baseline is not a baseline. Do not reintroduce noise
    here — `honest_eval.naive.verify_naive()` audits this file's output and
    fails if predictions deviate from the previous close.

95% prediction interval: Gaussian random-walk band around the previous price,
`prev ± 1.96·σ` where σ is the standard deviation of the 1-day price changes
of the training window (same convention as the other models' residual-based
intervals; grows like √h for multi-step, see benchmarks/multi_horizon.py).

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
Z_95          = 1.96   # 95% prediction interval z-score (Gaussian random walk)
DEFAULT_SEED  = 42     # kept for interface compatibility (model is deterministic)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """No-op kept for interface compatibility: the persistence baseline is
    deterministic by definition (callers such as model_artifacts/pipeline.py
    invoke set_seed on every model uniformly)."""
    np.random.seed(seed)


def train_sigma(train: pd.Series) -> float:
    """Std of the 1-step price changes of the training window — the Gaussian
    random-walk scale used for the 95% PI."""
    return float(np.std(np.diff(np.asarray(train, dtype=float))))


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
def run_naive(train: pd.Series, test: pd.Series,
              n_ensemble: int = 0, ensemble_seed=None) -> dict:
    """Rolling 1-step-ahead persistence forecast: pred_t = actual_{t-1}, exactly.

    Walk-forward: uses the realised previous price at every step (train's last price
    for the first test point, then test's own realised prices), never its own prediction.
    95% PI: prev ± 1.96·σ, σ = std of the train-set 1-day changes.

    `n_ensemble` (0 = off, default -- no cost for existing callers): at each step,
    additionally draws `n_ensemble` samples from the same Gaussian random-walk band
    already used for the 95% PI (prev ± 1.96σ) -- not a new distributional assumption,
    just materializing the existing one as a cloud. Populates result["ensemble"] (list
    of length n_test, one [n_ensemble] price array per step) for empirical CRPS
    (cf. model_artifacts/crps_kpis.py).
    """
    t0 = time.time()
    prev_prices = np.concatenate([[train.iloc[-1]], test.values[:-1].astype(float)])

    preds = prev_prices.copy()               # persistence: no drift, no noise
    sigma = train_sigma(train)
    lower = prev_prices - Z_95 * sigma
    upper = prev_prices + Z_95 * sigma

    ensemble = None
    if n_ensemble > 0:
        rng = np.random.default_rng(ensemble_seed)
        noise = rng.normal(0.0, sigma, size=(len(prev_prices), n_ensemble))
        ensemble = [prev_prices[t] + noise[t] for t in range(len(prev_prices))]

    train_time = time.time() - t0
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    result = {**metrics, "predictions": preds, "lower": lower, "upper": upper,
              "index": test.index, "actual": test.values}
    if ensemble is not None:
        result["ensemble"] = ensemble
    return result


def next_step_naive(series: pd.Series):
    """Single 1-step forecast beyond the last observation. Returns (pred, lo, hi)."""
    last_price = float(series.iloc[-1])
    sigma = train_sigma(series)
    return last_price, last_price - Z_95 * sigma, last_price + Z_95 * sigma


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
                    alpha=0.20, label="95% PI (Gaussian RW)")
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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="kept for interface compatibility (model is deterministic)")
    p.add_argument("--next-step", action="store_true", help="only forecast the next step")
    p.add_argument("--plot", metavar="PATH", default=None, help="save a forecast plot")
    args = p.parse_args()

    set_seed(args.seed)

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
