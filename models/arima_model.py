"""
ARIMA-GARCH Forecasting Model
=============================
Standalone port of the ARIMA model from the DEITA time-series benchmark.

What it does
------------
The "ARIMA" entry in the DEITA benchmark is really an **ARIMA(2,0,2)** mean
equation fitted on log-returns, paired with a **GARCH(1,1)** volatility model
that produces the 95% prediction intervals. Forecasting is done **walk-forward**
(rolling 1-step-ahead): each test point is predicted, then the realised value is
appended to the history before predicting the next.

This file is fully self-contained — it does not depend on any other DEITA
module. Run it directly from the command line.

Quick start
-----------
    pip install numpy pandas yfinance statsmodels arch scikit-learn matplotlib

    # default: BTC-USD, 2020-2024, last 15% as test set
    python arima_model.py

    # pick an asset / window
    python arima_model.py --ticker SPY --start 2018-01-01 --end 2024-12-31

    # save the forecast plot instead of just printing metrics
    python arima_model.py --ticker GC=F --plot forecast.png

    # one-step forecast beyond the last observation (no backtest)
    python arima_model.py --ticker BTC-USD --next-step
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Config (defaults; override via CLI) ──────────────────────────────────────
ARIMA_ORDER      = (2, 0, 2)   # mean equation on log-returns (d=0: returns are stationary)
GARCH_REFIT_FREQ = 20          # re-fit GARCH every N steps during the walk-forward
Z_95             = 1.96        # 95% prediction interval z-score


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily Close prices and return a clean, tz-naive Series."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker} between {start} and {end}.")

    # yfinance >= 0.2 may return MultiIndex columns for a single ticker
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

    actual_dir = np.sign(np.diff(actual))
    pred_dir   = np.sign(np.diff(predicted))
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    residuals = actual - predicted
    try:
        lb_p = acorr_ljungbox(residuals, lags=[10],
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


# ── ARIMA-GARCH walk-forward backtest ────────────────────────────────────────
def run_arima_garch(train_series: pd.Series, test_series: pd.Series,
                    order=ARIMA_ORDER, garch_refit_freq=GARCH_REFIT_FREQ,
                    n_ensemble: int = 0, ensemble_seed=None) -> dict:
    """Rolling 1-step-ahead ARIMA-GARCH forecast over the test window.

    Mean equation : ARIMA(order) on 100*log-returns
    Volatility    : GARCH(1,1) on the ARIMA residuals -> 95% intervals
    Returns a dict of metrics plus the raw prediction/interval arrays.

    `n_ensemble` (0 = off, default -- no cost for existing callers): at each step,
    additionally draws `n_ensemble` bootstrap samples of the next-step price by
    resampling (with replacement) the GARCH standardized residuals
    (resid / conditional_volatility) and applying them to that step's own mu/sigma
    -- a residual bootstrap around the already-fitted mean/volatility forecast,
    not a fresh distributional assumption. Populates `result["ensemble"]` (list of
    length n_test, one [n_ensemble] price array per step) for empirical CRPS
    (cf. model_artifacts/crps_kpis.py).
    """
    t0 = time.time()

    train_prices = train_series.astype(float).values
    test_prices  = test_series.astype(float).values
    n_test       = len(test_prices)

    log_prices = np.log(train_prices)
    returns    = np.diff(log_prices) * 100.0   # percent log-returns

    preds  = np.empty(n_test)
    lower  = np.empty(n_test)
    upper  = np.empty(n_test)
    sigmas = np.empty(n_test)
    ensembles = [] if n_ensemble > 0 else None
    rng = np.random.default_rng(ensemble_seed) if n_ensemble > 0 else None

    arima_res = ARIMA(
        returns, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit()

    resid = np.asarray(arima_res.resid, dtype=float)
    garch_res = arch_model(
        resid, vol="Garch", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")

    last_price = train_prices[-1]

    for t in range(n_test):
        mu = arima_res.forecast(steps=1)
        mu = float(mu.iloc[0] if hasattr(mu, "iloc") else mu[0]) / 100.0

        if t % garch_refit_freq == 0:
            resid = np.asarray(arima_res.resid, dtype=float)
            garch_res = arch_model(
                resid, vol="Garch", p=1, q=1, dist="normal", rescale=False
            ).fit(disp="off")

        garch_fc = garch_res.forecast(horizon=1, reindex=False)
        sigma    = np.sqrt(garch_fc.variance.values[-1, 0]) / 100.0

        preds[t]  = last_price * np.exp(mu)
        lower[t]  = last_price * np.exp(mu - Z_95 * sigma)
        upper[t]  = last_price * np.exp(mu + Z_95 * sigma)
        sigmas[t] = sigma

        if n_ensemble > 0:
            std_resid = np.asarray(garch_res.resid, dtype=float) / \
                np.asarray(garch_res.conditional_volatility, dtype=float)
            z_boot = rng.choice(std_resid, size=n_ensemble, replace=True)
            ensembles.append(last_price * np.exp(mu + sigma * z_boot))

        # walk forward: append the realised return, then move on
        actual_price = test_prices[t]
        actual_ret   = np.log(actual_price / last_price) * 100.0
        arima_res    = arima_res.append([actual_ret], refit=False)
        last_price   = actual_price

    train_time = time.time() - t0

    metrics = compute_metrics(
        test_prices, preds, pi_lower=lower, pi_upper=upper, train_time=train_time
    )
    metrics["Avg GARCH sigma"] = round(float(np.mean(sigmas)), 6)

    result = {
        **metrics,
        "predictions": preds,
        "lower": lower,
        "upper": upper,
        "index": test_series.index,
        "actual": test_prices,
    }
    if ensembles is not None:
        result["ensemble"] = ensembles
    return result


def next_step_arima_garch(series: pd.Series, order=ARIMA_ORDER):
    """Single 1-step forecast beyond the last observation (no backtest).

    Returns (point_forecast, pi_low_95, pi_high_95).
    """
    prices  = series.astype(float).values
    returns = np.diff(np.log(prices)) * 100.0

    arima_res = ARIMA(
        returns, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit()
    fc = arima_res.forecast(steps=1)
    mu = float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0]) / 100.0

    resid = np.asarray(arima_res.resid, dtype=float)
    garch_res = arch_model(
        resid, vol="Garch", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")
    garch_fc = garch_res.forecast(horizon=1, reindex=False)
    sigma    = np.sqrt(garch_fc.variance.values[-1, 0]) / 100.0

    last_price = prices[-1]
    pred   = last_price * np.exp(mu)
    pi_low = last_price * np.exp(mu - Z_95 * sigma)
    pi_hi  = last_price * np.exp(mu + Z_95 * sigma)
    return pred, pi_low, pi_hi


# ── Plot (optional) ──────────────────────────────────────────────────────────
def save_plot(result: dict, ticker: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = result["index"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, result["actual"], label="Actual", color="black", lw=1.3)
    ax.plot(idx, result["predictions"], label="ARIMA-GARCH forecast",
            color="tab:blue", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:blue",
                    alpha=0.20, label="95% PI")
    ax.set_title(f"ARIMA{ARIMA_ORDER}-GARCH(1,1) — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="ARIMA-GARCH forecasting (DEITA port)")
    p.add_argument("--ticker", default="BTC-USD", help="yfinance ticker (e.g. BTC-USD, SPY, GC=F)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15,
                   help="fraction of the series held out as the test set")
    p.add_argument("--order", default="2,0,2",
                   help="ARIMA order p,d,q on log-returns (default 2,0,2)")
    p.add_argument("--next-step", action="store_true",
                   help="only forecast the single next step (no backtest)")
    p.add_argument("--plot", metavar="PATH", default=None,
                   help="save a forecast plot to PATH (e.g. forecast.png)")
    args = p.parse_args()

    order = tuple(int(x) for x in args.order.split(","))

    print(f"Downloading {args.ticker} [{args.start} -> {args.end}] ...")
    prices = fetch_data(args.ticker, args.start, args.end)
    print(f"  {len(prices)} daily observations.\n")

    if args.next_step:
        pred, lo, hi = next_step_arima_garch(prices, order=order)
        last = prices.iloc[-1]
        print(f"Last close      : {last:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  "
          f"ARIMA order: {order}, GARCH(1,1)\n")

    result = run_arima_garch(train, test, order=order)

    print(f"=== ARIMA{order}-GARCH(1,1) — {args.ticker} ===")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
