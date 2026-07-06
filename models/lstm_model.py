"""
LSTM Forecasting Model
======================
Standalone port of the LSTM model from the DEITA time-series benchmark.

What it does
------------
A single-layer **LSTM (64 units) -> Dense(1)** trained on MinMax-scaled prices
over a 30-step look-back window. Forecasting is **walk-forward** (rolling
1-step-ahead): the network predicts the next scaled value, the realised value is
fed back into the look-back buffer, and so on. Prediction intervals are
+/- 1.96 * std(training residuals).

This file is fully self-contained — no dependency on any other DEITA module.

Quick start
-----------
    pip install numpy pandas yfinance tensorflow scikit-learn statsmodels matplotlib

    python lstm_model.py                              # BTC-USD backtest
    python lstm_model.py --ticker SPY --plot out.png  # + save forecast plot
    python lstm_model.py --ticker GC=F --next-step    # single next-step forecast

Note: training a neural net is CPU/GPU intensive — the backtest takes a while.
"""

import argparse
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TensorFlow info/warning logs

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ── Config (defaults; override via CLI) ──────────────────────────────────────
SEQ_LEN      = 30     # look-back window
UNITS        = 64     # LSTM hidden units
EPOCHS       = 30
BATCH_SIZE   = 32
DEFAULT_SEED = 42     # --seed default: TF training isn't bit-exact across machines,
                      # but fixing this makes a given run reproducible on the same machine.


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed numpy/tensorflow (and Python's hash-based RNG) for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def make_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm(seq_len: int = SEQ_LEN, units: int = UNITS) -> Sequential:
    model = Sequential()
    model.add(LSTM(units, input_shape=(seq_len, 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


# ── LSTM walk-forward backtest ───────────────────────────────────────────────
def run_lstm(train: pd.Series, test: pd.Series,
             seq_len=SEQ_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE) -> dict:
    """Train on the scaled train window, roll 1-step-ahead over the test window."""
    if len(train) <= seq_len:
        raise ValueError(
            f"train series has {len(train)} points, but seq_len={seq_len} requires "
            f"more than {seq_len} points to build at least one training sequence."
        )
    t0 = time.time()
    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled  = scaler.transform(test.values.reshape(-1, 1))

    X_train, y_train = make_sequences(train_scaled.flatten(), seq_len)
    X_train = X_train.reshape(-1, seq_len, 1)

    model = build_lstm(seq_len)
    es    = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, callbacks=[es], verbose=0)

    preds_scaled = []
    buffer = list(train_scaled.flatten()[-seq_len:])
    for i in range(len(test)):
        x = np.array(buffer[-seq_len:]).reshape(1, seq_len, 1)
        preds_scaled.append(model.predict(x, verbose=0)[0, 0])
        buffer.append(test_scaled[i, 0])     # walk forward with the realised value

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    train_preds = scaler.inverse_transform(
        model.predict(X_train, verbose=0).reshape(-1, 1)).flatten()
    std   = np.std(train.values[seq_len:] - train_preds)
    lower = preds - 1.96 * std
    upper = preds + 1.96 * std

    train_time = time.time() - t0
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    return {**metrics, "predictions": preds, "lower": lower, "upper": upper,
            "index": test.index, "actual": test.values}


def next_step_lstm(series: pd.Series, seq_len=SEQ_LEN, epochs=EPOCHS,
                   batch_size=BATCH_SIZE):
    """Single 1-step forecast beyond the last observation. Returns (pred, lo, hi)."""
    if len(series) <= seq_len:
        raise ValueError(
            f"series has {len(series)} points, but seq_len={seq_len} requires "
            f"more than {seq_len} points to build at least one training sequence."
        )
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y   = make_sequences(scaled, seq_len)
    X      = X.reshape(-1, seq_len, 1)

    model = build_lstm(seq_len)
    es    = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, callbacks=[es], verbose=0)

    x_next      = scaled[-seq_len:].reshape(1, seq_len, 1)
    pred_scaled = model.predict(x_next, verbose=0)[0, 0]
    pred        = float(scaler.inverse_transform([[pred_scaled]])[0, 0])

    train_preds = scaler.inverse_transform(
        model.predict(X, verbose=0).reshape(-1, 1)).flatten()
    std = np.std(series.values[seq_len:] - train_preds)
    return pred, pred - 1.96 * std, pred + 1.96 * std


# ── Plot (optional) ──────────────────────────────────────────────────────────
def save_plot(result: dict, ticker: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = result["index"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, result["actual"], label="Actual", color="black", lw=1.3)
    ax.plot(idx, result["predictions"], label="LSTM forecast", color="tab:red", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:red",
                    alpha=0.20, label="95% PI")
    ax.set_title(f"LSTM({UNITS}) seq={SEQ_LEN} — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="LSTM forecasting (DEITA port)")
    p.add_argument("--ticker", default="BTC-USD", help="yfinance ticker (BTC-USD, SPY, GC=F)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for reproducible training (numpy/tensorflow)")
    p.add_argument("--next-step", action="store_true", help="only forecast the next step")
    p.add_argument("--plot", metavar="PATH", default=None, help="save a forecast plot")
    args = p.parse_args()

    set_seed(args.seed)

    print(f"Downloading {args.ticker} [{args.start} -> {args.end}] ...")
    prices = fetch_data(args.ticker, args.start, args.end)
    print(f"  {len(prices)} daily observations.\n")

    if args.next_step:
        pred, lo, hi = next_step_lstm(prices, epochs=args.epochs)
        print(f"Last close      : {prices.iloc[-1]:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  "
          f"LSTM({UNITS}) seq={SEQ_LEN} epochs={args.epochs}\n")
    print("Note: training a neural net + rolling forecast can take several minutes.\n")

    result = run_lstm(train, test, epochs=args.epochs)
    print(f"=== LSTM({UNITS}) seq={SEQ_LEN} — {args.ticker} ===")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
