"""
DeepEdgeBenchmark — visual model comparison
===========================================
Runs all four forecasting models (ARIMA, SARIMA, Prophet, LSTM) on a single
asset over a common train/test split and renders one comparison figure:

  * top    : 1-step-ahead forecasts of every model overlaid on the actual price
  * bottom : a metrics table (best RMSE and best Directional Accuracy highlighted)

The figure is saved to a PNG and the metrics are also printed to the console.

Quick start
-----------
    pip install -r requirements.txt

    python run_benchmark.py                              # SPY, 2023-2024
    python run_benchmark.py --ticker BTC-USD             # different asset
    python run_benchmark.py --start 2020-01-01 --end 2024-12-31  # full window (slow)
    python run_benchmark.py --epochs 30 --out spy.png    # more LSTM epochs / custom path

Notes
-----
SARIMA re-fits every step and LSTM trains a network, so wider windows take
several minutes. Prophet is fit once then batch-predicts the test window.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
import arima_model
import sarima_model
import prophet_model
import lstm_model


METRIC_KEYS = ["RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "Dir. Acc (%)",
               "PI Cov 95% (%)", "Train Time (s)"]
COLORS = {"ARIMA": "tab:blue", "SARIMA": "tab:green",
          "Prophet": "tab:purple", "LSTM": "tab:red"}


def run_all(ticker, start, end, test_ratio, epochs):
    print(f"Fetching {ticker} [{start} -> {end}] ...")
    prices = arima_model.fetch_data(ticker, start, end)
    split = int(len(prices) * (1 - test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"  Train {len(train)}  Test {len(test)}\n")

    runs = {}

    def timed(name, fn):
        print(f"  running {name} ...", flush=True)
        t0 = time.time()
        runs[name] = fn()
        print(f"    done in {time.time() - t0:.1f}s")

    timed("ARIMA",   lambda: arima_model.run_arima_garch(train, test))
    timed("SARIMA",  lambda: sarima_model.run_sarima(train, test))
    timed("Prophet", lambda: prophet_model.run_prophet(train, test))
    timed("LSTM",    lambda: lstm_model.run_lstm(train, test, epochs=epochs))
    return test, runs


def render(test, runs, ticker, out):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.25)

    # forecast overlay
    ax = fig.add_subplot(gs[0])
    idx = test.index
    ax.plot(idx, test.values, color="black", lw=2.0, label="Actual", zorder=5)
    for name, r in runs.items():
        ax.plot(idx, r["predictions"], color=COLORS[name], lw=1.4, alpha=0.9, label=name)
    ax.set_title(f"DeepEdgeBenchmark — 1-step forecasts vs actual  ({ticker}, test window)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend(ncol=5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    # metrics table
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    col_labels = ["Model"] + METRIC_KEYS
    cell_text, row_colors = [], []
    for name, r in runs.items():
        cell_text.append([name] + [str(r.get(k, "N/A")) for k in METRIC_KEYS])
        row_colors.append(COLORS[name])
    tbl = ax2.table(cellText=cell_text, colLabels=col_labels,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#333333")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, rc in enumerate(row_colors, start=1):
        tbl[i, 0].set_text_props(color=rc, fontweight="bold")

    names = list(runs)
    rmse_vals = [float(runs[n]["RMSE"]) for n in names]
    dir_vals  = [float(runs[n]["Dir. Acc (%)"]) for n in names]
    tbl[int(np.argmin(rmse_vals)) + 1, 1].set_facecolor("#c8f7c5")
    tbl[int(np.argmax(dir_vals))  + 1, 5].set_facecolor("#c8f7c5")
    ax2.set_title("Metrics  (green = best;  lower RMSE/MAE/MAPE better, higher Dir.Acc better)",
                  fontsize=11, pad=12)

    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure -> {out}")


def print_table(runs):
    print("\n=== METRICS ===")
    hdr = "Model".ljust(9) + "".join(k.rjust(15) for k in METRIC_KEYS)
    print(hdr); print("-" * len(hdr))
    for name, r in runs.items():
        print(name.ljust(9) + "".join(str(r.get(k, "N/A")).rjust(15) for k in METRIC_KEYS))


def main():
    p = argparse.ArgumentParser(description="DeepEdgeBenchmark visual model comparison")
    p.add_argument("--ticker", default="SPY", help="yfinance ticker (SPY, BTC-USD, GC=F)")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=20, help="LSTM training epochs")
    p.add_argument("--out", default="benchmark_visual.png", help="output PNG path")
    args = p.parse_args()

    test, runs = run_all(args.ticker, args.start, args.end, args.test_ratio, args.epochs)
    print_table(runs)
    render(test, runs, args.ticker, args.out)


if __name__ == "__main__":
    main()
