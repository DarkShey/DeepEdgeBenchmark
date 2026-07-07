"""
DeepEdgeBenchmark — visual model comparison
===========================================
Runs all four forecasting models (ARIMA, SARIMA, Prophet, LSTM) on a single
asset over a common train/test split and renders one comparison figure:

  * top    : 1-step-ahead forecasts of every model overlaid on the actual price
  * bottom : a metrics table (best RMSE and best Directional Accuracy highlighted)

The figure is saved to a PNG and the metrics are also printed to the console.
It also exports the run's artifacts to ../artifacts/ (metrics.csv/json, one
forecast PNG per model, the comparison PNG, and a dated REPORT.md) — see
models/BRIEF_finalisation_modeles.md §6. Models themselves are not serialized:
ARIMA/SARIMA re-fit at every walk-forward step (no single fitted model to
save), and for Prophet/LSTM there's no production-serving use case here.

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
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
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
# module + short filename key, one per model — used for save_plot()/artifact filenames.
MODEL_MODULES = {
    "ARIMA":   (arima_model,   "arima"),
    "SARIMA":  (sarima_model,  "sarima"),
    "Prophet": (prophet_model, "prophet"),
    "LSTM":    (lstm_model,    "lstm"),
}
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
VERSION_LIBS = ["numpy", "pandas", "statsmodels", "sklearn", "tensorflow", "prophet", "arch", "yfinance"]


def run_all(ticker, start, end, test_ratio, epochs, seed):
    lstm_model.set_seed(seed)   # only LSTM has RNG-dependent training; harmless for the rest

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


# ── Artifacts (BRIEF_finalisation_modeles.md §6) ─────────────────────────────

METRICS_CSV_FIELDS = [
    "model", "ticker", "start", "end", "test_ratio", "n_test",
    "RMSE", "MAE", "MAPE", "DirAcc_%", "PI_cov95_%", "runtime_s", "seed",
]


def build_metrics_rows(runs, ticker, start, end, test_ratio, n_test, seed):
    """One row per model — the main deliverable per the brief: a reproducible,
    out-of-sample metrics table, not a serialized model (see module docstring)."""
    rows = []
    for name, r in runs.items():
        rows.append({
            "model": name, "ticker": ticker, "start": start, "end": end,
            "test_ratio": test_ratio, "n_test": n_test,
            "RMSE": r.get("RMSE"), "MAE": r.get("MAE"), "MAPE": r.get("MAPE (%)"),
            "DirAcc_%": r.get("Dir. Acc (%)"), "PI_cov95_%": r.get("PI Cov 95% (%)"),
            "runtime_s": r.get("Train Time (s)"), "seed": seed,
        })
    return rows


def write_metrics_files(rows, artifacts_dir):
    csv_path = artifacts_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = artifacts_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved metrics -> {csv_path}, {json_path}")


def save_forecast_plots(runs, ticker, artifacts_dir):
    """Reuses each model's own save_plot() (actual vs predicted + 95% PI band) —
    no new plotting logic, just routes the output into artifacts/."""
    for name, result in runs.items():
        module, short_key = MODEL_MODULES[name]
        path = artifacts_dir / f"forecast_{short_key}_{ticker}.png"
        module.save_plot(result, ticker, str(path))


def get_lib_versions():
    import numpy, pandas, statsmodels, sklearn, tensorflow, prophet, arch, yfinance
    mods = {"numpy": numpy, "pandas": pandas, "statsmodels": statsmodels, "sklearn": sklearn,
            "tensorflow": tensorflow, "prophet": prophet, "arch": arch, "yfinance": yfinance}
    return {name: mod.__version__ for name, mod in mods.items()}


def run_test_suite():
    """Re-runs models/ pytest suite live so REPORT.md reflects the actual current
    state of the repo, not a stale/manually-copied result (cf. brief §6-C)."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(models_dir), capture_output=True, text=True,
    )
    duration = time.time() - t0
    summary_line = next(
        (line for line in reversed(proc.stdout.strip().splitlines())
         if "passed" in line or "failed" in line or "error" in line),
        proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)",
    )
    return {"passed": proc.returncode == 0, "summary": summary_line, "duration_s": round(duration, 1)}


def write_report(rows, ticker, start, end, test_ratio, seed, lib_versions, test_result, artifacts_dir):
    lines = [
        "# DeepEdgeBenchmark — run report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Run configuration",
        "",
        f"- Ticker: `{ticker}`",
        f"- Window: `{start}` -> `{end}`",
        f"- Test ratio: `{test_ratio}`",
        f"- Seed: `{seed}`",
        "",
        "## Library versions",
        "",
    ]
    for name, version in lib_versions.items():
        lines.append(f"- {name}: `{version}`")
    lines += [
        "",
        "## Test suite (models/)",
        "",
        f"- Result: {'**SKIPPED**' if test_result['passed'] is None else ('**PASS**' if test_result['passed'] else '**FAIL**')}",
        f"- Summary: `{test_result['summary']}`",
        f"- Duration: {test_result['duration_s']}s",
        "",
        "## Metrics (out-of-sample, walk-forward)",
        "",
        "| " + " | ".join(METRICS_CSV_FIELDS) + " |",
        "|" + "---|" * len(METRICS_CSV_FIELDS),
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in METRICS_CSV_FIELDS) + " |")
    lines.append("")

    report_path = artifacts_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved report -> {report_path}")


def main():
    p = argparse.ArgumentParser(description="DeepEdgeBenchmark visual model comparison")
    p.add_argument("--ticker", default="SPY", help="yfinance ticker (SPY, BTC-USD, GC=F)")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=20, help="LSTM training epochs")
    p.add_argument("--seed", type=int, default=lstm_model.DEFAULT_SEED,
                   help="RNG seed for reproducible LSTM training")
    p.add_argument("--out", default=None,
                   help="comparison PNG path (default: artifacts/benchmark_visual.png)")
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR),
                   help="directory to export metrics/plots/report into")
    p.add_argument("--skip-tests", action="store_true",
                   help="skip re-running the models/ pytest suite for REPORT.md")
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(artifacts_dir / "benchmark_visual.png")

    test, runs = run_all(args.ticker, args.start, args.end, args.test_ratio, args.epochs, args.seed)
    print_table(runs)
    render(test, runs, args.ticker, out_path)

    rows = build_metrics_rows(runs, args.ticker, args.start, args.end,
                              args.test_ratio, len(test), args.seed)
    write_metrics_files(rows, artifacts_dir)
    save_forecast_plots(runs, args.ticker, artifacts_dir)

    if args.skip_tests:
        test_result = {"passed": None, "summary": "(skipped, --skip-tests)", "duration_s": 0.0}
    else:
        print("\nRe-running models/ test suite for the report ...")
        test_result = run_test_suite()
        print(f"  {test_result['summary']} ({test_result['duration_s']}s)")

    write_report(rows, args.ticker, args.start, args.end, args.test_ratio, args.seed,
                get_lib_versions(), test_result, artifacts_dir)


if __name__ == "__main__":
    main()
