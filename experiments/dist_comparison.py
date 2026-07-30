"""
dist_comparison.py — compare ARIMA-GARCH innovation distributions (normal vs GED vs
skew-t) on PI calibration (alternatives_distributions_pi.pdf, option 1: "quasi
gratuit" -- dist='ged'/'skewt' in arch_model, no other code path touched).

Context: ARIMA-GARCH (and the other 4 models) over-cover at the 50% PI (~0.57-0.66
observed vs 0.50 target) while being roughly correct at 95% -- diagnostic of a
too-wide-at-the-center gaussian/symmetric assumption. This reruns the SAME
train/test split once per distribution (models.arima_model.run_arima_garch(...,
dist=...)) -- the ARIMA mean equation is refit identically each time (dist only
changes the GARCH innovation law), so this is N short backtests, not N full
retrains: no other model (SARIMA/LSTM/Naive), no data-prep change.

KPIs reported per distribution:
  - PI coverage at 50/80/95% (target: match the nominal level -- the direct
    over-coverage diagnostic from the PDF)
  - PI width at 50/80/95% (sharpness -- a wider interval "cheats" coverage, so
    read both together)
  - CRPS, computed on samples drawn from the FITTED distribution's own ppf
    (run_arima_garch(..., n_crps_samples=...) -- NOT the model's existing
    bootstrap-based `ensemble` used for the production Gate 2 KPI, which
    resamples empirical residuals regardless of `dist` and would therefore barely
    differ between normal/ged/skewt)
  - RMSE/MAE (point accuracy -- should barely move, same mean equation)

Usage (from repo root; set CURL_CA_BUNDLE first if yfinance SSL fails locally --
see weekly_multimodel.py's Avast Web Shield note):
    python experiments/dist_comparison.py
    python experiments/dist_comparison.py --tickers BTC-USD SPY --dists normal ged skewt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

import arima_model as am           # noqa: E402
from crps_metrics import crps_empirical  # noqa: E402

DEFAULT_DISTS = ("normal", "ged", "skewt")
PI_COLS = ("PI Cov 50% (%)", "PI Width 50% (%)",
           "PI Cov 80% (%)", "PI Width 80% (%)",
           "PI Cov 95% (%)", "PI Width 95% (%)")


def crps_from_ensembles(ensembles, actual) -> float:
    actual = np.asarray(actual, dtype=float)
    scores = [crps_empirical(e, a) for e, a in zip(ensembles, actual)]
    return float(np.mean(scores))


def run_one(ticker: str, start: str, end: str, test_ratio: float, dist: str,
           n_crps_samples: int, seed: int) -> dict:
    prices = am.fetch_data(ticker, start, end)
    split = int(len(prices) * (1 - test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]

    result = am.run_arima_garch(train, test, dist=dist,
                                n_crps_samples=n_crps_samples, ensemble_seed=seed)
    crps = crps_from_ensembles(result["crps_ensemble"], result["actual"])

    row = {"Ticker": ticker, "Dist": dist, "CRPS": round(crps, 4),
          "RMSE": result["RMSE"], "MAE": result["MAE"],
          "Train Time (s)": result["Train Time (s)"]}
    row.update({col: result[col] for col in PI_COLS})
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["BTC-USD"])
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--dists", nargs="+", default=list(DEFAULT_DISTS))
    p.add_argument("--n-crps-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "dist_comparison_results.json"))
    args = p.parse_args()

    rows = []
    for ticker in args.tickers:
        for dist in args.dists:
            print(f"[{ticker}] ARIMA-GARCH(dist={dist}) ...")
            row = run_one(ticker, args.start, args.end, args.test_ratio,
                          dist, args.n_crps_samples, args.seed)
            rows.append(row)
            print(f"  -> CRPS={row['CRPS']}  "
                 f"Cov50={row['PI Cov 50% (%)']}%  Cov80={row['PI Cov 80% (%)']}%  "
                 f"Cov95={row['PI Cov 95% (%)']}%")

    df = pd.DataFrame(rows).set_index(["Ticker", "Dist"])
    print(f"\n=== ARIMA-GARCH -- distribution comparison ===")
    print(df.to_string())

    Path(args.out).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
