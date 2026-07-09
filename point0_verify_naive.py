"""
point0_verify_naive.py — audit a dashboard naive baseline (Point 0)
===================================================================
Point 0 of IMPROVEMENTS_BRIEF.md: the external `Run` pipeline's "Naive" baseline
injects ~3% Gaussian noise, inflating its RMSE ×1.5–×10 and invalidating every
"vs Naive" column and Diebold-Mariano test.  Feed that pipeline's prices +
naive predictions here to confirm whether the injection has been removed.

Acceptance criterion:
  * the naive prediction equals the previous close *exactly*;
  * |RMSE_dashboard − RMSE_recomputed| / RMSE_recomputed < 0.1%.

Usage
-----
  # CSV needs a date index + a 'Close' column; naive predictions optional
  python point0_verify_naive.py --prices prices.csv --naive naive_preds.csv
  python point0_verify_naive.py --prices prices.csv --test-ratio 0.15
  python point0_verify_naive.py --demo        # show the check on a clean vs noisy baseline
"""

import argparse
import sys

import numpy as np
import pandas as pd

from honest_eval import naive


def _load_series(path, col=None):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    c = col or ("Close" if "Close" in df.columns else df.columns[0])
    s = pd.to_numeric(df[c], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s.astype(float)


def _report(name, rep):
    status = "PASS" if rep["passed"] else "FAIL"
    print(f"\n[{status}] {name}")
    print(f"  recomputed naive RMSE : {rep['ref_rmse']}")
    if "dashboard_rmse" in rep:
        print(f"  dashboard naive RMSE  : {rep['dashboard_rmse']}")
    if "max_abs_pred_diff" in rep:
        print(f"  max |pred − prev close|: {rep['max_abs_pred_diff']}")
    if "rmse_rel_error" in rep:
        print(f"  RMSE relative error   : {rep['rmse_rel_error']*100:.4f}%")
    for issue in rep["issues"]:
        print(f"  ! {issue}")


def main():
    p = argparse.ArgumentParser(description="Point 0 — verify the naive baseline")
    p.add_argument("--prices", help="CSV of prices (date index, Close column)")
    p.add_argument("--naive", help="CSV of the pipeline's naive predictions (test window)")
    p.add_argument("--dashboard-rmse", type=float, default=None,
                   help="the RMSE the dashboard reported for its naive baseline")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-01", periods=400, freq="D")
        prices = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 400)), index=idx)
        split = int(len(prices) * 0.85)
        train, test = prices.iloc[:split], prices.iloc[split:]
        clean = naive.naive_random_walk(train, test)["predictions"]
        noisy = clean + rng.normal(0, 0.03 * test.mean(), len(test))
        _report("clean baseline (previous close)",
                naive.verify_naive(train, test, dashboard_predictions=clean))
        _report("noise-injected baseline (the bug)",
                naive.verify_naive(train, test, dashboard_predictions=noisy))
        return

    if not args.prices:
        sys.exit("Provide --prices CSV (or --demo).")
    prices = _load_series(args.prices)
    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]

    dash_preds = None
    if args.naive:
        dash_preds = _load_series(args.naive).values
        if len(dash_preds) != len(test):
            print(f"[warn] naive length {len(dash_preds)} != test length {len(test)}; "
                  "check the test window alignment.")

    rep = naive.verify_naive(train, test, dashboard_predictions=dash_preds,
                             dashboard_rmse=args.dashboard_rmse)
    _report(f"{args.prices}", rep)
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
