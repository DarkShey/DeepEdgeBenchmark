"""
honest_benchmark.py — regenerated, honest DeepEdgeBenchmark (IMPROVEMENTS_BRIEF.md)
==================================================================================
Ties the ``honest_eval`` library to the repo's ARIMA/SARIMA/Prophet/LSTM runners
and produces a dashboard where every performance claim is measured against the
*corrected* naive with explicit statistical uncertainty.

What it does, per the brief:
  Point 0  verifies the naive baseline == previous close (no injected noise)
  Point 1  D+1 metrics on CHANGES: MASE, Theil's U, ρ(Δ), DirAcc±95% CI, DM
  Point 3  dense daily rolling-origin D+1/7/30 (+ error-vs-horizon curve),
           Diebold-Mariano with Newey-West variance
  Point 4  volatility (QLIKE/PIT/Winkler) & direction (AUC/Brier) targets

Usage
-----
  # real data (needs network; heavy models refit per origin — use --step)
  python honest_benchmark.py --ticker BTC-USD --models arima sarima --step 3
  python honest_benchmark.py --offline prices.csv --models arima
  python honest_benchmark.py --demo            # synthetic, offline smoke test

  --no-multistep / --no-targets  skip the expensive stages
  --out honest.png               save PNGs (honest_variations.png, …)
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from honest_eval import metrics, naive, multistep, targets, report


MODEL_LABELS = {"arima": "ARIMA-GARCH", "sarima": "SARIMA",
                "prophet": "Prophet", "lstm": "LSTM"}


def _add_models_dir_to_path() -> None:
    """Les forecasters vivent dans models/ depuis la réorganisation du repo
    (anciennement à la racine) — même convention que benchmarks/multi_horizon.py."""
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    for d in (root, os.path.join(root, "models")):
        if d not in sys.path:
            sys.path.insert(0, d)


# ── data loading ─────────────────────────────────────────────────────────────

def load_prices(args) -> pd.Series:
    if args.demo:
        rng = np.random.default_rng(args.seed)
        n = args.demo_n
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        rets = rng.normal(0.0003, 0.02, n)          # geometric RW ~ crypto-ish
        return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="Close")
    if args.offline:
        df = pd.read_csv(args.offline, index_col=0, parse_dates=True)
        col = "Close" if "Close" in df.columns else df.columns[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s.astype(float)
    _add_models_dir_to_path()
    from arima_model import fetch_data                # reuse the repo's loader
    return fetch_data(args.ticker, args.start, args.end)


# ── D+1 model runs (Point 1) ─────────────────────────────────────────────────

def run_d1(train, test, model_keys, epochs):
    """Run each model's 1-step walk-forward and return {key: result dict}."""
    _add_models_dir_to_path()
    out = {}
    for k in model_keys:
        print(f"[D+1] {MODEL_LABELS[k]} ...", flush=True)
        try:
            if k == "arima":
                from arima_model import run_arima_garch; r = run_arima_garch(train, test)
            elif k == "sarima":
                from sarima_model import run_sarima; r = run_sarima(train, test)
            elif k == "prophet":
                from prophet_model import run_prophet; r = run_prophet(train, test)
            elif k == "lstm":
                from lstm_model import run_lstm; r = run_lstm(train, test, epochs=epochs)
            else:
                continue
            out[k] = r
        except Exception as exc:
            print(f"[D+1] {MODEL_LABELS[k]} FAILED: {exc}")
    return out


def d1_kpi_rows(d1_results, naive_rows):
    """Build honest KPI rows (Point 1) from D+1 model results vs the naive."""
    prev = naive_rows["prev"]; actual = naive_rows["actual"]
    err_n = actual - naive_rows["predictions"]
    rows = [{
        "name": "Naive (ref)", "rmse": metrics.rmse(actual, naive_rows["predictions"]),
        "mase": 1.0, "theil_u": 1.0, "change_corr": float("nan"),
        "dir_acc": float("nan"), "dir_ci95": (float("nan"), float("nan")),
        "dir_p": float("nan"),
        "coverage": metrics.coverage(actual, naive_rows["lower"], naive_rows["upper"]),
        "dm": float("nan"), "dm_p": float("nan"), "verdict": "reference",
    }]
    d1_view = {}
    for k, r in d1_results.items():
        pred = np.asarray(r["predictions"], float)
        err_m = actual - pred
        dm, p, _ = metrics.diebold_mariano(err_m, err_n, h=1)
        da = metrics.directional_accuracy(pred, prev, actual)
        u = metrics.theil_u(actual, pred, naive_rows["predictions"])
        cov = (metrics.coverage(actual, r["lower"], r["upper"])
               if r.get("lower") is not None else float("nan"))
        rows.append({
            "name": MODEL_LABELS[k], "rmse": metrics.rmse(actual, pred),
            "mase": metrics.mase(actual, pred, naive_rows["predictions"]),
            "theil_u": u, "change_corr": metrics.change_correlation(pred, prev, actual),
            "dir_acc": da["acc"], "dir_ci95": da["ci95"], "dir_p": da["p_vs_coin"],
            "coverage": cov, "dm": dm, "dm_p": p,
            "verdict": metrics.skill_verdict(u, p),
        })
        d1_view[k] = {"prev": prev, "actual": actual, "pred": pred, "index": naive_rows["index"]}
    return rows, d1_view


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Honest DeepEdgeBenchmark (IMPROVEMENTS_BRIEF.md)",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--ticker", default="BTC-USD")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--models", nargs="+", default=["arima", "sarima"],
                   choices=list(MODEL_LABELS))
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 30])
    p.add_argument("--step", type=int, default=1,
                   help="rolling-origin stride for multistep (raise for LSTM/SARIMA cost)")
    p.add_argument("--curve-max", type=int, default=0,
                   help="if >0, compute the RMSE/MASE-vs-horizon curve up to this h")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--offline", metavar="CSV", default=None, help="load prices from CSV")
    p.add_argument("--demo", action="store_true", help="synthetic offline data")
    p.add_argument("--demo-n", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-multistep", action="store_true")
    p.add_argument("--no-targets", action="store_true")
    p.add_argument("--out", default=None, help="save PNGs to this base path")
    p.add_argument("--show", action="store_true", help="open interactive windows")
    args = p.parse_args()

    prices = load_prices(args)
    n = len(prices)
    split = int(n * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"\n{'='*70}\n  HONEST BENCHMARK — {args.ticker}  ({n} obs, "
          f"train {len(train)} / test {len(test)})\n{'='*70}")

    # ── Point 0 ──
    ref = naive.naive_random_walk(train, test)
    v = naive.verify_naive(train, test, dashboard_predictions=ref["predictions"])
    print(f"\n[Point 0] naive baseline: predictions == previous close  → "
          f"{'OK' if v['passed'] else 'FAIL'}  (ref RMSE={v['ref_rmse']})")

    # ── Point 1 ──
    d1 = run_d1(train, test, args.models, args.epochs)
    kpi_rows, d1_view = d1_kpi_rows(d1, ref)
    print("\n[Point 1] Honest D+1 KPIs (vs corrected naive):")
    print(f"  {'Model':<14}{'RMSE':>10}{'MASE':>8}{'TheilU':>8}{'rho(D)':>8}"
          f"{'DirAcc%':>9}{'p_coin':>8}{'DM_p':>8}  verdict")

    def _fmt(val, spec, na="--"):
        return format(val, spec) if np.isfinite(val) else na

    for r in kpi_rows:
        rho = _fmt(r["change_corr"], "+.2f", "n/a")
        da = _fmt(r["dir_acc"] * 100, ".1f") if np.isfinite(r["dir_acc"]) else "--"
        pcoin = _fmt(r["dir_p"], ".3f")
        dmp = _fmt(r["dm_p"], ".3f")
        print(f"  {r['name']:<14}{r['rmse']:>10.3f}{r['mase']:>8.3f}"
              f"{r['theil_u']:>8.3f}{rho:>8}{da:>9}{pcoin:>8}{dmp:>8}  {r['verdict']}")

    # ── Point 3 ──
    curves, multi_scores = {}, {}
    if not args.no_multistep:
        print(f"\n[Point 3] dense rolling-origin (step={args.step}) horizons={args.horizons} ...")
        for k in args.models:
            fac = multistep.MODEL_FORECASTERS[k]
            fc = (fac(order=(2, 0, 2)) if k == "arima" else
                  fac(epochs=args.epochs) if k == "lstm" else fac())
            print(f"  [multistep] {MODEL_LABELS[k]} ...", flush=True)
            sc = multistep.evaluate_model_multih(
                prices, fc, horizons=tuple(args.horizons),
                test_start=split, step=args.step)
            multi_scores[k] = sc
            for h in args.horizons:
                s = sc[h]
                print(f"    D+{h:<2} n={s['n']:<4} RMSE={s['rmse']:.3f} "
                      f"MASE={s['mase']:.3f} U={s['theil_u']:.3f} "
                      f"DirAcc={s['dir_acc']*100:.1f}% DM_p={s['dm_p']:.3f}  {s['verdict']}")
            if args.curve_max > 0:
                curves[k] = multistep.error_vs_horizon(
                    prices, fc, horizons=range(1, args.curve_max + 1),
                    test_start=split, step=args.step)

    # ── Point 4 ──
    vol_res = dir_res = None
    if not args.no_targets:
        print("\n[Point 4] reformulated targets:")
        vol_res = targets.evaluate_volatility(prices.values, test_ratio=args.test_ratio)
        print(f"  volatility  persistence QLIKE={vol_res['_baseline']['qlike']:.4f} | "
              f"ewma={vol_res['ewma']['qlike']:.4f} "
              f"({'beats' if vol_res['ewma']['beats_persistence'] else 'no'}) | "
              f"garch={vol_res['garch']['qlike']:.4f} "
              f"({'beats' if vol_res['garch']['beats_persistence'] else 'no'})")
        y_up = targets.direction_labels(prices.values)[split:]
        # direction probability implied by the first model's D+1 predictive law
        if d1:
            k0 = args.models[0]
            r0 = d1[k0]
            sig = np.maximum((np.asarray(r0["upper"]) - np.asarray(r0["lower"])) / (2 * 1.96), 1e-9)
            prob_up = targets.implied_prob_up(r0["predictions"], sig, ref["prev"])
            m = min(len(prob_up), len(y_up))
            dir_res = targets.evaluate_direction(y_up[-m:], prob_up[-m:])
            print(f"  direction ({MODEL_LABELS[k0]})  AUC={dir_res['auc']} "
                  f"Brier={dir_res['brier']} (alwaysup={dir_res['brier_alwaysup']}) "
                  f"p_coin={dir_res['binom_p_vs_coin']}  → {dir_res['verdict']}")

    # ── render ──
    if args.out or args.show:
        figs = {"variations": report.plot_variations_panel(d1_view, ref, args.ticker),
                "kpi": report.plot_kpi_table(kpi_rows, args.ticker)}
        if curves:
            figs["horizon"] = report.plot_error_vs_horizon(curves, ticker=args.ticker)
        if vol_res or dir_res:
            figs["targets"] = report.plot_targets_panel(vol_res, dir_res, args.ticker)
        if args.out:
            paths = report.save_all(figs, args.out)
            print("\n[report] saved: " + ", ".join(paths))
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()

    print(f"\n{'='*70}\n  Done.\n{'='*70}")


if __name__ == "__main__":
    main()
