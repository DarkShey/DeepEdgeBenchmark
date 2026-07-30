"""
weekly_headtohead_v2.py — final TSDiff-W vs TSDiff-D vs Random-Walk head-to-head,
each TSDiff model trained at its OWN epoch* (selected beforehand by epoch_sweep.py
on a disjoint validation block never touched here), evaluated on the 30 test
origins. Same target dates for all three "models". See BRIEF_weekly_prediction_v2.md
§§5-7.

Guardrail: this script re-derives train_end_pos/val_pos/test_pos with the exact
same three_way_split() call, seed, start/end as epoch_sweep.py, so the test
block is identical to (and disjoint from) the block epoch_sweep.py reserved and
NEVER scored during epoch selection. Epochs come from epoch_sweep_results.json's
"selected_epochs" (argmin CRPS on validation) — never re-derived from test
performance here.

Usage:
    python weekly_headtohead_v2.py                              # reads epoch_sweep_results.json
    python weekly_headtohead_v2.py --epochs-w 60 --epochs-d 100  # override selection (all assets)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
import tsdiff_model as td                                      # noqa: E402

from crps_metrics import crps_empirical                         # noqa: E402
from paired_test import paired_bootstrap_test                   # noqa: E402
from weekly_headtohead import (                                 # noqa: E402
    ASSETS, HORIZON_LABELS, HORIZON_WEEKLY, HORIZON_DAILY,
    build_weekly, standardized_returns,
)
from epoch_sweep import (                                       # noqa: E402
    three_way_split, week_targets, DEFAULT_N_VAL, DEFAULT_N_TEST, DEFAULT_SEED,
)

MODELS = ("TSDiff-W", "TSDiff-D", "RandomWalk")
COMPARISON_PAIRS = (("TSDiff-W", "TSDiff-D"), ("TSDiff-W", "RandomWalk"), ("TSDiff-D", "RandomWalk"))


def random_walk_samples(historical_returns: np.ndarray, h: int) -> np.ndarray:
    """Empirical distribution of h-week cumulative log-returns, built from every
    overlapping h-length window of `historical_returns` (weekly, realised
    strictly before the origin) — the non-parametric floor (brief §5): no model,
    just "how much has price historically moved over h weeks"."""
    historical_returns = np.asarray(historical_returns, dtype=float)
    if len(historical_returns) < h:
        raise ValueError(f"only {len(historical_returns)} historical weekly returns, need >= {h}")
    return np.array([historical_returns[i:i + h].sum()
                     for i in range(len(historical_returns) - h + 1)])


def load_selected_epochs(path) -> dict:
    return json.loads(Path(path).read_text())["selected_epochs"]


def run_pair_v2(asset: str, ticker: str, epochs_w: int, epochs_d: int, n_val: int,
                n_test: int, seed: int, n_samples: int, k_denoise: int,
                start: str, end: str, include_d: bool = True) -> list:
    """`include_d=False` skips fitting/sampling TSDiff-D entirely (only TSDiff-W
    + RandomWalk records are produced) — the daily model's verdict (structurally
    under-calibrated) is already established elsewhere; this is a pure compute
    saving for runs that don't need it (e.g. a multi-asset TSDiff-W-vs-RW sweep).
    `epochs_d` is ignored when include_d=False."""
    print(f"[{asset}] downloading {ticker} ({start} -> {end}) ...")
    daily = td.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)
    train_end_pos, val_pos, test_pos = three_way_split(weekly, n_val, n_test)
    T0_date = weekly_dates.iloc[train_end_pos]
    train_daily = daily.loc[:T0_date]
    train_weekly = weekly.iloc[:train_end_pos + 1]
    print(f"[{asset}] train <= {T0_date.date()} | "
          f"test {weekly_dates.iloc[test_pos[0]].date()} -> "
          f"{weekly_dates.iloc[test_pos[-1]].date()} ({len(test_pos)}) | "
          f"epochs* W={epochs_w}" + (f" D={epochs_d}" if include_d else " (D skipped)"))

    t0 = time.time()
    td.set_seed(seed)
    model_w, mu_w, sd_w = td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY, epochs=epochs_w)
    print(f"[{asset}] TSDiff-W fitted in {time.time() - t0:.0f}s")

    model_d = mu_d = sd_d = daily_z = None
    if include_d:
        t0 = time.time()
        td.set_seed(seed)
        model_d, mu_d, sd_d = td.fit_tsdiff(train_daily, horizon=HORIZON_DAILY, epochs=epochs_d)
        print(f"[{asset}] TSDiff-D fitted in {time.time() - t0:.0f}s")
        daily_z = standardized_returns(daily, mu_d, sd_d)

    weekly_z = standardized_returns(weekly, mu_w, sd_w)
    weekly_r_raw = td._log_returns(weekly.values.astype(float))   # RAW (non-standardized) weekly returns, for RW

    records = []
    for k, m in enumerate(test_pos):
        origin_date, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
        last_price = float(weekly.iloc[m])
        actuals = [float(weekly.iloc[m + h]) for h in (1, 2, 3)]
        hist_weekly_returns = weekly_r_raw[:m]     # guardrail: realised strictly <= origin only

        td.set_seed(seed + k)
        samples_w = td.forecast_from_fitted(model_w, weekly_z[:m], mu_w, sd_w, last_price,
                                            horizons=[1, 2, 3], n_samples=n_samples,
                                            k_denoise=k_denoise)
        samples_d = None
        if include_d:
            td.set_seed(seed + k)
            samples_d = td.forecast_from_fitted(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price,
                                                horizons=daily_horizons, n_samples=n_samples,
                                                k_denoise=k_denoise)

        for wi, w_label in enumerate(HORIZON_LABELS):
            actual = actuals[wi]
            h_d = daily_horizons[wi]
            model_samples = {
                "TSDiff-W": samples_w[wi + 1],
                "RandomWalk": last_price * np.exp(random_walk_samples(hist_weekly_returns, wi + 1)),
            }
            if include_d:
                model_samples["TSDiff-D"] = samples_d[h_d]
            for model_name, samples in model_samples.items():
                # brief §5: RW's point forecast is "tomorrow = today", NOT the sample mean
                point = last_price if model_name == "RandomWalk" else float(np.mean(samples))
                lo, hi = (float(q) for q in np.quantile(samples, [0.025, 0.975]))
                records.append({
                    "asset": asset, "horizon": w_label, "model": model_name,
                    "origin": k, "origin_date": str(origin_date.date()),
                    "target_date": str(target_dates[wi].date()),
                    "daily_steps": h_d if model_name == "TSDiff-D" else None,
                    "actual": actual, "point": point, "lower": lo, "upper": hi,
                    "width": hi - lo,
                    "crps": crps_empirical(samples, actual),
                    "in_interval": bool(lo <= actual <= hi),
                })
        print(f"[{asset}] test origin {k + 1}/{len(test_pos)} ({origin_date.date()}) done")

    return records


def summarize_v2(records: list) -> dict:
    """{asset: {horizon: {model: {RMSE, Cov95, AvgWidth, CRPS, n_origins}}}}"""
    df = pd.DataFrame(records)
    out = {}
    for (asset, horizon, model), g in df.groupby(["asset", "horizon", "model"]):
        rmse = float(np.sqrt(np.mean((g["point"] - g["actual"]) ** 2)))
        cov95 = float(g["in_interval"].mean())
        crps = float(g["crps"].mean())
        width = float(g["width"].mean())
        out.setdefault(asset, {}).setdefault(horizon, {})[model] = {
            "RMSE": round(rmse, 4), "Cov95": round(cov95, 4),
            "AvgWidth": round(width, 4), "CRPS": round(crps, 4), "n_origins": int(len(g)),
        }
    return out


def run_paired_tests(records: list, n_boot: int = 10000, seed: int = 0) -> dict:
    """Per (asset, horizon, pair): paired bootstrap on per-origin CRPS
    differences (brief §6). Keyed "asset|horizon|A vs B", diff = CRPS_A - CRPS_B
    (negative mean_diff => A has lower/better CRPS than B)."""
    df = pd.DataFrame(records)
    out = {}
    for (asset, horizon), g in df.groupby(["asset", "horizon"]):
        piv = g.pivot(index="origin", columns="model", values="crps")
        for a, b in COMPARISON_PAIRS:
            if a not in piv.columns or b not in piv.columns:
                continue    # e.g. TSDiff-D absent when run_pair_v2(include_d=False)
            diffs = (piv[a] - piv[b]).values
            out[f"{asset}|{horizon}|{a} vs {b}"] = paired_bootstrap_test(diffs, n_boot=n_boot, seed=seed)
    return out


def g1_g2_flags(paired_tests: dict) -> dict:
    """Mechanical (non-narrative) verdict flags per (asset, horizon) from the
    paired tests — brief §7. G1: does TSDiff-W or TSDiff-D beat RandomWalk
    significantly on CRPS (lower is better -> a significant NEGATIVE diff means
    the TSDiff model wins). G2: is TSDiff-W vs TSDiff-D significant. These flags
    are inputs to the honest written verdict, not a substitute for it — a
    non-significant result at n=30 is not proof of "no difference"."""
    by_key = {}
    for key, res in paired_tests.items():
        asset, horizon, comparison = key.split("|")
        by_key.setdefault(asset, {}).setdefault(horizon, {})[comparison] = res

    flags = {}
    for asset, by_h in by_key.items():
        flags[asset] = {}
        for horizon, comps in by_h.items():
            w_vs_rw = comps.get("TSDiff-W vs RandomWalk")
            d_vs_rw = comps.get("TSDiff-D vs RandomWalk")
            w_vs_d = comps.get("TSDiff-W vs TSDiff-D")
            g1_w = bool(w_vs_rw and w_vs_rw["significant_at_05"] and w_vs_rw["mean_diff"] < 0)
            g1_d = bool(d_vs_rw and d_vs_rw["significant_at_05"] and d_vs_rw["mean_diff"] < 0)
            g2 = bool(w_vs_d and w_vs_d["significant_at_05"])
            flags[asset][horizon] = {
                "G1_W_beats_RW": g1_w, "G1_D_beats_RW": g1_d, "G1_any_beats_RW": g1_w or g1_d,
                "G2_W_vs_D_significant": g2,
            }
    return flags


def print_table_v2(summary: dict) -> None:
    print(f"\n{'Actif':<6}{'Horizon':<8}{'Modele':<12}{'RMSE':>10}{'Cov95':>8}{'Width':>10}{'CRPS':>10}")
    print("-" * 64)
    for asset, by_h in summary.items():
        for horizon in HORIZON_LABELS:
            for model in MODELS:
                m = by_h.get(horizon, {}).get(model)
                if not m:
                    continue
                print(f"{asset:<6}{horizon:<8}{model:<12}{m['RMSE']:>10.2f}{m['Cov95']:>8.2f}"
                      f"{m['AvgWidth']:>10.2f}{m['CRPS']:>10.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--sweep-file", default=str(Path(__file__).resolve().parent
                                              / "epoch_sweep_results.json"))
    p.add_argument("--epochs-w", type=int, default=None,
                   help="override the selected TSDiff-W epochs for ALL assets")
    p.add_argument("--epochs-d", type=int, default=None,
                   help="override the selected TSDiff-D epochs for ALL assets")
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=max(50, td.N_SAMPLES))
    p.add_argument("--k-denoise", type=int, default=td.K_DENOISE)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "weekly_headtohead_v2_results.json"))
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    selected = {}
    if args.epochs_w is None or args.epochs_d is None:
        selected = load_selected_epochs(args.sweep_file)

    all_records, epochs_used = [], {}
    t0 = time.time()
    for asset in args.assets:
        ew = args.epochs_w if args.epochs_w is not None else selected[f"{asset}|TSDiff-W"]["epochs"]
        ed = args.epochs_d if args.epochs_d is not None else selected[f"{asset}|TSDiff-D"]["epochs"]
        epochs_used[asset] = {"TSDiff-W": ew, "TSDiff-D": ed}
        records = run_pair_v2(asset, ASSETS[asset], ew, ed, args.n_val, args.n_test, args.seed,
                              args.n_samples, args.k_denoise, args.start, end)
        all_records.extend(records)
    elapsed = time.time() - t0

    summary = summarize_v2(all_records)
    paired_tests = run_paired_tests(all_records, n_boot=args.n_boot)
    flags = g1_g2_flags(paired_tests)

    payload = {
        "config": {
            "epochs_used": epochs_used, "n_val": args.n_val, "n_test": args.n_test,
            "seed": args.seed, "n_samples": args.n_samples, "k_denoise": args.k_denoise,
            "n_boot": args.n_boot, "assets": args.assets, "start": args.start, "end": end,
            "elapsed_s": round(elapsed, 1),
        },
        "summary": summary,
        "paired_tests": paired_tests,
        "flags": flags,
        "records": all_records,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min)")
    print_table_v2(summary)


if __name__ == "__main__":
    main()
