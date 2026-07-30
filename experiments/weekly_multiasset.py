"""
weekly_multiasset.py — extend the TSDiff-W-vs-RandomWalk head-to-head to a
5-asset panel (BTC-USD, ETH-USD, SPY, ZN=F, TLT — cf. calibration/regime/assets.py)
for statistical power. TSDiff-D is OUT OF SCOPE here (already established as
structurally under-calibrated in weekly_headtohead_v2_results.json) — this run
only fits TSDiff-W + the RandomWalk baseline, per asset.

Two phases, run separately (this script is meant to be gated: show the sweep
before launching the expensive final run):

  --phase sweep   Epoch-sweep {10,20,30,40,60,80,100,120} on the NEW assets only
                  (ETH, ZN, TLT) — SPY (epochs*=80) and BTC (epochs*=30) already
                  have a validated epoch* in epoch_sweep_results.json and are NOT
                  re-swept. Selection rule identical to epoch_sweep.py: argmin
                  CRPS on the validation block, never the test block. Merges into
                  epoch_sweep_results.json.

  --phase final   Head-to-head on the 30 test origins for all 5 assets (TSDiff-W
                  + RandomWalk only), using each asset's epochs* from
                  epoch_sweep_results.json. Adds a panel-level test: per-asset
                  paired bootstrap (TSDiff-W vs RandomWalk CRPS), then a Stouffer
                  combination across 4 INDEPENDENT buckets (SPY, BTC, ETH, and a
                  "Bonds" bucket = ZN=F+TLT averaged together first, since both
                  are the same underlying US Treasury exposure and must not be
                  double-counted as two independent observations). Saves to
                  experiments/weekly_multiasset_results.json.

Usage:
    python weekly_multiasset.py --phase sweep
    python weekly_multiasset.py --phase final
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
import tsdiff_model as td                                       # noqa: E402

from weekly_headtohead import ASSETS, HORIZON_LABELS             # noqa: E402
from epoch_sweep import sweep_asset, select_epochs, DEFAULT_N_VAL, DEFAULT_N_TEST, DEFAULT_SEED  # noqa: E402
from weekly_headtohead_v2 import (                                # noqa: E402
    run_pair_v2, summarize_v2, run_paired_tests, print_table_v2,
)
from paired_test import paired_bootstrap_test                    # noqa: E402
from cross_asset_test import merge_correlated_diffs, stouffer_combine  # noqa: E402

NEW_ASSETS = ["ETH", "ZN", "TLT"]
ALL_ASSETS = ["SPY", "BTC", "ETH", "ZN", "TLT"]
EPOCH_CANDIDATES = (10, 20, 30, 40, 60, 80, 100, 120)
BOND_ASSETS = ("ZN", "TLT")   # same underlying exposure -> merged into one bucket, never counted twice
SWEEP_FILE = Path(__file__).resolve().parent / "epoch_sweep_results.json"


def run_sweep_phase(assets, candidates, n_val, n_test, seed, n_samples, k_denoise,
                    start, end, sweep_file: Path):
    all_records = []
    meta = {}
    t0 = time.time()
    for asset in assets:
        records, asset_meta = sweep_asset(asset, ASSETS[asset], candidates, n_val, n_test,
                                          seed, n_samples, k_denoise, start, end,
                                          models=("TSDiff-W",))
        all_records.extend(records)
        meta[asset] = asset_meta
    elapsed = time.time() - t0

    base = json.loads(sweep_file.read_text()) if sweep_file.exists() else \
        {"config": {}, "meta": {}, "records": [], "selected_epochs": {}}
    merged_records = base["records"] + all_records
    selected = select_epochs(merged_records)
    base["records"] = merged_records
    base["selected_epochs"] = selected
    base["meta"].update(meta)
    sweep_file.write_text(json.dumps(base, indent=2))

    return all_records, selected, elapsed


def run_final_phase(sweep_file: Path, n_val, n_test, seed, n_samples, k_denoise,
                    start, end, out_path: Path):
    selected = json.loads(sweep_file.read_text())["selected_epochs"]
    for asset in ALL_ASSETS:
        key = f"{asset}|TSDiff-W"
        if key not in selected:
            raise SystemExit(f"No selected epoch* for {key} in {sweep_file} -- run --phase sweep first.")

    all_records = []
    epochs_used = {}
    t0 = time.time()
    for asset in ALL_ASSETS:
        ew = selected[f"{asset}|TSDiff-W"]["epochs"]
        epochs_used[asset] = ew
        records = run_pair_v2(asset, ASSETS[asset], epochs_w=ew, epochs_d=None,
                              n_val=n_val, n_test=n_test, seed=seed, n_samples=n_samples,
                              k_denoise=k_denoise, start=start, end=end, include_d=False)
        all_records.extend(records)
    elapsed = time.time() - t0

    summary = summarize_v2(all_records)
    paired_tests = run_paired_tests(all_records, n_boot=10000, seed=0)

    # ── per-asset bucket: pool all 3 horizons x 30 origins into ONE paired test ──
    # Indexed by (origin_date, horizon) -- REAL calendar dates, not positional
    # origin numbers (0..n_test-1 would align by position even if the underlying
    # walk-forward origins fell on different actual dates for two assets).
    df = pd.DataFrame(all_records)
    per_asset_diffs = {}
    for asset in ALL_ASSETS:
        g = df[df.asset == asset]
        piv = g.pivot_table(index=["origin_date", "horizon"], columns="model", values="crps")
        per_asset_diffs[asset] = (piv["TSDiff-W"] - piv["RandomWalk"]).sort_index()

    # guardrail: ZN and TLT must share the exact same (origin_date, horizon)
    # index before being averaged into one "Bonds" bucket -- otherwise the merge
    # would silently misalign two different origin/target dates.
    zn_idx, tlt_idx = per_asset_diffs["ZN"].index, per_asset_diffs["TLT"].index
    if not zn_idx.equals(tlt_idx):
        raise SystemExit("ZN and TLT origin/horizon index mismatch -- cannot merge into one Bonds bucket.")
    bonds_diff = merge_correlated_diffs(per_asset_diffs["ZN"].values, per_asset_diffs["TLT"].values)

    bucket_diffs = {
        "SPY": per_asset_diffs["SPY"].values,
        "BTC": per_asset_diffs["BTC"].values,
        "ETH": per_asset_diffs["ETH"].values,
        "Bonds (ZN+TLT)": bonds_diff,
    }
    bucket_tests = {label: paired_bootstrap_test(diffs, n_boot=10000, seed=0)
                    for label, diffs in bucket_diffs.items()}
    panel_test = stouffer_combine(bucket_tests)

    payload = {
        "config": {
            "epochs_used": epochs_used, "n_val": n_val, "n_test": n_test, "seed": seed,
            "n_samples": n_samples, "k_denoise": k_denoise, "assets": ALL_ASSETS,
            "start": start, "end": end, "elapsed_s": round(elapsed, 1),
            "note": "TSDiff-D out of scope for this run (already established as structurally "
                   "under-calibrated) -- TSDiff-W + RandomWalk only, 5 assets.",
        },
        "summary": summary,
        "paired_tests": paired_tests,
        "bucket_tests": bucket_tests,
        "panel_test": panel_test,
        "records": all_records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}  ({elapsed / 60:.1f} min)")
    print_table_v2(summary)
    print(f"\n{'Bucket':<18}{'mean_diff':>12}{'p_value':>10}")
    print("-" * 40)
    for label, res in bucket_tests.items():
        print(f"{label:<18}{res['mean_diff']:>12.3f}{res['p_value']:>10.4f}")
    print(f"\nPanel (Stouffer, k={panel_test['k']}): z={panel_test['z_combined']:.3f}  "
          f"p={panel_test['p_combined']:.4f}  favors_W={panel_test['favors_first_term']}  "
          f"significant={panel_test['significant_at_05']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["sweep", "final"], default="sweep")
    p.add_argument("--assets", nargs="+", default=NEW_ASSETS, choices=list(ASSETS),
                   help="[sweep phase only] which assets to sweep")
    p.add_argument("--candidates", nargs="+", type=int, default=list(EPOCH_CANDIDATES))
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=max(50, td.N_SAMPLES))
    p.add_argument("--k-denoise", type=int, default=td.K_DENOISE)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--sweep-file", default=str(SWEEP_FILE))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "weekly_multiasset_results.json"))
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    sweep_file = Path(args.sweep_file)

    if args.phase == "sweep":
        records, selected, elapsed = run_sweep_phase(
            args.assets, args.candidates, args.n_val, args.n_test, args.seed,
            args.n_samples, args.k_denoise, args.start, end, sweep_file)
        print(f"\nSwept {args.assets} in {elapsed / 60:.1f} min -> merged into {sweep_file}")
        print(f"\n{'Actif':<6}{'Epochs':>8}{'CRPS_val':>12}{'Cov95_val':>11}{'rel_std%':>10}")
        print("-" * 47)
        df = pd.DataFrame(records)
        for asset in args.assets:
            sub = df[df.asset == asset].sort_values("epochs")
            best_ep = selected[f"{asset}|TSDiff-W"]["epochs"]
            for _, r in sub.iterrows():
                mark = " *" if r["epochs"] == best_ep else ""
                print(f"{asset:<6}{r['epochs']:>8}{r['crps_val']:>12.4f}"
                      f"{r['cov95_val']:>11.2f}{r['rel_std_pct_val']:>10.3f}{mark}")
    else:
        run_final_phase(sweep_file, args.n_val, args.n_test, args.seed, args.n_samples,
                        args.k_denoise, args.start, end, Path(args.out))


if __name__ == "__main__":
    main()
