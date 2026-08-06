"""
oos_tsdiff_daily_weekly.py -- TSDiff regime B (daily -> weekly, W+1/2/3) AND
regime C (weekly-native, W+1/2/3), on the EXACT SAME (asset, cutoff_date,
target_date) triplets as the other 6 models' existing `oos` rows.

MIRROR of `oos_nsdiff_daily_weekly.py` (declared, not silently duplicated --
see BRIEF_dashboard_multiseed_200.md §4/§8): same train-once-forward
mechanism, same verbatim-origin reading (`load_baseline_triplets` /
`load_baseline_triplets_daily`, both IMPORTED from `oos_nsdiff_daily_weekly.py`
-- not redefined), same `fetch_verified` price-verification helper (also
imported unmodified -- it only downloads/verifies OHLC price data by ticker,
nothing NsDiff-specific despite living in that module). The only real
differences from the NsDiff mirror: `tsdiff_model` instead of `nsdiff_model`,
and TWO epoch budgets (TSDiff-W / TSDiff-D are swept independently per asset,
`experiments/epoch_sweep_results.json`'s `selected_epochs` -- reused via
`weekly_headtohead_v2.load_selected_epochs`, NOT re-swept here) instead of
NsDiff's single shared budget.

IMPORTANT -- written but NOT executed on this machine (BRIEF_dashboard_
multiseed_200.md, priorité corrigée : le compute lourd tourne chez le
tuteur). See RUNBOOK_regeneration_multiseed_200.md for how/when to run this.

Why a NEW script instead of replaying `weekly_headtohead_v2.py` +
`backfill_weekly_predictions.py` (how the CURRENT TSDiff oos rows were
actually produced, cf. that script's docstring): `run_pair_v2` re-derives its
30/90 test origins from a FRESH `three_way_split()` call on freshly
re-downloaded data -- reproducing the exact existing origins would require
freezing `end=` to the historical run's date (data-drift risk, brief §9 "ne
pas régénérer des origines à peu près"). Reading origins VERBATIM from the DB
(this script's approach, identical to NsDiff's) sidesteps that risk entirely
and is the approach the brief mandates.

Insertion: NOT done by this script directly (single-seed generation only, cf.
`generate_tsdiff_asset`) -- the ensemble writer
(`oos_ensemble_tsdiff_daily_weekly.py`) and the CV/badge script
(`tsdiff_daily_weekly_multiseed.py`) both reuse `generate_tsdiff_asset` from
here, exactly mirroring how `oos_nsdiff_daily_weekly.generate_nsdiff_asset`
is reused by `nsdiff_daily_weekly_multiseed.py` /
`oos_ensemble_nsdiff_daily_weekly.py`.

Usage (once the tutor runs it):
    python oos_tsdiff_daily_weekly.py --smoke                  # 1 asset, tiny epochs/samples -- plumbing only
    python oos_tsdiff_daily_weekly.py --dry-run                # compute + verify + print, no DB writes
    python oos_tsdiff_daily_weekly.py                          # full run, all 5 assets, DB write
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tsdiff_model as td                                              # noqa: E402
from weekly_headtohead import (                                        # noqa: E402
    ASSETS as ASSET_TICKERS, HORIZON_WEEKLY, HORIZON_DAILY, build_weekly,
    standardized_returns,
)
from epoch_sweep import week_targets                                    # noqa: E402
from backtest_rolling_tsdiffw import (                                  # noqa: E402
    DB_PATH, HORIZON_UNITS, FETCH_START, FETCH_END, load_baseline_triplets,
    _weekly_position,
)
from oos_nsdiff_daily_weekly import (                                   # noqa: E402
    load_baseline_triplets_daily, fetch_verified,
)
from weekly_headtohead_v2 import load_selected_epochs                   # noqa: E402
from validation import sim_trades as st                                 # noqa: E402

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 50             # matches the historical single-seed oos config (n90 run)
DEFAULT_K_DENOISE = td.K_DENOISE
DEFAULT_SWEEP_FILE = ROOT / "experiments" / "epoch_sweep_results.json"
RUN_ID = "oos_tsdiff_daily_weekly"  # (re)generation run id -- single-seed path


# ── NsDiff-mirror: TSDiff generation (train-once-forward) ───────────────────

def generate_tsdiff_asset(asset: str, daily: pd.Series, weekly: pd.Series, weekly_dates: pd.Series,
                          origins_c: pd.DataFrame, origins_b: pd.DataFrame, epochs_w: int, epochs_d: int,
                          seed: int, n_samples: int, k_denoise: int, collect_samples: bool = False) -> tuple:
    """Exact mirror of `oos_nsdiff_daily_weekly.generate_nsdiff_asset` (see that
    function's docstring for the full rationale) -- `nm.*` -> `td.*`, single
    `epochs` -> `epochs_w`/`epochs_d` (TSDiff-W and TSDiff-D are swept
    independently per asset, unlike NsDiff's single shared budget).
    `collect_samples=True` stashes the raw sample cloud under row["_samples"],
    additive, same convention as the NsDiff mirror (needed for the ensemble)."""
    sym_diff = set(origins_c["cutoff_date"]) ^ set(origins_b["cutoff_date"])
    if sym_diff:
        raise SystemExit(f"[{asset}] regime B/C cutoff sets differ ({len(sym_diff)} dates not shared) -- "
                         "cannot reuse a single train-once split for both regimes.")

    cutoffs = sorted(origins_c["cutoff_date"].unique())
    pos_of = {c: _weekly_position(weekly_dates, c) for c in cutoffs}
    missing = [c for c, p in pos_of.items() if p is None]
    if missing:
        raise SystemExit(f"[{asset}] {len(missing)} cutoff date(s) not found on the "
                         f"W-FRI weekly grid: {missing[:5]}...")
    cutoffs = sorted(cutoffs, key=lambda c: pos_of[c])
    earliest_pos = pos_of[cutoffs[0]]
    T0_date = weekly_dates.iloc[earliest_pos]
    T0_daily_pos = int(daily.index.get_loc(T0_date))

    train_weekly = weekly.iloc[:earliest_pos]      # strictly before the earliest reused origin
    train_daily = daily.iloc[:T0_daily_pos]         # strictly before the earliest reused origin
    print(f"[{asset}] train-once: weekly {len(train_weekly)} obs (< {T0_date.date()}), "
         f"daily {len(train_daily)} obs (< {T0_date.date()}) | "
         f"{len(cutoffs)} origins {cutoffs[0]} -> {cutoffs[-1]}")

    t0 = time.time()
    td.set_seed(seed)
    model_w, mu_w, sd_w = td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY, epochs=epochs_w)
    print(f"[{asset}] TSDiff-W (regime C) fitted in {time.time() - t0:.0f}s (mu={mu_w:.6f}, sd={sd_w:.6f})")

    t0 = time.time()
    td.set_seed(seed)
    model_d, mu_d, sd_d = td.fit_tsdiff(train_daily, horizon=HORIZON_DAILY, epochs=epochs_d)
    print(f"[{asset}] TSDiff-D (regime B) fitted in {time.time() - t0:.0f}s (mu={mu_d:.6f}, sd={sd_d:.6f})")

    weekly_z = standardized_returns(weekly, mu_w, sd_w)
    daily_z = standardized_returns(daily, mu_d, sd_d)

    origins_c_by_cutoff = dict(tuple(origins_c.groupby("cutoff_date")))
    origins_b_by_cutoff = dict(tuple(origins_b.groupby("cutoff_date")))

    rows_c, rows_b = [], []
    for k, cutoff_date in enumerate(cutoffs):
        m = pos_of[cutoff_date]
        origin_date, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
        last_price = float(weekly.iloc[m])

        grp_c = origins_c_by_cutoff[cutoff_date]
        grp_b = origins_b_by_cutoff[cutoff_date]
        needed_h = sorted(int(h) for h in set(grp_c["horizon"]) | set(grp_b["horizon"]))
        h_d_needed = [daily_horizons[h - 1] for h in needed_h]

        td.set_seed(seed + k)
        samples_w = td.forecast_from_fitted(model_w, weekly_z[:m], mu_w, sd_w, last_price,
                                            horizons=needed_h, n_samples=n_samples, k_denoise=k_denoise)
        td.set_seed(seed + k)
        samples_d = td.forecast_from_fitted(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price,
                                            horizons=h_d_needed, n_samples=n_samples, k_denoise=k_denoise)

        for h in needed_h:
            wi = h - 1
            target_date = target_dates[wi]
            h_d = daily_horizons[wi]

            row_c = grp_c[grp_c["horizon"] == h]
            if len(row_c):
                stored_target = row_c["target_date"].iloc[0]
                if str(target_date.date()) != stored_target:
                    raise SystemExit(f"[{asset}] regime C target_date mismatch at cutoff={cutoff_date} "
                                    f"h={h}: computed={target_date.date()} vs baseline={stored_target}")
                s = samples_w[h]
                point = float(np.mean(s))
                lo, hi = (float(q) for q in np.quantile(s, [0.025, 0.975]))
                row_out = {
                    "run_id": RUN_ID, "model": "TSDiff", "asset": asset, "horizon": h,
                    "regime": "unknown", "cutoff_date": cutoff_date, "target_date": stored_target,
                    "last_close": last_price, "y_pred": point, "y_lower": lo, "y_upper": hi,
                    "y_true": float(row_c["y_true"].iloc[0]), "source": "oos",
                    "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": HORIZON_UNITS[h],
                }
                if collect_samples:
                    row_out["_samples"] = [float(x) for x in np.asarray(s).ravel()]
                rows_c.append(row_out)

            row_b = grp_b[grp_b["horizon"] == h]
            if len(row_b):
                stored_target = row_b["target_date"].iloc[0]
                if str(target_date.date()) != stored_target:
                    raise SystemExit(f"[{asset}] regime B target_date mismatch at cutoff={cutoff_date} "
                                    f"h={h}: computed={target_date.date()} vs baseline={stored_target}")
                s = samples_d[h_d]
                point = float(np.mean(s))
                lo, hi = (float(q) for q in np.quantile(s, [0.025, 0.975]))
                row_out = {
                    "run_id": RUN_ID, "model": "TSDiff", "asset": asset, "horizon": h,
                    "regime": "unknown", "cutoff_date": cutoff_date, "target_date": stored_target,
                    "last_close": last_price, "y_pred": point, "y_lower": lo, "y_upper": hi,
                    "y_true": float(row_b["y_true"].iloc[0]), "source": "oos",
                    "frequence": "daily", "horizon_type": "weekly", "horizon_unit": HORIZON_UNITS[h],
                }
                if collect_samples:
                    row_out["_samples"] = [float(x) for x in np.asarray(s).ravel()]
                rows_b.append(row_out)
        if (k + 1) % 15 == 0 or k == len(cutoffs) - 1:
            print(f"[{asset}] origin {k + 1}/{len(cutoffs)} ({cutoff_date}) done")

    return rows_c, rows_b


# ── main (single-seed, historical-config regen -- for the ensemble/CV loop, ──
# ── see tsdiff_daily_weekly_multiseed.py / oos_ensemble_tsdiff_daily_weekly.py) ──

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()),
                   help="tickers, e.g. SPY BTC-USD (default: all 5)")
    p.add_argument("--sweep-file", default=str(DEFAULT_SWEEP_FILE))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true", help="compute + verify + print, no DB writes")
    p.add_argument("--smoke", action="store_true",
                   help="tiny plumbing check: 1 asset (SPY), epochs=2, n-samples=6 -- overrides other flags")
    p.add_argument("--out", default=str(ROOT / "experiments" / "oos_tsdiff_daily_weekly_summary.json"))
    args = p.parse_args()

    assets = ["SPY"] if args.smoke else args.assets
    n_samples = 6 if args.smoke else args.n_samples

    selected = load_selected_epochs(args.sweep_file)

    t_start = time.time()
    print("=== 1. baseline origins (reused verbatim, regime C via load_baseline_triplets, "
         "regime B via load_baseline_triplets_daily) ===")
    origins_c_all = load_baseline_triplets(assets, db_path=args.db_path)
    origins_b_all = load_baseline_triplets_daily(assets, db_path=args.db_path)
    print(f"regime C: {len(origins_c_all)} triplets | regime B: {len(origins_b_all)} triplets")

    all_rows_c, all_rows_b, skipped, per_asset_meta = [], [], [], {}
    for asset in assets:
        origins_c = origins_c_all[origins_c_all["asset"] == asset].reset_index(drop=True)
        origins_b = origins_b_all[origins_b_all["asset"] == asset].reset_index(drop=True)
        if origins_c.empty or origins_b.empty:
            print(f"[{asset}] no baseline origins on one side (C={len(origins_c)}, B={len(origins_b)}) -- skipping")
            skipped.append({"asset": asset, "reason": "no baseline origins on one side"})
            continue

        print(f"\n=== [{asset}] price fetch + baseline verification ===")
        fetched = fetch_verified(asset, origins_c, origins_b)
        if fetched is None:
            print(f"[{asset}] SKIPPED: neither live yfinance nor the offline cache reproduces the "
                 f"baselines' stored last_close at every reused origin (yfinance drift).")
            skipped.append({"asset": asset, "reason": "price mismatch on both live and offline sources"})
            continue
        daily, weekly, weekly_dates, price_source = fetched

        ew = 2 if args.smoke else selected[f"{asset}|TSDiff-W"]["epochs"]
        ed = 2 if args.smoke else selected[f"{asset}|TSDiff-D"]["epochs"]

        print(f"=== [{asset}] TSDiff generation (train-once-forward, epochs W={ew}/D={ed}) ===")
        try:
            rows_c, rows_b = generate_tsdiff_asset(asset, daily, weekly, weekly_dates, origins_c, origins_b,
                                                   ew, ed, args.seed, n_samples, args.k_denoise)
        except Exception as exc:
            print(f"[{asset}] SKIPPED: {type(exc).__name__}: {exc}")
            skipped.append({"asset": asset, "reason": f"{type(exc).__name__}: {exc}", "price_source": price_source})
            continue
        all_rows_c += rows_c
        all_rows_b += rows_b
        per_asset_meta[asset] = {"price_source": price_source, "epochs_w": ew, "epochs_d": ed,
                                 "n_rows_c": len(rows_c), "n_rows_b": len(rows_b)}

    all_rows = all_rows_c + all_rows_b
    print(f"\n{len(all_rows_c)} regime-C rows + {len(all_rows_b)} regime-B rows = {len(all_rows)} total")
    if skipped:
        print(f"Skipped assets: {skipped}")

    if args.dry_run or args.smoke:
        print("\n--dry-run/--smoke: skipping DB write.")
        n_written = 0
    else:
        n_written = st.insert_oos_predictions(all_rows, db_path=args.db_path)
        print(f"\nInserted/updated {n_written} rows into {args.db_path} (source='oos', model='TSDiff', run_id={RUN_ID}).")

    summary = {
        "config": {
            "assets": assets, "seed": args.seed, "n_samples": n_samples,
            "k_denoise": args.k_denoise, "fetch_start": FETCH_START, "fetch_end": FETCH_END,
            "mechanism": "train-once-forward (mirrors oos_nsdiff_daily_weekly.generate_nsdiff_asset)",
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "per_asset": per_asset_meta,
        "skipped": skipped,
        "n_rows_regime_c": len(all_rows_c), "n_rows_regime_b": len(all_rows_b),
        "n_written": n_written,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary -> {args.out}")
    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
