"""
backtest_rolling_nsdiffw.py — rolling-origin backtest of NsDiff-W (weekly-
native, prod definition: nsdiff_model.fit_nsdiff fed weekly data, seq_len=30,
mirrors weekly_nsdiff_production.py) against the 5 baselines, on STRICTLY
IDENTICAL (asset, cutoff, horizon) triplets. Mirror of
backtest_rolling_tsdiffw.py, per BRIEF_nsdiff_weekly_parite_et_compa.md §4.2.

Origins: reused VERBATIM from backtest_rolling_tsdiffw.load_baseline_triplets
-- the 5 baselines' already-existing native-weekly walk-forward backtest in
tracking.db (frequence='weekly', horizon_type='weekly', daily_duplicate=0,
source='oos'). This buys exact triplet alignment "for free" -- no baseline
recomputation, and this backtest's rows land on the EXACT SAME origins as
backtest_rolling_tsdiffw.py's own TSDiff-W rows, so the two are directly
superposable (same (asset, cutoff, horizon) grid for both diffusion models).

Anti-leakage for NsDiff-W: EXPANDING-WINDOW PERIODIC REFIT every REFIT_EVERY
origins (default 15, same as TSDiff-W), identical mechanism to
backtest_rolling_tsdiffw.generate_tsdiffw_asset -- each refit trains
fit_nsdiff on weekly data <= that block's first reused cutoff, then every
origin in the block is forecast from that ONE frozen model via
forecast_from_fitted(..., hist_window=wz[:m]).

Epoch-selection leakage guard: DOES NOT APPLY here, unlike TSDiff-W. TSDiff-W's
epoch_sweep_val_dates exclusion exists because its epochs* were CHOSEN by
argmin-CRPS on a validation block (leakage risk: reusing those dates as test
origins here would let a hyperparameter-selection signal leak into this
backtest). NsDiff-W uses a FIXED, DECLARED epoch budget
(weekly_nsdiff_production.NSDIFF_EPOCHS_W=40, never selected on any
validation/test split -- brief §4.1 option (a)) -- there is no
hyperparameter-selection step to guard against, so ALL of the baselines'
surviving triplets are used as origins here, none excluded (documented here,
not silently done: if NsDiff-W ever switches to a validation-swept epoch
budget -- brief §5 dette déclarée -- this guard would need to be added back,
mirroring TSDiff-W's).

Asymmetry documented, not hidden (identical caveat to backtest_rolling_
tsdiffw.py, reproduced verbatim): NsDiff-W samples are GENUINE diffusion draws
(model.sample_paths); the 5 baselines' N=500 clouds are GAUSSIAN, drawn
parametrically from their already-stored 95% PI (prob_kpi_common.
sample_parametric) -- matches PRODUCTION's own weekly protocol for those 5
models, not a shortcut invented for this backtest.

Storage: NsDiff-W backtest rows are inserted with source='backtest_rolling_
nsdiff' (own value, own partial unique index, mirroring TSDiff-W's
'backtest_rolling') -- 100% isolated from 'live'/'oos'/'backtest_rolling' rows
(TSDiff/baselines), idempotent reruns. Baseline rows are NEVER written here --
only read (they already exist).

Metrics / tests: identical primitives to backtest_rolling_tsdiffw.py --
prob_kpi_common.row_kpis (via compute_prob_kpi_pilot.compute_row_kpis/
add_mase), build_kpi_probabilistes.aggregate_per_cell, paired_test.
paired_block_bootstrap_test, pooled_analysis's class/pooled/dual-test/Holm --
zero reimplementation.

Usage:
    python backtest_rolling_nsdiffw.py --smoke                  # 1 asset, few origins, tiny samples -- plumbing only
    python backtest_rolling_nsdiffw.py                          # full run, all 5 assets
    python backtest_rolling_nsdiffw.py --assets SPY BTC-USD      # subset
    python backtest_rolling_nsdiffw.py --dry-run                # compute + print, no DB writes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import nsdiff_model as nm                                               # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly     # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W                    # noqa: E402
from backtest_rolling_tsdiffw import (                                  # noqa: E402
    DB_PATH, BASELINE_MODELS, HORIZON_UNITS, FETCH_START, FETCH_END,
    load_baseline_triplets, generate_baselines, ensure_backtest_rolling_index,
    _weekly_position,
)
from prob_kpi_common import PARAMETRIC_MODELS, HORIZON_LABEL_ORDER       # noqa: E402
from compute_prob_kpi_pilot import compute_row_kpis, add_mase            # noqa: E402
from build_kpi_probabilistes import ASSET_CLASS, MODEL_ORDER             # noqa: E402
from honest_eval.metrics import mase as mase_fn                          # noqa: E402
from paired_test import paired_block_bootstrap_test                      # noqa: E402
from pooled_analysis import (                                            # noqa: E402
    compute_asset_scales, class_series, pooled_diff_series, dual_test, holm_correction,
)
from validation import tracking_db as tdb                                # noqa: E402

DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 42
DEFAULT_REFIT_EVERY = 15
RUN_ID = "backtest-rolling-nsdiffw-v1"


# ── 1. NsDiff-W generation (periodic expanding refit, no epoch-leakage guard) ─

def generate_nsdiffw_asset(asset: str, ticker: str, epochs: int, origin_rows: pd.DataFrame,
                           refit_every: int, n_samples: int, seed: int) -> list:
    """Periodic expanding-window refit NsDiff-W over ALL of this asset's
    baseline-shared origins (no epoch-sweep exclusion -- see module docstring).
    Mirrors backtest_rolling_tsdiffw.generate_tsdiffw_asset exactly, with
    nm.fit_nsdiff/nm.forecast_from_fitted swapped in for td.fit_tsdiff/
    td.forecast_from_fitted."""
    print(f"[{asset}] downloading {ticker} ({FETCH_START} -> {FETCH_END}) ...")
    daily = nm.fetch_data(ticker, FETCH_START, FETCH_END)
    weekly, weekly_dates = build_weekly(daily)

    cutoffs = sorted(origin_rows["cutoff_date"].unique())
    pos_of = {c: _weekly_position(weekly_dates, c) for c in cutoffs}
    missing = [c for c, p in pos_of.items() if p is None]
    if missing:
        raise SystemExit(f"[{asset}] {len(missing)} cutoff date(s) not found on the "
                         f"W-FRI weekly grid: {missing[:5]}...")
    cutoffs = sorted(cutoffs, key=lambda c: pos_of[c])

    blocks = [cutoffs[i:i + refit_every] for i in range(0, len(cutoffs), refit_every)]
    records = []
    for b_idx, block in enumerate(blocks):
        train_end_pos = pos_of[block[0]]
        train_weekly = weekly.iloc[:train_end_pos + 1]
        t0 = time.time()
        nm.set_seed(seed + b_idx)
        model, mu, sd = nm.fit_nsdiff(train_weekly, horizon=3, epochs=epochs)
        fit_s = time.time() - t0
        print(f"[{asset}] block {b_idx + 1}/{len(blocks)}: fit <= {block[0]} "
              f"({len(train_weekly)} weekly obs) in {fit_s:.0f}s, {len(block)} origins")

        wr = nm._log_returns(weekly.values.astype(float))
        wz = (wr - mu) / sd

        for k, cutoff_date in enumerate(block):
            m = pos_of[cutoff_date]
            if m == 0:
                continue
            last_price = float(weekly.iloc[m])
            grp = origin_rows[(origin_rows["asset"] == asset) & (origin_rows["cutoff_date"] == cutoff_date)]
            needed_h = sorted(int(h) for h in grp["horizon"].unique())
            nm.set_seed(seed + b_idx * 1000 + k)
            samples_by_h = nm.forecast_from_fitted(
                model, wz[:m], mu, sd, last_price, horizons=needed_h, n_samples=n_samples)
            for _, row in grp.iterrows():
                h = int(row["horizon"])
                actual_target = str(weekly_dates.iloc[m + h].date())
                if actual_target != row["target_date"]:
                    raise SystemExit(f"[{asset}] target_date mismatch at cutoff={cutoff_date} "
                                    f"h={h}: NsDiff grid={actual_target} vs baseline={row['target_date']}")
                if abs(last_price - row["last_close"]) > 1e-6 * max(1.0, abs(row["last_close"])):
                    raise SystemExit(f"[{asset}] last_close mismatch at cutoff={cutoff_date}: "
                                    f"NsDiff={last_price} vs baseline={row['last_close']}")
                records.append({
                    "model": "NsDiff", "asset": asset, "horizon": h,
                    "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": HORIZON_UNITS[h],
                    "cutoff_date": cutoff_date, "target_date": row["target_date"],
                    "last_close": last_price, "y_true": row["y_true"],
                    "method": "native_nsdiff_regimeC_rolling_backtest",
                    "refit_block": b_idx, "samples": samples_by_h[h],
                })
    return records


# ── 2. DB storage (source='backtest_rolling_nsdiff' by default, isolated) ────

def save_backtest_rolling_rows_nsdiff(nsdiff_index: pd.DataFrame, nsdiff_samples: np.ndarray,
                                      source_tag: str, run_id: str = RUN_ID,
                                      db_path: str = DB_PATH) -> int:
    """Mirror of backtest_rolling_tsdiffw.save_backtest_rolling_rows with
    model='NsDiff' hardcoded instead of 'TSDiff' -- own partial unique index
    per source_tag (ensure_backtest_rolling_index, reused as-is), so reruns
    are idempotent and never collide with TSDiff-W's own 'backtest_rolling'
    rows or with any other tag."""
    ensure_backtest_rolling_index(source_tag, db_path)
    import sqlite3
    con = sqlite3.connect(db_path)
    n = 0
    try:
        for i, row in nsdiff_index.iterrows():
            s = nsdiff_samples[i]
            y_pred = float(np.mean(s))
            y_lower, y_upper = (float(q) for q in np.quantile(s, [0.025, 0.975]))
            rec = {
                "run_id": run_id, "model": "NsDiff", "asset": row["asset"],
                "horizon": int(row["horizon"]), "regime": "unknown",
                "cutoff_date": row["cutoff_date"], "target_date": row["target_date"],
                "last_close": float(row["last_close"]), "y_pred": y_pred,
                "y_lower": y_lower, "y_upper": y_upper, "y_true": float(row["y_true"]),
                "source": source_tag, "frequence": "weekly", "horizon_type": "weekly",
                "horizon_unit": row["horizon_unit"],
                "real_flag": tdb.compute_real_flag("NsDiff", row["cutoff_date"]),
            }
            cols = list(rec.keys())
            placeholders = ", ".join(f":{c}" for c in cols)
            columns = ", ".join(cols)
            con.execute(f"""
                INSERT INTO predictions ({columns}) VALUES ({placeholders})
                ON CONFLICT (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
                WHERE source='{source_tag}'
                DO UPDATE SET
                    run_id=excluded.run_id, target_date=excluded.target_date,
                    last_close=excluded.last_close, y_pred=excluded.y_pred,
                    y_lower=excluded.y_lower, y_upper=excluded.y_upper,
                    y_true=excluded.y_true, real_flag=excluded.real_flag
            """, rec)
            n += 1
        con.commit()
    finally:
        con.close()
    return n


# ── 3. KPI matrix + paired tests (identical shape to backtest_rolling_tsdiffw) ─

def aggregate_per_cell_nsdiff(df: pd.DataFrame) -> pd.DataFrame:
    """Local mirror of build_kpi_probabilistes.aggregate_per_cell -- NOT a
    reimplementation for its own sake, but a required fix: that shared
    function casts the per-cell `model` column to
    `pd.Categorical(..., categories=MODEL_ORDER)` where MODEL_ORDER =
    ["ARIMA-GARCH","SARIMA","Prophet","Naive","LSTM","TSDiff"] (verified
    directly in build_kpi_probabilistes.py) -- any model name outside that
    list, e.g. "NsDiff", silently becomes NaN/null in the per-cell output
    (the per-ROW KPIs from compute_row_kpis are unaffected, only this
    aggregate's label). Not modifying that shared file (other callers, e.g.
    build_kpi_probabilistes.py's own CLI, still rely on the 6-model list) --
    this local copy is identical except MODEL_ORDER + ["NsDiff"]."""
    rows = []
    group_cols = ["model", "asset", "frequence", "horizon_type", "horizon_unit"]
    for keys, grp in df.groupby(group_cols):
        model, asset, frequence, horizon_type, horizon_unit = keys
        n = len(grp)
        entry = {
            "model": model, "asset": asset, "asset_class": ASSET_CLASS.get(asset),
            "frequence": frequence, "horizon_type": horizon_type, "horizon_unit": horizon_unit,
            "n_origins": n,
            "crps_mean": float(grp["crps"].mean()),
            "coverage_50": float(grp["cov50"].mean()),
            "coverage_80": float(grp["cov80"].mean()),
            "coverage_95": float(grp["cov95"].mean()),
            "sharpness_50": float(grp["sharp50"].mean()),
            "sharpness_80": float(grp["sharp80"].mean()),
            "sharpness_95": float(grp["sharp95"].mean()),
            "winkler_50": float(grp["winkler50"].mean()),
            "winkler_80": float(grp["winkler80"].mean()),
            "winkler_95": float(grp["winkler95"].mean()),
            "pit_mean": float(grp["pit"].mean()),
            "pit_std": float(grp["pit"].std()) if n > 1 else None,
        }
        matched = grp[grp["_has_naive"]] if "_has_naive" in grp else grp.iloc[0:0]
        entry["mase_n_matched"] = int(len(matched))
        if model == "Naive":
            entry["mase"] = 1.0
        elif len(matched) > 0:
            entry["mase"] = float(mase_fn(matched["y_true"], matched["sample_mean"], matched["naive_pred"]))
        else:
            entry["mase"] = None
        rows.append(entry)
    out = pd.DataFrame(rows)
    out["horizon_unit"] = pd.Categorical(out["horizon_unit"], categories=HORIZON_LABEL_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=list(MODEL_ORDER) + ["NsDiff"], ordered=True)
    return out.sort_values(["asset", "horizon_unit", "model"]).reset_index(drop=True)


def build_kpi_matrix(nsdiff_index: pd.DataFrame, nsdiff_samples: np.ndarray,
                     baseline_index: pd.DataFrame, baseline_samples: np.ndarray) -> pd.DataFrame:
    full_index = pd.concat([
        nsdiff_index.drop(columns=["samples"], errors="ignore").reset_index(drop=True),
        baseline_index.drop(columns=["y_pred", "y_lower", "y_upper"], errors="ignore").reset_index(drop=True),
    ], ignore_index=True)
    full_samples = np.concatenate([nsdiff_samples, baseline_samples], axis=0)
    df = compute_row_kpis(full_index, full_samples)
    df = add_mase(df)
    return df


def run_paired_tests(df: pd.DataFrame, assets: list, start: str, end_for_scales: str) -> dict:
    scales_by_ticker = compute_asset_scales(start, end_for_scales)
    df = df.copy()
    df["asset_class"] = df["asset"].map(ASSET_CLASS)

    out = {"per_asset": {}, "pooled": {}}
    for horizon_unit in ("W+1", "W+2", "W+3"):
        sub_h = df[df["horizon_unit"] == horizon_unit]
        t = sub_h[sub_h["model"] == "NsDiff"]
        pvals_this_horizon, comparisons_this_horizon = [], []
        for baseline in BASELINE_MODELS:
            b = sub_h[sub_h["model"] == baseline]
            for asset in assets:
                ta = t[t["asset"] == asset].set_index("cutoff_date").sort_index()
                ba = b[b["asset"] == asset].set_index("cutoff_date").sort_index()
                joined = ta[["crps"]].join(ba[["crps"]], lsuffix="_nsdiff", rsuffix="_baseline", how="inner")
                if len(joined) < 4:
                    continue
                diffs = (joined["crps_nsdiff"] - joined["crps_baseline"]).values
                res = paired_block_bootstrap_test(diffs, block_length=3)
                out["per_asset"][f"{asset}|{horizon_unit}|NsDiff vs {baseline}"] = res

            t_norm = t.copy()
            b_norm = b.copy()
            t_norm["crps_norm"] = t_norm["crps"] / t_norm["asset"].map(scales_by_ticker)
            b_norm["crps_norm"] = b_norm["crps"] / b_norm["asset"].map(scales_by_ticker)
            diffs_pooled = pooled_diff_series(t_norm, b_norm, "crps_norm", date_col="cutoff_date")
            res_pooled = dual_test(diffs_pooled, h=int(horizon_unit[-1]))
            out["pooled"][f"{horizon_unit}|NsDiff vs {baseline}"] = res_pooled
            if res_pooled.get("status") == "tested":
                pvals_this_horizon.append(res_pooled["p_value_bootstrap"])
                comparisons_this_horizon.append(f"{horizon_unit}|NsDiff vs {baseline}")

        if pvals_this_horizon:
            adjusted = holm_correction(pvals_this_horizon)
            for key, p_adj in zip(comparisons_this_horizon, adjusted):
                out["pooled"][key]["p_value_bootstrap_holm"] = p_adj
                out["pooled"][key]["significant_after_holm"] = bool(p_adj < 0.05)
    return out


# ── 4. main ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()),
                   help="tickers, e.g. SPY BTC-USD (default: all 5)")
    p.add_argument("--refit-every", type=int, default=DEFAULT_REFIT_EVERY)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS_W,
                   help="fixed epoch budget (declared, mirrors weekly_nsdiff_production.py -- "
                       "no validation sweep for NsDiff-W today, cf. brief §4.1/§5)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dry-run", action="store_true", help="compute + print, no DB writes")
    p.add_argument("--smoke", action="store_true",
                   help="tiny plumbing check: 1 asset (SPY), refit-every=100 (1 block), "
                       "epochs=2, n-samples=6 -- overrides other flags")
    p.add_argument("--out-prefix", default=str(ROOT / "experiments" / "backtest_rolling_nsdiffw"))
    p.add_argument("--source-tag", default="backtest_rolling_nsdiff",
                   help="DB source value + own partial unique index -- isolates this run's "
                       "rows from 'backtest_rolling' (TSDiff-W), 'live', 'oos', and any other tag.")
    args = p.parse_args()

    if args.source_tag in ("live", "oos", "backtest_rolling"):
        raise SystemExit(f"--source-tag={args.source_tag!r} would collide with existing "
                        "TSDiff/baseline rows -- refusing (isolation non-negotiable).")

    assets = ["SPY"] if args.smoke else args.assets
    refit_every = 100 if args.smoke else args.refit_every
    n_samples = 6 if args.smoke else args.n_samples
    epochs = 2 if args.smoke else args.epochs

    t_start = time.time()

    print("=== 1. baseline triplets (reused verbatim from backtest_rolling_tsdiffw) ===")
    triplets = load_baseline_triplets(assets)
    print(f"{len(triplets)} raw (asset,cutoff,horizon) rows across {len(assets)} asset(s)")

    print("\n=== 2. epoch-selection leakage guard: N/A (fixed epoch budget, no sweep) ===")
    origins = triplets
    print(f"{len(origins)} triplets used as-is, {origins['cutoff_date'].nunique()} unique cutoffs total "
         f"(no exclusion -- NsDiff-W epochs are FIXED/declared, not validation-selected)")

    print("\n=== 3. NsDiff-W generation (periodic expanding refit) ===")
    nsdiff_records = []
    for asset in assets:
        origin_rows = origins[origins["asset"] == asset]
        nsdiff_records += generate_nsdiffw_asset(
            asset, asset, epochs, origin_rows, refit_every, n_samples, args.seed)
    nsdiff_samples = np.stack([r.pop("samples") for r in nsdiff_records]).astype(np.float64)
    nsdiff_index = pd.DataFrame(nsdiff_records)
    nsdiff_index["n_samples"] = n_samples
    print(f"NsDiff-W: {len(nsdiff_index)} rows x {n_samples} samples")

    print("\n=== 4. baseline resampling (Gaussian, matches prod weekly) ===")
    baseline_index, baseline_samples = generate_baselines(assets, origins, n_samples, args.seed)
    print(f"Baselines: {len(baseline_index)} rows x {n_samples} samples")

    if args.dry_run or args.smoke:
        print("\n--dry-run/--smoke: skipping DB write.")
    else:
        print(f"\n=== 5. DB write (source='{args.source_tag}') ===")
        run_id = f"{RUN_ID}-{args.source_tag}" if args.source_tag != "backtest_rolling_nsdiff" else RUN_ID
        n_written = save_backtest_rolling_rows_nsdiff(nsdiff_index, nsdiff_samples,
                                                       source_tag=args.source_tag, run_id=run_id)
        print(f"Upserted {n_written} NsDiff-W rows into {DB_PATH} (source='{args.source_tag}').")

    print("\n=== 6. KPI matrix ===")
    df = build_kpi_matrix(nsdiff_index, nsdiff_samples, baseline_index, baseline_samples)
    per_cell = aggregate_per_cell_nsdiff(df)
    print(f"{len(df)} per-row KPI records, {len(per_cell)} per-cell aggregates")

    print("\n=== 7. paired tests ===")
    paired = run_paired_tests(df, assets, FETCH_START, origins["cutoff_date"].min())

    payload_kpi = {
        "config": {
            "assets": assets, "refit_every": refit_every, "n_samples": n_samples,
            "epochs": epochs, "epoch_selection": "fixed/declared (no validation sweep), "
                             "mirrors weekly_nsdiff_production.NSDIFF_EPOCHS_W",
            "seed": args.seed, "fetch_start": FETCH_START, "fetch_end": FETCH_END,
            "source_tag": args.source_tag,
            "baseline_models": BASELINE_MODELS, "baseline_sampling": "gaussian_parametric_from_stored_PI",
            "nsdiff_sampling": "native_diffusion_samples",
            "definition": "nsdiff_model.fit_nsdiff (seq_len=30, daily module fed weekly data) -- "
                         "mirrors weekly_nsdiff_production.py's own definition, NOT nsdiff_weekly."
                         "fit_weekly (seq_len=26, the module used in duel_backtest_nsdiff.json) -- "
                         "declared mismatch, brief §5.",
            "asymmetry_caveat": "NsDiff-W draws genuine diffusion samples; the 5 baselines draw "
                               "Gaussian samples parametrically from their stored 95% PI, matching "
                               "PRODUCTION's own weekly protocol (Monte-Carlo/bootstrap too costly "
                               "at D+7+) -- not a shortcut invented for this backtest. If NsDiff-W "
                               "wins, check the gain isn't just the baselines' Gaussian handicap; "
                               "if it loses despite that potential handicap, the result is more robust.",
            "leakage_guard": "N/A -- NsDiff-W epochs are fixed/declared, never validation-selected, "
                             "so there is no epoch-selection signal to guard against (unlike "
                             "TSDiff-W's epoch_sweep_val_dates exclusion). All baseline-shared "
                             "origins are used.",
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "per_row": json.loads(df.astype(object).where(pd.notnull(df), None).to_json(orient="records")),
        "per_cell": json.loads(per_cell.astype(object).where(pd.notnull(per_cell), None).to_json(orient="records")),
    }
    Path(f"{args.out_prefix}.json").write_text(json.dumps(payload_kpi, indent=2, default=str))
    print(f"Saved -> {args.out_prefix}.json")

    Path(f"{args.out_prefix}_paired_tests.json").write_text(json.dumps(paired, indent=2, default=str))
    print(f"Saved -> {args.out_prefix}_paired_tests.json")

    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
