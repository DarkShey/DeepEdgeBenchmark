"""
backtest_rolling_tsdiffw.py -- BRIEF_backtest_rolling_origin_tsdiffw.md: rolling-
origin backtest of TSDiff-W (weekly-native) against the 5 baselines, on
STRICTLY IDENTICAL (asset, cutoff, horizon) triplets, to settle whether P1
(the weekly TSDiff calibration collapse -- an artefact of scoring TSDiff-D
pushed multi-step, not TSDiff-W) is an artefact or a real deficit.

Origins: reused verbatim from the 5 baselines' ALREADY-EXISTING native-weekly
walk-forward backtest in tracking.db (model in ARIMA-GARCH/SARIMA/Prophet/
Naive/LSTM, frequence='weekly', horizon_type='weekly', daily_duplicate=0,
source='oos' -- experiments/weekly_multimodel_n90.json's n_test=90 run,
2024-10-18 -> 2026-07-02/03, one retrain per origin per baseline). This buys
exact triplet alignment "for free" -- no baseline recomputation, and no risk
of TSDiff's split landing on different Fridays than the baselines' split.

Anti-leakage for TSDiff-W (non-negotiable #1):
  - EXPANDING-WINDOW PERIODIC REFIT every REFIT_EVERY origins (default 15,
    ~6 refits/asset): each refit trains fit_tsdiff on weekly data <= that
    block's first reused cutoff (own mu/sd, never global), then every origin
    in the block is forecast from that ONE frozen model via
    forecast_from_fitted(..., hist_window=weekly_z[:m]) -- conditioning
    history is always <= the origin, weights are never updated on data past
    the block's training cutoff. Documented approximation (per brief: "si
    trop coûteux de réentraîner à chaque cutoff, documenter l'approximation
    et son biais"): staleness is bounded to <= REFIT_EVERY weeks instead of
    the ~90 weeks a pure train-once-forward pass (weekly_multiasset.py's
    existing protocol) would carry for the latest origins. This is strictly
    MORE conservative than the existing validated TSDiff-W tooling
    (weekly_headtohead_v2.run_pair_v2 / weekly_multiasset.py --phase final),
    which this script otherwise mirrors line-for-line (same fit_tsdiff /
    forecast_from_fitted calls, same standardized-return convention).
  - Epoch-selection leakage guard: epoch_sweep_results.json's TSDiff-W
    epochs* were chosen by argmin-CRPS on a 12-origin validation block
    positioned relative to n_test=30 (n_val=12, end="2026-07-16" -- read
    directly from that file's config). At n_test=90 that block's dates fall
    INSIDE the new, larger test range (three_way_split shifts the val block
    earlier as n_test grows) -- so those ~12 dates/asset are excluded from
    this backtest's origin set per asset (recomputed exactly, not
    eyeballed: refit epoch_sweep's own three_way_split with n_val=12,
    n_test=30, end="2026-07-16" and drop the resulting val dates).

Asymmetry documented, not hidden (per project convention, cf.
pooled_analysis_prob.json's own caveat): TSDiff-W samples are GENUINE
diffusion draws (model.sample_paths); the 5 baselines' N=500 clouds are
GAUSSIAN, drawn parametrically from their already-stored 95% PI
(prob_kpi_common.sample_parametric) -- this mirrors PRODUCTION exactly (the
weekly pipeline uses a Gaussian PI for these 5 models at weekly horizon,
Monte-Carlo/bootstrap being too costly at D+7+; see BRIEF's follow-up note),
not a shortcut invented for this backtest. If TSDiff-W wins on CRPS/
calibration despite producing a richer (non-Gaussian) distribution, check
the gain doesn't simply come from the baselines' Gaussian handicap; if it
loses despite that potential baseline handicap, the result is only more
robust. The NOTE this script produces reports TSDiff-W's OWN coverage
50/80/95 standalone (method-independent) alongside the head-to-head.

Storage: TSDiff-W backtest rows are inserted with source='backtest_rolling'
(new column value, no schema change -- `source` is free TEXT) via a NEW
partial unique index mirroring idx_predictions_oos_unique, so reruns are
idempotent and 100% isolated from 'live'/'oos' rows (non-negotiable #4).
Baseline rows are NEVER written here -- only read (they already exist).

Metrics: crps_metrics.crps_empirical + prob_kpi_common.row_kpis (coverage/
sharpness/Winkler/PIT at 50/80/95) -- zero reimplementation (non-negotiable
#3). Aggregation reuses compute_prob_kpi_pilot.compute_row_kpis/add_mase and
build_kpi_probabilistes.aggregate_per_cell verbatim, so the output matrix is
byte-for-byte the same *shape* as kpi_probabilistes.json.

Paired tests: per (asset, horizon, TSDiff-W vs baseline) block bootstrap
(paired_test.paired_block_bootstrap_test, block_length=3 -- W1-W3 overlap
convention, mandatory per BRIEF_soir_D7_tests_apparies.md) PLUS a pooled,
asset-class-clustered, scale-normalised test per horizon
(pooled_analysis.class_series/pooled_diff_series/dual_test/holm_correction,
same primitives as pooled_analysis_prob.json), Holm-corrected across the 5
baseline comparisons separately per horizon.

Usage:
    python backtest_rolling_tsdiffw.py --smoke                  # 1 asset, few origins, tiny epochs/N -- plumbing only
    python backtest_rolling_tsdiffw.py                          # full run, all 5 assets
    python backtest_rolling_tsdiffw.py --assets SPY BTC-USD      # subset
    python backtest_rolling_tsdiffw.py --dry-run                # compute + print, no DB writes
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

import tsdiff_model as td                                              # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly     # noqa: E402
from epoch_sweep import three_way_split                                 # noqa: E402
from weekly_tsdiff_production import load_epochs as load_tsdiffw_epochs  # noqa: E402
from prob_kpi_common import sample_parametric, PARAMETRIC_MODELS, Z95    # noqa: E402
from compute_prob_kpi_pilot import compute_row_kpis, add_mase            # noqa: E402
from build_kpi_probabilistes import aggregate_per_cell, ASSET_CLASS      # noqa: E402
from paired_test import paired_block_bootstrap_test                      # noqa: E402
from pooled_analysis import (                                            # noqa: E402
    compute_asset_scales, class_series, pooled_diff_series, dual_test, holm_correction,
)
from validation import tracking_db as tdb                                # noqa: E402

DB_PATH = str(ROOT / "validation" / "tracking.db")
BASELINE_MODELS = list(PARAMETRIC_MODELS)          # ARIMA-GARCH, SARIMA, Prophet, Naive, LSTM
HORIZONS = (1, 2, 3)
HORIZON_UNITS = {1: "W+1", 2: "W+2", 3: "W+3"}

DEFAULT_N_SAMPLES = 500
DEFAULT_K_DENOISE = td.K_DENOISE
DEFAULT_SEED = 42
DEFAULT_REFIT_EVERY = 15

# Epoch-sweep leakage guard: the exact (n_val, n_test, end) that produced the
# TSDiff-W epochs* currently in epoch_sweep_results.json (verified directly
# from that file's "config" key, not assumed).
SWEEP_N_VAL = 12
SWEEP_N_TEST = 30
SWEEP_END = "2026-07-16"
SWEEP_START = "2015-01-01"

FETCH_START = "2015-01-01"
FETCH_END = "2026-07-24"    # matches weekly_multimodel_n90.json's baseline run -- fixed, not "today"

RUN_ID = "backtest-rolling-tsdiffw-v1"


# ── 1. origin alignment ──────────────────────────────────────────────────────

def load_baseline_triplets(assets: list, db_path: str = DB_PATH) -> pd.DataFrame:
    """The already-realised (asset, cutoff_date) x (W+1,W+2,W+3) triplets shared
    by ALL 5 baselines -- the ground truth for exact alignment (brief
    non-negotiable #2). One row per (asset, cutoff_date, horizon_unit), with
    target_date/last_close/y_true asserted identical across the 5 models."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT model, asset, horizon, horizon_unit, cutoff_date, target_date,
                   last_close, y_true
            FROM predictions
            WHERE model IN ({}) AND asset IN ({})
                  AND frequence='weekly' AND horizon_type='weekly'
                  AND daily_duplicate=0 AND source='oos' AND y_true IS NOT NULL
            """.format(",".join("?" * len(BASELINE_MODELS)), ",".join("?" * len(assets))),
            con, params=[*BASELINE_MODELS, *assets],
        )
    finally:
        con.close()

    rows = []
    for (asset, cutoff_date, horizon_unit), grp in df.groupby(["asset", "cutoff_date", "horizon_unit"]):
        if set(grp["model"]) != set(BASELINE_MODELS):
            continue    # not all 5 baselines resolved this triplet yet -- skip
        target_dates = grp["target_date"].unique()
        last_closes = grp["last_close"].unique()
        y_trues = grp["y_true"].unique()
        if len(target_dates) != 1 or len(y_trues) != 1:
            raise SystemExit(f"baseline disagreement on target_date/y_true at "
                            f"{(asset, cutoff_date, horizon_unit)}: {grp}")
        if not np.allclose(last_closes, last_closes[0], rtol=1e-6):
            raise SystemExit(f"baseline disagreement on last_close at "
                            f"{(asset, cutoff_date, horizon_unit)}: {last_closes}")
        rows.append({
            "asset": asset, "cutoff_date": cutoff_date, "horizon_unit": horizon_unit,
            "horizon": int(grp["horizon"].iloc[0]), "target_date": target_dates[0],
            "last_close": float(last_closes[0]), "y_true": float(y_trues[0]),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["asset", "cutoff_date", "horizon"]).reset_index(drop=True)


def epoch_sweep_val_dates(ticker: str) -> set:
    """The ~12 origin dates used as TSDiff-W's epoch-selection validation block
    (see module docstring, 'Epoch-selection leakage guard') -- to be excluded
    from this backtest's origin set for this asset."""
    daily = td.fetch_data(ticker, SWEEP_START, SWEEP_END)
    weekly, weekly_dates = build_weekly(daily)
    _, val_pos, _ = three_way_split(weekly, SWEEP_N_VAL, SWEEP_N_TEST)
    return {str(weekly_dates.iloc[p].date()) for p in val_pos}


def select_origins(triplets: pd.DataFrame, assets: list) -> pd.DataFrame:
    """Drop, per asset, any cutoff_date inside that asset's epoch-sweep
    validation block (leakage guard). Returns the surviving triplets."""
    keep_frames = []
    for asset in assets:
        ticker = asset   # `asset` column already stores the ticker (BTC-USD etc.)
        excluded = epoch_sweep_val_dates(ticker)
        sub = triplets[triplets["asset"] == asset]
        n_before = sub["cutoff_date"].nunique()
        sub = sub[~sub["cutoff_date"].isin(excluded)]
        n_after = sub["cutoff_date"].nunique()
        print(f"[{asset}] origins: {n_before} baseline cutoffs -> {n_after} after "
              f"excluding {len(excluded)} epoch-sweep validation dates "
              f"({n_before - n_after} actually overlapped)")
        keep_frames.append(sub)
    return pd.concat(keep_frames, ignore_index=True)


# ── 2. TSDiff-W generation (periodic expanding refit) ────────────────────────

def _weekly_position(weekly_dates: pd.Series, date_str: str):
    wd = weekly_dates.values
    matches = np.where(wd == np.datetime64(pd.Timestamp(date_str)))[0]
    return int(matches[0]) if len(matches) else None


def generate_tsdiffw_asset(asset: str, ticker: str, epochs: int, origin_rows: pd.DataFrame,
                            refit_every: int, n_samples: int, k_denoise: int, seed: int) -> list:
    """Periodic expanding-window refit TSDiff-W over this asset's surviving
    origins. Mirrors generate_samples_tsdiff.py::_gen_cell_C's fit/sample
    mechanics exactly, applied per REFIT_EVERY-sized chronological block
    instead of once for the whole span."""
    print(f"[{asset}] downloading {ticker} ({FETCH_START} -> {FETCH_END}) ...")
    daily = td.fetch_data(ticker, FETCH_START, FETCH_END)
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
        td.set_seed(seed + b_idx)
        model, mu, sd = td.fit_tsdiff(train_weekly, horizon=3, epochs=epochs)
        fit_s = time.time() - t0
        print(f"[{asset}] block {b_idx + 1}/{len(blocks)}: fit <= {block[0]} "
              f"({len(train_weekly)} weekly obs) in {fit_s:.0f}s, {len(block)} origins")

        wr = td._log_returns(weekly.values.astype(float))
        wz = (wr - mu) / sd

        for k, cutoff_date in enumerate(block):
            m = pos_of[cutoff_date]
            if m == 0:
                continue
            last_price = float(weekly.iloc[m])
            grp = origin_rows[(origin_rows["asset"] == asset) & (origin_rows["cutoff_date"] == cutoff_date)]
            needed_h = sorted(int(h) for h in grp["horizon"].unique())
            td.set_seed(seed + b_idx * 1000 + k)
            samples_by_h = td.forecast_from_fitted(
                model, wz[:m], mu, sd, last_price, horizons=needed_h,
                n_samples=n_samples, k_denoise=k_denoise)
            for _, row in grp.iterrows():
                h = int(row["horizon"])
                actual_target = str(weekly_dates.iloc[m + h].date())
                if actual_target != row["target_date"]:
                    raise SystemExit(f"[{asset}] target_date mismatch at cutoff={cutoff_date} "
                                    f"h={h}: TSDiff grid={actual_target} vs baseline={row['target_date']}")
                if abs(last_price - row["last_close"]) > 1e-6 * max(1.0, abs(row["last_close"])):
                    raise SystemExit(f"[{asset}] last_close mismatch at cutoff={cutoff_date}: "
                                    f"TSDiff={last_price} vs baseline={row['last_close']}")
                records.append({
                    "model": "TSDiff", "asset": asset, "horizon": h,
                    "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": HORIZON_UNITS[h],
                    "cutoff_date": cutoff_date, "target_date": row["target_date"],
                    "last_close": last_price, "y_true": row["y_true"],
                    "method": "native_tsdiff_regimeC_rolling_backtest",
                    "refit_block": b_idx, "samples": samples_by_h[h],
                })
    return records


# ── 3. baseline resampling (Gaussian, matches production's weekly protocol) ──

def generate_baselines(assets: list, origin_rows: pd.DataFrame, n_samples: int, seed: int,
                        db_path: str = DB_PATH) -> tuple:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT model, asset, horizon, frequence, horizon_type, horizon_unit,
                   cutoff_date, target_date, last_close, y_pred, y_lower, y_upper, y_true
            FROM predictions
            WHERE model IN ({}) AND asset IN ({})
                  AND frequence='weekly' AND horizon_type='weekly'
                  AND daily_duplicate=0 AND source='oos' AND y_true IS NOT NULL
            """.format(",".join("?" * len(BASELINE_MODELS)), ",".join("?" * len(assets))),
            con, params=[*BASELINE_MODELS, *assets],
        )
    finally:
        con.close()

    key_cols = ["asset", "cutoff_date", "horizon_unit"]
    keep_keys = set(map(tuple, origin_rows[key_cols].drop_duplicates().values.tolist()))
    df["_key"] = list(zip(df["asset"], df["cutoff_date"], df["horizon_unit"]))
    df = df[df["_key"].isin(keep_keys)].drop(columns="_key").reset_index(drop=True)

    n = len(df)
    samples = np.empty((n, n_samples), dtype=np.float64)
    for i, row in df.iterrows():
        row_seed = (seed, row["model"], row["frequence"], row["horizon_type"],
                    row["horizon_unit"], row["cutoff_date"])
        rng = np.random.default_rng(abs(hash(row_seed)) % (2**32))
        samples[i] = sample_parametric(row["model"], row["y_pred"], row["y_lower"], row["y_upper"],
                                       row["last_close"], n_samples, rng)
    df["method"] = "parametric_gaussian_prod_weekly"
    df["n_samples"] = n_samples
    return df.reset_index(drop=True), samples


# ── 4. DB storage (source='backtest_rolling', isolated + idempotent) ─────────

def ensure_backtest_rolling_index(source_tag: str, db_path: str = DB_PATH) -> None:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        con.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_{source_tag}_unique
            ON predictions (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
            WHERE source = '{source_tag}'
        """)
        con.commit()
    finally:
        con.close()


def save_backtest_rolling_rows(tsdiff_index: pd.DataFrame, tsdiff_samples: np.ndarray,
                                source_tag: str = "backtest_rolling", run_id: str = RUN_ID,
                                db_path: str = DB_PATH) -> int:
    """`source_tag` isolates distinct epoch/protocol variants from each other AND
    from the original 'backtest_rolling' (epochs=30/60 for BTC/ETH) run -- each
    tag gets its own partial unique index, so reruns are idempotent per-tag and
    variants never collide/overwrite one another (non-negotiable #5)."""
    ensure_backtest_rolling_index(source_tag, db_path)
    import sqlite3
    con = sqlite3.connect(db_path)
    n = 0
    try:
        for i, row in tsdiff_index.iterrows():
            s = tsdiff_samples[i]
            y_pred = float(np.mean(s))
            y_lower, y_upper = (float(q) for q in np.quantile(s, [0.025, 0.975]))
            rec = {
                "run_id": run_id, "model": "TSDiff", "asset": row["asset"],
                "horizon": int(row["horizon"]), "regime": "unknown",
                "cutoff_date": row["cutoff_date"], "target_date": row["target_date"],
                "last_close": float(row["last_close"]), "y_pred": y_pred,
                "y_lower": y_lower, "y_upper": y_upper, "y_true": float(row["y_true"]),
                "source": source_tag, "frequence": "weekly", "horizon_type": "weekly",
                "horizon_unit": row["horizon_unit"],
                "real_flag": tdb.compute_real_flag("TSDiff", row["cutoff_date"]),
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


# ── 5. KPI matrix + paired tests ──────────────────────────────────────────────

def build_kpi_matrix(tsdiff_index: pd.DataFrame, tsdiff_samples: np.ndarray,
                     baseline_index: pd.DataFrame, baseline_samples: np.ndarray) -> pd.DataFrame:
    full_index = pd.concat([
        tsdiff_index.drop(columns=["samples"], errors="ignore").reset_index(drop=True),
        baseline_index.drop(columns=["y_pred", "y_lower", "y_upper"], errors="ignore").reset_index(drop=True),
    ], ignore_index=True)
    full_samples = np.concatenate([tsdiff_samples, baseline_samples], axis=0)
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
        t = sub_h[sub_h["model"] == "TSDiff"]
        pvals_this_horizon, comparisons_this_horizon = [], []
        for baseline in BASELINE_MODELS:
            b = sub_h[sub_h["model"] == baseline]
            # per-asset block bootstrap
            for asset in assets:
                ta = t[t["asset"] == asset].set_index("cutoff_date").sort_index()
                ba = b[b["asset"] == asset].set_index("cutoff_date").sort_index()
                joined = ta[["crps"]].join(ba[["crps"]], lsuffix="_tsdiff", rsuffix="_baseline", how="inner")
                if len(joined) < 4:
                    continue
                diffs = (joined["crps_tsdiff"] - joined["crps_baseline"]).values
                res = paired_block_bootstrap_test(diffs, block_length=3)
                out["per_asset"][f"{asset}|{horizon_unit}|TSDiff vs {baseline}"] = res

            # pooled, asset-class-clustered, scale-normalised
            t_norm = t.copy()
            b_norm = b.copy()
            t_norm["crps_norm"] = t_norm["crps"] / t_norm["asset"].map(scales_by_ticker)
            b_norm["crps_norm"] = b_norm["crps"] / b_norm["asset"].map(scales_by_ticker)
            diffs_pooled = pooled_diff_series(t_norm, b_norm, "crps_norm", date_col="cutoff_date")
            res_pooled = dual_test(diffs_pooled, h=int(horizon_unit[-1]))
            out["pooled"][f"{horizon_unit}|TSDiff vs {baseline}"] = res_pooled
            if res_pooled.get("status") == "tested":
                pvals_this_horizon.append(res_pooled["p_value_bootstrap"])
                comparisons_this_horizon.append(f"{horizon_unit}|TSDiff vs {baseline}")

        if pvals_this_horizon:
            adjusted = holm_correction(pvals_this_horizon)
            for key, p_adj in zip(comparisons_this_horizon, adjusted):
                out["pooled"][key]["p_value_bootstrap_holm"] = p_adj
                out["pooled"][key]["significant_after_holm"] = bool(p_adj < 0.05)
    return out


# ── 6. main ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()),
                   help="tickers, e.g. SPY BTC-USD (default: all 5)")
    p.add_argument("--refit-every", type=int, default=DEFAULT_REFIT_EVERY)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dry-run", action="store_true", help="compute + print, no DB writes")
    p.add_argument("--smoke", action="store_true",
                   help="tiny plumbing check: 1 asset (SPY), refit-every=100 (1 block), "
                       "epochs=2, n-samples=6, k-denoise=3 -- overrides other flags")
    p.add_argument("--out-prefix", default=str(ROOT / "experiments" / "backtest_rolling_tsdiffw"))
    p.add_argument("--epochs-override", type=int, default=None,
                   help="force this epoch count for every --assets asset (bypasses "
                       "epoch_sweep_results.json selection) -- for the P1bis epoch-bracket "
                       "robustness check. Requires --source-tag != 'backtest_rolling' so the "
                       "original epochs=30/60 rows are never overwritten.")
    p.add_argument("--source-tag", default="backtest_rolling",
                   help="DB source value + own partial unique index -- isolates this run's "
                       "rows from the original 'backtest_rolling' run and from other tags.")
    args = p.parse_args()

    if args.epochs_override is not None and args.source_tag == "backtest_rolling" and not args.dry_run:
        raise SystemExit("--epochs-override requires a distinct --source-tag (would otherwise "
                        "overwrite the original epochs=30/60 backtest_rolling rows).")

    assets = ["SPY"] if args.smoke else args.assets
    refit_every = 100 if args.smoke else args.refit_every
    n_samples = 6 if args.smoke else args.n_samples
    k_denoise = 3 if args.smoke else args.k_denoise
    epochs_override = 2 if args.smoke else args.epochs_override

    t_start = time.time()

    print("=== 1. baseline triplets ===")
    triplets = load_baseline_triplets(assets)
    print(f"{len(triplets)} raw (asset,cutoff,horizon) rows across {len(assets)} asset(s)")

    print("\n=== 2. epoch-sweep leakage exclusion ===")
    origins = select_origins(triplets, assets)
    print(f"{len(origins)} triplets after exclusion, {origins['cutoff_date'].nunique()} unique cutoffs total")

    print("\n=== 3. TSDiff-W generation (periodic expanding refit) ===")
    epochs_by_short = load_tsdiffw_epochs()   # keyed by short code (SPY/BTC/ETH/ZN/TLT)
    epochs_by_asset = {ASSET_TICKERS[short]: v for short, v in epochs_by_short.items()}   # -> ticker-keyed
    tsdiff_records = []
    for asset in assets:
        ep = epochs_override or epochs_by_asset[asset]
        origin_rows = origins[origins["asset"] == asset]
        tsdiff_records += generate_tsdiffw_asset(
            asset, asset, ep, origin_rows, refit_every, n_samples, k_denoise, args.seed)
    tsdiff_samples = np.stack([r.pop("samples") for r in tsdiff_records]).astype(np.float64)
    tsdiff_index = pd.DataFrame(tsdiff_records)
    tsdiff_index["n_samples"] = n_samples
    print(f"TSDiff-W: {len(tsdiff_index)} rows x {n_samples} samples")

    print("\n=== 4. baseline resampling (Gaussian, matches prod weekly) ===")
    baseline_index, baseline_samples = generate_baselines(assets, origins, n_samples, args.seed)
    print(f"Baselines: {len(baseline_index)} rows x {n_samples} samples")

    if args.dry_run or args.smoke:
        print("\n--dry-run/--smoke: skipping DB write.")
    else:
        print(f"\n=== 5. DB write (source='{args.source_tag}') ===")
        run_id = RUN_ID if args.source_tag == "backtest_rolling" else f"{RUN_ID}-{args.source_tag}"
        n_written = save_backtest_rolling_rows(tsdiff_index, tsdiff_samples,
                                               source_tag=args.source_tag, run_id=run_id)
        print(f"Upserted {n_written} TSDiff-W rows into {DB_PATH} (source='{args.source_tag}').")

    print("\n=== 6. KPI matrix ===")
    df = build_kpi_matrix(tsdiff_index, tsdiff_samples, baseline_index, baseline_samples)
    per_cell = aggregate_per_cell(df)
    print(f"{len(df)} per-row KPI records, {len(per_cell)} per-cell aggregates")

    print("\n=== 7. paired tests ===")
    paired = run_paired_tests(df, assets, FETCH_START, origins["cutoff_date"].min())

    payload_kpi = {
        "config": {
            "assets": assets, "refit_every": refit_every, "n_samples": n_samples,
            "k_denoise": k_denoise, "seed": args.seed, "fetch_start": FETCH_START, "fetch_end": FETCH_END,
            "source_tag": args.source_tag, "epochs_override": args.epochs_override,
            "baseline_models": BASELINE_MODELS, "baseline_sampling": "gaussian_parametric_from_stored_PI",
            "tsdiff_sampling": "native_diffusion_samples",
            "asymmetry_caveat": "TSDiff-W draws genuine diffusion samples; the 5 baselines draw "
                               "Gaussian samples parametrically from their stored 95% PI, matching "
                               "PRODUCTION's own weekly protocol (Monte-Carlo/bootstrap too costly "
                               "at D+7+) -- not a shortcut invented for this backtest. If TSDiff-W "
                               "wins, check the gain isn't just the baselines' Gaussian handicap; "
                               "if it loses despite that potential handicap, the result is more robust.",
            "leakage_guard": f"excluded per-asset epoch-sweep validation dates "
                             f"(n_val={SWEEP_N_VAL}, n_test={SWEEP_N_TEST}, end={SWEEP_END})",
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
