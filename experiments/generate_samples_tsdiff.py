"""
generate_samples_tsdiff.py — N=500 NATIVE TSDiff samples at the exact origins
already in tracking.db, without inventing any new protocol.

Why this can't be "just reload and resample" (verified directly, not assumed):
model_artifacts/pipeline.py never serializes a TSDiff checkpoint ("pas
d'artefact Gate 1 serialise, l'entrainement reel a lieu en Gate 2"), and no
.pt/.pkl for TSDiff exists anywhere in the repo. Every TSDiff point/PI
currently in tracking.db -- for ANY origin -- was itself produced by a single
train-once-forward pass (one fit_tsdiff call, then walk-forward sampling,
discarding the sample cloud down to mean + 2.5/97.5 quantiles). Confirmed by
reading Run/20260716-TSDiff-SPY-D1/{metadata,predictions}: one run_tsdiff()
call, train_end=2026-01-30, exactly matches metrics.json's n_val=113.

So getting genuine samples means re-running that SAME already-established
protocol (same frozen/live price series, same seed=42, same per-asset epoch
count already selected in epoch_sweep_results.json / weekly_headtohead_v2's
epochs_used) and keeping the N=500 draws instead of throwing them away. This
is a real fit per (asset, regime cell) -- not new methodology, just capturing
what the existing pipeline already computes internally at every origin.

Three regime cells, one fit each (all origins within a cell share the one
fitted model, walked forward -- exactly the "train-once-forward" compute
trick already used in weekly_multimodel.py / weekly_headtohead_v2.py):

  A. daily-native  (frequence=daily,  horizon_type=daily,  D+1 & D+7)
     -> tsdiff_model.py production defaults: horizon=7, epochs=40, seed=42
        (model_artifacts/pipeline.py's Gate2 TSDiff call takes no overrides),
        trained on a LOOKBACK_ROWS_A=700 rolling window (see below -- epochs=40
        was calibrated against that window size, not the full price history).
  B. daily->weekly (frequence=daily,  horizon_type=weekly, W+1/2/3, "TSDiff-D")
     -> horizon=HORIZON_DAILY=15 (weekly_headtohead.py), epochs = EPOCHS_TSDIFF_D
        (per-asset epoch_sweep selection, verified from epoch_sweep_results_
        d_extend.json [SPY,BTC] + epoch_sweep_results_ethzntlt_d.json [ETH,ZN,TLT]).
  C. weekly-native (frequence=weekly, horizon_type=weekly, W+1/2/3, "TSDiff-W")
     -> horizon=HORIZON_WEEKLY=3, epochs = EPOCHS_TSDIFF_W (per-asset selection
        from weekly_multiasset_results.json's config.epochs_used, all 5 assets).

Origins are NEVER re-derived from three_way_split/epoch_sweep -- they come
directly from tracking.db (whatever's already there is "the current matrix"),
so this is robust to any drift between the original generation scripts. Each
regime cell trains on data strictly before its earliest origin (no lookahead,
brief guardrail); within a cell the standardized-return buffer is only ever
extended with realised prices up to each origin (walk-forward, no leakage).

Persists to experiments/samples/<ASSET>_tsdiff.{index.parquet,npz}.

Usage:
    python generate_samples_tsdiff.py --asset SPY
    python generate_samples_tsdiff.py --asset SPY --cells A          # just D+1/D+7 (fast smoke test)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import tsdiff_model as td                                        # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly  # noqa: E402
from prob_kpi_common import load_matrix_rows, NATIVE_MODELS       # noqa: E402

SAMPLES_DIR = ROOT / "experiments" / "samples"

# Per-asset epochs already selected by epoch_sweep.py (argmin CRPS on a disjoint
# validation block, never touched by this pilot/run) -- verified directly from
# the json files, not guessed:
#   TSDiff-D (regime B): epoch_sweep_results_d_extend.json (SPY, BTC) +
#                         epoch_sweep_results_ethzntlt_d.json (ETH, ZN, TLT)
#   TSDiff-W (regime C): weekly_multiasset_results.json's config.epochs_used
#                         (all 5 assets in one run; BTC cross-checked against
#                         epoch_sweep_results_btcw_extend.json, matches)
EPOCHS_TSDIFF_D = {"SPY": 20, "BTC-USD": 20, "ETH-USD": 30, "ZN=F": 20, "TLT": 20}
EPOCHS_TSDIFF_W = {"SPY": 80, "BTC-USD": 30, "ETH-USD": 60, "ZN=F": 80, "TLT": 80}

HORIZON_DAILY_B = 15    # weekly_headtohead.py HORIZON_DAILY (regime B / TSDiff-D)
HORIZON_WEEKLY_C = 3    # weekly_headtohead.py HORIZON_WEEKLY (regime C / TSDiff-W)
DEFAULT_EPOCHS_A = td.EPOCHS          # 40 -- tsdiff_model.py production default (regime A)
FETCH_START = "2015-01-01"
# Regime A (tsdiff_model.py / pipeline.py Gate2) was always trained on a short
# ROLLING window (~2.5y -- checked directly: Run/*/metadata.json window_start
# is 2023-07-1x for a train_end around 2026-01/02, i.e. ~700 trading days), never
# the full history back to 2015. epochs=40 was tuned for that dataset size --
# feeding it the full 2015-> history (~2800 rows, ~4x the windows at the same
# batch_size) pushes far more gradient steps at the same epoch count and
# triggers TSDiff's already-documented epoch-collapse pathology (see
# weekly_multimodel.py's docstring), collapsing the sample cloud to near-zero
# variance. Bounding the lookback here is what keeps epochs=40 meaningful --
# not a new modelling choice, just matching the window size the epoch count
# was actually calibrated against.
LOOKBACK_ROWS_A = 700
K_DENOISE = td.K_DENOISE
SEED = 42


def _load_epochs_used(ticker: str) -> dict:
    """{'TSDiff-D': epochs, 'TSDiff-W': epochs} for this asset (ticker, e.g.
    'ZN=F'), from the already-completed epoch_sweep selections (see the
    EPOCHS_TSDIFF_* tables above). Falls back to tsdiff_model.py's own
    default (documented, not silently guessed) only if genuinely absent."""
    return {
        "TSDiff-D": EPOCHS_TSDIFF_D.get(ticker, DEFAULT_EPOCHS_A),
        "TSDiff-W": EPOCHS_TSDIFF_W.get(ticker, DEFAULT_EPOCHS_A),
    }


def _fetch(ticker: str) -> pd.Series:
    end = str((pd.Timestamp.today() + pd.Timedelta(days=1)).date())
    print(f"  fetching {ticker} [{FETCH_START} -> {end}] ...")
    return td.fetch_data(ticker, FETCH_START, end)


def _daily_pos(daily: pd.Series, date_str: str) -> int:
    return int(daily.index.get_loc(pd.Timestamp(date_str)))


def _gen_cell_A(daily: pd.Series, rows: pd.DataFrame, n_samples: int) -> list:
    """Regime A: daily-native D+1 & D+7, one fit (horizon=7) shared by both."""
    if rows.empty:
        return []
    min_cutoff = min(rows["cutoff_date"])
    cutoff_pos = _daily_pos(daily, min_cutoff)
    train_start_pos = max(0, cutoff_pos - LOOKBACK_ROWS_A)
    train = daily.iloc[train_start_pos:cutoff_pos]   # strictly before the earliest origin -- no lookahead
    print(f"  [A] train n={len(train)} ({train.index[0].date()} -> {train.index[-1].date()}), "
          f"{len(rows)} origins, horizon=7, epochs={DEFAULT_EPOCHS_A}")

    td.set_seed(SEED)
    t0 = time.time()
    model, mu, sd = td.fit_tsdiff(train, horizon=7, epochs=DEFAULT_EPOCHS_A)
    print(f"  [A] fitted in {time.time() - t0:.0f}s")

    r = td._log_returns(daily.values.astype(float))
    z = (r - mu) / sd

    # group rows by origin so D+1 and D+7 at the same cutoff_date share one sample_paths call
    out = []
    by_origin = rows.groupby("cutoff_date")
    for cutoff_date, grp in by_origin:
        pos = _daily_pos(daily, cutoff_date)
        last_price = float(daily.iloc[pos])
        needed_h = sorted(int(h) for h in grp["horizon"].unique())   # [1] and/or [7]
        samples_by_h = td.forecast_from_fitted(
            model, z[:pos], mu, sd, last_price, horizons=needed_h,
            n_samples=n_samples, k_denoise=K_DENOISE)
        for _, row in grp.iterrows():
            out.append({**row.to_dict(), "samples": samples_by_h[int(row["horizon"])],
                        "method": "native_tsdiff_regimeA"})
    return out


def _gen_cell_B(daily: pd.Series, rows: pd.DataFrame, epochs: int, n_samples: int) -> list:
    """Regime B: TSDiff-D, daily-trained, multi-step to weekly target (horizon=15)."""
    if rows.empty:
        return []
    min_cutoff = min(rows["cutoff_date"])
    cutoff_pos = _daily_pos(daily, min_cutoff)
    train = daily.iloc[:cutoff_pos]
    print(f"  [B] train n={len(train)} (<= {train.index[-1].date()}), "
          f"{len(rows)} rows, horizon={HORIZON_DAILY_B}, epochs={epochs}")

    td.set_seed(SEED)
    t0 = time.time()
    model, mu, sd = td.fit_tsdiff(train, horizon=HORIZON_DAILY_B, epochs=epochs)
    print(f"  [B] fitted in {time.time() - t0:.0f}s")

    r = td._log_returns(daily.values.astype(float))
    z = (r - mu) / sd

    out = []
    by_origin = sorted(rows.groupby("cutoff_date"), key=lambda kv: kv[0])
    for k, (cutoff_date, grp) in enumerate(by_origin):
        pos = _daily_pos(daily, cutoff_date)
        last_price = float(daily.iloc[pos])
        daily_horizons = sorted({_daily_pos(daily, td_) - pos for td_ in grp["target_date"]})
        daily_horizons = [h for h in daily_horizons if 1 <= h <= HORIZON_DAILY_B]
        if not daily_horizons:
            continue
        td.set_seed(SEED + k)
        samples_by_h = td.forecast_from_fitted(
            model, z[:pos], mu, sd, last_price, horizons=daily_horizons,
            n_samples=n_samples, k_denoise=K_DENOISE)
        for _, row in grp.iterrows():
            h_d = _daily_pos(daily, row["target_date"]) - pos
            if h_d not in samples_by_h:
                continue
            out.append({**row.to_dict(), "samples": samples_by_h[h_d],
                        "method": "native_tsdiff_regimeB"})
    return out


def _gen_cell_C(daily: pd.Series, rows: pd.DataFrame, epochs: int, n_samples: int) -> list:
    """Regime C: TSDiff-W, weekly-native (horizon=3 weekly steps)."""
    if rows.empty:
        return []
    weekly, weekly_dates = build_weekly(daily)
    min_cutoff = min(rows["cutoff_date"])
    # position in `weekly` of the last weekly point strictly before the earliest origin
    pos_of = {}
    wd_values = weekly_dates.values
    for cutoff_date in rows["cutoff_date"].unique():
        matches = np.where(wd_values == np.datetime64(pd.Timestamp(cutoff_date)))[0]
        if len(matches):
            pos_of[cutoff_date] = int(matches[0])
    if not pos_of:
        print("  [C] no origin date matches a weekly (Friday-anchored) trading date -- skipping.")
        return []
    train_end_pos = min(pos_of.values())
    train_weekly = weekly.iloc[:train_end_pos + 1]
    print(f"  [C] train n={len(train_weekly)} weekly obs "
          f"(<= {weekly_dates.iloc[train_end_pos].date()}), {len(rows)} rows, "
          f"horizon={HORIZON_WEEKLY_C}, epochs={epochs}")

    td.set_seed(SEED)
    t0 = time.time()
    model, mu, sd = td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY_C, epochs=epochs)
    print(f"  [C] fitted in {time.time() - t0:.0f}s")

    wr = td._log_returns(weekly.values.astype(float))
    wz = (wr - mu) / sd

    out = []
    by_origin = sorted(((d, g) for d, g in rows.groupby("cutoff_date") if d in pos_of),
                       key=lambda kv: pos_of[kv[0]])
    for k, (cutoff_date, grp) in enumerate(by_origin):
        m = pos_of[cutoff_date]
        if m == 0:
            continue   # no history to condition on
        last_price = float(weekly.iloc[m])
        needed_h = sorted(int(h) for h in grp["horizon"].unique())
        needed_h = [h for h in needed_h if 1 <= h <= HORIZON_WEEKLY_C]
        if not needed_h:
            continue
        td.set_seed(SEED + k)
        samples_by_h = td.forecast_from_fitted(
            model, wz[:m], mu, sd, last_price, horizons=needed_h,
            n_samples=n_samples, k_denoise=K_DENOISE)
        for _, row in grp.iterrows():
            h = int(row["horizon"])
            if h not in samples_by_h:
                continue
            out.append({**row.to_dict(), "samples": samples_by_h[h],
                        "method": "native_tsdiff_regimeC"})
    return out


def generate(asset: str, n_samples: int = 500, cells: str = "ABC",
             db_path: str = None) -> tuple:
    kwargs = {} if db_path is None else {"db_path": db_path}
    rows = load_matrix_rows(asset, models=list(NATIVE_MODELS), **kwargs)
    if rows.empty:
        raise SystemExit(f"no TSDiff rows found for asset={asset!r}")

    ticker = ASSET_TICKERS.get(asset, asset)
    daily = _fetch(ticker)

    rows_A = rows[rows["horizon_type"] == "daily"]
    rows_B = rows[(rows["horizon_type"] == "weekly") & (rows["frequence"] == "daily")]
    rows_C = rows[(rows["horizon_type"] == "weekly") & (rows["frequence"] == "weekly")]

    epochs_used = _load_epochs_used(ticker)

    records = []
    if "A" in cells:
        print(f"[{asset}] regime A (daily-native D+1/D+7): {len(rows_A)} rows")
        records += _gen_cell_A(daily, rows_A, n_samples)
    if "B" in cells:
        print(f"[{asset}] regime B (TSDiff-D, daily->weekly): {len(rows_B)} rows")
        records += _gen_cell_B(daily, rows_B, epochs_used.get("TSDiff-D", DEFAULT_EPOCHS_A), n_samples)
    if "C" in cells:
        print(f"[{asset}] regime C (TSDiff-W, weekly-native): {len(rows_C)} rows")
        records += _gen_cell_C(daily, rows_C, epochs_used.get("TSDiff-W", DEFAULT_EPOCHS_A), n_samples)

    if not records:
        raise SystemExit(f"no samples generated for asset={asset!r}, cells={cells!r}")

    samples = np.stack([r.pop("samples") for r in records]).astype(np.float64)
    index = pd.DataFrame(records)
    index["n_samples"] = n_samples
    return index.reset_index(drop=True), samples


def save(asset: str, index: pd.DataFrame, samples: np.ndarray, suffix: str = "tsdiff") -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    index_path = SAMPLES_DIR / f"{asset}_{suffix}.index.parquet"
    samples_path = SAMPLES_DIR / f"{asset}_{suffix}.samples.npz"
    index.to_parquet(index_path)
    np.savez_compressed(samples_path, samples=samples)
    print(f"[{asset}] {suffix}: {len(index)} rows x {samples.shape[1]} samples -> "
          f"{index_path.name}, {samples_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset", required=True)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--cells", default="ABC", help="subset of ABC, e.g. 'A' for a fast smoke test")
    p.add_argument("--db-path", default=None)
    args = p.parse_args()

    index, samples = generate(args.asset, args.n_samples, args.cells.upper(), args.db_path)
    save(args.asset, index, samples)


if __name__ == "__main__":
    main()
