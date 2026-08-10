"""
oos_nsdiff_daily_weekly.py -- NsDiff regime B (daily -> weekly, W+1/2/3) AND
regime C (weekly-native, W+1/2/3), inserted as `source='oos'`, on the EXACT
SAME (asset, cutoff_date, target_date) triplets as the 6 models already in
`tracking.db`'s oos rows. Closes BRIEF_nsdiff_dashboard_daily_oos.md: without
this, NsDiff has no daily side and no oos rows at all, so
`dashboard_d7_w1.py` (which reads `source='oos'` only and pairs regime B with
regime C per model) never sees it.

Mechanism: TRAIN-ONCE-FORWARD, mirroring `weekly_headtohead_v2.run_pair_v2`
line-for-line (swap `tsdiff_model` -> `nsdiff_model`) -- this is how TSDiff's
OWN oos rows (regime B "TSDiff-D" + regime C "TSDiff-W",
`weekly_headtohead_v2_n90.json`, `--n-test 90 --seed 42 --n-samples 50`) were
actually produced, verified by inspecting that file's `config` (n_test=90,
seed=42, n_samples=50, k_denoise=20) and by tracing `backfill_weekly_
predictions.py` (`source='oos'`, `st.insert_oos_predictions`). NOT the
periodic-refit mechanism of `backtest_rolling_nsdiffw.py` (that one targets a
separate, isolated `source='backtest_rolling_nsdiff'`, a different chantier).
Fit once per asset on data strictly before the earliest reused origin, then
forecast forward at every origin from growing (but never leaking) history
slices -- no retraining in the loop. Cheaper than per-origin retrain (this
repo's classical-model convention, `weekly_multimodel.py`) and faithful to
"reproduce identically" for a diffusion model, whose retrain cost is the
whole reason TSDiff itself never uses per-origin retrain either.

Origins: NOT recomputed via `epoch_sweep.three_way_split` (a fresh yfinance
pull could drift vs. the frozen baseline data -- exactly what happened to TLT,
see `NOTE_backtest_rolling_nsdiffw.md` §0). Instead read VERBATIM from the 5
baselines' (ARIMA-GARCH/SARIMA/Prophet/Naive/LSTM) existing oos rows:
regime C via `backtest_rolling_tsdiffw.load_baseline_triplets` (imported,
unmodified, already used by `backtest_rolling_nsdiffw.py` for the same
purpose); regime B via `load_baseline_triplets_daily` below (frequence=
'daily' mirror of that function -- no equivalent existed).

`load_baseline_triplets_daily` deliberately does NOT filter
`daily_duplicate=0` the way `load_baseline_triplets` does. Verified directly
against tracking.db: `validation.tracking_db.flag_daily_duplicates()`
partitions by `(source, model, asset, horizon, cutoff_date, target_date)` --
it does NOT include `frequence`/`horizon_type`. At W+1/2/3 a model's
regime-B (daily) and regime-C (weekly) rows share that exact tuple (only
`frequence`/`horizon_type` differ), so the partition incorrectly treats them
as duplicates of each other and flags whichever has the lower `id` as
daily_duplicate=1 -- confirmed: ARIMA-GARCH's daily W+1 rows show only 60/90
cutoffs at daily_duplicate=0 (30 "duplicates"), yet grouping ALL of
source='oos' by (model,asset,frequence,horizon_type,horizon_unit,cutoff_date)
at W+1/2/3 finds ZERO groups with count>1 -- there is no real duplication to
filter. Pre-existing bug in a shared file (validation/tracking_db.py), out of
scope here (not modified) -- worked around locally instead.

TLT / yfinance drift (BRIEF §7): each asset's freshly-downloaded weekly
series is checked against the baselines' stored `last_close` at EVERY reused
origin BEFORE any model is fit (cheap, fails fast). On mismatch, retries once
with `experiments/offline_prices.fetch_data_offline` (frozen DONNEE~1.XLS
snapshot, verified elsewhere to match yfinance to 4e-6 relative). If both
sources mismatch, that asset is skipped (not the whole run) and reported in
the JSON summary + printed clearly -- never silently dropped.

Epochs/seq_len/k_denoise (BRIEF §5, "declared, coherent with the NsDiff-W
definition already used"): NSDIFF_EPOCHS=40 for BOTH regimes (=
`weekly_nsdiff_production.NSDIFF_EPOCHS_W`; no separate daily budget exists
anywhere in this repo -- inventing one would be an undeclared new number, so
the existing declared weekly budget is reused for daily too), seq_len=30 and
k_denoise=20 (nsdiff_model.py defaults, same ones weekly_nsdiff_production.py
and backtest_rolling_nsdiffw.py already use). n_samples=50 (matches
weekly_headtohead_v2_n90.json's own config -- NOT backtest_rolling_nsdiffw.py's
500, which serves a different, more precision-hungry KPI backtest).

Insertion: `validation.sim_trades.insert_oos_predictions` (same idempotent
upsert -- `ON CONFLICT (source, model, asset, horizon, frequence,
horizon_type, cutoff_date) WHERE source='oos'` -- as every other oos
backfill script; model='NsDiff' collides with nothing). Run
`experiments/backfill_eval_metrics.py` afterward (existing script, global
`WHERE abs_error IS NULL`, touches only the newly-inserted NULL rows) to
populate abs_error/in_interval/etc., matching the other 6 models' rows.

Usage:
    python oos_nsdiff_daily_weekly.py --smoke                  # 1 asset, tiny epochs/samples -- plumbing only
    python oos_nsdiff_daily_weekly.py --dry-run                # compute + verify + print, no DB writes
    python oos_nsdiff_daily_weekly.py                          # full run, all 5 assets, DB write
    python oos_nsdiff_daily_weekly.py --assets SPY BTC-USD      # subset
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nsdiff_model as nm                                               # noqa: E402
from weekly_headtohead import (                                         # noqa: E402
    ASSETS as ASSET_TICKERS, HORIZON_WEEKLY, HORIZON_DAILY, build_weekly,
)
from epoch_sweep import week_targets                                    # noqa: E402
from backtest_rolling_tsdiffw import (                                  # noqa: E402
    DB_PATH, BASELINE_MODELS, HORIZON_UNITS, FETCH_START, FETCH_END,
    load_baseline_triplets, _weekly_position,
)
from weekly_nsdiff_production import NSDIFF_EPOCHS_W as NSDIFF_EPOCHS   # noqa: E402
from offline_prices import fetch_data_offline                           # noqa: E402
from validation import sim_trades as st                                 # noqa: E402

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 50            # matches weekly_headtohead_v2_n90.json's own oos-generation config
DEFAULT_K_DENOISE = nm.K_DENOISE  # fit-time only for NsDiff (sampling-time kwarg is absorbed/ignored)
RUN_ID = "20260804-nsdiff-daily-weekly-oos"


class OriginMismatchError(Exception):
    pass


# ── 1. origin alignment ──────────────────────────────────────────────────────

def load_baseline_triplets_daily(assets: list, db_path: str = DB_PATH) -> pd.DataFrame:
    """Regime-B mirror of `load_baseline_triplets` (imported above, reused
    as-is for regime C) -- see module docstring for why the
    `daily_duplicate=0` filter that function applies is deliberately NOT
    reused here (pre-existing cross-regime collision bug, worked around, not
    fixed in the shared file)."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT model, asset, horizon, horizon_unit, cutoff_date, target_date,
                   last_close, y_true
            FROM predictions
            WHERE model IN ({}) AND asset IN ({})
                  AND frequence='daily' AND horizon_type='weekly'
                  AND source='oos' AND y_true IS NOT NULL
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
            raise SystemExit(f"baseline disagreement on target_date/y_true (regime B) at "
                            f"{(asset, cutoff_date, horizon_unit)}: {grp}")
        if not np.allclose(last_closes, last_closes[0], rtol=1e-6):
            raise SystemExit(f"baseline disagreement on last_close (regime B) at "
                            f"{(asset, cutoff_date, horizon_unit)}: {last_closes}")
        rows.append({
            "asset": asset, "cutoff_date": cutoff_date, "horizon_unit": horizon_unit,
            "horizon": int(grp["horizon"].iloc[0]), "target_date": target_dates[0],
            "last_close": float(last_closes[0]), "y_true": float(y_trues[0]),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["asset", "cutoff_date", "horizon"]).reset_index(drop=True)


# ── 2. price fetch with baseline-price verification + offline fallback ──────

def _verify_origin_prices(weekly: pd.Series, weekly_dates: pd.Series, origins: pd.DataFrame,
                          asset: str, tol: float = 1e-6) -> None:
    for cutoff_date, grp in origins.groupby("cutoff_date"):
        pos = _weekly_position(weekly_dates, cutoff_date)
        if pos is None:
            raise OriginMismatchError(f"[{asset}] cutoff {cutoff_date} not found on the W-FRI weekly grid")
        live_price = float(weekly.iloc[pos])
        stored = float(grp["last_close"].iloc[0])
        if abs(live_price - stored) > tol * max(1.0, abs(stored)):
            raise OriginMismatchError(
                f"[{asset}] last_close mismatch at cutoff={cutoff_date}: "
                f"fetched={live_price} vs baseline={stored}")


def fetch_verified(ticker: str, origins_c: pd.DataFrame, origins_b: pd.DataFrame):
    """Try live yfinance first, then the frozen offline cache
    (experiments/offline_prices.py) -- returns (daily, weekly, weekly_dates,
    source_label) for the first source whose prices match ALL reused baseline
    origins within tolerance, or None if both mismatch (asset skipped, not
    fatal -- see module docstring / BRIEF §7)."""
    for label, fetch_fn in (("live/yfinance", nm.fetch_data), ("offline/DONNEE~1.XLS", fetch_data_offline)):
        try:
            daily = fetch_fn(ticker, FETCH_START, FETCH_END)
        except Exception as exc:
            print(f"[{ticker}] {label} fetch raised {type(exc).__name__}: {exc} -- trying next source")
            continue
        weekly, weekly_dates = build_weekly(daily)
        try:
            _verify_origin_prices(weekly, weekly_dates, origins_c, ticker)
            _verify_origin_prices(weekly, weekly_dates, origins_b, ticker)
        except OriginMismatchError as exc:
            print(f"[{ticker}] {label}: {exc} -- trying next source")
            continue
        print(f"[{ticker}] price source: {label} -- verified against "
             f"{origins_c['cutoff_date'].nunique() + origins_b['cutoff_date'].nunique()} baseline last_close values")
        return daily, weekly, weekly_dates, label
    return None


# ── 3. diffusion generation (train-once-forward, mirrors weekly_headtohead_v2) ──

class DiffusionEngine(NamedTuple):
    """Ce qui distingue NsDiff de TSDiff dans la boucle de generation -- rien
    d'autre. Les deux modules exposent la meme API (`set_seed`,
    `_log_returns`, `forecast_from_fitted`) ; seuls le nom du fit et le
    traitement de `k_denoise` different (fit-time pour NsDiff, sampling-time
    pour TSDiff). Parametrer ce triplet est ce qui permet de generer les deux
    modeles par LA MEME boucle, au meme grain d'origines et avec la meme
    lecture des quantiles -- condition d'un match a budget egal (un duel dont
    les deux bras passent par deux codes differents n'en est pas un)."""
    name: str                # valeur du champ `model` des lignes produites
    module: object           # nsdiff_model / tsdiff_model
    fit: object              # (train, horizon, epochs, k_denoise) -> (model, mu, sd)
    forecast: object         # (model, hist_z, mu, sd, last_price, horizons, n_samples, k_denoise) -> {h: samples}


def nsdiff_engine() -> DiffusionEngine:
    """Moteur par defaut -- reproduit exactement les appels d'origine (en
    particulier `forecast_from_fitted` SANS `k_denoise`, que NsDiff absorbe
    dans `**_ignored` : il est fit-time pour lui, pas sampling-time)."""
    return DiffusionEngine(
        "NsDiff", nm,
        lambda train, horizon, epochs, k: nm.fit_nsdiff(train, horizon=horizon, epochs=epochs, k_denoise=k),
        lambda model, hist, mu, sd, last_price, horizons, n_samples, k: nm.forecast_from_fitted(
            model, hist, mu, sd, last_price, horizons=horizons, n_samples=n_samples),
    )


def _standardized_returns(prices: pd.Series, mu: float, sd: float, module=nm) -> np.ndarray:
    r = module._log_returns(prices.values.astype(float))
    return (r - mu) / sd


def generate_nsdiff_asset(asset: str, daily: pd.Series, weekly: pd.Series, weekly_dates: pd.Series,
                          origins_c: pd.DataFrame, origins_b: pd.DataFrame, epochs: int, seed: int,
                          n_samples: int, k_denoise: int, collect_samples: bool = False,
                          engine: DiffusionEngine = None, epochs_daily: int = None) -> tuple:
    """`collect_samples=True` (additive, default False -> bit-for-bit unchanged
    behaviour for every existing caller) stashes the raw sample cloud used for
    each row's point/quantiles under row["_samples"] (list[float], length
    n_samples), on top of the already-computed y_pred/y_lower/y_upper. The
    leading underscore is the convention the whole oos/DB path relies on:
    `insert_oos_predictions` never sees the key because the ensemble scripts
    strip `k != "_samples"` before inserting. Needed by the multi-seed ENSEMBLE
    (BRIEF_dashboard_multiseed_200.md §4) and by any consumer that must
    recompute quantiles at other levels or warp the cloud (conformal
    recalibration) rather than re-fit: concatenate the 5 seeds' clouds into 1
    of 5*n_samples before reading quantiles, instead of averaging 5
    already-quantiled bounds.

    `engine=None` -> `nsdiff_engine()`, i.e. the original NsDiff code path,
    call for call. Passing another `DiffusionEngine` (e.g. TSDiff) runs the
    SAME loop, same origins, same quantile reading, only swapping the model --
    see the class docstring.

    `epochs_daily=None` -> `epochs`, i.e. le meme budget pour les deux regimes
    (comportement d'origine). Le dissocier est necessaire des lors que le
    modele est sensible au sur-entrainement : la serie QUOTIDIENNE compte ~5x
    plus d'observations que l'hebdomadaire, donc a nombre d'epoques egal elle
    subit ~5x plus de pas de gradient. Mesure sur TSDiff-D/SPY : la largeur du
    PI a 95% s'effondre de 12.4% du prix a 10 epoques, a 3.4% a 20, 0.46% a 40
    et 0.12% a 80 -- au-dela de ~20 l'echantillonneur perd sa variance et la
    couverture tombe a quelques pour cent. NsDiff ne montre pas cette
    fragilite, mais la reutilisation aveugle du budget hebdomadaire cote daily
    n'est donc PAS une convention sure en general."""
    engine = engine or nsdiff_engine()
    epochs_daily = epochs if epochs_daily is None else epochs_daily
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
    engine.module.set_seed(seed)
    model_w, mu_w, sd_w = engine.fit(train_weekly, HORIZON_WEEKLY, epochs, k_denoise)
    print(f"[{asset}] {engine.name}-W (regime C) fitted in {time.time() - t0:.0f}s (mu={mu_w:.6f}, sd={sd_w:.6f})")

    t0 = time.time()
    engine.module.set_seed(seed)
    model_d, mu_d, sd_d = engine.fit(train_daily, HORIZON_DAILY, epochs_daily, k_denoise)
    print(f"[{asset}] {engine.name}-D (regime B, epochs={epochs_daily}) fitted in "
         f"{time.time() - t0:.0f}s (mu={mu_d:.6f}, sd={sd_d:.6f})")

    weekly_z = _standardized_returns(weekly, mu_w, sd_w, engine.module)
    daily_z = _standardized_returns(daily, mu_d, sd_d, engine.module)

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

        engine.module.set_seed(seed + k)
        samples_w = engine.forecast(model_w, weekly_z[:m], mu_w, sd_w, last_price,
                                    needed_h, n_samples, k_denoise)
        engine.module.set_seed(seed + k)
        samples_d = engine.forecast(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price,
                                    h_d_needed, n_samples, k_denoise)

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
                    "run_id": RUN_ID, "model": engine.name, "asset": asset, "horizon": h,
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
                    "run_id": RUN_ID, "model": engine.name, "asset": asset, "horizon": h,
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


# ── 4. main ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()),
                   help="tickers, e.g. SPY BTC-USD (default: all 5)")
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS,
                   help="fixed epoch budget, BOTH regimes (declared, mirrors "
                       "weekly_nsdiff_production.NSDIFF_EPOCHS_W -- no separate daily "
                       "budget exists, brief §5)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true", help="compute + verify + print, no DB writes")
    p.add_argument("--smoke", action="store_true",
                   help="tiny plumbing check: 1 asset (SPY), epochs=2, n-samples=6 -- overrides other flags")
    p.add_argument("--out", default=str(ROOT / "experiments" / "oos_nsdiff_daily_weekly_summary.json"))
    args = p.parse_args()

    assets = ["SPY"] if args.smoke else args.assets
    epochs = 2 if args.smoke else args.epochs
    n_samples = 6 if args.smoke else args.n_samples

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
                 f"baselines' stored last_close at every reused origin (yfinance drift, cf. BRIEF §7).")
            skipped.append({"asset": asset, "reason": "price mismatch on both live and offline sources"})
            continue
        daily, weekly, weekly_dates, price_source = fetched

        print(f"=== [{asset}] NsDiff generation (train-once-forward) ===")
        try:
            rows_c, rows_b = generate_nsdiff_asset(asset, daily, weekly, weekly_dates, origins_c, origins_b,
                                                   epochs, args.seed, n_samples, args.k_denoise)
        except Exception as exc:
            # One asset's data problems (e.g. TLT: offline cache ends exactly at the
            # last reused cutoff, no margin left for W+2/W+3 target dates -- see
            # module docstring) must not sink the whole run (BRIEF §7, non-negotiable).
            print(f"[{asset}] SKIPPED: {type(exc).__name__}: {exc}")
            skipped.append({"asset": asset, "reason": f"{type(exc).__name__}: {exc}", "price_source": price_source})
            continue
        all_rows_c += rows_c
        all_rows_b += rows_b
        per_asset_meta[asset] = {"price_source": price_source, "n_rows_c": len(rows_c), "n_rows_b": len(rows_b)}

    all_rows = all_rows_c + all_rows_b
    print(f"\n{len(all_rows_c)} regime-C rows + {len(all_rows_b)} regime-B rows = {len(all_rows)} total")
    if skipped:
        print(f"Skipped assets: {skipped}")

    if args.dry_run or args.smoke:
        print("\n--dry-run/--smoke: skipping DB write.")
        n_written = 0
    else:
        n_written = st.insert_oos_predictions(all_rows, db_path=args.db_path)
        print(f"\nInserted/updated {n_written} rows into {args.db_path} (source='oos', model='NsDiff', run_id={RUN_ID}).")

    summary = {
        "config": {
            "assets": assets, "epochs": epochs, "seed": args.seed, "n_samples": n_samples,
            "k_denoise": args.k_denoise, "fetch_start": FETCH_START, "fetch_end": FETCH_END,
            "mechanism": "train-once-forward (mirrors weekly_headtohead_v2.run_pair_v2)",
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
