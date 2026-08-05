"""
oos_nsdiff_monthly.py -- NsDiff at MONTHLY horizon (M+1/M+2/M+3), mirroring
`oos_nsdiff_daily_weekly.py` one level up the horizon ladder. Closes
BRIEF_nsdiff_mensuel_M1M2M3.md.

=== READ THIS FIRST: power warning (brief §0) ===
~2015 -> today is only ~130-140 monthly closes (vs ~570 weeks for the weekly
side). After reserving a train block and a small validation block, the
number of TEST origins is a few dozen at best, and M+1/M+2/M+3 targets
overlap (block-bootstrap correlated) -> `effective_n` of order ~10, not the
~30 the weekly side gets with n_test=90/block_length=3. Expect most cellule
verdicts to be "indistinguishable" -- that is the EXPECTED, honest outcome of
low power, not evidence the two regimes perform identically. Never present a
monthly "indistinguishable" as demonstrated equivalence (brief §0, §9).

Two regimes, same target (dernier jour de bourse du mois, i.e. the actual
last TRADED day of the calendar month -- never a month still in progress):
  - Regime B (daily -> month-end): NsDiff fit on DAILY returns, forecast read
    off at the REAL trading-day distance to the month-end target
    (`month_targets`, exact mirror of `epoch_sweep.week_targets`).
    frequence='daily', horizon_type='monthly'.
  - Regime C (monthly-native): NsDiff fit DIRECTLY on monthly returns
    (`nsdiff_model.fit_nsdiff`/`forecast_from_fitted` -- frequency-agnostic,
    verified: they only ever see a `pd.Series` + a horizon int, nothing
    daily/weekly-specific), horizon=3, M+1/M+2/M+3 in one shot.
    frequence='monthly', horizon_type='monthly'.

Origins: UNLIKE the weekly side, no model has ANY monthly oos rows in
tracking.db to reuse -- there is no baseline triplet set to read verbatim
(brief §2). Origins are GENERATED here via `three_way_split_monthly` (exact
mirror of `epoch_sweep.three_way_split`, MONTH_MARGIN replacing WEEK_MARGIN)
on each asset's own monthly series. Documented per-asset in the run summary
(train_end/val_origins/test_origins), not silently chosen.

Because origins are self-generated (not baseline-matched), the elaborate
baseline-last_close verification dance `oos_nsdiff_daily_weekly.fetch_verified`
does for TLT/yfinance drift is NOT needed here: there is no stored baseline
price to reproduce bit-for-bit. `fetch_prices` below just needs ONE
internally-consistent price series per asset (regime B and regime C both
read off the SAME `daily`/`monthly` arrays from that single fetch) -- live
yfinance first, `experiments/offline_prices.fetch_data_offline` fallback on
failure (never mixed within one asset's run).

Budgets (brief §6, declared):
  - epochs=40 (`weekly_nsdiff_production.NSDIFF_EPOCHS_W`, same constant the
    daily/weekly side already reuses -- no separate monthly budget invented).
  - k_denoise=20 (`nsdiff_model.K_DENOISE` default, unchanged).
  - n_samples=50 (matches the daily/weekly side's own `DEFAULT_N_SAMPLES`).
  - seq_len: regime B (daily) uses `nsdiff_model.SEQ_LEN`=30 UNCHANGED
    (default, not overridden -- same daily lookback convention already used
    for both regimes on the weekly side, brief precedent). Regime C
    (monthly) uses `SEQ_LEN_MONTHLY=18` (~1.5y of monthly returns) -- a NEW,
    declared choice: reusing 30 (the daily/weekly-side default) would eat 30
    of our ~130-140 total monthly points before a single test origin, which
    the weekly side can afford (30 of ~570 weeks) but the monthly side
    cannot (brief §6: "seq_len mensuel ~12-24 justifié"). 18 sits mid-range:
    long enough that `sigma_kernel=8` (must be < seq_len) still has margin,
    short enough to leave most of the history for train+test origins.
  - `model_d.horizon` (regime B's diffusion generation length) is NOT a
    fixed guess -- it is computed PER ASSET as the actual max real
    trading-day distance to the M+3 target across that asset's test
    origins (crypto trades 7d/week -> needs ~90 vs ~65 for
    business-day-only assets over 3 months; hard-coding one constant for
    both would either truncate crypto or waste equity/bond training). See
    `generate_nsdiff_asset_monthly`. `forecast_from_fitted` SILENTLY
    truncates `hh = min(h, model.horizon)` (nsdiff_model.py) if this is
    under-sized -- computing the exact value from real origins instead of
    guessing avoids that trap outright, not just guards against it.

Origin split (brief §2, "viser le max d'origines de test possible tout en
gardant un train décent"): MONTH_MARGIN=3 (mirrors WEEK_MARGIN), a small
reserved-but-UNUSED validation block N_VAL_MONTHLY=6 (documented per the
brief's "train/validation/test" ask -- there is no monthly epoch sweep here,
epochs=40 is a declared constant, so validation is reserved for symmetry
with the three-way-split convention and future use, not consumed), and
N_TEST_MONTHLY=40 test origins by default (checked empirically: all 5 assets
support this -- SPY/BTC/ZN=F/TLT have 139 complete months, ETH-USD has 105
(launched 2017-11) -- leaving >=55 months of train for every asset, well
above `seq_len_monthly + horizon = 21`).

Insertion: `validation.sim_trades.insert_oos_predictions`, reused unmodified
-- same idempotent upsert as every other oos backfill script. `horizon_unit`
is ALWAYS passed explicitly as 'M+1'/'M+2'/'M+3' in every row (never left for
`insert_oos_predictions`'s own fallback `f"{'W' if horizon_type=='weekly'
else 'D'}+{horizon}"` to compute -- that fallback has no 'M' case and would
silently mislabel every monthly row 'D+1'/'D+2'/'D+3'). Also note (documented,
not fixed, out of scope, same posture as `oos_nsdiff_daily_weekly.py`'s own
`daily_duplicate` caveat): `tracking_db.flag_daily_duplicates()` partitions
by `(source, model, asset, horizon, cutoff_date)` -- NOT `frequence`/
`horizon_type` -- so IF that correction script is ever run, a monthly
cutoff_date that happens to coincide with an existing weekly cutoff_date for
the same asset/horizon could be mis-flagged. Nothing in this pipeline calls
`flag_daily_duplicates()`, so daily/weekly rows are never touched by running
this script (verified by row-count before/after, see NOTE).

Usage:
    python oos_nsdiff_monthly.py --smoke                  # 1 asset, tiny epochs/samples -- plumbing only
    python oos_nsdiff_monthly.py --dry-run                # compute + verify + print, no DB writes
    python oos_nsdiff_monthly.py                          # full run, all 5 assets, DB write, seed=42
    python oos_nsdiff_monthly.py --assets SPY BTC-USD      # subset
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

import nsdiff_model as nm                                               # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS                   # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                            # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W as NSDIFF_EPOCHS   # noqa: E402
from offline_prices import fetch_data_offline                           # noqa: E402
from validation import sim_trades as st                                 # noqa: E402

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 50                # mirrors oos_nsdiff_daily_weekly.DEFAULT_N_SAMPLES
DEFAULT_K_DENOISE = nm.K_DENOISE      # 20, unchanged
SEQ_LEN_MONTHLY = 18                  # declared, ~1.5y -- see module docstring
HORIZON_MONTHLY = 3                   # M+1/M+2/M+3, exact (mirrors HORIZON_WEEKLY=3)
MONTH_MARGIN = 3                      # monthly points reserved past the last origin (mirrors WEEK_MARGIN)
N_VAL_MONTHLY = 6                     # reserved, unused (no monthly epoch sweep -- epochs=40 declared)
N_TEST_MONTHLY = 40                   # default test-origin count (brief §2: "max possible, train décent")
MONTHLY_HORIZON_UNITS = {1: "M+1", 2: "M+2", 3: "M+3"}
RUN_ID = "20260805-nsdiff-monthly-oos"

FETCH_START = "2015-01-01"


# ── 1. monthly series (mirror of weekly_headtohead.build_weekly) ────────────

def build_monthly(daily: pd.Series):
    """Calendar-month-end close series (`.last()` within each month), plus,
    aligned 1:1, the ACTUAL trading date realising that close. Mirrors
    `build_weekly`'s W-FRI convention with 'ME' (month-end); like W-FRI, the
    bin label is the calendar month-end (not necessarily a trading day), so
    the label must never be used to index the daily series directly -- use
    `monthly_dates` for that.

    The current, still-in-progress calendar month is ALWAYS dropped (brief
    §0/§9: "jamais un mois partiel en cours") -- `.resample("ME").last()`
    would otherwise happily return today's last available close labelled
    under the current month's end, which is not that month's true close."""
    monthly = daily.resample("ME").last().dropna()
    monthly_dates = daily.index.to_series().resample("ME").last().dropna()
    if len(monthly) and monthly.index[-1].to_period("M") == pd.Timestamp.today().to_period("M"):
        monthly = monthly.iloc[:-1]
        monthly_dates = monthly_dates.iloc[:-1]
    return monthly, monthly_dates


def month_targets(monthly_dates: pd.Series, daily: pd.Series, m: int):
    """Exact mirror of `epoch_sweep.week_targets`, one rung up: for
    monthly-origin position m, (origin_date, daily_pos, target_dates[3],
    daily_horizons[3]) -- target_dates are the actual trading days realising
    M+1/M+2/M+3, daily_horizons their trading-day distance from the origin."""
    origin_date = monthly_dates.iloc[m]
    daily_pos = int(daily.index.get_loc(origin_date))
    target_dates = [monthly_dates.iloc[m + h] for h in (1, 2, 3)]
    daily_horizons = [int(daily.index.get_loc(d) - daily_pos) for d in target_dates]
    return origin_date, daily_pos, target_dates, daily_horizons


def three_way_split_monthly(monthly: pd.Series, n_val: int, n_test: int):
    """Exact mirror of `epoch_sweep.three_way_split`: positions into
    `monthly` -- (train_end_pos, val_origins_pos, test_origins_pos), three
    disjoint chronologically-ordered blocks, MONTH_MARGIN monthly points
    reserved past the last test origin for its M1-M3 targets."""
    n_m = len(monthly)
    needed = n_val + n_test + MONTH_MARGIN
    if n_m < needed + 1:
        raise ValueError(f"only {n_m} monthly points available, need >= {needed + 1} "
                          f"for {n_val} validation + {n_test} test origins with a "
                          f"{MONTH_MARGIN}-month margin.")
    train_end_pos = n_m - needed - 1
    val_origins_pos = list(range(train_end_pos + 1, train_end_pos + 1 + n_val))
    test_origins_pos = list(range(val_origins_pos[-1] + 1, val_origins_pos[-1] + 1 + n_test))
    return train_end_pos, val_origins_pos, test_origins_pos


# ── 2. price fetch -- single consistent source per asset, no baseline to match ──

def fetch_prices(ticker: str, start: str, end: str):
    """Live yfinance first, `offline_prices.fetch_data_offline` fallback on
    failure. Unlike `oos_nsdiff_daily_weekly.fetch_verified`, there is no
    stored baseline `last_close` to reproduce bit-for-bit (origins are
    self-generated here, brief §2) -- so no cross-source verification is
    needed, only ONE source used consistently for a given asset's whole run
    (both regimes read off the same `daily`/`monthly` arrays)."""
    try:
        daily = nm.fetch_data(ticker, start, end)
        return daily, "live/yfinance"
    except Exception as exc:
        print(f"[{ticker}] live/yfinance fetch raised {type(exc).__name__}: {exc} -- "
              f"falling back to offline/DONNEE~1.XLS")
        daily = fetch_data_offline(ticker, start, end)
        return daily, "offline/DONNEE~1.XLS"


# ── 3. NsDiff generation (train-once-forward, mirrors generate_nsdiff_asset) ──

def _standardized_returns(prices: pd.Series, mu: float, sd: float) -> np.ndarray:
    r = nm._log_returns(prices.values.astype(float))
    return (r - mu) / sd


def generate_nsdiff_asset_monthly(asset: str, daily: pd.Series, monthly: pd.Series,
                                  monthly_dates: pd.Series, test_pos: list, epochs: int,
                                  seed: int, n_samples: int, k_denoise: int,
                                  seq_len_monthly: int = SEQ_LEN_MONTHLY) -> tuple:
    if not test_pos:
        raise ValueError(f"[{asset}] no test origins.")

    targets = [month_targets(monthly_dates, daily, m) for m in test_pos]
    max_daily_h = max(dh[-1] for (_, _, _, dh) in targets)   # M+3 daily-horizon, always the largest

    earliest_pos = test_pos[0]
    T0_date = monthly_dates.iloc[earliest_pos]
    T0_daily_pos = int(daily.index.get_loc(T0_date))
    train_monthly = monthly.iloc[:earliest_pos]      # strictly before the earliest test origin
    train_daily = daily.iloc[:T0_daily_pos]           # strictly before the earliest test origin
    print(f"[{asset}] train-once: monthly {len(train_monthly)} obs (< {T0_date.date()}), "
          f"daily {len(train_daily)} obs (< {T0_date.date()}) | {len(test_pos)} test origins "
          f"{monthly_dates.iloc[test_pos[0]].date()} -> {monthly_dates.iloc[test_pos[-1]].date()} | "
          f"model_d.horizon (regime B, real max M+3 trading-day distance) = {max_daily_h}")

    t0 = time.time()
    nm.set_seed(seed)
    model_c, mu_c, sd_c = nm.fit_nsdiff(train_monthly, seq_len=seq_len_monthly,
                                        horizon=HORIZON_MONTHLY, epochs=epochs, k_denoise=k_denoise)
    print(f"[{asset}] NsDiff-M (regime C) fitted in {time.time() - t0:.0f}s (mu={mu_c:.6f}, sd={sd_c:.6f})")

    t0 = time.time()
    nm.set_seed(seed)
    model_d, mu_d, sd_d = nm.fit_nsdiff(train_daily, horizon=max_daily_h, epochs=epochs, k_denoise=k_denoise)
    print(f"[{asset}] NsDiff-D (regime B) fitted in {time.time() - t0:.0f}s (mu={mu_d:.6f}, sd={sd_d:.6f})")

    monthly_z = _standardized_returns(monthly, mu_c, sd_c)
    daily_z = _standardized_returns(daily, mu_d, sd_d)

    rows_c, rows_b = [], []
    for k, m in enumerate(test_pos):
        origin_date, daily_pos, target_dates, daily_horizons = targets[k]
        cutoff_date = str(origin_date.date())
        last_price = float(monthly.iloc[m])   # SAME origin price for both regimes (brief §1)

        nm.set_seed(seed + k)
        samples_c = nm.forecast_from_fitted(model_c, monthly_z[:m], mu_c, sd_c, last_price,
                                            horizons=[1, 2, 3], n_samples=n_samples)
        nm.set_seed(seed + k)
        samples_b = nm.forecast_from_fitted(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price,
                                            horizons=daily_horizons, n_samples=n_samples)

        for h in (1, 2, 3):
            wi = h - 1
            target_date = target_dates[wi]
            h_d = daily_horizons[wi]
            target_date_str = str(target_date.date())
            y_true = float(monthly.iloc[m + h])   # SAME target close for both regimes

            s_c = samples_c[h]
            point_c = float(np.mean(s_c))
            lo_c, hi_c = (float(q) for q in np.quantile(s_c, [0.025, 0.975]))
            rows_c.append({
                "run_id": RUN_ID, "model": "NsDiff", "asset": asset, "horizon": h,
                "regime": "unknown", "cutoff_date": cutoff_date, "target_date": target_date_str,
                "last_close": last_price, "y_pred": point_c, "y_lower": lo_c, "y_upper": hi_c,
                "y_true": y_true, "source": "oos",
                "frequence": "monthly", "horizon_type": "monthly", "horizon_unit": MONTHLY_HORIZON_UNITS[h],
            })

            s_b = samples_b[h_d]
            point_b = float(np.mean(s_b))
            lo_b, hi_b = (float(q) for q in np.quantile(s_b, [0.025, 0.975]))
            rows_b.append({
                "run_id": RUN_ID, "model": "NsDiff", "asset": asset, "horizon": h,
                "regime": "unknown", "cutoff_date": cutoff_date, "target_date": target_date_str,
                "last_close": last_price, "y_pred": point_b, "y_lower": lo_b, "y_upper": hi_b,
                "y_true": y_true, "source": "oos",
                "frequence": "daily", "horizon_type": "monthly", "horizon_unit": MONTHLY_HORIZON_UNITS[h],
            })
        if (k + 1) % 10 == 0 or k == len(test_pos) - 1:
            print(f"[{asset}] origin {k + 1}/{len(test_pos)} ({cutoff_date}) done")

    meta = {"max_daily_horizon": max_daily_h, "train_monthly_len": len(train_monthly),
            "train_daily_len": len(train_daily), "n_test_origins": len(test_pos),
            "test_origin_range": [str(monthly_dates.iloc[test_pos[0]].date()),
                                  str(monthly_dates.iloc[test_pos[-1]].date())]}
    return rows_c, rows_b, meta


# ── 4. main ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()),
                   help="tickers, e.g. SPY BTC-USD (default: all 5)")
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--seq-len-monthly", type=int, default=SEQ_LEN_MONTHLY)
    p.add_argument("--n-val", type=int, default=N_VAL_MONTHLY)
    p.add_argument("--n-test", type=int, default=N_TEST_MONTHLY)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true", help="compute + verify + print, no DB writes")
    p.add_argument("--smoke", action="store_true",
                   help="tiny plumbing check: 1 asset (SPY), epochs=2, n-samples=6, n-test=5 -- overrides other flags")
    p.add_argument("--out", default=str(ROOT / "experiments" / "oos_nsdiff_monthly_summary.json"))
    args = p.parse_args()

    assets = ["SPY"] if args.smoke else args.assets
    epochs = 2 if args.smoke else args.epochs
    n_samples = 6 if args.smoke else args.n_samples
    n_test = 5 if args.smoke else args.n_test
    n_val = 2 if args.smoke else args.n_val
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    t_start = time.time()
    all_rows_c, all_rows_b, skipped, per_asset_meta = [], [], [], {}
    for asset in assets:
        print(f"\n=== [{asset}] price fetch ===")
        try:
            daily, price_source = fetch_prices(asset, FETCH_START, end)
        except Exception as exc:
            print(f"[{asset}] SKIPPED: both live and offline price fetch failed: {type(exc).__name__}: {exc}")
            skipped.append({"asset": asset, "reason": f"price fetch failed: {exc}"})
            continue
        monthly, monthly_dates = build_monthly(daily)

        try:
            train_end_pos, val_pos, test_pos = three_way_split_monthly(monthly, n_val, n_test)
        except ValueError as exc:
            print(f"[{asset}] SKIPPED: {exc}")
            skipped.append({"asset": asset, "reason": str(exc)})
            continue
        print(f"[{asset}] split: train <= {monthly_dates.iloc[train_end_pos].date()} "
              f"({train_end_pos + 1} months) | validation {monthly_dates.iloc[val_pos[0]].date()} -> "
              f"{monthly_dates.iloc[val_pos[-1]].date()} ({len(val_pos)}, reserved/unused) | "
              f"test {monthly_dates.iloc[test_pos[0]].date()} -> {monthly_dates.iloc[test_pos[-1]].date()} "
              f"({len(test_pos)})")

        try:
            rows_c, rows_b, meta = generate_nsdiff_asset_monthly(
                asset, daily, monthly, monthly_dates, test_pos, epochs, args.seed,
                n_samples, args.k_denoise, args.seq_len_monthly)
        except Exception as exc:
            print(f"[{asset}] SKIPPED: {type(exc).__name__}: {exc}")
            skipped.append({"asset": asset, "reason": f"{type(exc).__name__}: {exc}", "price_source": price_source})
            continue
        all_rows_c += rows_c
        all_rows_b += rows_b
        per_asset_meta[asset] = {
            "price_source": price_source, "n_rows_c": len(rows_c), "n_rows_b": len(rows_b),
            "train_end": str(monthly_dates.iloc[train_end_pos].date()),
            "val_origins": [str(monthly_dates.iloc[i].date()) for i in val_pos],
            "n_test_origins": len(test_pos), **meta,
        }

    all_rows = all_rows_c + all_rows_b
    print(f"\n{len(all_rows_c)} regime-C rows + {len(all_rows_b)} regime-B rows = {len(all_rows)} total")
    if skipped:
        print(f"Skipped assets: {skipped}")

    if args.dry_run or args.smoke:
        print("\n--dry-run/--smoke: skipping DB write.")
        n_written = 0
    else:
        n_written = st.insert_oos_predictions(all_rows, db_path=args.db_path)
        print(f"\nInserted/updated {n_written} rows into {args.db_path} "
              f"(source='oos', model='NsDiff', frequence in daily/monthly, run_id={RUN_ID}).")

    summary = {
        "config": {
            "assets": assets, "epochs": epochs, "seed": args.seed, "n_samples": n_samples,
            "k_denoise": args.k_denoise, "seq_len_monthly": args.seq_len_monthly,
            "n_val": n_val, "n_test": n_test, "month_margin": MONTH_MARGIN,
            "fetch_start": FETCH_START, "fetch_end": end,
            "mechanism": "train-once-forward (mirrors oos_nsdiff_daily_weekly.generate_nsdiff_asset)",
            "origins": "self-generated via three_way_split_monthly (no baseline monthly oos rows exist, brief §2)",
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
