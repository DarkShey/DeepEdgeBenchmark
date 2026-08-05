"""
oos_tsdiff_monthly.py -- TSDiff at MONTHLY horizon (M+1/M+2/M+3), mirroring
`oos_nsdiff_monthly.py` (swap `nsdiff_model` -> `tsdiff_model`), which itself
mirrors `oos_nsdiff_daily_weekly.py` one rung up the horizon ladder. Closes
the "optionnel / next" item of BRIEF_nsdiff_mensuel_M1M2M3.md §4 ("ajouter
TSDiff mensuel pour la comparaison diffusion-vs-diffusion").

Reuses, UNMODIFIED, from `oos_nsdiff_monthly.py` (100% model-agnostic: pure
date/position bookkeeping, no `nsdiff_model` call inside any of them):
`build_monthly`, `month_targets`, `three_way_split_monthly`,
`MONTHLY_HORIZON_UNITS`, `MONTH_MARGIN`, `N_VAL_MONTHLY`, `N_TEST_MONTHLY`,
`SEQ_LEN_MONTHLY`, `HORIZON_MONTHLY`, `FETCH_START`. See that module's own
docstring for the reasoning behind each of these (power warning, split
sizing, seq_len choice) -- identical here, not repeated.

What's genuinely different from the NsDiff version (`tsdiff_model` API, not
a copy-paste error):
  - `fit_tsdiff`'s `T` parameter is `T_DIFFUSION` (training diffusion steps,
    default 1000) -- NOT the DDIM inference step count. Left at its default,
    never overridden here (unlike NsDiff, where `k_denoise` plays double
    duty as both training AND ancestral-sampling step count -- TSDiff
    decouples the two, cf. `tsdiff_model.py`'s own module docstring).
  - `forecast_from_fitted` takes explicit `k_denoise` (DDIM steps, default
    `tsdiff_model.K_DENOISE`=20) AND `ddim_eta` (default
    `tsdiff_model.DDIM_ETA`=1.0, "needed for real PI" per that module's own
    comment) -- both passed through explicitly below, never left to a
    silently-ignored kwarg (unlike NsDiff's `forecast_from_fitted`, which
    absorbs a stray `k_denoise` via `**_ignored` since ancestral sampling
    always uses every step).
  - Epochs: no separate monthly TSDiff budget exists, and this is
    deliberately NOT re-running a fresh per-asset epoch-sweep (that's what
    `weekly_headtohead_v2.py` does for the WEEKLY TSDiff oos rows, via
    `epoch_sweep_results.json` -- out of scope for this "optional/next"
    addition). `epochs=40` reused: it is `tsdiff_model.EPOCHS`'s own module
    default (same numeric value the NsDiff monthly side already uses,
    coincidentally -- not copied from there, independently the model's own
    declared default).

Usage: identical CLI shape to `oos_nsdiff_monthly.py`.
    python oos_tsdiff_monthly.py --smoke
    python oos_tsdiff_monthly.py --dry-run
    python oos_tsdiff_monthly.py
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

import tsdiff_model as td                                               # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS                   # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                            # noqa: E402
from offline_prices import fetch_data_offline                           # noqa: E402
from validation import sim_trades as st                                 # noqa: E402
from oos_nsdiff_monthly import (                                        # noqa: E402
    build_monthly, month_targets, three_way_split_monthly,
    MONTHLY_HORIZON_UNITS, MONTH_MARGIN, N_VAL_MONTHLY, N_TEST_MONTHLY,
    SEQ_LEN_MONTHLY, HORIZON_MONTHLY, FETCH_START,
)

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 50
DEFAULT_K_DENOISE = td.K_DENOISE   # 20, DDIM inference steps
DEFAULT_DDIM_ETA = td.DDIM_ETA     # 1.0, "needed for real PI" (tsdiff_model.py)
DEFAULT_EPOCHS = td.EPOCHS         # 40, tsdiff_model's own module default
RUN_ID = "20260805-tsdiff-monthly-oos"


def fetch_prices(ticker: str, start: str, end: str):
    """TSDiff-namespace mirror of `oos_nsdiff_monthly.fetch_prices` (same
    live-then-offline-fallback logic, `td.fetch_data` instead of
    `nm.fetch_data` -- the two are copied-verbatim-identical implementations
    per `nsdiff_model.py`'s own docstring, but kept in the calling model's
    own namespace here rather than cross-importing, for clarity)."""
    try:
        daily = td.fetch_data(ticker, start, end)
        return daily, "live/yfinance"
    except Exception as exc:
        print(f"[{ticker}] live/yfinance fetch raised {type(exc).__name__}: {exc} -- "
              f"falling back to offline/DONNEE~1.XLS")
        daily = fetch_data_offline(ticker, start, end)
        return daily, "offline/DONNEE~1.XLS"


def _standardized_returns(prices: pd.Series, mu: float, sd: float) -> np.ndarray:
    r = td._log_returns(prices.values.astype(float))
    return (r - mu) / sd


def generate_tsdiff_asset_monthly(asset: str, daily: pd.Series, monthly: pd.Series,
                                  monthly_dates: pd.Series, test_pos: list, epochs: int,
                                  seed: int, n_samples: int, k_denoise: int, ddim_eta: float,
                                  seq_len_monthly: int = SEQ_LEN_MONTHLY) -> tuple:
    """Exact mirror of `oos_nsdiff_monthly.generate_nsdiff_asset_monthly`,
    `tsdiff_model` calls instead of `nsdiff_model`."""
    if not test_pos:
        raise ValueError(f"[{asset}] no test origins.")

    targets = [month_targets(monthly_dates, daily, m) for m in test_pos]
    max_daily_h = max(dh[-1] for (_, _, _, dh) in targets)

    earliest_pos = test_pos[0]
    T0_date = monthly_dates.iloc[earliest_pos]
    T0_daily_pos = int(daily.index.get_loc(T0_date))
    train_monthly = monthly.iloc[:earliest_pos]
    train_daily = daily.iloc[:T0_daily_pos]
    print(f"[{asset}] train-once: monthly {len(train_monthly)} obs (< {T0_date.date()}), "
          f"daily {len(train_daily)} obs (< {T0_date.date()}) | {len(test_pos)} test origins "
          f"{monthly_dates.iloc[test_pos[0]].date()} -> {monthly_dates.iloc[test_pos[-1]].date()} | "
          f"model_d.horizon (regime B, real max M+3 trading-day distance) = {max_daily_h}")

    t0 = time.time()
    td.set_seed(seed)
    model_c, mu_c, sd_c = td.fit_tsdiff(train_monthly, seq_len=seq_len_monthly,
                                        horizon=HORIZON_MONTHLY, epochs=epochs)
    print(f"[{asset}] TSDiff-M (regime C) fitted in {time.time() - t0:.0f}s (mu={mu_c:.6f}, sd={sd_c:.6f})")

    t0 = time.time()
    td.set_seed(seed)
    model_d, mu_d, sd_d = td.fit_tsdiff(train_daily, horizon=max_daily_h, epochs=epochs)
    print(f"[{asset}] TSDiff-D (regime B) fitted in {time.time() - t0:.0f}s (mu={mu_d:.6f}, sd={sd_d:.6f})")

    monthly_z = _standardized_returns(monthly, mu_c, sd_c)
    daily_z = _standardized_returns(daily, mu_d, sd_d)

    rows_c, rows_b = [], []
    for k, m in enumerate(test_pos):
        origin_date, daily_pos, target_dates, daily_horizons = targets[k]
        cutoff_date = str(origin_date.date())
        last_price = float(monthly.iloc[m])

        td.set_seed(seed + k)
        samples_c = td.forecast_from_fitted(model_c, monthly_z[:m], mu_c, sd_c, last_price,
                                            horizons=[1, 2, 3], n_samples=n_samples,
                                            k_denoise=k_denoise, ddim_eta=ddim_eta)
        td.set_seed(seed + k)
        samples_b = td.forecast_from_fitted(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price,
                                            horizons=daily_horizons, n_samples=n_samples,
                                            k_denoise=k_denoise, ddim_eta=ddim_eta)

        for h in (1, 2, 3):
            wi = h - 1
            target_date = target_dates[wi]
            h_d = daily_horizons[wi]
            target_date_str = str(target_date.date())
            y_true = float(monthly.iloc[m + h])

            s_c = samples_c[h]
            point_c = float(np.mean(s_c))
            lo_c, hi_c = (float(q) for q in np.quantile(s_c, [0.025, 0.975]))
            rows_c.append({
                "run_id": RUN_ID, "model": "TSDiff", "asset": asset, "horizon": h,
                "regime": "unknown", "cutoff_date": cutoff_date, "target_date": target_date_str,
                "last_close": last_price, "y_pred": point_c, "y_lower": lo_c, "y_upper": hi_c,
                "y_true": y_true, "source": "oos",
                "frequence": "monthly", "horizon_type": "monthly", "horizon_unit": MONTHLY_HORIZON_UNITS[h],
            })

            s_b = samples_b[h_d]
            point_b = float(np.mean(s_b))
            lo_b, hi_b = (float(q) for q in np.quantile(s_b, [0.025, 0.975]))
            rows_b.append({
                "run_id": RUN_ID, "model": "TSDiff", "asset": asset, "horizon": h,
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--ddim-eta", type=float, default=DEFAULT_DDIM_ETA)
    p.add_argument("--seq-len-monthly", type=int, default=SEQ_LEN_MONTHLY)
    p.add_argument("--n-val", type=int, default=N_VAL_MONTHLY)
    p.add_argument("--n-test", type=int, default=N_TEST_MONTHLY)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=str(ROOT / "experiments" / "oos_tsdiff_monthly_summary.json"))
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
              f"({train_end_pos + 1} months) | validation ({len(val_pos)}, reserved/unused) | "
              f"test {monthly_dates.iloc[test_pos[0]].date()} -> {monthly_dates.iloc[test_pos[-1]].date()} "
              f"({len(test_pos)})")

        try:
            rows_c, rows_b, meta = generate_tsdiff_asset_monthly(
                asset, daily, monthly, monthly_dates, test_pos, epochs, args.seed,
                n_samples, args.k_denoise, args.ddim_eta, args.seq_len_monthly)
        except Exception as exc:
            print(f"[{asset}] SKIPPED: {type(exc).__name__}: {exc}")
            skipped.append({"asset": asset, "reason": f"{type(exc).__name__}: {exc}", "price_source": price_source})
            continue
        all_rows_c += rows_c
        all_rows_b += rows_b
        per_asset_meta[asset] = {
            "price_source": price_source, "n_rows_c": len(rows_c), "n_rows_b": len(rows_b),
            "train_end": str(monthly_dates.iloc[train_end_pos].date()),
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
              f"(source='oos', model='TSDiff', frequence in daily/monthly, run_id={RUN_ID}).")

    summary = {
        "config": {
            "assets": assets, "epochs": epochs, "seed": args.seed, "n_samples": n_samples,
            "k_denoise": args.k_denoise, "ddim_eta": args.ddim_eta, "seq_len_monthly": args.seq_len_monthly,
            "n_val": n_val, "n_test": n_test, "month_margin": MONTH_MARGIN,
            "fetch_start": FETCH_START, "fetch_end": end,
            "mechanism": "train-once-forward (mirrors oos_nsdiff_monthly.generate_nsdiff_asset_monthly)",
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
