"""
oos_nsdiff_tlt.py -- TLT-only extension of oos_nsdiff_daily_weekly.py
(BRIEF_nsdiff_ameliorer_limites.md Fix 1). Adds the 5th asset NsDiff was
missing (`NOTE_nsdiff_dashboard_daily_oos.md` §3: offline cache ends
2026-07-02, no margin for the W+2/W+3 target dates of the last origins).

Root cause (diagnosed, not re-litigated here): yfinance `auto_adjust=True`
back-adjusts historical Close for dividends RELATIVE TO THE FETCH TIME -- a
fresh fetch today re-adjusts old dates using dividends paid since the
original baseline fetch, so old cutoffs drift (~86.59 vs baseline's stored
86.941032 at 2024-10-18, ~0.4% -- confirmed empirically). Raw close
(`auto_adjust=False`) doesn't have that drift (time-invariant), but is on
the WRONG SCALE for old dates (93.87 vs 86.94 at 2024-10-18, ~8%, dividend
yield accumulated over the whole history) -- so neither "just refetch" mode
passes `_verify_origin_prices` (tol=1e-6) on its own, confirmed by testing
both against the 90 reused TLT origins.

What DOES pass (verified below, not asserted): the existing offline cache
(`experiments/offline_prices.py`, `DONNEE~1.XLS`) matches the stored
baseline `last_close`/`y_true` EXACTLY through its last date (2026-07-02:
87.038544 @ 2026-06-26, 86.941032 @ 2024-10-18, etc.) -- it captures the
SAME adjustment snapshot the baselines were originally fetched with, it
just stops 3 weeks short of the last target_date (~2026-07-23). For dates
close to that original fetch time, the auto-adjustment factor was already
~1.0 (no known future dividends yet to adjust for at fetch time) -- so raw
close for 2026-07-03..2026-07-24 (time-invariant, safe to fetch NOW even
weeks later) reproduces the same convention. Verified directly: raw close
today for 2026-07-10/17/23 (84.470001 / 84.519997 / 83.169998) matches the
stored `y_true` at those target_dates EXACTLY.

So the patched source here is: offline cache for [FETCH_START, 2026-07-03)
+ live raw close (auto_adjust=False) for [2026-07-03, FETCH_END) --
verified against ALL 90 reused TLT origins (both regimes) via the SAME
`_verify_origin_prices` guard-rail already used by
`oos_nsdiff_daily_weekly.fetch_verified`, imported, not reimplemented. If
this guard ever fails (e.g. rerun after a real TLT dividend lands in that
window and the repo's `DONNEE~1.XLS` isn't updated), this script raises and
does NOT write to the DB -- no invented price, per BRIEF Fix 1 point 3.

Everything else (NsDiff generation, epochs/seed/n_samples/k_denoise,
insertion) is `oos_nsdiff_daily_weekly.generate_nsdiff_asset` /
`validation.sim_trades.insert_oos_predictions`, imported and called as-is --
mirrors the other 4 assets' config exactly, no new budget invented.

Usage:
    python oos_nsdiff_tlt.py --dry-run     # verify + compute, no DB write
    python oos_nsdiff_tlt.py               # full run, DB write
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from weekly_headtohead import build_weekly                              # noqa: E402
from backtest_rolling_tsdiffw import (                                  # noqa: E402
    DB_PATH, load_baseline_triplets, FETCH_START, FETCH_END,
)
from offline_prices import fetch_data_offline                           # noqa: E402
from oos_nsdiff_daily_weekly import (                                   # noqa: E402
    load_baseline_triplets_daily, _verify_origin_prices, OriginMismatchError,
    generate_nsdiff_asset, DEFAULT_SEED, DEFAULT_N_SAMPLES, DEFAULT_K_DENOISE,
)
from weekly_nsdiff_production import NSDIFF_EPOCHS_W as NSDIFF_EPOCHS   # noqa: E402
from validation import sim_trades as st                                 # noqa: E402

TICKER = "TLT"
OFFLINE_CUTOVER = "2026-07-03"   # offline_prices.py's TLT sheet covers up to (excl.) this date
RUN_ID = "20260804-nsdiff-daily-weekly-oos-tlt"


def fetch_tlt_patched(ticker: str, start: str, end: str) -> pd.Series:
    assert ticker == TICKER
    offline = fetch_data_offline(ticker, start, OFFLINE_CUTOVER)
    raw = yf.download(ticker, start=OFFLINE_CUTOVER, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise SystemExit(f"No live data for {ticker} between {OFFLINE_CUTOVER} and {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    combined = pd.concat([offline, close.astype(float)]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=str(ROOT / "experiments" / "oos_nsdiff_tlt_summary.json"))
    args = p.parse_args()

    t_start = time.time()
    print("=== 1. baseline origins (verbatim) ===")
    origins_c = load_baseline_triplets([TICKER], db_path=args.db_path)
    origins_b = load_baseline_triplets_daily([TICKER], db_path=args.db_path)
    print(f"regime C: {len(origins_c)} triplets | regime B: {len(origins_b)} triplets "
          f"({origins_c['cutoff_date'].nunique()} / {origins_b['cutoff_date'].nunique()} unique origins)")
    if origins_c.empty or origins_b.empty:
        raise SystemExit("No baseline origins for TLT on one side -- nothing to mirror.")

    print("\n=== 2. patched price source (offline < 2026-07-03 + live raw close >= 2026-07-03) ===")
    daily = fetch_tlt_patched(TICKER, FETCH_START, FETCH_END)
    print(f"  combined daily series: {daily.index.min().date()} -> {daily.index.max().date()} ({len(daily)} obs)")
    weekly, weekly_dates = build_weekly(daily)

    print("=== 3. verification vs stored baseline last_close (tol=1e-6 relative, ALL reused origins) ===")
    result = {"ticker": TICKER, "n_origins_c": int(origins_c["cutoff_date"].nunique()),
              "n_origins_b": int(origins_b["cutoff_date"].nunique())}
    try:
        _verify_origin_prices(weekly, weekly_dates, origins_c, TICKER)
        _verify_origin_prices(weekly, weekly_dates, origins_b, TICKER)
    except OriginMismatchError as exc:
        print(f"VERIFICATION FAILED: {exc}")
        print("TLT stays excluded -- no price invented (BRIEF Fix 1 point 3). Nothing written to the DB.")
        result.update({"status": "skipped", "reason": str(exc)})
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        return
    print(f"  OK -- patched source matches stored last_close at all "
          f"{origins_c['cutoff_date'].nunique() + origins_b['cutoff_date'].nunique()} baseline checks.")

    print("\n=== 4. NsDiff generation (train-once-forward, mirrors the other 4 assets) ===")
    rows_c, rows_b = generate_nsdiff_asset(
        TICKER, daily, weekly, weekly_dates, origins_c, origins_b,
        args.epochs, args.seed, args.n_samples, args.k_denoise,
    )
    for row in rows_c + rows_b:
        row["run_id"] = RUN_ID
    print(f"  {len(rows_c)} regime-C rows + {len(rows_b)} regime-B rows")

    if args.dry_run:
        print("\n--dry-run: skipping DB write.")
        n_written = 0
    else:
        n_written = st.insert_oos_predictions(rows_c + rows_b, db_path=args.db_path)
        print(f"\nInserted/updated {n_written} rows into {args.db_path} "
              f"(source='oos', model='NsDiff', asset='TLT', run_id={RUN_ID}).")

    result.update({
        "status": "written" if not args.dry_run else "dry_run_ok",
        "price_source": f"offline<{OFFLINE_CUTOVER} + live_raw_close>={OFFLINE_CUTOVER}",
        "config": {"epochs": args.epochs, "seed": args.seed, "n_samples": args.n_samples,
                   "k_denoise": args.k_denoise, "fetch_start": FETCH_START, "fetch_end": FETCH_END},
        "n_rows_regime_c": len(rows_c), "n_rows_regime_b": len(rows_b), "n_written": n_written,
        "elapsed_s": round(time.time() - t_start, 1),
    })
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"Summary -> {args.out}")
    print(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
