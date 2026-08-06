"""
oos_ensemble_tsdiff_daily_weekly.py -- writes the ENSEMBLE (5 seeds 42-46 x
200 samples each, concatenated into ONE cloud of 1000) TSDiff `oos` row per
origin into `tracking.db`, replacing the existing single-seed/50 row on the
EXACT SAME origin.

MIRROR of `oos_ensemble_nsdiff_daily_weekly.py` -- same reasoning, same
non-negotiables (concatenate before quantiles, one row per origin, verify
against existing origins before upsert, reset+backfill the derived-metric
columns the upsert doesn't touch). See that script's docstring for the full
rationale; not repeated here beyond what differs (model='TSDiff').

IMPORTANT -- written but NOT executed on this machine (compute lourd chez le
tuteur). See RUNBOOK_regeneration_multiseed_200.md.

Usage (once the tutor runs it):
    python oos_ensemble_tsdiff_daily_weekly.py --dry-run   # compute + verify, no DB write
    python oos_ensemble_tsdiff_daily_weekly.py             # full run
"""

import argparse
import subprocess
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tsdiff_daily_weekly_multiseed import (                             # noqa: E402
    SEEDS, ASSETS, N_SAMPLES, load_price_data, run_seed,
)
from validation import sim_trades as st                                 # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                            # noqa: E402

ENSEMBLE_RUN_ID = "oos_ensemble_5s200_tsdiff_daily_weekly"


def row_key(r: dict) -> tuple:
    return (r["asset"], r["cutoff_date"], r["horizon"], r["frequence"])


def build_ensemble_rows(rows_by_seed: dict) -> list:
    by_key: dict = {}
    for seed in SEEDS:
        for r in rows_by_seed[seed]:
            by_key.setdefault(row_key(r), []).append(r)

    ensemble_rows = []
    for key, rs in by_key.items():
        if len(rs) != len(SEEDS):
            raise SystemExit(f"origin {key}: {len(rs)}/{len(SEEDS)} seeds present -- "
                             "incomplete checkpoints, cannot build ensemble.")
        cloud = np.concatenate([np.asarray(r["_samples"], dtype=float) for r in rs])
        expected_n = len(SEEDS) * N_SAMPLES
        if cloud.size != expected_n:
            raise SystemExit(f"origin {key}: cloud has {cloud.size} samples, expected {expected_n}.")
        ref = rs[0]
        y_pred = float(np.mean(cloud))
        y_lower, y_upper = (float(q) for q in np.quantile(cloud, [0.025, 0.975]))
        y_trues = {r["y_true"] for r in rs}
        target_dates = {r["target_date"] for r in rs}
        last_closes = {round(r["last_close"], 6) for r in rs}
        if len(y_trues) != 1 or len(target_dates) != 1 or len(last_closes) != 1:
            raise SystemExit(f"origin {key}: seeds disagree on y_true/target_date/last_close "
                             f"(should be seed-independent) -- {y_trues} {target_dates} {last_closes}")
        ensemble_rows.append({
            "run_id": ENSEMBLE_RUN_ID, "model": "TSDiff", "asset": ref["asset"],
            "horizon": ref["horizon"], "regime": "unknown",
            "cutoff_date": ref["cutoff_date"], "target_date": ref["target_date"],
            "last_close": ref["last_close"], "y_pred": y_pred, "y_lower": y_lower, "y_upper": y_upper,
            "y_true": ref["y_true"], "source": "oos",
            "frequence": ref["frequence"], "horizon_type": "weekly", "horizon_unit": ref["horizon_unit"],
        })
    return ensemble_rows


def verify_against_existing(ensemble_rows: list, db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        existing = set(con.execute(
            "SELECT asset, cutoff_date, horizon, frequence FROM predictions "
            "WHERE source='oos' AND model='TSDiff' AND horizon_type='weekly'"
        ).fetchall())
    finally:
        con.close()
    built = {row_key(r) for r in ensemble_rows}
    missing_in_db = built - existing
    missing_in_ensemble = existing - built
    if missing_in_db:
        raise SystemExit(f"{len(missing_in_db)} ensemble row(s) don't match any existing oos origin "
                         f"(would INSERT instead of REPLACE): {sorted(missing_in_db)[:5]}...")
    if missing_in_ensemble:
        raise SystemExit(f"{len(missing_in_ensemble)} existing oos origin(s) not covered by the ensemble "
                         f"(would be left stale/single-seed): {sorted(missing_in_ensemble)[:5]}...")
    print(f"Origin check OK: {len(built)} ensemble rows == {len(existing)} existing oos "
          f"(TSDiff, horizon_type='weekly') origins, exact match.")


def reset_stale_eval_metrics(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "UPDATE predictions SET abs_error=NULL, abs_error_naif=NULL, beats_naif=NULL, "
            "direction_correct=NULL, in_interval=NULL, evaluated_at=NULL "
            "WHERE source='oos' AND model='TSDiff' AND horizon_type='weekly'"
        )
        n = cur.rowcount
        con.commit()
        return n
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true", help="compute + verify, no DB write")
    args = p.parse_args()

    print("=== 1. loading verified price data + selected epochs (once, seed-independent) ===")
    prices = load_price_data()

    print(f"\n=== 2. per-seed TSDiff rows (reuses checkpoints if present, n_samples={N_SAMPLES}) ===")
    rows_by_seed = {}
    for seed in SEEDS:
        print(f"--- seed={seed} ---")
        rows_by_seed[seed] = run_seed(seed, prices)

    print("\n=== 3. building ensemble (concatenate 5x200 -> 1000 samples/origin) ===")
    ensemble_rows = build_ensemble_rows(rows_by_seed)
    print(f"{len(ensemble_rows)} ensemble origins built.")

    print("\n=== 4. verifying against existing oos origins ===")
    verify_against_existing(ensemble_rows, args.db_path)

    if args.dry_run:
        print("\n--dry-run: no DB write.")
        return

    print("\n=== 5. upsert into predictions (source='oos') ===")
    n_written = st.insert_oos_predictions(
        [{k: v for k, v in r.items() if k != "_samples"} for r in ensemble_rows],
        db_path=args.db_path,
    )
    print(f"Inserted/updated {n_written} rows (run_id={ENSEMBLE_RUN_ID}).")

    print("\n=== 6. resetting stale derived-metric columns ===")
    n_reset = reset_stale_eval_metrics(args.db_path)
    print(f"NULLed abs_error/in_interval/etc on {n_reset} rows.")

    print("\n=== 7. backfill_eval_metrics.py (recompute from the ensemble y_pred/y_lower/y_upper) ===")
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "backfill_eval_metrics.py"),
                    "--db-path", args.db_path], check=True)


if __name__ == "__main__":
    main()
