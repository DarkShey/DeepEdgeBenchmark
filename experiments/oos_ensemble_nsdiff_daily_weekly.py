"""
oos_ensemble_nsdiff_daily_weekly.py -- writes the ENSEMBLE (5 seeds 42-46 x 200
samples each, concatenated into ONE cloud of 1000) NsDiff `oos` row per origin
into `tracking.db`, replacing the existing single-seed/50 row on the EXACT
SAME origin (BRIEF_dashboard_multiseed_200.md §4, tâche 6). One line per
origin -- never one line per seed (brief §9: that would double-count in
`build_daily_weekly_pairs`/pooling).

Reuses, does not reimplement:
  - `nsdiff_daily_weekly_multiseed.{load_price_data, run_seed, SEEDS, ASSETS,
    N_SAMPLES}` -- same checkpoints (n_samples=200, collect_samples=True) the
    CV-table/badge script already produces; running that script first (or
    letting this one trigger the same `run_seed`) means NO duplicate NsDiff
    fit -- checkpoints are shared.
  - `validation.sim_trades.insert_oos_predictions` -- same idempotent upsert
    every other oos backfill script uses (`ON CONFLICT ... WHERE source='oos'`,
    key excludes seed/n_samples, so this REPLACES the single-seed row).
  - `backfill_eval_metrics.py` -- run as a subprocess afterward.

Non-negotiable (brief §9): CONCATENATE the 5 raw clouds (1000 samples) before
reading mean/quantiles -- never average 5 already-computed bounds. Point =
np.mean(cloud), bounds = np.quantile(cloud, [0.025, 0.975]) -- same convention
as every existing oos row (oos_nsdiff_daily_weekly.py:287,304), just on 1000
samples instead of 50.

Gotcha found reading `validation/sim_trades.py:insert_oos_predictions`: its
`ON CONFLICT ... DO UPDATE SET` does NOT touch abs_error/in_interval/
abs_error_naif/beats_naif/direction_correct/evaluated_at -- those stay at
their STALE single-seed values after the upsert. `backfill_eval_metrics.py`
only fills rows `WHERE abs_error IS NULL`, so it would silently SKIP every row
we just updated (all already non-NULL from the single-seed generation). This
script explicitly NULLs those columns for exactly the rows it touches (scoped
`model='NsDiff' AND horizon_type='weekly' AND source='oos'`, verified in step
3 to be exactly the ensemble's coverage) right after the upsert, then runs
backfill_eval_metrics.py so it actually recomputes them from the new
y_pred/y_lower/y_upper.

Usage:
    python oos_ensemble_nsdiff_daily_weekly.py --dry-run   # compute + verify, no DB write
    python oos_ensemble_nsdiff_daily_weekly.py             # full run
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

from nsdiff_daily_weekly_multiseed import (                             # noqa: E402
    SEEDS, ASSETS, N_SAMPLES, load_price_data, run_seed,
)
from validation import sim_trades as st                                 # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                            # noqa: E402

ENSEMBLE_RUN_ID = "oos_ensemble_5s200_nsdiff_daily_weekly"


def row_key(r: dict) -> tuple:
    return (r["asset"], r["cutoff_date"], r["horizon"], r["frequence"])


def build_ensemble_rows(rows_by_seed: dict) -> list:
    """`rows_by_seed`: {seed: [row, ...]} (each row from generate_nsdiff_asset,
    collect_samples=True, i.e. carrying "_samples"). Groups across seeds by
    origin (asset, cutoff_date, horizon, frequence), concatenates the 5 raw
    clouds, computes ensemble point/bounds -- one output row per origin."""
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
            "run_id": ENSEMBLE_RUN_ID, "model": "NsDiff", "asset": ref["asset"],
            "horizon": ref["horizon"], "regime": "unknown",
            "cutoff_date": ref["cutoff_date"], "target_date": ref["target_date"],
            "last_close": ref["last_close"], "y_pred": y_pred, "y_lower": y_lower, "y_upper": y_upper,
            "y_true": ref["y_true"], "source": "oos",
            "frequence": ref["frequence"], "horizon_type": "weekly", "horizon_unit": ref["horizon_unit"],
        })
    return ensemble_rows


def verify_against_existing(ensemble_rows: list, db_path: str) -> None:
    """Non-négociable brief §6 : chaque ligne ensemble doit tomber EXACTEMENT
    sur une origine (model, asset, frequence, horizon_type, horizon_unit,
    cutoff_date) déjà présente dans `oos` -- ni ligne nouvelle, ni origine
    orpheline."""
    con = sqlite3.connect(db_path)
    try:
        existing = set(con.execute(
            "SELECT asset, cutoff_date, horizon, frequence FROM predictions "
            "WHERE source='oos' AND model='NsDiff' AND horizon_type='weekly'"
        ).fetchall())
    finally:
        con.close()
    built = {row_key(r) for r in ensemble_rows}
    missing_in_db = built - existing        # ensemble row with no matching existing origin
    missing_in_ensemble = existing - built  # existing origin the ensemble didn't cover
    if missing_in_db:
        raise SystemExit(f"{len(missing_in_db)} ensemble row(s) don't match any existing oos origin "
                         f"(would INSERT instead of REPLACE): {sorted(missing_in_db)[:5]}...")
    if missing_in_ensemble:
        raise SystemExit(f"{len(missing_in_ensemble)} existing oos origin(s) not covered by the ensemble "
                         f"(would be left stale/single-seed): {sorted(missing_in_ensemble)[:5]}...")
    print(f"Origin check OK: {len(built)} ensemble rows == {len(existing)} existing oos "
          f"(NsDiff, horizon_type='weekly') origins, exact match.")


def reset_stale_eval_metrics(db_path: str) -> int:
    """Cf. docstring du module : l'upsert ne touche pas abs_error & co, donc on
    les remet à NULL pour exactement le périmètre que l'ensemble couvre
    (vérifié 1:1 par verify_against_existing juste avant), afin que
    backfill_eval_metrics.py (WHERE abs_error IS NULL) les recalcule vraiment."""
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "UPDATE predictions SET abs_error=NULL, abs_error_naif=NULL, beats_naif=NULL, "
            "direction_correct=NULL, in_interval=NULL, evaluated_at=NULL "
            "WHERE source='oos' AND model='NsDiff' AND horizon_type='weekly'"
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

    print("=== 1. loading verified price data (once, seed-independent) ===")
    prices = load_price_data()

    print(f"\n=== 2. per-seed NsDiff rows (reuses checkpoints if present, n_samples={N_SAMPLES}) ===")
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

    print("\n=== 6. resetting stale derived-metric columns (see module docstring) ===")
    n_reset = reset_stale_eval_metrics(args.db_path)
    print(f"NULLed abs_error/in_interval/etc on {n_reset} rows.")

    print("\n=== 7. backfill_eval_metrics.py (recompute from the ensemble y_pred/y_lower/y_upper) ===")
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "backfill_eval_metrics.py"),
                    "--db-path", args.db_path], check=True)


if __name__ == "__main__":
    main()
