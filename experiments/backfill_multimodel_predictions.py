"""
backfill_multimodel_predictions.py — load weekly_multimodel.py's results into
validation/tracking.db as source='oos' rows (BRIEF_audit_combinaisons.md step 3).

Maps regime B -> frequence='daily', horizon_type='weekly' ; regime C ->
frequence='weekly', horizon_type='weekly' -- same mapping as backfill_weekly_predictions.py
used for TSDiff. `regime` (market regime, distinct concept from B/C) is left at the
insert_oos_predictions default ('unknown') here -- run backfill_regime.py again
afterward (idempotent, only touches rows still 'unknown') rather than recomputing it
inline, and backfill_eval_metrics.py again to populate abs_error/in_interval for
these new rows.

Usage:
    python backfill_multimodel_predictions.py --in weekly_multimodel_checkpoint.json
    python backfill_multimodel_predictions.py --in weekly_multimodel_checkpoint.json --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from validation import sim_trades as st                        # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS           # noqa: E402

RUN_ID = "20260717-weekly-multimodel"
REGIME_TO_FREQUENCE = {"B": "daily", "C": "weekly"}


def rows_from_file(path: Path) -> list:
    payload = json.loads(path.read_text())
    rows = []
    for r in payload["records"]:
        week_n = int(r["horizon"][1:])   # "W1" -> 1
        rows.append({
            "run_id": RUN_ID, "model": r["model"], "asset": ASSET_TICKERS[r["asset"]],
            "horizon": week_n, "regime": "unknown",
            "cutoff_date": r["origin_date"], "target_date": r["target_date"],
            "last_close": float(r["last_close"]), "y_pred": float(r["point"]),
            "y_lower": float(r["lower"]), "y_upper": float(r["upper"]),
            "y_true": float(r["actual"]), "source": "oos",
            "frequence": REGIME_TO_FREQUENCE[r["regime"]], "horizon_type": "weekly",
            "horizon_unit": f"W+{week_n}",
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_file", required=True)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows = rows_from_file(Path(args.in_file))
    by_key = {}
    for r in rows:
        key = (r["model"], r["asset"], r["frequence"])
        by_key[key] = by_key.get(key, 0) + 1

    print(f"{len(rows)} rows from {args.in_file}")
    for (model, asset, freq), n in sorted(by_key.items()):
        print(f"  {model:<14}{asset:<10}frequence={freq:<7}n={n}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    n = st.insert_oos_predictions(rows, db_path=args.db_path)
    print(f"\nInserted/updated {n} rows into {args.db_path} (source='oos', run_id={RUN_ID}).")


if __name__ == "__main__":
    main()
