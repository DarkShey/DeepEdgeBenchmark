"""
backfill_eval_metrics.py — compute and persist abs_error/abs_error_naif/beats_naif/
direction_correct/in_interval/evaluated_at for existing OOS rows (currently NULL for
all 4801 of them: only tracking_db.evaluate_pending(), scoped to source='live', ever
populates these columns; the historical OOS/backtest ingestion path never has).
Backfilled so future queries don't need to recompute these on the fly (as
audit_performance.py currently does) and so 'oos' and 'live' rows are homogeneous.

direction_correct uses the EXACT same convention as tracking_db._sign()/
evaluate_pending() (sign(y_pred-last_close) == sign(y_true-last_close), sign(0)=0)
-- not the stricter "product > 0" used in audit_performance.py's exploratory query,
so this persisted column means the same thing for 'oos' and 'live' rows.

`--source` (défaut 'oos', chemin historique inchangé) : la piste à backfiller.
Le chantier R2 du BRIEF_nsdiff_regeneration_oos_et_famille3.md écrit la grille
régénérée sous une piste distincte pour ne pas mélanger ancienne et nouvelle
grille, et a besoin des mêmes colonnes dérivées dessus.

Usage:
    python backfill_eval_metrics.py
    python backfill_eval_metrics.py --dry-run
    python backfill_eval_metrics.py --source oos2020
"""

import argparse
import sqlite3
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source", default="oos", help="piste à backfiller (défaut : oos)")
    args = p.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        n_target = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source=? AND y_true IS NOT NULL "
            "AND abs_error IS NULL", (args.source,)
        ).fetchone()[0]
        print(f"Rows to backfill (source={args.source!r}, y_true known, abs_error still NULL): "
              f"{n_target}")

        if args.dry_run:
            print("--dry-run: nothing written.")
            return

        today = date.today().isoformat()
        cur = conn.execute("""
            UPDATE predictions
            SET
                in_interval = CASE WHEN y_true BETWEEN y_lower AND y_upper THEN 1 ELSE 0 END,
                abs_error = ABS(y_pred - y_true),
                abs_error_naif = ABS(last_close - y_true),
                beats_naif = CASE WHEN ABS(y_pred - y_true) <= ABS(last_close - y_true)
                                  THEN 1 ELSE 0 END,
                direction_correct = CASE
                    WHEN (CASE WHEN y_pred > last_close THEN 1
                              WHEN y_pred < last_close THEN -1 ELSE 0 END)
                       = (CASE WHEN y_true > last_close THEN 1
                              WHEN y_true < last_close THEN -1 ELSE 0 END)
                    THEN 1 ELSE 0 END,
                evaluated_at = ?
            WHERE source = ? AND y_true IS NOT NULL AND abs_error IS NULL
        """, (today, args.source))
        n_updated = cur.rowcount
        conn.commit()
        print(f"Updated {n_updated} rows.")

        remaining = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source=? AND y_true IS NOT NULL "
            "AND abs_error IS NULL", (args.source,)
        ).fetchone()[0]
        print(f"Remaining unbackfilled (source={args.source!r}, y_true known): {remaining}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
