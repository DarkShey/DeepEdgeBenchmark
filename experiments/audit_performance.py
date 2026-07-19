"""
audit_performance.py — performance table for every non-empty cell of the coverage
matrix (BRIEF_audit_combinaisons.md step 2): RMSE, interval coverage (target ~0.95),
directional accuracy, skill vs naive. Sorted to surface the top combos per asset and
per asset class.

Metrics are computed ON THE FLY from y_true/y_pred/y_lower/y_upper/last_close, NOT
from the stored abs_error/in_interval/beats_naif/direction_correct columns -- those
are NULL for every source='oos' row (audit finding: only evaluate_pending(), scoped
to source='live', ever populates them; the historical OOS/backtest ingestion path
never has). This applies uniformly to the pre-existing daily OOS data and this
brief's new weekly backfill, so all cells are computed the same way.

Regime segmentation (brief's "segmentable par régime de marché") is NOT available
for source='oos': `regime` is 'unknown' for all 4801 OOS rows (both old and new),
only the 488 'live' rows carry a real regime label. Noted, not fixed here (out of
this brief's scope, cf. "pas de recalibration des modèles existants").

Usage:
    python audit_performance.py
"""

import argparse
import json
import sqlite3
from pathlib import Path

ASSET_CLASS = {
    "BTC-USD": "crypto", "ETH-USD": "crypto", "SPY": "index", "ZN=F": "bond", "TLT": "bond",
}


def build_performance_table(conn) -> list:
    rows = conn.execute("""
        SELECT model, frequence, horizon_type, horizon_unit, asset,
               COUNT(*) AS n,
               SUM(CASE WHEN y_true IS NOT NULL THEN 1 ELSE 0 END) AS n_evaluated,
               AVG(CASE WHEN y_true IS NOT NULL THEN (y_pred - y_true) * (y_pred - y_true) END) AS mse,
               AVG(CASE WHEN y_true IS NOT NULL THEN ABS(y_pred - y_true) END) AS mae,
               AVG(CASE WHEN y_true IS NOT NULL
                        THEN CASE WHEN y_true BETWEEN y_lower AND y_upper THEN 1.0 ELSE 0.0 END END) AS coverage,
               AVG(CASE WHEN y_true IS NOT NULL AND y_pred != last_close AND y_true != last_close
                        THEN CASE WHEN (y_pred - last_close) * (y_true - last_close) > 0 THEN 1.0 ELSE 0.0 END END) AS dir_acc,
               AVG(CASE WHEN y_true IS NOT NULL
                        THEN CASE WHEN ABS(y_pred - y_true) <= ABS(last_close - y_true) THEN 1.0 ELSE 0.0 END END) AS beats_naif_rate
        FROM predictions
        WHERE source = 'oos'
        GROUP BY model, frequence, horizon_type, horizon_unit, asset
        HAVING n_evaluated > 0
    """).fetchall()

    table = []
    for r in rows:
        rmse = r["mse"] ** 0.5 if r["mse"] is not None else None
        table.append({
            "model": r["model"], "frequence": r["frequence"], "horizon_type": r["horizon_type"],
            "horizon_unit": r["horizon_unit"], "asset": asset_full_and_class(r["asset"])[0],
            "asset_class": asset_full_and_class(r["asset"])[1],
            "n": r["n"], "n_evaluated": r["n_evaluated"],
            "RMSE": round(rmse, 6) if rmse is not None else None,
            "MAE": round(r["mae"], 6) if r["mae"] is not None else None,
            "Cov95": round(r["coverage"], 4) if r["coverage"] is not None else None,
            "DirAcc": round(r["dir_acc"], 4) if r["dir_acc"] is not None else None,
            "BeatsNaifRate": round(r["beats_naif_rate"], 4) if r["beats_naif_rate"] is not None else None,
            # "good combo" per brief §4: precise AND well-calibrated, not sur-confident
            "well_calibrated": bool(r["coverage"] is not None and 0.85 <= r["coverage"] <= 1.0),
        })
    return table


def asset_full_and_class(asset: str):
    return asset, ASSET_CLASS.get(asset, "unknown")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(Path(__file__).resolve().parent.parent
                                           / "validation" / "tracking.db"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "audit_performance.json"))
    args = p.parse_args()

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    try:
        table = build_performance_table(conn)
    finally:
        conn.close()

    payload = {
        "scope": "source='oos' only, metrics computed on the fly (see module docstring)",
        "regime_segmentation": "unavailable for OOS data -- regime='unknown' on all rows",
        "n_cells": len(table),
        "table": table,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    # top combo per asset (sorted by RMSE, tie-break by calibration)
    print("=== Top combo per asset (by RMSE, among well-calibrated i.e. Cov95 in [0.85,1.0]) ===")
    for asset in sorted({r["asset"] for r in table}):
        candidates = [r for r in table if r["asset"] == asset and r["well_calibrated"]]
        pool = candidates or [r for r in table if r["asset"] == asset]
        best = min(pool, key=lambda r: r["RMSE"])
        flag = "" if candidates else "  (aucune combo bien calibree, meilleure RMSE brute)"
        print(f"  {asset:<10} {best['model']:<14} {best['frequence']}/{best['horizon_type']}/"
              f"{best['horizon_unit']:<5} RMSE={best['RMSE']:.4f} Cov95={best['Cov95']} "
              f"DirAcc={best['DirAcc']}{flag}")

    print("\n=== Top combo per asset class ===")
    for cls in sorted({r["asset_class"] for r in table}):
        candidates = [r for r in table if r["asset_class"] == cls and r["well_calibrated"]]
        pool = candidates or [r for r in table if r["asset_class"] == cls]
        # RMSE not comparable across assets of different price scale -> rank by Cov95 proximity to .95, then RMSE within
        best = min(pool, key=lambda r: (abs((r["Cov95"] or 0) - 0.95), r["RMSE"]))
        print(f"  {cls:<8} {best['model']:<14} {best['asset']:<10} "
              f"{best['frequence']}/{best['horizon_type']}/{best['horizon_unit']:<5} "
              f"RMSE={best['RMSE']:.4f} Cov95={best['Cov95']} DirAcc={best['DirAcc']}")

    print(f"\nSaved -> {args.out} ({len(table)} non-empty cells)")


if __name__ == "__main__":
    main()
