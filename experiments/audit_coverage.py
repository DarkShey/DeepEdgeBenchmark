"""
audit_coverage.py — coverage matrix (model x frequence x horizon_type x horizon_unit
x asset), BRIEF_audit_combinaisons.md step 1. Confirms by REAL query (not assumption)
which cells of the full configuration space have been tested and which are empty.

Scope: source='oos' only (systematic historical backtest, same evaluation protocol
for all models/assets). 'live' rows (488, ongoing production forecasts, small/recent
sample) are a different evaluation protocol and excluded from this audit.

Also runs the consistency checks demanded by the brief (§0/§5a): no horizon_unit
inconsistent with its horizon_type (e.g. a 'W+2' row labeled horizon_type='daily'),
and reports the frequence/horizon_type value sets actually observed.

Usage:
    python audit_coverage.py
    python audit_coverage.py --db-path /path/to/tracking.db --out audit_coverage.json
"""

import argparse
import json
import sqlite3
from itertools import product
from pathlib import Path

MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive", "TSDiff")
ASSETS = ("BTC-USD", "ETH-USD", "SPY", "ZN=F", "TLT")
# (frequence, horizon_type, horizon_unit) -- the only combinations that make sense
# (brief §2): regime A (daily native), B (daily->weekly), C (weekly native).
REGIME_CELLS = (
    ("daily", "daily", "D+1"), ("daily", "daily", "D+7"),                 # regime A
    ("daily", "weekly", "W+1"), ("daily", "weekly", "W+2"), ("daily", "weekly", "W+3"),   # regime B
    ("weekly", "weekly", "W+1"), ("weekly", "weekly", "W+2"), ("weekly", "weekly", "W+3"),  # regime C
)


def check_consistency(conn) -> dict:
    bad_unit = conn.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE (horizon_type='daily' AND horizon_unit NOT LIKE 'D+%')
           OR (horizon_type='weekly' AND horizon_unit NOT LIKE 'W+%')
    """).fetchone()[0]
    freq_values = [r[0] for r in conn.execute("SELECT DISTINCT frequence FROM predictions")]
    horizon_type_values = [r[0] for r in conn.execute("SELECT DISTINCT horizon_type FROM predictions")]
    dupe_oos = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT source, model, asset, horizon, frequence, horizon_type, cutoff_date, COUNT(*) c
            FROM predictions WHERE source='oos'
            GROUP BY source, model, asset, horizon, frequence, horizon_type, cutoff_date
            HAVING c > 1
        )
    """).fetchone()[0]
    return {
        "n_rows_with_horizon_unit_inconsistent_with_horizon_type": bad_unit,
        "distinct_frequence_values": sorted(freq_values),
        "distinct_horizon_type_values": sorted(horizon_type_values),
        "n_duplicate_oos_business_keys": dupe_oos,
    }


def build_matrix(conn) -> list:
    rows = conn.execute("""
        SELECT model, frequence, horizon_type, horizon_unit, asset,
               COUNT(*) AS n_predictions,
               SUM(CASE WHEN y_true IS NOT NULL THEN 1 ELSE 0 END) AS n_evaluated
        FROM predictions
        WHERE source = 'oos'
        GROUP BY model, frequence, horizon_type, horizon_unit, asset
    """).fetchall()
    observed = {(r["model"], r["frequence"], r["horizon_type"], r["horizon_unit"], r["asset"]):
               (r["n_predictions"], r["n_evaluated"]) for r in rows}

    matrix = []
    for model, (frequence, horizon_type, horizon_unit), asset in product(MODELS, REGIME_CELLS, ASSETS):
        n_pred, n_eval = observed.get((model, frequence, horizon_type, horizon_unit, asset), (0, 0))
        matrix.append({
            "model": model, "frequence": frequence, "horizon_type": horizon_type,
            "horizon_unit": horizon_unit, "asset": asset,
            "n_predictions": n_pred, "n_evaluated": n_eval, "empty": n_pred == 0,
        })
    return matrix


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(Path(__file__).resolve().parent.parent
                                           / "validation" / "tracking.db"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "audit_coverage.json"))
    args = p.parse_args()

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    try:
        consistency = check_consistency(conn)
        matrix = build_matrix(conn)
    finally:
        conn.close()

    n_total_cells = len(matrix)
    n_empty = sum(1 for c in matrix if c["empty"])
    n_filled = n_total_cells - n_empty

    payload = {
        "scope": "source='oos' only (systematic historical backtest)",
        "dimensions": {"models": list(MODELS), "assets": list(ASSETS),
                      "regime_cells (frequence, horizon_type, horizon_unit)": list(REGIME_CELLS)},
        "consistency_checks": consistency,
        "coverage_summary": {"n_total_cells": n_total_cells, "n_filled": n_filled, "n_empty": n_empty},
        "matrix": matrix,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"Consistency checks: {json.dumps(consistency, indent=2)}")
    print(f"\nCoverage: {n_filled}/{n_total_cells} cells filled, {n_empty} empty.\n")

    print(f"{'Model':<14}{'Regime':<26}", end="")
    for a in ASSETS:
        print(f"{a:<10}", end="")
    print()
    print("-" * (14 + 26 + 10 * len(ASSETS)))
    for model in MODELS:
        for frequence, horizon_type, horizon_unit in REGIME_CELLS:
            regime_label = f"{frequence}/{horizon_type}/{horizon_unit}"
            print(f"{model:<14}{regime_label:<26}", end="")
            for asset in ASSETS:
                cell = next(c for c in matrix if c["model"] == model and c["frequence"] == frequence
                           and c["horizon_type"] == horizon_type and c["horizon_unit"] == horizon_unit
                           and c["asset"] == asset)
                mark = "." if cell["empty"] else str(cell["n_predictions"])
                print(f"{mark:<10}", end="")
            print()

    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
