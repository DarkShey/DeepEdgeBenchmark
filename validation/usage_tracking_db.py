"""
usage_tracking_db.py — démonstration bout-en-bout de tracking_db.py (Partie B).

    init_db -> save_prediction (records d'exemple, conformes au contrat §3)
            -> evaluate_pending (price_fetcher yfinance de price_fetcher.py)
            -> report (par modèle, puis par modèle x régime)

Exécution (depuis DeepEdgeBenchmark/) :
    python -m validation.usage_tracking_db
"""

from datetime import datetime

from validation import tracking_db as td
from validation.price_fetcher import yfinance_price_fetcher

DB_PATH = "validation/tracking_demo.db"

# 3 records d'exemple conformes au contrat (§3) — réutilisables tels quels par
# Kyrio pour tester son côté (save_prediction côté Partie A).
EXAMPLE_RECORDS = [
    {
        "run_id": "demo-run-1", "tc_id": "TC1", "model": "arima", "asset": "BTC-USD",
        "horizon": 1, "cutoff_date": "2026-06-01", "target_date": "2026-06-02",
        "regime": "calm", "last_close": 68000.0, "y_pred": 68500.0,
        "y_lower": 65000.0, "y_upper": 72000.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": datetime.now().isoformat(),
    },
    {
        "run_id": "demo-run-1", "tc_id": "TC2", "model": "sarima", "asset": "SPY",
        "horizon": 1, "cutoff_date": "2026-06-01", "target_date": "2026-06-02",
        "regime": "calm", "last_close": 580.0, "y_pred": 581.5,
        "y_lower": 574.0, "y_upper": 589.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": datetime.now().isoformat(),
    },
    {
        "run_id": "demo-run-1", "tc_id": "TC3", "model": "lstm", "asset": "GC=F",
        "horizon": 7, "cutoff_date": "2026-06-01", "target_date": "2026-06-10",
        "regime": "unknown", "last_close": 2350.0, "y_pred": 2400.0,
        "y_lower": 2280.0, "y_upper": 2500.0,
        "verdict_integrite": 1, "verdict_plausibilite": 0,
        "created_at": datetime.now().isoformat(),
    },
]


def main():
    td.init_db(DB_PATH)
    print(f"init_db -> {DB_PATH}")

    for record in EXAMPLE_RECORDS:
        inserted = td.save_prediction(record, db_path=DB_PATH)
        print(f"  save_prediction({record['tc_id']}, {record['model']}) -> inserted={inserted}")

    n_evaluated = td.evaluate_pending(yfinance_price_fetcher, db_path=DB_PATH)
    print(f"\nevaluate_pending -> {n_evaluated} prédiction(s) évaluée(s)")

    print("\n=== report(group_by=('model',)) ===")
    for row in td.report(group_by=("model",), db_path=DB_PATH):
        print(row)

    print("\n=== report(group_by=('model', 'regime')) ===")
    for row in td.report(group_by=("model", "regime"), db_path=DB_PATH):
        print(row)

    n_csv = td.export_csv("validation/tracking_demo_export.csv", db_path=DB_PATH)
    print(f"\nexport_csv -> tracking_demo_export.csv ({n_csv} lignes)")


if __name__ == "__main__":
    main()
