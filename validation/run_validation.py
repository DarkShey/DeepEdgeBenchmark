"""
validation/run_validation.py — Orchestration bout-en-bout Partie A + Partie B
==============================================================================
Enchaîne :
  1. init_db(db)                                  — schéma créé si absent.
  2. génération des test cases (5 actifs x 5 modèles x 2 horizons), via
     generate_test_cases.py (Partie A, Kyrio) inchangé — appelé en sous-processus
     pour ne pas interférer avec son propre argparse ni le modifier.
  3. evaluate_pending(price_fetcher.fetch, db)     — remplit y_true + métriques
     pour les prédictions déjà échues (target_date <= aujourd'hui).
  4. report(group_by=("model",)) puis report(group_by=("model","regime"))
     — tables agrégées affichées + export CSV.

Exécution (depuis DeepEdgeBenchmark/) :
    python -m validation.run_validation
"""

import subprocess
import sys
from pathlib import Path

from validation import tracking_db as td
from validation.price_fetcher import yfinance_price_fetcher

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "validation/tracking.db"


def _print_report(rows, title):
    print(f"\n=== {title} ===")
    if not rows:
        print("(aucune ligne)")
        return
    for row in rows:
        print(row)


def main():
    td.init_db(DB_PATH)
    print(f"init_db -> {DB_PATH}")

    print("\n=== Génération des test cases (5 actifs x 5 modèles x 2 horizons) ===")
    subprocess.run(
        [sys.executable, "-m", "validation.generate_test_cases", "--db-path", DB_PATH],
        cwd=str(REPO_ROOT), check=True,
    )

    n_evaluated = td.evaluate_pending(yfinance_price_fetcher, db_path=DB_PATH)
    print(f"\nevaluate_pending -> {n_evaluated} prédiction(s) évaluée(s)")

    _print_report(td.report(group_by=("model",), db_path=DB_PATH), "report(group_by=('model',))")
    _print_report(td.report(group_by=("model", "regime"), db_path=DB_PATH),
                  "report(group_by=('model', 'regime'))")

    csv_path = "validation/tracking_export.csv"
    n_csv = td.export_csv(csv_path, db_path=DB_PATH)
    print(f"\nexport_csv -> {csv_path} ({n_csv} lignes)")


if __name__ == "__main__":
    main()
