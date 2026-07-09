"""
evaluate_daily.py — résolution quotidienne des prédictions en attente (consigne tuteur)
========================================================================================
Prévu pour tourner une fois par jour (cron) : pour chaque prédiction déjà en base dont
la date cible (target_date) est atteinte ou dépassée et dont la vraie valeur n'est pas
encore connue (y_true IS NULL), télécharge le prix de marché réel de ce jour-là et le
sauve dans tracking.db (cf. tracking_db.evaluate_pending, déjà utilisé par
model_artifacts/pipeline.py juste après avoir créé de nouvelles prédictions — ce
script couvre le cas où personne ne relance le pipeline entre-temps).

Idempotent : si rien n'est échu ou que le marché n'a pas encore publié le prix (jour
férié, données pas encore à jour chez le fournisseur), ne fait rien et ne casse rien —
sûr à relancer tous les jours, y compris plusieurs fois le même jour.

Après résolution, réécrit business_validation.json dans les dossiers Run/ concernés
(cf. model_artifacts/pipeline.py, combo_dir) pour que les métriques y_true/abs_error/...
soient visibles là où on regarde déjà les résultats, sans attendre un nouveau run complet.

Usage (depuis DeepEdgeBenchmark/) :
    python -m validation.evaluate_daily
    python -m validation.evaluate_daily --db-path validation/tracking.db

Cron (tous les jours à 22h, après clôture des marchés) :
    0 22 * * * cd /chemin/vers/DeepEdgeBenchmark && .venv/bin/python -m validation.evaluate_daily >> validation/evaluate_daily.log 2>&1
"""

import argparse
from datetime import date, timedelta

from benchmarks.run_benchmark import download_full_data
from model_artifacts import pipeline as mp
from validation import tracking_db as td

DEFAULT_DB_PATH = "validation/tracking.db"
DEFAULT_LOOKBACK_DAYS = 30


def build_live_price_fetcher(assets, lookback_days=DEFAULT_LOOKBACK_DAYS):
    """Télécharge les derniers jours de cours réels pour ces actifs (vraies données de
    marché du jour, pas celles en mémoire d'un run précédent — ce script tourne
    indépendamment, potentiellement des jours après la prédiction) et construit un
    price_fetcher(asset, target_date) -> float | None conforme à evaluate_pending()."""
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = (date.today() + timedelta(days=1)).isoformat()  # borne "end" exclusive côté yfinance
    lookup = {}
    for asset in assets:
        try:
            data = download_full_data(asset, start, end)
        except SystemExit as exc:
            print(f"  {asset} : ECHEC téléchargement ({exc})")
            continue
        lookup[asset] = {str(ts.date()): float(c) for ts, c in data["Close"].items()}

    def price_fetcher(asset, target_date):
        return lookup.get(asset, {}).get(target_date)

    return price_fetcher


def main():
    p = argparse.ArgumentParser(
        description="Résout quotidiennement les prédictions en attente déjà échues (Partie B, cron)")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help="fenêtre de téléchargement pour retrouver le prix des target_date échues")
    p.add_argument("--no-run-refresh", action="store_true",
                   help="ne pas réécrire business_validation.json dans Run/ après résolution")
    p.add_argument("--run-dir", default=str(mp.RUN_ROOT),
                   help="dossier Run/ où réécrire business_validation.json (cf. Run/readme.md)")
    args = p.parse_args()

    today_iso = date.today().isoformat()
    assets = td.pending_assets(db_path=args.db_path)
    if not assets:
        print(f"[evaluate_daily] {today_iso} : rien en attente échue — rien à faire.")
        return

    print(f"[evaluate_daily] {today_iso} : {len(assets)} actif(s) à vérifier : {', '.join(assets)}")
    price_fetcher = build_live_price_fetcher(assets, lookback_days=args.lookback_days)
    n_evaluated = td.evaluate_pending(price_fetcher, db_path=args.db_path, today=today_iso)
    print(f"[evaluate_daily] {n_evaluated} prédiction(s) résolue(s)")

    if n_evaluated and not args.no_run_refresh:
        run_ids = td.run_ids_evaluated_on(today_iso, db_path=args.db_path)
        for run_id in run_ids:
            mp.export_business_validation(run_id, db_path=args.db_path, run_dir_root=args.run_dir)
        print(f"[evaluate_daily] business_validation.json rafraîchi pour {len(run_ids)} run(s) : "
              f"{', '.join(run_ids)}")


if __name__ == "__main__":
    main()
