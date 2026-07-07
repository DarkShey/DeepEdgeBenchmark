"""
generate_test_cases.py — Partie A (Kyrio) : génération des test cases de validation
============================================================================
Génère un jeu de test cases (actif x horizon x modèle) : une prédiction par
combinaison, verdictée immédiatement (intégrité + plausibilité, cf.
verdict_rules.py) et persistée via tracking_db.save_prediction() — le contrat
d'interface documenté dans BRIEF_tracking_db.md (Partie B, Maéva) : le champ
`horizon` est un entier en JOURS DE BOURSE (1 ou 7), `target_date` est la date
calendaire réelle du n-ième jour de trading après `cutoff_date`.

Mécanisme de holdout (--holdout-days, défaut 7, même logique que le paramètre
T de benchmarks/config.py) : les N derniers jours calendaires réellement
téléchargés sont mis de côté et jamais montrés aux modèles (ni pour
l'entraînement, ni pour la calibration de régime — contrainte point-in-time).
Comme ces jours sont déjà connus dans le même téléchargement, ce script peut
appeler evaluate_pending() (Partie B) tout de suite après avoir sauvegardé les
prédictions, avec un price_fetcher alimenté par les données déjà en mémoire —
aucun appel réseau supplémentaire. Avec --holdout-days 0 (mode "live"), rien
n'est encore évaluable : evaluate_pending() ne trouvera aucun target_date déjà
échu, ce qui est géré normalement (0 évaluation, pas d'erreur).

Test cases couverts par défaut :
  actifs   : BTC-USD, ETH-USD, SPY, ZN=F, TLT   (calibration/regime/assets.py — les
             5 actifs déjà utilisés par tout le reste du benchmark)
  horizons : 1 et 7 jours de bourse (BRIEF_tracking_db.md §3)
  modèles  : ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive (benchmarks/multi_horizon.py)
  -> jusqu'à 5 x 2 x 5 = 50 lignes en base par run (moins si un modèle échoue,
     ex. dépendance optionnelle absente — géré comme dans run_benchmark.py :
     un échec de modèle n'interrompt pas les autres).

tc_id volontairement auto-descriptif : "TC_<ticker>_H<horizon>" (ex.
"TC_BTC-USD_H1"). C'est ce qui rend la clé de dédoublonnage de save_prediction
(tc_id, model, cutoff_date) sûre : deux actifs différents ne peuvent jamais
partager un tc_id, donc ne peuvent jamais se faire passer pour un doublon l'un
de l'autre.

Usage (depuis DeepEdgeBenchmark/) :
    python -m validation.generate_test_cases
    python -m validation.generate_test_cases --assets "BTC-USD,SPY" --models "Naive,SARIMA"
    python -m validation.generate_test_cases --db-path validation/tracking.db
"""

import argparse
import json
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from benchmarks import config
from benchmarks.multi_horizon import MODEL_ADAPTERS
from benchmarks.regime_overlay import fit_predict_regime
from benchmarks.run_benchmark import download_full_data
from calibration.regime.assets import ASSETS

from validation import tracking_db as td
from validation import verdict_rules

# Horizons en JOURS DE BOURSE, conformes au contrat RECORD_FIELDS de
# BRIEF_tracking_db.md §3 ("horizon : int, 1 ou 7 (jours de bourse)").
HORIZONS_TRADING_DAYS = (1, 7)

DEFAULT_DB_PATH = "validation/tracking.db"
DEFAULT_RUN_DIR = "Run"


def build_price_fetcher(full_by_asset):
    """Construit un price_fetcher(asset, target_date) -> float | None conforme
    au §8 du brief, alimenté par les données déjà téléchargées cette session
    (pas d'appel réseau supplémentaire). Découplé de tracking_db.py comme
    demandé : c'est la Partie A qui fournit l'implémentation, ici en mémoire.
    Retourne None si la date n'est pas dans les données visibles ce run (le
    prochain run, avec des données plus fraîches, pourra alors la résoudre)."""
    lookup = {
        ticker: {str(ts.date()): float(close) for ts, close in full_close.items()}
        for ticker, full_close in full_by_asset.items()
    }

    def price_fetcher(asset, target_date):
        return lookup.get(asset, {}).get(target_date)

    return price_fetcher


def export_run_bundle(run_id, visible_by_asset, args, run_dir_root=DEFAULT_RUN_DIR):
    """Exporte le bundle du run dans Run/<YYYYMMDD>-run-complet/, suivant la
    convention de Run/readme.md~: "output de trainings (cf. doc de Kyrio,
    Data Readiness)" + tests. Le pipeline actuel n'a ni Parquet ni PostgreSQL
    (SQLite + CSV suffisent à cette échelle) ; on reprend la structure du bundle
    Corpus décrite dans ce doc (Training Data Set + Meta Data + .py), adaptée~:
      training_data/<ticker>.csv  -- série Close visible (post-holdout) utilisée
                                     à l'entraînement pour cet actif.
      results.csv                 -- dump de la table predictions (tracking_db.export_csv),
                                     donc déjà à jour des évaluations faites ce run.
      meta_data.json               -- paramètres du run (modèles, actifs, horizons,
                                     holdout, dates de cutoff).
      scripts/                     -- copie de tracking_db.py, verdict_rules.py,
                                     generate_test_cases.py (le ".py" du bundle).
    Les fichiers XLS de tests (smoke tests Claude + Test Cases) restent à ajouter
    manuellement dans ce même dossier, cf. Run/readme.md.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    run_folder = Path(run_dir_root) / f"{date_str}-run-complet"
    (run_folder / "training_data").mkdir(parents=True, exist_ok=True)
    (run_folder / "scripts").mkdir(parents=True, exist_ok=True)

    for ticker, series in visible_by_asset.items():
        series.rename("close").to_csv(run_folder / "training_data" / f"{ticker}.csv")

    n_rows = td.export_csv(str(run_folder / "results.csv"), db_path=args.db_path)

    meta = {
        "run_id": run_id,
        "date": date_str,
        "holdout_days": args.holdout_days,
        "db_path": args.db_path,
        "assets": sorted(visible_by_asset.keys()),
        "horizons_trading_days": list(HORIZONS_TRADING_DAYS),
        "cutoff_dates": {t: str(s.index[-1].date()) for t, s in visible_by_asset.items()},
        "n_rows_in_db_export": n_rows,
        "source": "yfinance (daily, benchmarks.run_benchmark.download_full_data)",
    }
    with open(run_folder / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    this_dir = Path(__file__).resolve().parent
    for fname in ("tracking_db.py", "verdict_rules.py", "generate_test_cases.py"):
        src = this_dir / fname
        if src.exists():
            shutil.copy(src, run_folder / "scripts" / fname)

    print(f"[generate_test_cases] bundle Run exporté -> {run_folder}")
    return run_folder


def build_records(ticker, asset_class, model_name, adapter_fn, visible_close,
                  full_close, regime_tag, run_id, epochs, seed):
    """Une prédiction par horizon (1 et 7 jours de bourse) pour (ticker, model),
    conforme au contrat RECORD_FIELDS de tracking_db.py — pas de champ
    d'évaluation ici, `evaluate_pending()` (Partie B) s'en charge après coup.

    `visible_close` = ce que le modèle a le droit de voir (post-holdout).
    `full_close` = tout ce qui a été téléchargé (peut être identique à
    `visible_close` si --holdout-days 0) ; utilisé UNIQUEMENT pour retrouver la
    date calendaire réelle du n-ième jour de bourse (target_date), jamais pour
    entraîner/calibrer — la contrainte point-in-time porte sur `visible_close`.
    """
    horizons = list(HORIZONS_TRADING_DAYS)
    if model_name == "LSTM":
        raw = adapter_fn(visible_close, horizons, epochs=epochs, seed=seed)
    else:
        raw = adapter_fn(visible_close, horizons)

    cutoff_date = visible_close.index[-1].date()
    last_close = float(visible_close.iloc[-1])
    now = datetime.now().isoformat(timespec="seconds")
    n_visible = len(visible_close)

    records = []
    for h_days in horizons:
        point, lo, hi = raw[h_days]

        future_idx = n_visible - 1 + h_days
        # target_date = date calendaire réelle du h_days-ième jour de bourse
        # après cutoff, si déjà dans les données téléchargées (holdout) ; sinon
        # calculée approximativement (jours calendaires) — provisoire, seule la
        # comparaison à un vrai jour de bourse (via evaluate_pending) compte.
        if future_idx < len(full_close):
            target_date = full_close.index[future_idx].date()
        else:
            target_date = cutoff_date + timedelta(days=h_days)

        record = {
            "run_id": run_id,
            "tc_id": f"TC_{ticker}_H{h_days}",
            "model": model_name,
            "asset": ticker,
            "horizon": h_days,
            "cutoff_date": str(cutoff_date),
            "target_date": str(target_date),
            "regime": regime_tag,
            "last_close": last_close,
            "y_pred": float(point),
            "y_lower": float(lo),
            "y_upper": float(hi),
            "created_at": now,
        }
        record["verdict_integrite"] = verdict_rules.check_integrity(record)
        # Une prédiction structurellement cassée n'est pas "plausible" non plus —
        # pas de calcul de mouvement sur des bornes déjà incohérentes.
        record["verdict_plausibilite"] = (
            verdict_rules.check_plausibility(record, asset_class, h_days)
            if record["verdict_integrite"] else 0
        )
        records.append(record)
    return records


def main():
    p = argparse.ArgumentParser(description="Test cases de base — validation business")
    p.add_argument("--assets", default=None,
                   help="tickers séparés par des virgules (défaut : les 5 de calibration/regime/assets.py)")
    p.add_argument("--models", default=None,
                   help="modèles séparés par des virgules (défaut : tous ceux de MODEL_ADAPTERS)")
    p.add_argument("--epochs", type=int, default=20, help="épochs LSTM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--holdout-days", type=int, default=7,
                   help="jours calendaires réels mis de côté avant cutoff, pour que "
                        "evaluate_pending() puisse résoudre des cas tout de suite "
                        "(0 = mode live, rien d'évaluable avant le vrai futur)")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                   help="dossier Run/ où exporter le bundle du run (cf. Run/readme.md)")
    p.add_argument("--no-run-export", action="store_true",
                   help="désactive l'export du bundle Run/ (actif par défaut)")
    args = p.parse_args()

    selected_assets = ASSETS
    if args.assets:
        wanted = {t.strip() for t in args.assets.split(",")}
        selected_assets = [a for a in ASSETS if a["ticker"] in wanted]

    selected_models = list(MODEL_ADAPTERS.items())
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        selected_models = [(n, f) for n, f in selected_models if n in wanted]

    if not selected_assets or not selected_models:
        raise SystemExit("Aucun actif ou aucun modèle sélectionné — rien à faire.")

    run_id = f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    data_end = config.DATA_END or datetime.today().strftime("%Y-%m-%d")
    print(f"[generate_test_cases] run_id={run_id}  fenêtre {config.DATA_START} -> {data_end}")

    if args.holdout_days > 0:
        print(f"[generate_test_cases] holdout={args.holdout_days}j -> cutoff décalé dans le "
              f"passé, evaluate_pending() pourra résoudre certains cas tout de suite")

    inserted = duplicates = errors = 0
    visible_by_asset = {}
    full_by_asset = {}
    for asset in selected_assets:
        ticker, asset_class = asset["ticker"], asset["asset_class"]
        print(f"\n[generate_test_cases] {ticker} : téléchargement ...")
        try:
            full_data = download_full_data(ticker, config.DATA_START, data_end)
        except SystemExit as exc:
            print(f"  ECHEC téléchargement : {exc}")
            errors += len(selected_models) * len(HORIZONS_TRADING_DAYS)
            continue

        if args.holdout_days > 0:
            cutoff_limit = full_data.index[-1] - timedelta(days=args.holdout_days)
            visible_data = full_data[full_data.index <= cutoff_limit]
        else:
            visible_data = full_data
        if visible_data.empty:
            print(f"  ECHEC : --holdout-days {args.holdout_days} trop grand, "
                  f"aucune donnée visible restante pour {ticker}")
            errors += len(selected_models) * len(HORIZONS_TRADING_DAYS)
            continue
        visible_close, full_close = visible_data["Close"], full_data["Close"]
        visible_by_asset[ticker] = visible_close
        full_by_asset[ticker] = full_close
        print(f"  {len(full_data)} jours téléchargés -> {len(visible_data)} visibles "
              f"(cutoff {visible_data.index[-1].date()})")

        print(f"[generate_test_cases] {ticker} : calibration régime ...")
        try:
            # fit_predict_regime attend le DataFrame OHLCV complet (comme
            # split["train"] dans run_benchmark.py), pas juste la colonne Close —
            # RegimeAgent.fit() indexe d'autres colonnes (ex. High/Low/Volume).
            # On ne lui montre que `visible_data` : contrainte point-in-time,
            # le holdout ne doit pas fuiter dans la calibration non plus.
            regime_state = fit_predict_regime(visible_data, visible_data.index[-1])
            regime_tag = regime_state.dominant_regime()
        except Exception as exc:
            print(f"  régime indisponible ({exc}) -> régime='unknown'")
            regime_tag = "unknown"

        for model_name, adapter_fn in selected_models:
            try:
                records = build_records(ticker, asset_class, model_name, adapter_fn,
                                        visible_close, full_close, regime_tag, run_id,
                                        args.epochs, args.seed)
            except Exception:
                print(f"  {model_name:<12} ECHEC génération prédiction :")
                traceback.print_exc()
                errors += len(HORIZONS_TRADING_DAYS)
                continue

            for record in records:
                try:
                    if td.save_prediction(record, db_path=args.db_path):
                        inserted += 1
                    else:
                        duplicates += 1
                except ValueError as exc:
                    print(f"  {model_name:<12} {record['tc_id']} REJETE : {exc}")
                    errors += 1

            detail = ", ".join(f"{r['tc_id']} pred={r['y_pred']:.2f}" for r in records)
            print(f"  {model_name:<12} ok — {len(records)} test cases traités ({detail})")

    print(f"\n[generate_test_cases] terminé -> {inserted} insérés, "
          f"{duplicates} doublons ignorés, {errors} échecs. DB : {args.db_path}")

    # Partie B : évaluation immédiate de ce qui est déjà connu (holdout), sans
    # appel réseau supplémentaire (price_fetcher alimenté par full_by_asset).
    price_fetcher = build_price_fetcher(full_by_asset)
    n_evaluated = td.evaluate_pending(price_fetcher, db_path=args.db_path)
    print(f"[generate_test_cases] evaluate_pending -> {n_evaluated} prédiction(s) évaluée(s)")

    if n_evaluated:
        print("\n[generate_test_cases] report(group_by=('model',)) :")
        for row in td.report(group_by=("model",), db_path=args.db_path):
            print(f"  {row}")

    if not args.no_run_export and visible_by_asset:
        export_run_bundle(run_id, visible_by_asset, args, run_dir_root=args.run_dir)


if __name__ == "__main__":
    main()
