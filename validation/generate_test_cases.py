"""
generate_test_cases.py — Test cases de base pour la validation business
============================================================================
Génère un jeu de test cases (actif x horizon x modèle) : une prédiction par
combinaison, verdictée immédiatement (intégrité + plausibilité, cf.
verdict_rules.py) et persistée via tracking_db.save_prediction().

Mécanisme de holdout (--holdout-days, défaut 7, même logique que le paramètre
T de benchmarks/config.py) : les N derniers jours calendaires réellement
téléchargés sont mis de côté et jamais montrés aux modèles (ni pour
l'entraînement, ni pour la calibration de régime — contrainte point-in-time).
Comme ces jours sont déjà connus dans le même téléchargement, la vraie valeur
(`actual`) est immédiatement disponible et est renseignée tout de suite dans
le record — un fait brut, pas un verdict de précision. Avec --holdout-days 0
(mode "live", ancien comportement par défaut), `actual` reste NULL en
attendant que le vrai futur arrive ; c'est alors evaluate_pending()/report()
(gérés ailleurs, hors scope de ce script) qui le renseignent et calculent les
taux agrégés.

Test cases couverts par défaut :
  actifs   : BTC-USD, ETH-USD, SPY, ZN=F, TLT   (calibration/regime/assets.py — les
             5 actifs déjà utilisés par tout le reste du benchmark)
  horizons : D+1 (1 jour de trading suivant), D+7 (~1 semaine calendaire,
             assimilée à 5 jours de trading — même longueur que S+1 dans
             benchmarks/config.py)
  modèles  : ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive (benchmarks/multi_horizon.py)
  -> jusqu'à 5 x 2 x 5 = 50 lignes en base par run (moins si un modèle échoue,
     ex. dépendance optionnelle absente — géré comme dans run_benchmark.py :
     un échec de modèle n'interrompt pas les autres).

tc_id volontairement auto-descriptif : "TC_<ticker>_D<horizon>" (ex.
"TC_BTC-USD_D1"). C'est ce qui rend la clé de dédoublonnage de save_prediction
(tc_id, model, cutoff_date) sûre : deux actifs différents ne peuvent jamais
partager un tc_id, donc ne peuvent jamais se faire passer pour un doublon l'un
de l'autre. Voir la note dans tracking_db.py.

Usage (depuis DeepEdgeBenchmark/) :
    python -m validation.generate_test_cases
    python -m validation.generate_test_cases --assets "BTC-USD,SPY" --models "Naive,SARIMA"
    python -m validation.generate_test_cases --db-path validation/tracking.db
"""

import argparse
import csv
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

# D+n (jours calendaires "métier", ceux demandés pour la validation business)
# -> horizon en jours de TRADING attendu par les adaptateurs de
# benchmarks/multi_horizon.py. D+1 = le jour de trading suivant. D+7 = ~1
# semaine calendaire ≈ 5 jours de trading (weekend exclu) : le target_date
# stocké reste en jours calendaires (cutoff + 7j), seul l'appel au modèle
# utilise l'équivalent trading-day.
D_TO_TRADING_DAYS = {1: 1, 7: 5}

DEFAULT_DB_PATH = "validation/tracking.db"
DEFAULT_RUN_DIR = "Run"


def export_run_bundle(run_id, all_records, visible_by_asset, args, run_dir_root=DEFAULT_RUN_DIR):
    """Exporte le bundle du run dans Run/<YYYYMMDD>-run-complet/, suivant la
    convention de Run/readme.md~: "output de trainings (cf. doc de Kyrio,
    Data Readiness)" + tests. Le pipeline actuel n'a ni Parquet ni PostgreSQL
    (SQLite + CSV suffisent à cette échelle) ; on reprend la structure du bundle
    Corpus décrite dans ce doc (Training Data Set + Meta Data + .py), adaptée~:
      training_data/<ticker>.csv  -- série Close visible (post-holdout) utilisée
                                     à l'entraînement pour cet actif.
      results.csv                 -- toutes les prédictions générées ce run.
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

    if all_records:
        with open(run_folder / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

    meta = {
        "run_id": run_id,
        "date": date_str,
        "holdout_days": args.holdout_days,
        "db_path": args.db_path,
        "assets": sorted(visible_by_asset.keys()),
        "models": sorted({r["model"] for r in all_records}),
        "horizons_D": sorted({r["horizon"] for r in all_records}),
        "cutoff_dates": {t: str(s.index[-1].date()) for t, s in visible_by_asset.items()},
        "n_test_cases": len(all_records),
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
    """Une prédiction par horizon D pour (ticker, model), déjà verdictée.

    `visible_close` = ce que le modèle a le droit de voir (post-holdout).
    `full_close` = tout ce qui a été téléchargé (peut être identique à
    `visible_close` si --holdout-days 0) ; utilisé UNIQUEMENT pour relire la
    vraie valeur déjà connue, jamais pour entraîner/calibrer — la contrainte
    point-in-time porte sur `visible_close`, pas sur la lecture de l'actual.
    Le lookup se fait par décalage en jours de TRADING (comme le fait déjà
    benchmarks/run_benchmark.py::compute_verdict), pas par date calendaire,
    pour rester cohérent avec l'horizon réellement utilisé par le modèle.
    """
    trading_days = sorted(set(D_TO_TRADING_DAYS.values()))
    if model_name == "LSTM":
        raw = adapter_fn(visible_close, trading_days, epochs=epochs, seed=seed)
    else:
        raw = adapter_fn(visible_close, trading_days)

    cutoff_date = visible_close.index[-1].date()
    last_close = float(visible_close.iloc[-1])
    now = datetime.now().isoformat(timespec="seconds")
    n_visible = len(visible_close)

    records = []
    for d, h_days in D_TO_TRADING_DAYS.items():
        point, lo, hi = raw[h_days]
        target_date = cutoff_date + timedelta(days=d)

        future_idx = n_visible - 1 + h_days
        if future_idx < len(full_close):
            actual = float(full_close.iloc[future_idx])
            evaluated_at = now
        else:
            actual = None
            evaluated_at = None

        record = {
            "run_id": run_id,
            "tc_id": f"TC_{ticker}_D{d}",
            "model": model_name,
            "asset": ticker,
            "horizon": d,
            "cutoff_date": str(cutoff_date),
            "target_date": str(target_date),
            "regime": regime_tag,
            "last_close": last_close,
            "y_pred": float(point),
            "y_lower": float(lo),
            "y_upper": float(hi),
            "created_at": now,
            "actual": actual,
            "evaluated_at": evaluated_at,
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
                        "target_date soit déjà connu (0 = mode live, actual reste NULL)")
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
              f"passé, actual rempli immédiatement quand déjà connu")

    inserted = duplicates = errors = 0
    all_records = []
    visible_by_asset = {}
    for asset in selected_assets:
        ticker, asset_class = asset["ticker"], asset["asset_class"]
        print(f"\n[generate_test_cases] {ticker} : téléchargement ...")
        try:
            full_data = download_full_data(ticker, config.DATA_START, data_end)
        except SystemExit as exc:
            print(f"  ECHEC téléchargement : {exc}")
            errors += len(selected_models) * len(D_TO_TRADING_DAYS)
            continue

        if args.holdout_days > 0:
            cutoff_limit = full_data.index[-1] - timedelta(days=args.holdout_days)
            visible_data = full_data[full_data.index <= cutoff_limit]
        else:
            visible_data = full_data
        if visible_data.empty:
            print(f"  ECHEC : --holdout-days {args.holdout_days} trop grand, "
                  f"aucune donnée visible restante pour {ticker}")
            errors += len(selected_models) * len(D_TO_TRADING_DAYS)
            continue
        visible_close, full_close = visible_data["Close"], full_data["Close"]
        visible_by_asset[ticker] = visible_close
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
                errors += len(D_TO_TRADING_DAYS)
                continue

            all_records.extend(records)
            for record in records:
                try:
                    if td.save_prediction(record, db_path=args.db_path):
                        inserted += 1
                    else:
                        duplicates += 1
                except ValueError as exc:
                    print(f"  {model_name:<12} {record['tc_id']} REJETE : {exc}")
                    errors += 1
            def _fmt_actual(r):
                return "—" if r["actual"] is None else f"{r['actual']:.2f}"

            detail = ", ".join(
                f"{r['tc_id']} pred={r['y_pred']:.2f} actual={_fmt_actual(r)}"
                for r in records
            )
            print(f"  {model_name:<12} ok — {len(records)} test cases traités ({detail})")

    print(f"\n[generate_test_cases] terminé -> {inserted} insérés, "
          f"{duplicates} doublons ignorés, {errors} échecs. DB : {args.db_path}")

    if not args.no_run_export and all_records:
        export_run_bundle(run_id, all_records, visible_by_asset, args, run_dir_root=args.run_dir)


if __name__ == "__main__":
    main()
