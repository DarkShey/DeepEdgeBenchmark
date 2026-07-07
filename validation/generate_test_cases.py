"""
generate_test_cases.py — Test cases de base pour la validation business
============================================================================
Génère un jeu de test cases (actif x horizon x modèle) : une vraie prédiction
"aujourd'hui" par combinaison, verdictée immédiatement (intégrité +
plausibilité, cf. verdict_rules.py) et persistée via tracking_db.save_prediction().
Ne calcule PAS de précision (pas de comparaison à un `actual` réel) — c'est le
rôle d'evaluate_pending()/report(), gérés ailleurs et volontairement hors
scope de ce script.

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
import traceback
from datetime import datetime, timedelta

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


def build_records(ticker, asset_class, model_name, adapter_fn, train_close,
                  regime_tag, run_id, epochs, seed):
    """Une prédiction par horizon D pour (ticker, model), déjà verdictée."""
    trading_days = sorted(set(D_TO_TRADING_DAYS.values()))
    if model_name == "LSTM":
        raw = adapter_fn(train_close, trading_days, epochs=epochs, seed=seed)
    else:
        raw = adapter_fn(train_close, trading_days)

    cutoff_date = train_close.index[-1].date()
    last_close = float(train_close.iloc[-1])
    now = datetime.now().isoformat(timespec="seconds")

    records = []
    for d, h_days in D_TO_TRADING_DAYS.items():
        point, lo, hi = raw[h_days]
        target_date = cutoff_date + timedelta(days=d)
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

    inserted = duplicates = errors = 0
    for asset in selected_assets:
        ticker, asset_class = asset["ticker"], asset["asset_class"]
        print(f"\n[generate_test_cases] {ticker} : téléchargement ...")
        try:
            full_data = download_full_data(ticker, config.DATA_START, data_end)
        except SystemExit as exc:
            print(f"  ECHEC téléchargement : {exc}")
            errors += len(selected_models) * len(D_TO_TRADING_DAYS)
            continue
        train_close = full_data["Close"]

        print(f"[generate_test_cases] {ticker} : calibration régime ...")
        try:
            # fit_predict_regime attend le DataFrame OHLCV complet (comme
            # split["train"] dans run_benchmark.py), pas juste la colonne Close —
            # RegimeAgent.fit() indexe d'autres colonnes (ex. High/Low/Volume).
            regime_state = fit_predict_regime(full_data, full_data.index[-1])
            regime_tag = regime_state.dominant_regime()
        except Exception as exc:
            print(f"  régime indisponible ({exc}) -> régime='unknown'")
            regime_tag = "unknown"

        for model_name, adapter_fn in selected_models:
            try:
                records = build_records(ticker, asset_class, model_name, adapter_fn,
                                        train_close, regime_tag, run_id,
                                        args.epochs, args.seed)
            except Exception:
                print(f"  {model_name:<12} ECHEC génération prédiction :")
                traceback.print_exc()
                errors += len(D_TO_TRADING_DAYS)
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
            print(f"  {model_name:<12} ok — {len(records)} test cases traités "
                  f"(intégrité/plausibilité : "
                  f"{[(r['verdict_integrite'], r['verdict_plausibilite']) for r in records]})")

    print(f"\n[generate_test_cases] terminé -> {inserted} insérés, "
          f"{duplicates} doublons ignorés, {errors} échecs. DB : {args.db_path}")


if __name__ == "__main__":
    main()
