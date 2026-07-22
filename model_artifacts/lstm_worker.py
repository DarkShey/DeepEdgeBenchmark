"""
model_artifacts/lstm_worker.py — worker LSTM isolé, jamais dans le même process que
benchmarks.multi_horizon.
========================================================================================
Investigation (session du 2026-07-08, cf. sample du PID pendant le blocage) : importer
arima_model/sarima_model/naive_model/regime_overlay — même transitivement, via
`from benchmarks import multi_horizon as mh` ou `from benchmarks.run_benchmark import
download_full_data`, qui importent tous les deux benchmarks.regime_overlay ->
calibration.regime.regime_hmm (pandas_ta + arch + hmmlearn) — dans le même process que
TensorFlow bloque indéfiniment le premier model.fit() (deadlock confirmé : pile figée
dans TFE_Execute -> Notification jamais signalée, dans le pool de threads interne de
TF). Ce n'est ni une histoire de nombre de threads (testé 1 et 2, pareil), ni
spécifique à Prophet (LSTM seul suffit), ni réglable par les variables d'environnement
OpenMP usuelles (KMP_DUPLICATE_LIB_OK, TF_ENABLE_ONEDNN_OPTS=0, testées, pareil) :
seule l'isolation complète du process fonctionne, vérifiée empiriquement.

Ce script n'importe donc QUE numpy/pandas/lstm_model — jamais benchmarks.*, jamais
calibration.*, jamais model_artifacts.pipeline lui-même (qui importe benchmarks.multi_horizon
sans condition dès son chargement, cf. pipeline.py ligne ~91). Deuxième piège trouvé en
testant CE script isolément : lstm_model.py importe lui-même yfinance/statsmodels/sklearn
AVANT tensorflow (son propre ordre interne, cf. ses imports) — cet ordre-là aussi
suffit à reproduire le deadlock. D'où l'import + configuration explicite de tensorflow
TOUT EN HAUT de ce fichier, avant tout le reste (même trick que le haut de pipeline.py) :
quand `import lstm_model` s'exécute ensuite, son propre `import tensorflow` interne ne
fait plus que référencer le module déjà chargé, sans jamais laisser yfinance/statsmodels
s'exécuter avant l'initialisation de TF.

Contrat : lit (train, validation) -- 2 pd.Series picklées -- depuis --data-pickle,
calcule gate1 (fit + contrôle qualité) et gate2 D1/D7 (walk-forward + origines
glissantes, même logique que _run_model_d1/_run_model_d7_rolling de pipeline.py, dupliquée
ici pour ne dépendre d'aucun import de pipeline.py), écrit model.h5/scaler.pkl dans
--out-dir et un résultat JSON (--result-json). Le process parent (model_artifacts/pipeline.py,
qui lui peut importer benchmarks.multi_horizon sans risque puisqu'il ne touche jamais TF
lui-même) relit ce JSON pour finir la sérialisation standard (metadata.json,
predictions.parquet, etc. -- pure I/O, aucun besoin d'isolement).

Usage : python -m model_artifacts.lstm_worker --data-pickle <path> --out-dir <path>
        --result-json <path> --seed 42 --epochs 30 --max-d7-origins 10 --horizons D1,D7
"""

import os

# Doit s'exécuter avant TOUT le reste (numpy/pandas compris, cf. docstring de module) :
# le pool de threads de TensorFlow peut être initialisé au chargement de la lib native,
# avant qu'un appel Python à set_*_threads() ne puisse le reconfigurer.
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
    _tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

# Même convention que model_artifacts/pipeline.py (D7 = 5 jours de bourse, pas 7).
HORIZON_TRADING_DAYS = {"D1": 1, "D7": 5}


def _fit_lstm(train, epochs=None, seed=None):
    """Dupliqué de benchmarks/multi_horizon.py::fit_lstm (pas importable depuis ce
    worker sans réintroduire la chaîne benchmarks.multi_horizon -> arima_model/regime,
    cf. docstring de module). Ne dépend que de lstm_model, comme l'original."""
    import lstm_model
    seq_len = lstm_model.SEQ_LEN
    epochs = lstm_model.EPOCHS if epochs is None else epochs
    seed = lstm_model.DEFAULT_SEED if seed is None else seed
    lstm_model.set_seed(seed)

    if len(train) <= seq_len:
        raise ValueError(
            f"train series has {len(train)} points, but seq_len={seq_len} requires "
            f"more than {seq_len} points to build at least one training sequence."
        )

    scaler = lstm_model.MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    X, y = lstm_model.make_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = lstm_model.build_lstm(seq_len)
    es = lstm_model.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=lstm_model.BATCH_SIZE,
             validation_split=0.1, callbacks=[es], verbose=0)

    train_preds = scaler.inverse_transform(
        model.predict(X, verbose=0).reshape(-1, 1)).flatten()
    std = float(np.std(train.values[seq_len:] - train_preds))
    return model, scaler, std, scaled


def _forecast_from_fitted_lstm(model, scaler, std, scaled, horizons):
    """Dupliqué de benchmarks/multi_horizon.py::forecast_from_fitted_lstm — même raison."""
    import lstm_model
    seq_len = lstm_model.SEQ_LEN
    max_h = max(horizons)

    buffer = list(scaled[-seq_len:])
    rollout_scaled = []
    for _ in range(max_h):
        x = np.array(buffer[-seq_len:]).reshape(1, seq_len, 1)
        p_scaled = model.predict(x, verbose=0)[0, 0]
        rollout_scaled.append(p_scaled)
        buffer.append(p_scaled)

    rollout_prices = scaler.inverse_transform(
        np.array(rollout_scaled).reshape(-1, 1)).flatten()

    results = {}
    for h in horizons:
        i = h - 1
        point = float(rollout_prices[i])
        sigma_h = std * np.sqrt(h)
        results[h] = (point, point - 1.96 * sigma_h, point + 1.96 * sigma_h)
    return results


def _d1_walk_forward(train, validation, seed, epochs):
    import lstm_model
    from model_artifacts import crps_kpis
    lstm_model.set_seed(seed if seed is not None else lstm_model.DEFAULT_SEED)
    result = lstm_model.run_lstm(train, validation, epochs=epochs or lstm_model.EPOCHS,
                                 n_ensemble=crps_kpis.DEFAULT_N_ENSEMBLE)
    dates = [str(d) for d in result["index"]]
    actual = np.asarray(result["actual"], dtype=float).tolist()
    pred = np.asarray(result["predictions"], dtype=float).tolist()
    lo = np.asarray(result["lower"], dtype=float).tolist()
    hi = np.asarray(result["upper"], dtype=float).tolist()
    metrics = lstm_model.compute_metrics(actual, pred, pi_lower=lo, pi_upper=hi)
    # Nuage MC-Dropout jamais sérialisé dans le JSON de résultat (--result-json) --
    # seul le scalaire crps y figure, comme les autres métriques (cf. crps_kpis.py).
    metrics["crps"] = crps_kpis.crps_from_step_ensembles(result.get("ensemble"), actual)
    return metrics, dates, actual, pred, lo, hi


def _d7_rolling_origins(train, validation, h_days, seed, epochs, max_origins):
    import lstm_model
    n_val = len(validation)
    max_origin = n_val - h_days
    if max_origin < 1:
        raise ValueError(f"validation trop courte ({n_val} points) pour horizon={h_days}")

    n_origins = min(max_origins, max_origin)
    origins = sorted(set(np.linspace(0, max_origin - 1, n_origins, dtype=int).tolist()))

    dates, actuals, preds, los, his = [], [], [], [], []
    for origin in origins:
        extended_train = pd.concat([train, validation.iloc[:origin]]) if origin > 0 else train
        target_idx = origin + h_days - 1
        actual = float(validation.iloc[target_idx])
        model, scaler, std, scaled = _fit_lstm(extended_train, epochs=epochs, seed=seed)
        point, lo, hi = _forecast_from_fitted_lstm(model, scaler, std, scaled, [h_days])[h_days]
        dates.append(str(validation.index[target_idx]))
        actuals.append(actual); preds.append(float(point)); los.append(float(lo)); his.append(float(hi))

    metrics = lstm_model.compute_metrics(actuals, preds, pi_lower=los, pi_upper=his)
    return metrics, dates, actuals, preds, los, his


def main():
    p = argparse.ArgumentParser(description="Worker LSTM isolé (jamais dans le process de benchmarks.multi_horizon)")
    p.add_argument("--data-pickle", required=True, help="pickle de (train, validation) -- 2 pd.Series")
    p.add_argument("--out-dir", required=True, help="dossier où écrire model.h5/scaler.pkl (gate1)")
    p.add_argument("--result-json", required=True, help="chemin où écrire le résultat (gate1_ok, gate2 par horizon)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-d7-origins", type=int, default=10)
    p.add_argument("--horizons", default="D1,D7")
    p.add_argument("--live-horizons", default="",
                   help="jours de bourse (ex. 1,5) pour la prévision hors-échantillon sur "
                        "train+validation combinés -- cf. process_asset_model/_forecast_all_horizons "
                        "de pipeline.py, qui ne peut pas appeler mh.forecast_horizons_lstm dans son "
                        "propre process (même deadlock que Gate1/Gate2, cf. docstring de module)")
    p.add_argument("--skip-training", action="store_true",
                   help="saute le fit Gate1 (85% train, sérialisation) -- cf. pipeline.py "
                        "--full-retrain=False : le process parent recopie alors model.h5/scaler.pkl "
                        "d'un run antérieur plutôt que de les régénérer. N'affecte ni le Gate2 "
                        "(contrôlé indépendamment par --horizons, qui peut déjà être restreint aux "
                        "seuls horizons non réutilisables) ni --live-horizons : les deux font leur "
                        "propre fit, jamais celui de Gate1 (cf. docstring de module).")
    args = p.parse_args()

    import lstm_model

    with open(args.data_pickle, "rb") as f:
        train, validation = pickle.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [h for h in args.horizons.split(",") if h]

    result = {"gate1_ok": False, "gate2": {}}
    if args.skip_training:
        # Rien à faire ici : le process parent recopie model.h5/scaler.pkl/hyperparams.json
        # d'un run antérieur (cf. --skip-training ci-dessus) -- ne pas toucher out_dir.
        result["gate1_skipped"] = True
    else:
        try:
            model, scaler, std, scaled = _fit_lstm(train, epochs=args.epochs, seed=args.seed)
            weights_ok = all(bool(np.all(np.isfinite(w))) for w in model.get_weights())
            gate1_ok = bool(weights_ok and np.isfinite(std))
            result["gate1_ok"] = gate1_ok
            if gate1_ok:
                model.save(str(out_dir / "model.h5"))
                with open(out_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
                result["hyperparams"] = {
                    "seq_len": lstm_model.SEQ_LEN, "units": lstm_model.UNITS,
                    "epochs": lstm_model.EPOCHS, "batch_size": lstm_model.BATCH_SIZE,
                    "optimizer": "adam",
                }
        except Exception as exc:
            result["gate1_error"] = str(exc)

    for horizon_label in horizons:
        h_days = HORIZON_TRADING_DAYS[horizon_label]
        try:
            if horizon_label == "D1":
                metrics, dates, actual, pred, lo, hi = _d1_walk_forward(train, validation, args.seed, args.epochs)
            else:
                metrics, dates, actual, pred, lo, hi = _d7_rolling_origins(
                    train, validation, h_days, args.seed, args.epochs, args.max_d7_origins)
            result["gate2"][horizon_label] = {
                "ok": True, "metrics": metrics,
                "dates": dates, "actual": actual, "predicted": pred, "pi_lower": lo, "pi_upper": hi,
            }
        except Exception as exc:
            result["gate2"][horizon_label] = {"ok": False, "error": str(exc)}

    result["live_forecast"] = {}
    if args.live_horizons:
        h_days_list = [int(h) for h in args.live_horizons.split(",")]
        try:
            full_series = pd.concat([train, validation])
            model, scaler, std, scaled = _fit_lstm(full_series, epochs=args.epochs, seed=args.seed)
            forecasts = _forecast_from_fitted_lstm(model, scaler, std, scaled, h_days_list)
            result["live_forecast"] = {
                str(h): [float(point), float(lo), float(hi)]
                for h, (point, lo, hi) in forecasts.items()
            }
        except Exception as exc:
            result["live_forecast_error"] = str(exc)

    Path(args.result_json).write_text(json.dumps(result, default=str, indent=2))
    gate2_summary = ", ".join(f"{h}={result['gate2'][h]['ok']}" for h in horizons)
    gate1_summary = "skipped" if result.get("gate1_skipped") else result["gate1_ok"]
    print(f"[lstm_worker] gate1_ok={gate1_summary} gate2=({gate2_summary})")


if __name__ == "__main__":
    main()
