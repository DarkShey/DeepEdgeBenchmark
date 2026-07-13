"""
test_cases/lstm_subprocess_forecast.py — worker LSTM isolé pour les test cases
================================================================================
Même piège documenté dans model_artifacts/lstm_worker.py et models/conftest.py :
importer arima_model/sarima_model (même transitivement, via `benchmarks.multi_horizon`)
dans le même process que TensorFlow bloque indéfiniment le premier `model.fit()`
(deadlock confirmé, cf. docstrings citées). La parade vérifiée est double : (1) ce
script n'importe QUE numpy/pandas/lstm_model — jamais benchmarks.* ni calibration.* —
et (2) TensorFlow est importé et configuré tout en haut, avant tout le reste.

Duplique donc ici la même glue que benchmarks/multi_horizon.py::fit_lstm /
forecast_from_fitted_lstm (~25 lignes d'adaptation, pas de logique de modèle : le LSTM
lui-même reste entièrement dans models/lstm_model.py, non modifié) — même choix que
model_artifacts/lstm_worker.py, pour la même raison (cf. sa docstring).

Contrat : lit une série d'entraînement (pd.Series picklée) depuis --data-pickle, calcule
forecast_horizons_lstm(train, horizons) et écrit {h: [point, lo, hi]} en JSON dans
--result-json.

Usage : python -m test_cases.lstm_subprocess_forecast --data-pickle <path>
        --horizons 1,7 --result-json <path> [--seed 42] [--epochs 30]
"""

import os

# Doit s'exécuter avant TOUT le reste (numpy/pandas compris) : le pool de threads de
# TensorFlow peut être initialisé au chargement de la lib native, avant qu'un appel
# Python à set_*_threads() ne puisse le reconfigurer.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))


def _fit_lstm_local(train, epochs=None, seed=None):
    """Copie de benchmarks/multi_horizon.py::fit_lstm (glue, pas de logique de modèle) —
    dupliquée pour ne jamais importer benchmarks.multi_horizon dans ce process isolé."""
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


def _forecast_from_fitted_lstm_local(model, scaler, std, scaled, horizons):
    """Copie de benchmarks/multi_horizon.py::forecast_from_fitted_lstm (glue)."""
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


def forecast_horizons_lstm_isolated(train, horizons, epochs=None, seed=None):
    model, scaler, std, scaled = _fit_lstm_local(train, epochs=epochs, seed=seed)
    return _forecast_from_fitted_lstm_local(model, scaler, std, scaled, horizons)


def main():
    p = argparse.ArgumentParser(description="Worker LSTM isolé (test cases)")
    p.add_argument("--data-pickle", required=True, help="pd.Series (train) picklée")
    p.add_argument("--horizons", required=True, help="ex. '1,7'")
    p.add_argument("--result-json", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    with open(args.data_pickle, "rb") as f:
        train = pickle.load(f)
    horizons = [int(h) for h in args.horizons.split(",")]

    try:
        result = forecast_horizons_lstm_isolated(train, horizons, epochs=args.epochs, seed=args.seed)
        payload = {"ok": True, "forecasts": {str(h): list(v) for h, v in result.items()}}
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}

    Path(args.result_json).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
