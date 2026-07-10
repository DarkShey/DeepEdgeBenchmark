"""
model_artifacts/pipeline.py — Pipeline « train + validate » -> artefacts modèles (Run/)
=========================================================================================
Aligne les artefacts produits par les 4 modèles réels (ARIMA-GARCH, SARIMA, Prophet, LSTM)
+ Naive sur le doc « DEITA — Artifacts des Modèles » (cf. BRIEF_model_artifacts.md).

Décision de conception assumée (le code existant ne sérialisait pas, walk-forward pur) :
on introduit ici une étape d'entraînement + sérialisation, en réutilisant SANS LES
MODIFIER (sauf extraction pure dans benchmarks/multi_horizon.py) les modèles de models/
et les adaptateurs de benchmarks/multi_horizon.py. Ce module est uniquement de
l'orchestration : split, 2 quality gates, écriture des fichiers d'artefacts.

2 quality gates par combinaison (modèle x actif x horizon) :
  - Gate 1 (Training)   : fit sur les 85% de début -> si OK, sérialise le modèle.
  - Gate 2 (Validation) : évalue sur les 15% de fin (jamais vus) -> si OK, sauve les métriques.
Un échec de gate sur une combinaison n'interrompt pas les autres (cf. run_benchmark.py).

Le modèle entraîné est le même quel que soit l'horizon (fit une fois, cf. §12 du brief) :
Gate 1 tourne une fois par (modèle, actif) et ses fichiers sont copiés à l'identique dans
les dossiers ...-D1 et ...-D7. Gate 2, en revanche, est spécifique à l'horizon :
  - D1 : réutilise tel quel run_<model>(train, validation) de models/*.py (walk-forward
    1-step déjà existant et testé) -> RMSE/MAE/... sur n_val = len(validation) points.
  - D7 : jours de trading (D+7 calendaire ~ 1 semaine ~ 5 jours de trading -- backtest
    Gate2, distinct de l'horizon business=7 j. de bourse, cf. BUSINESS_HORIZONS_TRADING_DAYS
    plus bas). Aucune fonction multi-step "sans refit" n'existe pour tous les modèles
    (Prophet/LSTM en particulier n'ont pas d'API d'état incrémental) -> évaluation par
    origines glissantes : à chaque
    origine, ré-appelle forecast_horizons_<model>(train_étendu, [7]) tel quel (refit inclus,
    exactement comme l'existant), MAX_D7_ROLLING_ORIGINS origines réparties sur la
    validation (borne le temps de calcul — limitation documentée, pas une approximation
    cachée).

Exécution (depuis DeepEdgeBenchmark/) :
    python -m model_artifacts.pipeline
    python -m model_artifacts.pipeline --assets "BTC-USD,SPY" --models "ARIMA-GARCH,Naive"
"""

import argparse
import ast
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Sur cette machine, laisser TensorFlow initialiser le backend Metal/GPU bloque le
# process indéfiniment (confirmé par un `sample` du PID pendant le blocage : pile
# Python figée dans les frameworks GPUCompiler/MPS, 0% CPU ensuite, aucun accès
# réseau en cause). Un LSTM de cette taille (seq_len=30, 64 unités) n'a de toute
# façon rien à gagner du GPU. Note : CUDA_VISIBLE_DEVICES est une variable NVIDIA,
# sans effet sur Metal (Apple Silicon) — le vrai réglage est set_visible_devices([],
# 'GPU'), posé ici avant tout import de lstm_model (donc avant le premier fit LSTM),
# pur réglage d'environnement, aucune logique de modélisation touchée.
#
# Variables d'environnement posées AVANT le import (le pool de threads de TF peut être
# initialisé au chargement de la lib native, avant qu'un appel Python à set_*_threads()
# ne puisse le reconfigurer) — le deadlock persistait avec les seuls appels API.
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
    # Deadlock distinct constaté ensuite lors du run réel (pas seulement des tests
    # unitaires) : `sample` sur le PID a montré le thread principal bloqué dans
    # TFE_Execute -> absl::Mutex::Block -> attente d'une Notification jamais signalée
    # — un deadlock dans le pool de threads interne de TensorFlow (connu sur certaines
    # configs Apple Silicon en cas de sur-souscription de threads). Un LSTM aussi petit
    # (seq_len=30, 64 unités) n'a rien à gagner d'un pool multi-thread ; le forcer en
    # mono-thread élimine la classe de deadlock sans coût de performance mesurable.
    _tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    # best-effort : si TF a déjà exécuté une op ailleurs (import concurrent, config déjà
    # verrouillée), on ne bloque pas le pipeline pour un réglage best-effort.
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from calibration.regime.assets import ASSETS
from benchmarks.run_benchmark import download_full_data
from benchmarks import multi_horizon as mh
from benchmarks.regime_overlay import fit_predict_regime

from validation import tracking_db as td
from validation import verdict_rules
from honest_eval import metrics as he_metrics

RUN_ROOT = REPO_ROOT / "Run"
DEFAULT_DB_PATH = "validation/tracking.db"

MODELS = ["ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive"]
MODEL_FOLDER_NAME = {
    "ARIMA-GARCH": "ARIMA", "SARIMA": "SARIMA", "Prophet": "Prophet",
    "LSTM": "LSTM", "Naive": "Naive",
}
# D+1/D+7 en jours de TRADING (backtest Gate2) ; D+7 ~ 1 semaine calendaire ~ 5 jours
# de trading -- distinct de BUSINESS_HORIZONS_TRADING_DAYS plus bas (contrat tracking.db).
HORIZON_TRADING_DAYS = {"D1": 1, "D7": 5}

# Contrat business tracking.db (BRIEF_tracking_db.md §3) : horizon = 1 ou 7 JOURS DE
# BOURSE exactement -- distinct du "D7" Gate2 ci-dessus qui est un horizon de backtest
# à 5 jours de bourse. On calcule donc en plus, dans la prévision live (jamais dans
# Gate2), le point à horizon=7 jours de bourse pour nourrir tracking.db sans toucher à
# la méthodologie Gate2 existante.
BUSINESS_HORIZONS_TRADING_DAYS = (1, 7)
PIPELINE_HORIZON_LABEL = {1: "D1", 7: "D7"}

TRAIN_RATIO = 0.85
WINDOW_YEARS = 3
DEFAULT_SEED = 42
MAX_D7_ROLLING_ORIGINS = 10   # borne le coût du refit répété (cf. docstring module)


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _num(v):
    """float JSON-safe : NaN/inf -> None."""
    if v is None:
        return None
    f = float(v)
    return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)


def chronological_split(close: pd.Series, train_ratio: float = TRAIN_RATIO):
    """Split chronologique strict train (début) / validation (fin) — jamais de mélange."""
    split_idx = int(len(close) * train_ratio)
    return close.iloc[:split_idx], close.iloc[split_idx:]


def combo_dir(run_date_str: str, model_folder_name: str, asset: str, horizon_label: str) -> Path:
    return RUN_ROOT / f"{run_date_str}-{model_folder_name}-{asset}-{horizon_label}"


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def get_lib_versions() -> dict:
    import numpy, pandas, statsmodels, sklearn, arch, yfinance
    versions = {
        "numpy": numpy.__version__, "pandas": pandas.__version__,
        "statsmodels": statsmodels.__version__, "sklearn": sklearn.__version__,
        "arch": arch.__version__, "yfinance": yfinance.__version__,
    }
    try:
        import tensorflow
        versions["tensorflow"] = tensorflow.__version__
    except ImportError:
        pass
    try:
        import prophet
        versions["prophet"] = prophet.__version__
    except ImportError:
        pass
    return versions


# ── Gate 1 : fit + quality check + sérialisation, par modèle ────────────────────
# Chaque handler délègue le fit à benchmarks/multi_horizon.py (fit_<model>, déjà
# extrait sans changement de comportement) — aucune logique de modélisation ici.

def _arima_fit(train, seed=None, epochs=None):
    arima_res, garch_res = mh.fit_arima(train)
    return {"arima_res": arima_res, "garch_res": garch_res, "last_price": float(train.iloc[-1])}


def _arima_quality_ok(fitted) -> bool:
    ok1 = np.all(np.isfinite(np.asarray(fitted["arima_res"].resid, dtype=float)))
    ok2 = np.all(np.isfinite(np.asarray(fitted["garch_res"].params.values, dtype=float)))
    return bool(ok1 and ok2)


def _arima_serialize(fitted, out_dir: Path):
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({"arima_res": fitted["arima_res"], "garch_res": fitted["garch_res"]}, f)
    pd.DataFrame({"resid": np.asarray(fitted["arima_res"].resid, dtype=float)}).to_parquet(
        out_dir / "residuals.parquet"
    )


def _arima_hyperparams(fitted) -> dict:
    import arima_model
    return {
        "order": list(arima_model.ARIMA_ORDER), "garch_p": 1, "garch_q": 1, "garch_dist": "normal",
        "aic": _num(fitted["arima_res"].aic), "bic": _num(fitted["arima_res"].bic),
    }


def _arima_forecast(fitted, horizons):
    return mh.forecast_from_fitted_arima(fitted["arima_res"], fitted["garch_res"],
                                         fitted["last_price"], horizons)


def _sarima_fit(train, seed=None, epochs=None):
    return {"result": mh.fit_sarima(train)}


def _sarima_quality_ok(fitted) -> bool:
    return bool(np.all(np.isfinite(np.asarray(fitted["result"].fittedvalues, dtype=float))))


def _sarima_serialize(fitted, out_dir: Path):
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(fitted["result"], f)
    pd.DataFrame({"resid": np.asarray(fitted["result"].resid, dtype=float)}).to_parquet(
        out_dir / "residuals.parquet"
    )


def _sarima_hyperparams(fitted) -> dict:
    import sarima_model
    return {
        "order": list(sarima_model.ORDER), "seasonal_order": list(sarima_model.SEASONAL_ORDER),
        "aic": _num(fitted["result"].aic), "bic": _num(fitted["result"].bic),
    }


def _sarima_forecast(fitted, horizons):
    return mh.forecast_from_fitted_sarima(fitted["result"], horizons)


def _prophet_fit(train, seed=None, epochs=None):
    return {"model": mh.fit_prophet(train), "last_date": train.index[-1]}


def _prophet_quality_ok(fitted) -> bool:
    df = fitted["model"].history[["ds"]]
    pred = fitted["model"].predict(df)
    return bool(np.all(np.isfinite(pred["yhat"].values)))


def _prophet_serialize(fitted, out_dir: Path):
    from prophet.serialize import model_to_json
    (out_dir / "model.json").write_text(model_to_json(fitted["model"]))


def _prophet_hyperparams(fitted) -> dict:
    m = fitted["model"]
    return {
        "changepoint_prior_scale": m.changepoint_prior_scale,
        "seasonality_prior_scale": m.seasonality_prior_scale,
        "interval_width": m.interval_width,
        "weekly_seasonality": True, "yearly_seasonality": True, "daily_seasonality": False,
    }


def _prophet_forecast(fitted, horizons):
    return mh.forecast_from_fitted_prophet(fitted["model"], fitted["last_date"], horizons)


def _lstm_fit(train, seed=None, epochs=None):
    model, scaler, std, scaled = mh.fit_lstm(train, epochs=epochs, seed=seed)
    return {"model": model, "scaler": scaler, "std": std, "scaled": scaled}


def _lstm_quality_ok(fitted) -> bool:
    weights_ok = all(np.all(np.isfinite(w)) for w in fitted["model"].get_weights())
    return bool(weights_ok and np.isfinite(fitted["std"]))


def _lstm_serialize(fitted, out_dir: Path):
    fitted["model"].save(str(out_dir / "model.h5"))
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(fitted["scaler"], f)


def _lstm_hyperparams(fitted) -> dict:
    import lstm_model
    return {
        "seq_len": lstm_model.SEQ_LEN, "units": lstm_model.UNITS,
        "epochs": lstm_model.EPOCHS, "batch_size": lstm_model.BATCH_SIZE,
        "optimizer": "adam",
    }


def _lstm_forecast(fitted, horizons):
    return mh.forecast_from_fitted_lstm(fitted["model"], fitted["scaler"], fitted["std"],
                                        fitted["scaled"], horizons)


HANDLERS = {
    "ARIMA-GARCH": {"fit": _arima_fit, "quality_ok": _arima_quality_ok,
                    "serialize": _arima_serialize, "hyperparams": _arima_hyperparams,
                    "forecast": _arima_forecast},
    "SARIMA": {"fit": _sarima_fit, "quality_ok": _sarima_quality_ok,
               "serialize": _sarima_serialize, "hyperparams": _sarima_hyperparams,
               "forecast": _sarima_forecast},
    "Prophet": {"fit": _prophet_fit, "quality_ok": _prophet_quality_ok,
                "serialize": _prophet_serialize, "hyperparams": _prophet_hyperparams,
                "forecast": _prophet_forecast},
    "LSTM": {"fit": _lstm_fit, "quality_ok": _lstm_quality_ok,
             "serialize": _lstm_serialize, "hyperparams": _lstm_hyperparams,
             "forecast": _lstm_forecast},
}


def _reload_arima(out_dir: Path, train: pd.Series):
    with open(out_dir / "model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return {"arima_res": bundle["arima_res"], "garch_res": bundle["garch_res"],
            "last_price": float(train.iloc[-1])}


def _reload_sarima(out_dir: Path, train: pd.Series):
    with open(out_dir / "model.pkl", "rb") as f:
        result = pickle.load(f)
    return {"result": result}


def _reload_prophet(out_dir: Path, train: pd.Series):
    from prophet.serialize import model_from_json
    model = model_from_json((out_dir / "model.json").read_text())
    return {"model": model, "last_date": train.index[-1]}


def _reload_lstm(out_dir: Path, train: pd.Series):
    import lstm_model
    # compile=False : on ne recharge que pour prédire (jamais pour continuer l'entraînement),
    # et l'état de compilation (optimizer/loss) sérialisé en HDF5 legacy par Keras 3 ne se
    # désérialise pas proprement ("keras.metrics.mse n'est pas un KerasSaveable") — un problème
    # de (dé)sérialisation de la config de compilation, pas des poids du modèle eux-mêmes.
    model = lstm_model.tf.keras.models.load_model(str(out_dir / "model.h5"), compile=False)
    with open(out_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    seq_len = lstm_model.SEQ_LEN
    scaled = scaler.transform(train.values.reshape(-1, 1)).flatten()
    X, _ = lstm_model.make_sequences(scaled, seq_len)
    train_preds = scaler.inverse_transform(
        model.predict(X.reshape(-1, seq_len, 1), verbose=0).reshape(-1, 1)).flatten()
    std = float(np.std(train.values[seq_len:] - train_preds))
    return {"model": model, "scaler": scaler, "std": std, "scaled": scaled}


RELOADERS = {
    "ARIMA-GARCH": _reload_arima, "SARIMA": _reload_sarima,
    "Prophet": _reload_prophet, "LSTM": _reload_lstm,
}


def reload_model(model_key: str, out_dir: Path, train: pd.Series):
    """Recharge le modèle sérialisé par fit_and_serialize() depuis out_dir, en
    recalculant à partir de `train` le contexte éphémère non sérialisé (dernier
    prix/date, série normalisée) — utilisé par le test de round-trip (§10) pour
    vérifier que le modèle rechargé redonne les mêmes prévisions que l'original."""
    return RELOADERS[model_key](out_dir, train)


def fit_and_serialize(model_key: str, train: pd.Series, out_dir: Path, seed=None, epochs=None):
    """Gate 1 : fit (délégué à benchmarks/multi_horizon.py) + contrôle qualité
    (valeurs finies) + sérialisation + hyperparams.json si OK.
    Retourne (fitted, gate1_ok). fitted est None si le fit a levé une exception."""
    handler = HANDLERS[model_key]
    try:
        fitted = handler["fit"](train, seed=seed, epochs=epochs)
        gate1_ok = handler["quality_ok"](fitted)
    except Exception as exc:
        print(f"    [Gate1 FAIL] {model_key} : {exc}")
        return None, False

    if gate1_ok:
        out_dir.mkdir(parents=True, exist_ok=True)
        handler["serialize"](fitted, out_dir)
        hp = handler["hyperparams"](fitted)
        (out_dir / "hyperparams.json").write_text(json.dumps(hp, indent=2))
    else:
        print(f"    [Gate1 FAIL] {model_key} : contrôle qualité échoué (NaN/inf détecté)")
    return fitted, gate1_ok


SERIALIZED_FILES = {
    "ARIMA-GARCH": ["model.pkl", "residuals.parquet", "hyperparams.json"],
    "SARIMA": ["model.pkl", "residuals.parquet", "hyperparams.json"],
    "Prophet": ["model.json", "hyperparams.json"],
    "LSTM": ["model.h5", "scaler.pkl", "hyperparams.json"],
}


def copy_serialized_artifacts(src_dir: Path, dst_dir: Path, model_key: str):
    """§12 : le modèle entraîné est le même quel que soit l'horizon (fit une fois) ->
    copie à l'identique les fichiers de Gate 1 plutôt que de refitter pour chaque
    dossier d'horizon."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in SERIALIZED_FILES[model_key]:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)


# ── Gate 2 : évaluation hors-échantillon, par horizon ───────────────────────────

def _run_model_d1(model_key: str, train: pd.Series, validation: pd.Series, seed, epochs) -> dict:
    """D+1 : réutilise tel quel le run_<model>(train, validation) walk-forward
    1-step déjà existant dans models/*.py (aucune nouvelle logique de modélisation)."""
    if model_key == "ARIMA-GARCH":
        import arima_model
        return arima_model.run_arima_garch(train, validation)
    if model_key == "SARIMA":
        import sarima_model
        return sarima_model.run_sarima(train, validation)
    if model_key == "Prophet":
        import prophet_model
        return prophet_model.run_prophet(train, validation)
    if model_key == "LSTM":
        import lstm_model
        lstm_model.set_seed(seed or lstm_model.DEFAULT_SEED)
        return lstm_model.run_lstm(train, validation, epochs=epochs or lstm_model.EPOCHS)
    if model_key == "Naive":
        import naive_model
        naive_model.set_seed(seed or naive_model.DEFAULT_SEED)
        return naive_model.run_naive(train, validation)
    raise ValueError(model_key)


def _compute_metrics_for(model_key: str, actual, predicted, pi_lower, pi_upper) -> dict:
    """compute_metrics est identique dans les 4 modules + naive_model — on prend
    celui du modèle concerné plutôt que d'en réimplémenter un."""
    import importlib
    module_name = {
        "ARIMA-GARCH": "arima_model", "SARIMA": "sarima_model",
        "Prophet": "prophet_model", "LSTM": "lstm_model", "Naive": "naive_model",
    }[model_key]
    mod = importlib.import_module(module_name)
    return mod.compute_metrics(actual, predicted, pi_lower=pi_lower, pi_upper=pi_upper)


def _forecast_horizon(model_key: str, train_extended: pd.Series, h_days: int, seed, epochs):
    if model_key == "ARIMA-GARCH":
        return mh.forecast_horizons_arima(train_extended, [h_days])[h_days]
    if model_key == "SARIMA":
        return mh.forecast_horizons_sarima(train_extended, [h_days])[h_days]
    if model_key == "Prophet":
        return mh.forecast_horizons_prophet(train_extended, [h_days])[h_days]
    if model_key == "LSTM":
        return mh.forecast_horizons_lstm(train_extended, [h_days], epochs=epochs, seed=seed)[h_days]
    if model_key == "Naive":
        return mh.forecast_horizons_naive(train_extended, [h_days])[h_days]
    raise ValueError(model_key)


def _forecast_all_horizons(model_key: str, train_extended: pd.Series, horizons_days: list, seed, epochs) -> dict:
    """Comme _forecast_horizon mais pour plusieurs horizons en un seul fit (contrat
    forecast_horizons_<model> : fit once puis prévoit tous les horizons demandés) —
    utilisé pour la prévision live (hors-échantillon, au-delà de window_end), fittée
    une fois par (modèle, actif) et réutilisée pour D1 et D7 (cf. Gate 1 qui fait de
    même via copy_serialized_artifacts)."""
    if model_key == "ARIMA-GARCH":
        return mh.forecast_horizons_arima(train_extended, horizons_days)
    if model_key == "SARIMA":
        return mh.forecast_horizons_sarima(train_extended, horizons_days)
    if model_key == "Prophet":
        return mh.forecast_horizons_prophet(train_extended, horizons_days)
    if model_key == "LSTM":
        return mh.forecast_horizons_lstm(train_extended, horizons_days, epochs=epochs, seed=seed)
    if model_key == "Naive":
        return mh.forecast_horizons_naive(train_extended, horizons_days)
    raise ValueError(model_key)


def _run_model_d7_rolling(model_key: str, train: pd.Series, validation: pd.Series,
                          h_days: int, seed, epochs, max_origins: int) -> dict:
    """D+7 (ou plus généralement h_days > 1) : aucune API d'état incrémental commune
    aux 4 modèles (Prophet/LSTM n'en ont pas) -> évaluation par origines glissantes,
    chaque origine ré-appelant forecast_horizons_<model> tel quel (refit inclus,
    exactement le comportement déjà existant de cette fonction). max_origins borne
    le coût total (limitation documentée, cf. docstring de module)."""
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
        point, lo, hi = _forecast_horizon(model_key, extended_train, h_days, seed, epochs)
        dates.append(validation.index[target_idx])
        actuals.append(actual); preds.append(point); los.append(lo); his.append(hi)

    metrics = _compute_metrics_for(model_key, actuals, preds, los, his)
    # Clés préfixées `_` (même convention que `_n_val` déjà en place) : poppées par
    # evaluate_gate2 avant de construire le payload de métriques, portées jusque-là
    # pour permettre l'écriture de predictions.parquet (cf. write_predictions_parquet).
    metrics["_n_val"] = len(origins)
    metrics["_dates"] = dates
    metrics["_actuals"] = actuals
    metrics["_preds"] = preds
    metrics["_los"] = los
    metrics["_his"] = his
    return metrics


def _naive_baseline_for_comparison(model_key: str, train: pd.Series, validation: pd.Series,
                                   horizon_label: str, h_days: int, seed, epochs,
                                   max_d7_origins: int, pred_arr, payload) -> tuple:
    """Prévisions Naive alignées sur les mêmes dates que le modèle évalué (mêmes
    train/validation/horizon -> mêmes origines D7, cf. _run_model_d7_rolling), pour
    les métriques honest_eval (Point 1 du brief d'amélioration). Naive n'a aucun coût
    de fit : recalculer sa série pour chaque modèle plutôt que de la partager entre
    modèles est trivial en coût CPU et évite tout risque de désalignement de dates."""
    if model_key == "Naive":
        return pred_arr, payload["MAE"], payload["RMSE"]
    if horizon_label == "D1":
        naive_result = _run_model_d1("Naive", train, validation, seed, epochs)
        naive_pred_arr = list(np.asarray(naive_result["predictions"], dtype=float))
    else:
        naive_result = _run_model_d7_rolling("Naive", train, validation, h_days, seed, epochs,
                                             max_d7_origins)
        naive_pred_arr = naive_result.pop("_preds")
    return naive_pred_arr, naive_result["MAE"], naive_result["RMSE"]


def _to_metrics_payload(result: dict, model_key: str, asset: str, horizon_label: str, n_val: int,
                        pi_lower, pi_upper) -> dict:
    pi_cov = result.get("PI Cov 95% (%)")
    widths = np.asarray(pi_upper, dtype=float) - np.asarray(pi_lower, dtype=float)
    return {
        "RMSE": _num(result.get("RMSE")),
        "MAE": _num(result.get("MAE")),
        "MAPE": _num(result.get("MAPE (%)")),
        "directional_accuracy": _num(result.get("Dir. Acc (%)")),
        "pi_coverage_95": _num(pi_cov) if pi_cov != "N/A" else None,
        "pi_width_min": _num(np.min(widths)) if widths.size else None,
        "pi_width_mean": _num(np.mean(widths)) if widths.size else None,
        "pi_width_max": _num(np.max(widths)) if widths.size else None,
        "n_val": n_val,
        "horizon": horizon_label,
        "asset": asset,
        "model": model_key,
    }


def _gate2_metrics_ok(payload: dict) -> bool:
    required = ("RMSE", "MAE", "MAPE", "directional_accuracy", "pi_coverage_95")
    return all(payload.get(k) is not None for k in required)


def evaluate_gate2(model_key: str, asset: str, train: pd.Series, validation: pd.Series,
                   horizon_label: str, seed=None, epochs=None,
                   max_d7_origins: int = MAX_D7_ROLLING_ORIGINS):
    """Gate 2 : évalue sur les 15% de fin, retourne (payload, gate2_ok, series). payload
    et series sont None si le calcul a levé une exception. `series` (dict avec
    dates/actual/predicted/pi_lower/pi_upper, alignés point à point) alimente
    write_predictions_parquet — pour D1 c'est le validation set complet (walk-forward
    1-step), pour D7 c'est un point par origine glissante (cf. _run_model_d7_rolling)."""
    h_days = HORIZON_TRADING_DAYS[horizon_label]
    try:
        if horizon_label == "D1":
            result = _run_model_d1(model_key, train, validation, seed, epochs)
            n_val = len(validation)
            dates = list(result["index"])
            actual_arr = list(np.asarray(result["actual"], dtype=float))
            pred_arr = list(np.asarray(result["predictions"], dtype=float))
            lo_arr = list(np.asarray(result["lower"], dtype=float))
            hi_arr = list(np.asarray(result["upper"], dtype=float))
        else:
            result = _run_model_d7_rolling(model_key, train, validation, h_days, seed, epochs, max_d7_origins)
            n_val = result.pop("_n_val")
            dates = result.pop("_dates")
            actual_arr = result.pop("_actuals")
            pred_arr = result.pop("_preds")
            lo_arr = result.pop("_los")
            hi_arr = result.pop("_his")
        payload = _to_metrics_payload(result, model_key, asset, horizon_label, n_val, lo_arr, hi_arr)
        series = {"dates": dates, "actual": actual_arr, "predicted": pred_arr,
                  "pi_lower": lo_arr, "pi_upper": hi_arr}
    except Exception as exc:
        print(f"    [Gate2 FAIL] {model_key} {horizon_label} : {exc}")
        return None, False, None

    # Point 1 du brief d'amélioration : le modèle n'a de skill que s'il bat la baseline
    # Naive sur les VARIATIONS, pas les niveaux (cf. honest_eval/metrics.py). Best-effort
    # : un échec ici (ex. baseline Naive elle-même en échec pour cet actif) ne doit pas
    # faire échouer tout Gate 2, qui a déjà ses propres métriques en niveaux valides.
    try:
        naive_pred_arr, naive_mae, naive_rmse = _naive_baseline_for_comparison(
            model_key, train, validation, horizon_label, h_days, seed, epochs,
            max_d7_origins, pred_arr, payload)
        payload["honest_eval"] = he_metrics.variation_metrics(
            actual=actual_arr, predicted=pred_arr, naive_predicted=naive_pred_arr,
            mae_model=payload["MAE"], mae_naive=naive_mae,
            rmse_model=payload["RMSE"], rmse_naive=naive_rmse,
        )
    except Exception as exc:
        print(f"    [honest_eval] {model_key} {horizon_label} : échec métriques vs naïf ({exc})")
        payload["honest_eval"] = None

    ok = _gate2_metrics_ok(payload)
    if not ok:
        print(f"    [Gate2 FAIL] {model_key} {horizon_label} : métriques non calculables ({payload})")
    return payload, ok, series


def write_metadata_json(out_dir: Path, asset: str, asset_class: str, window_start: str,
                        window_end: str, train_end: str, run_date: str, seed: int) -> None:
    payload = {
        "asset": asset, "asset_class": asset_class, "frequency": "1d",
        "window_start": window_start, "window_end": window_end, "train_end": train_end,
        "run_date": run_date, "git_commit": get_git_commit(), "seed": seed,
        "lib_versions": get_lib_versions(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2))


def write_predictions_parquet(out_dir: Path, dates, actual, predicted, pi_lower, pi_upper) -> None:
    """Série datée réel/prédit/PI 95% de la validation Gate 2 (D1 : un point par jour de
    validation ; D7 : un point par origine glissante) — alimente le graphe du dashboard."""
    df = pd.DataFrame({
        "date": pd.DatetimeIndex(dates),
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
        "pi_lower": np.asarray(pi_lower, dtype=float),
        "pi_upper": np.asarray(pi_upper, dtype=float),
    })
    df.to_parquet(out_dir / "predictions.parquet")


def write_prices_parquet(out_dir: Path, train: pd.Series, validation: pd.Series) -> None:
    """Historique complet (train + validation) — permet au graphe du dashboard de tracer
    la courbe de prix réelle avant même la fenêtre de validation, avec la coupure
    train/validation (cf. metadata.json.train_end). Même redondance assumée que
    metadata.json (cf. BRIEF §12) : un dossier de combinaison reste auto-suffisant."""
    full = pd.concat([train, validation])
    df = pd.DataFrame({"date": pd.DatetimeIndex(full.index), "close": full.values.astype(float)})
    df.to_parquet(out_dir / "prices.parquet")


# ── Orchestration d'une combinaison / du pipeline complet ───────────────────────

MODEL_TEST_FILTER = {
    # Bracketé : "arima_model" seul matcherait aussi les tests de "sarima_model" (sous-chaîne).
    "ARIMA-GARCH": "[arima_model]", "SARIMA": "[sarima_model]", "Prophet": "[prophet_model]",
    "LSTM": "lstm",  # pas "[lstm_model]" : capte aussi les 2 tests spécifiques LSTM de
                      # test_models_common.py (noms contenant "lstm" mais hors fixture paramétrée)
    "Naive": "naive_model",  # matche models/test_naive_model.py (5 tests dédiés, cf. Point 0
                              # du brief d'amélioration -- persistance stricte + PI gaussien)
}

_unit_test_cache = {}

# Fichiers de tests couvrant models/*.py, sourcés par MODEL_TEST_FILTER ci-dessus.
UNIT_TEST_SOURCE_FILES = ["models/test_models_common.py", "models/test_metrics.py",
                          "models/test_naive_model.py"]


def _module_docstring(relative_path: str) -> str:
    """Docstring de tête du fichier de test, lue à chaque run (jamais périmée, contrairement
    à une doc à maintenir à la main) — répond à "à quoi correspondent ces tests"."""
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    return (ast.get_docstring(tree) or "").strip()


def _model_unit_test_results(model_key: str) -> dict:
    """Résultats des tests unitaires (models/) propres à ce modèle : statut global, détail
    test par test (via le rapport JUnit, plus fiable qu'un parsing du texte de sortie), et
    la docstring des fichiers de tests sources pour documenter ce qu'ils vérifient. Mis en
    cache pour la durée du process : le code testé ne change pas d'un actif/horizon à
    l'autre dans un même run, inutile de relancer pytest pour un résultat identique."""
    if model_key in _unit_test_cache:
        return _unit_test_cache[model_key]

    filter_expr = MODEL_TEST_FILTER[model_key]
    with tempfile.TemporaryDirectory() as tmp:
        junit_path = Path(tmp) / "report.xml"
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "models", "-k", filter_expr,
             "-q", "--no-header", f"--junitxml={junit_path}"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        tests = []
        if junit_path.exists():
            root = ET.parse(junit_path).getroot()
            suite = root.find("testsuite") if root.tag == "testsuites" else root
            for case in suite.findall("testcase"):
                outcome = "passed"
                if case.find("failure") is not None:
                    outcome = "failed"
                elif case.find("error") is not None:
                    outcome = "error"
                elif case.find("skipped") is not None:
                    outcome = "skipped"
                module = case.get("classname", "").replace(".", "/") + ".py"
                tests.append({"name": f"{module}::{case.get('name')}", "outcome": outcome})

    summary_line = next((l for l in reversed(result.stdout.strip().splitlines()) if l.strip()), "")
    status = "no_tests_collected" if result.returncode == 5 else ("pass" if result.returncode == 0 else "fail")

    summary = {
        "model": model_key, "pytest_filter": filter_expr,
        "status": status, "exit_code": result.returncode, "summary": summary_line,
        "tests": tests,
        "what_these_tests_check": {f: _module_docstring(f) for f in UNIT_TEST_SOURCE_FILES},
    }
    _unit_test_cache[model_key] = summary
    return summary


def _run_lstm_via_worker(train: pd.Series, validation: pd.Series, out_dir: Path, seed, epochs,
                         max_d7_origins: int, horizons: list, business_h_days: set) -> dict:
    """LSTM tourne dans un sous-processus neuf et isolé (model_artifacts/lstm_worker.py),
    jamais dans CE process : ce module importe benchmarks.multi_horizon (donc
    arima_model/regime_overlay) sans condition dès son chargement (cf. import plus haut),
    et cette combinaison avec TensorFlow bloque indéfiniment le premier model.fit()
    (deadlock confirmé le 2026-07-08, cf. docstring de lstm_worker.py) -- ni une
    question de nombre de threads, ni spécifique à Prophet, ni réglable par les
    variables d'environnement OpenMP usuelles. Seule l'isolation de process règle
    le problème, vérifiée empiriquement."""
    with tempfile.TemporaryDirectory() as tmp:
        data_pickle = Path(tmp) / "data.pkl"
        result_json = Path(tmp) / "result.json"
        with open(data_pickle, "wb") as f:
            pickle.dump((train, validation), f)
        # Toujours inclure les horizons business (1 et 7 j. de bourse, tracking.db, décalés
        # de business_lag si besoin -- cf. process_asset_model) en plus des horizons Gate2
        # demandés -- une seule invocation du worker isolé couvre donc à la fois le
        # backtest ML et la prévision live business.
        live_horizons_days = sorted({HORIZON_TRADING_DAYS[h] for h in horizons} | business_h_days)
        cmd = [sys.executable, "-m", "model_artifacts.lstm_worker",
               "--data-pickle", str(data_pickle), "--out-dir", str(out_dir),
               "--result-json", str(result_json), "--max-d7-origins", str(max_d7_origins),
               "--horizons", ",".join(horizons),
               "--live-horizons", ",".join(str(h) for h in live_horizons_days)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        result = json.loads(result_json.read_text())
        # Clés JSON toujours strings -- reconverties en int (h_days) pour matcher
        # HORIZON_TRADING_DAYS et le format déjà utilisé par _forecast_all_horizons.
        result["live_forecast"] = {int(k): v for k, v in result.get("live_forecast", {}).items()}
        return result


def _save_business_predictions(run_id: str, model_key: str, ticker: str, asset_class: str,
                               regime_tag: str, full_series: pd.Series, forecasts_by_h: dict,
                               db_path: str, business_lag: int) -> None:
    """Persiste la prévision live à horizon=1 et horizon=7 jours de bourse dans
    tracking.db (contrat BRIEF_tracking_db.md §3, Partie B) -- même dict forecasts_by_h
    que celui republié dans metrics.json (clé "forecast") pour le Gate2 horizon_label,
    mais ici horizon=1/7 jours de bourse (contrat business), potentiellement différent
    de l'horizon Gate2 D7=5 jours de bourse (cf. BUSINESS_HORIZONS_TRADING_DAYS plus haut).

    "D+1" doit toujours vouloir dire "demain PAR RAPPORT À AUJOURD'HUI", jamais "demain
    par rapport au dernier close connu" : si le run tourne en journée (avant la clôture
    du marché du jour même), cutoff_date = hier (dernier close déjà publié), et un
    horizon de 1 jour de bourse à partir de là tombe sur AUJOURD'HUI, pas demain (bug
    constaté le 2026-07-09). business_lag (calculé une fois dans process_asset_model,
    partagé avec le worker LSTM) compense cet écart, ajouté à l'horizon business avant
    de chercher la prévision correspondante dans forecasts_by_h."""
    cutoff_date = full_series.index[-1].date()
    last_close = float(full_series.iloc[-1])
    now = datetime.now()

    for h_days in BUSINESS_HORIZONS_TRADING_DAYS:
        effective_h = h_days + business_lag
        if effective_h not in forecasts_by_h:
            continue
        point, lo, hi = forecasts_by_h[effective_h]
        target_date = cutoff_date + pd.Timedelta(days=effective_h)

        record = {
            "run_id": run_id,
            "tc_id": f"TC_{ticker}_D{h_days}",
            "model": model_key,
            "asset": ticker,
            "horizon": h_days,
            "cutoff_date": str(cutoff_date),
            "target_date": str(target_date),
            "regime": regime_tag,
            "last_close": last_close,
            "y_pred": float(point),
            "y_lower": float(lo),
            "y_upper": float(hi),
            "created_at": now.isoformat(timespec="seconds"),
        }
        record["verdict_integrite"] = verdict_rules.check_integrity(record)
        record["verdict_plausibilite"] = (
            verdict_rules.check_plausibility(record, asset_class, h_days)
            if record["verdict_integrite"] else 0
        )
        td.save_prediction(record, db_path=db_path)


def export_business_validation(run_id: str, db_path: str = DEFAULT_DB_PATH,
                               run_dir_root=RUN_ROOT) -> list:
    """Range les résultats de validation business (prédiction + évaluation dès que
    connue) de ce run_id dans Run/<date>-<modèle>-<actif>-<horizon>/business_validation.json
    (mêmes dossiers combo que combo_dir/MODEL_FOLDER_NAME ci-dessus). Appelé en fin de
    run_pipeline() et par validation/evaluate_daily.py (cron) après résolution
    quotidienne des prédictions échues, pour rafraîchir y_true/abs_error/... sans
    attendre un nouveau run complet."""
    date_str = run_id.removeprefix("run_").split("T", 1)[0]
    rows = td.fetch_predictions_for_run(run_id, db_path=db_path)

    written = []
    for row in rows:
        model_folder = MODEL_FOLDER_NAME.get(row["model"], row["model"])
        horizon_label = PIPELINE_HORIZON_LABEL.get(row["horizon"], f"H{row['horizon']}")
        out_dir = Path(run_dir_root) / f"{date_str}-{model_folder}-{row['asset']}-{horizon_label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = dict(row)
        payload["horizon_trading_days"] = row["horizon"]
        if row["horizon"] != HORIZON_TRADING_DAYS["D7"] and horizon_label == "D7":
            payload["note"] = (
                "horizon_trading_days=7 ici (test case business), à ne pas confondre "
                "avec les métriques Gate2 D7 de ce même dossier qui portent sur "
                f"{HORIZON_TRADING_DAYS['D7']} jours de bourse (cf. PIPELINE_HORIZON_LABEL)."
            )

        (out_dir / "business_validation.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        written.append(out_dir)

    print(f"[pipeline] business_validation.json écrit dans {len(written)} dossier(s) Run/")
    return written


def process_asset_model(model_key: str, ticker: str, asset_class: str, train: pd.Series,
                        validation: pd.Series, run_date_str: str, run_date_iso: str,
                        window_start: str, window_end: str, seed: int, epochs,
                        max_d7_origins: int, horizons: list, run_id: str, regime_tag: str,
                        db_path: str) -> list:
    """Gate 1 une fois (sauf Naive, rien à entraîner) puis Gate 2 par horizon.
    Retourne la liste des logs (un par horizon)."""
    train_end = str(train.index[-1].date())
    logs = []

    first_out_dir = combo_dir(run_date_str, MODEL_FOLDER_NAME[model_key], ticker, horizons[0])

    # cf. docstring de _save_business_predictions : décalage d'un jour de bourse entre
    # cutoff (dernier close connu, ici = fin de validation) et aujourd'hui si le run
    # tourne avant la clôture du marché du jour même -- les horizons business doivent
    # alors viser 1 jour de bourse plus loin pour que "D+1"/"D+7" restent "demain"/"dans
    # 1 semaine" par rapport à aujourd'hui, pas par rapport au dernier close. Calculé ici
    # (avant l'appel LSTM) pour que le worker isolé forecaste aussi le bon horizon.
    business_lag = 1 if validation.index[-1].date() < datetime.now().date() else 0
    business_h_days = {h + business_lag for h in BUSINESS_HORIZONS_TRADING_DAYS}

    lstm_worker_result = None
    if model_key == "LSTM":
        lstm_worker_result = _run_lstm_via_worker(train, validation, first_out_dir, seed, epochs,
                                                  max_d7_origins, horizons, business_h_days)
        gate1_ok = lstm_worker_result["gate1_ok"]
        if gate1_ok:
            (first_out_dir / "hyperparams.json").write_text(
                json.dumps(lstm_worker_result["hyperparams"], indent=2))
        else:
            print(f"    [Gate1 FAIL] LSTM : {lstm_worker_result.get('gate1_error')}")
    else:
        fitted, gate1_ok = None, True
        if model_key != "Naive":
            fitted, gate1_ok = fit_and_serialize(model_key, train, first_out_dir, seed=seed, epochs=epochs)
    print(f"  [{model_key:<12} {ticker:<8}] Gate1 (training)   : {'PASS' if gate1_ok else 'FAIL'}")

    unit_test_result = _model_unit_test_results(model_key)

    # Prévision live (hors-échantillon) : fit une seule fois sur train+validation (toute la
    # donnée connue, ancre = window_end) et prévoit tous les horizons demandés d'un coup —
    # distinct de Gate 2 (backtest) qui ne voit jamais au-delà de la validation.
    full_series = pd.concat([train, validation])
    h_days_list = sorted({HORIZON_TRADING_DAYS[h] for h in horizons} | business_h_days)
    if model_key == "LSTM":
        # _forecast_all_horizons("LSTM", ...) appellerait mh.forecast_horizons_lstm dans CE
        # process (celui-ci a déjà importé benchmarks.multi_horizon plus haut) -> même
        # deadlock TF que Gate1/Gate2 (cf. _run_lstm_via_worker). La prévision live est
        # déjà calculée par le worker (voir --live-horizons dans _run_lstm_via_worker).
        forecasts_by_h = lstm_worker_result.get("live_forecast", {}) if lstm_worker_result else {}
    else:
        try:
            forecasts_by_h = _forecast_all_horizons(model_key, full_series, h_days_list, seed, epochs)
        except Exception as exc:
            forecasts_by_h = {}
            print(f"  [{model_key:<12} {ticker:<8}] Prévision live     : ECHEC ({exc})")

    _save_business_predictions(run_id, model_key, ticker, asset_class, regime_tag,
                               full_series, forecasts_by_h, db_path, business_lag)

    for horizon_label in horizons:
        out_dir = combo_dir(run_date_str, MODEL_FOLDER_NAME[model_key], ticker, horizon_label)
        out_dir.mkdir(parents=True, exist_ok=True)
        if model_key != "Naive" and gate1_ok and out_dir != first_out_dir:
            copy_serialized_artifacts(first_out_dir, out_dir, model_key)

        if model_key == "LSTM":
            g2 = lstm_worker_result["gate2"].get(horizon_label, {"ok": False})
            if g2["ok"]:
                payload = _to_metrics_payload(g2["metrics"], model_key, ticker, horizon_label,
                                              len(g2["actual"]), g2["pi_lower"], g2["pi_upper"])
                gate2_ok = _gate2_metrics_ok(payload)
                series = {"dates": g2["dates"], "actual": g2["actual"], "predicted": g2["predicted"],
                          "pi_lower": g2["pi_lower"], "pi_upper": g2["pi_upper"]}
                # LSTM tourne dans un sous-process isolé (cf. _run_lstm_via_worker) et ne
                # passe jamais par evaluate_gate2 -- honest_eval (Point 1) doit donc être
                # recalculé ici explicitement, sinon LSTM resterait le seul modèle sans ces
                # métriques (constaté : honest_eval=None sur le run réel du 2026-07-10).
                try:
                    naive_pred_arr, naive_mae, naive_rmse = _naive_baseline_for_comparison(
                        model_key, train, validation, horizon_label,
                        HORIZON_TRADING_DAYS[horizon_label], seed, epochs, max_d7_origins,
                        g2["predicted"], payload)
                    payload["honest_eval"] = he_metrics.variation_metrics(
                        actual=g2["actual"], predicted=g2["predicted"], naive_predicted=naive_pred_arr,
                        mae_model=payload["MAE"], mae_naive=naive_mae,
                        rmse_model=payload["RMSE"], rmse_naive=naive_rmse,
                    )
                except Exception as exc:
                    print(f"    [honest_eval] LSTM {horizon_label} : échec métriques vs naïf ({exc})")
                    payload["honest_eval"] = None
            else:
                payload, gate2_ok, series = None, False, None
                print(f"    [Gate2 FAIL] LSTM {horizon_label} : {g2.get('error')}")
        else:
            payload, gate2_ok, series = evaluate_gate2(model_key, ticker, train, validation, horizon_label,
                                                       seed=seed, epochs=epochs, max_d7_origins=max_d7_origins)
        if gate2_ok:
            # Prévision hors-échantillon (au-delà de last_date, la dernière clôture connue)
            # repliée directement dans metrics.json plutôt que dans un forecast.json séparé
            # -- ni l'un ni l'autre n'est dans les "5 fichiers du doc" (BRIEF_model_artifacts.md
            # §5), et un fichier à part faisait doublon avec business_validation.json (même
            # prévision, deux endroits, risque de désynchronisation constaté en pratique).
            h_days = HORIZON_TRADING_DAYS[horizon_label]
            if h_days in forecasts_by_h:
                point, lo, hi = forecasts_by_h[h_days]
                payload["forecast"] = {
                    "last_date": str(full_series.index[-1].date()),
                    "last_price": _num(float(full_series.iloc[-1])),
                    "predicted": _num(point), "pi_lower": _num(lo), "pi_upper": _num(hi),
                }
            (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
            write_predictions_parquet(out_dir, **series)

        write_prices_parquet(out_dir, train, validation)
        write_metadata_json(out_dir, ticker, asset_class, window_start, window_end,
                            train_end, run_date_iso, seed)
        (out_dir / "unit_tests.json").write_text(json.dumps(unit_test_result, indent=2, ensure_ascii=False))

        status = f"RMSE={payload['RMSE']} DirAcc={payload['directional_accuracy']}%" if gate2_ok else "—"
        print(f"  [{model_key:<12} {ticker:<8} {horizon_label}] Gate2 (validation) : "
              f"{'PASS' if gate2_ok else 'FAIL'}  {status}")

        logs.append({
            "model": model_key, "asset": ticker, "horizon": horizon_label,
            "gate1": gate1_ok, "gate2": gate2_ok, "dir": str(out_dir),
        })
    return logs


def run_pipeline(models=None, assets=None, horizons=None, run_date=None, seed=DEFAULT_SEED,
                 epochs=None, max_d7_origins=MAX_D7_ROLLING_ORIGINS, run_id=None,
                 db_path=DEFAULT_DB_PATH) -> list:
    models = models or MODELS
    assets = assets or ASSETS
    horizons = horizons or list(HORIZON_TRADING_DAYS)
    run_date = run_date or datetime.now()
    run_date_str = run_date.strftime("%Y%m%d")
    run_date_iso = run_date.strftime("%Y-%m-%d")
    data_end = run_date_iso
    data_start = (run_date - pd.DateOffset(years=WINDOW_YEARS)).strftime("%Y-%m-%d")
    # Un run_id partagé entre les 2 sous-processus (non-LSTM puis LSTM, cf. main()) --
    # passé explicitement par main() pour que les deux moitiés d'un même run complet
    # écrivent sous le même run_id en base (tracking.db).
    run_id = run_id or f"run_{run_date.strftime('%Y%m%dT%H%M%S')}"

    all_logs = []
    for asset_info in assets:
        ticker, asset_class = asset_info["ticker"], asset_info["asset_class"]
        print(f"\n=== {ticker} : téléchargement ({data_start} -> {data_end}) ===")
        try:
            full_data = download_full_data(ticker, data_start, data_end)
        except SystemExit as exc:
            print(f"  ECHEC téléchargement : {exc}")
            continue
        full_close = full_data["Close"]
        train, validation = chronological_split(full_close)
        window_start, window_end = str(full_close.index[0].date()), str(full_close.index[-1].date())
        print(f"  fenêtre {window_start} -> {window_end}  |  train={len(train)}  validation={len(validation)}")

        try:
            # RegimeAgent.fit() indexe High/Low/Volume -- on lui passe le DataFrame
            # OHLCV complet (full_data), pas juste la colonne Close.
            regime_tag = fit_predict_regime(full_data, full_data.index[-1]).dominant_regime()
        except Exception as exc:
            print(f"  régime indisponible ({exc}) -> régime='unknown'")
            regime_tag = "unknown"

        for model_key in models:
            logs = process_asset_model(model_key, ticker, asset_class, train, validation,
                                       run_date_str, run_date_iso, window_start, window_end,
                                       seed, epochs, max_d7_origins, horizons, run_id,
                                       regime_tag, db_path)
            all_logs.extend(logs)

    export_business_validation(run_id, db_path=db_path, run_dir_root=RUN_ROOT)
    return all_logs


# Isolation par process, pas par thread : constaté empiriquement que faire tourner Prophet
# (qui lance un sous-processus cmdstan/Stan à chaque fit — potentiellement 10+ fois pour le
# rolling D7) PUIS LSTM dans le MÊME process Python provoque un deadlock reproductible dans
# le thread pool interne de TensorFlow (TFE_Execute -> absl::Notification jamais signalée),
# même GPU désactivé et TF forcé en mono-thread. LSTM seul, ou Prophet seul, fonctionnent
# parfaitement (vérifié séparément) — le problème est la combinaison des deux dans un seul
# process, pas l'un ou l'autre individuellement. Plutôt que de chasser plus loin un bug
# d'interaction entre bibliothèques natives, on isole : le run complet (sans --models
# explicite) lance deux sous-processus séparés (le reste, puis LSTM seul).
_MODELS_BEFORE_LSTM = [m for m in MODELS if m != "LSTM"]


def main():
    p = argparse.ArgumentParser(description="Pipeline train+validate -> artefacts modèles (Run/)")
    p.add_argument("--assets", default=None, help="tickers séparés par des virgules (défaut : les 5 de assets.py)")
    p.add_argument("--models", default=None, help="modèles séparés par des virgules (défaut : tous)")
    p.add_argument("--horizons", default=None, help="D1,D7 (défaut : les deux)")
    p.add_argument("--epochs", type=int, default=None, help="épochs LSTM (défaut : lstm_model.EPOCHS)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-d7-origins", type=int, default=MAX_D7_ROLLING_ORIGINS)
    p.add_argument("--no-process-isolation", action="store_true",
                   help="désactive l'isolation LSTM/autres-modèles en 2 sous-processus (déconseillé)")
    p.add_argument("--run-date", default=None,
                   help="YYYY-MM-DD : date à utiliser pour le nom des dossiers Run/ (défaut : "
                        "aujourd'hui) — pour compléter rétroactivement un run passé")
    p.add_argument("--run-id", default=None,
                   help="run_YYYYMMDDTHHMMSS (défaut : généré) — interne, sert à partager le "
                        "même run_id entre les 2 sous-processus non-LSTM/LSTM d'un run complet")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH,
                   help="tracking.db où persister les prédictions business (Partie B)")
    args = p.parse_args()
    run_date = datetime.strptime(args.run_date, "%Y-%m-%d") if args.run_date else None

    if args.models is None and not args.no_process_isolation:
        # Run complet : on ré-invoke ce même script deux fois (sous-processus séparés) plutôt
        # que d'exécuter tous les modèles dans le process courant.
        run_id = args.run_id or f"run_{(run_date or datetime.now()).strftime('%Y%m%dT%H%M%S')}"
        common = []
        if args.assets: common += ["--assets", args.assets]
        if args.horizons: common += ["--horizons", args.horizons]
        if args.epochs is not None: common += ["--epochs", str(args.epochs)]
        if args.run_date: common += ["--run-date", args.run_date]
        common += ["--seed", str(args.seed), "--max-d7-origins", str(args.max_d7_origins),
                  "--run-id", run_id, "--db-path", args.db_path]

        print(f"=== Sous-processus 1/2 : {', '.join(_MODELS_BEFORE_LSTM)} ===")
        subprocess.run([sys.executable, "-m", "model_artifacts.pipeline",
                        "--models", ",".join(_MODELS_BEFORE_LSTM), *common], check=True)

        print("\n=== Sous-processus 2/2 : LSTM (isolé) ===")
        subprocess.run([sys.executable, "-m", "model_artifacts.pipeline",
                        "--models", "LSTM", *common], check=True)
        return

    models = args.models.split(",") if args.models else None
    horizons = args.horizons.split(",") if args.horizons else None
    assets = None
    if args.assets:
        wanted = {t.strip() for t in args.assets.split(",")}
        assets = [a for a in ASSETS if a["ticker"] in wanted]

    logs = run_pipeline(models=models, assets=assets, horizons=horizons, run_date=run_date,
                        seed=args.seed, epochs=args.epochs, max_d7_origins=args.max_d7_origins,
                        run_id=args.run_id, db_path=args.db_path)

    n_gate1_pass = sum(1 for l in logs if l["gate1"])
    n_gate2_pass = sum(1 for l in logs if l["gate2"])
    print(f"\n=== Terminé : {len(logs)} combinaisons — Gate1 PASS {n_gate1_pass}/{len(logs)}, "
          f"Gate2 PASS {n_gate2_pass}/{len(logs)} ===")


if __name__ == "__main__":
    main()
