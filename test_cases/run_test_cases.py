"""
test_cases/run_test_cases.py — Orchestrateur : lance les modèles sur chaque occurrence
========================================================================================
Pour chaque (test case × actif × occurrence retenue × modèle) : entraîne le modèle sur
tout l'historique connu jusqu'au cutoff (dernier jour de l'ancien régime, aucune fuite du
futur) et prévoit les horizons demandés (jours de bourse), via
`benchmarks.multi_horizon.forecast_horizons_<model>` — réutilisé tel quel, aucune logique
de modèle réécrite ici. LSTM est isolé dans un sous-processus dédié
(test_cases/lstm_subprocess_forecast.py, même deadlock TensorFlow documenté que
model_artifacts/pipeline.py).

La cible réelle (y_true) est déjà connue (toutes les transitions étudiées sont passées) :
lue directement dans test_cases/data/<ticker>.csv, pas besoin de retélécharger ni
d'attendre — contrairement au pipeline `Run/` qui prédit hors-échantillon en direct.

Écrit results/<tc_id>/<ticker>/<transition_date>/<model_id>.json.

Usage (depuis DeepEdgeBenchmark/) :
    python -m test_cases.run_test_cases
    python -m test_cases.run_test_cases --test-cases TC1.3 --assets BTC-USD --models ARIMA-GARCH,SARIMA
    python -m test_cases.run_test_cases --max-occurrences 2 --epochs 10
"""

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from test_cases.registry_models import MODELS, MODEL_BY_ID
from test_cases.registry_assets import ASSETS, ASSET_BY_TICKER
from test_cases.registry_test_cases import TEST_CASES, TEST_CASE_BY_ID, MAX_OCCURRENCES_PER_ASSET
from test_cases.transitions import load_series, select_occurrences

RUN_ROOT = REPO_ROOT / "Run"
RESULTS_ROOT = Path(__file__).parent / "results"
DEFAULT_SEED = 42


def _num(v):
    """float JSON-safe : NaN/inf -> None."""
    if v is None:
        return None
    f = float(v)
    return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)


def _find_latest_hyperparams(model_folder: str, ticker: str):
    """Dernier Run/<date>-<model_folder>-<ticker>-D1/hyperparams.json en date — purement
    informatif/traçabilité (répond à "reprendre les paramètres de sortie, derniers en
    date, des modèles du fichier Run, et indiquer la date"). N'affecte jamais le fit
    lui-même (toujours refait à la date historique du cutoff, sans quoi ce serait une
    fuite du futur — cf. model_artifacts/pipeline.py §12 : les hyperparamètres sont de
    toute façon des constantes de modèle, identiques d'un run à l'autre)."""
    candidates = sorted(RUN_ROOT.glob(f"*-{model_folder}-{ticker}-D1"), reverse=True)
    for d in candidates:
        hp_path = d / "hyperparams.json"
        if hp_path.exists():
            return {
                "source_run_date": d.name.split("-", 1)[0],
                "source_run_dir": d.name,
                "hyperparams": json.loads(hp_path.read_text()),
            }
    return None


def _train_series(df: pd.DataFrame, cutoff_idx: int) -> pd.Series:
    window = df.iloc[: cutoff_idx + 1]
    return pd.Series(window["Close"].values, index=pd.DatetimeIndex(window["Date"]))


def _forecast_non_lstm(model_id: str, train: pd.Series, horizons: list) -> dict:
    from benchmarks import multi_horizon as mh
    forecaster = getattr(mh, MODEL_BY_ID[model_id]["forecaster"])
    return forecaster(train, horizons)


def _forecast_lstm(train: pd.Series, horizons: list, seed: int, epochs) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        data_pickle = Path(tmp) / "train.pkl"
        result_json = Path(tmp) / "result.json"
        with open(data_pickle, "wb") as f:
            pickle.dump(train, f)
        cmd = [sys.executable, "-m", "test_cases.lstm_subprocess_forecast",
               "--data-pickle", str(data_pickle), "--horizons", ",".join(str(h) for h in horizons),
               "--result-json", str(result_json), "--seed", str(seed)]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        payload = json.loads(result_json.read_text())
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "échec LSTM (sous-processus)"))
    return {int(h): tuple(v) for h, v in payload["forecasts"].items()}


def run_one(test_case: dict, ticker: str, occurrence: dict, model_id: str, seed: int, epochs) -> dict:
    model = MODEL_BY_ID[model_id]
    df = load_series(ticker)
    train = _train_series(df, occurrence["cutoff_idx"])
    horizons = test_case["horizons_days"]

    if model["isolated_subprocess"]:
        forecasts = _forecast_lstm(train, horizons, seed=seed, epochs=epochs)
    else:
        forecasts = _forecast_non_lstm(model_id, train, horizons)

    last_close = occurrence["last_close"]
    by_horizon = {}
    for h in horizons:
        if h not in forecasts:
            continue
        point, lo, hi = forecasts[h]
        y_true = occurrence["target_actuals"][h]
        by_horizon[str(h)] = {
            "target_date": str(occurrence["target_dates"][h].date()),
            "y_pred": _num(point), "y_lower": _num(lo), "y_upper": _num(hi),
            "y_true": _num(y_true),
            "in_interval": bool(lo <= y_true <= hi),
            "direction_correct": bool(np.sign(point - last_close) == np.sign(y_true - last_close)),
            "abs_error": _num(abs(y_true - point)),
            "abs_error_naif": _num(abs(y_true - last_close)),
            "beats_naif": bool(abs(y_true - point) <= abs(y_true - last_close)),
        }

    hp_info = _find_latest_hyperparams(model["folder"], ticker)

    return {
        "tc_id": test_case["id"], "tc_name": test_case["name"],
        "model": model_id, "asset": ticker,
        "transition_date": str(occurrence["transition_date"].date()),
        "cutoff_date": str(occurrence["cutoff_date"].date()),
        "regime_from": test_case["regime_from"], "regime_to": test_case["regime_to"],
        "last_close": _num(last_close),
        "horizons": by_horizon,
        "hyperparams_source": hp_info,
    }


def run_pipeline(test_case_ids=None, tickers=None, model_ids=None, max_occurrences=None,
                 seed=DEFAULT_SEED, epochs=None) -> list:
    test_cases = [TEST_CASE_BY_ID[i] for i in test_case_ids] if test_case_ids else TEST_CASES
    assets = [ASSET_BY_TICKER[t] for t in tickers] if tickers else ASSETS
    models = [MODEL_BY_ID[m] for m in model_ids] if model_ids else MODELS

    written = []
    for test_case in test_cases:
        tc_id = test_case["id"]
        print(f"\n=== {tc_id} — {test_case['label']} ===")
        for asset in assets:
            ticker = asset["ticker"]
            occurrences = select_occurrences(ticker, test_case, max_n=max_occurrences)
            if not occurrences:
                print(f"  {ticker:<8} : aucune occurrence trouvée")
                continue
            print(f"  {ticker:<8} : {len(occurrences)} occurrence(s) — "
                  f"{[str(o['transition_date'].date()) for o in occurrences]}")
            for occ in occurrences:
                transition_str = str(occ["transition_date"].date())
                out_dir = RESULTS_ROOT / tc_id / ticker / transition_str
                out_dir.mkdir(parents=True, exist_ok=True)
                for model in models:
                    model_id = model["id"]
                    try:
                        record = run_one(test_case, ticker, occ, model_id, seed=seed, epochs=epochs)
                        (out_dir / f"{model_id}.json").write_text(
                            json.dumps(record, indent=2, ensure_ascii=False))
                        written.append(out_dir / f"{model_id}.json")
                        print(f"    [{model_id:<12} {transition_str}] OK")
                    except Exception as exc:
                        print(f"    [{model_id:<12} {transition_str}] ECHEC : {exc}")
    return written


def main():
    p = argparse.ArgumentParser(description="Lance les modèles sur les test cases de transition de régime")
    p.add_argument("--test-cases", default=None, help="ids séparés par des virgules (ex. TC1.2,TC1.3)")
    p.add_argument("--assets", default=None, help="tickers séparés par des virgules")
    p.add_argument("--models", default=None, help="ids de modèle séparés par des virgules")
    p.add_argument("--max-occurrences", type=int, default=None,
                   help=f"défaut : {MAX_OCCURRENCES_PER_ASSET} (registry_test_cases.py)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=None, help="épochs LSTM (défaut : lstm_model.EPOCHS)")
    args = p.parse_args()

    test_case_ids = args.test_cases.split(",") if args.test_cases else None
    tickers = args.assets.split(",") if args.assets else None
    model_ids = args.models.split(",") if args.models else None

    written = run_pipeline(test_case_ids=test_case_ids, tickers=tickers, model_ids=model_ids,
                           max_occurrences=args.max_occurrences, seed=args.seed, epochs=args.epochs)
    print(f"\n=== Terminé : {len(written)} fichier(s) résultat écrit(s) sous {RESULTS_ROOT} ===")


if __name__ == "__main__":
    main()
