"""
test_cases/registry_models.py — Tableau des modèles couverts par les test cases
================================================================================
Source unique de vérité pour la liste des modèles de prévision évalués par
test_cases/run_test_cases.py. Pour ajouter un modèle, voir test_cases/README.md.

`forecaster` référence par son nom une fonction de benchmarks/multi_horizon.py,
contrat commun : forecast_horizons_<x>(train: pd.Series, horizons: list[int])
-> {h_days: (point, lo, hi)} — fit une seule fois, prévoit tous les horizons
demandés. Aucune logique de modèle n'est dupliquée ici : run_test_cases.py fait
juste `getattr(multi_horizon, forecaster)`.
"""

MODELS = [
    {
        "id": "ARIMA-GARCH",
        "label": "ARIMA-GARCH",
        "folder": "ARIMA",          # préfixe utilisé dans Run/<date>-<folder>-<asset>-<horizon>/
        "forecaster": "forecast_horizons_arima",
        "isolated_subprocess": False,
    },
    {
        "id": "SARIMA",
        "label": "SARIMA",
        "folder": "SARIMA",
        "forecaster": "forecast_horizons_sarima",
        "isolated_subprocess": False,
    },
    {
        "id": "Prophet",
        "label": "Prophet",
        "folder": "Prophet",
        "forecaster": "forecast_horizons_prophet",
        "isolated_subprocess": False,
    },
    {
        "id": "LSTM",
        "label": "LSTM",
        "folder": "LSTM",
        "forecaster": "forecast_horizons_lstm",
        # Deadlock TensorFlow documenté (models/conftest.py, model_artifacts/pipeline.py) si
        # statsmodels/yfinance sont importés avant TF dans le même process -> tourne dans un
        # sous-processus isolé (test_cases/lstm_subprocess_forecast.py), jamais en direct ici.
        "isolated_subprocess": True,
    },
    {
        "id": "TSDiff",
        "label": "TSDiff",
        "folder": "TSDiff",
        "forecaster": "forecast_horizons_tsdiff",
        "isolated_subprocess": False,
    },
]

MODEL_BY_ID = {m["id"]: m for m in MODELS}
