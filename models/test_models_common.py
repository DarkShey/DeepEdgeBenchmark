"""
Tests factorisés sur les 4 forecasters (arima, sarima, prophet, lstm), via l'interface
commune fetch_data/run_<model>/next_step_<model>. Zéro réseau : fetch_data est soit
monkeypatché, soit jamais appelé (les données de test viennent de synthetic_series()).

Contrat de sortie observé (identique sur les 4 modules, aucune normalisation nécessaire) :
run_<model>(train, test) -> dict avec au moins "predictions", "lower", "upper", "index",
"actual" + les clés de compute_metrics aplaties (RMSE, MAE, "Dir. Acc (%)", "PI Cov 95% (%)", ...).
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from conftest import make_train_test, reset_seeds, synthetic_series

REQUIRED_RESULT_KEYS = (
    "predictions", "lower", "upper", "index", "actual",
    "RMSE", "MAE", "Dir. Acc (%)", "PI Cov 95% (%)",
)


def test_fetch_data_monkeypatched_no_network(model_module, monkeypatch):
    fixed = synthetic_series(n=50, seed=1)
    monkeypatch.setattr(model_module, "fetch_data", lambda ticker, start, end: fixed)

    result = model_module.fetch_data("FAKE-TICKER", "2020-01-01", "2020-06-01")

    pd.testing.assert_series_equal(result, fixed)


def test_run_model_returns_expected_contract(run_fn, run_kwargs):
    train, test = make_train_test()

    result = run_fn(train, test, **run_kwargs)

    for key in REQUIRED_RESULT_KEYS:
        assert key in result, f"missing key {key!r} in run_<model> result"
    assert len(result["predictions"]) == len(test)
    assert np.all(np.isfinite(np.asarray(result["predictions"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["lower"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["upper"], dtype=float)))


def test_next_step_returns_finite_point_and_interval(next_step_fn, run_kwargs):
    series = synthetic_series(n=120)

    pred, lo, hi = next_step_fn(series, **run_kwargs)

    assert all(np.isfinite([pred, lo, hi]))
    assert lo <= pred <= hi


def test_point_in_time_no_lookahead(model_module, run_fn, run_kwargs):
    """Truncate the test window and re-predict: the shared prefix of predictions must
    match (within tolerance — TF/optimizer convergence isn't perfectly bit-exact), which
    would not hold if run_<model> ever peeked at test rows beyond the one being predicted."""
    train, test = make_train_test()

    reset_seeds(0)
    full_result = run_fn(train, test, **run_kwargs)

    reset_seeds(0)
    truncated_result = run_fn(train, test.iloc[:3], **run_kwargs)

    full_prefix = np.asarray(full_result["predictions"], dtype=float)[:3]
    truncated = np.asarray(truncated_result["predictions"], dtype=float)

    assert truncated == pytest.approx(full_prefix, rel=5e-2, abs=5e-2)


def test_run_model_n_ensemble_zero_omits_ensemble_key(run_fn, run_kwargs):
    """Non-régression : n_ensemble=0 (défaut) reste le comportement actuel exact pour
    tous les appelants existants (CLI standalone, next_step_*, experiments/) -- pas de
    clé "ensemble" ajoutée au dict retourné."""
    train, test = make_train_test()

    result = run_fn(train, test, **run_kwargs)

    assert "ensemble" not in result


def test_run_model_n_ensemble_populates_step_clouds(run_fn, run_kwargs):
    """n_ensemble>0 -- bootstrap des résidus (ARIMA/SARIMA/Prophet) ou MC-Dropout
    (LSTM) -- doit produire un nuage fini de la bonne taille par pas de validation
    (cf. model_artifacts/crps_kpis.py, qui consomme result["ensemble"])."""
    train, test = make_train_test()
    n_ensemble = 20

    result = run_fn(train, test, n_ensemble=n_ensemble, **run_kwargs)

    assert "ensemble" in result
    assert len(result["ensemble"]) == len(test)
    for cloud in result["ensemble"]:
        cloud = np.asarray(cloud, dtype=float)
        assert cloud.shape == (n_ensemble,)
        assert np.all(np.isfinite(cloud))


def test_lstm_run_raises_clear_error_when_series_shorter_than_lookback():
    lstm_model = importlib.import_module("lstm_model")
    train = synthetic_series(n=10, seed=0)   # shorter than SEQ_LEN=30
    test = synthetic_series(n=5, seed=1)

    with pytest.raises(ValueError, match="seq_len"):
        lstm_model.run_lstm(train, test, epochs=1)


def test_lstm_next_step_raises_clear_error_when_series_shorter_than_lookback():
    lstm_model = importlib.import_module("lstm_model")
    series = synthetic_series(n=10, seed=0)   # shorter than SEQ_LEN=30

    with pytest.raises(ValueError, match="seq_len"):
        lstm_model.next_step_lstm(series, epochs=1)
