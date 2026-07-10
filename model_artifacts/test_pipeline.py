"""
model_artifacts/test_pipeline.py — suite hors-ligne (aucun accès réseau) pour le
pipeline train+validate (cf. BRIEF_model_artifacts.md §10).

Données synthétiques déterministes (jamais de fetch_data/yfinance) : vérifie que les
2 gates tournent et loguent PASS, que les fichiers d'artefacts attendus sont créés,
que metrics.json contient les bonnes clés, et surtout le round-trip de sérialisation
(le modèle rechargé redonne les mêmes prévisions que l'objet fitté original — la
preuve que l'artefact est exploitable pour un déploiement, pas juste un octet vide).
"""

# Le GPU/Metal est désactivé pour TensorFlow par model_artifacts/conftest.py
# (chargé par pytest avant ce module) — cf. sa docstring pour le détail du bug.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_artifacts import pipeline as mp

MODELS_TO_TEST = ["ARIMA-GARCH", "SARIMA", "Prophet", "LSTM"]
EXPECTED_METRICS_KEYS = {
    "RMSE", "MAE", "MAPE", "directional_accuracy", "pi_coverage_95",
    "pi_width_min", "pi_width_mean", "pi_width_max",
    "n_val", "horizon", "asset", "model", "honest_eval",
}
EXPECTED_HONEST_EVAL_KEYS = {
    "mase", "theils_u", "variation_correlation",
    "directional_accuracy_variations", "diebold_mariano", "no_better_than_naive",
}


def synthetic_series(n=160, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def synthetic_split():
    return mp.chronological_split(synthetic_series())


def _epochs_for(model_key):
    return 2 if model_key == "LSTM" else None


@pytest.mark.parametrize("model_key", MODELS_TO_TEST)
def test_gate1_passes_and_serializes_expected_files(model_key, tmp_path, synthetic_split):
    train, _ = synthetic_split
    out_dir = tmp_path / f"{model_key}-gate1"

    fitted, gate1_ok = mp.fit_and_serialize(model_key, train, out_dir, seed=0, epochs=_epochs_for(model_key))

    assert gate1_ok is True
    assert fitted is not None
    for filename in mp.SERIALIZED_FILES[model_key]:
        assert (out_dir / filename).exists(), f"{filename} manquant pour {model_key}"


@pytest.mark.parametrize("model_key", MODELS_TO_TEST + ["Naive"])
@pytest.mark.parametrize("horizon_label", ["D1", "D7"])
def test_gate2_passes_and_metrics_have_expected_keys(model_key, horizon_label, synthetic_split):
    train, validation = synthetic_split

    payload, gate2_ok, series = mp.evaluate_gate2(
        model_key, "SYN", train, validation, horizon_label,
        seed=0, epochs=_epochs_for(model_key), max_d7_origins=2,
    )

    assert gate2_ok is True
    assert set(payload.keys()) == EXPECTED_METRICS_KEYS
    assert payload["horizon"] == horizon_label
    assert payload["asset"] == "SYN"
    assert payload["model"] == model_key
    assert payload["n_val"] > 0
    assert all(np.isfinite(payload[k]) for k in
              ("RMSE", "MAE", "MAPE", "directional_accuracy", "pi_coverage_95",
               "pi_width_min", "pi_width_mean", "pi_width_max"))
    assert payload["pi_width_min"] <= payload["pi_width_mean"] <= payload["pi_width_max"]

    # Point 1 du brief d'amélioration : métriques vs baseline naïve sur les variations.
    assert payload["honest_eval"] is not None, "honest_eval n'a pas pu être calculé"
    assert set(payload["honest_eval"].keys()) == EXPECTED_HONEST_EVAL_KEYS
    if model_key == "Naive":
        # Naive comparé à lui-même : aucun skill par construction (U=1, MASE=1).
        assert payload["honest_eval"]["theils_u"] == pytest.approx(1.0)
        assert payload["honest_eval"]["mase"] == pytest.approx(1.0)
        assert payload["honest_eval"]["diebold_mariano"]["verdict"] == "identical_predictions"

    # series alimente predictions.parquet (cf. write_predictions_parquet) : un point par
    # jour de validation (D1) ou par origine glissante (D7), toutes les listes alignées.
    n_points = payload["n_val"]
    assert {"dates", "actual", "predicted", "pi_lower", "pi_upper"} == set(series.keys())
    assert all(len(series[k]) == n_points for k in series)


@pytest.mark.parametrize("model_key", MODELS_TO_TEST)
def test_round_trip_serialization_gives_same_forecast(model_key, tmp_path, synthetic_split):
    """La preuve que l'artefact est exploitable : recharger model.* depuis le disque
    doit redonner (à la tolérance numérique près) la même prévision que l'objet fitté
    en mémoire au moment de l'entraînement."""
    train, _ = synthetic_split
    out_dir = tmp_path / f"{model_key}-roundtrip"

    fitted, gate1_ok = mp.fit_and_serialize(model_key, train, out_dir, seed=0, epochs=_epochs_for(model_key))
    assert gate1_ok is True

    original = mp.HANDLERS[model_key]["forecast"](fitted, [1, 5])
    reloaded_fitted = mp.reload_model(model_key, out_dir, train)
    reloaded = mp.HANDLERS[model_key]["forecast"](reloaded_fitted, [1, 5])

    for h in (1, 5):
        orig_point, orig_lo, orig_hi = original[h]
        new_point, new_lo, new_hi = reloaded[h]
        assert new_point == pytest.approx(orig_point, rel=1e-3)
        assert new_lo == pytest.approx(orig_lo, rel=1e-2)
        assert new_hi == pytest.approx(orig_hi, rel=1e-2)


def test_process_asset_model_produces_all_five_files_for_lstm(tmp_path, monkeypatch, synthetic_split):
    """LSTM est le seul modèle qui peuple les 5 emplacements du doc DEITA
    (model + scaler + hyperparams + metrics + metadata) — cas le plus complet
    pour vérifier l'orchestration Gate1+Gate2+métadonnées bout en bout."""
    train, validation = synthetic_split
    monkeypatch.setattr(mp, "RUN_ROOT", tmp_path)

    logs = mp.process_asset_model(
        "LSTM", "SYN", "test", train, validation,
        run_date_str="20260101", run_date_iso="2026-01-01",
        window_start=str(train.index[0].date()), window_end=str(validation.index[-1].date()),
        seed=0, epochs=2, max_d7_origins=2, horizons=["D1", "D7"],
        run_id="test-run", regime_tag="unknown", db_path=str(tmp_path / "tracking.db"),
    )

    assert len(logs) == 2
    assert all(log["gate1"] for log in logs)
    assert all(log["gate2"] for log in logs)

    for log in logs:
        out_dir = Path(log["dir"])
        for filename in ("model.h5", "scaler.pkl", "hyperparams.json", "metrics.json", "metadata.json",
                         "predictions.parquet", "prices.parquet"):
            assert (out_dir / filename).exists(), f"{filename} manquant dans {out_dir}"
        # Prévision hors-échantillon repliée dans metrics.json (pas de forecast.json séparé,
        # cf. process_asset_model -- ex-doublon avec business_validation.json).
        assert not (out_dir / "forecast.json").exists()
        metrics = json.loads((out_dir / "metrics.json").read_text())
        assert {"last_date", "last_price", "predicted", "pi_lower", "pi_upper"} <= metrics["forecast"].keys()

    # même modèle (fit une fois) -> model.h5/scaler.pkl identiques dans D1 et D7 (§12)
    d1_dir = Path(logs[0]["dir"])
    d7_dir = Path(logs[1]["dir"])
    assert (d1_dir / "model.h5").read_bytes() == (d7_dir / "model.h5").read_bytes()
    assert (d1_dir / "scaler.pkl").read_bytes() == (d7_dir / "scaler.pkl").read_bytes()

    # predictions.parquet (D1) : un point par jour de validation, colonnes attendues.
    preds_d1 = pd.read_parquet(d1_dir / "predictions.parquet")
    assert list(preds_d1.columns) == ["date", "actual", "predicted", "pi_lower", "pi_upper"]
    assert len(preds_d1) == len(validation)

    # prices.parquet : historique complet (train + validation), identique dans D1 et D7
    # (même redondance assumée que metadata.json, cf. BRIEF §12).
    prices_d1 = pd.read_parquet(d1_dir / "prices.parquet")
    prices_d7 = pd.read_parquet(d7_dir / "prices.parquet")
    assert list(prices_d1.columns) == ["date", "close"]
    assert len(prices_d1) == len(train) + len(validation)
    pd.testing.assert_frame_equal(prices_d1, prices_d7)
