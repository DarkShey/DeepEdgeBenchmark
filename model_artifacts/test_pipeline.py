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
    "pi_width_min", "pi_width_mean", "pi_width_max", "crps",
    "n_val", "horizon", "asset", "model",
    # Métriques de skill vs baseline persistence (Point 1 du brief, cf.
    # _honest_skill_metrics dans pipeline.py) : préexistant, cette liste était restée
    # périmée depuis leur introduction (commit 33b98cc) -- corrigé au passage, sans
    # rapport avec le CRPS.
    "theil_u", "MASE", "change_corr", "dir_acc_change", "dir_acc_ci95",
    "dir_acc_p_vs_coin", "dm_stat", "dm_p", "dm_lag", "skill_vs_naive",
}

# CRPS n'est calculé que pour D1 (Gate2 walk-forward existant, cf. crps_kpis.py) --
# D7 (backtest rolling-origin, pas encore supporté) doit avoir crps=None, jamais un
# nombre halluciné. LSTM est aussi exclu ici : ce test appelle evaluate_gate2
# directement (pas le pipeline réel), qui pour LSTM bascule sur lstm_worker_result
# -- son CRPS est calculé côté worker, cf. model_artifacts/lstm_worker.py.
MODELS_WITH_D1_CRPS = {"ARIMA-GARCH", "SARIMA", "Prophet", "Naive"}


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

    # CRPS (bootstrap des résidus, bande gaussienne pour Naive, cf. crps_kpis.py) :
    # seulement pour D1 sur MODELS_WITH_D1_CRPS -- D7 et LSTM (calculé côté worker en
    # pipeline réel, pas ici) doivent rester à None, jamais halluciné.
    if horizon_label == "D1" and model_key in MODELS_WITH_D1_CRPS:
        assert payload["crps"] is not None and np.isfinite(payload["crps"])
        assert payload["crps"] >= 0
    else:
        assert payload["crps"] is None

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


# ── --full-retrain=False (défaut) : réutilisation Gate1/Gate2 d'un run antérieur ────────

def test_find_reusable_run_dir_ignores_same_or_later_dates(tmp_path):
    (tmp_path / "20260101-SARIMA-SYN-D1").mkdir()
    (tmp_path / "20260101-SARIMA-SYN-D1" / "metrics.json").write_text("{}")
    (tmp_path / "20260103-SARIMA-SYN-D1").mkdir()
    (tmp_path / "20260103-SARIMA-SYN-D1" / "metrics.json").write_text("{}")

    found = mp.find_reusable_run_dir("SARIMA", "SYN", "D1", "20260102", "metrics.json", run_root=tmp_path)
    assert found is not None and found.name == "20260101-SARIMA-SYN-D1"

    # 20260101 n'est pas strictement antérieur à lui-même -> pas réutilisable pour ce jour-là.
    assert mp.find_reusable_run_dir("SARIMA", "SYN", "D1", "20260101", "metrics.json", run_root=tmp_path) is None
    # Aucun run pour cet horizon/actif -> None (bascule sur calcul complet côté appelant).
    assert mp.find_reusable_run_dir("SARIMA", "OTHER", "D1", "20260201", "metrics.json", run_root=tmp_path) is None


def test_reuse_gate2_payload_returns_none_if_metrics_incomplete(tmp_path):
    prev_dir, out_dir = tmp_path / "prev", tmp_path / "out"
    prev_dir.mkdir(); out_dir.mkdir()
    (prev_dir / "metrics.json").write_text(json.dumps({"RMSE": 1.0}))  # clés Gate2 manquantes

    assert mp.reuse_gate2_payload(prev_dir, out_dir) is None


def test_reuse_gate2_payload_copies_predictions_and_strips_forecast(tmp_path):
    prev_dir, out_dir = tmp_path / "prev", tmp_path / "out"
    prev_dir.mkdir(); out_dir.mkdir()
    metrics = {
        "RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "directional_accuracy": 50.0, "pi_coverage_95": 95.0,
        "forecast": {"predicted": 999.0},
    }
    (prev_dir / "metrics.json").write_text(json.dumps(metrics))
    (prev_dir / "predictions.parquet").write_bytes(b"parquet-bytes")

    payload = mp.reuse_gate2_payload(prev_dir, out_dir)

    assert payload is not None
    assert "forecast" not in payload
    assert payload["RMSE"] == 1.0
    assert (out_dir / "predictions.parquet").read_bytes() == b"parquet-bytes"


def test_full_retrain_false_reuses_previous_run_without_recomputing(tmp_path, monkeypatch, synthetic_split):
    """Cas nominal de --full-retrain=False : un run antérieur exploitable existe -> Gate1
    (fit_and_serialize) et Gate2 (evaluate_gate2) ne sont PAS rappelés (prouvé en les
    faisant lever si appelés, pas juste en comparant des valeurs -- qui seraient
    trivialement égales sur des données synthétiques identiques). Seule la prévision live
    et les fichiers non liés à Gate1/Gate2 (prices.parquet, unit_tests.json) sont
    recalculés chaque jour, comme aujourd'hui."""
    train, validation = synthetic_split
    monkeypatch.setattr(mp, "RUN_ROOT", tmp_path)
    model_key = "SARIMA"

    common = dict(
        model_key=model_key, ticker="SYN", asset_class="test", train=train, validation=validation,
        window_start=str(train.index[0].date()), window_end=str(validation.index[-1].date()),
        seed=0, epochs=None, max_d7_origins=2, horizons=["D1", "D7"],
        regime_tag="unknown", db_path=str(tmp_path / "tracking.db"),
    )

    # Run 1 : rien à réutiliser encore (Run/ vide) -> calcul complet malgré full_retrain=False.
    logs1 = mp.process_asset_model(run_date_str="20260101", run_date_iso="2026-01-01",
                                   run_id="run-1", full_retrain=False, **common)
    assert all(log["gate1"] and log["gate2"] for log in logs1)

    monkeypatch.setattr(mp, "fit_and_serialize",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Gate1 rappelé à tort")))
    monkeypatch.setattr(mp, "evaluate_gate2",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Gate2 rappelé à tort")))

    # Run 2 (le lendemain) : le run 1 est exploitable -> Gate1/Gate2 réutilisés tels quels.
    logs2 = mp.process_asset_model(run_date_str="20260102", run_date_iso="2026-01-02",
                                   run_id="run-2", full_retrain=False, **common)
    assert all(log["gate1"] and log["gate2"] for log in logs2)

    for log1, log2 in zip(logs1, logs2):
        dir1, dir2 = Path(log1["dir"]), Path(log2["dir"])
        for filename in mp.SERIALIZED_FILES[model_key]:
            assert (dir1 / filename).read_bytes() == (dir2 / filename).read_bytes(), filename
        assert (dir1 / "predictions.parquet").read_bytes() == (dir2 / "predictions.parquet").read_bytes()

        metrics1 = json.loads((dir1 / "metrics.json").read_text())
        metrics2 = json.loads((dir2 / "metrics.json").read_text())
        metrics1.pop("forecast", None)
        metrics2.pop("forecast", None)
        assert metrics1 == metrics2

        metadata2 = json.loads((dir2 / "metadata.json").read_text())
        assert metadata2["gate1_reused_from"] == "20260101"
        assert metadata2["gate2_reused_from"] == "20260101"

    # Run 1 (le tout premier) n'a, lui, rien réutilisé.
    metadata1 = json.loads((Path(logs1[0]["dir"]) / "metadata.json").read_text())
    assert metadata1["gate1_reused_from"] is None
    assert metadata1["gate2_reused_from"] is None


def test_gate2_reuse_recomputes_when_prior_d1_run_predates_crps(tmp_path, monkeypatch, synthetic_split):
    """Bug réel constaté en prod (run du 23/07) : --full-retrain=False réutilisait
    indéfiniment un run D1 antérieur au déploiement du CRPS (metrics.json sans la clé
    "crps"), sans jamais le recalculer -- _gate2_metrics_ok ne vérifiait pas sa présence.
    Un tel run doit être traité comme non exploitable et déclencher un recalcul complet,
    qui doit alors produire un CRPS non-None (auto-guérison, sans --full-retrain manuel)."""
    train, validation = synthetic_split
    monkeypatch.setattr(mp, "RUN_ROOT", tmp_path)
    model_key = "SARIMA"

    # Simule un run antérieur "légitime" mais écrit avant l'existence du CRPS : metrics.json
    # a toutes les clés attendues par _gate2_metrics_ok, juste pas "crps".
    prev_dir = tmp_path / "20260101-SARIMA-SYN-D1"
    prev_dir.mkdir()
    legacy_payload = {
        "RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "directional_accuracy": 50.0,
        "pi_coverage_95": 95.0, "horizon": "D1", "asset": "SYN", "model": model_key,
        "n_val": len(validation),
    }
    (prev_dir / "metrics.json").write_text(json.dumps(legacy_payload))
    (prev_dir / "predictions.parquet").write_bytes(b"stub-parquet")
    (prev_dir / "hyperparams.json").write_text("{}")
    for filename in mp.SERIALIZED_FILES[model_key]:
        (prev_dir / filename).write_bytes(b"stub")

    logs = mp.process_asset_model(
        model_key=model_key, ticker="SYN", asset_class="test", train=train, validation=validation,
        window_start=str(train.index[0].date()), window_end=str(validation.index[-1].date()),
        seed=0, epochs=None, max_d7_origins=2, horizons=["D1"],
        regime_tag="unknown", db_path=str(tmp_path / "tracking.db"),
        run_date_str="20260102", run_date_iso="2026-01-02", run_id="run-2", full_retrain=False,
    )

    assert all(log["gate1"] and log["gate2"] for log in logs)
    out_dir = Path(logs[0]["dir"])
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["crps"] is not None and np.isfinite(metrics["crps"])
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["gate2_reused_from"] is None, "aurait dû recalculer, pas réutiliser le run sans CRPS"


def test_full_retrain_true_ignores_previous_run(tmp_path, monkeypatch, synthetic_split):
    """--full-retrain=True (opt-in) : comportement historique, tout est recalculé même si
    un run antérieur exploitable existe -- ne doit jamais tenter de réutiliser."""
    train, validation = synthetic_split
    monkeypatch.setattr(mp, "RUN_ROOT", tmp_path)
    common = dict(
        model_key="SARIMA", ticker="SYN", asset_class="test", train=train, validation=validation,
        window_start=str(train.index[0].date()), window_end=str(validation.index[-1].date()),
        seed=0, epochs=None, max_d7_origins=2, horizons=["D1", "D7"],
        regime_tag="unknown", db_path=str(tmp_path / "tracking.db"),
    )

    mp.process_asset_model(run_date_str="20260101", run_date_iso="2026-01-01",
                           run_id="run-1", full_retrain=True, **common)
    logs2 = mp.process_asset_model(run_date_str="20260102", run_date_iso="2026-01-02",
                                   run_id="run-2", full_retrain=True, **common)

    for log2 in logs2:
        metadata2 = json.loads((Path(log2["dir"]) / "metadata.json").read_text())
        assert metadata2["gate1_reused_from"] is None
        assert metadata2["gate2_reused_from"] is None


def test_lstm_full_retrain_false_asks_worker_to_skip_training(tmp_path, monkeypatch, synthetic_split):
    """LSTM tourne dans un sous-processus isolé (lstm_worker.py) -- ici on vérifie que
    process_asset_model lui demande bien de sauter Gate1 (skip_training=True) et ne lui
    redemande que les horizons Gate2 non réutilisables (aucun ici), sans lancer le vrai
    worker (subprocess + TensorFlow), pour un test rapide et ciblé sur le câblage."""
    train, validation = synthetic_split
    monkeypatch.setattr(mp, "RUN_ROOT", tmp_path)

    # Simule un run antérieur exploitable pour LSTM (D1 et D7).
    for horizon_label in ("D1", "D7"):
        prev_dir = tmp_path / f"20260101-LSTM-SYN-{horizon_label}"
        prev_dir.mkdir()
        for filename in mp.SERIALIZED_FILES["LSTM"]:
            (prev_dir / filename).write_bytes(b"stub")
        (prev_dir / "metrics.json").write_text(json.dumps({
            "RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "directional_accuracy": 50.0, "pi_coverage_95": 95.0,
        }))
        (prev_dir / "predictions.parquet").write_bytes(b"stub-parquet")

    captured = {}

    def fake_worker(train, validation, out_dir, seed, epochs, max_d7_origins,
                    all_horizons, gate2_horizons, business_h_days, skip_training):
        captured["skip_training"] = skip_training
        captured["gate2_horizons"] = list(gate2_horizons)
        return {"gate1_ok": False, "gate2": {},
                "live_forecast": {1: (100.0, 90.0, 110.0), 7: (105.0, 95.0, 115.0)}}
    monkeypatch.setattr(mp, "_run_lstm_via_worker", fake_worker)

    logs = mp.process_asset_model(
        "LSTM", "SYN", "test", train, validation,
        run_date_str="20260102", run_date_iso="2026-01-02",
        window_start=str(train.index[0].date()), window_end=str(validation.index[-1].date()),
        seed=0, epochs=2, max_d7_origins=2, horizons=["D1", "D7"],
        run_id="test-run", regime_tag="unknown", db_path=str(tmp_path / "tracking.db"),
        full_retrain=False,
    )

    assert captured["skip_training"] is True
    assert captured["gate2_horizons"] == []  # D1 et D7 tous deux réutilisables ici
    assert all(log["gate1"] and log["gate2"] for log in logs)
    for log in logs:
        out_dir = Path(log["dir"])
        assert (out_dir / "model.h5").read_bytes() == b"stub"
        metadata = json.loads((out_dir / "metadata.json").read_text())
        assert metadata["gate1_reused_from"] == "20260101"
        assert metadata["gate2_reused_from"] == "20260101"
