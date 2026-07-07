"""
validation/test_integration.py — flux complet save_prediction -> evaluate_pending -> report

Un seul test, hors-ligne (price_fetcher mocké, aucun accès réseau) : vérifie que la
consolidation (une seule tracking_db, auto-initialisante) fonctionne de bout en bout
sur un jeu de records conformes au contrat des 15 champs (RECORD_FIELDS).
"""

from datetime import datetime, timedelta

from validation import tracking_db as td


def _record(**overrides):
    record = {
        "run_id": "integ-run", "tc_id": "TC_BTC-USD_D1", "model": "arima", "asset": "BTC-USD",
        "horizon": 1, "cutoff_date": "2026-06-01", "target_date": "2026-06-02",
        "regime": "calm", "last_close": 100.0, "y_pred": 101.0,
        "y_lower": 95.0, "y_upper": 107.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": "2026-06-01T00:00:00",
    }
    record.update(overrides)
    return record


def test_full_flow_save_evaluate_report(tmp_path):
    db_path = str(tmp_path / "integration.db")
    # pas d'appel explicite à init_db() : on vérifie l'auto-initialisation paresseuse
    # (Étape 3) exactement comme le fait validation/generate_test_cases.py.

    record = _record()
    inserted = td.save_prediction(record, db_path=db_path)
    assert inserted is True

    # rejouer le même record ne duplique rien (idempotence bout-en-bout).
    assert td.save_prediction(record, db_path=db_path) is False

    prices = {("BTC-USD", "2026-06-02"): 102.0}
    fake_fetcher = lambda asset, target_date: prices.get((asset, target_date))

    n_evaluated = td.evaluate_pending(fake_fetcher, db_path=db_path, today="2026-06-03")
    assert n_evaluated == 1

    by_model = {row["model"]: row for row in td.report(group_by=("model",), db_path=db_path)}
    assert by_model["arima"]["n_total"] == 1
    assert by_model["arima"]["n_evalues"] == 1
    assert by_model["arima"]["couverture_ic95"] == 1.0     # 95 <= 102 <= 107
    assert by_model["arima"]["taux_bat_naif"] == 1.0        # |102-101|=1 <= |102-100|=2
    assert by_model["arima"]["exactitude_dir"] == 1.0       # sign(101-100)==sign(102-100)

    by_model_regime = {(row["model"], row["regime"]): row
                       for row in td.report(group_by=("model", "regime"), db_path=db_path)}
    assert ("arima", "calm") in by_model_regime
    assert by_model_regime[("arima", "calm")]["n_evalues"] == 1
