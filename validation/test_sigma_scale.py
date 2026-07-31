"""
validation/test_sigma_scale.py — Chantier 2 de BRIEF_branchement_prod_calibration_sigma.md.

Vérifie sigma_scale.sigma_scale() : causalité stricte (aucune origine >= la date de la
nouvelle prévision), formule EWMA(z^2) exacte, filtrage (modèle x actif x horizon), état
neutre (1.0) sans historique. Insertion via tracking_db.save_prediction (jamais d'INSERT
maison, cf. garde-fou du brief) + tracking_db.evaluate_pending (même chemin que le cron
quotidien) pour résoudre y_true -- pas de SQL d'écriture direct sur `predictions`.
"""

import math

import pytest

from validation import tracking_db as td
from validation import sigma_scale as ss

_Z975 = 1.959963984540054


def make_record(**overrides):
    record = {
        "run_id": "run1", "tc_id": None, "model": "SARIMA", "asset": "BTC-USD",
        "horizon": 1, "cutoff_date": "2026-07-01", "target_date": "2026-07-02",
        "regime": "calm", "last_close": 100.0, "y_pred": 100.0,
        "y_lower": 95.0, "y_upper": 105.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": "2026-07-01T12:00:00",
    }
    record.update(overrides)
    record["tc_id"] = record["tc_id"] or f"TC_{record['asset']}_D{record['horizon']}_{record['cutoff_date']}"
    return record


def _insert_realized(db_path, cutoff_date, target_date, y_true, **overrides):
    """Insère via save_prediction (chemin d'insertion existant, contrat RECORD_FIELDS)
    puis résout y_true via evaluate_pending (même mécanisme que le cron quotidien,
    validation/evaluate_daily.py) -- jamais d'UPDATE/INSERT maison sur `predictions`."""
    record = make_record(cutoff_date=cutoff_date, target_date=target_date, **overrides)
    assert td.save_prediction(record, db_path=db_path) is True
    n = td.evaluate_pending(lambda asset, tdate: y_true, db_path=db_path, today="2099-01-01")
    assert n == 1


def test_sigma_scale_is_neutral_one_without_history(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    assert ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-15", db_path) == pytest.approx(1.0)


def test_sigma_scale_excludes_rows_at_or_after_as_of_date(tmp_path):
    """Causalité stricte : une origine (cutoff_date) EGALE à as_of_cutoff_date, ou
    postérieure, ne doit jamais entrer dans l'état EWMA."""
    db_path = str(tmp_path / "tracking.db")
    # y_true très loin de y_pred -> z énorme si (a tort) inclus dans l'état.
    _insert_realized(db_path, "2026-07-10", "2026-07-11", y_true=100.0)   # z=0 (neutre)
    _insert_realized(db_path, "2026-07-11", "2026-07-12", y_true=500.0)   # z énorme
    _insert_realized(db_path, "2026-07-12", "2026-07-13", y_true=900.0)   # z énorme

    # as_of = 2026-07-11 : seule la ligne d'origine 07-10 (z=0) est strictement antérieure
    # -- un seul pas EWMA depuis l'état neutre 1.0 : s2 = lambda*1 + (1-lambda)*0 = lambda.
    # Si les lignes 07-11/07-12 (z énorme) fuitaient, sqrt(s2) serait >> 1 (loin de sqrt(lambda)).
    scale = ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-11", db_path)
    assert scale == pytest.approx(math.sqrt(ss.EWMA_LAMBDA), rel=1e-12)
    assert scale < 1.5   # garde-fou lisible : bien loin de ce que donnerait une fuite

    # as_of encore plus tôt : rien d'antérieur du tout -> état neutre 1.0.
    scale_none = ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-10", db_path)
    assert scale_none == pytest.approx(1.0)


def test_sigma_scale_matches_manual_ewma_formula(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    sigma_own = (105.0 - 95.0) / (2.0 * _Z975)   # y_lower/y_upper par défaut de make_record

    y_trues = [103.0, 90.0, 108.0]   # z != 0 à chaque étape
    dates = ["2026-07-10", "2026-07-11", "2026-07-12"]
    for cutoff, y_true in zip(dates, y_trues):
        target = {"2026-07-10": "2026-07-11", "2026-07-11": "2026-07-12",
                  "2026-07-12": "2026-07-13"}[cutoff]
        _insert_realized(db_path, cutoff, target, y_true=y_true)

    got = ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-13", db_path)

    s2 = 1.0
    for y_true in y_trues:
        z = (y_true - 100.0) / sigma_own
        s2 = ss.EWMA_LAMBDA * s2 + (1 - ss.EWMA_LAMBDA) * z * z
    expected = math.sqrt(s2)

    assert got == pytest.approx(expected, rel=1e-12)


def test_sigma_scale_is_scoped_to_model_asset_horizon(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    # Ligne mécalibrée pour un AUTRE modèle sur le même actif/horizon/date -- ne doit
    # jamais fuiter dans le calcul pour "SARIMA".
    _insert_realized(db_path, "2026-07-10", "2026-07-11", y_true=900.0, model="Prophet")
    # Ligne mécalibrée pour le même modèle mais un AUTRE actif -- idem.
    _insert_realized(db_path, "2026-07-10", "2026-07-11", y_true=900.0, model="SARIMA", asset="SPY")
    # Ligne mécalibrée pour le même modèle/actif mais un AUTRE horizon -- idem.
    _insert_realized(db_path, "2026-07-10", "2026-07-11", y_true=900.0, model="SARIMA", horizon=7)

    scale = ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-15", db_path)
    assert scale == pytest.approx(1.0)   # aucune ligne SARIMA/BTC-USD/horizon=1 réalisée


def test_sigma_scale_ignores_daily_duplicate_rows(tmp_path):
    """daily_duplicate=1 (cf. flag_daily_duplicates) ne doit jamais entrer dans l'état,
    même mécanisme d'exclusion que experiments/prob_kpi_common.load_matrix_rows."""
    db_path = str(tmp_path / "tracking.db")
    _insert_realized(db_path, "2026-07-10", "2026-07-11", y_true=900.0)   # z énorme
    # Marque la ligne comme doublon quotidien (même mécanisme que flag_daily_duplicates,
    # pas un INSERT -- une mise à jour de métadonnées sur une ligne déjà légitimement
    # insérée via save_prediction).
    conn = td._connect(db_path)
    try:
        conn.execute("UPDATE predictions SET daily_duplicate = 1 WHERE model='SARIMA'")
        conn.commit()
    finally:
        conn.close()

    scale = ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-15", db_path)
    assert scale == pytest.approx(1.0)


def test_sigma_scale_widens_after_a_miscalibrated_realized_prediction(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    _insert_realized(db_path, "2026-07-10", "2026-07-11", y_true=300.0)   # très loin de y_pred=100
    scale = ss.sigma_scale("SARIMA", "BTC-USD", 1, "2026-07-15", db_path)
    assert scale > 1.5   # z large -> EWMA(z^2) > 1 -> sqrt > 1 (élargit la bande)
