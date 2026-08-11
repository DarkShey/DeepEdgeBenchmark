import json
from pathlib import Path

import pytest

from model_artifacts import generate_dashboard as gd


def test_collect_run_data_computes_rmse_ratio_vs_naive(tmp_path):
    run_root = tmp_path / "Run"
    run_root.mkdir()

    def write_combo(name, model, asset, horizon, rmse):
        combo_dir = run_root / name
        combo_dir.mkdir()
        (combo_dir / "metrics.json").write_text(json.dumps({
            "model": model,
            "asset": asset,
            "horizon": horizon,
            "RMSE": rmse,
            "MAE": 1.0,
            "MAPE": 0.0,
            "directional_accuracy": 0.0,
            "pi_coverage_95": 0.0,
            "pi_width_min": 0.0,
            "pi_width_mean": 0.0,
            "pi_width_max": 0.0,
            "n_val": 5,
        }))

    write_combo("20260101-ARIMA-BTC-USD-D1", "ARIMA-GARCH", "BTC-USD", "D1", 2.0)
    write_combo("20260101-Naive-BTC-USD-D1", "Naive", "BTC-USD", "D1", 1.0)
    write_combo("20260101-LSTM-BTC-USD-D1", "LSTM", "BTC-USD", "D1", 4.0)

    data = gd.collect_run_data(run_root)
    records = data["records"]

    arima = next(r for r in records if r["model"] == "ARIMA-GARCH")
    lstm = next(r for r in records if r["model"] == "LSTM")
    naive = next(r for r in records if r["model"] == "Naive")

    assert arima["rmse_vs_naive"] == 2.0
    assert lstm["rmse_vs_naive"] == 4.0
    assert naive["rmse_vs_naive"] is None


# ── collect_rolling_coverage (monitoring calibration, 2026-07-31) ─────────────

from validation import tracking_db as td


def _insert_realized_pred(db_path, cutoff, target, y_true, horizon=1, model="SARIMA",
                          asset="BTC-USD"):
    """Même idiome que validation/test_sigma_scale.py : insertion via
    save_prediction (contrat RECORD_FIELDS) puis résolution via evaluate_pending
    (le mécanisme du cron quotidien) -- jamais de SQL d'écriture maison."""
    record = {
        "run_id": "run1", "tc_id": f"TC_{asset}_D{horizon}_{cutoff}",
        "model": model, "asset": asset,
        "horizon": horizon, "cutoff_date": cutoff, "target_date": target,
        "regime": "calm", "last_close": 100.0, "y_pred": 100.0,
        "y_lower": 95.0, "y_upper": 105.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": f"{cutoff}T12:00:00",
    }
    assert td.save_prediction(record, db_path=db_path) is True
    n = td.evaluate_pending(lambda a, t: y_true, db_path=db_path, today="2099-01-01")
    assert n == 1


def test_collect_rolling_coverage_counts_hits_and_flags_undercoverage(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    # 10 prédictions D+1 résolues : 5 dans la bande [95, 105], 5 dehors -> 50 %
    for i in range(10):
        y_true = 100.0 if i % 2 == 0 else 120.0
        _insert_realized_pred(db_path, f"2026-07-{i + 1:02d}", f"2026-07-{i + 2:02d}", y_true)

    out = gd.collect_rolling_coverage(db_path=db_path, window=4)
    cell = out["BTC-USD"]["SARIMA"]["D+1"]   # horizon_unit NULL -> fallback D+1

    assert cell["n_all"] == 10
    assert cell["cov95_all"] == 50.0
    assert cell["n_window"] == 4
    assert cell["status"] == "low"           # 50 % << 95 % - 2 sigma binomial
    assert cell["n_live"] == 10              # save_prediction -> source='live'
    assert cell["cov95_live"] == 50.0


def test_collect_rolling_coverage_full_hit_window_is_flagged_wide(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    for i in range(6):
        _insert_realized_pred(db_path, f"2026-07-{i + 1:02d}", f"2026-07-{i + 2:02d}", 100.0)

    out = gd.collect_rolling_coverage(db_path=db_path, window=6)
    cell = out["BTC-USD"]["SARIMA"]["D+1"]
    assert cell["cov95_window"] == 100.0
    assert cell["status"] == "wide"          # 100 % sur fenêtre pleine


def test_collect_rolling_coverage_missing_db_is_silent():
    assert gd.collect_rolling_coverage(db_path="/nonexistent/tracking.db") == {}


# ── collect_monthly_kpis (BRIEF_dashboard_mensuel_et_maj_bdd.md, régime C) ─────


def _insert_monthly_pred(db_path, cutoff, target, y_true=None, model="TSDiff",
                         asset="BTC-USD", horizon_unit="M+1", frequence="monthly",
                         horizon_type="monthly", y_lower=95.0, y_upper=105.0,
                         y_pred=100.0):
    """Même idiome que _insert_realized_pred (via save_prediction -- jamais de SQL
    d'écriture maison), avec frequence/horizon_type/horizon_unit explicites -- ceux-ci
    sont respectés tels quels par save_prediction quand fournis (cf. sa docstring).
    y_true=None laisse la ligne non résolue (n_total > n_realized)."""
    horizon = {"M+1": 1, "M+2": 2, "M+3": 3}[horizon_unit]
    record = {
        "run_id": "monthlytestrun", "tc_id": f"TC_{asset}_{model}_{horizon_unit}_{cutoff}",
        "model": model, "asset": asset,
        "horizon": horizon, "cutoff_date": cutoff, "target_date": target,
        "regime": "calm", "last_close": 100.0, "y_pred": y_pred,
        "y_lower": y_lower, "y_upper": y_upper,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": f"{cutoff}T12:00:00",
        "frequence": frequence, "horizon_type": horizon_type, "horizon_unit": horizon_unit,
    }
    assert td.save_prediction(record, db_path=db_path) is True
    if y_true is not None:
        n = td.evaluate_pending(lambda a, t: y_true, db_path=db_path, today="2099-01-01")
        assert n == 1


def test_collect_monthly_kpis_basic_cells_and_cov95(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    # 3 lignes M+1 résolues pour TSDiff/BTC-USD : 2 dans la bande [95,105], 1 dehors
    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=100.0)
    _insert_monthly_pred(db_path, "2026-02-28", "2026-03-31", y_true=102.0)
    _insert_monthly_pred(db_path, "2026-03-31", "2026-04-30", y_true=120.0)
    # NsDiff, même actif, autre horizon -- doit apparaître comme cellule séparée
    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=101.0,
                         model="NsDiff", horizon_unit="M+2")
    # Un autre actif (5 actifs du panel, cf. brief §6)
    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=100.0,
                         asset="SPY")
    # 1 ligne non résolue insérée EN DERNIER (compte dans n_total, pas dans
    # n_realized) -- évaluée après les autres, evaluate_pending() résoudrait sinon
    # TOUTES les lignes pending du run (pas seulement la dernière insérée).
    _insert_monthly_pred(db_path, "2026-04-30", "2026-05-31", y_true=None)

    out = gd.collect_monthly_kpis(db_path=db_path)

    btc_tsdiff_m1 = out["BTC-USD"]["TSDiff"]["M+1"]
    assert btc_tsdiff_m1["n_total"] == 4
    assert btc_tsdiff_m1["n_realized"] == 3
    assert btc_tsdiff_m1["cov95_exact"] == pytest.approx(2 / 3)
    assert btc_tsdiff_m1["crps_is_approx"] is True   # TSDiff/NsDiff toujours approx

    assert "NsDiff" in out["BTC-USD"]
    assert "M+2" in out["BTC-USD"]["NsDiff"]
    assert out["SPY"]["TSDiff"]["M+1"]["n_total"] == 1


def test_collect_monthly_kpis_missing_db_is_silent():
    assert gd.collect_monthly_kpis(db_path="/nonexistent/tracking.db") == {}


def test_collect_monthly_kpis_excludes_regime_b(tmp_path):
    """Régime B (frequence='daily', horizon_type='monthly', daily poussé à la fin de
    mois) reste hors affichage mensuel -- seul le régime C (frequence='monthly')
    doit remonter (brief §3.1)."""
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=100.0)   # régime C
    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=999.0,   # régime B
                         frequence="daily", model="ARIMA-GARCH")

    out = gd.collect_monthly_kpis(db_path=db_path)

    assert out["BTC-USD"]["TSDiff"]["M+1"]["n_total"] == 1   # pas 2
    assert "ARIMA-GARCH" not in out.get("BTC-USD", {})       # pas un modèle monthly de toute façon


def test_weekly_row_samples_reused_by_monthly(tmp_path, monkeypatch):
    """_weekly_row_samples (agnostique à la fréquence) doit être RÉUTILISÉE par
    collect_monthly_kpis, pas réimplémentée -- test d'identité par espionnage plutôt
    que par lecture de source (brief §5 test 4)."""
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=100.0)

    calls = []
    original = gd._weekly_row_samples

    def spy(*args, **kwargs):
        calls.append(args[0])   # le modèle
        return original(*args, **kwargs)

    monkeypatch.setattr(gd, "_weekly_row_samples", spy)
    gd.collect_monthly_kpis(db_path=db_path)
    assert calls == ["TSDiff"]


# ── collect_data_freshness (B1 -- stamp de provenance) ─────────────────────────


def test_collect_data_freshness_reports_max_cutoff_and_run_id(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    _insert_monthly_pred(db_path, "2026-01-31", "2026-02-28", y_true=100.0)
    _insert_monthly_pred(db_path, "2026-02-28", "2026-03-31", y_true=100.0)
    _insert_realized_pred(db_path, "2026-07-01", "2026-07-08", 100.0,
                          horizon=1, model="TSDiff")
    # Cette ligne weekly doit être ignorée par la piste "monthly" (et réciproquement)
    td.save_prediction({
        "run_id": "weeklytestrun", "tc_id": "TC_weekly_1", "model": "TSDiff",
        "asset": "BTC-USD", "horizon": 1, "cutoff_date": "2026-07-05",
        "target_date": "2026-07-12", "regime": "calm", "last_close": 100.0,
        "y_pred": 100.0, "y_lower": 95.0, "y_upper": 105.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1, "created_at": "2026-07-05T12:00:00",
        "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": "W+1",
    }, db_path=db_path)

    out = gd.collect_data_freshness(db_path=db_path)

    assert out["monthly"]["max_cutoff_date"] == "2026-02-28"
    assert out["monthly"]["n_origins"] == 2
    assert out["monthly"]["run_id_latest"] == "monthlytestrun"
    assert out["weekly"]["max_cutoff_date"] == "2026-07-05"
    assert out["weekly"]["n_origins"] == 1
    assert out["weekly"]["run_id_latest"] == "weeklytestrun"


def test_collect_data_freshness_missing_db_is_silent():
    assert gd.collect_data_freshness(db_path="/nonexistent/tracking.db") == {}
