import json
from pathlib import Path

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
