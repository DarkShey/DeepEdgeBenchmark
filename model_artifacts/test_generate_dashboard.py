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
