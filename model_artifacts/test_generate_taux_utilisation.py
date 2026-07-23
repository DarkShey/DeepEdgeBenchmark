from model_artifacts import generate_taux_utilisation as gtu


def test_band_thresholds():
    assert gtu._band(49.9) == "band-red"
    assert gtu._band(50.0) == "band-yellow"
    assert gtu._band(59.9) == "band-yellow"
    assert gtu._band(60.0) == "band-blue"
    assert gtu._band(69.9) == "band-blue"
    assert gtu._band(70.0) == "band-green"
    assert gtu._band(100.0) == "band-green"


def _row(counters):
    return {"signals": [{"counter": c} for c in counters]}


def test_usage_row_usable_if_any_signal_counter_positive():
    rows = [
        _row([1]),          # usable
        _row([-1, 2]),      # usable (au moins un signal positif)
        _row([-1, -2]),     # non usable
        _row([]),           # non usable (aucun signal déclenché)
        _row([None]),       # non usable (signal non résolu)
    ]
    usable, total = gtu._usage(rows)
    assert (usable, total) == (2, 5)


def test_compute_matrix_filters_real_flag_live(monkeypatch):
    fake_rows = {
        "BTC-USD": [
            {"model": "ARIMA-GARCH", "real_flag": "live", "signals": [{"counter": 1}]},
            {"model": "ARIMA-GARCH", "real_flag": "oos", "signals": [{"counter": 1}]},
            {"model": "LSTM", "real_flag": "live", "signals": [{"counter": -1}]},
        ],
    }

    def fake_daily_detail(db_path, asset, models):
        return fake_rows[asset]

    monkeypatch.setattr(gtu.st, "daily_detail", fake_daily_detail)

    matrix = gtu.compute_matrix(db_path="unused", models=["ARIMA-GARCH", "LSTM"],
                                 assets=["BTC-USD"])

    arima = next(r for r in matrix if r["model"] == "ARIMA-GARCH")
    lstm = next(r for r in matrix if r["model"] == "LSTM")
    assert arima["cells"] == [(1, 1)]  # la ligne 'oos' est bien exclue
    assert lstm["cells"] == [(0, 1)]


def test_render_html_applies_band_class_and_shows_fraction():
    matrix = [{"model": "ARIMA-GARCH", "cells": [(7, 10), (0, 0)]}]
    html = gtu.render_html(matrix, ["BTC-USD", "ETH-USD"], "2026-07-21 12:00 +0200")

    assert 'class="cell band-green"' in html  # 7/10 = 70,0 %
    assert "7 / 10" in html
    assert ">—<" in html  # cellule à total=0
    assert "ARIMA-GARCH" in html
    assert "BTC-USD" in html and "ETH-USD" in html
