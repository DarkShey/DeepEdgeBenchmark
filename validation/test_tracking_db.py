"""
test_tracking_db.py — suite hors-ligne (aucun accès réseau) pour tracking_db.py.

Chaque test utilise une base SQLite temporaire (tmp_path de pytest) et, quand un
price_fetcher est nécessaire, un mock déterministe — jamais price_fetcher.py /
yfinance (cf. BRIEF_tracking_db.md §9.7).
"""

import sqlite3

import pytest

from validation import tracking_db as td


def make_record(**overrides):
    record = {
        "run_id": "run1",
        "tc_id": "TC1",
        "model": "arima",
        "asset": "BTC-USD",
        "horizon": 1,
        "cutoff_date": "2026-06-01",
        "target_date": "2026-06-02",
        "regime": "calm",
        "last_close": 100.0,
        "y_pred": 101.0,
        "y_lower": 95.0,
        "y_upper": 107.0,
        "verdict_integrite": 1,
        "verdict_plausibilite": 1,
        "created_at": "2026-06-01T12:00:00",
    }
    record.update(overrides)
    return record


# ── 1. init_db ────────────────────────────────────────────────────────────────

def test_init_db_creates_tables_and_is_idempotent(tmp_path):
    db_path = str(tmp_path / "tracking.db")

    td.init_db(db_path)
    td.init_db(db_path)  # ré-appel : ne doit pas lever

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"test_cases", "predictions"} <= tables


# ── register_test_case (upsert) ──────────────────────────────────────────────

def test_register_test_case_upserts_on_conflict(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    td.register_test_case("TC1", "BTC-USD", 1, "desc v1", db_path=db_path)
    td.register_test_case("TC1", "BTC-USD", 7, "desc v2", db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM test_cases WHERE tc_id='TC1'").fetchone()
    n = conn.execute("SELECT COUNT(*) FROM test_cases").fetchone()[0]
    conn.close()

    assert n == 1
    assert row["horizon"] == 7
    assert row["description"] == "desc v2"


# ── 2. save_prediction : insertion + doublon ─────────────────────────────────

def test_save_prediction_inserts_and_duplicate_returns_false(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    record = make_record()

    assert td.save_prediction(record, db_path=db_path) is True
    assert td.save_prediction(record, db_path=db_path) is False  # doublon (tc_id/model/cutoff_date)

    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert n == 1


# ── 3. save_prediction : record incomplet ────────────────────────────────────

def test_save_prediction_missing_field_raises_value_error(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    record = make_record()
    del record["y_pred"]

    with pytest.raises(ValueError):
        td.save_prediction(record, db_path=db_path)


# ── 4. evaluate_pending : calcul exact des métriques ─────────────────────────

def test_evaluate_pending_computes_metrics_on_a_hit_case(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    record = make_record(
        tc_id="TC1", target_date="2026-06-02",
        last_close=100.0, y_pred=101.0, y_lower=95.0, y_upper=107.0,
    )
    td.save_prediction(record, db_path=db_path)

    n = td.evaluate_pending(lambda asset, target_date: 102.0, db_path=db_path, today="2026-06-03")
    assert n == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions").fetchone()
    conn.close()

    assert row["y_true"] == 102.0
    assert row["in_interval"] == 1                       # 95 <= 102 <= 107
    assert row["abs_error"] == pytest.approx(1.0)        # |102-101|
    assert row["abs_error_naif"] == pytest.approx(2.0)   # |102-100|
    assert row["beats_naif"] == 1                        # 1.0 <= 2.0
    assert row["direction_correct"] == 1                 # sign(101-100)=+1 == sign(102-100)=+1
    assert row["evaluated_at"] is not None


def test_evaluate_pending_computes_metrics_on_a_miss_case(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    record = make_record(
        tc_id="TC2", target_date="2026-06-02",
        last_close=100.0, y_pred=90.0, y_lower=85.0, y_upper=95.0,
    )
    td.save_prediction(record, db_path=db_path)

    td.evaluate_pending(lambda asset, target_date: 110.0, db_path=db_path, today="2026-06-03")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE tc_id='TC2'").fetchone()
    conn.close()

    assert row["in_interval"] == 0                        # 110 not in [85, 95]
    assert row["abs_error"] == pytest.approx(20.0)        # |110-90|
    assert row["abs_error_naif"] == pytest.approx(10.0)   # |110-100|
    assert row["beats_naif"] == 0                         # 20 > 10
    assert row["direction_correct"] == 0                  # sign(90-100)=-1 != sign(110-100)=+1


# ── 5. evaluate_pending : ignore non-échues et fetcher -> None ───────────────

def test_evaluate_pending_skips_not_due_and_unavailable_prices(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    future_record = make_record(tc_id="TC3", model="sarima", target_date="2099-01-01")
    td.save_prediction(future_record, db_path=db_path)

    due_but_unavailable = make_record(tc_id="TC4", model="prophet", target_date="2026-06-02")
    td.save_prediction(due_but_unavailable, db_path=db_path)

    calls = []

    def fetcher(asset, target_date):
        calls.append((asset, target_date))
        return None  # donnée pas encore disponible

    n = td.evaluate_pending(fetcher, db_path=db_path, today="2026-06-03")

    assert n == 0
    # le fetcher ne doit être appelé QUE pour la prédiction échue (TC4), pas la future (TC3)
    assert calls == [("BTC-USD", "2026-06-02")]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT tc_id, y_true FROM predictions").fetchall()
    conn.close()
    assert all(row["y_true"] is None for row in rows)


# ── 6. report : agrégations, y compris group_by=("model","regime") ──────────

def test_report_aggregates_by_model_and_by_model_regime(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    td.save_prediction(make_record(
        tc_id="TC1", model="arima", asset="BTC-USD", regime="calm",
        last_close=100.0, y_pred=101.0, y_lower=95.0, y_upper=107.0,
        verdict_integrite=1, verdict_plausibilite=1, target_date="2026-06-02",
    ), db_path=db_path)
    td.save_prediction(make_record(
        tc_id="TC2", model="arima", asset="ETH-USD", regime="stress",
        last_close=100.0, y_pred=90.0, y_lower=85.0, y_upper=95.0,
        verdict_integrite=1, verdict_plausibilite=0, target_date="2026-06-02",
    ), db_path=db_path)
    td.save_prediction(make_record(
        tc_id="TC3", model="lstm", asset="SPY", regime="calm",
        last_close=100.0, y_pred=104.0, y_lower=95.0, y_upper=110.0,
        verdict_integrite=0, verdict_plausibilite=1, target_date="2026-06-02",
    ), db_path=db_path)
    td.save_prediction(make_record(
        tc_id="TC4", model="lstm", asset="QQQ", regime="stress",
        verdict_integrite=1, verdict_plausibilite=1, target_date="2099-01-01",  # jamais échue ici
    ), db_path=db_path)

    prices = {
        ("BTC-USD", "2026-06-02"): 102.0,   # TC1 : in_interval, beats_naif, direction OK
        ("ETH-USD", "2026-06-02"): 110.0,   # TC2 : hors intervalle, perd au naif, direction fausse
        ("SPY", "2026-06-02"): 99.0,        # TC3 : in_interval mais perd au naif et direction fausse
    }

    n = td.evaluate_pending(lambda asset, target_date: prices.get((asset, target_date)),
                            db_path=db_path, today="2026-06-03")
    assert n == 3   # TC4 reste non-évaluée (target_date dans le futur)

    by_model = {row["model"]: row for row in td.report(group_by=("model",), db_path=db_path)}

    assert by_model["arima"]["n_total"] == 2
    assert by_model["arima"]["n_evalues"] == 2
    assert by_model["arima"]["taux_integrite"] == 1.0
    assert by_model["arima"]["taux_plausibilite"] == 0.5
    assert by_model["arima"]["couverture_ic95"] == 0.5
    assert by_model["arima"]["taux_bat_naif"] == 0.5
    assert by_model["arima"]["exactitude_dir"] == 0.5

    assert by_model["lstm"]["n_total"] == 2
    assert by_model["lstm"]["n_evalues"] == 1
    assert by_model["lstm"]["taux_integrite"] == 0.5
    assert by_model["lstm"]["taux_plausibilite"] == 1.0
    assert by_model["lstm"]["couverture_ic95"] == 1.0
    assert by_model["lstm"]["taux_bat_naif"] == 0.0
    assert by_model["lstm"]["exactitude_dir"] == 0.0

    by_model_regime = {(row["model"], row["regime"]): row
                       for row in td.report(group_by=("model", "regime"), db_path=db_path)}
    assert set(by_model_regime) == {
        ("arima", "calm"), ("arima", "stress"), ("lstm", "calm"), ("lstm", "stress"),
    }
    # groupe sans aucune prédiction évaluée : AVG() sur des NULL -> None, pas 0 ni une erreur
    assert by_model_regime[("lstm", "stress")]["n_evalues"] == 0
    assert by_model_regime[("lstm", "stress")]["couverture_ic95"] is None


def test_report_rejects_invalid_group_by(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)

    with pytest.raises(ValueError):
        td.report(group_by=("not_a_column",), db_path=db_path)


# ── export_csv ────────────────────────────────────────────────────────────────

def test_export_csv_writes_all_rows(tmp_path):
    db_path = str(tmp_path / "tracking.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)
    td.save_prediction(make_record(tc_id="TC2", cutoff_date="2026-06-08"), db_path=db_path)

    csv_path = tmp_path / "export.csv"
    n = td.export_csv(str(csv_path), db_path=db_path)

    assert n == 2
    content = csv_path.read_text()
    assert content.count("\n") == 3  # header + 2 lignes (+ newline final)
    assert "tc_id" in content.splitlines()[0]
