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


# ═══════════════════════════════════════════════════════════════════════════════
# Unification predictions (BRIEF_db_unification.md) : migration + isolation source
# ═══════════════════════════════════════════════════════════════════════════════

def _create_legacy_predictions_schema(db_path):
    """Reproduit exactement le schéma pré-unification (pas de `source`, tc_id/
    verdict_*/created_at NOT NULL) pour tester la migration en conditions réelles."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL, tc_id TEXT NOT NULL, model TEXT NOT NULL, asset TEXT NOT NULL,
            horizon INTEGER NOT NULL, cutoff_date TEXT NOT NULL, target_date TEXT NOT NULL,
            regime TEXT NOT NULL, last_close REAL NOT NULL, y_pred REAL NOT NULL,
            y_lower REAL NOT NULL, y_upper REAL NOT NULL, verdict_integrite INTEGER NOT NULL,
            verdict_plausibilite INTEGER NOT NULL, created_at TEXT NOT NULL, y_true REAL,
            in_interval INTEGER, abs_error REAL, abs_error_naif REAL, beats_naif INTEGER,
            direction_correct INTEGER, evaluated_at TEXT,
            UNIQUE (tc_id, model, cutoff_date)
        )
    """)
    conn.commit()
    conn.close()


def _insert_oos_row(db_path, **overrides):
    """INSERT direct dans predictions avec source='oos' (simule ce que fera
    sim_trades.py -- tc_id/verdict_*/created_at à NULL, comme une vraie ligne OOS)."""
    row = {
        "run_id": "20260707-Prophet-SPY-D1", "model": "Prophet", "asset": "SPY",
        "horizon": 1, "cutoff_date": "2026-02-01", "target_date": "2026-02-02",
        "regime": "unknown", "last_close": 100.0, "y_pred": 101.0,
        "y_lower": 95.0, "y_upper": 107.0, "y_true": 102.0, "source": "oos",
    }
    row.update(overrides)
    conn = sqlite3.connect(db_path)
    cols = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({placeholders})", row)
    conn.commit()
    conn.close()


def _drop_oos_unique_index(db_path):
    """Retire idx_predictions_oos_unique -- depuis BRIEF_prevention_doublons.md, cet
    index (désormais sans run_id) rend un doublon métier OOS impossible à insérer, y
    compris via _insert_oos_row. Utilisé uniquement par les tests qui exercent
    flag_daily_duplicates/reconcile_oos_sim_trades (conservées à titre défensif/
    historique, cf. leurs docstrings) sur un scénario de doublon qui ne peut plus se
    produire par le chemin d'ingestion normal."""
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX idx_predictions_oos_unique")
    conn.commit()
    conn.close()


def test_migration_preserves_ids_and_sequence_from_legacy_schema(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _create_legacy_predictions_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO predictions (run_id, tc_id, model, asset, horizon, cutoff_date, "
        "target_date, regime, last_close, y_pred, y_lower, y_upper, verdict_integrite, "
        "verdict_plausibilite, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("run1", "TC1", "arima", "SPY", 1, "2026-01-01", "2026-01-02", "calm",
             100.0, 101.0, 95.0, 107.0, 1, 1, "2026-01-01T00:00:00"),
            ("run2", "TC2", "lstm", "BTC-USD", 1, "2026-01-01", "2026-01-02", "calm",
             200.0, 201.0, 195.0, 207.0, 1, 1, "2026-01-01T00:00:00"),
            ("run3", "TC3", "prophet", "ETH-USD", 7, "2026-01-01", "2026-01-08", "bull",
             300.0, 301.0, 295.0, 307.0, 0, 1, "2026-01-01T00:00:00"),
        ],
    )
    conn.commit()
    conn.execute("DELETE FROM predictions WHERE tc_id='TC2'")   # trou dans les id (comme la vraie base)
    conn.execute("UPDATE sqlite_sequence SET seq = 392 WHERE name='predictions'")
    conn.commit()
    ids_before = [row[0] for row in conn.execute("SELECT id FROM predictions ORDER BY id")]
    conn.close()

    td.init_db(db_path)   # déclenche la migration (pas de colonne source)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, tc_id, verdict_integrite, source FROM predictions ORDER BY id").fetchall()
    seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='predictions'").fetchone()[0]
    conn.close()

    assert [row["id"] for row in rows] == ids_before == [1, 3]   # trou préservé, pas de renumérotation
    assert all(row["source"] == "live" for row in rows)          # lignes pré-existantes -> live
    assert rows[0]["tc_id"] == "TC1" and rows[1]["tc_id"] == "TC3"
    assert seq == 392   # compteur historique préservé, PAS recalculé depuis MAX(id)=3

    # la prochaine insertion continue bien après le compteur restauré
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO predictions (run_id, tc_id, model, asset, horizon, cutoff_date, "
        "target_date, regime, last_close, y_pred, y_lower, y_upper, verdict_integrite, "
        "verdict_plausibilite, created_at) VALUES "
        "('run4','TC4','naive','SPY',1,'2026-01-02','2026-01-03','calm',100,100,90,110,1,1,'2026-01-02T00:00:00')"
    )
    conn.commit()
    new_id = conn.execute("SELECT id FROM predictions WHERE tc_id='TC4'").fetchone()[0]
    conn.close()
    assert new_id == 393


def test_migration_is_idempotent_and_noop_on_already_migrated_db(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)   # déjà au nouveau schéma dès la création
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)
    td.init_db(db_path)   # rejouer ne doit rien casser ni dupliquer

    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert n == 1


def test_save_prediction_writes_source_live(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)

    conn = sqlite3.connect(db_path)
    source = conn.execute("SELECT source FROM predictions WHERE tc_id='TC1'").fetchone()[0]
    conn.close()
    assert source == "live"


def test_oos_row_accepts_null_business_columns(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _insert_oos_row(db_path)   # ne doit pas lever malgré tc_id/verdict_* absents

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE source='oos'").fetchone()
    conn.close()
    assert row["tc_id"] is None
    assert row["verdict_integrite"] is None
    assert row["verdict_plausibilite"] is None
    assert row["created_at"] is None


def test_oos_row_idempotent_via_partial_unique_index(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _insert_oos_row(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("""
        INSERT OR IGNORE INTO predictions
            (run_id, model, asset, horizon, cutoff_date, target_date, regime,
             last_close, y_pred, y_lower, y_upper, y_true, source)
        VALUES ('20260707-Prophet-SPY-D1','Prophet','SPY',1,'2026-02-01','2026-02-02',
                'unknown',100.0,101.0,95.0,107.0,102.0,'oos')
    """)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]
    conn.close()
    assert cur.rowcount == 0   # rejeu ignoré
    assert n == 1              # pas de doublon


def test_evaluate_pending_ignores_oos_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(
        tc_id="TC1", target_date="2026-06-02", last_close=100.0, y_pred=101.0,
    ), db_path=db_path)
    _insert_oos_row(db_path, target_date="2026-06-02", y_true=None)   # OOS "non résolu"

    calls = []
    n = td.evaluate_pending(
        lambda asset, target_date: calls.append((asset, target_date)) or 102.0,
        db_path=db_path, today="2026-06-03",
    )

    assert n == 1                 # seule la ligne live est résolue
    assert calls == [("BTC-USD", "2026-06-02")]   # jamais appelé pour la ligne oos (asset SPY ici)

    conn = sqlite3.connect(db_path)
    oos_y_true = conn.execute("SELECT y_true FROM predictions WHERE source='oos'").fetchone()[0]
    conn.close()
    assert oos_y_true is None   # la ligne oos n'a pas été touchée


def test_pending_assets_ignores_oos_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _insert_oos_row(db_path, asset="ZN=F", target_date="2026-06-02", y_true=None)

    assert td.pending_assets(db_path=db_path) == []   # la ligne oos non résolue n'apparaît pas


def test_fetch_predictions_for_run_ignores_oos_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _insert_oos_row(db_path, run_id="shared_run_id")
    td.save_prediction(make_record(tc_id="TC1", run_id="shared_run_id"), db_path=db_path)

    rows = td.fetch_predictions_for_run("shared_run_id", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["source"] == "live"


def test_report_ignores_oos_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1", model="arima"), db_path=db_path)
    _insert_oos_row(db_path, model="arima")

    by_model = {row["model"]: row for row in td.report(group_by=("model",), db_path=db_path)}
    assert by_model["arima"]["n_total"] == 1   # la ligne oos n'est pas comptée


def test_export_csv_default_is_live_only(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)
    _insert_oos_row(db_path)

    csv_path = tmp_path / "export.csv"
    n = td.export_csv(str(csv_path), db_path=db_path)
    assert n == 1   # comportement inchangé de tracking_export.csv : live seulement


def test_export_csv_source_none_dumps_everything(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)
    _insert_oos_row(db_path)

    csv_path = tmp_path / "export_all.csv"
    n = td.export_csv(str(csv_path), db_path=db_path, source=None)
    assert n == 2


# ── flag_daily_duplicates (BRIEF_correction_doublons.md) ─────────────────────


def test_flag_daily_duplicates_keeps_latest_run_id(tmp_path):
    """Défensif (cf. docstring de reconcile_oos_sim_trades) : ce scénario de doublon
    n'est plus atteignable par le chemin d'ingestion normal depuis
    BRIEF_prevention_doublons.md (idx_predictions_oos_unique ne porte plus run_id) --
    construit ici en désactivant temporairement l'index pour vérifier que la logique
    de flag_daily_duplicates reste correcte si des doublons existaient malgré tout
    (ex. base migrée depuis l'ancien schéma, avant que le nouvel index soit posé)."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _drop_oos_unique_index(db_path)
    common = dict(model="Prophet", asset="SPY", horizon=1,
                  cutoff_date="2026-02-01", target_date="2026-02-02")
    _insert_oos_row(db_path, run_id="20260701-Prophet-SPY-D1", y_pred=100.0, **common)
    _insert_oos_row(db_path, run_id="20260710-Prophet-SPY-D1", y_pred=102.0, **common)
    _insert_oos_row(db_path, run_id="20260705-Prophet-SPY-D1", y_pred=101.0, **common)

    n_flagged = td.flag_daily_duplicates(db_path=db_path)
    assert n_flagged == 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT run_id, daily_duplicate FROM predictions WHERE source='oos'"
    ).fetchall()
    conn.close()

    survivors = {r["run_id"] for r in rows if r["daily_duplicate"] == 0}
    flagged = {r["run_id"] for r in rows if r["daily_duplicate"] == 1}
    assert survivors == {"20260710-Prophet-SPY-D1"}   # run_id lexicographiquement le plus récent
    assert flagged == {"20260701-Prophet-SPY-D1", "20260705-Prophet-SPY-D1"}


def test_flag_daily_duplicates_ties_on_run_id_broken_by_id(tmp_path):
    """Cas défensif : l'index partiel idx_predictions_oos_unique interdit normalement
    deux lignes oos avec le même (run_id, model, asset, horizon, cutoff_date) -- une
    égalité de run_id ne peut donc pas se produire en usage réel (cf. audit : la cause
    des doublons est toujours des run_id distincts). On désactive temporairement
    l'index pour simuler ce cas et vérifier que le dernier niveau de départage
    (id DESC) fonctionne bien si cette hypothèse venait à être violée."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX idx_predictions_oos_unique")
    conn.commit()
    conn.close()

    _insert_oos_row(db_path, run_id="20260707-Prophet-SPY-D1", y_pred=100.0)
    _insert_oos_row(db_path, run_id="20260707-Prophet-SPY-D1", y_pred=101.0)   # id plus grand

    n_flagged = td.flag_daily_duplicates(db_path=db_path)
    assert n_flagged == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, daily_duplicate FROM predictions WHERE source='oos' ORDER BY id"
    ).fetchall()
    conn.close()

    assert rows[0]["daily_duplicate"] == 1   # id le plus petit -> flaggé
    assert rows[1]["daily_duplicate"] == 0   # id le plus grand -> survivant (départage ultime)


def test_flag_daily_duplicates_never_flags_live_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)
    # save_prediction() ré-appelle init_db() -- dropper l'index APRÈS, sinon il serait
    # simplement recréé (aucun doublon oos encore présent à ce stade) et la 2e
    # _insert_oos_row ci-dessous violerait la contrainte au lieu de la contourner.
    _drop_oos_unique_index(db_path)   # cf. docstring de test_flag_daily_duplicates_keeps_latest_run_id
    _insert_oos_row(db_path, run_id="20260701-Prophet-SPY-D1")
    _insert_oos_row(db_path, run_id="20260710-Prophet-SPY-D1")

    td.flag_daily_duplicates(db_path=db_path)

    conn = sqlite3.connect(db_path)
    live_total = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='live'").fetchone()[0]
    live_flagged = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE source='live' AND daily_duplicate=1"
    ).fetchone()[0]
    conn.close()

    assert live_total == 1
    assert live_flagged == 0


def test_flag_daily_duplicates_is_idempotent(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _drop_oos_unique_index(db_path)   # cf. docstring de test_flag_daily_duplicates_keeps_latest_run_id
    _insert_oos_row(db_path, run_id="20260701-Prophet-SPY-D1")
    _insert_oos_row(db_path, run_id="20260710-Prophet-SPY-D1")

    n_first = td.flag_daily_duplicates(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    state_first = {r["run_id"]: r["daily_duplicate"] for r in conn.execute(
        "SELECT run_id, daily_duplicate FROM predictions WHERE source='oos'"
    )}
    conn.close()

    n_second = td.flag_daily_duplicates(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    state_second = {r["run_id"]: r["daily_duplicate"] for r in conn.execute(
        "SELECT run_id, daily_duplicate FROM predictions WHERE source='oos'"
    )}
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()

    assert n_first == n_second == 1
    assert state_first == state_second
    assert total == 2   # aucune ligne supprimée par le rejeu


# ── flag_oos_superseded_by_live ───────────────────────────────────────────────


def test_flag_oos_superseded_by_live_flags_matching_cutoff_date(tmp_path):
    """Même si le target_date diverge (décalage business_lag), une ligne oos partage
    la même clé (model, asset, horizon, frequence, horizon_type, cutoff_date) qu'une
    ligne live -> doit être flaguée : c'est la même date de prédiction en double."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(
        model="Prophet", asset="SPY", cutoff_date="2026-07-06", target_date="2026-07-07",
    ), db_path=db_path)
    _insert_oos_row(db_path, model="Prophet", asset="SPY",
                     cutoff_date="2026-07-06", target_date="2026-07-08")   # target divergent

    n_flagged = td.flag_oos_superseded_by_live(db_path=db_path)
    assert n_flagged == 1

    conn = sqlite3.connect(db_path)
    oos_flag = conn.execute(
        "SELECT daily_duplicate FROM predictions WHERE source='oos'"
    ).fetchone()[0]
    conn.close()
    assert oos_flag == 1


def test_flag_oos_superseded_by_live_leaves_unmatched_oos_alone(tmp_path):
    """Pas de ligne live pour ce cutoff_date -> le backtest oos reste la seule source
    d'information pour cette date, il ne doit pas être flagué."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(
        model="Prophet", asset="SPY", cutoff_date="2026-07-06",
    ), db_path=db_path)
    _insert_oos_row(db_path, model="Prophet", asset="SPY", cutoff_date="2025-12-05")

    n_flagged = td.flag_oos_superseded_by_live(db_path=db_path)
    assert n_flagged == 0

    conn = sqlite3.connect(db_path)
    oos_flag = conn.execute(
        "SELECT daily_duplicate FROM predictions WHERE source='oos'"
    ).fetchone()[0]
    conn.close()
    assert oos_flag == 0


def test_flag_oos_superseded_by_live_never_flags_live_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(model="Prophet", asset="SPY", cutoff_date="2026-07-06"),
                        db_path=db_path)
    _insert_oos_row(db_path, model="Prophet", asset="SPY", cutoff_date="2026-07-06")

    td.flag_oos_superseded_by_live(db_path=db_path)

    conn = sqlite3.connect(db_path)
    live_flagged = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE source='live' AND daily_duplicate=1"
    ).fetchone()[0]
    conn.close()
    assert live_flagged == 0


def test_flag_oos_superseded_by_live_is_idempotent(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(model="Prophet", asset="SPY", cutoff_date="2026-07-06"),
                        db_path=db_path)
    _insert_oos_row(db_path, model="Prophet", asset="SPY", cutoff_date="2026-07-06")

    n_first = td.flag_oos_superseded_by_live(db_path=db_path)
    n_second = td.flag_oos_superseded_by_live(db_path=db_path)

    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    oos_flag = conn.execute(
        "SELECT daily_duplicate FROM predictions WHERE source='oos'"
    ).fetchone()[0]
    conn.close()

    assert n_first == 1
    assert n_second == 0   # déjà flaguée, rien de nouveau au 2e appel
    assert oos_flag == 1
    assert total == 2   # aucune ligne supprimée


def test_export_csv_oos_daily_duplicate_filter_is_a_noop_once_no_duplicates_remain(tmp_path):
    """Depuis BRIEF_prevention_doublons.md, export_csv (comme toute fonction qui
    s'auto-initialise) ne peut plus être appelée sur une base contenant encore des
    doublons OOS physiques (même flagués, pas supprimés) : son self-init tenterait de
    reposer l'index dur idx_predictions_oos_unique et lèverait une IntegrityError --
    ce scénario n'est de toute façon plus jamais atteignable par le chemin
    d'ingestion normal. Le filtre `AND daily_duplicate=0` de export_csv (posé par
    BRIEF_correction_doublons.md) devient donc un no-op permanent et inoffensif :
    vérifie qu'il n'exclut plus rien sur une base normale (aucun doublon)."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(tc_id="TC1"), db_path=db_path)
    _insert_oos_row(db_path, run_id="20260710-Prophet-SPY-D1")

    n_oos = td.export_csv(str(tmp_path / "export_oos.csv"), db_path=db_path, source="oos")
    assert n_oos == 1

    n_all = td.export_csv(str(tmp_path / "export_all.csv"), db_path=db_path, source=None)
    assert n_all == 2   # 1 live + 1 oos, rien filtré par daily_duplicate=0


# ── BRIEF_prevention_doublons.md : index dur sans run_id ─────────────────────

def test_oos_unique_index_rejects_raw_duplicate_business_key(tmp_path):
    """§4/§9 du brief : l'index OOS ne porte plus run_id -- un INSERT brut (sans
    gestion de conflit) pour la même clé métier (source, model, asset, horizon,
    cutoff_date) avec un run_id différent doit être rejeté par SQLite lui-même."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    _insert_oos_row(db_path, run_id="20260701-Prophet-SPY-D1")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_oos_row(db_path, run_id="20260710-Prophet-SPY-D1")   # même clé métier, run_id différent

    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]
    conn.close()
    assert n == 1   # le premier insert reste seul en base, le second a été rejeté (pas d'empilement)


def test_oos_unique_index_supports_upsert_keep_latest(tmp_path):
    """L'index (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
    doit servir de cible de conflit valide pour un upsert SQL direct -- prérequis pour
    sim_trades.insert_oos_predictions (§5 du brief, testé au niveau applicatif dans
    test_sim_trades.py). Ici, au niveau SQL brut : deux INSERT ... ON CONFLICT ... DO
    UPDATE sur la même clé métier avec des run_id différents -> 1 seule ligne, dont le
    contenu est celui du dernier insert (keep-latest).

    `frequence`/`horizon_type` (BRIEF_audit_combinaisons.md) omis de l'INSERT ->
    defaultent à 'daily'/'daily' (identiques aux deux appels), donc la collision sur
    la clé métier étendue se produit toujours comme avant leur ajout."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)

    def upsert(run_id, y_pred):
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO predictions (run_id, model, asset, horizon, regime, cutoff_date,
                                     target_date, last_close, y_pred, y_lower, y_upper, y_true, source)
            VALUES (?, 'Prophet', 'SPY', 1, 'unknown', '2026-02-01', '2026-02-02',
                    100.0, ?, 95.0, 107.0, 102.0, 'oos')
            ON CONFLICT (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
            WHERE source='oos'
            DO UPDATE SET run_id=excluded.run_id, y_pred=excluded.y_pred
        """, (run_id, y_pred))
        conn.commit()
        conn.close()

    upsert("20260701-Prophet-SPY-D1", 100.0)
    upsert("20260710-Prophet-SPY-D1", 105.0)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT run_id, y_pred FROM predictions WHERE source='oos'").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["run_id"] == "20260710-Prophet-SPY-D1"
    assert rows[0]["y_pred"] == pytest.approx(105.0)


# ── frequence / horizon_type / horizon_unit (BRIEF_audit_combinaisons.md) ──────

def test_fresh_db_has_frequency_horizon_columns_defaulting_daily(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    conn.close()
    assert {"frequence", "horizon_type", "horizon_unit"} <= cols


def test_save_prediction_defaults_to_daily_native(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.save_prediction(make_record(horizon=7), db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions").fetchone()
    conn.close()
    assert row["frequence"] == "daily"
    assert row["horizon_type"] == "daily"
    assert row["horizon_unit"] == "D+7"


def test_save_prediction_respects_explicit_weekly_fields(tmp_path):
    db_path = str(tmp_path / "t.db")
    record = make_record(horizon=2, tc_id="TC_W2", frequence="weekly",
                         horizon_type="weekly", horizon_unit="W+2")
    td.save_prediction(record, db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions").fetchone()
    conn.close()
    assert row["frequence"] == "weekly"
    assert row["horizon_type"] == "weekly"
    assert row["horizon_unit"] == "W+2"


def test_legacy_db_migration_backfills_frequency_horizon(tmp_path):
    """Une base créée AVANT ce brief (colonnes absentes) doit se retrouver, après
    un simple appel à init_db(), avec toutes ses lignes existantes correctement
    étiquetées daily natif -- ce sont réellement des prédictions daily, pas un
    remplissage arbitraire."""
    db_path = str(tmp_path / "t.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, tc_id TEXT,
            model TEXT NOT NULL, asset TEXT NOT NULL, horizon INTEGER NOT NULL,
            cutoff_date TEXT NOT NULL, target_date TEXT NOT NULL, regime TEXT NOT NULL,
            last_close REAL NOT NULL, y_pred REAL NOT NULL, y_lower REAL NOT NULL,
            y_upper REAL NOT NULL, verdict_integrite INTEGER, verdict_plausibilite INTEGER,
            created_at TEXT, y_true REAL, in_interval INTEGER, abs_error REAL,
            abs_error_naif REAL, beats_naif INTEGER, direction_correct INTEGER,
            evaluated_at TEXT, source TEXT NOT NULL DEFAULT 'live',
            daily_duplicate INTEGER NOT NULL DEFAULT 0,
            UNIQUE (tc_id, model, cutoff_date)
        )
    """)
    conn.execute("""
        INSERT INTO predictions (run_id, model, asset, horizon, cutoff_date, target_date,
                                 regime, last_close, y_pred, y_lower, y_upper)
        VALUES ('r1', 'ARIMA-GARCH', 'SPY', 7, '2026-01-01', '2026-01-08',
                'calm', 100.0, 101.0, 95.0, 107.0)
    """)
    conn.commit()
    conn.close()

    td.init_db(db_path)   # doit migrer sans lever

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions").fetchone()
    conn.close()
    assert row["frequence"] == "daily"
    assert row["horizon_type"] == "daily"
    assert row["horizon_unit"] == "D+7"


def test_frequency_horizon_migration_is_idempotent(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    td.save_prediction(make_record(), db_path=db_path)
    td.init_db(db_path)   # ré-appel, ne doit pas lever ni modifier les données
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions").fetchone()
    conn.close()
    assert row["frequence"] == "daily"


def test_oos_index_lets_daily_and_weekly_coexist_same_asset_horizon_cutoff(tmp_path):
    """Le coeur du problème BRIEF_audit_combinaisons.md §0 : un TSDiff-D (frequence=daily,
    horizon_type=weekly, horizon=1 = 'W+1') et un TSDiff-W (frequence=weekly,
    horizon_type=weekly, horizon=1) sur le MEME actif/horizon/cutoff_date sont deux
    prédictions différentes -- l'ancien index (sans frequence/horizon_type) les aurait
    fait collisionner (silently ignorées ou écrasées). Elles doivent coexister."""
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)
    conn = sqlite3.connect(db_path)

    def insert(frequence):
        conn.execute("""
            INSERT INTO predictions (run_id, model, asset, horizon, regime, cutoff_date,
                                     target_date, last_close, y_pred, y_lower, y_upper,
                                     y_true, source, frequence, horizon_type, horizon_unit)
            VALUES ('r', 'TSDiff', 'BTC-USD', 1, 'unknown', '2025-12-05', '2025-12-12',
                    100.0, 101.0, 95.0, 107.0, 102.0, 'oos', ?, 'weekly', 'W+1')
        """, (frequence,))

    insert("daily")     # TSDiff-D visant W+1
    insert("weekly")    # TSDiff-W natif W+1
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]
    conn.close()
    assert n == 2   # les deux coexistent, aucune n'a écrasé/ignoré l'autre
