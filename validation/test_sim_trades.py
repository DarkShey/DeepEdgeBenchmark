"""
test_sim_trades.py — suite du §10 de BRIEF_bull_calm_d1.md (hors-ligne, aucun accès réseau).

Convention identique à test_tracking_db.py : sqlite temporaire (tmp_path), assertions
directes en SQL quand on teste la persistance, sinon appel direct aux fonctions pures.
"""

import math
import sqlite3

import pandas as pd
import pytest

from validation import sim_trades as st
from validation import tracking_db as td


# ── Fixtures / builders ───────────────────────────────────────────────────────

def make_prediction_record(**overrides):
    record = {
        "run_id": "run1", "tc_id": "TC_SPY_D1", "model": "ARIMA-GARCH", "asset": "SPY",
        "horizon": 1, "cutoff_date": "2026-06-01", "target_date": "2026-06-02",
        "regime": "calm", "last_close": 100.0, "y_pred": 101.0,
        "y_lower": 95.0, "y_upper": 107.0,
        "verdict_integrite": 1, "verdict_plausibilite": 1,
        "created_at": "2026-06-01T00:00:00",
    }
    record.update(overrides)
    return record


def write_fake_run_dir(tmp_path, name="20260707-ARIMA-SPY-D1", model="ARIMA-GARCH",
                       asset="SPY", horizon_label="D1", rows=None):
    """Fabrique un dossier Run/<name>/ minimal (predictions.parquet + metrics.json)
    conforme au contrat lu par sim_trades.build_daily_oos_log_rows."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "predictions.parquet")
    (run_dir / "metrics.json").write_text(
        '{"model": "%s", "asset": "%s", "horizon": "%s", "n_val": %d}'
        % (model, asset, horizon_label, len(df))
    )
    return run_dir


# ── 1. test_branches_exhaustives ──────────────────────────────────────────────

@pytest.mark.parametrize("realized,expected_branch", [
    (120.0, 1),   # realized > pi_high (110)
    (105.0, 2),   # ref < realized <= pi_high
    (100.0, 3),   # pi_low <= realized <= ref
    (85.0, 4),    # realized < pi_low
])
def test_branches_exhaustives(realized, expected_branch):
    ref, pi_low, pi_high = 100.0, 90.0, 110.0
    branch, counter, exit_px = st._resolve_branches(ref, pi_low, pi_high, realized)
    assert branch == expected_branch
    # une seule branche vraie : re-vérifie que les 4 conditions sont mutuellement exclusives
    conditions = [
        realized > pi_high,
        pi_high >= realized > ref,
        ref >= realized >= pi_low,
        realized < pi_low,
    ]
    assert sum(conditions) == 1


# ── 2. test_frontieres ────────────────────────────────────────────────────────

def test_frontieres_realized_equals_pi_high_is_branch_2_not_1():
    branch, counter, exit_px = st._resolve_branches(100.0, 90.0, 110.0, 110.0)
    assert branch == 2
    assert counter == 1
    assert exit_px == 110.0


def test_frontieres_realized_equals_ref_is_branch_3():
    branch, counter, exit_px = st._resolve_branches(100.0, 90.0, 110.0, 100.0)
    assert branch == 3
    assert counter == -1


def test_frontieres_realized_equals_pi_low_is_branch_3_not_4():
    branch, counter, exit_px = st._resolve_branches(100.0, 90.0, 110.0, 90.0)
    assert branch == 3
    assert counter == -1


# ── 3. test_counter_values ────────────────────────────────────────────────────

@pytest.mark.parametrize("realized,expected_counter", [
    (120.0, 2), (105.0, 1), (100.0, -1), (85.0, -2),
])
def test_counter_values(realized, expected_counter):
    _, counter, _ = st._resolve_branches(100.0, 90.0, 110.0, realized)
    assert counter == expected_counter


# ── 4. test_roi_formulas ──────────────────────────────────────────────────────

def test_roi_formula_branch_1_capped_at_pi_high():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=130.0)
    assert branch == 1
    assert roi == pytest.approx((110.0 - 100.0) / 100.0)


def test_roi_formula_branch_2():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=104.0)
    assert branch == 2
    assert roi == pytest.approx((104.0 - 100.0) / 100.0)


def test_roi_formula_branch_3():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=95.0)
    assert branch == 3
    assert roi == pytest.approx((95.0 - 100.0) / 100.0)


def test_roi_formula_branch_4_uses_realized_close_not_pi_low_in_v1():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=80.0)
    assert branch == 4
    # décision v1 (§8/§11.1) : exit_px = realized (daily close prudent), PAS pi_low
    assert roi == pytest.approx((80.0 - 100.0) / 100.0)
    assert roi != pytest.approx((90.0 - 100.0) / 100.0)


def test_roi_formula_applies_fee_bps():
    _, _, _, roi_no_fee, _ = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=104.0, fee_bps=0.0)
    _, _, _, roi_with_fee, _ = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=104.0, fee_bps=10.0)
    assert roi_no_fee - roi_with_fee == pytest.approx(10.0 / 1e4)


# ── 5. test_no_lookahead ──────────────────────────────────────────────────────

def test_no_lookahead_reference_price_is_actual_t_minus_1(tmp_path):
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 105.0, 103.0],
        "predicted": [999.0, 106.0, 104.0],   # predicted[0] n'est jamais utilisé (t=0 ignoré)
        "pi_lower":  [999.0,  95.0,  93.0],
        "pi_upper":  [999.0, 115.0, 113.0],
    }
    run_dir = write_fake_run_dir(tmp_path, rows=rows)
    log_rows, n_dropped = st.build_daily_oos_log_rows(run_dir)

    assert n_dropped == 0
    assert len(log_rows) == 2   # t=1 et t=2 ; t=0 ignoré (§6.1, pas de t-1)

    row_t1 = log_rows[0]
    assert row_t1["d_date"] == "2026-02-01"
    assert row_t1["target_date"] == "2026-02-02"
    assert row_t1["reference_price"] == pytest.approx(100.0)   # actual[t-1=0], PAS actual[t=1]=105.0
    assert row_t1["reference_price"] != pytest.approx(105.0)
    assert row_t1["realized_price"] == pytest.approx(105.0)    # actual[t=1], révélé seulement à D+1

    row_t2 = log_rows[1]
    assert row_t2["reference_price"] == pytest.approx(105.0)   # actual[t-1=1], PAS actual[t=2]=103.0
    assert row_t2["realized_price"] == pytest.approx(103.0)


def test_no_lookahead_swap_to_actual_t_would_fail(tmp_path):
    """Verrou anti-régression : si l'implémentation utilisait actual[t] au lieu de
    actual[t-1] comme reference_price, ce test doit échouer."""
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual":    [100.0, 999.0],   # valeur volontairement absurde pour actual[t=1]
        "predicted": [999.0, 106.0],
        "pi_lower":  [999.0,  95.0],
        "pi_upper":  [999.0, 115.0],
    }
    run_dir = write_fake_run_dir(tmp_path, rows=rows)
    log_rows, _ = st.build_daily_oos_log_rows(run_dir)
    # si le code (buggé) faisait reference_price = actual[t], on lirait 999.0 ici
    assert log_rows[0]["reference_price"] == pytest.approx(100.0)


# ── 6. test_signal_flat ───────────────────────────────────────────────────────

def test_signal_flat_when_predicted_not_above_ref():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=100.0, pi_low=90.0, pi_high=110.0, realized=105.0)
    assert signal_valid is False
    assert branch is None
    assert counter == 0
    assert roi == 0.0


def test_signal_flat_rows_not_stored_as_sim_trades(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 99.0, 101.0],
        "predicted": [999.0, 98.0, 108.0],   # t=1 : predicted(99->98) <= ref -> flat
        "pi_lower":  [999.0, 90.0, 90.0],
        "pi_upper":  [999.0, 108.0, 120.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_daily_oos_log_rows(run_dir)
    st.insert_daily_oos_log(log_rows, db_path=db_path)
    n_trades = st.generate_sim_trades(db_path=db_path)

    assert n_trades == 1   # seule la ligne t=2 (predicted=108 > ref=99) génère un trade
    conn = sqlite3.connect(db_path)
    n_total_log = conn.execute("SELECT COUNT(*) FROM daily_oos_log").fetchone()[0]
    n_total_trades = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
    conn.close()
    assert n_total_log == 2     # les 2 lignes (flat incluse) vivent dans daily_oos_log
    assert n_total_trades == 1  # le flat n'est jamais un sim_trade


# ── 7. test_degenerate_pi ─────────────────────────────────────────────────────

def test_degenerate_pi_flag_when_pi_high_leq_ref():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=101.0, pi_low=90.0, pi_high=100.0, realized=102.0)
    assert degenerate == 1
    assert signal_valid is True   # la règle s'exécute quand même (§6.2)
    assert branch is not None


def test_degenerate_pi_excluded_from_kpis_by_default(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual":    [100.0, 102.0],
        "predicted": [999.0, 101.0],
        "pi_lower":  [999.0,  90.0],
        "pi_upper":  [999.0, 100.0],   # pi_high(100) <= ref(100) -> dégénéré
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_daily_oos_log_rows(run_dir)
    st.insert_daily_oos_log(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades").fetchone()
    conn.close()
    assert trade["degenerate_pi"] == 1   # le trade existe bien...

    report = st.kpi_report(db_path=db_path, source="oos", group_by=("asset",))
    assert report[0]["n_signaux"] == 0   # ...mais est exclu du KPI par défaut


# ── 8. test_live_open ─────────────────────────────────────────────────────────

def test_live_open_when_realized_is_none():
    signal_valid, branch, counter, roi, degenerate = st.bull_calm_d1(
        ref=100.0, predicted=105.0, pi_low=90.0, pi_high=110.0, realized=None)
    assert signal_valid is True
    assert branch is None
    assert counter is None
    assert roi is None


def test_live_open_trade_stored_with_open_status(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    td.save_prediction(make_prediction_record(target_date="2099-01-01"), db_path=db_path)

    st.ingest_live_daily_oos_log(db_path=db_path)
    n_new = st.generate_sim_trades(db_path=db_path, source="live")
    assert n_new == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE source='live'").fetchone()
    conn.close()
    assert trade["status"] == "open"
    assert trade["realized_price"] is None
    assert trade["branch"] is None
    assert trade["counter"] is None

    report = st.kpi_report(db_path=db_path, source="live", group_by=("asset",))
    assert report[0]["n_signaux"] == 0   # non résolu -> pas compté dans les KPIs (§6.5)


# ── 9. test_idempotent_resolution ─────────────────────────────────────────────

def test_idempotent_resolution_no_duplicate_and_no_reprocess(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    td.save_prediction(make_prediction_record(
        tc_id="TC1", target_date="2026-06-02", last_close=100.0, y_pred=101.0,
        y_lower=95.0, y_upper=107.0,
    ), db_path=db_path)

    result1 = st.sync_live_trades(db_path=db_path)
    assert result1["new_trades"] == 1
    assert result1["resolved"] == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades").fetchone()
    conn.close()
    assert trade["status"] == "open"

    # le marché résout la prédiction
    td.evaluate_pending(lambda asset, target_date: 102.0, db_path=db_path, today="2026-06-03")

    result2 = st.sync_live_trades(db_path=db_path)
    assert result2["new_trades"] == 0
    assert result2["resolved"] == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute("SELECT * FROM sim_trades").fetchall()
    conn.close()
    assert len(trades) == 1   # toujours un seul trade, pas de doublon
    assert trades[0]["status"] == "closed"
    assert trades[0]["branch"] == 2
    assert trades[0]["roi"] == pytest.approx((102.0 - 100.0) / 100.0)

    # rejouer encore : rien ne doit bouger (idempotence complète)
    result3 = st.sync_live_trades(db_path=db_path)
    assert result3["new_trades"] == 0
    assert result3["resolved"] == 0
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
    conn.close()
    assert n == 1


# ── 10. test_vs_naive (baseline reproductible always-long) ───────────────────

def test_vs_naive_pure_persistence_generates_zero_signals(tmp_path):
    """Un prédicteur naïf pur (predicted[t] == actual[t-1], persistance stricte)
    ne déclenche jamais bull_calm_d1 (predicted > ref est toujours faux) : le
    benchmark 'signal-filtré' serait vide -- constaté empiriquement sur les runs
    Naive-*-D1 récents (cf. discussion), d'où le remplacement par un benchmark
    always-long pour le KPI 8 (naive_always_long_report)."""
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 105.0, 103.0],
        "predicted": [999.0, 100.0, 105.0],   # persistance stricte : predicted[t] == actual[t-1]
        "pi_lower":  [999.0,  90.0,  95.0],
        "pi_upper":  [999.0, 110.0, 115.0],
    }
    run_dir = write_fake_run_dir(tmp_path, model="Naive", horizon_label="D1", rows=rows)
    log_rows, _ = st.build_daily_oos_log_rows(run_dir)
    for row in log_rows:
        signal_valid, *_ = st.bull_calm_d1(
            row["reference_price"], row["predicted"], row["pi_lower"],
            row["pi_upper"], row["realized_price"])
        assert signal_valid is False


def test_naive_always_long_benchmark_ignores_signal_filter(tmp_path):
    """Le benchmark always-long (KPI 8, décision tuteur) applique la résolution des
    branches à CHAQUE jour, sans filtrer sur predicted > ref -- reproductible et
    déterministe sur un jeu de données fixe."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 105.0, 103.0],
        "predicted": [999.0, 100.0, 105.0],   # persistance stricte -> 0 signal sous bull_calm_d1
        "pi_lower":  [999.0,  90.0,  95.0],
        "pi_upper":  [999.0, 110.0, 115.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", model="Naive", horizon_label="D1", rows=rows)
    log_rows, _ = st.build_daily_oos_log_rows(run_dir)
    st.insert_daily_oos_log(log_rows, db_path=db_path)

    # sous la règle normale : aucun sim_trade (confirme test précédent au niveau DB)
    n_trades = st.generate_sim_trades(db_path=db_path)
    assert n_trades == 0

    report = st.naive_always_long_report(db_path=db_path, source="oos", group_by=("asset",))
    assert len(report) == 1
    entry = report[0]
    assert entry["n_days"] == 2   # les 2 lignes du log, malgré 0 signal
    # ligne t=1 : ref=100, pi=[90,110], realized=105 -> branch2, roi=+0.05
    # ligne t=2 : ref=105, pi=[95,115], realized=103 -> branch3, roi=(103-105)/105
    expected_roi_sum = (105.0 - 100.0) / 100.0 + (103.0 - 105.0) / 105.0
    assert entry["roi_sum"] == pytest.approx(expected_roi_sum, abs=1e-6)   # report arrondi à 6 décimales


# ── Bonus : regime="unknown" forcé en OOS, jamais lu depuis business_validation ─

def test_oos_regime_is_always_unknown_even_if_business_validation_present(tmp_path):
    run_dir = write_fake_run_dir(tmp_path, rows={
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 105.0], "predicted": [999.0, 106.0],
        "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 115.0],
    })
    (run_dir / "business_validation.json").write_text('{"regime": "bull"}')

    log_rows, _ = st.build_daily_oos_log_rows(run_dir)
    assert all(row["regime"] == "unknown" for row in log_rows)


def test_kpi_report_rejects_regime_groupby_on_oos_source(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    with pytest.raises(ValueError):
        st.kpi_report(db_path=db_path, source="oos", group_by=("asset", "regime"))


def test_kpi_report_allows_regime_groupby_on_live_source(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    # ne doit pas lever, même si la table est vide
    report = st.kpi_report(db_path=db_path, source="live", group_by=("asset", "regime"))
    assert report == []
