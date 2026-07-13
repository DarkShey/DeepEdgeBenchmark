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
    conforme au contrat lu par sim_trades.build_oos_prediction_rows."""
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
    log_rows, n_dropped = st.build_oos_prediction_rows(run_dir)

    assert n_dropped == 0
    assert len(log_rows) == 2   # t=1 et t=2 ; t=0 ignoré (§6.1, pas de t-1)

    row_t1 = log_rows[0]
    assert row_t1["cutoff_date"] == "2026-02-01"
    assert row_t1["target_date"] == "2026-02-02"
    assert row_t1["last_close"] == pytest.approx(100.0)   # actual[t-1=0], PAS actual[t=1]=105.0
    assert row_t1["last_close"] != pytest.approx(105.0)
    assert row_t1["y_true"] == pytest.approx(105.0)    # actual[t=1], révélé seulement à D+1

    row_t2 = log_rows[1]
    assert row_t2["last_close"] == pytest.approx(105.0)   # actual[t-1=1], PAS actual[t=2]=103.0
    assert row_t2["y_true"] == pytest.approx(103.0)


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
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    # si le code (buggé) faisait reference_price = actual[t], on lirait 999.0 ici
    assert log_rows[0]["last_close"] == pytest.approx(100.0)


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
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    n_trades = st.generate_sim_trades(db_path=db_path)

    assert n_trades == 1   # seule la ligne t=2 (predicted=108 > ref=99) génère un trade
    conn = sqlite3.connect(db_path)
    n_total_log = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]
    n_total_trades = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
    conn.close()
    assert n_total_log == 2     # les 2 lignes (flat incluse) vivent dans predictions
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
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
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

    # plus d'étape d'ingestion séparée depuis BRIEF_db_unification.md : save_prediction
    # écrit directement dans predictions, all_predictions la voit immédiatement.
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
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    for row in log_rows:
        signal_valid, *_ = st.bull_calm_d1(
            row["last_close"], row["y_pred"], row["y_lower"],
            row["y_upper"], row["y_true"])
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
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Bear (bear_calm_d1 = TC1.3, bear_stress_d1 = TC1.4) — miroir exact des tests Bull
# ci-dessus (BRIEF_bull_calm_d1.md §3bis) : position courte, profit quand le prix baisse,
# ROI = (ref - exit_px)/ref, +2 si realized<PI_low, +1 si realized<ref, -1 si
# ref<=realized<=PI_high, -2 si realized>PI_high.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. test_bear_branches_exhaustives ────────────────────────────────────────

@pytest.mark.parametrize("realized,expected_branch", [
    (80.0, 1),    # realized < pi_low (90)
    (95.0, 2),    # pi_low <= realized < ref
    (100.0, 3),   # ref <= realized <= pi_high
    (120.0, 4),   # realized > pi_high (110)
])
def test_bear_branches_exhaustives(realized, expected_branch):
    ref, pi_low, pi_high = 100.0, 90.0, 110.0
    branch, counter, exit_px = st._resolve_branches_bear(ref, pi_low, pi_high, realized)
    assert branch == expected_branch
    conditions = [
        realized < pi_low,
        pi_low <= realized < ref,
        ref <= realized <= pi_high,
        realized > pi_high,
    ]
    assert sum(conditions) == 1


# ── 2. test_bear_frontieres ───────────────────────────────────────────────────

def test_bear_frontieres_realized_equals_pi_low_is_branch_2_not_1():
    branch, counter, exit_px = st._resolve_branches_bear(100.0, 90.0, 110.0, 90.0)
    assert branch == 2
    assert counter == 1
    assert exit_px == 90.0


def test_bear_frontieres_realized_equals_ref_is_branch_3():
    branch, counter, exit_px = st._resolve_branches_bear(100.0, 90.0, 110.0, 100.0)
    assert branch == 3
    assert counter == -1


def test_bear_frontieres_realized_equals_pi_high_is_branch_3_not_4():
    branch, counter, exit_px = st._resolve_branches_bear(100.0, 90.0, 110.0, 110.0)
    assert branch == 3
    assert counter == -1


# ── 3. test_bear_counter_values ───────────────────────────────────────────────

@pytest.mark.parametrize("realized,expected_counter", [
    (80.0, 2), (95.0, 1), (100.0, -1), (120.0, -2),
])
def test_bear_counter_values(realized, expected_counter):
    _, counter, _ = st._resolve_branches_bear(100.0, 90.0, 110.0, realized)
    assert counter == expected_counter


# ── 4. test_bear_roi_formulas ─────────────────────────────────────────────────

def test_bear_roi_formula_branch_1_capped_at_pi_low():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=70.0)
    assert branch == 1
    assert roi == pytest.approx((100.0 - 90.0) / 100.0)


def test_bear_roi_formula_branch_2():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=96.0)
    assert branch == 2
    assert roi == pytest.approx((100.0 - 96.0) / 100.0)


def test_bear_roi_formula_branch_3():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=105.0)
    assert branch == 3
    assert roi == pytest.approx((100.0 - 105.0) / 100.0)


def test_bear_roi_formula_branch_4_uses_realized_close_not_pi_high_in_v1():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=120.0)
    assert branch == 4
    # décision v1 (miroir §8/§11.1) : exit_px = realized (daily close prudent), PAS pi_high
    assert roi == pytest.approx((100.0 - 120.0) / 100.0)
    assert roi != pytest.approx((100.0 - 110.0) / 100.0)


def test_bear_roi_formula_applies_fee_bps():
    _, _, _, roi_no_fee, _ = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=96.0, fee_bps=0.0)
    _, _, _, roi_with_fee, _ = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=96.0, fee_bps=10.0)
    assert roi_no_fee - roi_with_fee == pytest.approx(10.0 / 1e4)


# ── 5. test_bear_signal_flat ──────────────────────────────────────────────────

def test_bear_signal_flat_when_predicted_not_below_ref():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=100.0, pi_low=90.0, pi_high=110.0, realized=95.0)
    assert signal_valid is False
    assert branch is None
    assert counter == 0
    assert roi == 0.0


def test_bear_signal_requires_ref_leq_pi_high():
    # ref > pi_high -> relève de TC1.4 (bear_stress_d1), pas de TC1.3 (garde-fou d'étanchéité)
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=112.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=95.0)
    assert signal_valid is False


# ── 6. test_bear_degenerate_pi ────────────────────────────────────────────────

def test_bear_degenerate_pi_flag_when_pi_low_geq_ref():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=99.0, pi_low=100.0, pi_high=110.0, realized=98.0)
    assert degenerate == 1
    assert signal_valid is True   # la règle s'exécute quand même (miroir §6.2)
    assert branch is not None


def test_bear_degenerate_pi_excluded_from_kpis_by_default(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual":    [100.0, 98.0],
        "predicted": [999.0, 99.0],
        "pi_lower":  [999.0, 100.0],   # pi_low(100) >= ref(100) -> dégénéré
        "pi_upper":  [999.0, 110.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="bear_calm_d1", source="oos")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='bear_calm_d1'").fetchone()
    conn.close()
    assert trade["degenerate_pi"] == 1

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="bear_calm_d1", group_by=("asset",))
    assert report[0]["n_signaux"] == 0


# ── 7. test_bear_live_open ────────────────────────────────────────────────────

def test_bear_live_open_when_realized_is_none():
    signal_valid, branch, counter, roi, degenerate = st.bear_calm_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=None)
    assert signal_valid is True
    assert branch is None
    assert counter is None
    assert roi is None


def test_bear_live_open_trade_stored_with_open_status(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    td.save_prediction(make_prediction_record(
        target_date="2099-01-01", last_close=100.0, y_pred=95.0, y_lower=90.0, y_upper=110.0,
    ), db_path=db_path)

    n_new = st.generate_sim_trades(db_path=db_path, rule_version="bear_calm_d1", source="live")
    assert n_new == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='bear_calm_d1'").fetchone()
    conn.close()
    assert trade["status"] == "open"
    assert trade["realized_price"] is None
    assert trade["branch"] is None
    assert trade["counter"] is None

    report = st.kpi_report(db_path=db_path, source="live", rule_version="bear_calm_d1", group_by=("asset",))
    assert report[0]["n_signaux"] == 0


# ── 8. test_bear_idempotent_resolution ────────────────────────────────────────

def test_bear_idempotent_resolution_no_duplicate_and_no_reprocess(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    td.save_prediction(make_prediction_record(
        target_date="2026-06-02", last_close=100.0, y_pred=95.0,
        y_lower=90.0, y_upper=110.0,
    ), db_path=db_path)

    result1 = st.sync_live_trades(db_path=db_path, rule_version="bear_calm_d1")
    assert result1["new_trades"] == 1
    assert result1["resolved"] == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='bear_calm_d1'").fetchone()
    conn.close()
    assert trade["status"] == "open"

    # le marché résout la prédiction : le prix baisse (thèse courte confirmée)
    td.evaluate_pending(lambda asset, target_date: 96.0, db_path=db_path, today="2026-06-03")

    result2 = st.sync_live_trades(db_path=db_path, rule_version="bear_calm_d1")
    assert result2["new_trades"] == 0
    assert result2["resolved"] == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute("SELECT * FROM sim_trades WHERE rule_version='bear_calm_d1'").fetchall()
    conn.close()
    assert len(trades) == 1
    assert trades[0]["status"] == "closed"
    assert trades[0]["branch"] == 2
    assert trades[0]["roi"] == pytest.approx((100.0 - 96.0) / 100.0)

    result3 = st.sync_live_trades(db_path=db_path, rule_version="bear_calm_d1")
    assert result3["new_trades"] == 0
    assert result3["resolved"] == 0


# ── 9. bear_stress_d1 (TC1.4) — signal de conviction forte, résolution partagée ──

def test_bear_stress_signal_requires_pi_high_below_ref():
    # pi_high < ref -> toute la bande est sous ref, baisse quasi certaine
    signal_valid, branch, counter, roi, degenerate = st.bear_stress_d1(
        ref=112.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=95.0)
    assert signal_valid is True

    # pi_high >= ref -> pas de conviction forte (relève de bear_calm_d1 ou flat)
    signal_valid, branch, counter, roi, degenerate = st.bear_stress_d1(
        ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=95.0)
    assert signal_valid is False


def test_bear_stress_shares_resolution_branches_bear_helper():
    # bear_calm_d1 et bear_stress_d1 sont mutuellement exclusives par construction (la
    # même journée ne peut valider qu'une seule des deux, cf. garde-fous d'étanchéité) --
    # ce test vérifie qu'elles délèguent bien à la MÊME fonction de résolution
    # (_resolve_branches_bear), pas qu'elles donnent le même résultat sur un jour donné.
    ref, pi_low, pi_high, realized = 112.0, 90.0, 110.0, 95.0
    expected = st._resolve_branches_bear(ref, pi_low, pi_high, realized)

    _, branch_stress, counter_stress, roi_stress, _ = st.bear_stress_d1(
        ref=ref, predicted=95.0, pi_low=pi_low, pi_high=pi_high, realized=realized)
    assert (branch_stress, counter_stress) == (expected[0], expected[1])
    assert roi_stress == pytest.approx((ref - expected[2]) / ref)


# ── 10. RULES dispatch registers the two new rule versions ──────────────────

def test_bear_rules_registered_in_dispatch():
    assert "bear_calm_d1" in st.RULES
    assert "bear_stress_d1" in st.RULES
    result = st.RULES["bear_calm_d1"](ref=100.0, predicted=95.0, pi_low=90.0, pi_high=110.0, realized=96.0)
    signal_valid, branch, counter, roi, direction_ok, in_band, degenerate = result
    assert signal_valid is True
    assert direction_ok == 1   # realized(96) < ref(100) : thèse courte confirmée
    assert in_band is None


# ── Bonus : regime="unknown" forcé en OOS, jamais lu depuis business_validation ─

def test_oos_regime_is_always_unknown_even_if_business_validation_present(tmp_path):
    run_dir = write_fake_run_dir(tmp_path, rows={
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 105.0], "predicted": [999.0, 106.0],
        "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 115.0],
    })
    (run_dir / "business_validation.json").write_text('{"regime": "bull"}')

    log_rows, _ = st.build_oos_prediction_rows(run_dir)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Sideways (sideways_d1) — BRIEF_sideways_d1.md §9. Test de justesse pur : roi et
# direction_ok sont TOUJOURS None. Exemple mental repris du §4 du brief :
# ref=100, pi_low=96, pi_high=104 (W=8), predicted=100.3, k=0.10 -> eps=0.8, m=2, h=4.
# ═══════════════════════════════════════════════════════════════════════════════

SW_REF, SW_PI_LOW, SW_PI_HIGH = 100.0, 96.0, 104.0   # W=8


# ── test_sideways_signal_flat_vs_directional ─────────────────────────────────

def test_sideways_signal_flat_vs_directional():
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5)
    assert signal is True   # |100.3-100|=0.3 <= eps=0.8

    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=105.0, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5)
    assert signal is False  # |105-100|=5 > eps=0.8 : mouvement directionnel, pas sideways
    assert branch is None
    assert counter == 0


# ── test_sideways_signal_requires_ref_in_band ────────────────────────────────

def test_sideways_signal_requires_ref_in_band():
    # |predicted-ref| minuscule mais ref hors bande -> jamais Sideways (§5.2)
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=90.0, predicted=90.05, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=90.1)
    assert signal is False


# ── test_sideways_branches_exhaustives ───────────────────────────────────────

@pytest.mark.parametrize("realized,expected_branch", [
    (100.5, 1),   # quasi immobile : |100.5-100|=0.5 <= m=2
    (103.0, 2),   # dans la bande, hors coeur
    (106.0, 3),   # hors bande, dist=106-104=2 <= h=4
    (111.0, 4),   # hors bande, dist=111-104=7 > h=4
])
def test_sideways_branches_exhaustives(realized, expected_branch):
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized)
    assert signal is True
    assert branch == expected_branch
    # une seule zone vraie (§3 exhaustivité)
    in_band_check = SW_PI_LOW <= realized <= SW_PI_HIGH
    core_check = in_band_check and abs(realized - SW_REF) <= 2.0
    small_breakout = (not in_band_check) and min(abs(SW_PI_LOW - realized), abs(realized - SW_PI_HIGH)) <= 4.0
    big_breakout = (not in_band_check) and not small_breakout
    assert sum([core_check, in_band_check and not core_check, small_breakout, big_breakout]) == 1


# ── test_sideways_frontieres ──────────────────────────────────────────────────

def test_sideways_frontieres_pi_high_and_pi_low_are_in_band_inclusive():
    # realized == pi_high (104) : dans la bande (inclusif), hors coeur (dist=4 > m=2) -> branche 2
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=104.0)
    assert in_band == 1
    assert branch == 2

    # realized == pi_low (96) : idem, symétrique
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=96.0)
    assert in_band == 1
    assert branch == 2


def test_sideways_frontieres_m_boundary_inclusive_in_branch_1():
    # realized == ref + m (100+2=102) : |102-100|=2 <= m=2 -> branche 1 inclusif
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=102.0)
    assert branch == 1


def test_sideways_frontieres_h_boundary_inclusive_in_branch_3():
    # realized == pi_high + h (104+4=108) : dist=4 <= h=4 -> branche 3 inclusif (pas 4)
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=108.0)
    assert branch == 3

    # juste au-delà -> branche 4
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=108.01)
    assert branch == 4


# ── test_sideways_counter_values ─────────────────────────────────────────────

@pytest.mark.parametrize("realized,expected_counter", [
    (100.5, 2), (103.0, 1), (106.0, -1), (111.0, -2),
])
def test_sideways_counter_values(realized, expected_counter):
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized)
    assert counter == expected_counter


# ── test_sideways_symmetry ────────────────────────────────────────────────────

def test_sideways_symmetry_small_breakout_same_counter_both_directions():
    # dist=4 des deux côtés (h=4) -> même branche/counter, haussier ou baissier
    _, branch_up, counter_up, *_ = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=108.0)
    _, branch_down, counter_down, *_ = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=92.0)
    assert (branch_up, counter_up) == (branch_down, counter_down) == (3, -1)


def test_sideways_symmetry_big_breakout_same_counter_both_directions():
    # dist=7 des deux côtés -> même branche/counter
    _, branch_up, counter_up, *_ = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=111.0)
    _, branch_down, counter_down, *_ = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=89.0)
    assert (branch_up, counter_up) == (branch_down, counter_down) == (4, -2)


# ── test_sideways_roi_is_none ─────────────────────────────────────────────────

@pytest.mark.parametrize("realized", [100.5, 103.0, 106.0, 111.0, None])
def test_sideways_roi_is_none(realized):
    result = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized)
    roi = result[3]
    assert roi is None


# ── test_sideways_direction_ok_is_none (niveau DB : direction_ok n'existe même pas
#    dans le retour de sideways_d1, seul le stockage sim_trades peut le vérifier) ──

def test_sideways_direction_ok_is_none_in_sim_trades(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.5], "predicted": [999.0, 100.3],
        "pi_lower": [999.0, 96.0], "pi_upper": [999.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    n = st.generate_sim_trades(db_path=db_path, rule_version="sideways_d1", source="oos")
    assert n == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_d1'").fetchone()
    conn.close()
    assert trade["direction_ok"] is None
    assert trade["roi"] is None
    assert trade["in_band"] == 1
    assert trade["branch"] == 1
    assert trade["counter"] == 2


# ── test_sideways_no_lookahead ────────────────────────────────────────────────

def test_sideways_no_lookahead_reference_price_is_actual_t_minus_1(tmp_path):
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 100.5, 999.0],   # actual[t=2] volontairement absurde
        "predicted": [999.0, 100.3, 100.2],
        "pi_lower":  [999.0,  96.0,  96.0],
        "pi_upper":  [999.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path, rows=rows)
    log_rows, n_dropped = st.build_oos_prediction_rows(run_dir)

    assert n_dropped == 0
    row_t1 = log_rows[0]
    assert row_t1["last_close"] == pytest.approx(100.0)   # actual[t-1=0], pas actual[t=1]
    assert row_t1["y_true"] == pytest.approx(100.5)

    # si le code (buggé) utilisait actual[t] comme référence, on lirait 100.5 pour t=2, pas 100.5 (t-1=1)
    row_t2 = log_rows[1]
    assert row_t2["last_close"] == pytest.approx(100.5)   # actual[t-1=1]
    assert row_t2["last_close"] != pytest.approx(999.0)


# ── test_sideways_live_open ───────────────────────────────────────────────────

def test_sideways_live_open_when_realized_is_none():
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=None)
    assert signal is True
    assert branch is None
    assert counter is None
    assert roi is None
    assert in_band is None


def test_sideways_live_open_trade_stored_with_open_status(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    td.save_prediction(make_prediction_record(
        target_date="2099-01-01", last_close=100.0, y_pred=100.3, y_lower=96.0, y_upper=104.0,
    ), db_path=db_path)

    n_new = st.generate_sim_trades(db_path=db_path, rule_version="sideways_d1", source="live")
    assert n_new == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_d1'").fetchone()
    conn.close()
    assert trade["status"] == "open"
    assert trade["realized_price"] is None
    assert trade["branch"] is None
    assert trade["counter"] is None
    assert trade["roi"] is None
    assert trade["in_band"] is None

    report = st.kpi_report(db_path=db_path, source="live", rule_version="sideways_d1", group_by=("asset",))
    assert report[0]["n_signaux"] == 0   # non résolu -> hors KPIs (§5.4)


# ── test_sideways_degenerate_pi ───────────────────────────────────────────────

def test_sideways_degenerate_pi_when_w_is_zero():
    signal, branch, counter, roi, in_band, degenerate = st.sideways_d1(
        ref=100.0, predicted=100.0, pi_low=100.0, pi_high=100.0, realized=100.0)
    assert degenerate == 1


# ── test_sideways_k_sensitivity ───────────────────────────────────────────────

def test_sideways_k_sensitivity_is_monotonic_in_number_of_signals(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    # écarts |predicted-ref| croissants relatifs à W=8 : 0.3 (6.25% de piste), 1.0, 1.5, 3.0
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]),
        "actual":    [100.0, 100.5, 101.0, 100.8, 100.2],
        "predicted": [999.0, 100.3, 102.0, 102.3, 103.8],
        "pi_lower":  [999.0,  96.0,  97.0,  96.8,  96.2],
        "pi_upper":  [999.0, 104.0, 105.0, 104.8, 104.2],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_d1",
                           group_by=("asset",), k_values=(0.05, 0.10, 0.15, 0.20))
    sensitivity = report[0]["k_sensitivity"]
    assert [s["k"] for s in sensitivity] == [0.05, 0.10, 0.15, 0.20]
    n_signals = [s["n_signaux"] for s in sensitivity]
    assert n_signals == sorted(n_signals)   # monotone non-décroissant
    assert n_signals[0] < n_signals[-1]     # variation réelle sur ce jeu de données


def test_sideways_k_values_rejected_for_other_rule_version(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    with pytest.raises(ValueError):
        st.kpi_report(db_path=db_path, source="oos", rule_version="bull_calm_d1",
                      group_by=("asset",), k_values=(0.10,))


# ── kpi_report variante justesse (pas de ROI) ────────────────────────────────

def test_sideways_kpi_report_has_no_roi_fields(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.5], "predicted": [999.0, 100.3],
        "pi_lower": [999.0, 96.0], "pi_upper": [999.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_d1", source="oos")

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_d1", group_by=("asset",))
    entry = report[0]
    assert "roi_sum" not in entry
    assert "roi_mean" not in entry
    assert entry["n_signaux"] == 1
    assert entry["taux_justesse"] == 1.0     # counter=+2 >= 1
    assert entry["taux_immobile"] == 1.0     # counter==+2
    assert entry["taux_breakout"] == 0.0
    assert entry["taux_breakout_haussier"] == 0.0
    assert entry["taux_breakout_baissier"] == 0.0
    assert entry["in_band_coverage"] == 1.0


def test_sideways_kpi_report_breakout_decomposed_haussier_baissier(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 111.0, 89.0],   # t1 : breakout haussier fort ; t2 : ref=111 (hors bande, pas sideways)
        "predicted": [999.0, 100.3, 100.3],
        "pi_lower":  [999.0,  96.0,  96.0],
        "pi_upper":  [999.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_d1", source="oos")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_d1'").fetchall()
    conn.close()
    # seul t=1 (ref=100, dans la bande) est un signal sideways ; t=2 a ref=111 hors bande -> pas de trade
    assert len(trades) == 1
    assert trades[0]["branch"] == 4   # realized=111, dist=7 > h=4
    assert trades[0]["counter"] == -2

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_d1", group_by=("asset",))
    entry = report[0]
    assert entry["taux_breakout"] == 1.0
    assert entry["taux_breakout_haussier"] == 1.0
    assert entry["taux_breakout_baissier"] == 0.0


# ── k_values ne doit rien écrire (lecture seule) ─────────────────────────────

def test_sideways_k_sweep_does_not_write_sim_trades(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 100.5, 101.0],
        "predicted": [999.0, 100.3, 102.0],
        "pi_lower":  [999.0,  96.0,  97.0],
        "pi_upper":  [999.0, 104.0, 105.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)

    conn = sqlite3.connect(db_path)
    n_before = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
    conn.close()
    assert n_before == 0

    st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_d1",
                  group_by=("asset",), k_values=(0.05, 0.10, 0.15, 0.20))

    conn = sqlite3.connect(db_path)
    n_after = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
    conn.close()
    assert n_after == 0   # le balayage k_values ne persiste rien


# ═══════════════════════════════════════════════════════════════════════════════
# Unification predictions (BRIEF_db_unification.md) : coexistence live/oos + horizon=1
# ═══════════════════════════════════════════════════════════════════════════════

def test_coexistence_live_and_oos_isolated_by_source_in_generate_sim_trades(tmp_path):
    """live + oos dans la même table predictions, distingués par source : chaque appel
    de generate_sim_trades(source=...) n'agit que sur sa source, jamais sur l'autre."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    td.save_prediction(make_prediction_record(
        tc_id="TC_LIVE", target_date="2026-06-02", last_close=100.0, y_pred=101.0,
        y_lower=95.0, y_upper=107.0,
    ), db_path=db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [200.0, 210.0], "predicted": [999.0, 205.0],
        "pi_lower": [999.0, 195.0], "pi_upper": [999.0, 215.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", asset="BTC-USD", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)

    n_live = st.generate_sim_trades(db_path=db_path, source="live")
    n_oos = st.generate_sim_trades(db_path=db_path, source="oos")
    assert n_live == 1 and n_oos == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    live_trade = conn.execute("SELECT * FROM sim_trades WHERE source='live'").fetchone()
    oos_trade = conn.execute("SELECT * FROM sim_trades WHERE source='oos'").fetchone()
    conn.close()
    assert live_trade["asset"] == "SPY"
    assert oos_trade["asset"] == "BTC-USD"
    # rejouer generate_sim_trades sans filtre de source ne doit rien dupliquer
    n_replay = st.generate_sim_trades(db_path=db_path)
    assert n_replay == 0


def test_live_horizon_7_never_leaks_into_all_predictions_or_kpis(tmp_path):
    """predictions contient du live horizon=1 ET horizon=7 (mêmes runs) ; la vue
    all_predictions (et donc generate_sim_trades/kpi_report) doit filtrer horizon=1
    strictement -- une ligne h=7 ne doit jamais apparaître dans le backtest D+1."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    td.save_prediction(make_prediction_record(
        tc_id="TC_H1", horizon=1, target_date="2026-06-02",
        last_close=100.0, y_pred=101.0, y_lower=95.0, y_upper=107.0,
    ), db_path=db_path)
    td.save_prediction(make_prediction_record(
        tc_id="TC_H7", horizon=7, target_date="2026-06-08",
        last_close=100.0, y_pred=110.0, y_lower=95.0, y_upper=120.0,
    ), db_path=db_path)

    conn = sqlite3.connect(db_path)
    n_all_predictions = conn.execute("SELECT COUNT(*) FROM all_predictions").fetchone()[0]
    horizons_seen = {row[0] for row in conn.execute("SELECT DISTINCT horizon FROM all_predictions")}
    conn.close()
    assert n_all_predictions == 1
    assert horizons_seen == {1}

    n_new = st.generate_sim_trades(db_path=db_path, source="live")
    assert n_new == 1   # jamais 2 : la ligne h=7 n'a pas généré de sim_trade

    report = st.kpi_report(db_path=db_path, source="live", group_by=())
    assert report[0]["n_total"] == 1   # la ligne h=7 n'est pas comptée dans N_total non plus


def test_init_db_drops_legacy_daily_oos_log_when_all_rows_duplicated(tmp_path):
    """Une base créée avant BRIEF_db_unification.md peut encore porter une
    daily_oos_log orpheline (plus jamais écrite) -- init_db() doit la supprimer, à
    condition que chaque ligne y soit déjà un doublon exact de predictions."""
    db_path = str(tmp_path / "t.db")
    td.save_prediction(make_prediction_record(tc_id="TC1", run_id="run1"), db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE daily_oos_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, model TEXT, asset TEXT,
            horizon INTEGER, d_date TEXT, source TEXT
        )
    """)
    conn.execute(
        "INSERT INTO daily_oos_log (run_id, model, asset, horizon, d_date, source) "
        "VALUES ('run1','ARIMA-GARCH','SPY',1,'2026-06-01','live')"
    )
    conn.commit()
    conn.close()

    st.init_db(db_path)   # doit supprimer daily_oos_log sans lever

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "daily_oos_log" not in tables


def test_init_db_refuses_to_drop_daily_oos_log_with_orphan_rows(tmp_path):
    db_path = str(tmp_path / "t.db")
    td.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE daily_oos_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, model TEXT, asset TEXT,
            horizon INTEGER, d_date TEXT, source TEXT
        )
    """)
    conn.execute(
        "INSERT INTO daily_oos_log (run_id, model, asset, horizon, d_date, source) "
        "VALUES ('run_orphan','ARIMA-GARCH','SPY',1,'2026-06-01','live')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError):
        st.init_db(db_path)   # aucune ligne predictions correspondante -> refuse de perdre la donnée
