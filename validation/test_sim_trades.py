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
                       asset="SPY", horizon_label="D1", rows=None, prices=None):
    """Fabrique un dossier Run/<name>/ minimal (predictions.parquet + metrics.json,
    + prices.parquet si fourni) conforme au contrat lu par
    sim_trades.build_oos_prediction_rows. `prices` (dict avec cles "date"/"close")
    est nécessaire pour un horizon rolling-origin (D7) -- reconstruction de
    cutoff_date/last_close, cf. docstring du module."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "predictions.parquet")
    (run_dir / "metrics.json").write_text(
        '{"model": "%s", "asset": "%s", "horizon": "%s", "n_val": %d}'
        % (model, asset, horizon_label, len(df))
    )
    if prices is not None:
        pd.DataFrame(prices).to_parquet(run_dir / "prices.parquet")
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


# ── BRIEF_correction_sim_trades.md : daily_duplicate propagé à all_predictions/sim_trades ──

_DUP_ROWS = {
    "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
    "actual": [100.0, 102.0], "predicted": [999.0, 103.0],
    "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 110.0],
}


def _drop_oos_unique_index(db_path):
    """Retire idx_predictions_oos_unique -- depuis BRIEF_prevention_doublons.md, cet
    index (désormais sans run_id) rend un doublon métier OOS impossible à insérer via
    insert_oos_predictions (upsert « garde le dernier »). Utilisé uniquement par les
    tests qui exercent flag_daily_duplicates/reconcile_oos_sim_trades (conservées à
    titre défensif/historique, cf. leurs docstrings) sur un scénario de doublon qui ne
    peut plus se produire par le chemin d'ingestion normal."""
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX idx_predictions_oos_unique")
    conn.commit()
    conn.close()


def _insert_oos_row_bypassing_index(db_path, **overrides):
    """INSERT direct dans predictions (bypass insert_oos_predictions, qui refuserait
    désormais un doublon business-key) -- l'appelant doit avoir appelé
    _drop_oos_unique_index avant d'insérer plus d'une ligne pour la même clé métier."""
    row = {
        "run_id": "20260707-ARIMA-SPY-D1", "model": "ARIMA-GARCH", "asset": "SPY",
        "horizon": 1, "regime": "unknown", "cutoff_date": "2026-02-01",
        "target_date": "2026-02-02", "last_close": 100.0, "y_pred": 101.0,
        "y_lower": 95.0, "y_upper": 107.0, "y_true": 102.0, "source": "oos",
    }
    row.update(overrides)
    conn = sqlite3.connect(db_path)
    cols = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({placeholders})", row)
    conn.commit()
    conn.close()


def test_all_predictions_view_excludes_flagged_oos_duplicates(tmp_path):
    """Défensif (cf. docstring de reconcile_oos_sim_trades) : ce scénario de doublon
    physique n'est plus atteignable via insert_oos_predictions depuis
    BRIEF_prevention_doublons.md (index sans run_id) -- construit ici en bypassant
    l'index pour vérifier que la vue continue de bien filtrer daily_duplicate=1 si des
    lignes flaguées existaient malgré tout (ex. base migrée depuis l'ancien schéma)."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    _drop_oos_unique_index(db_path)

    _insert_oos_row_bypassing_index(db_path, run_id="20260701-ARIMA-SPY-D1")
    _insert_oos_row_bypassing_index(db_path, run_id="20260710-ARIMA-SPY-D1")

    conn = sqlite3.connect(db_path)
    n_before = conn.execute("SELECT COUNT(*) FROM all_predictions WHERE source='oos'").fetchone()[0]
    conn.close()
    assert n_before == 2   # avant flag, la vue voit encore les 2 copies

    td.flag_daily_duplicates(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows_after = conn.execute("SELECT run_id FROM all_predictions WHERE source='oos'").fetchall()
    conn.close()

    assert [r["run_id"] for r in rows_after] == ["20260710-ARIMA-SPY-D1"]   # seul le survivant reste visible


def test_all_predictions_view_excludes_weekly_horizon_1_rows(tmp_path):
    """BRIEF_audit_combinaisons.md : horizon=1 signifie 'D+1' pour un daily natif mais
    'W+1' pour un weekly natif -- meme valeur numerique, prediction totalement
    differente. La vue all_predictions (lue par generate_sim_trades, des regles de
    trading pensees pour du D+1 uniquement) ne doit jamais laisser passer un W+1."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    daily_row = {
        "run_id": "r1", "model": "TSDiff", "asset": "BTC-USD", "horizon": 1,
        "regime": "unknown", "cutoff_date": "2025-12-05", "target_date": "2025-12-06",
        "last_close": 100.0, "y_pred": 101.0, "y_lower": 95.0, "y_upper": 107.0,
        "y_true": 102.0, "source": "oos",
    }
    weekly_row = {
        **daily_row, "run_id": "r2", "target_date": "2025-12-12",
        "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": "W+1",
    }
    st.insert_oos_predictions([daily_row, weekly_row], db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT run_id FROM all_predictions WHERE source='oos'").fetchall()
    conn.close()
    assert [r["run_id"] for r in rows] == ["r1"]   # seule la ligne daily/D+1 est visible


def test_reconcile_never_touches_live_sim_trades(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    td.save_prediction(make_prediction_record(
        tc_id="TC_LIVE", model="ARIMA-GARCH", asset="SPY", target_date="2026-06-02",
        last_close=100.0, y_pred=101.0, y_lower=95.0, y_upper=107.0,
    ), db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="bull_calm_d1", source="live")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    live_trade_before = dict(conn.execute("SELECT * FROM sim_trades WHERE source='live'").fetchone())
    conn.close()
    assert live_trade_before is not None

    # lignes oos sans rapport (upsert ordinaire, plus de doublon possible depuis
    # BRIEF_prevention_doublons.md), pour s'assurer que la réconciliation tourne pour
    # de vrai même quand elle n'a structurellement plus rien à faire
    for name in ("20260701-ARIMA-BTC-USD-D1", "20260710-ARIMA-BTC-USD-D1"):
        run_dir = write_fake_run_dir(tmp_path / "runs", name=name, asset="BTC-USD", rows=_DUP_ROWS)
        log_rows, _ = st.build_oos_prediction_rows(run_dir)
        st.insert_oos_predictions(log_rows, db_path=db_path)
    td.flag_daily_duplicates(db_path=db_path)

    st.reconcile_oos_sim_trades(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    live_trade_after = dict(conn.execute("SELECT * FROM sim_trades WHERE source='live'").fetchone())
    n_live = conn.execute("SELECT COUNT(*) FROM sim_trades WHERE source='live'").fetchone()[0]
    conn.close()

    assert n_live == 1
    assert live_trade_after == live_trade_before   # strictement intact, aucune colonne modifiée


def test_reconcile_oos_sim_trades_is_a_safe_noop_once_no_duplicates_remain(tmp_path):
    """Depuis BRIEF_prevention_doublons.md, reconcile_oos_sim_trades ne peut plus être
    appelée en toute sécurité que sur une base sans doublon OOS physique (cf. sa
    docstring : son étape de régénération s'auto-initialise et l'auto-init pose
    désormais l'index dur, qui refuserait des doublons encore présents). C'est donc
    structurellement son SEUL mode d'usage supporté désormais -- vérifie qu'appelée
    deux fois sur une base normale (aucun doublon, ingérée via le chemin standard),
    elle ne supprime ni ne régénère jamais rien (no-op), et ne casse rien."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    run_dir = write_fake_run_dir(tmp_path / "runs", name="20260710-ARIMA-SPY-D1", rows=_DUP_ROWS)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="bull_calm_d1", source="oos")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    state_before = [dict(r) for r in conn.execute(
        "SELECT * FROM sim_trades WHERE source='oos' ORDER BY id"
    ).fetchall()]
    conn.close()
    assert len(state_before) == 1

    result_first = st.reconcile_oos_sim_trades(db_path=db_path)
    assert result_first == {"rule_versions": ["bull_calm_d1"], "n_deleted": 0, "n_regenerated": 0}

    result_second = st.reconcile_oos_sim_trades(db_path=db_path)
    assert result_second == result_first

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    state_after = [dict(r) for r in conn.execute(
        "SELECT * FROM sim_trades WHERE source='oos' ORDER BY id"
    ).fetchall()]
    conn.close()
    assert state_after == state_before


# ── BRIEF_prevention_doublons.md : index dur sans run_id + upsert keep-latest ──

def test_insert_oos_predictions_upsert_keeps_latest_run(tmp_path):
    """§5/§9 du brief : deux ingestions sur la même clé métier (source, model, asset,
    horizon, cutoff_date) avec des run_id différents -> 1 seule ligne en base,
    contenu = celui du DERNIER run (upsert « garde le dernier », pas le premier)."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    rows_early = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 102.0], "predicted": [999.0, 103.0],
        "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 110.0],
    }
    rows_late = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 102.0], "predicted": [999.0, 108.0],   # valeurs différentes au 2e run
        "pi_lower": [999.0, 96.0], "pi_upper": [999.0, 112.0],
    }
    run_a = write_fake_run_dir(tmp_path / "runs", name="20260701-ARIMA-SPY-D1", rows=rows_early)
    run_b = write_fake_run_dir(tmp_path / "runs", name="20260710-ARIMA-SPY-D1", rows=rows_late)

    log_a, _ = st.build_oos_prediction_rows(run_a)
    st.insert_oos_predictions(log_a, db_path=db_path)
    log_b, _ = st.build_oos_prediction_rows(run_b)
    st.insert_oos_predictions(log_b, db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT run_id, y_pred, y_lower, y_upper FROM predictions WHERE source='oos'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1   # pas d'empilement : la clé métier est unique par construction
    assert rows[0]["run_id"] == "20260710-ARIMA-SPY-D1"   # provenance = le dernier run
    assert rows[0]["y_pred"] == pytest.approx(108.0)      # valeurs du dernier run, pas du premier
    assert rows[0]["y_lower"] == pytest.approx(96.0)


def test_insert_oos_predictions_sets_real_flag_independently_of_source(tmp_path):
    """Cas exact des backfills (8, 11, 13, 14, 17-20/07) : une prédiction réelle rejouée
    via le chemin OOS après une panne de prod doit porter real_flag='live' MEME si
    source='oos' -- les deux colonnes portent des sens indépendants (cf.
    tracking_db.compute_real_flag)."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    rows_fake = {
        "date": pd.to_datetime(["2026-07-05", "2026-07-06"]),   # cutoff_date=07-05 < seuil réel
        "actual": [100.0, 102.0], "predicted": [999.0, 103.0],
        "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 110.0],
    }
    rows_real = {
        "date": pd.to_datetime(["2026-07-06", "2026-07-07"]),   # cutoff_date=07-06 >= seuil réel
        "actual": [100.0, 102.0], "predicted": [999.0, 103.0],
        "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 110.0],
    }
    run_fake = write_fake_run_dir(tmp_path / "runs", name="20260705-ARIMA-SPY-D1", rows=rows_fake)
    run_real = write_fake_run_dir(tmp_path / "runs", name="20260706-ARIMA-SPY-D1", rows=rows_real)

    log_fake, _ = st.build_oos_prediction_rows(run_fake)
    log_real, _ = st.build_oos_prediction_rows(run_real)
    st.insert_oos_predictions(log_fake, db_path=db_path)
    st.insert_oos_predictions(log_real, db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = {r["cutoff_date"]: r for r in conn.execute("SELECT * FROM predictions WHERE source='oos'")}
    conn.close()
    assert rows["2026-07-05"]["source"] == "oos"
    assert rows["2026-07-05"]["real_flag"] == "oos"      # fausse : avant le seuil
    assert rows["2026-07-06"]["source"] == "oos"
    assert rows["2026-07-06"]["real_flag"] == "live"     # vraie malgré source='oos' (backfill)


def test_insert_oos_predictions_defaults_missing_frequency_fields_to_daily(tmp_path):
    """build_oos_prediction_rows() (daily uniquement) ne fournit pas frequence/
    horizon_type/horizon_unit -- insert_oos_predictions doit les défaulter à
    'daily'/'daily'/'D+{horizon}' plutôt que planter ou les laisser NULL."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 102.0], "predicted": [999.0, 103.0],
        "pi_lower": [999.0, 95.0], "pi_upper": [999.0, 110.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", name="20260701-ARIMA-SPY-D1", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    assert "frequence" not in log_rows[0]   # confirme la prémisse : le builder ne les fournit pas

    st.insert_oos_predictions(log_rows, db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE source='oos'").fetchone()
    conn.close()
    assert row["frequence"] == "daily"
    assert row["horizon_type"] == "daily"
    assert row["horizon_unit"] == "D+1"


def test_insert_oos_predictions_weekly_row_coexists_with_daily_same_key(tmp_path):
    """Coeur du garde-fou BRIEF_audit_combinaisons.md : une ligne 'weekly' passée
    explicitement à insert_oos_predictions doit coexister avec une ligne 'daily' sur
    le même (model, asset, horizon, cutoff_date) -- pas de collision silencieuse sur
    l'index OOS étendu."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    daily_row = {
        "run_id": "r1", "model": "TSDiff", "asset": "BTC-USD", "horizon": 1,
        "regime": "unknown", "cutoff_date": "2025-12-05", "target_date": "2025-12-06",
        "last_close": 100.0, "y_pred": 101.0, "y_lower": 95.0, "y_upper": 107.0,
        "y_true": 102.0, "source": "oos",
    }
    weekly_row = {
        **daily_row, "run_id": "r2", "target_date": "2025-12-12",
        "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": "W+1",
    }
    st.insert_oos_predictions([daily_row], db_path=db_path)
    st.insert_oos_predictions([weekly_row], db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT frequence, horizon_type FROM predictions WHERE source='oos' ORDER BY frequence"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert {r["frequence"] for r in rows} == {"daily", "weekly"}


def test_ingest_oos_reingestion_is_idempotent(tmp_path):
    """§8/§9 : rejouer ingest_oos sur les mêmes Run/*-D1 n'ajoute aucune ligne
    predictions ni ne duplique un signal (upsert idempotent : mêmes données -> même
    ligne écrasée par elle-même)."""
    run_root = tmp_path / "Run"
    write_fake_run_dir(run_root, name="20260710-ARIMA-SPY-D1", rows=_DUP_ROWS)
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    stats_first = st.ingest_oos(run_root=str(run_root), db_path=db_path)
    stats_second = st.ingest_oos(run_root=str(run_root), db_path=db_path)

    conn = sqlite3.connect(db_path)
    n_predictions = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]
    n_bull_calm_trades = conn.execute(
        "SELECT COUNT(*) FROM sim_trades WHERE source='oos' AND rule_version='bull_calm_d1'"
    ).fetchone()[0]
    conn.close()

    assert n_predictions == 1                       # aucun doublon créé par le rejeu
    assert stats_first["combos"] == stats_second["combos"] == 1
    assert n_bull_calm_trades == 1                   # signal reconstruit, jamais dupliqué


def _fake_d7_prices(n_days=20, start="2026-01-05", base=100.0):
    """20 jours de bourse consécutifs, close = base + i (valeurs connues, faciles
    à vérifier dans les tests de reconstruction cutoff/last_close)."""
    dates = pd.bdate_range(start, periods=n_days)
    return {"date": dates, "close": [base + i for i in range(n_days)]}


def test_ingest_oos_also_processes_d7_folders(tmp_path):
    """BRIEF_audit_combinaisons.md : ingest_oos doit ingerer Run/*-D7/ en plus de
    Run/*-D1/ (regression du trou de couverture regime A / D+7 -- glob D1 seul
    laissait des dossiers D7 complets sur disque jamais ingeres)."""
    run_root = tmp_path / "Run"
    write_fake_run_dir(run_root, name="20260710-ARIMA-SPY-D1", rows=_DUP_ROWS)

    prices = _fake_d7_prices()
    target_date = prices["date"][10]   # position 10 -> cutoff a la position 10-5=5
    d7_rows = {
        "date": [target_date], "actual": [111.0], "predicted": [110.5],
        "pi_lower": [105.0], "pi_upper": [115.0],
    }
    write_fake_run_dir(run_root, name="20260710-ARIMA-SPY-D7", horizon_label="D7",
                       rows=d7_rows, prices=prices)
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    stats = st.ingest_oos(run_root=str(run_root), db_path=db_path)
    assert stats["combos"] == 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT horizon, frequence, horizon_type, horizon_unit, cutoff_date, target_date, last_close "
        "FROM predictions WHERE source='oos'"
    ).fetchall()
    conn.close()

    by_horizon = {r["horizon"]: r for r in rows}
    assert set(by_horizon) == {1, 7}
    assert by_horizon[7]["frequence"] == "daily"
    assert by_horizon[7]["horizon_type"] == "daily"
    assert by_horizon[7]["horizon_unit"] == "D+7"
    # cutoff/last_close correctement reconstruits depuis prices.parquet (position
    # 10-5=5 -> date de bourse n5, close=100+5=105.0) -- PAS depuis une ligne t-1
    # inexistante dans predictions.parquet (une seule ligne ici).
    assert by_horizon[7]["cutoff_date"] == str(prices["date"][5].date())
    assert by_horizon[7]["target_date"] == str(target_date.date())
    assert by_horizon[7]["last_close"] == pytest.approx(105.0)

    # the D+7 row must NOT leak into all_predictions (D+1-only trading rules)
    conn = sqlite3.connect(db_path)
    n_all_pred = conn.execute("SELECT COUNT(*) FROM all_predictions WHERE source='oos'").fetchone()[0]
    conn.close()
    assert n_all_pred == 1


def test_build_oos_prediction_rows_d7_drops_all_rows_without_prices_parquet(tmp_path):
    """Garde-fou du bug BRIEF_comparaison_rigoureuse.md : sans prices.parquet, aucune
    reconstruction fiable de cutoff_date/last_close n'est possible pour du D7 --
    on droppe TOUT plutot que de retomber sur l'ancienne logique t-1 fausse."""
    run_dir = write_fake_run_dir(tmp_path, name="fake-D7", horizon_label="D7", rows=_DUP_ROWS)
    rows, n_dropped = st.build_oos_prediction_rows(run_dir)
    assert rows == []
    assert n_dropped == len(_DUP_ROWS["date"])


def test_build_oos_prediction_rows_d7_drops_target_too_close_to_price_history_start(tmp_path):
    """Un target dans les 5 premiers jours de prices.parquet n'a pas assez
    d'historique pour reculer de 5 pas de bourse -- droppé, pas une exception ni
    une fausse date."""
    prices = _fake_d7_prices()
    target_date = prices["date"][2]   # position 2 < steps=5 -> pas de cutoff valide
    d7_rows = {
        "date": [target_date], "actual": [102.0], "predicted": [101.5],
        "pi_lower": [98.0], "pi_upper": [105.0],
    }
    run_dir = write_fake_run_dir(tmp_path, name="fake-D7", horizon_label="D7",
                                 rows=d7_rows, prices=prices)
    rows, n_dropped = st.build_oos_prediction_rows(run_dir)
    assert rows == []
    assert n_dropped == 1


def test_rebuild_oos_sim_trades_never_touches_live(tmp_path):
    """§6 : rebuild_oos_sim_trades (appelée par ingest_oos) ne touche jamais
    source='live', même quand elle wipe-et-régénère tout l'OOS."""
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)

    td.save_prediction(make_prediction_record(
        tc_id="TC_LIVE", target_date="2026-06-02",
        last_close=100.0, y_pred=101.0, y_lower=95.0, y_upper=107.0,
    ), db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="bull_calm_d1", source="live")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    live_trade_before = dict(conn.execute("SELECT * FROM sim_trades WHERE source='live'").fetchone())
    conn.close()

    run_dir = write_fake_run_dir(tmp_path / "runs", name="20260710-ARIMA-SPY-D1", rows=_DUP_ROWS)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    result = st.rebuild_oos_sim_trades(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    live_trade_after = dict(conn.execute("SELECT * FROM sim_trades WHERE source='live'").fetchone())
    n_live = conn.execute("SELECT COUNT(*) FROM sim_trades WHERE source='live'").fetchone()[0]
    n_oos_bull_calm = conn.execute(
        "SELECT COUNT(*) FROM sim_trades WHERE source='oos' AND rule_version='bull_calm_d1'"
    ).fetchone()[0]
    conn.close()

    assert n_live == 1
    assert live_trade_after == live_trade_before   # strictement intact
    assert n_oos_bull_calm == 1                     # le combo oos a bien été reconstruit
    assert result["n_regenerated"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Sideways v2 gaté (sideways_gated_d1) — BRIEF_sideways_v2.md §9. Extension de
# sideways_d1 (qui reste strictement inchangée, cf. tests ci-dessus, tous verts sans
# modification). Même bande que les tests sideways_d1 : ref=100, pi_low=96, pi_high=104
# (W=8, Hb=4), predicted=100.3, k=0.10 -> eps=0.8, m=2, h=4. Defaults gate : vb_max=1,
# stress_max=0.30 (§2 du brief). roi <- pnl_shortvol (proxy short-straddle d'évaluation,
# JAMAIS un rendement exécuté -- ne jamais le lire comme un ROI directionnel).
# ═══════════════════════════════════════════════════════════════════════════════

def _fake_gated_trade(d_date, roi, branch, counter, vol_bucket=0, source="oos",
                      reference_price=100.0, pi_lower=96.0, pi_upper=104.0,
                      realized_price=100.0, in_band=1, stress_score=None):
    """Ligne sim_trades gated minimale (dict, supporte t["champ"] comme sqlite3.Row) --
    pour unit-tester _summarize_group_sideways_gated/_gate_sweep sans DB."""
    return {
        "d_date": d_date, "roi": roi, "branch": branch, "counter": counter,
        "vol_bucket": vol_bucket, "source": source, "reference_price": reference_price,
        "pi_lower": pi_lower, "pi_upper": pi_upper, "realized_price": realized_price,
        "in_band": in_band, "stress_score": stress_score,
    }


# ── test_gated_signal_passes_when_calm ────────────────────────────────────────

def test_gated_signal_passes_when_calm():
    signal, branch, counter, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5,
        vol_bucket=1, stress_score=0.18, source="live")
    assert signal is True
    assert gated_out == 0
    assert branch == 1
    assert counter == 2
    assert pnl == pytest.approx(1.0 - 0.5 / 4.0)


# ── test_gated_signal_blocked_high_vol ────────────────────────────────────────

def test_gated_signal_blocked_high_vol():
    signal, branch, counter, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5,
        vol_bucket=2, stress_score=0.10, source="live", vb_max=1)
    assert signal is False
    assert gated_out == 1          # flat suspect, pas un simple flat (signal_v1 était vrai)
    assert branch is None
    assert counter == 0
    assert pnl is None


# ── test_gated_signal_blocked_high_stress_live_only ──────────────────────────

def test_gated_signal_blocked_high_stress_live_only():
    signal_live, branch, counter, pnl, in_band, gated_out_live, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5,
        vol_bucket=0, stress_score=0.50, source="live", stress_max=0.30)
    assert signal_live is False
    assert gated_out_live == 1

    # même stress_score élevé, mais source="oos" -> le terme stress est neutralisé (§5.2)
    signal_oos, branch, counter, pnl, in_band, gated_out_oos, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5,
        vol_bucket=0, stress_score=0.50, source="oos", stress_max=0.30)
    assert signal_oos is True
    assert gated_out_oos == 0


# ── test_gate_undefined_when_vol_bucket_none ─────────────────────────────────

def test_gate_undefined_when_vol_bucket_none():
    signal, branch, counter, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5,
        vol_bucket=None, stress_score=None, source="live")
    assert signal is True
    assert gated_out == 0   # vol_bucket indisponible -> gate dégénère en pass (§5.1)


def test_kpi_report_counts_n_gate_undefined(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.5], "predicted": [999.0, 100.3],
        "pi_lower": [999.0, 96.0], "pi_upper": [999.0, 104.0],
    }
    # groupe (asset, model) d'une seule ligne -> terciles indisponibles (< 3, §5.1)
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos")

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",))
    entry = report[0]
    assert entry["n_signaux"] == 1
    assert entry["n_gate_undefined"] == 1


# ── test_counter_identical_to_v1 ──────────────────────────────────────────────

@pytest.mark.parametrize("realized,expected_branch,expected_counter", [
    (100.5, 1, 2), (103.0, 2, 1), (106.0, 3, -1), (111.0, 4, -2),
])
def test_counter_identical_to_v1(realized, expected_branch, expected_counter):
    signal_v1, branch_v1, counter_v1, *_rest_v1 = st.sideways_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized)
    signal_v2, branch_v2, counter_v2, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized,
        vol_bucket=0, stress_score=0.0, source="live")
    assert (branch_v2, counter_v2) == (branch_v1, counter_v1) == (expected_branch, expected_counter)
    assert signal_v1 is True and signal_v2 is True and gated_out == 0


# ── test_pnl_shortvol_monotone ────────────────────────────────────────────────

def test_pnl_shortvol_monotone():
    pnls = []
    for realized in (100.5, 101.5, 103.0, 104.0):   # move croissant : 0.5, 1.5, 3.0, 4.0
        _, _, _, pnl, *_ = st.sideways_gated_d1(
            ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized,
            vol_bucket=0, source="oos")
        pnls.append(pnl)
    assert pnls == sorted(pnls, reverse=True)   # décroît strictement quand move croît
    assert len(set(pnls)) == len(pnls)

    _, _, _, pnl_immobile, *_ = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=SW_REF,
        vol_bucket=0, source="oos")
    assert pnl_immobile == pytest.approx(1.0)   # immobile parfait -> +1

    _, _, _, pnl_breakeven, *_ = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=SW_PI_HIGH,
        vol_bucket=0, source="oos")
    assert pnl_breakeven == pytest.approx(0.0)   # pile sur une borne du PI -> breakeven


# ── test_pnl_shortvol_clip_branch4 ────────────────────────────────────────────

@pytest.mark.parametrize("realized", [111.0, 150.0, 500.0])
def test_pnl_shortvol_clip_branch4(realized):
    signal, branch, counter, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=realized,
        vol_bucket=0, source="oos")
    assert branch == 4
    assert pnl == pytest.approx(-1.0)   # plancher defined-risk, quelle que soit l'ampleur


# ── test_pnl_shortvol_none_when_open ──────────────────────────────────────────

def test_pnl_shortvol_none_when_open():
    signal, branch, counter, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=None,
        vol_bucket=0, source="live")
    assert signal is True
    assert branch is None
    assert counter is None
    assert pnl is None
    assert gated_out == 0


# ── test_v1_roi_still_null ────────────────────────────────────────────────────

def test_v1_roi_still_null(tmp_path):
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
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    v1_trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_d1'").fetchone()
    v2_trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_gated_d1'").fetchone()
    conn.close()
    assert v1_trade["roi"] is None            # v1 non contaminée (§0/§3 du brief)
    assert v2_trade["roi"] is not None         # v2 stocke bien pnl_shortvol dans roi


# ── test_direction_ok_null ────────────────────────────────────────────────────

def test_direction_ok_null():
    result = st.RULES["sideways_gated_d1"](
        ref=SW_REF, predicted=100.3, pi_low=SW_PI_LOW, pi_high=SW_PI_HIGH, realized=100.5,
        vol_bucket=0, stress_score=0.0, source="live")
    signal_valid, branch, counter, roi, direction_ok, in_band, degenerate, gated_out = result
    assert direction_ok is None
    assert signal_valid is True
    assert gated_out == 0


def test_gated_rule_registered_in_dispatch():
    assert "sideways_gated_d1" in st.RULES


# ── test_degenerate_pi_excluded ───────────────────────────────────────────────

def test_degenerate_pi_excluded():
    signal, branch, counter, pnl, in_band, gated_out, degenerate = st.sideways_gated_d1(
        ref=100.0, predicted=100.0, pi_low=100.0, pi_high=100.0, realized=100.0,
        vol_bucket=0, source="oos")
    assert degenerate == 1
    assert pnl is None   # §5.3 : pas de division par zéro (W=0), pnl_shortvol non défini


def test_degenerate_pi_excluded_from_gated_kpis_by_default(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.0], "predicted": [999.0, 100.0],
        "pi_lower": [999.0, 100.0], "pi_upper": [999.0, 100.0],   # W=0 -> dégénéré
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_gated_d1'").fetchone()
    conn.close()
    assert trade["degenerate_pi"] == 1

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",))
    assert report[0]["n_signaux"] == 0


# ── test_vol_tercile_proxy_per_group ──────────────────────────────────────────

def test_vol_tercile_proxy_per_group():
    rows = [
        {"id": 1, "asset": "SPY", "model": "M", "pi_lower": 95.0, "pi_upper": 100.0},   # W=5
        {"id": 2, "asset": "SPY", "model": "M", "pi_lower": 90.0, "pi_upper": 100.0},   # W=10
        {"id": 3, "asset": "SPY", "model": "M", "pi_lower": 80.0, "pi_upper": 100.0},   # W=20
        {"id": 4, "asset": "BTC-USD", "model": "M", "pi_lower": 190.0, "pi_upper": 200.0},  # groupe séparé
    ]
    buckets = st._vol_bucket_proxy_for_rows(rows)
    assert buckets[1] == 0   # plus petite largeur du groupe SPY -> tercile bas
    assert buckets[2] == 1
    assert buckets[3] == 2   # plus grande largeur -> tercile haut
    assert buckets[4] is None   # groupe BTC-USD n'a qu'une ligne -> terciles indisponibles (§5.1)


def test_vol_tercile_proxy_grouped_by_asset_and_model_not_asset_alone():
    rows = [
        {"id": 1, "asset": "SPY", "model": "A", "pi_lower": 95.0, "pi_upper": 100.0},
        {"id": 2, "asset": "SPY", "model": "B", "pi_lower": 90.0, "pi_upper": 100.0},
    ]
    buckets = st._vol_bucket_proxy_for_rows(rows)
    # chaque groupe (SPY,A) et (SPY,B) n'a qu'1 ligne -> indisponible, même si même asset
    assert buckets[1] is None
    assert buckets[2] is None


# ── test_gate_sweep_monotone ──────────────────────────────────────────────────

def test_gate_sweep_monotone(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime([f"2026-02-{i:02d}" for i in range(1, 8)]),
        "actual":    [100.0, 100.5, 100.3, 100.6, 100.2, 100.4, 100.1],
        "predicted": [999.0, 100.3, 100.3, 100.3, 100.3, 100.3, 100.3],
        "pi_lower":  [999.0,  96.0,  94.0,  92.0,  90.0,  88.0,  86.0],   # W croissant : 8,10,12,14,16,18
        "pi_upper":  [999.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    # vb_max=2 à la génération : permissif, tous les signaux v1 deviennent tradables
    # (le proxy assigne buckets 0,0,1,1,2,2 par rang -- vérifié par le test précédent)
    n = st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=2)
    assert n == 6

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",), gate_values=[(2, 0.30), (1, 0.30), (0, 0.30)])
    sweep = report[0]["gate_sensitivity"]
    n_signals = [s["n_signal_tradable"] for s in sweep]
    assert n_signals == sorted(n_signals, reverse=True)   # durcir vb_max réduit (monotone) le volume
    assert n_signals[0] > n_signals[-1]
    assert n_signals == [6, 4, 2]


def test_gate_values_rejected_for_other_rule_version(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    with pytest.raises(ValueError):
        st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_d1",
                      group_by=("asset",), gate_values=[(1, 0.30)])


# ── test_sharpness_flags_wide_band ────────────────────────────────────────────

def test_sharpness_flags_wide_band(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows_narrow = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.0], "predicted": [999.0, 100.05],
        "pi_lower": [999.0, 98.0], "pi_upper": [999.0, 102.0],   # W=4
    }
    rows_wide = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.0], "predicted": [999.0, 100.05],
        "pi_lower": [999.0, 90.0], "pi_upper": [999.0, 110.0],   # W=20
    }
    run_narrow = write_fake_run_dir(tmp_path / "runs", name="20260701-M-NARROW-D1",
                                    asset="NARROW", rows=rows_narrow)
    run_wide = write_fake_run_dir(tmp_path / "runs", name="20260701-M-WIDE-D1",
                                  asset="WIDE", rows=rows_wide)
    for run_dir in (run_narrow, run_wide):
        log_rows, _ = st.build_oos_prediction_rows(run_dir)
        st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=2)

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",))
    by_asset = {r["asset"]: r for r in report}
    assert by_asset["NARROW"]["taux_justesse"] == by_asset["WIDE"]["taux_justesse"] == 1.0
    # même justesse, mais la bande large "triche" -- rel_width_mean le révèle (§8.4)
    assert by_asset["WIDE"]["rel_width_mean"] > by_asset["NARROW"]["rel_width_mean"]


# ── test_cvar_and_freq_floor ──────────────────────────────────────────────────

def test_cvar_and_freq_floor(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    n_calm, n_breakout = 8, 2
    dates = pd.to_datetime([f"2026-02-{i:02d}" for i in range(1, 1 + n_calm + n_breakout + 1)])
    actual = [100.0] + [100.1] * n_calm + [130.0] * n_breakout
    predicted = [999.0] + [100.05] * (n_calm + n_breakout)
    pi_lower = [999.0] + [96.0] * (n_calm + n_breakout)
    pi_upper = [999.0] + [104.0] * (n_calm + n_breakout)
    rows = {"date": dates, "actual": actual, "predicted": predicted,
            "pi_lower": pi_lower, "pi_upper": pi_upper}
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=2)

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",))
    entry = report[0]
    assert entry["n_signaux"] > 0
    assert entry["branch_distribution"][4] > 0   # au moins un gros breakout dans ce jeu de données
    assert entry["freq_floor"] == round(entry["branch_distribution"][4] / entry["n_signaux"], 4)
    assert entry["cvar_5_shortvol"] <= entry["pnl_shortvol_mean"]
    assert entry["pnl_skew"] is not None


# ── test_calmar_uses_cumulative_drawdown ──────────────────────────────────────

def test_calmar_uses_cumulative_drawdown():
    signals = [
        _fake_gated_trade("2026-02-01", 0.3, 2, 1),
        _fake_gated_trade("2026-02-02", -0.5, 3, -1),
        _fake_gated_trade("2026-02-03", 0.3, 2, 1),
        _fake_gated_trade("2026-02-04", -0.5, 3, -1),
        _fake_gated_trade("2026-02-05", 0.3, 2, 1),
    ]
    entry = st._summarize_group_sideways_gated(log_rows=[], signals=signals, n_open=0, n_gated_out=0)
    pnl_mean = sum(s["roi"] for s in signals) / len(signals)
    # cumulative (ordre d_date) : 0.3, -0.2, 0.1, -0.4, -0.1 -> peak=0.3 -> max_drawdown=0.7
    # (PAS |min(pnl)|=0.5 : le pnl le plus bas n'est pas le creux de la série cumulée -- ce
    # test casserait une implémentation naïve basée sur pnl_min au lieu du drawdown cumulé)
    expected_max_drawdown = 0.7
    assert entry["calmar_shortvol"] == round(pnl_mean / expected_max_drawdown, 6)
    assert entry["calmar_shortvol"] != pytest.approx(
        pnl_mean / abs(min(s["roi"] for s in signals)))


# ── test_no_lookahead (gated) ─────────────────────────────────────────────────

def test_no_lookahead_vol_bucket_computed_from_pi_bounds_not_realized():
    rows = [
        {"id": 1, "asset": "SPY", "model": "M", "pi_lower": 95.0, "pi_upper": 100.0},
        {"id": 2, "asset": "SPY", "model": "M", "pi_lower": 90.0, "pi_upper": 100.0},
        {"id": 3, "asset": "SPY", "model": "M", "pi_lower": 80.0, "pi_upper": 100.0},
    ]
    buckets_a = st._vol_bucket_proxy_for_rows(rows)
    # ajouter realized_price n'a aucun effet -- le proxy ne le lit jamais (aucun look-ahead)
    rows_with_realized = [dict(r, realized_price=999.0) for r in rows]
    buckets_b = st._vol_bucket_proxy_for_rows(rows_with_realized)
    assert buckets_a == buckets_b


def test_no_lookahead_gated_trade_vol_bucket_dated_at_d(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime([f"2026-02-{i:02d}" for i in range(1, 6)]),
        "actual":    [100.0, 100.1, 100.0, 100.2, 100.1],
        "predicted": [999.0, 100.05, 100.05, 100.05, 100.05],
        "pi_lower":  [999.0,  96.0,  94.0,  92.0,  90.0],   # W croissant, connu à D
        "pi_upper":  [999.0, 104.0, 104.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=2)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute(
        "SELECT d_date, vol_bucket, pi_lower, pi_upper FROM sim_trades "
        "WHERE rule_version='sideways_gated_d1' ORDER BY d_date"
    ).fetchall()
    conn.close()
    widths = [t["pi_upper"] - t["pi_lower"] for t in trades]
    assert widths == sorted(widths)   # confirme la construction du fixture
    # vol_bucket croît avec W (bornes connues à D) -- jamais influencé par realized (D+1)
    assert [t["vol_bucket"] for t in trades] == sorted(t["vol_bucket"] for t in trades)


def test_no_lookahead_reference_price_is_actual_t_minus_1_for_gated(tmp_path):
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 100.1, 999.0],   # actual[t=2] volontairement absurde
        "predicted": [999.0, 100.05, 100.05],
        "pi_lower":  [999.0,  96.0,  96.0],
        "pi_upper":  [999.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path, rows=rows)
    log_rows, n_dropped = st.build_oos_prediction_rows(run_dir)
    assert n_dropped == 0
    row_t1 = log_rows[0]
    assert row_t1["last_close"] == pytest.approx(100.0)   # actual[t-1=0], jamais actual[t=1]
    assert row_t1["y_true"] == pytest.approx(100.1)


# ── gated_out : persistance + exclusion des KPI tradables ────────────────────

def test_gated_out_row_persisted_but_excluded_from_kpi_signaux(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04"]),
        "actual":    [100.0, 100.1, 100.0, 100.2],
        "predicted": [999.0, 100.05, 100.05, 100.05],
        "pi_lower":  [999.0,  96.0,  96.0,  96.0],
        "pi_upper":  [999.0, 104.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    # vb_max=-1 : aucun vol_bucket (0/1/2) n'est <= -1 -> bloque tout (groupe de 3 lignes,
    # terciles bien définis) -- force gated_out=1 sur toutes les lignes sideways
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=-1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_gated_d1'").fetchall()
    conn.close()
    assert len(trades) == 3   # journalisées (flat suspect), pas "aucune ligne" comme un flat ordinaire
    assert all(t["gated_out"] == 1 for t in trades)
    assert all(t["signal_valid"] == 0 for t in trades)
    assert all(t["branch"] is None and t["roi"] is None for t in trades)

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",))
    entry = report[0]
    assert entry["n_signaux"] == 0
    assert entry["n_gated_out"] == 3
    assert entry["n_signal_v1"] == 3


def test_gated_out_rows_are_idempotent_on_replay(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "actual":    [100.0, 100.1, 100.0],
        "predicted": [999.0, 100.05, 100.05],
        "pi_lower":  [999.0,  96.0,  96.0],
        "pi_upper":  [999.0, 104.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    n1 = st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=-1)
    n2 = st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos", vb_max=-1)
    assert n1 == 2   # 2 lignes sideways (t=1, t=2)
    assert n2 == 0   # rejouer ne duplique rien


def test_gated_live_open_when_realized_none(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    td.save_prediction(make_prediction_record(
        target_date="2099-01-01", last_close=100.0, y_pred=100.05, y_lower=96.0, y_upper=104.0,
    ), db_path=db_path)
    n_new = st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="live")
    assert n_new == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trade = conn.execute("SELECT * FROM sim_trades WHERE rule_version='sideways_gated_d1'").fetchone()
    conn.close()
    assert trade["status"] == "open"
    assert trade["signal_valid"] == 1
    assert trade["gated_out"] == 0
    assert trade["roi"] is None


def test_gated_kpi_report_never_labels_pnl_as_roi(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    rows = {
        "date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        "actual": [100.0, 100.1], "predicted": [999.0, 100.05],
        "pi_lower": [999.0, 96.0], "pi_upper": [999.0, 104.0],
    }
    run_dir = write_fake_run_dir(tmp_path / "runs", rows=rows)
    log_rows, _ = st.build_oos_prediction_rows(run_dir)
    st.insert_oos_predictions(log_rows, db_path=db_path)
    st.generate_sim_trades(db_path=db_path, rule_version="sideways_gated_d1", source="oos")

    report = st.kpi_report(db_path=db_path, source="oos", rule_version="sideways_gated_d1",
                           group_by=("asset",))
    entry = report[0]
    assert not any(k.startswith("roi_") for k in entry)   # jamais "roi_*" -- toujours pnl_shortvol_*
    assert "pnl_shortvol_mean" in entry


def test_migration_adds_gated_columns_to_pre_v2_sim_trades(tmp_path):
    db_path = str(tmp_path / "t.db")
    st.init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE sim_trades")
    conn.execute("""
        CREATE TABLE sim_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, rule_version TEXT NOT NULL, run_id TEXT NOT NULL,
            model TEXT NOT NULL, asset TEXT NOT NULL, horizon INTEGER NOT NULL, regime TEXT NOT NULL,
            source TEXT NOT NULL, d_date TEXT NOT NULL, target_date TEXT NOT NULL,
            reference_price REAL NOT NULL, predicted REAL NOT NULL, pi_lower REAL NOT NULL,
            pi_upper REAL NOT NULL, realized_price REAL, signal_valid INTEGER NOT NULL,
            direction_ok INTEGER, branch INTEGER, counter INTEGER, roi REAL,
            degenerate_pi INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
            evaluated_at TEXT,
            UNIQUE (rule_version, source, run_id, model, asset, horizon, d_date)
        )
    """)   # schéma pré-v2 : sans in_band/vol_bucket/stress_score/gated_out
    conn.commit()
    conn.close()

    st.init_db(db_path)   # doit migrer sans lever

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sim_trades)")}
    conn.close()
    assert {"in_band", "vol_bucket", "stress_score", "gated_out"} <= cols
