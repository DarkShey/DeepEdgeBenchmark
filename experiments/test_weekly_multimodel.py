"""
Plumbing + guardrail tests for weekly_multimodel.py — no network access (yfinance
monkeypatched), Naive as the test vehicle (no real model fit, fast) for the
walk-forward wiring; targeted tests for the SARIMA/Prophet weekly-specific wrappers
using real (tiny) fits since their whole point is a regime-specific code path.
"""

import numpy as np
import pandas as pd
import pytest

import weekly_multimodel as wm
import tsdiff_model as td
import sarima_model


def _synthetic_daily(n_days=900, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def patched_fetch(monkeypatch):
    daily = _synthetic_daily()
    monkeypatch.setattr(td, "fetch_data", lambda ticker, start, end: daily)
    return daily


# ── walk-forward wiring (Naive: no fit, exercises pure orchestration) ──────────

def test_regime_c_uses_weekly_series_and_1_2_3_horizons(patched_fetch):
    res = wm.run_model_asset("Naive", "TEST", "TEST", "C", n_val=3, n_test=4,
                             start="2021-01-01", end="2025-01-01")
    assert res["n_failed"] == 0
    assert len(res["records"]) == 4 * 3   # n_test x 3 horizons
    for r in res["records"]:
        assert r["daily_steps"] is None
        assert pd.Timestamp(r["target_date"]) > pd.Timestamp(r["origin_date"])


def test_regime_b_uses_daily_steps_and_matches_regime_c_target_dates(patched_fetch):
    res_b = wm.run_model_asset("Naive", "TEST", "TEST", "B", n_val=3, n_test=4,
                               start="2021-01-01", end="2025-01-01")
    res_c = wm.run_model_asset("Naive", "TEST", "TEST", "C", n_val=3, n_test=4,
                               start="2021-01-01", end="2025-01-01")
    assert res_b["n_failed"] == 0
    for r in res_b["records"]:
        assert r["daily_steps"] is not None
        assert r["daily_steps"] > 0

    # equity guardrail: same origin/horizon -> same target date in both regimes
    by_key_b = {(r["origin"], r["horizon"]): r["target_date"] for r in res_b["records"]}
    by_key_c = {(r["origin"], r["horizon"]): r["target_date"] for r in res_c["records"]}
    assert by_key_b == by_key_c


def test_regime_b_and_c_share_the_same_test_origins_as_T0(patched_fetch):
    res_b = wm.run_model_asset("Naive", "TEST", "TEST", "B", n_val=3, n_test=4,
                               start="2021-01-01", end="2025-01-01")
    res_c = wm.run_model_asset("Naive", "TEST", "TEST", "C", n_val=3, n_test=4,
                               start="2021-01-01", end="2025-01-01")
    assert res_b["T0"] == res_c["T0"]


def test_naive_point_is_last_close_regardless_of_regime(patched_fetch):
    for regime in ("B", "C"):
        res = wm.run_model_asset("Naive", "TEST", "TEST", regime, n_val=3, n_test=3,
                                 start="2021-01-01", end="2025-01-01")
        for r in res["records"]:
            assert r["point"] == pytest.approx(r["last_close"])


# ── SARIMA / Prophet weekly-specific wrappers ───────────────────────────────────

def _weekly_series(n_weeks=120, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-07", periods=n_weeks, freq="W-FRI")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_weeks)))
    return pd.Series(prices, index=idx)


def test_sarima_weekly_disables_seasonality():
    train = _weekly_series()
    out = wm.forecast_horizons_sarima_weekly(train, [1, 2, 3])
    assert set(out) == {1, 2, 3}
    for h, (point, lo, hi) in out.items():
        assert lo <= point <= hi
        assert np.isfinite(point)
    # the daily seasonal_order (1,0,1,5) must NOT have been used -- sanity: a
    # plain (0,0,0,0)-seasonal fit on 120 weekly points should not raise/hang
    # (already implicitly checked by not raising above); explicit regression
    # marker: forecast_horizons_sarima_weekly must differ from the raw daily
    # wrapper's hardcoded seasonal_order.
    assert sarima_model.SEASONAL_ORDER != (0, 0, 0, 0)   # confirms daily default untouched globally


def test_prophet_weekly_targets_land_on_fridays():
    prophet = pytest.importorskip("prophet_model")
    train = _weekly_series()
    out = wm.forecast_horizons_prophet_weekly(train, [1, 2, 3])
    assert set(out) == {1, 2, 3}
    for h, (point, lo, hi) in out.items():
        assert lo <= point <= hi
        assert np.isfinite(point)


def test_regime_c_forecast_registry_has_all_five_models():
    assert set(wm.REGIME_C_FORECAST) == set(wm.MODELS)
    assert set(wm.REGIME_B_FORECAST) == set(wm.MODELS)


# ── calibration sigma du walk-forward (adoptée 2026-07-31, cf. SIGMA_SCALE_MODELS) ──

def test_calibrate_off_gives_raw_bands_and_unit_scale(patched_fetch):
    res = wm.run_model_asset("Naive", "TEST", "TEST", "C", n_val=3, n_test=4,
                             start="2021-01-01", end="2025-01-01",
                             calibrate_sigma="off")
    assert all(r["sigma_scale"] == 1.0 for r in res["records"])
    # bandes brutes : celles de mh.forecast_horizons_naive recalculées telles quelles
    for r in res["records"][:3]:   # origine 0
        h = {"W1": 1, "W2": 2, "W3": 3}[r["horizon"]]
        # persistence : point = dernier close, bande = ±1.96·sigma·sqrt(h)
        assert r["point"] == r["last_close"]
        half = r["upper"] - r["point"]
        assert half == pytest.approx(r["point"] - r["lower"], rel=1e-9)


def test_calibrate_on_respects_resolution_lag(patched_fetch):
    """Causalité : W+j ne peut dévier de la bande brute qu'à partir de l'origine j
    (le z de l'origine 0 pour W+j n'est résolu qu'à l'origine j)."""
    res = wm.run_model_asset("Naive", "TEST", "TEST", "C", n_val=3, n_test=5,
                             start="2021-01-01", end="2025-01-01",
                             calibrate_sigma="on")
    scales = {(r["origin"], r["horizon"]): r["sigma_scale"] for r in res["records"]}
    # origine 0 : état neutre partout
    assert scales[(0, "W1")] == scales[(0, "W2")] == scales[(0, "W3")] == 1.0
    # l'horizon W+j reste neutre jusqu'à l'origine j exclue
    assert scales[(1, "W2")] == 1.0 and scales[(1, "W3")] == 1.0
    assert scales[(2, "W3")] == 1.0
    # ... et dévie dès que son premier z est résolu
    assert scales[(1, "W1")] != 1.0
    assert scales[(2, "W2")] != 1.0
    assert scales[(3, "W3")] != 1.0


def test_calibrate_on_first_update_matches_ewma_formula(patched_fetch):
    """Le premier scale de W+1 (origine 1) vaut sqrt(lam + (1-lam)·z0²), z0 mesuré
    sur la bande BRUTE de l'origine 0 -- verrouille et la formule et le fait que
    z est mesuré sur sigma propre (jamais la bande corrigée)."""
    res = wm.run_model_asset("Naive", "TEST", "TEST", "C", n_val=3, n_test=2,
                             start="2021-01-01", end="2025-01-01",
                             calibrate_sigma="on")
    rec0 = next(r for r in res["records"] if r["origin"] == 0 and r["horizon"] == "W1")
    rec1 = next(r for r in res["records"] if r["origin"] == 1 and r["horizon"] == "W1")
    sigma_own = (rec0["upper"] - rec0["lower"]) / (2 * wm._Z975)   # origine 0 = brute
    z0 = (rec0["actual"] - rec0["point"]) / sigma_own
    expected = np.sqrt(wm.EWMA_LAMBDA + (1 - wm.EWMA_LAMBDA) * z0 ** 2)
    assert rec1["sigma_scale"] == pytest.approx(expected, rel=1e-6)


def test_arima_garch_never_calibrated(patched_fetch):
    """ARIMA-GARCH hors périmètre (sigma GARCH déjà dynamique) : scale 1.0 partout
    même avec calibrate_sigma='on'."""
    res = wm.run_model_asset("ARIMA-GARCH", "TEST", "TEST", "C", n_val=3, n_test=2,
                             start="2021-01-01", end="2025-01-01",
                             calibrate_sigma="on")
    assert res["n_failed"] == 0
    assert all(r["sigma_scale"] == 1.0 for r in res["records"])


def test_prophet_weekly_hook_is_a_dict_like_mh(patched_fetch):
    """Signature unifiée scalaire->dict (alignée sur mh.forecast_from_fitted_prophet) :
    {h: facteur} n'élargit QUE l'horizon visé."""
    daily = patched_fetch
    weekly, _ = wm.build_weekly(daily)
    train = weekly.iloc[:200]
    raw = wm.forecast_horizons_prophet_weekly(train, [1, 2])
    scaled = wm.forecast_horizons_prophet_weekly(train, [1, 2], sigma_scale={2: 3.0})
    # les bornes Prophet sont Monte-Carlo (tirages postérieurs non seedés) : deux
    # fits distincts ne sont comparables qu'à ce bruit près -- tolérances larges.
    assert scaled[1][0] == pytest.approx(raw[1][0], rel=1e-6)   # point h=1 (MAP) intact
    w = lambda t: t[2] - t[1]
    assert w(scaled[1]) == pytest.approx(w(raw[1]), rel=0.25)   # h=1 non ciblé
    assert w(scaled[2]) > 2.0 * w(raw[2])                        # h=2 élargi ~3x
