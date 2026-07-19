"""
Plumbing + guardrail tests for weekly_multiasset.py — no network access
(yfinance monkeypatched), tiny hyperparameters. Checks: sweep-phase merging,
TSDiff-D is never fitted (include_d=False plumbed through), and the ZN/TLT
index-alignment guardrail before they're merged into one "Bonds" bucket.
"""

import json

import numpy as np
import pandas as pd
import pytest

import weekly_multiasset as wm
import tsdiff_model as td


def _synthetic_daily(n_days=900, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def patched_fetch_uniform(monkeypatch):
    """All tickers resolve to the SAME synthetic series -- fine for plumbing
    tests where we don't care about realism, only about wiring."""
    daily = _synthetic_daily()
    monkeypatch.setattr(td, "fetch_data", lambda ticker, start, end: daily)
    return daily


def test_run_sweep_phase_merges_into_existing_file(tmp_path, patched_fetch_uniform):
    sweep_file = tmp_path / "sweep.json"
    sweep_file.write_text(json.dumps({
        "config": {}, "meta": {}, "records": [
            {"asset": "SPY", "model": "TSDiff-W", "epochs": 80, "crps_val": 1.0,
             "cov95_val": 0.9, "rel_std_pct_val": 1.5, "n_val_origins": 12},
        ], "selected_epochs": {"SPY|TSDiff-W": {"epochs": 80, "crps_val": 1.0,
                                                "cov95_val": 0.9, "rel_std_pct_val": 1.5}},
    }))
    td.set_seed(0)
    records, selected, elapsed = wm.run_sweep_phase(
        ["ETH"], candidates=[2, 4], n_val=3, n_test=5, seed=0, n_samples=6,
        k_denoise=3, start="2021-01-01", end="2025-01-01", sweep_file=sweep_file)

    merged = json.loads(sweep_file.read_text())
    # old SPY record preserved, new ETH records added, both present in selection
    assert any(r["asset"] == "SPY" for r in merged["records"])
    assert any(r["asset"] == "ETH" for r in merged["records"])
    assert "SPY|TSDiff-W" in merged["selected_epochs"]
    assert "ETH|TSDiff-W" in merged["selected_epochs"]
    assert selected["ETH|TSDiff-W"]["epochs"] in (2, 4)
    # only TSDiff-W was swept (models=("TSDiff-W",) in run_sweep_phase)
    assert all(r["model"] == "TSDiff-W" for r in records)


def test_run_final_phase_skips_tsdiff_d_and_computes_panel_test(tmp_path, patched_fetch_uniform):
    sweep_file = tmp_path / "sweep.json"
    sweep_file.write_text(json.dumps({
        "selected_epochs": {f"{a}|TSDiff-W": {"epochs": 2} for a in wm.ALL_ASSETS},
    }))
    out_path = tmp_path / "final.json"
    td.set_seed(0)
    wm.run_final_phase(sweep_file, n_val=3, n_test=4, seed=0, n_samples=6, k_denoise=3,
                       start="2021-01-01", end="2025-01-01", out_path=out_path)

    payload = json.loads(out_path.read_text())
    assert set(payload["summary"].keys()) == set(wm.ALL_ASSETS)
    for asset, by_h in payload["summary"].items():
        for horizon, by_m in by_h.items():
            assert set(by_m.keys()) == {"TSDiff-W", "RandomWalk"}   # never TSDiff-D

    # raw per-origin records must be persisted too (needed for any downstream
    # consumer, e.g. a DB backfill) -- not just the aggregated summary/tests.
    assert "records" in payload
    assert len(payload["records"]) == 5 * 3 * 4 * 2   # assets x horizons x n_test x (W+RW)
    assert {r["asset"] for r in payload["records"]} == set(wm.ALL_ASSETS)

    assert set(payload["bucket_tests"].keys()) == {"SPY", "BTC", "ETH", "Bonds (ZN+TLT)"}
    assert payload["panel_test"]["k"] == 4
    assert "p_combined" in payload["panel_test"]


def test_run_final_phase_raises_on_missing_epoch_selection(tmp_path, patched_fetch_uniform):
    sweep_file = tmp_path / "sweep.json"
    sweep_file.write_text(json.dumps({"selected_epochs": {"SPY|TSDiff-W": {"epochs": 2}}}))
    with pytest.raises(SystemExit):
        wm.run_final_phase(sweep_file, n_val=3, n_test=4, seed=0, n_samples=6, k_denoise=3,
                           start="2021-01-01", end="2025-01-01", out_path=tmp_path / "out.json")


def test_run_final_phase_raises_on_zn_tlt_index_mismatch(tmp_path, monkeypatch):
    """ZN and TLT must share identical (origin, horizon) indices before being
    averaged into the 'Bonds' bucket -- feed them different-length series to
    force a mismatch and check the guardrail actually fires."""
    long_series = _synthetic_daily(n_days=900, seed=1)
    short_series = _synthetic_daily(n_days=850, seed=2)   # different end date -> different origins

    def fake_fetch(ticker, start, end):
        return short_series if ticker == "ZN=F" else long_series

    monkeypatch.setattr(td, "fetch_data", fake_fetch)
    sweep_file = tmp_path / "sweep.json"
    sweep_file.write_text(json.dumps({
        "selected_epochs": {f"{a}|TSDiff-W": {"epochs": 2} for a in wm.ALL_ASSETS},
    }))
    td.set_seed(0)
    with pytest.raises(SystemExit, match="ZN and TLT"):
        wm.run_final_phase(sweep_file, n_val=3, n_test=4, seed=0, n_samples=6, k_denoise=3,
                           start="2021-01-01", end="2025-01-01", out_path=tmp_path / "out.json")
