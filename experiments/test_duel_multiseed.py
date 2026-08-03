"""Correctness tests for duel_multiseed.py's aggregation functions (pure,
synthetic per-seed inputs — no heavy backtest run) and the checkpoint
load/skip contract (BRIEF_multigraines.md §5, "run reprenable")."""

import json

import pandas as pd
import pytest

import duel_multiseed as dms


# ── aggregate_crps_dispersion ───────────────────────────────────────────────

def test_aggregate_crps_dispersion_mean_and_std():
    per_seed_df = {
        1: pd.DataFrame([
            {"asset": "SPY", "horizon": "W1", "model": "TSDiff", "crps": 1.0},
            {"asset": "SPY", "horizon": "W1", "model": "TSDiff", "crps": 3.0},
        ]),
        2: pd.DataFrame([
            {"asset": "SPY", "horizon": "W1", "model": "TSDiff", "crps": 5.0},
        ]),
    }
    out = {(r["asset"], r["horizon"], r["model"]): r for r in dms.aggregate_crps_dispersion(per_seed_df)}
    row = out[("SPY", "W1", "TSDiff")]
    # seed 1's per-seed mean crps = 2.0, seed 2's = 5.0 -> mean=3.5, n_seeds=2
    assert row["crps_mean"] == pytest.approx(3.5)
    assert row["n_seeds"] == 2
    assert row["crps_std"] == pytest.approx(pd.Series([2.0, 5.0]).std())


# ── aggregate_mcs_stability ──────────────────────────────────────────────────

def _fake_analysis(mcs=None, holm=None, pooled=None, spa=None):
    return {
        "model_confidence_set": mcs or {},
        "pairwise_vs_diffusion": holm or {},
        "pooled_pair_verdict_by_horizon": pooled or {},
        "spa_vs_garch": spa or {},
    }


def test_aggregate_mcs_stability_fractions():
    per_seed_analysis = {
        1: _fake_analysis(mcs={"SPY|W1": {"mcs": ["TSDiff", "Naive"]}}),
        2: _fake_analysis(mcs={"SPY|W1": {"mcs": ["TSDiff"]}}),
        3: _fake_analysis(mcs={"SPY|W1": {"mcs": ["Naive"]}}),
    }
    out = dms.aggregate_mcs_stability(per_seed_analysis)
    assert out["SPY|W1"]["TSDiff"] == pytest.approx(2 / 3)
    assert out["SPY|W1"]["Naive"] == pytest.approx(2 / 3)
    assert out["SPY|W1"]["Prophet"] == 0.0


# ── aggregate_holm_stability ─────────────────────────────────────────────────

def test_aggregate_holm_stability_stable_and_flipping_cells():
    key_stable = "SPY|W1|TSDiff vs Prophet"
    key_flip = "SPY|W1|TSDiff vs ARIMA-GARCH"
    per_seed_analysis = {
        1: _fake_analysis(holm={key_stable: {"significant_after_holm": True},
                                key_flip: {"significant_after_holm": True}}),
        2: _fake_analysis(holm={key_stable: {"significant_after_holm": True},
                                key_flip: {"significant_after_holm": False}}),
    }
    out = dms.aggregate_holm_stability(per_seed_analysis)
    assert out[key_stable]["fraction_significant"] == 1.0
    assert out[key_stable]["stable"] is True
    assert out[key_flip]["fraction_significant"] == 0.5
    assert out[key_flip]["stable"] is False
    assert out[key_flip]["seeds_significant"] == [1]


# ── aggregate_pooled_stability ───────────────────────────────────────────────

def test_aggregate_pooled_stability():
    per_seed_analysis = {
        1: _fake_analysis(pooled={"TSDiff vs Prophet": {"W1": {"significant_bootstrap": True}}}),
        2: _fake_analysis(pooled={"TSDiff vs Prophet": {"W1": {"significant_bootstrap": True}}}),
    }
    out = dms.aggregate_pooled_stability(per_seed_analysis)
    assert out["TSDiff vs Prophet"]["W1"]["fraction_significant"] == 1.0
    assert out["TSDiff vs Prophet"]["W1"]["stable"] is True


# ── aggregate_spa_stability ──────────────────────────────────────────────────

def test_aggregate_spa_stability():
    per_seed_analysis = {
        1: _fake_analysis(spa={"SPY|W1": {"reject_no_model_beats_benchmark": False}}),
        2: _fake_analysis(spa={"SPY|W1": {"reject_no_model_beats_benchmark": False}}),
        3: _fake_analysis(spa={"SPY|W1": {"reject_no_model_beats_benchmark": True}}),
    }
    out = dms.aggregate_spa_stability(per_seed_analysis)
    assert out["SPY|W1"]["fraction_reject"] == pytest.approx(1 / 3)
    assert out["SPY|W1"]["stable"] is False
    assert out["SPY|W1"]["seeds_reject"] == [3]


# ── checkpointing (brief §5, "run reprenable") ───────────────────────────────

def test_load_or_run_asset_skips_recomputation_when_checkpoint_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(dms, "CHECKPOINT_DIR", tmp_path)
    ckpt = dms.checkpoint_path(7, "SPY")
    ckpt.write_text(json.dumps({"records": [{"crps": 1.23}], "meta": {"epochs_tsdiff_w": 60}}))

    def _boom(*a, **kw):
        raise AssertionError("run_asset_duel must NOT be called when a checkpoint exists")

    monkeypatch.setattr(dms.db, "run_asset_duel", _boom)
    records, meta = dms.load_or_run_asset(7, "SPY", "SPY", args=None)
    assert records == [{"crps": 1.23}]
    assert meta == {"epochs_tsdiff_w": 60}


def test_load_or_run_asset_computes_and_saves_checkpoint_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(dms, "CHECKPOINT_DIR", tmp_path)
    calls = []

    def _fake_run(asset_code, ticker, args):
        calls.append((asset_code, ticker))
        return [{"crps": 4.56}], {"epochs_tsdiff_w": 40}

    monkeypatch.setattr(dms.db, "run_asset_duel", _fake_run)
    records, meta = dms.load_or_run_asset(9, "BTC", "BTC-USD", args=None)
    assert calls == [("BTC", "BTC-USD")]
    assert records == [{"crps": 4.56}]
    ckpt = dms.checkpoint_path(9, "BTC")
    assert ckpt.exists()
    saved = json.loads(ckpt.read_text())
    assert saved["records"] == [{"crps": 4.56}]

    # second call must now load from the checkpoint, not recompute
    calls.clear()
    records2, meta2 = dms.load_or_run_asset(9, "BTC", "BTC-USD", args=None)
    assert calls == []
    assert records2 == records


# ── NsDiff epoch sweep (BRIEF_nsdiff_epoch_sweep.md) — additive, opt-in ──────

def test_build_arg_parser_nsdiff_defaults():
    """New flags exist with the documented defaults; --include-nsdiff itself
    defaults to False (unchanged)."""
    args = dms.build_arg_parser().parse_args([])
    assert args.include_nsdiff is False
    assert args.nsdiff_fixed_epochs is None
    assert args.nsdiff_epoch_candidates == [40, 60, 80]
    assert args.nsdiff_hp_samples == 100


def test_run_one_seed_without_include_nsdiff_never_touches_nsdiff_path(monkeypatch):
    """The new NsDiff-sweep flags/plumbing must have zero effect on a run
    without --include-nsdiff (brief §5's second required test) -- proven by
    making the NsDiff code path raise if it is EVER reached, with `args`
    carrying no `include_nsdiff` attribute at all (the getattr(..., False)
    fallback exercised by a bare args namespace, same as argparse's own
    store_true default of False)."""
    monkeypatch.setattr(dms, "load_or_run_asset",
                        lambda seed, asset_code, ticker, args: ([{"crps": 1.0}], {"m": 1}))

    def _boom_nsdiff(*a, **kw):
        raise AssertionError("load_or_run_asset_nsdiff must NOT be called without --include-nsdiff")

    def _boom_analysis(*a, **kw):
        raise AssertionError("build_grid_analysis_with_nsdiff must NOT be used without --include-nsdiff")

    monkeypatch.setattr(dms, "load_or_run_asset_nsdiff", _boom_nsdiff)
    monkeypatch.setattr(dms.db, "build_grid_analysis", lambda df, meta, args: {"ok": True})
    monkeypatch.setattr(dms.db, "build_grid_analysis_with_nsdiff", _boom_analysis)

    class Args:
        assets = ["SPY"]

    args = Args()   # deliberately no `include_nsdiff` attribute at all
    df, all_meta, analysis = dms.run_one_seed(1, args)
    assert analysis == {"ok": True}
    assert all_meta == {"SPY": {"m": 1}}
