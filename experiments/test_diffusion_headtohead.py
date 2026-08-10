"""Correctness tests for diffusion_headtohead.py -- la machinerie partagee par
les deux matchs inter-modeles (vs ARIMA-GARCH, vs TSDiff). Les conventions de
SIGNE sont l'endroit ou un duel peut s'inverser silencieusement (toutes les
metriques sont a minimiser SAUF la couverture), donc chacune est testee sur un
cas ou le gagnant est connu par construction. Le moteur de generation
(`DiffusionEngine`) est verifie sur son identite, pas sur ses poids : son
equivalence bit-a-bit avec le chemin d'origine a ete controlee sur donnees
reelles, trop lentement pour la suite unitaire."""

import numpy as np
import pandas as pd
import pytest

import diffusion_headtohead as h2h


def _arm(sq_error, winkler, in_interval, y_true=100.0, asset="SPY",
         regime="weekly", horizon_unit="W+1", seed=42):
    n = len(sq_error)
    dates = pd.date_range("2025-01-03", periods=n, freq="7D").astype(str)
    return pd.DataFrame({
        "asset": asset, "frequence": regime, "horizon_unit": horizon_unit, "seed": seed,
        "cutoff_date": dates, "target_date": dates, "last_close": 100.0, "y_true": y_true,
        "sq_error": np.asarray(sq_error, dtype=float),
        "winkler": np.asarray(winkler, dtype=float),
        "in_interval": np.asarray(in_interval, dtype=float),
    })


# ── convention de signe : metriques a MINIMISER ──────────────────────────

def test_paired_verdict_declares_a_the_winner_when_its_metric_is_lower():
    rng = np.random.default_rng(0)
    a = rng.gamma(2.0, 1.0, 60)
    out = h2h.paired_verdict(a - (a + 3.0), "NsDiff", "TSDiff")   # A systematiquement plus bas
    assert out["verdict"] == "NsDiff_significantly_better"
    assert out["mean_diff"] < 0


def test_paired_verdict_declares_b_the_winner_when_its_metric_is_lower():
    rng = np.random.default_rng(1)
    a = rng.gamma(2.0, 1.0, 60)
    out = h2h.paired_verdict((a + 3.0) - a, "NsDiff", "TSDiff")
    assert out["verdict"] == "TSDiff_significantly_better"


def test_paired_verdict_is_indistinguishable_on_pure_noise():
    rng = np.random.default_rng(2)
    out = h2h.paired_verdict(rng.normal(0.0, 1.0, 60), "NsDiff", "TSDiff")
    assert out["verdict"] == "indistinguishable"


def test_paired_verdict_refuses_to_test_a_handful_of_points():
    assert h2h.paired_verdict([1.0, 2.0], "A", "B")["status"] == "insufficient_data"


# ── convention de signe INVERSE : la couverture se MAXIMISE ──────────────

def test_coverage_head_to_head_credits_the_arm_that_covers_more():
    """A couvre 100%, B couvre 60% -> c'est A qui doit etre credite, alors
    meme que le test interne partage la convention 'negatif favorise A'."""
    n = 60
    a = _arm(np.ones(n), np.ones(n), np.ones(n))
    b = _arm(np.ones(n), np.ones(n), [1.0] * 36 + [0.0] * 24)
    out = h2h.compare_cell(h2h.merge_pair(a, b, "SPY", "weekly", "W+1"), "NsDiff", "TSDiff")
    assert out["cov95_NsDiff"] == pytest.approx(1.0)
    assert out["cov95_TSDiff"] == pytest.approx(0.6)
    assert out["coverage_head_to_head"]["verdict"] == "NsDiff_covers_more"


def test_compare_cell_reports_both_arms_under_their_own_labels():
    n = 60
    a = _arm(np.full(n, 4.0), np.full(n, 10.0), np.ones(n))
    b = _arm(np.full(n, 9.0), np.full(n, 20.0), np.ones(n))
    out = h2h.compare_cell(h2h.merge_pair(a, b, "SPY", "weekly", "W+1"), "NsDiff", "TSDiff")
    assert out["rmse_NsDiff"] == pytest.approx(2.0)      # sqrt(4)
    assert out["rmse_TSDiff"] == pytest.approx(3.0)      # sqrt(9)
    assert out["winkler_NsDiff"] == pytest.approx(10.0)
    assert out["rmse_test"]["verdict"] == "NsDiff_significantly_better"
    assert out["winkler_test"]["verdict"] == "NsDiff_significantly_better"


def test_compare_cell_on_empty_overlap_is_not_an_error():
    assert h2h.compare_cell(pd.DataFrame(), "A", "B") == {"status": "no_overlap", "n": 0}


# ── appariement ──────────────────────────────────────────────────────────

def test_merge_pair_keeps_only_shared_origins_in_chronological_order():
    a = _arm(np.ones(10), np.ones(10), np.ones(10))
    b = _arm(np.ones(6), np.ones(6), np.ones(6))          # 6 premieres origines
    merged = h2h.merge_pair(a, b, "SPY", "weekly", "W+1")
    assert len(merged) == 6
    assert list(merged["cutoff_date_a"]) == sorted(merged["cutoff_date_a"])


def test_merge_pair_rejects_arms_that_do_not_share_the_ground_truth():
    a = _arm(np.ones(10), np.ones(10), np.ones(10), y_true=100.0)
    b = _arm(np.ones(10), np.ones(10), np.ones(10), y_true=101.0)
    with pytest.raises(SystemExit, match="y_true divergent"):
        h2h.merge_pair(a, b, "SPY", "weekly", "W+1")


def test_merge_pair_never_crosses_regimes():
    a = _arm(np.ones(10), np.ones(10), np.ones(10), regime="weekly")
    b = _arm(np.ones(10), np.ones(10), np.ones(10), regime="daily")
    assert h2h.merge_pair(a, b, "SPY", "weekly", "W+1").empty


# ── pooling des graines / modele deterministe ────────────────────────────

def test_seed_pooled_rows_averages_the_metric_and_keeps_the_origin_count():
    frames = [_arm(np.full(10, float(s)), np.full(10, float(s)), np.ones(10), seed=s)
              for s in (42, 43, 44)]
    pooled = h2h.seed_pooled_rows(pd.concat(frames, ignore_index=True))
    assert len(pooled) == 10                      # une ligne par ORIGINE, pas 30
    assert (pooled["seed"] == -1).all()
    assert pooled["sq_error"].iloc[0] == pytest.approx(43.0)   # moyenne de 42/43/44


def test_broadcast_seeds_replicates_a_seedless_model_identically():
    rows = _arm(np.ones(10), np.ones(10), np.ones(10)).drop(columns=["seed"])
    out = h2h.broadcast_seeds(rows, [42, 43])
    assert sorted(out["seed"].unique()) == [-1, 42, 43]
    assert len(out) == 30
    for s in (-1, 42, 43):
        assert out[out["seed"] == s]["sq_error"].tolist() == [1.0] * 10


# ── moteurs de generation ────────────────────────────────────────────────

def test_engines_are_registered_under_their_own_model_name():
    from diffusion_multiseed_v2 import ENGINES
    for name, factory in ENGINES.items():
        engine = factory()
        assert engine.name == name              # le champ `model` des lignes produites
        assert hasattr(engine.module, "set_seed")
        assert hasattr(engine.module, "forecast_from_fitted")
        assert hasattr(engine.module, "_log_returns")


def test_tsdiff_epoch_budgets_are_read_from_the_repo_never_invented():
    from diffusion_multiseed_v2 import load_tsdiff_epochs, epochs_for, NSDIFF_EPOCHS_W
    budgets = load_tsdiff_epochs()
    assert budgets, "aucun budget TSDiff relu"
    assert set(budgets.values()) <= {40, 60, 80}
    assert epochs_for("NsDiff", "SPY", 42, budgets) == NSDIFF_EPOCHS_W
    assert epochs_for("TSDiff", "SPY", 42, budgets) == budgets[("SPY", 42)]
    with pytest.raises(SystemExit, match="refus d'en inventer"):
        epochs_for("TSDiff", "SPY", 999, budgets)
