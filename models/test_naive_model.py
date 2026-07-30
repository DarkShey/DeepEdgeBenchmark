"""
Tests dédiés naive_model.py — Point 0 de IMPROVEMENTS_BRIEF.md.

La baseline persistence doit prédire EXACTEMENT la clôture précédente : c'est le
critère d'acceptation du brief, audité aussi par honest_eval.naive.verify_naive().
Zéro réseau (données synthétiques de conftest.py). Collectés par Gate 1 du
pipeline model_artifacts via MODEL_TEST_FILTER["Naive"] = "naive_model".
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_train_test, synthetic_series

import naive_model


def test_run_naive_predicts_previous_close_exactly():
    train, test = make_train_test()

    result = naive_model.run_naive(train, test)

    expected = np.concatenate([[float(train.iloc[-1])],
                               test.values[:-1].astype(float)])
    np.testing.assert_array_equal(np.asarray(result["predictions"], dtype=float),
                                  expected)


def test_run_naive_is_deterministic_across_seeds():
    """Point 0 : aucune perturbation aléatoire — deux runs avec des seeds
    différents doivent produire des prédictions identiques."""
    train, test = make_train_test()

    naive_model.set_seed(1)
    r1 = naive_model.run_naive(train, test)
    naive_model.set_seed(999)
    r2 = naive_model.run_naive(train, test)

    np.testing.assert_array_equal(r1["predictions"], r2["predictions"])
    np.testing.assert_array_equal(r1["lower"], r2["lower"])
    np.testing.assert_array_equal(r1["upper"], r2["upper"])


def test_run_naive_interval_is_gaussian_rw_band():
    train, test = make_train_test()

    result = naive_model.run_naive(train, test)

    sigma = float(np.std(np.diff(train.values.astype(float))))
    prev = np.asarray(result["predictions"], dtype=float)
    np.testing.assert_allclose(result["lower"], prev - naive_model.Z_95 * sigma)
    np.testing.assert_allclose(result["upper"], prev + naive_model.Z_95 * sigma)
    assert sigma > 0


def test_run_naive_n_ensemble_zero_omits_ensemble_key():
    """Non-régression : n_ensemble=0 (défaut) reste le comportement actuel exact pour
    tous les appelants existants (CLI standalone, pipeline avant ce changement)."""
    train, test = make_train_test()

    result = naive_model.run_naive(train, test)

    assert "ensemble" not in result


def test_run_naive_n_ensemble_populates_gaussian_rw_clouds():
    """n_ensemble>0 matérialise la bande gaussienne déjà utilisée pour l'IC95
    (prev ± 1.96σ) en nuage de tirages par pas, pour le CRPS empirique (cf.
    model_artifacts/crps_kpis.py) -- pas une nouvelle hypothèse de distribution."""
    train, test = make_train_test()
    n_ensemble = 500

    result = naive_model.run_naive(train, test, n_ensemble=n_ensemble, ensemble_seed=0)

    assert "ensemble" in result
    assert len(result["ensemble"]) == len(test)
    prev = np.asarray(result["predictions"], dtype=float)
    sigma = float(np.std(np.diff(train.values.astype(float))))
    for t, cloud in enumerate(result["ensemble"]):
        cloud = np.asarray(cloud, dtype=float)
        assert cloud.shape == (n_ensemble,)
        assert np.all(np.isfinite(cloud))
        assert cloud.mean() == pytest.approx(prev[t], abs=5 * sigma / np.sqrt(n_ensemble))


def test_next_step_naive_returns_last_price():
    series = synthetic_series(n=120)

    pred, lo, hi = next_step = naive_model.next_step_naive(series)

    assert pred == float(series.iloc[-1])
    assert lo < pred < hi


def test_run_naive_passes_verify_naive_audit():
    """Croise avec l'auditeur du brief quand honest_eval est disponible."""
    try:
        import os, sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from honest_eval.naive import verify_naive
    except ImportError:
        import pytest
        pytest.skip("honest_eval indisponible")

    train, test = make_train_test()
    result = naive_model.run_naive(train, test)

    report = verify_naive(train, test,
                          dashboard_predictions=result["predictions"],
                          dashboard_rmse=result["RMSE"])
    assert report["passed"], report["issues"]
