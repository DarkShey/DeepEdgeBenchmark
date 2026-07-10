"""
Tests dédiés à naive_model.py — verrouillent le critère d'acceptation du Point 0 du
brief d'amélioration (2026-07-08) : la baseline Naive doit être une persistance
stricte (pred_t = actual_{t-1} exactement, aucun tirage aléatoire) avec un PI 95%
gaussien (+/- 1.96 * sigma des variations du train), pas le +/-5% uniforme d'origine.

Zéro réseau : fetch_data n'est jamais appelé, les données viennent de synthetic_series()
(cf. conftest.py, partagé avec test_models_common.py).
"""

import numpy as np
import pytest

from conftest import make_train_test, synthetic_series

import naive_model


def test_predictions_equal_previous_actual_exactly():
    """Persistance stricte : pred_t == actual_{t-1}, au bit près (pas de +/-5% de bruit)."""
    train, test = make_train_test()

    result = naive_model.run_naive(train, test)

    expected = np.concatenate([[train.iloc[-1]], test.values[:-1].astype(float)])
    assert np.array_equal(result["predictions"], expected)


def test_pi_matches_gaussian_1_96_sigma_of_train_diffs():
    """Le PI 95% doit être pred +/- Z_95 * sigma, sigma = std(diff(train)) — pas un
    pourcentage fixe du prix (l'ancien bug : prev_price * [0.95, 1.05])."""
    train, test = make_train_test()

    result = naive_model.run_naive(train, test)

    sigma = np.std(np.diff(train.values.astype(float)), ddof=1)
    expected_half_width = naive_model.Z_95 * sigma
    actual_half_widths = (result["upper"] - result["lower"]) / 2.0

    assert actual_half_widths == pytest.approx(expected_half_width, rel=1e-9)
    # Le PI est centré sur la prédiction (pas sur le prix précédent brut avant PI).
    assert result["upper"] - result["predictions"] == pytest.approx(expected_half_width, rel=1e-9)
    assert result["predictions"] - result["lower"] == pytest.approx(expected_half_width, rel=1e-9)


def test_next_step_matches_run_naive_convention():
    """next_step_naive doit suivre exactement la même formule que run_naive : point =
    dernier prix, PI = point +/- Z_95 * sigma(série)."""
    series = synthetic_series(n=120)

    pred, lo, hi = naive_model.next_step_naive(series)

    assert pred == pytest.approx(float(series.iloc[-1]))
    sigma = np.std(np.diff(series.values.astype(float)), ddof=1)
    half = naive_model.Z_95 * sigma
    assert lo == pytest.approx(pred - half)
    assert hi == pytest.approx(pred + half)


def test_walk_forward_uses_realised_prices_never_own_predictions():
    """Tronquer la fenêtre de test et re-prédire : le préfixe partagé des prédictions
    doit être identique au bit près (déterministe maintenant, donc égalité stricte
    possible) — ne tiendrait pas si run_naive regardait au-delà du point courant ou
    dépendait d'un état aléatoire non reproductible."""
    train, test = make_train_test()

    full_result = naive_model.run_naive(train, test)
    truncated_result = naive_model.run_naive(train, test.iloc[:3])

    assert np.array_equal(full_result["predictions"][:3], truncated_result["predictions"])


def test_run_naive_returns_expected_contract():
    """Même contrat de sortie que les 4 autres forecasters (cf. test_models_common.py) :
    predictions/lower/upper/index/actual + métriques aplaties, toutes finies."""
    train, test = make_train_test()

    result = naive_model.run_naive(train, test)

    required_keys = ("predictions", "lower", "upper", "index", "actual",
                     "RMSE", "MAE", "Dir. Acc (%)", "PI Cov 95% (%)")
    for key in required_keys:
        assert key in result, f"missing key {key!r} in run_naive result"
    assert len(result["predictions"]) == len(test)
    assert np.all(np.isfinite(np.asarray(result["predictions"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["lower"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["upper"], dtype=float)))
