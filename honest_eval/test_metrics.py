"""
Tests de honest_eval/metrics.py — chaque fonction est vérifiée contre soit une valeur
calculée à la main, soit une implémentation de référence indépendante déjà présente
dans les dépendances du projet (statsmodels pour l'IC de Wilson), pour ne pas se
contenter de re-vérifier sa propre formule avec elle-même.
"""

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from honest_eval import metrics


# ── MASE / Theil's U ──────────────────────────────────────────────────────────
def test_mase_basic_ratio():
    assert metrics.mase(mae_model=2.0, mae_naive=4.0) == pytest.approx(0.5)


def test_mase_naive_zero_returns_nan():
    assert np.isnan(metrics.mase(mae_model=2.0, mae_naive=0.0))


def test_theils_u_basic_ratio():
    assert metrics.theils_u(rmse_model=3.0, rmse_naive=6.0) == pytest.approx(0.5)


def test_theils_u_equal_to_naive_is_one():
    assert metrics.theils_u(rmse_model=5.0, rmse_naive=5.0) == pytest.approx(1.0)


# ── Corrélation des variations ───────────────────────────────────────────────
def test_variation_correlation_perfect_positive():
    delta_pred = [1, 2, 3, 4, 5]
    delta_real = [2, 4, 6, 8, 10]
    assert metrics.variation_correlation(delta_pred, delta_real) == pytest.approx(1.0)


def test_variation_correlation_perfect_negative():
    delta_pred = [1, 2, 3, 4, 5]
    delta_real = [-1, -2, -3, -4, -5]
    assert metrics.variation_correlation(delta_pred, delta_real) == pytest.approx(-1.0)


def test_variation_correlation_no_variance_is_nan():
    """Un modèle naïf comparé à lui-même a delta_pred constant (= 0) -- corrélation
    non définie, pas 0 ni une erreur."""
    delta_pred = [0, 0, 0, 0]
    delta_real = [1, -2, 3, -1]
    assert np.isnan(metrics.variation_correlation(delta_pred, delta_real))


# ── Directional accuracy + IC Wilson ─────────────────────────────────────────
def test_directional_accuracy_matches_statsmodels_wilson_ci():
    """Vérifie l'IC de Wilson contre statsmodels.stats.proportion.proportion_confint
    (méthode 'wilson'), une implémentation indépendante déjà utilisée ailleurs dans
    le projet, plutôt que de re-dériver la même formule dans le test."""
    rng = np.random.default_rng(0)
    delta_pred = rng.normal(size=37)
    delta_real = rng.normal(size=37)
    correct = int(np.sum(np.sign(delta_pred) == np.sign(delta_real)))
    n = len(delta_pred)

    result = metrics.directional_accuracy_wilson(delta_pred, delta_real)
    ref_low, ref_high = proportion_confint(correct, n, alpha=0.05, method="wilson")

    # result[...] est arrondi à 2 décimales (contrat de l'API) -- tolérance cohérente
    # avec cet arrondi, pas avec la précision brute de la formule.
    assert result["accuracy"] == pytest.approx(100 * correct / n, abs=5e-3)
    assert result["ci_low"] == pytest.approx(100 * ref_low, abs=5e-3)
    assert result["ci_high"] == pytest.approx(100 * ref_high, abs=5e-3)


def test_directional_accuracy_wilson_narrower_than_naive_wald_on_small_n():
    """Sur un petit n avec un succès partiel (pas un cas dégénéré 0% ou 100%, où
    Wilson et Wald coïncident tous les deux exactement), l'IC de Wilson doit différer
    de l'IC de Wald naïf (p_hat +/- z*sqrt(p_hat*(1-p_hat)/n)) -- c'est précisément
    pour ce genre de n que le brief demande Wilson plutôt que Wald."""
    delta_real = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    delta_pred = [1, -1, 1, -1, 1, -1, 1, 1, -1, -1]   # 8/10 correct
    result = metrics.directional_accuracy_wilson(delta_pred, delta_real)

    n, correct = 10, 8
    p_hat = correct / n
    wald_half_width = metrics.Z_95 * np.sqrt(p_hat * (1 - p_hat) / n)
    wald_high = (p_hat + wald_half_width) * 100

    assert result["accuracy"] == pytest.approx(80.0)
    assert result["ci_high"] != pytest.approx(wald_high, abs=1e-6)


def test_directional_accuracy_empty_input():
    result = metrics.directional_accuracy_wilson([], [])
    assert result["n"] == 0
    assert result["accuracy"] is None


# ── Diebold-Mariano ───────────────────────────────────────────────────────────
def test_diebold_mariano_model_strictly_better_rejects_h0():
    """Le modèle a des erreurs quasi nulles, le naïf a des erreurs larges et
    constantes -- le test doit rejeter H0 et conclure 'beats_naive'."""
    rng = np.random.default_rng(1)
    errors_model = rng.normal(0, 0.01, size=60)
    errors_naive = rng.normal(0, 5.0, size=60)

    result = metrics.diebold_mariano(errors_model, errors_naive)

    assert result["verdict"] == "beats_naive"
    assert result["p_value"] < 0.05
    assert result["dm_stat"] > 0


def test_diebold_mariano_identical_predictions():
    """Modèle == naïf (même série d'erreurs) -- variance nulle, pas de division par
    zéro, verdict explicite plutôt qu'un NaN silencieux."""
    errors = [1.0, -2.0, 3.0, -1.5, 0.5]
    result = metrics.diebold_mariano(errors, errors)
    assert result["verdict"] == "identical_predictions"
    assert result["p_value"] is None


def test_diebold_mariano_no_significant_difference():
    """Erreurs de même distribution pour modèle et naïf (même seed) -- p-value élevée,
    verdict 'no_better_than_naive'."""
    rng = np.random.default_rng(2)
    errors_model = rng.normal(0, 1.0, size=50)
    errors_naive = rng.normal(0, 1.0, size=50)
    result = metrics.diebold_mariano(errors_model, errors_naive)
    assert result["verdict"] == "no_better_than_naive"
    assert result["p_value"] >= 0.05


def test_diebold_mariano_insufficient_data():
    result = metrics.diebold_mariano([1.0], [2.0])
    assert result["verdict"] == "insufficient_data"


def test_newey_west_variance_exceeds_iid_for_positively_autocorrelated_series():
    """Teste directement l'estimateur HAC de Newey-West (pas le test DM en entier, où
    le carré des erreurs peut brouiller le signal d'autocorrélation) : une série
    fortement autocorrélée positivement (moyenne mobile) doit avoir une variance
    longue-portée strictement supérieure à l'estimateur i.i.d. (gamma_0 seul) --
    exactement le problème que Newey-West est censé corriger (Point 3 du brief,
    rolling origin chevauchant)."""
    rng = np.random.default_rng(5)
    white_noise = rng.normal(0, 1.0, size=200)
    autocorrelated = np.convolve(white_noise, np.ones(5) / 5, mode="same")

    iid_var = metrics._newey_west_long_run_variance(autocorrelated, lag_truncation=0)
    hac_var = metrics._newey_west_long_run_variance(autocorrelated, lag_truncation=10)

    assert hac_var > iid_var


def test_newey_west_variance_close_to_iid_for_white_noise():
    """Sur du vrai bruit blanc (pas d'autocorrélation), l'estimateur HAC ne doit pas
    dériver arbitrairement loin de l'estimateur i.i.d. (les gamma_k>0 doivent être
    proches de 0 en espérance)."""
    rng = np.random.default_rng(6)
    white_noise = rng.normal(0, 1.0, size=500)

    iid_var = metrics._newey_west_long_run_variance(white_noise, lag_truncation=0)
    hac_var = metrics._newey_west_long_run_variance(white_noise, lag_truncation=10)

    assert hac_var == pytest.approx(iid_var, rel=0.3)


# ── variation_metrics (intégration) ──────────────────────────────────────────
def test_variation_metrics_naive_vs_itself_is_neutral():
    """Le naïf comparé à lui-même : U=1, MASE=1, DM='identical_predictions'."""
    actual = [100.0, 102.0, 101.0, 105.0]
    naive_pred = [99.0, 100.0, 102.0, 101.0]   # persistance : ancre = valeur précédente

    result = metrics.variation_metrics(
        actual=actual, predicted=naive_pred, naive_predicted=naive_pred,
        mae_model=2.0, mae_naive=2.0, rmse_model=2.5, rmse_naive=2.5,
    )

    assert result["theils_u"] == pytest.approx(1.0)
    assert result["mase"] == pytest.approx(1.0)
    assert result["diebold_mariano"]["verdict"] == "identical_predictions"
    assert result["no_better_than_naive"] is False  # p_value indisponible, pas "no better"


def test_variation_metrics_perfect_model_beats_naive():
    """Un modèle qui prédit exactement le futur bat significativement le naïf."""
    rng = np.random.default_rng(4)
    naive_pred = 100 + np.cumsum(rng.normal(0, 1, 60))
    actual = naive_pred + rng.normal(0, 3, 60)   # le naïf a une vraie erreur
    predicted = actual.copy()                     # le modèle prédit exactement

    result = metrics.variation_metrics(
        actual=actual, predicted=predicted, naive_predicted=naive_pred,
        mae_model=0.0, mae_naive=float(np.mean(np.abs(actual - naive_pred))),
        rmse_model=0.0, rmse_naive=float(np.sqrt(np.mean((actual - naive_pred) ** 2))),
    )

    assert result["theils_u"] == pytest.approx(0.0)
    assert result["mase"] == pytest.approx(0.0)
    assert result["diebold_mariano"]["verdict"] == "beats_naive"
    assert result["directional_accuracy_variations"]["accuracy"] == 100.0
