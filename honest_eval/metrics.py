"""
honest_eval/metrics.py — Métriques "honnêtes" pour comparer un modèle à la baseline
naïve (persistance), plutôt que de scorer des niveaux de prix bruts.
=====================================================================================
Cf. Point 1 du brief d'amélioration (IMPROVEMENTS_BRIEF, 2026-07-08) : un modèle qui
"prédit le prix d'hier +/- epsilon" affiche un RMSE en niveaux flatteur
(corr(pred_t, prix_{t-1}) proche de 1) sans avoir aucun pouvoir prédictif réel. Ces
métriques comparent explicitement chaque modèle à la baseline naïve (models/
naive_model.py, persistance stricte depuis le Point 0) sur les VARIATIONS de prix,
jamais les niveaux.

Convention partagée par toutes les fonctions de ce module : `naive_predicted` est déjà
le "prix de référence" (dernier prix connu au moment de la prévision) pour chaque
point, puisque naive_model prédit exactement ce prix (persistance stricte, Point 0).
  delta_reel_t  = actual_t - naive_predicted_t
  delta_pred_t  = predicted_t - naive_predicted_t
Un modèle sans aucun pouvoir prédictif a delta_pred proche de 0 (comme le naïf lui-
même) et une corrélation de variations proche de 0, même si sa corrélation en niveaux
est proche de 1.
"""

import numpy as np
from scipy import stats

Z_95 = 1.959963984540054   # scipy.stats.norm.ppf(0.975)


def mase(mae_model: float, mae_naive: float) -> float:
    """Mean Absolute Scaled Error = MAE(modèle) / MAE(naïf). < 1 : bat le naïf en MAE."""
    if mae_naive is None or mae_naive == 0 or mae_model is None:
        return float("nan")
    return float(mae_model / mae_naive)


def theils_u(rmse_model: float, rmse_naive: float) -> float:
    """Theil's U = RMSE(modèle) / RMSE(naïf). < 1 : bat le naïf, ~1 : aucun skill vs
    naïf, > 1 : pire que le naïf."""
    if rmse_naive is None or rmse_naive == 0 or rmse_model is None:
        return float("nan")
    return float(rmse_model / rmse_naive)


def variation_correlation(delta_pred, delta_real) -> float:
    """Corrélation de Pearson entre variations prédites et réelles (pas les niveaux)
    -- proche de 0 si le modèle n'a aucun pouvoir prédictif sur le sens/l'ampleur du
    mouvement, même si sa corrélation en niveaux est ~1 (cf. diagnostic du brief)."""
    delta_pred = np.asarray(delta_pred, dtype=float)
    delta_real = np.asarray(delta_real, dtype=float)
    if len(delta_pred) < 2 or np.std(delta_pred) == 0 or np.std(delta_real) == 0:
        return float("nan")
    return float(np.corrcoef(delta_pred, delta_real)[0, 1])


def directional_accuracy_wilson(delta_pred, delta_real) -> dict:
    """Précision directionnelle (sign(delta_pred) == sign(delta_real)) avec intervalle
    de confiance binomial de Wilson à 95% -- plus fiable que Wald sur les petits n
    (ex. n_val=10 des runs D+7 non-denses actuels, cf. Point 3 pour le rolling dense)."""
    delta_pred = np.asarray(delta_pred, dtype=float)
    delta_real = np.asarray(delta_real, dtype=float)
    n = len(delta_pred)
    if n == 0:
        return {"accuracy": None, "ci_low": None, "ci_high": None, "n": 0}

    correct = int(np.sum(np.sign(delta_pred) == np.sign(delta_real)))
    p_hat = correct / n
    z = Z_95
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return {
        "accuracy": round(p_hat * 100, 2),
        "ci_low": round(max(0.0, (center - margin) * 100), 2),
        "ci_high": round(min(100.0, (center + margin) * 100), 2),
        "n": n,
    }


def diebold_mariano(errors_model, errors_naive, power: int = 2, lag_truncation: int = 0) -> dict:
    """Test de Diebold-Mariano (modèle vs naïf) sur la perte quadratique, avec
    correction petit échantillon de Harvey-Leybourne-Newbold (HLN 1997) et p-value
    via la loi de Student(n-1) plutôt que la loi normale asymptotique (trop optimiste
    aux tailles d'échantillon rencontrées ici, n <= 165).

    d_t = |e_naif_t|^power - |e_modele_t|^power : d_t > 0 => le modèle fait mieux ce
    jour-là. H0 : E[d_t] = 0 (aucune différence de skill contre le naïf).

    lag_truncation=0 (défaut) : variance de dbar estimée i.i.d. (var(d)/n), valable
    pour des erreurs 1-step ou des origines D+h suffisamment espacées pour ne pas se
    chevaucher. lag_truncation>0 : variance HAC de Newey-West (troncature au lag
    donné) -- nécessaire dès que les erreurs sont autocorrélées par construction,
    typiquement le rolling origin quotidien chevauchant du Point 3.
    """
    e_model = np.asarray(errors_model, dtype=float)
    e_naive = np.asarray(errors_naive, dtype=float)
    n = len(e_model)
    if n < 2:
        return {"dm_stat": None, "p_value": None, "n": n, "verdict": "insufficient_data"}

    d = np.abs(e_naive) ** power - np.abs(e_model) ** power
    dbar = float(np.mean(d))

    if lag_truncation <= 0:
        var_dbar = float(np.var(d, ddof=1)) / n
    else:
        var_dbar = _newey_west_long_run_variance(d, lag_truncation) / n

    if var_dbar <= 0 or not np.isfinite(var_dbar):
        return {"dm_stat": None, "p_value": None, "n": n, "verdict": "identical_predictions"}

    dm_stat = dbar / np.sqrt(var_dbar)
    hln_factor = np.sqrt((n - 1) / n)   # h=1 : (n+1-2h+h(h-1)/n)/n = (n-1)/n
    dm_hln = dm_stat * hln_factor
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1)))

    if p_value < 0.05:
        verdict = "beats_naive" if dbar > 0 else "worse_than_naive"
    else:
        verdict = "no_better_than_naive"

    return {"dm_stat": round(float(dm_hln), 4), "p_value": round(p_value, 4),
            "n": n, "verdict": verdict}


def _newey_west_long_run_variance(d, lag_truncation: int) -> float:
    """Variance longue-portée de Newey & West (1987) pour une série d (les d_t du test
    DM) : var(dbar) = (gamma_0 + 2*sum_{k=1..L} w_k*gamma_k) / n, poids de Bartlett
    w_k = 1 - k/(L+1), L = lag_truncation."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    d_centered = d - np.mean(d)
    gamma_0 = float(np.sum(d_centered ** 2)) / n
    total = gamma_0
    for k in range(1, min(lag_truncation, n - 1) + 1):
        gamma_k = float(np.sum(d_centered[k:] * d_centered[:-k])) / n
        weight = 1 - k / (lag_truncation + 1)
        total += 2 * weight * gamma_k
    return total


def variation_metrics(actual, predicted, naive_predicted, mae_model, mae_naive,
                      rmse_model, rmse_naive, lag_truncation: int = 0) -> dict:
    """Regroupe toutes les métriques Point 1 pour un (modèle, actif, horizon) donné.
    `naive_predicted` sert d'ancre (prix de référence) pour delta_pred/delta_real,
    cf. docstring de module. `lag_truncation` : cf. diebold_mariano (0 par défaut,
    >0 pour du rolling origin chevauchant, Point 3)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    naive_predicted = np.asarray(naive_predicted, dtype=float)

    delta_real = actual - naive_predicted
    delta_pred = predicted - naive_predicted
    errors_model = actual - predicted
    errors_naive = actual - naive_predicted

    dm = diebold_mariano(errors_model, errors_naive, lag_truncation=lag_truncation)
    dir_acc = directional_accuracy_wilson(delta_pred, delta_real)
    corr = variation_correlation(delta_pred, delta_real)
    u = theils_u(rmse_model, rmse_naive)
    m = mase(mae_model, mae_naive)

    p_value = dm.get("p_value")
    no_better_than_naive = bool(
        u is not None and not np.isnan(u) and abs(u - 1.0) < 0.05
        and p_value is not None and p_value >= 0.05
    )

    return {
        "mase": round(m, 4) if np.isfinite(m) else None,
        "theils_u": round(u, 4) if np.isfinite(u) else None,
        "variation_correlation": round(corr, 4) if np.isfinite(corr) else None,
        "directional_accuracy_variations": dir_acc,
        "diebold_mariano": dm,
        "no_better_than_naive": no_better_than_naive,
    }
