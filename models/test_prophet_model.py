"""
test_prophet_model.py — BRIEF_correction_prophet.md : run_prophet doit être
walk-forward 1-pas (miroir de run_sarima), pas un fit unique + predict batch.
Zéro réseau (données synthétiques déterministes, cf. conftest.py).

Le test de non-régression de dérive compare le nouveau run_prophet (walk-forward) à
une réimplémentation locale de l'ANCIEN comportement (_legacy_batch_run_prophet,
fit unique + predict batch) -- l'ancien code n'existe plus dans prophet_model.py,
il est reconstitué ici uniquement pour prouver que le correctif élimine bien le
biais qu'il produisait (BRIEF_correction_prophet.md §0/§4).
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_train_test, reset_seeds

import prophet_model as pm


def _disable_yearly_seasonality(monkeypatch):
    """Force yearly_seasonality=False sur les modèles construits par prophet_model
    (train/test) et par `_legacy_batch_run_prophet` ci-dessous. Nos séries de test
    couvrent quelques dizaines de jours, largement moins qu'un an : la composante
    saisonnière annuelle y est sous-déterminée et produit des extrapolations erratiques
    sans rapport avec la logique testée ici (walk-forward vs batch), cf.
    BRIEF_correction_prophet.md §4 point 4 ("le test valide la logique walk-forward,
    pas la précision de Prophet"). Ne modifie pas prophet_model.py : monkeypatch du
    symbole `Prophet` importé dans le module, restauré automatiquement par pytest."""
    original = pm.Prophet

    def _factory(*args, **kwargs):
        kwargs["yearly_seasonality"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(pm, "Prophet", _factory)


def _legacy_batch_run_prophet(train, test):
    """Ancien run_prophet (avant BRIEF_correction_prophet.md) : un seul fit sur
    `train`, puis predict de toute la fenêtre `test` en un coup -- exactement le
    corps de fonction cité en §0 du brief. Conservé ici seulement pour la
    comparaison de non-régression ; le code de production ne fait plus ça. Utilise
    `pm.Prophet` (pas un import direct) pour rester soumis au même monkeypatch que
    le nouveau run_prophet dans les tests qui en ont besoin."""
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(train.index),
        "y": train.astype(float).values.flatten(),
    })
    model = pm.Prophet(interval_width=1 - pm.PI_ALPHA, daily_seasonality=False,
                       weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_train)
    forecast = model.predict(pd.DataFrame({"ds": pd.to_datetime(test.index)}))
    preds = forecast["yhat"].values
    lower = forecast["yhat_lower"].values
    upper = forecast["yhat_upper"].values
    metrics = pm.compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper)
    return {**metrics, "predictions": preds, "lower": lower, "upper": upper,
            "index": test.index, "actual": test.values}


def _trending_then_flat_series(n_train=60, n_test=20, seed=0):
    """Train : tendance haussière compoundée propre (~+0.5 %/jour, faible bruit) --
    mime la fenêtre d'entraînement haussière observée en pratique (BTC/ETH). Test :
    renversement vers un régime plat (~0 %/jour) -- mime le décrochage réel derrière
    le bug Prophet OOS (audit : forte tendance pendant le train, croissance qui cale
    au test ; un batch fige l'extrapolation haussière du train et sur-estime de plus
    en plus au fil de la fenêtre de test)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_train + n_test)

    train_returns = rng.normal(0.005, 0.001, n_train)
    train_prices = 100.0 * np.cumprod(1 + train_returns)

    test_returns = rng.normal(0.0, 0.001, n_test)
    test_prices = train_prices[-1] * np.cumprod(1 + test_returns)

    prices = np.concatenate([train_prices, test_prices])
    s = pd.Series(prices, index=idx, name="Close")
    return s.iloc[:n_train], s.iloc[n_train:]


# ── 1. Non-régression de dérive (§4, point 1) ────────────────────────────────

def test_walk_forward_removes_batch_overestimation_bias(monkeypatch):
    _disable_yearly_seasonality(monkeypatch)
    reset_seeds(0)
    train, test = _trending_then_flat_series(n_train=60, n_test=20, seed=0)

    old_batch = _legacy_batch_run_prophet(train, test)
    new_walk_forward = pm.run_prophet(train, test)

    level = float(np.mean(test.values))
    old_bias = float(np.mean(np.asarray(old_batch["predictions"]) - test.values))
    new_bias = float(np.mean(np.asarray(new_walk_forward["predictions"]) - test.values))

    assert old_bias > 0.05 * level            # ancien : sur-estimation nette et non ambiguë
    assert abs(new_bias) < 0.01 * level        # nouveau : biais quasi nul (seuil §4 : <1% du niveau moyen)
    assert new_walk_forward["MAE"] < old_batch["MAE"]   # nouveau nettement plus précis

    # l'ancien dérive de plus en plus loin dans la fenêtre de test (extrapolation
    # figée depuis la fin du train) ; le nouveau, reconditionné à chaque pas, ne
    # montre pas cette croissance de l'erreur avec l'horizon.
    old_err = np.asarray(old_batch["predictions"]) - test.values
    new_err = np.asarray(new_walk_forward["predictions"]) - test.values
    mid = len(test) // 2
    assert np.mean(old_err[mid:]) > np.mean(old_err[:mid])           # ancien : empire avec l'horizon
    assert abs(np.mean(new_err[mid:])) < abs(np.mean(old_err[mid:]))  # nouveau : ne dérive pas pareil


# ── 2. Contrat de retour (§4, point 2) ────────────────────────────────────────

def test_run_prophet_contract():
    train, test = make_train_test(n_train=50, n_test=8, seed=0)

    result = pm.run_prophet(train, test)

    for key in ("predictions", "lower", "upper", "index", "actual"):
        assert len(result[key]) == len(test)

    preds = np.asarray(result["predictions"], dtype=float)
    lower = np.asarray(result["lower"], dtype=float)
    upper = np.asarray(result["upper"], dtype=float)
    assert np.all(np.isfinite(preds)) and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= preds) and np.all(preds <= upper)   # point à point

    for key in ("RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "Dir. Acc (%)",
               "PI Cov 95% (%)", "Ljung-Box p", "Train Time (s)"):
        assert key in result


# ── 3. Cohérence 1-pas (§4, point 3) ──────────────────────────────────────────

def test_last_step_matches_next_step_prophet(monkeypatch):
    """Le dernier pas du walk-forward et next_step_prophet(historique jusqu'à
    l'avant-dernier point) sont EXACTEMENT le même calcul (même historique connu,
    même unique date à prédire, mêmes hyperparamètres Prophet) -- ils doivent
    produire le même point et le même intervalle, à tolérance numérique près."""
    _disable_yearly_seasonality(monkeypatch)
    train, test = make_train_test(n_train=50, n_test=6, seed=0)

    result = pm.run_prophet(train, test)

    history = pd.concat([train, test.iloc[:-1]])
    pred, lo, hi = pm.next_step_prophet(history, next_date=test.index[-1])

    # yhat (point) vient de l'optimisation MAP -- déterministe, tolérance serrée.
    assert pred == pytest.approx(result["predictions"][-1], rel=1e-3, abs=1e-3)
    # yhat_lower/yhat_upper viennent d'un échantillonnage Monte-Carlo de l'incertitude
    # (uncertainty_samples, RNG global non re-seedé identiquement entre les deux
    # instanciations de Prophet) -- même calcul en espérance, mais pas bit-exact ;
    # tolérance plus large en conséquence (2 %), toujours largement assez serrée pour
    # détecter un vrai bug de logique (mauvaise date, mauvais historique, etc.).
    assert lo == pytest.approx(result["lower"][-1], rel=2e-2)
    assert hi == pytest.approx(result["upper"][-1], rel=2e-2)


# ── 4. refit_freq : documenté, défaut à 1 ─────────────────────────────────────

def test_refit_freq_defaults_to_every_step():
    import inspect
    assert inspect.signature(pm.run_prophet).parameters["refit_freq"].default == 1


def test_refit_freq_greater_than_one_reintroduces_drift(monkeypatch):
    """`refit_freq>1` doit se comporter comme annoncé (§3) : moins de refits que de
    pas de test -- pas une garantie de précision, juste que le paramètre a un effet
    réel (sinon la doc du §3 serait fausse)."""
    _disable_yearly_seasonality(monkeypatch)
    train, test = _trending_then_flat_series(n_train=60, n_test=20, seed=0)

    refit_every_step = pm.run_prophet(train, test, refit_freq=1)
    refit_rarely = pm.run_prophet(train, test, refit_freq=len(test))   # un seul refit, comme l'ancien batch

    bias_every_step = abs(np.mean(np.asarray(refit_every_step["predictions"]) - test.values))
    bias_rarely = abs(np.mean(np.asarray(refit_rarely["predictions"]) - test.values))
    assert bias_rarely > bias_every_step
