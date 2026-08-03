"""
multi_horizon.py — Adaptateurs multi-horizon pour les modèles de models/
=========================================================================
Les 5 fichiers de models/ (arima, sarima, prophet, lstm, naive) ne savent faire que du
walk-forward 1-step (ou next_step_*, un seul pas au-delà du dernier point). Ce module les
étend à un forecast multi-horizon **sans les modifier** : chaque fonction ci-dessous fitte
le modèle **une seule fois** sur la série d'entraînement, puis produit en un seul passage
les prévisions (point, IC95 bas, IC95 haut) pour tous les horizons demandés.

Contrat commun : `forecast_horizons_<model>(train: pd.Series, horizons: list[int]) ->
dict[int, tuple[float, float, float]]` où les clés sont des horizons en JOURS DE TRADING
(1-indexé, 1 = le jour suivant train.index[-1]).

Extensibilité : pour ajouter un nouveau modèle au benchmark, écrire une fonction
`forecast_horizons_<nom>` suivant ce contrat et l'ajouter à MODEL_ADAPTERS.

Adoptions sigma (BRIEF_branchement_prod_calibration_sigma.md, suite de
HANDOFF_sigma_calibration_suivi.md §5/§8.2 -- les adoptions étaient déjà en place dans
models/*.py mais n'étaient JAMAIS exercées ici, donc jamais visibles côté prévision live
ni côté backtest D+7, cf. BRIEF §0) :
  - ARIMA-GARCH : `dist` (défaut `arima_model.GARCH_DIST` = "skewt") -- bornes recalculées
    depuis les quantiles de la loi ajustée (mirror `arima_model._dist_shape`/
    `_std_quantiles`) au lieu du multiplicateur normal figé `Z_95`. `dist="normal"`
    reproduit EXACTEMENT (bit-for-bit) l'ancien calcul symétrique `+/- Z_95*sigma`.
  - Prophet : `log_space` (défaut `prophet_model.LOG_SPACE` = True) -- fit sur log(price),
    bornes exponentiées (loi lognormale en prix, même spécification que
    `prophet_model.run_prophet`). `log_space=False` reproduit l'ancien calcul en espace prix.
  - SARIMA / Naive / LSTM : `sigma_scale` (dict optionnel `{h_days: facteur}`, défaut None)
    -- correction multiplicative autour du point, mirror `run_sarima`/`run_naive`/`run_lstm`
    (`lo = point - (point - lo)*corr`, `hi = point + (hi - point)*corr`). Un horizon absent
    du dict (ou `sigma_scale=None`) n'est PAS corrigé : bornes brutes inchangées, bit-for-bit.
    Le facteur lui-même (sqrt(EWMA(z^2)) causal depuis tracking.db) est calculé par
    l'appelant (cf. validation/sigma_scale.py + model_artifacts/pipeline.py), jamais ici --
    ce module n'a aucun accès à tracking.db.

Réversibilité : tous ces paramètres ont un défaut qui ACTIVE les adoptions (comportement
prod). `model_artifacts/pipeline.py --calibrate-sigma off` les repasse explicitement à
leur valeur legacy (dist="normal", log_space=False, sigma_scale=None partout) pour
restaurer le comportement historique de ce module bit-for-bit (cf. son test de
non-régression dédié).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
import arima_model
import sarima_model
import naive_model
# prophet_model / lstm_model pull in heavy optional deps (prophet/cmdstanpy, tensorflow) —
# imported lazily inside their adapter so a missing dependency only disables that one
# model (caught per-model in run_benchmark.py) instead of crashing this whole module.


def _scaled_bounds(point: float, lo_raw: float, hi_raw: float, h: int, sigma_scale: dict):
    """`lo_raw`/`hi_raw` corrigés multiplicativement par `sigma_scale.get(h)` autour de
    `point` -- mirror `sarima_model.run_sarima`/`naive_model.run_naive`
    (`lo = point - (point - lo)*corr`). Renvoie `(lo_raw, hi_raw)` TELS QUELS (aucune
    arithmétique) si `sigma_scale` est None ou n'a pas d'entrée pour `h` : garantit la
    reproduction bit-for-bit de l'ancien calcul quand la correction est absente, plutôt
    qu'un `* 1.0` qui resterait exposé à l'arrondi flottant d'une soustraction-addition."""
    if sigma_scale is None or h not in sigma_scale:
        return lo_raw, hi_raw
    corr = float(sigma_scale[h])
    lo = point - (point - lo_raw) * corr
    hi = point + (hi_raw - point) * corr
    return lo, hi


# ── ARIMA-GARCH ───────────────────────────────────────────────────────────────
def fit_arima(train: pd.Series, dist: str = None):
    """Fit ARIMA(order) puis GARCH(1,1) sur ses résidus, une seule fois.
    Extrait de forecast_horizons_arima (même calcul, exposé pour la sérialisation
    des artefacts modèles — cf. model_artifacts/pipeline.py).

    `dist` (défaut None -> `arima_model.GARCH_DIST`, adopté "skewt") : loi
    d'innovation du GARCH, cf. note d'adoption en tête de module. `dist="normal"`
    reproduit le fit historique de ce module (qui codait `dist="normal"` en dur)."""
    dist = arima_model.GARCH_DIST if dist is None else dist
    prices = train.astype(float).values
    returns = np.diff(np.log(prices)) * 100.0

    arima_res = arima_model.ARIMA(
        returns, order=arima_model.ARIMA_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit()
    resid = np.asarray(arima_res.resid, dtype=float)
    garch_res = arima_model.arch_model(
        resid, vol="Garch", p=1, q=1, dist=dist, rescale=False
    ).fit(disp="off")
    return arima_res, garch_res


def forecast_from_fitted_arima(arima_res, garch_res, last_price: float, horizons: list,
                               dist: str = None) -> dict:
    """Multi-step via ARIMA.forecast(steps=h) (retours cumulés) + variance GARCH
    cumulée (somme des variances par pas, hypothèse d'indépendance approx.),
    à partir d'objets déjà fittés (aucun nouveau fit ici).

    `dist` DOIT être la même valeur que celle passée à `fit_arima` pour ce `garch_res`
    (résout le multiplicateur de bande, cf. note d'adoption en tête de module) :
    `dist="normal"` -> bande symétrique `+/- Z_95` calculée EXACTEMENT comme l'ancien
    code (aucun passage par les quantiles de la loi, donc aucun risque de dérive
    numérique 1.96 vs norm.ppf(0.975)=1.959964) ; toute autre valeur -> quantiles de
    la loi réellement ajustée (`arima_model._dist_shape`/`_std_quantiles`), donc bande
    asymétrique pour skew-t/GED."""
    dist = arima_model.GARCH_DIST if dist is None else dist
    max_h = max(horizons)
    mean_fc = np.asarray(arima_res.forecast(steps=max_h), dtype=float) / 100.0
    garch_fc = garch_res.forecast(horizon=max_h, reindex=False)
    var_per_step = garch_fc.variance.values[-1, :] / (100.0 ** 2)

    cum_return = np.cumsum(mean_fc)
    cum_sigma = np.sqrt(np.cumsum(var_per_step))

    if dist == "normal":
        q_lo, q_hi = -arima_model.Z_95, arima_model.Z_95
    else:
        dist_obj, shape = arima_model._dist_shape(garch_res)
        q_lo, q_hi = arima_model._std_quantiles(dist_obj, shape, (0.95,))[0.95]

    results = {}
    for h in horizons:
        i = h - 1
        point = last_price * np.exp(cum_return[i])
        lo = last_price * np.exp(cum_return[i] + cum_sigma[i] * q_lo)
        hi = last_price * np.exp(cum_return[i] + cum_sigma[i] * q_hi)
        results[h] = (float(point), float(lo), float(hi))
    return results


def forecast_horizons_arima(train: pd.Series, horizons: list, dist: str = None) -> dict:
    """Fit once (fit_arima) puis forecast (forecast_from_fitted_arima) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables.
    Résout `dist` une seule fois ici pour garantir que fit et forecast utilisent
    exactement la même loi (cf. docstring de forecast_from_fitted_arima)."""
    dist = arima_model.GARCH_DIST if dist is None else dist
    arima_res, garch_res = fit_arima(train, dist=dist)
    last_price = train.astype(float).values[-1]
    return forecast_from_fitted_arima(arima_res, garch_res, last_price, horizons, dist=dist)


# ── SARIMA ────────────────────────────────────────────────────────────────────
def fit_sarima(train: pd.Series):
    """Fit SARIMAX une seule fois. Extrait de forecast_horizons_sarima."""
    history = train.astype(float).values.tolist()
    return sarima_model.SARIMAX(
        history, order=sarima_model.ORDER, seasonal_order=sarima_model.SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)


def forecast_from_fitted_sarima(result, horizons: list, sigma_scale: dict = None) -> dict:
    """Multi-step natif : SARIMAX.get_forecast(steps=h) donne predicted_mean et
    conf_int() pour chaque pas 1..h en un seul appel, à partir d'un résultat déjà fitté.

    `sigma_scale` : cf. note d'adoption en tête de module (_scaled_bounds)."""
    max_h = max(horizons)
    fc = result.get_forecast(steps=max_h)
    pred_mean = np.asarray(fc.predicted_mean, dtype=float)
    ci = np.asarray(fc.conf_int(alpha=sarima_model.PI_ALPHA), dtype=float)

    results = {}
    for h in horizons:
        i = h - 1
        point = float(pred_mean[i])
        lo_raw, hi_raw = float(ci[i, 0]), float(ci[i, 1])
        lo, hi = _scaled_bounds(point, lo_raw, hi_raw, h, sigma_scale)
        results[h] = (point, lo, hi)
    return results


def forecast_horizons_sarima(train: pd.Series, horizons: list, sigma_scale: dict = None) -> dict:
    """Fit once (fit_sarima) puis forecast (forecast_from_fitted_sarima) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables."""
    result = fit_sarima(train)
    return forecast_from_fitted_sarima(result, horizons, sigma_scale=sigma_scale)


# ── Prophet ───────────────────────────────────────────────────────────────────
def fit_prophet(train: pd.Series, log_space: bool = None):
    """Fit Prophet une seule fois. Extrait de forecast_horizons_prophet.

    `log_space` (défaut None -> `prophet_model.LOG_SPACE`, adopté True) : fit sur
    log(price) au lieu du prix brut, cf. note d'adoption en tête de module."""
    import prophet_model
    log_space = prophet_model.LOG_SPACE if log_space is None else log_space
    y = train.astype(float).values.flatten()
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(train.index),
        "y": np.log(y) if log_space else y,
    })
    model = prophet_model.Prophet(
        interval_width=1 - prophet_model.PI_ALPHA,
        daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
    )
    model.fit(df_train)
    return model


def forecast_from_fitted_prophet(model, last_date, horizons: list,
                                 log_space: bool = None, sigma_scale: dict = None) -> dict:
    """Interroge directement les dates futures (jours ouvrés au-delà de last_date)
    sur un modèle déjà fitté — Prophet élargit nativement l'IC avec la distance
    dans le futur, sans dépendre d'un état interne à mettre à jour.

    `log_space` DOIT être la même valeur que celle passée à `fit_prophet` pour ce
    `model` (résout l'exponentiation des bornes, sinon `yhat`/`yhat_lower/upper`
    sont en espace log alors qu'on les traiterait comme des prix). `sigma_scale`
    (dict optionnel `{h_days: facteur}`) : correction EWMA appliquée EN ESPACE DE
    FIT (log si log_space, mirror `prophet_model.next_step_prophet`/`run_prophet`)
    avant l'exponentiation -- pas après, l'ordre change le résultat."""
    import prophet_model
    log_space = prophet_model.LOG_SPACE if log_space is None else log_space
    max_h = max(horizons)
    last_date = pd.to_datetime(last_date)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max_h)
    forecast = model.predict(pd.DataFrame({"ds": future_dates}))

    results = {}
    for h in horizons:
        i = h - 1
        row = forecast.iloc[i]
        yhat = float(row["yhat"])
        lo, hi = _scaled_bounds(yhat, float(row["yhat_lower"]), float(row["yhat_upper"]),
                                h, sigma_scale)
        if log_space:
            results[h] = (float(np.exp(yhat)), float(np.exp(lo)), float(np.exp(hi)))
        else:
            results[h] = (yhat, lo, hi)
    return results


def forecast_horizons_prophet(train: pd.Series, horizons: list,
                              log_space: bool = None, sigma_scale: dict = None) -> dict:
    """Fit once (fit_prophet) puis forecast (forecast_from_fitted_prophet) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables.
    Résout `log_space` une seule fois ici pour garantir que fit et forecast utilisent
    exactement le même espace (cf. docstring de forecast_from_fitted_prophet)."""
    import prophet_model
    log_space = prophet_model.LOG_SPACE if log_space is None else log_space
    model = fit_prophet(train, log_space=log_space)
    return forecast_from_fitted_prophet(model, train.index[-1], horizons,
                                        log_space=log_space, sigma_scale=sigma_scale)


# ── LSTM ──────────────────────────────────────────────────────────────────────
def fit_lstm(train: pd.Series, epochs: int = None, seed: int = None, seq_len: int = None):
    """Fit le réseau une seule fois. Extrait de forecast_horizons_lstm.
    Retourne (model, scaler, std_résidus, série_scalée_complète) : tout ce qu'il
    faut pour forecaster (forecast_from_fitted_lstm) ou sérialiser l'artefact.

    `seq_len` (défaut `lstm_model.SEQ_LEN`, comportement daily inchangé) : régime
    hebdo natif (BRIEF_lstm_weekly_retune.md) a besoin d'un lookback différent de
    30 -- l'appelant doit repasser la MÊME valeur à `forecast_from_fitted_lstm`
    (pas de round-trip via le tuple retourné, pour ne pas casser les appelants
    existants qui dépaquettent (model, scaler, std, scaled))."""
    import lstm_model
    seq_len = lstm_model.SEQ_LEN if seq_len is None else seq_len
    epochs = lstm_model.EPOCHS if epochs is None else epochs
    seed = lstm_model.DEFAULT_SEED if seed is None else seed
    lstm_model.set_seed(seed)

    if len(train) <= seq_len:
        raise ValueError(
            f"train series has {len(train)} points, but seq_len={seq_len} requires "
            f"more than {seq_len} points to build at least one training sequence."
        )

    scaler = lstm_model.MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    X, y = lstm_model.make_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = lstm_model.build_lstm(seq_len)
    es = lstm_model.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=lstm_model.BATCH_SIZE,
             validation_split=0.1, callbacks=[es], verbose=0)

    train_preds = scaler.inverse_transform(
        model.predict(X, verbose=0).reshape(-1, 1)).flatten()
    std = float(np.std(train.values[seq_len:] - train_preds))
    return model, scaler, std, scaled


def forecast_from_fitted_lstm(model, scaler, std: float, scaled, horizons: list,
                              seq_len: int = None, sigma_scale: dict = None) -> dict:
    """Un seul rollout récursif de max(horizons) pas à partir d'objets déjà fittés :
    le réseau se ré-alimente de ses propres prédictions (jamais du vrai futur,
    contrainte point-in-time). L'IC s'élargit en sqrt(h) à partir de l'écart-type
    des résidus d'entraînement (même convention que next_step_lstm).

    `seq_len` : DOIT être la même valeur que celle passée à `fit_lstm` pour ce
    `model` (défaut `lstm_model.SEQ_LEN`, comportement daily inchangé). `sigma_scale`
    : cf. note d'adoption en tête de module (_scaled_bounds)."""
    import lstm_model
    seq_len = lstm_model.SEQ_LEN if seq_len is None else seq_len
    max_h = max(horizons)

    buffer = list(scaled[-seq_len:])
    rollout_scaled = []
    for _ in range(max_h):
        x = np.array(buffer[-seq_len:]).reshape(1, seq_len, 1)
        p_scaled = model.predict(x, verbose=0)[0, 0]
        rollout_scaled.append(p_scaled)
        buffer.append(p_scaled)   # recursif : jamais le vrai futur

    rollout_prices = scaler.inverse_transform(
        np.array(rollout_scaled).reshape(-1, 1)).flatten()

    results = {}
    for h in horizons:
        i = h - 1
        point = float(rollout_prices[i])
        sigma_h = std * np.sqrt(h)
        lo_raw, hi_raw = point - 1.96 * sigma_h, point + 1.96 * sigma_h
        lo, hi = _scaled_bounds(point, lo_raw, hi_raw, h, sigma_scale)
        results[h] = (point, lo, hi)
    return results


def forecast_horizons_lstm(train: pd.Series, horizons: list, epochs: int = None,
                           seed: int = None, seq_len: int = None,
                           sigma_scale: dict = None) -> dict:
    """Fit once (fit_lstm) puis forecast (forecast_from_fitted_lstm) — inchangé
    pour les appelants existants, juste réorganisé en 2 fonctions réutilisables.
    `seq_len=None` -> défaut `lstm_model.SEQ_LEN`, comportement daily inchangé."""
    model, scaler, std, scaled = fit_lstm(train, epochs=epochs, seed=seed, seq_len=seq_len)
    return forecast_from_fitted_lstm(model, scaler, std, scaled, horizons, seq_len=seq_len,
                                     sigma_scale=sigma_scale)


# ── Naive ─────────────────────────────────────────────────────────────────────
def forecast_horizons_naive(train: pd.Series, horizons: list, sigma_scale: dict = None) -> dict:
    """Persistence stricte (Point 0 du brief) : point_h = dernier_prix, exactement ;
    IC95 = dernier_prix ± 1.96·σ·sqrt(h), σ = écart-type des variations 1 jour du train
    (échelle marche aléatoire, même convention que les autres modèles).

    `sigma_scale` : cf. note d'adoption en tête de module (_scaled_bounds)."""
    last_price = float(train.iloc[-1])
    sigma = naive_model.train_sigma(train)
    results = {}
    for h in horizons:
        half = naive_model.Z_95 * sigma * np.sqrt(h)
        lo_raw, hi_raw = last_price - half, last_price + half
        lo, hi = _scaled_bounds(last_price, lo_raw, hi_raw, h, sigma_scale)
        results[h] = (last_price, lo, hi)
    return results


# ── TSDiff (diffusion, DEITA port) ──────────────────────────────────────────────
def forecast_horizons_tsdiff(train: pd.Series, horizons: list, epochs: int = None,
                             seed: int = None) -> dict:
    """Fit le denoiser de diffusion une fois sur les log-returns du train, puis
    échantillonne N chemins de retours (DDIM) et lit le prix à chaque horizon depuis
    le retour cumulé des `h` premiers pas. Point = moyenne du nuage d'échantillons ;
    IC95 = quantiles 2.5/97.5 (distribution prédictive du modèle). L'horizon généré
    (tsdiff_model.HORIZON) borne l'horizon exploitable — au-delà, on plafonne."""
    import tsdiff_model as td
    if seed is not None:
        td.set_seed(seed)
    ep = td.EPOCHS if epochs is None else epochs

    prices = train.values.astype(float)
    r = td._log_returns(prices)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = td._make_windows(z, td.SEQ_LEN, td.HORIZON)
    if len(H_win) == 0:
        raise ValueError("not enough return history to build TSDiff training windows.")

    model = td.TSDiff()
    model.train(H_win, T_win, epochs=ep)
    paths = model.sample_paths(z[-td.SEQ_LEN:].astype(np.float32),
                               n_samples=td.N_SAMPLES)   # [N, HORIZON] std step-returns

    last_price = float(prices[-1])
    results = {}
    for h in horizons:
        hh = min(int(h), td.HORIZON)                     # model generates HORIZON steps
        cum_r = paths[:, :hh].sum(axis=1) * sd + hh * mu  # de-standardized cumulative log-return
        price_samples = last_price * np.exp(cum_r)
        results[h] = (float(np.mean(price_samples)),
                      float(np.quantile(price_samples, 0.025)),
                      float(np.quantile(price_samples, 0.975)))
    return results


# ── NsDiff (diffusion, LSNM+UANS port, cf. models/nsdiff_model.py) ──────────────
def forecast_horizons_nsdiff(train: pd.Series, horizons: list, epochs: int = None,
                             seed: int = None) -> dict:
    """Calqué ligne à ligne sur forecast_horizons_tsdiff ci-dessus : fit le
    denoiser NsDiff une fois sur les log-returns du train, puis échantillonne N
    chemins de retours (ancestral, cf. nsdiff_model.py) et lit le prix à chaque
    horizon depuis le retour cumulé des `h` premiers pas. Point = moyenne du
    nuage d'échantillons ; IC95 = quantiles 2.5/97.5 (distribution prédictive du
    modèle). L'horizon généré (nsdiff_model.HORIZON) borne l'horizon
    exploitable -- au-delà, on plafonne."""
    import nsdiff_model as nd
    if seed is not None:
        nd.set_seed(seed)
    ep = nd.EPOCHS if epochs is None else epochs

    prices = train.values.astype(float)
    r = nd._log_returns(prices)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = nd._make_windows(z, nd.SEQ_LEN, nd.HORIZON)
    if len(H_win) == 0:
        raise ValueError("not enough return history to build NsDiff training windows.")

    model = nd.NsDiff()
    model.train(H_win, T_win, epochs=ep)
    paths = model.sample_paths(z[-nd.SEQ_LEN:].astype(np.float32),
                               n_samples=nd.N_SAMPLES)   # [N, HORIZON] std step-returns

    last_price = float(prices[-1])
    results = {}
    for h in horizons:
        hh = min(int(h), nd.HORIZON)                      # model generates HORIZON steps
        cum_r = paths[:, :hh].sum(axis=1) * sd + hh * mu   # de-standardized cumulative log-return
        price_samples = last_price * np.exp(cum_r)
        results[h] = (float(np.mean(price_samples)),
                      float(np.quantile(price_samples, 0.025)),
                      float(np.quantile(price_samples, 0.975)))
    return results


# ── Registre des modèles (point d'extension) ─────────────────────────────────
MODEL_ADAPTERS = {
    "ARIMA-GARCH": forecast_horizons_arima,
    "SARIMA":      forecast_horizons_sarima,
    "Prophet":     forecast_horizons_prophet,
    "LSTM":        forecast_horizons_lstm,
    "Naive":       forecast_horizons_naive,
    "TSDiff":      forecast_horizons_tsdiff,
    "NsDiff":      forecast_horizons_nsdiff,
}
