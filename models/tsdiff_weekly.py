"""
TSDiff-W — variante hebdomadaire du port TSDiff (PROTOTYPE, Phase 0)
====================================================================
Resample daily -> weekly (W-FRI, dernière clôture), entraîne le denoiser de
diffusion sur les rendements **hebdomadaires**, et génère **W+1 / W+2 / W+3 en un
seul tir** : le UNet1D de TSDiff produit tout le chemin d'horizon d'un coup, donc
AUCUNE récursion / AUCUN compounding d'erreur (c'est tout l'intérêt vs le
daily-multistep).

Réutilise entièrement l'architecture de models/tsdiff_model.py (classe TSDiff,
UNet1D, DDIM…) — on ne fait que changer la fréquence des données (weekly) et lire
l'horizon en semaines.

Prototype : additif, ne touche ni le pipeline ni le modèle daily.
"""

import numpy as np
import pandas as pd

import tsdiff_model as td   # réutilise TSDiff, _log_returns, _make_windows, set_seed


WEEK_RULE   = "W-FRI"   # ancre canonique : clôture du vendredi
SEQ_LEN_W   = 26        # 26 semaines de lookback (~6 mois)
HORIZON_W   = 3         # génère W+1, W+2, W+3
HIDDEN_W    = 64
DEPTH_W     = 2
T_DIFF_W    = 1000
EPOCHS_W    = 60
BATCH_W     = 32
N_SAMPLES_W = 100
K_DENOISE_W = 20


def to_weekly(prices: pd.Series) -> pd.Series:
    """Daily -> weekly (clôture du vendredi). Ne garde que les semaines complètes
    (dropna) : jamais de barre partielle du vendredi à venir (point-in-time)."""
    return prices.resample(WEEK_RULE).last().dropna()


def fit_weekly(train_weekly: pd.Series, seq_len=SEQ_LEN_W, horizon=HORIZON_W,
               hidden=HIDDEN_W, depth=DEPTH_W, T=T_DIFF_W, epochs=EPOCHS_W,
               batch_size=BATCH_W):
    """Entraîne un TSDiff sur les rendements hebdo standardisés. Retourne
    (model, z, mu, sd, last_price) — tout ce qu'il faut pour échantillonner."""
    p = train_weekly.values.astype(float)
    if len(p) <= seq_len + horizon:
        raise ValueError(
            f"série hebdo trop courte ({len(p)} barres) pour seq_len={seq_len} + "
            f"horizon={horizon} (il en faut > {seq_len + horizon}).")
    r = td._log_returns(p)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = td._make_windows(z, seq_len, horizon)
    if len(H_win) == 0:
        raise ValueError("pas assez d'historique hebdo pour construire des fenêtres.")

    model = td.TSDiff(seq_len=seq_len, horizon=horizon, hidden=hidden, depth=depth, T=T)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)
    return model, z, mu, sd, float(p[-1])


def forecast_from_fitted_weekly(model, mu: float, sd: float, lookback_z, last_price: float,
                                weeks=(1, 2, 3), n_samples=N_SAMPLES_W,
                                k_denoise=K_DENOISE_W) -> dict:
    """Échantillonne le chemin hebdo depuis un modèle DÉJÀ entraîné, conditionné sur
    `lookback_z` (les seq_len derniers rendements hebdo standardisés avec les mu/sd
    d'entraînement) et `last_price` (dernier prix à l'origine). Permet le protocole
    train-once-forward : fit une fois, forecast à N origines sans réentraînement.
    Retourne {k: {"point","lo","hi","samples"}}."""
    paths = model.sample_paths(np.asarray(lookback_z, dtype=np.float32),
                               n_samples=n_samples, k_denoise=k_denoise)   # [N, horizon]
    out = {}
    for k in weeks:
        cum_r = paths[:, :k].sum(axis=1) * sd + k * mu     # retour cumulé hebdo dé-standardisé
        px = last_price * np.exp(cum_r)                    # échantillons de prix à W+k
        out[k] = {
            "point": float(np.mean(px)),
            "lo": float(np.quantile(px, 0.025)),
            "hi": float(np.quantile(px, 0.975)),
            "samples": px,
        }
    return out


def forecast_weekly(train_weekly: pd.Series, weeks=(1, 2, 3),
                    n_samples=N_SAMPLES_W, k_denoise=K_DENOISE_W, **fit_kw) -> dict:
    """Entraîne (fit_weekly) puis échantillonne le chemin hebdo en UN tir (retrain
    à chaque appel). Voir forecast_from_fitted_weekly pour le mode train-once."""
    horizon = fit_kw.get("horizon", HORIZON_W)
    if max(weeks) > horizon:
        raise ValueError(f"weeks={weeks} dépasse l'horizon généré ({horizon}).")
    model, z, mu, sd, last = fit_weekly(train_weekly, **fit_kw)
    return forecast_from_fitted_weekly(model, mu, sd, z[-model.seq_len:], last,
                                       weeks=weeks, n_samples=n_samples, k_denoise=k_denoise)
