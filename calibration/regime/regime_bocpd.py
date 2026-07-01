"""
regime_bocpd.py — Détection Bayésienne de Changements de Régime pour DEITA

Implémentation from-scratch de BOCPD (Adams & MacKay, 2007) avec modèle
d'observation Gaussien et prior Normal-Normal conjugué.

Dépendances : numpy, pandas, math (bibliothèque standard Python uniquement).
"""

import math
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd


class RegimeBOCPD:
    """
    BOCPD (Bayesian Online Changepoint Detection) pour DEITA.

    Principe :
        À chaque instant t, on maintient la distribution P(r_t = r | x_{1:t}) sur la
        "longueur de run" r_t = nombre d'observations accumulées dans le segment courant
        depuis le dernier changement de régime.

        changepoint_prob = max_{s ∈ [t-RECENCY_DAYS, t]}  P(r_s ≤ RECENCY_DAYS | x_{1:s})
            → probabilité bayésienne qu'un nouveau segment a démarré dans les
              RECENCY_DAYS derniers jours. Robuste au décalage de détection.

    Modèle d'observation :
        x_t | μ, r_t  ~  N(μ, σ²)    avec σ² = 1 (signal préstandardisé)

    Prior conjugué sur μ en début de chaque segment :
        μ  ~  N(μ_0, σ²/κ_0)

    Après r observations dans le segment courant, la distribution prédictive est :
        P(x_{t+1} | x_{t-r+1:t})  ~  N(μ_r, σ²·(κ_r + 1)/κ_r)
        avec κ_r = κ_0 + r  et  μ_r = (κ_0·μ_0 + Σ x_i) / κ_r

    Fonction de hasard constante : h = HAZARD_RATE (probabilité de changer de régime
    à chaque pas de temps, ici ≈ 1/60 → durée moyenne d'un régime ≈ 60 jours).

    Signal :
        log(|returns| + ε), normalisé par les stats d'entraînement.
        Ce choix convertit les changements de volatilité (calme → stress) en
        changements de moyenne, détectables par le prior Gaussien.

    Complète les champs changepoint_prob et is_transitioning du RegimeState
    produit par RegimeHMM (Maéva). Ne redéfinit pas l'interface RegimeState.
    """

    CHANGEPOINT_THRESHOLD = 0.5   # seuil pour is_transitioning
    WINDOW_DAYS = 90              # fenêtre glissante (jours)
    RECENCY_DAYS = 3              # max sur les N derniers jours (3 : 0 faux positifs en calme pur)
    HAZARD_RATE = 1.0 / 60.0     # durée de régime attendue ≈ 60 jours
    PRIOR_KAPPA = 1.0             # force du prior sur μ_0 (équivalent à 1 obs fictive)
    LOG_ABS_EPSILON = 1e-4        # évite log(0)

    def __init__(self) -> None:
        self._signal_mean: float = 0.0
        self._signal_std: float = 1.0
        self._is_fitted: bool = False

    # ── API publique ───────────────────────────────────────────────────────────

    def fit(self, prices_train: pd.DataFrame) -> None:
        """
        Calibre les statistiques du signal log|returns| sur l'historique d'entraînement.

        Les seuils sont figés ici et ne sont pas recalculés à chaque predict().

        Paramètres
        ----------
        prices_train : pd.DataFrame
            Données OHLCV (colonnes Open/High/Low/Close/Volume, index DatetimeIndex).
            Minimum 30 jours.
        """
        returns = prices_train["Close"].pct_change().dropna()
        if len(returns) < 30:
            raise ValueError(
                f"Pas assez de données pour calibrer RegimeBOCPD : {len(returns)} < 30 jours."
            )
        log_abs = np.log(np.abs(returns.values) + self.LOG_ABS_EPSILON)
        self._signal_mean = float(np.mean(log_abs))
        self._signal_std = float(np.std(log_abs)) or 1.0
        self._is_fitted = True

    def predict(self, prices: pd.DataFrame, as_of: datetime) -> Tuple[float, bool]:
        """
        Estime la probabilité bayésienne qu'un changement de régime vient de se produire.

        Contrainte point-in-time : seules les données strictement antérieures à as_of
        sont utilisées (prices.index < as_of).

        Paramètres
        ----------
        prices : pd.DataFrame
            Données OHLCV complètes (le filtre point-in-time est appliqué ici).
        as_of : datetime
            Date de prédiction.

        Retourne
        --------
        changepoint_prob : float dans [0, 1]
            max P(changepoint à t) sur les RECENCY_DAYS derniers jours du window.
            Robuste à un léger décalage de détection (e.g. FTX détecté 1–2 jours
            après le choc, mais flagué both).
        is_transitioning : bool
            True si changepoint_prob > CHANGEPOINT_THRESHOLD.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RegimeBOCPD doit être entraîné avant predict() — appeler fit() d'abord."
            )

        data = prices[prices.index < as_of]
        returns = data["Close"].pct_change().dropna()

        if len(returns) < self.WINDOW_DAYS:
            return 0.0, False

        signal = self._to_signal(returns.iloc[-self.WINDOW_DAYS:])
        cp_probs = self._run_bocpd(signal)

        changepoint_prob = float(np.max(cp_probs[-self.RECENCY_DAYS:]))
        is_transitioning = changepoint_prob > self.CHANGEPOINT_THRESHOLD

        return changepoint_prob, bool(is_transitioning)

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def _to_signal(self, returns: pd.Series) -> np.ndarray:
        """Transforme les rendements en signal BOCPD : log|r| normalisé (centré, réduit)."""
        log_abs = np.log(np.abs(returns.values) + self.LOG_ABS_EPSILON)
        return (log_abs - self._signal_mean) / self._signal_std

    def _run_bocpd(self, data: np.ndarray) -> np.ndarray:
        """
        Algorithme BOCPD (Adams & MacKay, 2007) — modèle Normal avec variance connue.

        À chaque pas t :
          - Calcule la probabilité prédictive Gaussienne N(μ_r, σ²·(κ_r+1)/κ_r)
            pour chaque longueur de run r active.
          - Met à jour la distribution de run-length :
              P(r_t = r+1) ∝ P(r_{t-1} = r) · p(x_t | r) · (1 − h)   [croissance]
              P(r_t = 0)   ∝ Σ_r P(r_{t-1} = r) · p(x_t | r) · h     [changement]
          - Met à jour les statistiques suffisantes (κ_r, μ_r) par update bayésien.

        Paramètres
        ----------
        data : np.ndarray de taille n
            Signal normalisé (WINDOW_DAYS valeurs).

        Retourne
        --------
        cp_probs : np.ndarray de taille n
            cp_probs[t] = P(r_t ≤ RECENCY_DAYS | x_{1:t})
            = probabilité que le run courant ait démarré dans les RECENCY_DAYS derniers jours.
        """
        n = len(data)
        h = self.HAZARD_RATE
        kappa_0 = self.PRIOR_KAPPA
        mu_0 = 0.0
        obs_var = 1.0  # variance connue (signal standardisé → σ² ≈ 1)

        # R[r] = P(r_{t-1} = r | x_{1:t-1}), normalisé à chaque pas
        R = np.zeros(n + 1)
        R[0] = 1.0  # avant x_1 : run de longueur 0 (on n'a encore rien observé)

        # Statistiques suffisantes du posterior Normal par longueur de run
        # kappa[r] = κ_0 + r   (κ_0 "obs fictives" + r obs réelles)
        # mu_post[r] = μ a posteriori après r observations dans le segment courant
        kappa = np.full(n + 1, kappa_0, dtype=float)
        mu_post = np.full(n + 1, mu_0, dtype=float)

        cp_probs = np.zeros(n)

        for t in range(n):
            x = data[t]
            active = t + 1  # longueurs de run actives : 0, 1, ..., t

            ak = kappa[:active]    # κ_r pour chaque run length r
            am = mu_post[:active]  # μ_r pour chaque run length r

            # Variance prédictive : σ²·(κ_r + 1)/κ_r
            pred_var = obs_var * (ak + 1.0) / ak

            # Log-probabilité prédictive Gaussienne
            log_pred = -0.5 * (np.log(2.0 * math.pi * pred_var) + (x - am) ** 2 / pred_var)

            # Stabilité numérique : soustraction du max avant exp
            log_pred_shifted = log_pred - log_pred.max()
            pred_prob = np.exp(log_pred_shifted)  # proportionnel à la vraie pred_prob

            # Mise à jour de la distribution de run-length
            R_new = np.zeros(n + 1)
            R_new[1 : active + 1] = R[:active] * pred_prob * (1.0 - h)  # croissance
            R_new[0] = np.sum(R[:active] * pred_prob) * h                 # changement

            # Normalisation (évite l'accumulation d'erreur numérique)
            total = R_new[: active + 1].sum()
            if total > 1e-300:
                R_new[: active + 1] /= total
            else:
                R_new[0] = 1.0  # repli en cas de sous-dépassement numérique

            # Update bayésien des statistiques suffisantes (pour les runs croissants)
            new_kappa = ak + 1.0
            kappa[1 : active + 1] = new_kappa
            mu_post[1 : active + 1] = (ak * am + x) / new_kappa

            # Réinitialisation du prior pour le state changepoint (r = 0)
            kappa[0] = kappa_0
            mu_post[0] = mu_0

            R = R_new
            # P(run ≤ RECENCY_DAYS jours) = P(changepoint dans les RECENCY_DAYS derniers jours)
            # Note : P(r_t=0) = h toujours (identité mathématique), donc on ne peut pas
            # utiliser R[0] seul. On somme les petites longueurs de run pour capter
            # l'accumulation de masse sur les états récents après un vrai changement.
            cp_probs[t] = float(R[: self.RECENCY_DAYS + 1].sum())

        return cp_probs
