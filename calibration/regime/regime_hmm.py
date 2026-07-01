from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
from arch import arch_model
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from calibration.regime.regime_state import RegimeState


class RegimeHMM:
    """
    Moteur de détection de régime pour DEITA.
    Combine GARCH(1,1) pour la volatilité conditionnelle
    et un HMM gaussien à 3 états pour la classification de régime.
    """

    VERSION = "hmm-garch-adx-v1"
    N_STATES = 2  # HMM à 2 états : stress / non-stress (le split calm/trending se fait ensuite par seuil ADX)
    ADX_PERIOD = 14
    ADX_TRENDING_THRESHOLD = 25  # au sein du non-stress : ADX > 25 → trending, sinon calm
    GARCH_P = 1
    GARCH_Q = 1
    HMM_N_ITER = 200
    HMM_COV_TYPE = "diag"
    MIN_TRAIN_DAYS = 252  # 1 an de trading minimum
    CHANGEPOINT_THRESHOLD = 0.5  # seuil pour is_transitioning (défaut, écrasable par Kyrio)
    HMM_RESTART_SEEDS = [42, 0, 1, 7, 13]  # multi-restart : on garde le meilleur log-likelihood

    def __init__(self):
        self._scaler = None
        self._hmm = None
        self._vol_thresholds = None
        self._state_labels: Dict[int, str] = {}
        self._is_fitted = False

    def fit(self, prices: pd.DataFrame, train_end: str) -> None:
        prices_train = prices[prices.index <= train_end]

        if len(prices_train) < self.MIN_TRAIN_DAYS:
            raise ValueError(
                f"Pas assez de données d'entraînement : {len(prices_train)} < {self.MIN_TRAIN_DAYS}"
            )

        features = self._compute_features(prices_train)
        features = features.dropna()

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(features)

        best_model = None
        best_score = None
        for seed in self.HMM_RESTART_SEEDS:
            candidate = hmm.GaussianHMM(
                n_components=self.N_STATES,
                covariance_type=self.HMM_COV_TYPE,
                n_iter=self.HMM_N_ITER,
                random_state=seed,
            )
            candidate.fit(X_scaled)
            score = candidate.score(X_scaled)
            if best_score is None or score > best_score:
                best_score = score
                best_model = candidate
        self._hmm = best_model

        sigma_t = features["sigma_t"]
        q33, q66 = sigma_t.quantile([1 / 3, 2 / 3])
        self._vol_thresholds = (q33, q66)

        self._state_labels = self._assign_regime_labels()

        self._is_fitted = True

    def predict(self, prices: pd.DataFrame, as_of: datetime) -> RegimeState:
        if not self._is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant predict() — appeler fit() d'abord.")

        prices_to_use = prices[prices.index < as_of]

        features = self._compute_features(prices_to_use)
        features = features.dropna()

        if len(features) < 30:
            raise ValueError("Pas assez de données après suppression des NaN")

        X_scaled = self._scaler.transform(features)

        log_probs = self._hmm.predict_proba(X_scaled)
        last_probs = log_probs[-1]  # shape (2,) : stress / non_stress

        stress_idx = [i for i, label in self._state_labels.items() if label == "stress"][0]
        p_stress = float(last_probs[stress_idx])
        p_non_stress = 1.0 - p_stress

        adx_last = features["adx"].iloc[-1]
        if adx_last > self.ADX_TRENDING_THRESHOLD:
            p_trending, p_calm = p_non_stress, 0.0
        else:
            p_calm, p_trending = p_non_stress, 0.0

        probs = {"calm": p_calm, "trending": p_trending, "stress": p_stress}

        sigma_t_last = features["sigma_t"].iloc[-1]
        q33, q66 = self._vol_thresholds
        vol_bucket = 0 if sigma_t_last < q33 else (1 if sigma_t_last < q66 else 2)

        dominant_state_idx = int(np.argmax(last_probs))
        p_stay = self._hmm.transmat_[dominant_state_idx, dominant_state_idx]
        expected_duration_days = 1.0 / (1.0 - p_stay)

        state = RegimeState(
            probs=probs,
            vol_bucket=vol_bucket,
            stress_score=probs["stress"],
            expected_duration_days=expected_duration_days,
            as_of=as_of,
            version=self.VERSION,
        )
        state.validate()
        return state

    def _compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices["Close"].pct_change().dropna() * 100  # en pourcentage pour ARCH
        am = arch_model(returns, vol="Garch", p=self.GARCH_P, q=self.GARCH_Q, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        sigma_t = res.conditional_volatility  # pd.Series, même index que returns

        adx_df = ta.adx(prices["High"], prices["Low"], prices["Close"], length=self.ADX_PERIOD)
        adx = adx_df[f"ADX_{self.ADX_PERIOD}"]  # pd.Series

        # Ratio du volume du jour sur la moyenne mobile 30j du volume
        # → capture les explosions de volume relatives, indépendamment du niveau absolu
        volume_norm = prices["Volume"] / prices["Volume"].rolling(30).mean()

        features = pd.DataFrame({
            "sigma_t": sigma_t,
            "adx": adx,
            "volume_norm": volume_norm,
        }, index=prices.index)
        return features.dropna()

    def _assign_regime_labels(self) -> Dict[int, str]:
        means = self._hmm.means_  # shape (2, 3) → (n_states, n_features)
        # means[:, 0] = σ_t moyenne par état
        # means[:, 2] = volume_norm moyen par état

        # L'état STRESS a la σ_t la plus élevée ET le volume le plus élevé
        # → score de stress = rank(σ_t) + rank(volume_norm)
        stress_score = np.argsort(means[:, 0]) + np.argsort(means[:, 2])
        stress_idx = int(np.argmax(stress_score))
        non_stress_idx = 1 - stress_idx

        mapping = {stress_idx: "stress", non_stress_idx: "non_stress"}
        self._state_labels = mapping
        return mapping
