"""
regime_overlay.py — Couche de calibration régime pour le benchmark
=====================================================================
Wrapper fin autour de calibration.regime.RegimeAgent, et fonction d'élargissement
d'intervalle de confiance selon le régime détecté. C'est la SEULE façon dont les
modèles de models/ "utilisent" le régime : on ne modifie aucun fichier models/*.py,
on élargit (ou resserre) leur IC 95% après coup, ici, uniformément pour tous.

Rationale : en régime de stress, la volatilité réalisée dépasse largement ce que les
modèles calibrent sur l'historique récent -> leurs IC 95% sont en réalité sous-couverts.
Élargir par (1 + stress_score) (calme ~x1, stress plein ~x2) rapproche le taux de
couverture réel de 95%, ce qui est exactement ce que measure le système de verdicts.
"""

from datetime import timedelta

import pandas as pd

from calibration.regime.regime_agent import RegimeAgent
from calibration.regime.regime_state import RegimeState


def fit_predict_regime(ohlcv: pd.DataFrame, train_end) -> RegimeState:
    """Entraîne un RegimeAgent sur `ohlcv` jusqu'à `train_end` inclus, puis renvoie
    le RegimeState au jour suivant train_end (contrainte point-in-time : seules les
    données <= train_end sont utilisées)."""
    agent = RegimeAgent()
    train_end_str = str(pd.Timestamp(train_end).date())
    agent.fit(ohlcv, train_end=train_end_str)
    as_of = pd.Timestamp(train_end) + timedelta(days=1)
    return agent.predict(ohlcv, as_of=as_of)


def scale_interval(point: float, lo: float, hi: float, stress_score: float):
    """Élargit asymétriquement (point-lo) et (hi-point) par un facteur (1+stress_score).
    stress_score=0 (calme) -> IC inchangé ; stress_score=1 (stress plein) -> IC x2."""
    factor = 1.0 + stress_score
    lo2 = point - (point - lo) * factor
    hi2 = point + (hi - point) * factor
    return lo2, hi2
