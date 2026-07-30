"""
model_artifacts/crps_kpis.py — agrégation du CRPS empirique pour Gate 2 (D1).
================================================================================
Réutilise l'estimateur déjà testé de experiments/crps_metrics.py (Gneiting & Raftery
2007, eq. 20) plutôt que d'en réimplémenter un. Chaque modèle génère lui-même son
nuage d'échantillons par pas de validation (bootstrap des résidus pour ARIMA-GARCH/
SARIMA/Prophet, Monte Carlo Dropout pour LSTM, échantillons natifs pour TSDiff — cf.
paramètre `n_ensemble`/`keep_samples` de run_<model> dans models/*.py) ; ce module se
contente de transformer cette liste de nuages en un scalaire CRPS par walk-forward.

Aucun import lourd au niveau module (seulement numpy + crps_empirical, pur numpy) :
ce fichier doit rester importable depuis model_artifacts/lstm_worker.py, qui tourne
dans un sous-processus isolé pour éviter un deadlock TensorFlow (cf. docstring de ce
fichier) et n'importe donc jamais model_artifacts.pipeline ni benchmarks.*.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from crps_metrics import crps_empirical  # noqa: E402 (import après sys.path.insert)

# Nombre de tirages bootstrap/MC-Dropout par pas de validation — compromis coût/
# précision (identique pour tous les modèles, pour une comparaison équitable).
DEFAULT_N_ENSEMBLE = 200


def crps_from_step_ensembles(ensembles, actual) -> float | None:
    """Moyenne du CRPS empirique pas par pas sur un walk-forward D1.

    `ensembles[i]` : nuage de valeurs simulées au pas i (bootstrap ou MC-Dropout).
    `actual[i]` : valeur réalisée au pas i.
    Retourne None si `ensembles` est vide/absent (modèle ou horizon sans support
    CRPS, ex. n_ensemble=0 ou Gate2 D7 — cf. model_artifacts/pipeline.py)."""
    if not ensembles:
        return None
    actual = np.asarray(actual, dtype=float)
    if len(ensembles) != len(actual):
        raise ValueError(
            f"crps_from_step_ensembles: {len(ensembles)} nuages pour {len(actual)} "
            "valeurs réalisées — doivent être alignés pas à pas."
        )
    scores = [crps_empirical(e, a) for e, a in zip(ensembles, actual)]
    return float(np.mean(scores))
