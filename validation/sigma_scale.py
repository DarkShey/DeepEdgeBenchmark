"""
validation/sigma_scale.py — sigma_scale = sqrt(EWMA(z^2)), causal, depuis tracking.db.
=========================================================================================
Chantier 1b de BRIEF_branchement_prod_calibration_sigma.md.

`sarima_model.run_sarima`/`prophet_model.run_prophet` (adoptions du 2026-07-30,
cf. HANDOFF_sigma_calibration_suivi.md §5) calibrent leur correction multiplicative
sigma'_t = sigma_t * sqrt(EWMA(z_t^2)) depuis un état interne au backtest walk-forward
(les résidus in-sample qu'ils viennent de produire). La prévision LIVE n'a pas cet
historique in-sample : sa seule source de "prédictions déjà réalisées" est tracking.db
(les lignes business D+1 déjà écrites par de précédents runs, dont y_true est maintenant
connu). Ce module reproduit exactement le même état EWMA (lambda=0.94, seed=1.0) à partir
de CES lignes-là, pour nourrir `sigma_scale=` de benchmarks/multi_horizon.py.

Contrat de lecture identique à experiments/prob_kpi_common.load_matrix_rows : source IN
('oos', 'live'), daily_duplicate=0, y_true IS NOT NULL -- restreint en plus à
frequence='daily' AND horizon_type='daily' (seul grain que save_prediction écrit
aujourd'hui pour les modèles ARIMA-GARCH/SARIMA/Prophet/Naive/LSTM, cf.
model_artifacts/pipeline.py::_save_business_predictions) pour ne jamais mélanger un
horizon business daily avec un horizon weekly (TSDiff-W) qui partagerait la même valeur
entière de `horizon`.

Causalité stricte : seules les lignes dont `cutoff_date` est STRICTEMENT antérieure à
`as_of_cutoff_date` (la date de la NOUVELLE prévision à calibrer) entrent dans l'état
EWMA -- aucune ligne dont l'origine est postérieure ou égale ne peut fuiter dans le
calcul, quel que soit le moment où `y_true` a été résolu (cf. evaluate_pending).
"""

import sqlite3

import numpy as np

from validation import tracking_db as td

EWMA_LAMBDA = 0.94                     # RiskMetrics daily decay, même valeur que
                                        # sarima_model/prophet_model/naive_model/lstm_model.
_Z975 = 1.959963984540054              # norm.ppf(0.975) -- même convention que sarima_model,
                                        # prophet_model (sigma_own recouvré depuis les bornes).


def _load_causal_rows(model: str, asset: str, horizon: int, as_of_cutoff_date: str,
                      db_path: str) -> list:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            """
            SELECT cutoff_date, y_pred, y_lower, y_upper, y_true
            FROM predictions
            WHERE model = ? AND asset = ? AND horizon = ?
                  AND frequence = 'daily' AND horizon_type = 'daily'
                  AND source IN ('oos', 'live') AND daily_duplicate = 0
                  AND y_true IS NOT NULL
                  AND cutoff_date < ?
            ORDER BY cutoff_date ASC
            """,
            (model, asset, horizon, as_of_cutoff_date),
        )
        return cur.fetchall()
    finally:
        con.close()


def sigma_scale(model: str, asset: str, horizon: int, as_of_cutoff_date: str,
                db_path: str, lam: float = EWMA_LAMBDA) -> float:
    """sqrt(EWMA(z^2)) causal pour (model, asset, horizon), tel qu'observable juste
    AVANT `as_of_cutoff_date`.

    z_t = (y_true_t - y_pred_t) / sigma_own_t, sigma_own_t = (y_upper_t - y_lower_t) /
    (2*Z975) -- reconstruction du sigma propre au modèle depuis les bornes stockées,
    même convention que experiments/prob_kpi_common.sample_parametric. État initialisé
    à 1.0 ("faire confiance au sigma du modèle jusqu'à preuve du contraire", même
    convention que sarima_model.run_sarima / prophet_model.run_prophet) : retourne 1.0
    tel quel si aucune ligne réalisée n'existe encore (pas d'historique -> pas de
    correction, comportement neutre par construction, jamais une erreur).

    `td.init_db(db_path)` (paresseux et idempotent, même garde-fou que
    tracking_db.save_prediction) : un tout premier run, base encore inexistante ou sans
    la table `predictions`, ne doit pas lever une sqlite3.OperationalError ici -- juste
    ne rien trouver et retourner l'état neutre 1.0."""
    td.init_db(db_path)
    rows = _load_causal_rows(model, asset, horizon, as_of_cutoff_date, db_path)
    s2 = 1.0
    for _cutoff_date, y_pred, y_lower, y_upper, y_true in rows:
        sigma_own = max((y_upper - y_lower) / (2.0 * _Z975), 1e-12)
        z = (y_true - y_pred) / sigma_own
        s2 = lam * s2 + (1 - lam) * z * z
    return float(np.sqrt(s2))
