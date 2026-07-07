"""
verdict_rules.py — Règles par défaut pour verdict_integrite / verdict_plausibilite
=======================================================================================
`tracking_db.save_prediction()` persiste un record déjà verdicté ; ce module fournit
les deux fonctions qui calculent ces verdicts AVANT l'appel à save_prediction(). Deux
familles de checks, volontairement indépendantes :

  - intégrité   : la prédiction est-elle structurellement saine (pas de NaN, bornes
                  cohérentes, dates cohérentes) ? Un échec ici signale un bug/crash
                  du pipeline, PAS un mauvais modèle.
  - plausibilité : le mouvement prédit est-il raisonnable pour cet actif et cet
                  horizon ? Un échec ici signale une prédiction délirante (mais
                  techniquement valide) — utile pour attraper un modèle mal calibré
                  bien avant que le vrai futur ne soit connu.

Seuils par défaut, à ajuster : mouvement journalier "normal" par classe d'actif,
étendu en sqrt(horizon en jours de trading) — même convention que naive_model.py
et regime_overlay.py (élargissement en racine du temps).
"""

import math
from datetime import date

VALID_REGIMES = {"calm", "bull", "bear", "stress", "unknown"}

# Mouvement journalier "normal" par classe d'actif (cf. calibration/regime/assets.py
# `asset_class`). Ce sont des ordres de grandeur de vol quotidienne réaliste, pas des
# bornes de risque extrême — le but est d'attraper une prédiction délirante (ex. +80%
# en 1 jour sur un ETF obligataire), pas de juger la précision du modèle.
ASSET_MAX_DAILY_MOVE = {
    "crypto": 0.15,
    "index": 0.05,
    "bond": 0.03,
}
DEFAULT_MAX_DAILY_MOVE = 0.10  # classe d'actif inconnue -> fallback prudent

# Largeur d'IC 95% jugée "délirante" si elle dépasse ce multiple du mouvement max
# attendu sur l'horizon (attrape les GARCH/LSTM qui explosent, pas les IC normalement
# larges des horizons longs).
PI_WIDTH_MAX_MULTIPLE = 8.0


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_integrity(record: dict) -> int:
    """1 si la prédiction est structurellement saine, 0 sinon. Ne juge jamais la
    qualité du modèle — seulement l'absence de valeurs cassées/incohérentes."""
    last_close = record.get("last_close")
    y_pred = record.get("y_pred")
    y_lower = record.get("y_lower")
    y_upper = record.get("y_upper")

    if not all(_is_finite(v) for v in (last_close, y_pred, y_lower, y_upper)):
        return 0
    if last_close <= 0 or y_lower <= 0:
        return 0
    if not (y_lower <= y_pred <= y_upper):
        return 0
    if y_lower >= y_upper:
        return 0
    if record.get("horizon", 0) <= 0:
        return 0
    if record.get("regime") not in VALID_REGIMES:
        return 0
    try:
        cutoff = date.fromisoformat(str(record["cutoff_date"]))
        target = date.fromisoformat(str(record["target_date"]))
    except (KeyError, ValueError):
        return 0
    if target <= cutoff:
        return 0
    return 1


def check_plausibility(record: dict, asset_class: str, horizon_trading_days: int) -> int:
    """1 si le mouvement prédit (et la largeur de l'IC) reste dans un ordre de
    grandeur raisonnable pour cette classe d'actif et cet horizon, 0 sinon.
    Ne s'appelle qu'après check_integrity==1 (suppose des valeurs déjà saines)."""
    last_close = record["last_close"]
    y_pred = record["y_pred"]
    y_lower = record["y_lower"]
    y_upper = record["y_upper"]

    max_daily = ASSET_MAX_DAILY_MOVE.get(asset_class, DEFAULT_MAX_DAILY_MOVE)
    max_move = max_daily * math.sqrt(max(horizon_trading_days, 1))

    move_pct = abs(y_pred / last_close - 1.0)
    if move_pct > max_move:
        return 0

    width_pct = (y_upper - y_lower) / last_close
    if width_pct > PI_WIDTH_MAX_MULTIPLE * max_move:
        return 0

    return 1
