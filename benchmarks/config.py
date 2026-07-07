"""
config.py — Configuration du benchmark multi-actifs x multi-horizons
======================================================================
Source unique de vérité pour les actifs, la fenêtre de données, le split
train/validation et la grille d'horizons. Modifié à la main par l'utilisateur
selon les besoins (pas d'exposition dans l'UI interactive du programme).
"""

from calibration.regime.assets import ASSETS  # BTC-USD, ETH-USD, SPY, ZN=F, TLT

# ── Fenêtre de données ────────────────────────────────────────────────────────
DATA_START = "2018-01-01"
DATA_END = None   # None -> aujourd'hui (datetime.today())

# ── Split train/validation ───────────────────────────────────────────────────
TRAIN_VAL_SPLIT = 0.85   # 85% train / 15% validation, sur les données "effectives" (post-T)

# ── Paramètre T : jours à ignorer à la fin de la période de données ─────────
# Sert à simuler "aujourd'hui" T jours dans le passé, afin de valider les modèles à la
# main sans attendre le vrai futur : les T derniers jours réels téléchargés (déjà
# connus, mais volontairement mis de côté) servent de vérité terrain pour les horizons
# qui dépassent la validation (au-delà d'~15% des données effectives). Augmenter T
# permet de vérifier des horizons plus longs immédiatement. T=0 => mode "prod" (aucun
# futur caché, horizons au-delà de la validation restent "pending").
T = 30

# ── Grille d'horizons (en jours de TRADING, pas calendaires) ─────────────────
# J = jour de trading, S = semaine (5 jours de trading), M = mois (21 jours de trading).
HORIZONS = {
    "J+1": 1, "J+2": 2, "J+3": 3, "J+4": 4, "J+5": 5, "J+6": 6,
    "S+1": 5, "S+2": 10, "S+3": 15, "S+4": 20,
    "M+1": 21, "M+2": 42,
}
MAX_HORIZON_DAYS = max(HORIZONS.values())

DB_PATH = "benchmarks/benchmark_results.db"
