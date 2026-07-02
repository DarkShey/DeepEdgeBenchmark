"""
assets.py — Registre des actifs et événements pour le dashboard multi-actifs DEITA

Source unique de vérité pour la liste des actifs (BTC/ETH/SPY/ZN=F/TLT), la fenêtre de
données commune et les événements de marché affichés sur les graphiques. Consommé
par dashboard_builder.py et regime_agent.py (import, pas de duplication).
"""

# BRIEF_dashboard_v5_corrections.md : TLT (ETF, allocation variable/opaque) remplacé par ZN=F
# (futures CME sur le Treasury Note 10 ans) — le 10 ans est le benchmark mondial des taux et
# le titre du Trésor US le plus échangé.
# Données vérifiées avant bascule : 2139 jours depuis 2018, volume quasi jamais nul (0.37%).
# BRIEF_dashboard_v6_corrections.md §1 : TLT remis EN PLUS de ZN=F (pas à sa place) — les deux
# représentations du marché obligataire US (ETF et futures) cohabitent pour permettre de les
# comparer directement.
ASSETS = [
    {"ticker": "BTC-USD", "label": "Bitcoin",                 "short": "BTC", "asset_class": "crypto", "color": "#f7931a"},
    {"ticker": "ETH-USD", "label": "Ethereum",                 "short": "ETH", "asset_class": "crypto", "color": "#627eea"},
    {"ticker": "SPY",     "label": "S&P 500 (SPY)",            "short": "SPX", "asset_class": "index",  "color": "#2ecc71"},
    {"ticker": "ZN=F",    "label": "US Treasury 10Y Note Futures", "short": "ZN", "asset_class": "bond", "color": "#3498db"},
    {"ticker": "TLT",     "label": "US Treasury 20+Y (ETF)",   "short": "TLT", "asset_class": "bond",   "color": "#9b59b6"},
]

DATA_START = "2018-01-01"
DATA_END = None          # None → utiliser la date du jour (datetime.today())
TRAIN_END = "2023-12-31"

# Événements globaux : pertinents pour TOUS les actifs (macro, monétaire, géopolitique)
GLOBAL_EVENTS = {
    "2020-03-12": ("COVID crash",       "macro"),
    "2020-03-15": ("Fed taux 0%",       "monetaire"),
    "2022-01-05": ("Fed pivot hawkish", "monetaire"),
    "2022-02-24": ("Invasion Ukraine",  "geopolitique"),
    "2022-06-15": ("Fed +75bp",         "monetaire"),
    "2023-03-10": ("SVB faillite",      "macro"),
    "2023-07-26": ("Fed pic 5.25%",     "monetaire"),
    "2023-12-13": ("Fed pivot dovish",  "monetaire"),
}

# Événements spécifiques à un actif : affichés seulement sur l'onglet de cet actif
# (peuvent aussi être répliqués sur un autre actif si pertinent, ex. Merge ETH sur l'onglet ETH uniquement)
ASSET_EVENTS = {
    "BTC-USD": {
        "2017-11-29": ("BTC ATH $10k",     "crypto"),
        "2018-01-17": ("BTC ATH $20k",     "crypto"),
        "2018-12-15": ("BTC bas $3.2k",    "crypto"),
        "2020-05-11": ("BTC halving #3",   "crypto"),
        "2020-12-16": ("BTC franchit $20k","crypto"),
        "2021-02-08": ("Tesla 1.5G$ BTC",  "crypto"),
        "2021-09-07": ("El Salvador BTC",  "geopolitique"),
        "2021-11-10": ("BTC ATH $69k",     "crypto"),
        "2022-05-09": ("LUNA collapse",    "crypto"),
        "2022-11-08": ("FTX collapse",     "crypto"),
        "2024-01-10": ("BTC ETF spot",     "crypto"),
        "2024-03-14": ("BTC ATH $73k",     "crypto"),
        "2024-04-19": ("BTC halving #4",   "crypto"),
        "2025-01-23": ("BTC ATH $109k",    "crypto"),
    },
    "ETH-USD": {
        "2021-04-14": ("Coinbase IPO",     "crypto"),
        "2022-05-09": ("LUNA collapse",    "crypto"),
        "2022-09-15": ("ETH Merge PoS",    "crypto"),
        "2022-11-08": ("FTX collapse",     "crypto"),
        "2024-05-23": ("ETH ETF spot approuvé", "crypto"),
    },
    "SPY": {
        "2020-03-23": ("Plancher COVID S&P",   "macro"),
        "2022-10-12": ("Plancher bear 2022",   "macro"),
    },
    "ZN=F": {
        "2022-03-16": ("Début hausses de taux Fed",   "monetaire"),
        "2023-10-19": ("US 10Y touche ~5%",           "monetaire"),
    },
}

# TLT réutilise les mêmes événements que ZN=F : même sous-jacent macro (taux US).
ASSET_EVENTS["TLT"] = ASSET_EVENTS["ZN=F"]

_REGIME_BG = {
    "calm":  "rgba(46,204,113,0.24)",    # vert
    "bull":  "rgba(241,196,15,0.22)",    # ambre/or
    "bear":  "rgba(74,105,189,0.24)",    # bleu ardoise/indigo
    "stress":"rgba(231,76,60,0.26)",     # rouge
}
_REGIME_HEX = {
    "calm": "#2ecc71", "bull": "#f1c40f", "bear": "#4a69bd", "stress": "#e74c3c",
}

_EVENT_COLORS = {
    "crypto":       "#e67e22",  # orange (inchangé)
    "macro":        "#ff6ec7",  # rose/magenta (était #e74c3c -> collision avec stress)
    "monetaire":    "#2980b9",  # bleu (distinct de bull #f1c40f et bear #4a69bd)
    "geopolitique": "#8e44ad",  # violet (inchangé)
}

_REGIME_LABELS = {
    "calm": "Calme", "bull": "Haussier", "bear": "Baissier", "stress": "Stress",
}


def events_for_ticker(ticker: str) -> dict:
    """Fusionne GLOBAL_EVENTS + ASSET_EVENTS[ticker] pour l'affichage sur l'onglet de cet actif."""
    merged = dict(GLOBAL_EVENTS)
    merged.update(ASSET_EVENTS.get(ticker, {}))
    return merged
