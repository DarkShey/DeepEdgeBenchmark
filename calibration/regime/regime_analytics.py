"""
regime_analytics.py — Analyses statistiques sur les historiques de régime DEITA

Fonctions pures consommant les DataFrames déjà produits par
RegimeAgent.predict_history() (colonnes : regime, p_calm, p_trending, p_stress,
vol_bucket, sigma_t, vol_of_vol, changepoint_prob, indexé par date). Aucune fonction
ne télécharge de données ni ne fait tourner le HMM/BOCPD.

Couvre les 3 demandes du tuteur (cf. BRIEF_dashboard_multiasset.md §0) :
  - largeur des régimes (segment_regimes / regime_width_stats)
  - moyenne des régimes à 4 échelles (zoom temporel + donut recalculé côté JS, cf.
    BRIEF_dashboard_v3_corrections.md — pas de fonction Python dédiée)
  - vol comme déclencheur de changement de régime (vol_spike_hit_rate — statistique descriptive
    simple, cf. note de prudence affichée dans le dashboard : pas de test de significativité)
  - vol comme déclencheur de corrélation inter-actifs (rolling_cross_correlation /
    stress_conditioned_correlation)
"""

import itertools

import numpy as np
import pandas as pd


# ── 4.1 Largeur des régimes ─────────────────────────────────────────────────────

def segment_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Découpe la colonne 'regime' en segments contigus.
    Retourne un DataFrame avec colonnes : regime, start, end, n_days_trading, n_days_calendar.

    IMPORTANT — point de rigueur : utiliser n_days_calendar = (end - start).days + 1 pour toute
    comparaison INTER-actifs, jamais n_days_trading. BTC/ETH cotent 7j/7, SPY/TLT cotent ~5j/7
    (hors jours fériés) : comparer des comptages de lignes fausserait la comparaison de largeur
    de régime entre crypto et actifs traditionnels. n_days_trading reste utile pour des stats
    intra-actif (ex. "durée moyenne d'un régime stress sur BTC seul").
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["regime", "start", "end", "n_days_trading", "n_days_calendar"])

    regimes = df["regime"].tolist()
    idx = df.index
    n = len(regimes)

    segments = []
    i = 0
    while i < n:
        j = i
        r = regimes[i]
        while j < n and regimes[j] == r:
            j += 1
        start, end = idx[i], idx[j - 1]
        n_days_trading = j - i
        n_days_calendar = (end - start).days + 1
        segments.append({
            "regime": r,
            "start": start,
            "end": end,
            "n_days_trading": n_days_trading,
            "n_days_calendar": n_days_calendar,
        })
        i = j

    return pd.DataFrame(segments)


def regime_width_stats(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Groupby(regime) sur n_days_calendar : count, mean, median, std, min, max.
    Une ligne par régime (calm/trending/stress).
    """
    stats = segments.groupby("regime")["n_days_calendar"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    return stats.reset_index()


# ── 4.3 Vol comme déclencheur de changement de régime ───────────────────────────

def vol_spike_hit_rate(df: pd.DataFrame, lookback: int = 3, quantile: float = 0.75) -> float:
    """
    % des changements de régime précédés (dans les `lookback` jours précédents) d'un sigma_t
    dépassant le quantile `quantile` de sa distribution glissante sur 60 jours.
    Retourne un float dans [0, 1].

    ATTENTION — statistique descriptive simple, pas un test de significativité : l'échantillon
    d'événements (changements de régime) est faible par actif (quelques dizaines sur plusieurs
    années), donc l'incertitude d'échantillonnage sur ce pourcentage est large et non quantifiée
    ici. De plus, le régime "stress" est en partie défini par sigma_t élevée dans RegimeHMM, donc
    une partie du lien mesuré est mécanique (par construction) plutôt que prédictive. À utiliser
    comme indication exploratoire pour le tuteur, pas comme signal de trading validé.
    """
    rolling_q = df["sigma_t"].rolling(60).quantile(quantile)
    spike = df["sigma_t"] > rolling_q

    regime_change = df["regime"] != df["regime"].shift(1)
    if len(regime_change) > 0:
        regime_change.iloc[0] = False

    spike_recent = spike.shift(1).rolling(lookback, min_periods=1).max().fillna(0).astype(bool)

    n_changes = int(regime_change.sum())
    if n_changes == 0:
        return 0.0

    hits = int((regime_change & spike_recent).sum())
    return hits / n_changes


# ── 4.4 Vol comme déclencheur de corrélation inter-actifs ───────────────────────

def rolling_cross_correlation(returns_by_asset: dict, window: int = 63) -> pd.DataFrame:
    """
    returns_by_asset : {ticker: pd.Series des rendements journaliers}, même index aligné (inner join
    sur les dates communes aux 4 actifs — nécessaire à cause du calendrier crypto vs actions/bonds).
    Calcule la corrélation glissante (fenêtre `window` jours, ex. 63 ≈ 1 trimestre boursier) pour
    chacune des 6 paires uniques parmi les 4 actifs.
    Retourne un DataFrame indexé par date, colonnes du type "BTC-ETH", "BTC-SPY", "SPY-TLT", etc.
    """
    keys = list(returns_by_asset.keys())
    aligned = pd.concat(returns_by_asset, axis=1, join="inner")
    aligned.columns = keys

    out = pd.DataFrame(index=aligned.index)
    for a, b in itertools.combinations(keys, 2):
        out[f"{a}-{b}"] = aligned[a].rolling(window).corr(aligned[b])
    return out


def market_mask_union(masks: dict) -> pd.Series:
    """Union booléenne (OR) d'un dict {ticker: pd.Series bool} aligné sur l'index commun."""
    aligned = pd.concat(masks, axis=1, join="inner")
    return aligned.any(axis=1)


def market_mask_intersection(masks: dict) -> pd.Series:
    """Intersection booléenne (AND) d'un dict {ticker: pd.Series bool} aligné sur l'index commun."""
    aligned = pd.concat(masks, axis=1, join="inner")
    return aligned.all(axis=1)


def segment_boolean_mask(mask: pd.Series) -> list:
    """
    Découpe une série booléenne en segments contigus où mask == True.
    Retourne une liste de dicts {start, end} (mêmes conventions que segment_regimes,
    mais sur un mask binaire plutôt qu'une colonne 'regime' à 3 valeurs).
    Utilisé pour dessiner le fond "stress marché" sur le graphique de corrélation glissante.
    """
    idx = mask.index
    values = mask.tolist()
    n = len(values)
    segments = []
    i = 0
    while i < n:
        if not values[i]:
            i += 1
            continue
        j = i
        while j < n and values[j]:
            j += 1
        segments.append({"start": idx[i], "end": idx[j - 1]})
        i = j
    return segments


def stress_conditioned_correlation(returns_by_asset: dict, stress_masks: dict, calm_masks: dict) -> dict:
    """
    stress_masks : {ticker: bool série, True si p_stress > 0.5}.
    calm_masks   : {ticker: bool série, True si regime == "calm"}.

    "stress" = au moins 1 actif sur 4 en stress (union stress_masks).
    "calm"   = LES 4 ACTIFS SIMULTANÉMENT en régime calme (intersection stricte calm_masks) —
               remplace l'ancienne définition "aucun stress" qui laissait passer les jours trending.

    Retourne {"stress": corr_matrix, "calm": corr_matrix, "stress_mask": pd.Series, "calm_mask": pd.Series}.
    """
    keys = list(returns_by_asset.keys())
    aligned = pd.concat(returns_by_asset, axis=1, join="inner")
    aligned.columns = keys

    union_stress = market_mask_union(stress_masks)
    intersection_calm = market_mask_intersection(calm_masks)

    common_idx = aligned.index.intersection(union_stress.index).intersection(intersection_calm.index)
    aligned = aligned.loc[common_idx]
    union_stress = union_stress.reindex(common_idx).fillna(False)
    intersection_calm = intersection_calm.reindex(common_idx).fillna(False)

    stress_corr = aligned.loc[union_stress].corr()
    calm_corr = aligned.loc[intersection_calm].corr()

    return {
        "stress": stress_corr,
        "calm": calm_corr,
        "stress_mask": union_stress,
        "calm_mask": intersection_calm,
    }
