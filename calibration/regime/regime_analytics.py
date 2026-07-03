"""
regime_analytics.py — Analyses statistiques sur les historiques de régime DEITA

Fonctions pures consommant les DataFrames déjà produits par
RegimeAgent.predict_history() (colonnes : regime, p_calm, p_bull, p_bear, p_stress,
vol_bucket, sigma_t, vol_of_vol, changepoint_prob, indexé par date). Aucune fonction
ne télécharge de données ni ne fait tourner le HMM/BOCPD.

Couvre les objectifs des séances du 02/07 (cf. BRIEF_dashboard_multiasset.md §0 et
BRIEF_dashboard_v6_corrections.md §3) :
  - largeur des régimes (segment_regimes / regime_width_stats)
  - moyenne des régimes à 4 échelles (zoom temporel + donut recalculé côté JS, cf.
    BRIEF_dashboard_v3_corrections.md — pas de fonction Python dédiée)
  - vol comme signal avancé d'un changement de régime (regime_transition_vol_profile —
    étude d'événement : profil moyen de sigma_t autour des transitions de régime, remplace
    l'ancienne corrélation décalée vol/régime trop agrégée, cf. BRIEF v6 §3)
  - vol comme déclencheur de corrélation inter-actifs (rolling_cross_correlation /
    pairwise_stress_calm_correlation), avec test de significativité (fisher_r_critical /
    correlation_significance) pour distinguer un vrai effet du bruit d'échantillonnage. Chaque
    paire est conditionnée sur son propre sous-échantillon stress/calme (cf.
    BRIEF_dashboard_v9_corrections.md §2) — pas sur un mask global exigeant une condition
    simultanée sur tous les actifs du jeu de données, qui écrase artificiellement l'échantillon.
  - test formel de causalité (granger_causality_vol_to_stress / granger_causality_volume_to_stress) :
    la volatilité, puis le volume, passés aident-ils réellement à prédire le régime stress futur ?
    Contrairement à regime_transition_vol_profile (descriptif, "indicatif"), c'est un vrai test
    statistique (ADF + Granger + correction pour comparaisons multiples côté appelant).
  - entre volatilité et volume, lequel bouge en premier EN GÉNÉRAL, pas seulement autour des
    débuts de régime (lead_lag_cross_correlation) : corrélation croisée ±N jours sur les
    variations journalières, question distincte de granger_causality_* (celles-ci sont
    conditionnées aux transitions vers le régime stress ; celle-ci porte sur tout l'historique).

ensure_stationary (test ADF + différenciation conditionnelle) et fisher_r_critical_bonferroni
(seuil de Fisher corrigé pour comparaisons multiples) sont les deux prérequis statistiques
partagés par granger_causality_*_to_stress et lead_lag_cross_correlation — tout test qui scanne
plusieurs lags sur les mêmes données doit vérifier la stationnarité en amont et corriger son
seuil de significativité, sous peine de p-values/corrélations non interprétables.
"""

import itertools
import math

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


# ── 4.1 Largeur des régimes ─────────────────────────────────────────────────────

def segment_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Découpe la colonne 'regime' en segments contigus.
    Retourne un DataFrame avec colonnes : regime, start, end, n_days_trading, n_days_calendar.

    IMPORTANT — point de rigueur : utiliser n_days_calendar = (end - start).days + 1 pour toute
    comparaison INTER-actifs, jamais n_days_trading. BTC/ETH cotent 7j/7, SPY/ZN=F cotent ~5j/7
    (marchés fermés le week-end) : comparer des comptages de lignes fausserait la comparaison de
    largeur de régime entre crypto et actifs traditionnels. n_days_trading reste utile pour des
    stats intra-actif (ex. "durée moyenne d'un régime stress sur BTC seul").
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
    Une ligne par régime (calm/bull/bear/stress).
    """
    stats = segments.groupby("regime")["n_days_calendar"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    return stats.reset_index()


# ── 4.3 Vol comme signal avancé d'un changement de régime (étude d'événement) ───

def regime_transition_vol_profile(df: pd.DataFrame, window: int = 10, alignment: str = "start",
                                   only_into: str | None = None, column: str = "sigma_t") -> pd.DataFrame:
    """
    Étude d'événement : profil moyen d'une colonne (sigma_t par défaut) autour des transitions
    de régime.

    column : nom de la colonne de df à profiler (ex. "sigma_t" pour la volatilité, "volume_norm"
    pour le volume normalisé). N'affecte que la source des données ; les colonnes de sortie
    restent nommées mean_sigma/std_sigma quel que soit `column`, pour limiter le risque de
    régression sur les appelants existants.
    alignment : "start" (jour 0 = premier jour du nouveau régime) ou "end" (jour 0 = dernier
    jour de l'ancien régime, juste avant la transition).
    only_into : si fourni (ex. "stress"), ne considère que les transitions VERS ce régime
    (ex. "à quoi ressemble la vol juste avant qu'on bascule en stress ?"). Si None, toutes les
    transitions sont poolées ensemble. Pour alignment="start", c'est le régime du segment qui
    COMMENCE à l'événement qui est testé ; pour alignment="end", c'est le régime du segment
    SUIVANT (celui vers lequel on transitionne après la fin du segment courant).
    window : nombre de jours de trading de part et d'autre de l'événement.

    Pour chaque segment de segment_regimes (sauf le premier segment de l'historique en
    alignment="start", ou le dernier en alignment="end" — ils n'ont pas d'événement "avant"/
    "après" dans les données), extraire la colonne sur [event_pos - window, event_pos + window]
    en position entière (iloc), aligné sur un axe rel_day = -window..+window. Si la fenêtre déborde
    des bornes du DataFrame, le segment est ignoré (pas de padding artificiel). Les événements
    retenus sont ensuite empilés et moyennés (et écart-type) colonne par colonne (= jour relatif
    par jour relatif) pour obtenir le profil moyen.

    Retourne un DataFrame [rel_day, mean_sigma, std_sigma, n_events].
    """
    if alignment not in ("start", "end"):
        raise ValueError(f"alignment doit être 'start' ou 'end', got {alignment!r}")

    rel_day = np.arange(-window, window + 1)
    segments = segment_regimes(df)
    n_segments = len(segments)

    if n_segments < 2:
        return pd.DataFrame({"rel_day": rel_day, "mean_sigma": np.nan, "std_sigma": np.nan, "n_events": 0})

    if alignment == "start":
        # le tout premier segment n'a pas de "avant" dans les données -> exclu
        candidate_positions = range(1, n_segments)
        event_dates = segments["start"]
        regime_at_event = segments["regime"]
    else:
        # le tout dernier segment n'a pas de "après" dans les données -> exclu
        candidate_positions = range(0, n_segments - 1)
        event_dates = segments["end"]
        regime_at_event = segments["regime"].shift(-1)  # régime du segment SUIVANT

    sigma = df[column]
    profiles = []
    for i in candidate_positions:
        if only_into is not None and regime_at_event.iloc[i] != only_into:
            continue
        pos = df.index.get_loc(event_dates.iloc[i])
        lo, hi = pos - window, pos + window
        if lo < 0 or hi >= len(df):
            continue
        profiles.append(sigma.iloc[lo:hi + 1].to_numpy())

    if not profiles:
        return pd.DataFrame({"rel_day": rel_day, "mean_sigma": np.nan, "std_sigma": np.nan, "n_events": 0})

    stacked = np.stack(profiles, axis=0)
    return pd.DataFrame({
        "rel_day": rel_day,
        "mean_sigma": stacked.mean(axis=0),
        "std_sigma": stacked.std(axis=0),
        "n_events": stacked.shape[0],
    })


# ── 4.4 Vol comme déclencheur de corrélation inter-actifs ───────────────────────

def rolling_cross_correlation(returns_by_asset: dict, window: int = 63) -> pd.DataFrame:
    """
    returns_by_asset : {ticker: pd.Series des rendements journaliers}, même index aligné (inner join
    sur les dates communes aux 4 actifs — nécessaire à cause du calendrier crypto vs actions/bonds).
    Calcule la corrélation glissante (fenêtre `window` jours, ex. 63 ≈ 1 trimestre boursier) pour
    chacune des 6 paires uniques parmi les 4 actifs.
    Retourne un DataFrame indexé par date, colonnes du type "BTC-ETH", "BTC-SPY", "SPY-ZN", etc.
    """
    keys = list(returns_by_asset.keys())
    aligned = pd.concat(returns_by_asset, axis=1, join="inner")
    aligned.columns = keys

    out = pd.DataFrame(index=aligned.index)
    for a, b in itertools.combinations(keys, 2):
        out[f"{a}-{b}"] = aligned[a].rolling(window).corr(aligned[b])
    return out


def lead_lag_cross_correlation(series_a: pd.Series, series_b: pd.Series, max_lag: int = 5) -> dict:
    """
    Corrélation croisée entre deux séries (typiquement des VARIATIONS journalières déjà
    différenciées par l'appelant — sigma_t et volume_norm sont chacune autocorrélées à leur
    propre niveau, ce qui gonflerait artificiellement toute corrélation calculée sur les niveaux
    bruts), sur un éventail de décalages -max_lag à +max_lag jours de trading.

    Vérifie néanmoins la stationnarité de chaque série reçue (ADF, cf. ensure_stationary) et la
    différencie une seconde fois si elle ne l'est toujours pas — un filet de sécurité, pas le
    mécanisme principal : la différenciation "officielle" (variations journalières) reste la
    responsabilité de l'appelant, exigée par la question posée (cf. ci-dessous), pas seulement
    par la stationnarité. Sans cette vérification, les corrélations calculées sur des séries non
    stationnaires ne seraient pas interprétables (corrélation fallacieuse).

    Question à laquelle ça répond : entre deux signaux (typiquement vol et volume), lequel
    bouge en premier EN GÉNÉRAL — pas seulement autour d'un événement particulier (contrairement
    à regime_transition_vol_profile / granger_causality_*_to_stress, qui conditionnent sur les
    transitions vers le régime stress) ?

    Convention de signe (à ne pas confondre) :
      - lag > 0 : corrèle series_a(t) avec series_b(t - lag), c'est-à-dire series_b DÉCALÉE DANS
        LE PASSÉ. Une corrélation élevée ici signifie que series_b PASSÉE est liée à series_a
        AUJOURD'HUI -> series_b précède (mène) series_a.
      - lag < 0 : corrèle series_a(t) avec series_b(t + |lag|), c'est-à-dire series_b DÉCALÉE DANS
        LE FUTUR. Une corrélation élevée ici signifie que series_a AUJOURD'HUI est liée à
        series_b FUTURE -> series_a précède (mène) series_b.
      - lag = 0 : corrélation contemporaine (même jour), généralement la plus forte des deux
        séries (fait stylisé bien connu vol/volume) — ne pas la confondre avec un effet de
        décalage réel.

    Retourne {"ccf": DataFrame[lag, corr, n], "adf_a_p": float, "a_differenced": bool,
    "adf_b_p": float, "b_differenced": bool}. n (par lag) sert à calculer un seuil de
    significativité avec fisher_r_critical(n) ou, si plusieurs lags sont scannés (comme ici),
    fisher_r_critical_bonferroni(n, n_tests=2*max_lag+1) — cf. appelant.
    """
    a, adf_a_p, a_differenced = ensure_stationary(series_a)
    b, adf_b_p, b_differenced = ensure_stationary(series_b)

    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        b_shifted = b.shift(lag)
        valid = pd.concat([a, b_shifted], axis=1).dropna()
        n = len(valid)
        corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1])) if n > 3 else float("nan")
        rows.append({"lag": lag, "corr": corr, "n": n})

    return {
        "ccf": pd.DataFrame(rows),
        "adf_a_p": adf_a_p, "a_differenced": bool(a_differenced),
        "adf_b_p": adf_b_p, "b_differenced": bool(b_differenced),
    }


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


def pairwise_stress_calm_correlation(returns_by_asset: dict, stress_masks: dict, calm_masks: dict) -> pd.DataFrame:
    """
    Pour chaque paire (a, b), corrélation de Pearson des rendements conditionnée sur deux
    sous-échantillons SPÉCIFIQUES À LA PAIRE (pas un mask global sur tous les actifs du jeu de
    données) :
      - stress_pair = stress_masks[a] OR stress_masks[b]   (au moins l'un des deux stressé)
      - calm_pair   = calm_masks[a] AND calm_masks[b]       (les deux simultanément calmes)

    Justification : exiger qu'un actif sans rapport avec la paire testée (ex. TLT) soit lui aussi
    calme pour évaluer la corrélation BTC-SPX n'a pas de sens et réduit artificiellement
    l'échantillon (constaté : 59 jours sur 2134 avec la définition globale à 5 actifs). Chaque
    paire a maintenant son propre n_stress/n_calm et donc son propre seuil de significativité
    (fisher_r_critical) — les tailles d'échantillon diffèrent légitimement d'une paire à l'autre.

    Retourne un DataFrame, une ligne par paire :
    [pair, corr_stress, n_stress, r_crit_stress, stress_sig,
           corr_calm, n_calm, r_crit_calm, calm_sig].
    Si n_stress ou n_calm <= 3, la corrélation correspondante est NaN (échantillon insuffisant,
    cf. fisher_r_critical qui retourne déjà None dans ce cas).
    """
    keys = list(returns_by_asset.keys())
    rows = []
    for a, b in itertools.combinations(keys, 2):
        ra_, rb_ = returns_by_asset[a], returns_by_asset[b]
        common = ra_.index.intersection(rb_.index)
        ra_, rb_ = ra_.loc[common], rb_.loc[common]

        sa = stress_masks[a].reindex(common).fillna(False)
        sb = stress_masks[b].reindex(common).fillna(False)
        ca = calm_masks[a].reindex(common).fillna(False)
        cb = calm_masks[b].reindex(common).fillna(False)

        stress_pair = sa | sb
        calm_pair = ca & cb

        n_stress = int(stress_pair.sum())
        n_calm = int(calm_pair.sum())

        corr_stress = float(ra_.loc[stress_pair].corr(rb_.loc[stress_pair])) if n_stress > 3 else float("nan")
        corr_calm = float(ra_.loc[calm_pair].corr(rb_.loc[calm_pair])) if n_calm > 3 else float("nan")

        r_crit_s = fisher_r_critical(n_stress)
        r_crit_c = fisher_r_critical(n_calm)

        rows.append({
            "pair": f"{a}-{b}",
            "corr_stress": corr_stress, "n_stress": n_stress, "r_crit_stress": r_crit_s,
            "stress_sig": bool(r_crit_s is not None and corr_stress == corr_stress and abs(corr_stress) > r_crit_s),
            "corr_calm": corr_calm, "n_calm": n_calm, "r_crit_calm": r_crit_c,
            "calm_sig": bool(r_crit_c is not None and corr_calm == corr_calm and abs(corr_calm) > r_crit_c),
        })
    return pd.DataFrame(rows)


# ── 4.5 Significativité statistique des corrélations ────────────────────────────

def fisher_r_critical(n: int, z_crit: float = 1.959964) -> float | None:
    """
    Seuil critique |r| au-delà duquel une corrélation de Pearson calculée sur n observations
    est significativement différente de 0 (test bilatéral, transformation de Fisher).
    Retourne None si n <= 3 (transformation non définie, échantillon trop petit pour tester).
    """
    if n <= 3:
        return None
    return math.tanh(z_crit / math.sqrt(n - 3))


def correlation_significance(r: float, n: int, z_crit: float = 1.959964) -> dict:
    """
    Teste si r (calculé sur n observations) est significativement différent de 0.
    Retourne {"r_crit": float | None, "significant": bool, "n": int}.
    """
    r_crit = fisher_r_critical(n, z_crit)
    if r_crit is None:
        return {"r_crit": None, "significant": False, "n": n}
    return {"r_crit": r_crit, "significant": bool(abs(r) > r_crit), "n": n}


def fisher_r_critical_bonferroni(n: int, n_tests: int, alpha: float = 0.05) -> float | None:
    """
    Comme fisher_r_critical, mais avec le seuil alpha corrigé pour comparaisons multiples
    (Bonferroni : alpha / n_tests) — à utiliser quand plusieurs tests sont scannés sur les
    mêmes données (ex. lead_lag_cross_correlation sur 2*max_lag+1 lags), pour éviter de conclure
    à la significativité d'un pic isolé dû uniquement au nombre de tests effectués (le même
    principe déjà appliqué à granger_causality_*_to_stress sur ses lags, cf. appelant côté
    dashboard_builder.py — Bonferroni y est appliqué directement sur les p-values, ici il faut
    d'abord convertir le alpha corrigé en seuil |r| via la transformation de Fisher).
    """
    from scipy.stats import norm
    z_crit = float(norm.ppf(1 - (alpha / n_tests) / 2))
    return fisher_r_critical(n, z_crit)


def ensure_stationary(series: pd.Series) -> tuple[pd.Series, float, bool]:
    """
    Vérifie la stationnarité d'une série (test ADF, Augmented Dickey-Fuller) et la différencie
    (diff().dropna()) si l'ADF ne rejette pas l'hypothèse de racine unitaire (p > 0.05).

    Prérequis partagé par tout test qui suppose des séries stationnaires — sans ça, les p-values
    (Granger) ou les corrélations (lead_lag_cross_correlation) calculées sur des séries non
    stationnaires ne sont pas interprétables ("spurious regression"/corrélation fallacieuse).

    Retourne (série_résultante, p_value_ADF_de_la_série_ORIGINALE, a_été_différenciée).
    """
    p_value = float(adfuller(series.dropna(), autolag="AIC")[1])
    differenced = p_value > 0.05
    out = series.diff().dropna() if differenced else series
    return out, p_value, differenced


# ── 4.6 Test formel : un signal passé (vol ou volume) cause-t-il (au sens de Granger) le régime stress ? ──

def _granger_causality_column_to_stress(df: pd.DataFrame, column: str, maxlag: int) -> dict:
    """
    Implémentation générique du test formel de causalité de Granger : la colonne `column`
    (sigma_t pour la Q1 vol, volume_norm pour la Q2 volume) retardée aide-t-elle à prédire
    p_stress aujourd'hui, au-delà de ce que p_stress explique déjà de lui-même (persistance) ?

    Contrairement à regime_transition_vol_profile (profil descriptif, "indicatif, pas un test
    statistique formel"), ceci EST un test statistique formel, avec ses prérequis respectés :

    1. Test ADF (Augmented Dickey-Fuller) de stationnarité sur `column` et p_stress — un test de
       Granger sur des séries non stationnaires produit des p-values non interprétables
       (régression fallacieuse, "spurious regression").
    2. Différenciation (diff().dropna()) de chaque série dont l'ADF ne rejette pas l'hypothèse de
       racine unitaire (p > 0.05).
    3. Test F de causalité de Granger (statsmodels.tsa.stattools.grangercausalitytests) aux lags
       1..maxlag, sur les séries stationnaires (différenciées si besoin).

    Ne teste QUE le sens `column` -> stress (économiquement significatif). Le sens inverse
    (stress -> `column`) ne l'est pas : p_stress est en partie dérivé de sigma_t (et donc
    indirectement de volume_norm, corrélé) par le HMM (RegimeHMM._assign_regime_labels), donc
    régime persistant et signal élevé co-varient mécaniquement par construction — un test dans
    ce sens ne prouverait rien de causal.

    Retourne {"adf_source_p": float, "source_differenced": bool, "adf_pstress_p": float,
    "pstress_differenced": bool, "p_values": {lag: p_value}, "n_obs": int}.

    Note de rigueur : les p-values à chaque lag ne sont PAS indépendantes (tester 10 lags sur les
    mêmes données), donc appliquer une correction pour comparaisons multiples (ex. Bonferroni,
    seuil = 0.05 / maxlag) avant de conclure à la significativité d'un lag précis — cf. appelant.
    """
    data = df[[column, "p_stress"]].dropna()

    source, adf_source_p, source_differenced = ensure_stationary(data[column])
    pstress, adf_pstress_p, pstress_differenced = ensure_stationary(data["p_stress"])

    common = source.index.intersection(pstress.index)
    gc_df = pd.DataFrame({"pstress": pstress.loc[common], "source": source.loc[common]}).dropna()

    gc_result = grangercausalitytests(gc_df[["pstress", "source"]], maxlag=maxlag, verbose=False)
    p_values = {lag: float(gc_result[lag][0]["ssr_ftest"][1]) for lag in range(1, maxlag + 1)}

    return {
        "adf_source_p": adf_source_p,
        "source_differenced": bool(source_differenced),
        "adf_pstress_p": adf_pstress_p,
        "pstress_differenced": bool(pstress_differenced),
        "p_values": p_values,
        "n_obs": len(gc_df),
    }


def granger_causality_vol_to_stress(df: pd.DataFrame, maxlag: int = 10) -> dict:
    """
    Question 1 : la volatilité (sigma_t) passée cause-t-elle (au sens de Granger) le régime
    stress futur ? Cf. _granger_causality_column_to_stress pour la méthode complète (ADF +
    Granger, avec différenciation si nécessaire).
    """
    return _granger_causality_column_to_stress(df, "sigma_t", maxlag)


def granger_causality_volume_to_stress(df: pd.DataFrame, maxlag: int = 10) -> dict:
    """
    Question 2 : le volume normalisé (volume_norm) passé cause-t-il (au sens de Granger) le
    régime stress futur ? Même méthode exactement que granger_causality_vol_to_stress
    (cf. _granger_causality_column_to_stress), appliquée à volume_norm plutôt qu'à sigma_t —
    pour savoir si le volume est un signal précurseur utile, indépendamment de la volatilité.
    """
    return _granger_causality_column_to_stress(df, "volume_norm", maxlag)
