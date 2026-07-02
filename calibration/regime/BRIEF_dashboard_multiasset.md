# Brief d'implémentation — Dashboard Multi-Actifs & Analyses de Régime

**Projet :** DEITA — Moteur de Régime
**Contexte :** suite du travail livré dans `calibration/regime/` (RegimeHMM + RegimeBOCPD + RegimeAgent,
7/7 tests verts). Ce brief couvre les 3 points demandés par le tuteur pour la séance du jour, plus la
refonte du dashboard HTML demandée par Maéva pour les présenter proprement.
**Ce fichier est une spécification, pas du code** — à donner tel quel à Claude Code pour exécution.
**Fichiers existants à ne PAS modifier** (déjà testés, interface figée) : `regime_state.py`,
`regime_hmm.py`, `regime_bocpd.py`, `test_regime_agent.py`. Tout le nouveau travail se fait dans de
nouveaux fichiers + une refonte ciblée de `regime_agent.py` (uniquement la partie génération HTML,
qui n'est couverte par aucun test).

---

## 0. Les 3 demandes du tuteur (reformulées et opérationnalisées)

| # | Demande brute | Interprétation opérationnelle |
|---|---|---|
| 1 | Vol comme déclencheur de changement de régime et/ou corrélation | (a) La volatilité (σₜ GARCH) précède-t-elle statistiquement un changement de régime ? → corrélation décalée (lead-lag). (b) La corrélation entre actifs augmente-t-elle quand la volatilité/le stress augmente (contagion) ? → corrélation glissante inter-actifs superposée aux régimes. |
| 2 | Largeur des régimes (3), comparaison Crypto/Index/US Bond/ETH | Mesurer la durée (largeur temporelle) de chaque épisode `calm`/`trending`/`stress` par actif, comparer les distributions entre les 4 actifs. |
| 3 | Graphes "moyenne des régimes" sur 4 échelles (full, année, trimestre, mois) | Pour chaque actif, moyenne des probabilités de régime (`p_calm`, `p_trending`, `p_stress`) agrégée par période, avec un sélecteur d'échelle. |

---

## 1. Actifs à intégrer

Le dashboard actuel ne couvre que BTC-USD. Il faut ajouter 3 actifs en gardant **exactement la même
logique** (RegimeHMM + RegimeBOCPD, mêmes hyperparamètres, aucune modification du moteur).

| Actif demandé | Ticker Yahoo Finance retenu | Classe | Justification |
|---|---|---|---|
| Bitcoin (déjà en place) | `BTC-USD` | crypto | inchangé |
| ETH | `ETH-USD` | crypto | Le tuteur a dit "ETH", pas "ETF" — ce sont deux choses différentes (voir encadré ci-dessous). Dans une liste Crypto/Index/US Bond/ETH, ETH = Ethereum, la 2ᵉ crypto par capitalisation, cohérent avec BTC. |
| Un indice actions (S&P500) | `SPY` (ETF qui réplique le S&P 500) | index | `SPY` a un historique Volume propre et fiable sur Yahoo (contrairement à `^GSPC` dont le volume est moins standardisé), et c'est déjà le ticker par défaut de `run_benchmark.py` dans ce repo — cohérence avec l'existant. |
| Un exemple pertinent d'obligation US | `TLT` (iShares 20+ Year Treasury Bond ETF) | bond | Les obligations longue duration sont celles qui montrent le plus clairement des régimes de marché distincts (calme pendant les périodes de taux stables, stress/trending pendant les cycles de hausse de taux type 2022). Une obligation courte durée (ex. `SHY`) serait trop plate pour que le HMM détecte quoi que ce soit d'intéressant. Alternative de repli si `TLT` pose un problème de données : `IEF` (7-10 ans). |

> **ETH vs ETF — pour clarifier une fois pour toutes :**
> - **ETH** = Ethereum, une cryptomonnaie (comme Bitcoin). Elle a son propre prix, son propre marché, disponible sur Yahoo Finance sous le ticker `ETH-USD`.
> - **ETF** (*Exchange-Traded Fund*, "fonds indiciel coté") = un type de produit financier qui réplique la performance d'un panier d'actifs (un indice, un secteur, une classe d'actifs) et se négocie en bourse comme une action. `SPY` et `TLT` **sont** des ETF (l'un réplique le S&P 500, l'autre un panier d'obligations d'État américaines). Ethereum n'est pas un ETF.
> - Donc dans ce brief : ETH = la cryptomonnaie Ethereum. SPY et TLT = les ETF utilisés comme proxys pour "l'indice actions" et "l'obligation US".

**Fenêtre de données commune** : `BTC-USD` a un historique bien plus long que `ETH-USD` (qui n'a des
volumes fiables sur Yahoo qu'à partir de fin 2017). Pour que la comparaison inter-actifs (point 2 et
analyses de corrélation du point 1) soit rigoureuse, **utiliser la même fenêtre de dates pour les 4
actifs** :
- `DATA_START = "2018-01-01"`
- `DATA_END = today` (ou `"2025-01-01"` si on veut figer comme l'existant — à décider selon la fraîcheur voulue)
- `TRAIN_END = "2023-12-31"` (identique à l'existant, cohérence avec les tests déjà écrits sur BTC)

---

## 2. Architecture cible

```
DeepEdgeBenchmark/
└── calibration/
    └── regime/
        ├── regime_state.py         (existant, INCHANGÉ)
        ├── regime_hmm.py           (existant, INCHANGÉ)
        ├── regime_bocpd.py         (existant, INCHANGÉ)
        ├── regime_agent.py         (existant — garder generate_html() pour compat, ne pas casser)
        ├── test_regime_agent.py    (existant, INCHANGÉ)
        ├── assets.py               ← NOUVEAU : registre des 4 actifs + événements
        ├── regime_analytics.py     ← NOUVEAU : largeur des régimes, échelles, vol-trigger, corrélation
        ├── dashboard_builder.py    ← NOUVEAU : orchestration multi-actifs + génération HTML combiné
        ├── test_regime_analytics.py← NOUVEAU : tests unitaires des fonctions d'analytics
        └── output/
            ├── regime.html                (existant, single-asset BTC — laissé tel quel)
            └── regime_dashboard.html      ← NOUVEAU : livrable final, 4 onglets + comparaison
```

**Principe directeur** : `RegimeHMM`, `RegimeBOCPD`, `RegimeAgent.fit()` / `RegimeAgent.predict()` sont
déjà génériques (ils prennent n'importe quel DataFrame OHLCV en entrée, aucun code n'est hardcodé pour
BTC). Il n'y a donc **rien à changer dans le moteur de régime lui-même** — on l'appelle 4 fois, une par
actif. Tout le travail est dans : (1) le registre d'actifs, (2) les nouvelles analyses, (3) la
génération HTML multi-onglets.

---

## 3. Fichier `assets.py` (nouveau)

```python
ASSETS = [
    {"ticker": "BTC-USD", "label": "Bitcoin",        "short": "BTC", "asset_class": "crypto", "color": "#f7931a"},
    {"ticker": "ETH-USD", "label": "Ethereum",        "short": "ETH", "asset_class": "crypto", "color": "#627eea"},
    {"ticker": "SPY",     "label": "S&P 500 (SPY)",   "short": "SPX", "asset_class": "index",  "color": "#2ecc71"},
    {"ticker": "TLT",     "label": "US Treasury 20+Y","short": "TLT", "asset_class": "bond",   "color": "#3498db"},
]

DATA_START = "2018-01-01"
DATA_END   = None          # None → utiliser la date du jour (datetime.today())
TRAIN_END  = "2023-12-31"

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
    "TLT": {
        "2022-03-16": ("Début hausses de taux Fed",   "monetaire"),
        "2023-10-19": ("US 10Y touche ~5%",           "monetaire"),
    },
}

_EVENT_COLORS = {
    "crypto":       "#e67e22",
    "macro":        "#e74c3c",
    "monetaire":    "#2980b9",
    "geopolitique": "#8e44ad",
}

_REGIME_BG   = {"calm": "rgba(39,174,96,0.18)", "trending": "rgba(41,128,185,0.18)", "stress": "rgba(231,76,60,0.22)"}
_REGIME_HEX  = {"calm": "#27ae60", "trending": "#2980b9", "stress": "#e74c3c"}
_REGIME_LABELS = {"calm": "Calme", "trending": "Tendanciel", "stress": "Stress"}


def events_for_ticker(ticker: str) -> dict:
    """Fusionne GLOBAL_EVENTS + ASSET_EVENTS[ticker] pour l'affichage sur l'onglet de cet actif."""
    merged = dict(GLOBAL_EVENTS)
    merged.update(ASSET_EVENTS.get(ticker, {}))
    return merged
```

Ces constantes remplacent celles actuellement définies en tête de `regime_agent.py`
(`MARKET_EVENTS`, `_EVENT_COLORS`, `_REGIME_BG`, `_REGIME_HEX`, `_REGIME_LABELS`). `regime_agent.py`
peut soit les importer depuis `assets.py`, soit les garder en doublon pour ne rien casser — préférer
l'import pour éviter la duplication.

---

## 4. Fichier `regime_analytics.py` (nouveau)

Toutes les fonctions prennent en entrée le DataFrame produit par `RegimeAgent._predict_history(prices)`
(colonnes : `regime`, `p_calm`, `p_trending`, `p_stress`, `vol_bucket`, `sigma_t`, `vol_of_vol`,
`changepoint_prob`, indexé par date). Aucune fonction ne doit re-télécharger de données ou refaire
tourner le HMM — elles consomment uniquement les DataFrames déjà calculés.

### 4.1 Largeur des régimes (point 2)

```python
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

def regime_width_stats(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Groupby(regime) sur n_days_calendar : count, mean, median, std, min, max.
    Une ligne par régime (calm/trending/stress).
    """
```

Pour la comparaison Crypto/Index/US Bond/ETH : appeler `segment_regimes` sur les 4 DataFrames
(un par actif), concaténer avec une colonne `asset`, puis produire un **box plot** (pas un simple
barplot de moyennes — un box plot montre la dispersion, essentiel ici car les durées de régime ont
des distributions très asymétriques avec des outliers, ex. un régime `calm` qui dure 300 jours) :
- axe X : régime (`calm`, `trending`, `stress`)
- couleur : actif (4 couleurs, cf. `assets.py`)
- axe Y : `n_days_calendar`
- Plotly : `type: "box"`, `boxmode: "group"`, une trace par actif.

### 4.2 Moyenne des régimes à 4 échelles (point 3)

```python
def regime_scale_means(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq ∈ {"full", "Y", "Q", "M"}.
    - "full"  → une seule ligne : moyenne de p_calm/p_trending/p_stress sur tout l'historique.
    - "Y"/"Q"/"M" → groupby(pd.Grouper(freq=freq)) puis .mean() sur p_calm/p_trending/p_stress.
    Retourne un DataFrame indexé par période (ou une seule ligne "Total" si freq="full") avec
    colonnes p_calm, p_trending, p_stress (qui somment à 1 sur chaque ligne).
    """
```

Pourquoi utiliser la moyenne des probabilités (`p_calm`, `p_trending`, `p_stress`) plutôt que le
comptage du régime dominant (`argmax`) par période : la moyenne des probabilités est plus rigoureuse
statistiquement, elle ne perd pas d'information sur les jours ambigus (ex. 48%/47%/5%) que
l'`argmax` written binaire écraserait arbitrairement.

Rendu : pour `freq="full"`, un donut (comme le graphique de distribution déjà existant). Pour
`freq="Y"/"Q"/"M"`, un **stacked bar chart à 100 %** (une barre par période, 3 segments empilés
calm/trending/stress qui somment à 100 %). Un sélecteur (boutons ou dropdown : "Total | Année |
Trimestre | Mois") swap le trace Plotly via `Plotly.react()` — toutes les données des 4 échelles sont
pré-calculées côté Python et injectées en JSON dans le HTML (volumes petits : ≤ 10 lignes pour "Y",
~30 pour "Q", ~90 pour "M" sur 2018-2025 → aucun problème de poids de page).

### 4.3 Vol comme déclencheur de changement de régime (point 1a)

```python
def vol_regime_leadlag(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    Teste si sigma_t (vol GARCH) précède les changements de régime.

    regime_change[t] = 1 si df['regime'][t] != df['regime'][t-1], sinon 0.
    Pour lag = 0..max_lag :
        corr(lag) = pearson( sigma_t.shift(lag), regime_change )   [dropna avant corrélation]
    Interprétation : un pic de corrélation à lag > 0 signifie que la vol d'il y a `lag` jours est
    corrélée avec un changement de régime aujourd'hui → la vol est un signal avancé (déclencheur).
    Un pic à lag = 0 signifie une relation contemporaine (moins intéressant causalement).

    Retourne un DataFrame [lag, corr].
    """

def vol_spike_hit_rate(df: pd.DataFrame, lookback: int = 3, quantile: float = 0.75) -> float:
    """
    Statistique simple et lisible pour le tuteur, en complément de la corrélation décalée :
    % des changements de régime précédés (dans les `lookback` jours précédents) d'un sigma_t
    dépassant le quantile `quantile` de sa distribution glissante sur 60 jours.
    Retourne un float dans [0, 1].
    """
```

Affichage : un bar chart `corr(lag)` pour lag = 0..10, un par actif (grille 2x2 sur l'onglet
Comparaison), plus la `hit_rate` affichée en gros chiffre à côté (ex. "62 % des changements de régime
BTC sont précédés d'un pic de volatilité dans les 3 jours").

### 4.4 Vol comme déclencheur de corrélation inter-actifs (point 1b)

```python
def rolling_cross_correlation(returns_by_asset: dict[str, pd.Series], window: int = 63) -> pd.DataFrame:
    """
    returns_by_asset : {ticker: pd.Series des rendements journaliers}, même index aligné (inner join
    sur les dates communes aux 4 actifs — nécessaire à cause du calendrier crypto vs actions/bonds).
    Calcule la corrélation glissante (fenêtre `window` jours, ex. 63 ≈ 1 trimestre boursier) pour
    chacune des 6 paires uniques parmi les 4 actifs.
    Retourne un DataFrame indexé par date, colonnes du type "BTC-ETH", "BTC-SPY", "SPY-TLT", etc.
    """

def stress_conditioned_correlation(returns_by_asset: dict[str, pd.Series],
                                    stress_masks: dict[str, pd.Series]) -> dict:
    """
    stress_masks : {ticker: pd.Series booléenne, True si p_stress > 0.5 ce jour-là}.
    Compare la corrélation moyenne inter-actifs sur deux sous-échantillons :
      - "au moins un actif en stress" (union des masks)
      - "tous les actifs calmes" (aucun mask actif)
    Retourne {"stress": corr_matrix_moyenne, "calm": corr_matrix_moyenne} pour vérifier
    l'hypothèse de contagion (corrélations plus fortes en période de stress).
    """
```

Affichage : un graphique de corrélation glissante (les 6 paires, lignes de couleurs différentes) avec
en fond les bandes de régime `stress` de BTC (référence marché crypto/risque) en rouge transparent —
pour visuellement corréler "pic de corrélation inter-actifs" et "période de stress". En dessous, un
petit tableau récapitulatif des deux matrices de corrélation moyenne (`stress` vs `calm`).

---

## 5. Fichier `dashboard_builder.py` (nouveau) — orchestration + HTML multi-onglets

### 5.1 Pipeline de calcul

```python
def run_pipeline() -> dict:
    """
    Pour chaque actif de assets.ASSETS :
      1. Télécharge les prix via yfinance (DATA_START → DATA_END), auto_adjust=True.
      2. Aplati les colonnes si MultiIndex (comme déjà fait dans regime_agent.py __main__).
      3. Instancie un RegimeAgent, .fit(prices, train_end=TRAIN_END).
      4. .predict(prices) pour l'état courant (dernier RegimeState).
      5. ._predict_history(prices) pour le DataFrame complet (à exposer en méthode publique
         `predict_history()` — actuellement privée avec un underscore, renommer ou ajouter un
         alias public, SANS toucher à sa logique interne).
    Retourne { ticker: {"prices": df_ohlcv, "history": df_history, "state": RegimeState} }.
    """

def compute_all_analytics(results: dict) -> dict:
    """
    Appelle regime_analytics.* sur les résultats de run_pipeline() :
      - segments + width stats par actif et agrégés
      - regime_scale_means aux 4 échelles, par actif
      - vol_regime_leadlag + vol_spike_hit_rate, par actif
      - rolling_cross_correlation + stress_conditioned_correlation, inter-actifs
    Retourne un dict structuré prêt à sérialiser en JSON pour le template HTML.
    """

def main():
    results = run_pipeline()
    analytics = compute_all_analytics(results)
    html = build_multi_asset_html(results, analytics)
    out = Path(__file__).parent / "output" / "regime_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"[dashboard_builder] HTML généré -> {out.resolve()}")
```

### 5.2 Structure du HTML généré

**Navigation** : une barre d'onglets fixe en haut de page — 5 onglets : `BTC` · `ETH` · `S&P 500` ·
`US Bond (TLT)` · `Comparaison`. Implémentation en JS pur (pas de framework) : chaque onglet est une
`<div class="tab-panel" data-tab="BTC-USD">` cachée par défaut (`display:none`) sauf la première ;
un clic sur le bouton d'onglet bascule la visibilité et **initialise les graphiques Plotly de cet
onglet à la première ouverture seulement** (lazy init avec un flag `_initialized` par onglet, pour ne
pas payer le coût de rendu de 5 dashboards complets au chargement de la page).

**Contenu de chaque onglet actif (BTC/ETH/S&P500/TLT)** — reprend la logique déjà en place dans
`regime_agent._build_html`, à paramétrer par actif au lieu de hardcoder BTC-USD :

1. Légende avec **checkboxes** (nouveau, remplace la légende statique actuelle) :
   - 3 checkboxes régime (Calme/Tendanciel/Stress) — cochées par défaut, décochent le fond coloré
     correspondant sur les 3 graphiques temporels (filtrer le tableau `shapes` avant de rappeler
     `Plotly.relayout`).
   - 4 checkboxes catégorie d'événement (Crypto/Macro/Monétaire/Géopolitique) — cochées par défaut,
     décochent les lignes pointillées + labels de cette catégorie.
   - 1 checkbox "Afficher les libellés d'événements" — **décochée par défaut** (voir §6, allègement
     visuel) : par défaut on ne montre que les traits verticaux, pas le texte, pour ne pas
     surcharger. Cocher affiche les `annotations` texte.
2. Graphique prix (échelle log) + fond régime + lignes d'événements — identique à l'existant, mais
   les `shapes`/`annotations` doivent porter une métadonnée JS (pas Plotly, un objet JS parallèle)
   `{cat: "crypto"|"macro"|..., regime: "calm"|...}` pour permettre le filtrage par les checkboxes.
3. Graphique volatilité (σₜ GARCH + vol-of-vol) — identique à l'existant.
4. Graphique BOCPD (changepoint_prob + seuil 0.5) — identique à l'existant.
5. **Nouveau** : graphique "Composition moyenne des régimes" avec sélecteur Total/Année/Trimestre/Mois
   (§4.2) — remplace le donut statique actuel (le mode "Total" du nouveau composant EST l'ancien
   donut, pas de perte de fonctionnalité).
6. Table des événements (existant) — mettre dans un `<details>` repliable par défaut fermé, pour
   alléger la page (voir §6).

**Contenu de l'onglet Comparaison** (nouveau, §4.1 + §4.3 + §4.4) :
1. Box plot largeur des régimes (4 actifs × 3 régimes).
2. Grille 2×2 des bar charts corrélation décalée vol/changement de régime (un par actif) + les 4
   hit-rates affichés en chiffres clés au-dessus.
3. Graphique corrélation glissante inter-actifs (6 paires) avec fond régime stress BTC.
4. Tableau récapitulatif corrélation moyenne "stress" vs "calme".

**Synchronisation du zoom** : garder le comportement existant (zoom/pan liés) mais **seulement entre
les 3 graphiques temporels d'un même onglet actif** — ne pas essayer de synchroniser entre onglets
différents (complexité inutile, les onglets ne sont pas visibles simultanément).

---

## 6. Allègement visuel — recommandations concrètes

Le dashboard actuel (`regime.html`) devient vite chargé visuellement dès qu'on affiche ~20 événements
sur un historique de plusieurs années (labels en texte incliné à -40°, toujours visibles, se
chevauchent au dézoomé). Recommandations à implémenter :

1. **Labels d'événements masqués par défaut**, visibles seulement via la checkbox "Afficher les
   libellés" (§5.2). Par défaut, seules les lignes verticales pointillées restent visibles — assez
   pour repérer visuellement un choc, sans texte qui se chevauche.
2. **Déclutter dynamique au zoom** : sur l'event `plotly_relayout` (déjà utilisé pour la sync de
   zoom), recalculer quels événements tombent dans la plage X visible ; si le nombre d'événements
   visibles dépasse un seuil (ex. 8), ne pas afficher leurs labels même si la checkbox "libellés" est
   cochée (afficher un message discret "zoomez pour voir les libellés"). Si ≤ 8, les afficher.
3. **Table d'événements repliable** (`<details>`/`<summary>`), fermée par défaut.
4. **Palette et tailles cohérentes** entre les 5 onglets (réutiliser exactement `_REGIME_BG`,
   `_REGIME_HEX`, `_EVENT_COLORS` de `assets.py` partout, ne pas redéfinir de couleurs ad hoc par
   onglet).
5. **Barre d'onglets sticky** en haut de page (`position: sticky; top: 0`) pour ne pas avoir à
   remonter en scrollant.
6. Garder les graphiques `displayModeBar:false` (comme actuellement) pour une interface épurée —
   le zoom/pan reste possible à la souris/trackpad sans la barre d'outils Plotly.

---

## 7. Tests à écrire — `test_regime_analytics.py`

Sur des données synthétiques simples (pas besoin de télécharger yfinance dans les tests, construire
un petit DataFrame `history` à la main avec une colonne `regime` alternant sur des durées connues) :

1. `test_segment_regimes_basic` : séquence `["calm"]*5 + ["stress"]*3 + ["calm"]*2` → doit produire
   3 segments avec les bonnes longueurs (`n_days_trading` = 5, 3, 2).
2. `test_segment_regimes_calendar_vs_trading_days` : vérifier que `n_days_calendar` diffère bien de
   `n_days_trading` quand l'index a des trous (week-ends type SPY/TLT) — construire un index avec
   des sauts de 3 jours (vendredi → lundi) et vérifier `n_days_calendar > n_days_trading` sur ce
   segment.
3. `test_regime_scale_means_sums_to_one` : pour chaque ligne retournée à n'importe quelle échelle,
   `p_calm + p_trending + p_stress ≈ 1.0` (tolérance 1e-6).
4. `test_regime_scale_means_full_matches_global_mean` : le résultat `freq="full"` doit être égal à
   la moyenne simple des colonnes sur tout le DataFrame d'entrée.
5. `test_vol_regime_leadlag_shape` : la sortie a bien `max_lag + 1` lignes (lags 0 à max_lag), les
   corrélations sont dans `[-1, 1]`.
6. `test_vol_spike_hit_rate_bounds` : le résultat est dans `[0, 1]`.
7. `test_rolling_cross_correlation_pairs_count` : avec 4 actifs, la sortie a bien 6 colonnes
   (combinaisons 2 parmi 4), toutes les valeurs dans `[-1, 1]` (hors NaN de warmup).

Exécution : `pytest calibration/regime/test_regime_analytics.py -v`

---

## 8. Ordre d'implémentation recommandé

1. `assets.py` — registre des actifs + événements (base de tout le reste).
2. `regime_analytics.py` + `test_regime_analytics.py` — logique pure, testable sans télécharger de
   données, à valider avant de toucher au HTML.
3. Exposer `RegimeAgent.predict_history()` en public (alias de `_predict_history`, sans changer la
   logique) — petit changement dans `regime_agent.py`.
4. `dashboard_builder.run_pipeline()` — télécharger et fitter les 4 actifs, vérifier manuellement
   (print) que les 4 `RegimeState` et DataFrames d'historique sont cohérents avant de passer au HTML.
5. `dashboard_builder.compute_all_analytics()` — brancher les fonctions de `regime_analytics.py` sur
   les résultats du pipeline.
6. `dashboard_builder.build_multi_asset_html()` — construire le HTML par onglet en réutilisant/
   adaptant les blocs de `regime_agent._build_html` (prix, vol, BOCPD), puis ajouter les nouveaux
   composants (composition multi-échelle, checkboxes, onglet Comparaison).
7. Lancer `python -m calibration.regime.dashboard_builder`, ouvrir `output/regime_dashboard.html`
   dans un navigateur, vérifier visuellement les 5 onglets.
8. Vérification finale (voir §9).

---

## 9. Vérification finale à faire avant de livrer

- `pytest calibration/regime/ -v` → toujours 7/7 sur les tests existants + tous les nouveaux tests
  `test_regime_analytics.py` verts (aucune régression sur le module Maéva/Kyrio existant).
- Ouvrir `regime_dashboard.html` dans un navigateur : les 5 onglets se chargent, le zoom
  synchronisé fonctionne par onglet, les checkboxes cachent/affichent bien ce qu'elles annoncent,
  le sélecteur d'échelle (Total/Année/Trimestre/Mois) change bien le graphique.
- Vérifier à l'œil que les 4 actifs ont des régimes qui font sens (ex. TLT doit montrer du stress/
  trending marqué sur 2022, SPY du stress net en mars 2020 et 2022, ETH du stress en mai 2022 /
  LUNA et nov 2022 / FTX comme BTC).
- Vérifier que le box plot de largeur de régime utilise bien `n_days_calendar` (pas `n_days_trading`)
  — sinon la comparaison crypto vs actions/bonds est biaisée (cf. §4.1).

---

## 10. Hypothèses posées (à challenger si besoin)

- `TLT` retenu comme "exemple pertinent d'obligation US" plutôt que `IEF`/`SHY`/`^TNX` — car c'est
  l'ETF obligataire le plus liquide et le plus volatile, donc celui où les régimes seront les plus
  visibles. À changer facilement (un seul paramètre dans `assets.py`) si le tuteur préfère un autre
  proxy obligataire.
- Fenêtre commune 2018–aujourd'hui pour les 4 actifs (au lieu de 2017/2014 pour BTC) — sacrifie un
  peu d'historique BTC pour permettre une comparaison inter-actifs statistiquement propre.
- "Corrélation" (point 1) interprétée comme corrélation de rendements inter-actifs conditionnée au
  régime/stress (contagion), en complément de la corrélation décalée vol → changement de régime
  (causalité/anticipation). Si le tuteur voulait dire autre chose par "et/ou corrél", clarifier avec
  lui avant l'implémentation du §4.4.
