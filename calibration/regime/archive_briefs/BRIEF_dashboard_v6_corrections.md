# Corrections v6 — Dashboard Multi-Actifs DEITA (séance du 02/07/2026, après-midi)

**Précision** : "TFT" dans la consigne du jour est une coquille pour **TLT** (l'ETF iShares
20+ Year Treasury Bond, déjà utilisé puis retiré en v5 au profit de `ZN=F`) — confirmé.

## 0. Portée de ce patch — lire avant de commencer

Contrairement à `v2`/`v3`/`v4`/`v5` (patchs limités au dashboard HTML), **le point 2 ci-dessous
(bull/bear) touche le moteur de régime lui-même**, donc des fichiers jusqu'ici explicitement
protégés dans tous les briefs précédents : `regime_state.py`, `regime_hmm.py`,
`test_regime_agent.py`. C'est un choix assumé : la consigne du jour demande un changement de
modèle (4 régimes au lieu de 3), pas juste un changement d'affichage — il n'y a pas moyen de faire ça
uniquement côté dashboard. Cette exception ne s'applique qu'au point 2 ; les points 1, 3 et 4
restent des patchs dashboard-only (`assets.py`, `regime_analytics.py`, `dashboard_builder.py`).

Ordre d'implémentation recommandé : **2 d'abord** (c'est la fondation dont dépend le reste de
l'affichage), puis 1, 3, 4 dans n'importe quel ordre.

---

## 1. Remettre TLT dans les actifs (en plus de ZN=F, pas à sa place)

**`assets.py`** — rajouter TLT à `ASSETS`, sans retirer `ZN=F` (les deux représentations du marché
obligataire US — ETF et futures — cohabitent, ce qui permet justement de comparer les deux comme
discuté) :

```python
ASSETS = [
    {"ticker": "BTC-USD", "label": "Bitcoin",                 "short": "BTC", "asset_class": "crypto", "color": "#f7931a"},
    {"ticker": "ETH-USD", "label": "Ethereum",                 "short": "ETH", "asset_class": "crypto", "color": "#627eea"},
    {"ticker": "SPY",     "label": "S&P 500 (SPY)",            "short": "SPX", "asset_class": "index",  "color": "#2ecc71"},
    {"ticker": "ZN=F",    "label": "US Treasury 10Y Note Futures", "short": "ZN", "asset_class": "bond", "color": "#3498db"},
    {"ticker": "TLT",     "label": "US Treasury 20+Y (ETF)",   "short": "TLT", "asset_class": "bond",   "color": "#9b59b6"},
]
```

Rajouter une entrée `ASSET_EVENTS["TLT"]` (peut réutiliser les mêmes événements que `ZN=F`, ils
concernent le même sous-jacent macro) :

```python
ASSET_EVENTS["TLT"] = ASSET_EVENTS["ZN=F"]
```

**Conséquences mécaniques à vérifier** (aucun changement de logique requis, juste vérifier que ça
scale bien à 5 actifs) :
- `dashboard_builder.py` boucle déjà sur `ASSETS` pour tout (`run_pipeline`, `compute_all_analytics`,
  tabs, box plot, leadlag grid, corrélation croisée) — 5 actifs au lieu de 4 doit fonctionner sans
  changement de code, seulement plus de données à traiter.
- Le tableau de corrélation stress/calme et le graphique de corrélation glissante passeront de 6
  paires (`C(4,2)`) à **10 paires** (`C(5,2)`) — plus de lignes sur le graphique. Ne rien
  développer de spécial : Plotly permet déjà nativement de cliquer sur une entrée de légende pour
  masquer/afficher une courbe, ça suffit à gérer la lisibilité sans code additionnel.
- La grille `leadlag-grid` (actuellement CSS `.grid2x2`, 2 colonnes fixes) doit devenir responsive
  pour accueillir 5 actifs proprement. Remplacer :
  ```css
  .grid2x2{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin-bottom:14px; }
  ```
  (renommer la classe en `.gridN` si on veut, ou garder `.grid2x2` tel quel avec ce nouveau CSS —
  au choix, juste changer la règle `grid-template-columns`).

---

## 2. Diviser "trending" en bull / bear → 4 régimes (calm / bull / bear / stress)

### Principe

Le HMM garde exactement la même architecture en cascade (2 étages actuels), on ajoute un **3ᵉ
étage** : à l'intérieur de la masse "trending" (ADX > 25), on regarde le sens de la tendance via les
indicateurs directionnels **DI+ / DI-** que `pandas_ta.adx()` calcule déjà en même temps que l'ADX
(colonnes `DMP_{length}` et `DMN_{length}` — DMP = Directional Movement Plus, DMN = Directional
Movement Minus ; c'est le même appel `ta.adx(...)` déjà utilisé dans `_compute_features`, juste
récupérer 2 colonnes de plus au lieu d'une seule). **Vérifier ces noms de colonnes en environnement
réel avant d'implémenter** (`print(adx_df.columns.tolist())`) — non vérifiable dans cet
environnement de rédaction (pandas_ta absent), mais cohérent avec la convention déjà confirmée dans
le code existant (`adx_df[f"ADX_{self.ADX_PERIOD}"]` fonctionne, donc le même DataFrame a très
probablement aussi `DMP_{length}`/`DMN_{length}`).

```
DI+ > DI-  →  bull  (tendance haussière)
DI+ ≤ DI-  →  bear  (tendance baissière)
```

Nouvelle cascade complète :
1. HMM 2 états (stress / non_stress) — **inchangé**.
2. Seuil ADX sur la masse non_stress : ADX > 25 → "trending" (temporaire), sinon → `calm`.
3. **Nouveau** : à l'intérieur de "trending", signe de `DI+ - DI-` → `bull` ou `bear`.

Résultat final : `probs = {"calm": ..., "bull": ..., "bear": ..., "stress": ...}` (toujours une
seule masse non nulle parmi calm/bull/bear, plus stress — même logique "hard split" que
l'actuelle séparation calm/trending, pas de nouveauté conceptuelle, juste un niveau de plus).

### `regime_hmm.py`

`_compute_features` — récupérer aussi DMP/DMN et calculer `di_diff` :

```python
adx_df = ta.adx(prices["High"], prices["Low"], prices["Close"], length=self.ADX_PERIOD)
adx = adx_df[f"ADX_{self.ADX_PERIOD}"]
di_diff = adx_df[f"DMP_{self.ADX_PERIOD}"] - adx_df[f"DMN_{self.ADX_PERIOD}"]
...
features = pd.DataFrame({
    "sigma_t": sigma_t,
    "adx": adx,
    "di_diff": di_diff,
    "volume_norm": volume_norm,
}, index=prices.index)
return features.dropna()
```

`predict()` — remplacer le bloc de répartition calm/trending par calm/bull/bear :

```python
adx_last = features["adx"].iloc[-1]
di_diff_last = features["di_diff"].iloc[-1]

if adx_last > self.ADX_TRENDING_THRESHOLD:
    p_calm = 0.0
    if di_diff_last >= 0:
        p_bull, p_bear = p_non_stress, 0.0
    else:
        p_bull, p_bear = 0.0, p_non_stress
else:
    p_calm = p_non_stress
    p_bull, p_bear = 0.0, 0.0

probs = {"calm": p_calm, "bull": p_bull, "bear": p_bear, "stress": p_stress}
```

Bump de version (signale le changement de contrat de sortie) :
```python
VERSION = "hmm-garch-adx-v2"
```

### `regime_state.py`

- Docstring du champ `probs` : `"Clés exactes : calm, bull, bear, stress."`
- `validate()` :
  ```python
  if set(self.probs.keys()) != {"calm", "bull", "bear", "stress"}:
      raise ValueError(f"probs doit contenir exactement calm/bull/bear/stress, got {set(self.probs.keys())}")
  ```
- `dominant_regime()` : aucun changement (marche avec n'importe quel jeu de clés).

### `regime_agent.py` — `_predict_history` (classification batch pour tout l'historique)

```python
adx = features["adx"].values
di_diff = features["di_diff"].values
thresh = RegimeHMM.ADX_TRENDING_THRESHOLD
is_trending = adx > thresh

p_calm = np.where(~is_trending, p_non_stress, 0.0)
p_bull = np.where(is_trending & (di_diff >= 0), p_non_stress, 0.0)
p_bear = np.where(is_trending & (di_diff < 0), p_non_stress, 0.0)

stacked = np.stack([p_calm, p_bull, p_bear, p_stress], axis=1)
idx = np.argmax(stacked, axis=1)
_map = {0: "calm", 1: "bull", 2: "bear", 3: "stress"}
regimes = [_map[int(i)] for i in idx]

df = pd.DataFrame({
    "regime": regimes,
    "p_calm": p_calm, "p_bull": p_bull, "p_bear": p_bear, "p_stress": p_stress,
    "vol_bucket": vol_bucket.astype(int),
}, index=features.index)
```

(Adapter les noms de variables intermédiaires en conséquence — la logique de vol_bucket et de BOCPD
juste après dans la même méthode ne change pas.)

### `assets.py` — palette et libellés à 4 régimes

`_REGIME_BG`/`_REGIME_HEX` gardent `calm` (vert) et `stress` (rouge) inchangés (déjà corrigés en
v2, pas de collision). Remplacer `trending` par deux entrées `bull`/`bear`, avec des teintes
nouvelles, choisies pour rester distinctes des 4 couleurs d'événements déjà en place
(`crypto=#e67e22, macro=#ff6ec7, monetaire=#2980b9, geopolitique=#8e44ad`) :

```python
_REGIME_BG = {
    "calm":  "rgba(46,204,113,0.24)",
    "bull":  "rgba(241,196,15,0.22)",    # ambre/or
    "bear":  "rgba(74,105,189,0.24)",    # bleu ardoise/indigo
    "stress":"rgba(231,76,60,0.26)",
}
_REGIME_HEX = {
    "calm": "#2ecc71", "bull": "#f1c40f", "bear": "#4a69bd", "stress": "#e74c3c",
}
_REGIME_LABELS = {
    "calm": "Calme", "bull": "Haussier", "bear": "Baissier", "stress": "Stress",
}
```

(Ces couleurs sont une proposition raisonnable, pas une contrainte absolue — l'important est que
les 4 régimes + les 4 catégories d'événements restent 8 teintes visuellement distinctes entre
elles ; ajuster si besoin en gardant ce principe.)

### `dashboard_builder.py` — généraliser tout ce qui suppose 3 régimes

- `_asset_panel_html` : les 3 checkboxes régime deviennent 4 (`calm`, `bull`, `bear`, `stress`),
  même pattern que l'existant :
  ```html
  <div class="li"><input type="checkbox" class="regime-cb-{ticker}" value="calm" checked><div class="dot" style="background:{_REGIME_HEX['calm']}"></div>Calme</div>
  <div class="li"><input type="checkbox" class="regime-cb-{ticker}" value="bull" checked><div class="dot" style="background:{_REGIME_HEX['bull']}"></div>Haussier</div>
  <div class="li"><input type="checkbox" class="regime-cb-{ticker}" value="bear" checked><div class="dot" style="background:{_REGIME_HEX['bear']}"></div>Baissier</div>
  <div class="li"><input type="checkbox" class="regime-cb-{ticker}" value="stress" checked><div class="dot" style="background:{_REGIME_HEX['stress']}"></div>Stress</div>
  ```
- État JS `TABS[a.ticker].state.regimes` : `{calm:true, bull:true, bear:true, stress:true}`.
- `updateComposition(tabId)` (donut dynamique, v3) : généraliser le comptage à 4 catégories :
  ```js
  let nCalm=0, nBull=0, nBear=0, nStress=0, total=0;
  for (let i=0;i<d.dates.length;i++){
    if (d.dates[i] >= range[0] && d.dates[i] <= range[1]) {
      total++;
      if (d.regimes[i]==='calm') nCalm++;
      else if (d.regimes[i]==='bull') nBull++;
      else if (d.regimes[i]==='bear') nBear++;
      else nStress++;
    }
  }
  const trace = { type:'pie', labels:['Calme','Haussier','Baissier','Stress'],
    values:[pct(nCalm),pct(nBull),pct(nBear),pct(nStress)],
    marker:{colors:[REGIME_HEX.calm,REGIME_HEX.bull,REGIME_HEX.bear,REGIME_HEX.stress]},
    hole:0.42, textinfo:'label+percent', textfont:{size:11,color:'#ecf0f1'}, showlegend:false };
  ```
- Onglet Comparaison, box plot largeur des régimes : `categoryarray`/`ticktext`/`tickvals` passent
  de 3 à 4 valeurs : `['calm','bull','bear','stress']` / `['Calme','Haussier','Baissier','Stress']`.
- `regimeTraces` (légende factice sur le graphique prix, `['calm','trending','stress'].map(...)`)
  devient `['calm','bull','bear','stress'].map(...)`.

### `test_regime_agent.py` — mise à jour des assertions (fichier précédemment protégé, ici concerné)

- **TC4 (bull run 2020-2021)** — c'est littéralement le cas d'usage qui valide le nouveau split :
  ```python
  assert state.probs["bull"] > 0.4, (
      f"Bull attendu > 0.4 en bull run. Obtenu : {state.dominant_regime()} — probs : {state.probs}"
  )
  assert state.probs["calm"] < 0.4
  ```
- **TC5 (choppy 2019)** — l'intention initiale ("pas de tendance dominante") se vérifie maintenant
  sur la somme bull+bear :
  ```python
  assert (state.probs["bull"] + state.probs["bear"]) < 0.4, (
      f"Bull+Bear doit être faible en période choppy. Obtenu : {state.probs}"
  )
  ```
- **TC6 (point-in-time)** — la liste de clés comparées passe de 3 à 4 :
  ```python
  diffs = [abs(state_after.probs[k] - state_before.probs[k]) for k in ("calm", "bull", "bear", "stress")]
  ```
- **TC7 (validation RegimeState)** — reconstruire les 3 cas invalides avec le schéma à 4 clés, en
  gardant la même intention de test (juste ajouter une clé `bull`/`bear` cohérente à chaque exemple
  pour que la répartition reste réaliste, sauf pour le cas qui teste justement "la somme ne fait pas
  1") :
  ```python
  with pytest.raises(ValueError, match="somme"):
      RegimeState(probs={"calm": 0.5, "bull": 0.3, "bear": 0.2, "stress": 0.5}, ...).validate()

  with pytest.raises(ValueError, match="vol_bucket"):
      RegimeState(probs={"calm": 0.5, "bull": 0.2, "bear": 0.2, "stress": 0.1}, vol_bucket=3, ...).validate()

  with pytest.raises(ValueError, match="stress_score"):
      RegimeState(probs={"calm": 0.5, "bull": 0.2, "bear": 0.2, "stress": 0.1}, stress_score=0.9, ...).validate()
  ```
- TC1/TC2/TC3 : aucun changement (ne référencent ni "trending" ni bull/bear).

---

## 3. Remplacer la corrélation décalée vol/régime par une étude d'événement (début/fin de régime)

**Problème avec l'existant** (`vol_regime_leadlag`) : une simple corrélation de Pearson entre
`sigma_t` décalé et un indicateur binaire "le régime a changé ce jour-là" mélange tous les types de
transition (calm→stress, stress→calm, bull→bear, etc.) dans un seul chiffre par lag — trop agrégé
pour vraiment répondre à "la vol est-elle liée au **début** d'un changement de régime".

**Nouvelle méthode : étude d'événement** (event study — méthode standard en finance empirique).
Pour chaque transition de régime (bornes déjà disponibles via `segment_regimes`), on regarde le
profil moyen de `sigma_t` sur une fenêtre de ±N jours autour de la date de transition, aligné sur un
axe "temps relatif à l'événement", puis on moyenne sur toutes les transitions. Si la vol moyenne
grimpe juste avant/au moment `t=0`, la vol est un signal avancé du changement de régime.

### `regime_analytics.py`

Supprimer `vol_regime_leadlag` et `vol_spike_hit_rate` (remplacés), ajouter :

```python
def regime_transition_vol_profile(df: pd.DataFrame, window: int = 10, alignment: str = "start",
                                   only_into: str | None = None) -> pd.DataFrame:
    """
    Étude d'événement : profil moyen de sigma_t autour des transitions de régime.

    alignment : "start" (jour 0 = premier jour du nouveau régime) ou "end" (jour 0 = dernier
    jour de l'ancien régime, juste avant la transition).
    only_into : si fourni (ex. "stress"), ne considère que les transitions VERS ce régime
    (ex. "à quoi ressemble la vol juste avant qu'on bascule en stress ?"). Si None, toutes les
    transitions sont poolées ensemble.
    window : nombre de jours de trading de part et d'autre de l'événement.

    Pour chaque segment de regime_segments (sauf le premier et le dernier segment de
    l'historique, qui n'ont pas une fenêtre complète des deux côtés), extraire sigma_t sur
    [event_idx - window, event_idx + window] en position entière (iloc), aligner sur un axe
    rel_day = -window..+window, puis moyenner (et calculer l'écart-type) sur tous les
    événements retenus.

    Retourne un DataFrame [rel_day, mean_sigma, std_sigma, n_events].
    """
```

Implémentation : utiliser `segment_regimes(df)` pour obtenir les segments, prendre soit `start`
(alignment="start") soit `end` (alignment="end") de chaque segment (sauf le tout premier segment
si alignment="start", ou le tout dernier si alignment="end", qui n'ont pas de "avant"/"après"
dans les données), retrouver la position entière (`df.index.get_loc(...)`) de cette date, extraire
`df["sigma_t"].iloc[pos-window : pos+window+1]` si la fenêtre est entièrement dans les bornes du
DataFrame (sinon ignorer ce segment — pas de padding artificiel), réindexer sur `range(-window,
window+1)`, empiler tous les événements retenus dans une matrice, puis `mean(axis=0)`/`std(axis=0)`
colonne par colonne (= jour relatif par jour relatif) pour obtenir le profil moyen.

Si `only_into` est fourni, ne garder que les segments dont `regime == only_into` (pour
alignment="start", c'est le régime du segment qui COMMENCE à l'événement ; pour alignment="end", il
faudrait regarder le régime du segment SUIVANT — préciser ce sens dans le docstring et le code).

### `dashboard_builder.py`

Dans `compute_all_analytics`, remplacer les appels à `vol_regime_leadlag`/`vol_spike_hit_rate` par:

```python
profile_all = ra.regime_transition_vol_profile(history, window=10, alignment="start")
profile_into_stress = ra.regime_transition_vol_profile(history, window=10, alignment="start", only_into="stress")
```

Dans `_comparison_payload`, remplacer le bloc `leadlag` par un payload équivalent portant les deux
profils (`profile_all`, `profile_into_stress`) par actif : `rel_day`, `mean_sigma`, `std_sigma`,
`n_events` pour chacun.

### HTML/JS — remplacer la carte "Volatilité comme déclencheur..."

Remplacer le titre de la carte et son contenu (grille 2x2 actuelle de bar charts lag-corrélation +
hit-rates) par une grille (même layout responsive que décrit en §1) de graphiques ligne, un par
actif : `mean_sigma` en fonction de `rel_day` (axe -10 à +10), avec une ligne verticale à `rel_day=0`
(la transition), et idéalement une bande d'écart-type (`mean_sigma ± std_sigma`, en zone semi-
transparente autour de la ligne). Deux courbes superposées par graphique : "toutes transitions" et
"transitions vers stress uniquement" (légende à 2 entrées), pour voir si le pic de vol est
particulièrement marqué spécifiquement avant une bascule en stress.

Nouveau titre de carte : *"Volatilité autour des changements de régime (étude d'événement, ±10j)"*.
Sous-titre explicatif : *"Moyenne de σₜ sur une fenêtre de 10 jours avant/après chaque transition
de régime (alignée sur le premier jour du nouveau régime). Si σₜ augmente avant le jour 0, la
volatilité précède le changement de régime — c'est un signal avancé, pas seulement une
conséquence."*

---

## 4. Corrélation inter-actifs : ajouter volatilité, volume, swing

**Interprétation de "swing"** (à confirmer si ce n'est pas ce qui était voulu) : amplitude du
mouvement journalier, `(High - Low) / Close` — une mesure brute et immédiate de l'agitation du
prix, complémentaire à `sigma_t` (qui est une estimation GARCH lissée, pas une mesure brute
jour par jour).

**Bonne nouvelle : `rolling_cross_correlation` est déjà générique** (elle prend n'importe quel
dict `{ticker: pd.Series}`, ce n'est pas câblé sur les rendements). Pas besoin d'une nouvelle
fonction dans `regime_analytics.py` — juste construire 3 dicts de séries supplémentaires côté
`dashboard_builder.py` et appeler la fonction existante 3 fois de plus.

### `dashboard_builder.py` — `compute_all_analytics`

```python
returns_by_asset = {}
sigma_by_asset = {}
volume_by_asset = {}
swing_by_asset = {}
stress_masks = {}
calm_masks = {}

for asset in ASSETS:
    ...
    history = results[ticker]["history"]
    prices = results[ticker]["prices"]
    returns = prices["Close"].pct_change().dropna()

    returns_by_asset[short] = returns
    sigma_by_asset[short] = history["sigma_t"].reindex(returns.index)
    volume_by_asset[short] = (prices["Volume"] / prices["Volume"].rolling(30).mean()).reindex(returns.index)
    swing_by_asset[short] = ((prices["High"] - prices["Low"]) / prices["Close"]).reindex(returns.index)
    ...

cross_correlation = {
    "returns": ra.rolling_cross_correlation(returns_by_asset, window=63),
    "volatility": ra.rolling_cross_correlation(sigma_by_asset, window=63),
    "volume": ra.rolling_cross_correlation(volume_by_asset, window=63),
    "swing": ra.rolling_cross_correlation(swing_by_asset, window=63),
}
```

Dans `_comparison_payload`, sérialiser les 4 variantes (même format que l'actuel `cc_series`, pour
chacune des 4 clés `returns/volatility/volume/swing`).

### HTML/JS

Sur la carte "Corrélation glissante inter-actifs", ajouter un petit sélecteur (mêmes styles
`.scale-btn` déjà utilisés ailleurs, à réutiliser pour cohérence visuelle) :

```html
<div class="scale-sel">
  <button class="cc-btn cc-btn-active" data-signal="returns">Rendements</button>
  <button class="cc-btn" data-signal="volatility">Volatilit&#233;</button>
  <button class="cc-btn" data-signal="volume">Volume</button>
  <button class="cc-btn" data-signal="swing">Swing</button>
</div>
```

Au clic, reconstruire les traces à partir de `COMPARISON.cross_correlation[signal].series` (même
palette de couleurs par paire) et `Plotly.react('chart-crosscorr', ...)`. Le fond rouge (stress
marché, déjà en place depuis v2 §4) reste identique quel que soit le signal affiché — la
segmentation de stress ne dépend pas du signal choisi.

Mettre à jour le sous-titre pour préciser que la lecture change selon le signal choisi, ex. :
*"Rendements = co-mouvement des prix. Volatilité = les 4 actifs deviennent-ils risqués en même
temps ? Volume = panique/attention de marché partagée ? Swing = amplitude journalière (High-Low)/Close,
mesure brute non lissée, complémentaire à la volatilité GARCH."*

---

## 5. Tests à mettre à jour/ajouter

`test_regime_analytics.py` :
- Supprimer les tests de `vol_regime_leadlag`/`vol_spike_hit_rate` (fonctions supprimées).
- Ajouter des tests pour `regime_transition_vol_profile` : sur un DataFrame synthétique avec un
  sigma_t construit à la main (ex. un pic net juste avant une transition connue), vérifier que le
  profil retourné a bien un maximum de `mean_sigma` proche de `rel_day=0` ou juste avant, et que
  `n_events` correspond au nombre de transitions réellement utilisables (bornes complètes).
- Vérifier que le premier et dernier segment de l'historique sont bien exclus (fenêtre incomplète).

`test_regime_agent.py` : appliquer toutes les modifications listées en §2 (TC4, TC5, TC6, TC7).

---

## 6. Vérification finale

- `pytest calibration/regime/ -v` → tout vert.
- 5 onglets actifs (BTC/ETH/SPX/ZN/TLT) + Comparaison.
- Chaque onglet actif a 4 checkboxes régime (Calme/Haussier/Baissier/Stress), le donut composition
  a 4 parts, le fond du graphique prix distingue bien visuellement les 4 régimes.
- TC4 (bull run BTC 2020-2021) passe avec `probs["bull"] > 0.4` — bonne validation que le split
  fonctionne comme attendu sur un cas connu.
- Onglet Comparaison : le box plot largeur de régime a 4 catégories × 5 actifs ; la carte
  volatilité/régime affiche des courbes de profil (pas des bar charts de corrélation par lag) avec
  un pic visible autour de `rel_day=0` si l'hypothèse "vol = signal avancé" se vérifie ; la carte
  corrélation glissante a un sélecteur Rendements/Volatilité/Volume/Swing fonctionnel.
