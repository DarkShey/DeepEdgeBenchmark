# Corrections v2 — Dashboard Multi-Actifs DEITA (retour du 02/07/2026)

**Contexte :** `BRIEF_dashboard_multiasset.md` a été implémenté (`assets.py`, `regime_analytics.py`,
`dashboard_builder.py`, `output/regime_dashboard.html`). Ce fichier corrige 4 points précis relevés
après relecture du rendu. **Patch ciblé, pas une réécriture** : ne toucher que ce qui est listé
ci-dessous. Ne pas retoucher `regime_state.py`, `regime_hmm.py`, `regime_bocpd.py`,
`test_regime_agent.py`.

---

## 1. Le sélecteur d'échelle doit vivre dans le graphe principal, pas seulement dans la composition

**Problème :** actuellement le sélecteur Total/Année/Trimestre/Mois (`.scale-sel`,
`_asset_panel_html`) ne pilote que la carte "Composition moyenne des régimes"
(`renderComposition`). Le graphique prix garde toujours le fond régime au jour le jour.

**Attendu :** un seul sélecteur, affiché sur la carte du **graphique prix**, qui pilote à la fois :
(a) la résolution du fond coloré régime sur les 3 graphiques temporels (prix/vol/BOCPD), et
(b) la carte composition (comportement déjà en place, à réutiliser tel quel).

**Renommage des 4 modes** : `Jour` (= comportement actuel exact, fond régime au jour le jour) /
`Année` / `Trimestre` / `Mois`. Le mode `Jour` correspond à ce qui s'appelait `full` pour la
composition (donut du total) — les deux significations coïncident : "pas d'agrégation" = le fond
journalier réel sur le prix, et le donut total sur la composition.

### Changements `regime_analytics.py`

Ajouter :

```python
def regime_scale_segments(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq ∈ {"Y", "Q", "M"}. Calcule le régime dominant (argmax de p_calm/p_trending/p_stress
    moyennés) par période via regime_scale_means(df, freq), puis retourne un DataFrame
    [start, end, regime] avec une ligne par période — utilisé pour dessiner un fond régime
    "grossier" sur le graphique prix (une couleur par année/trimestre/mois au lieu du détail
    quotidien).
    start/end = bornes calendaires de la période (Period.start_time / .end_time).
    """
    means = regime_scale_means(df, freq)  # index = périodes, colonnes p_calm/p_trending/p_stress
    dominant = means.idxmax(axis=1).str.replace("p_", "", regex=False)
    rows = []
    for period, regime in zip(means.index, dominant):
        rows.append({"start": period.start_time, "end": period.end_time, "regime": regime})
    return pd.DataFrame(rows)
```

### Changements `dashboard_builder.py`

Dans `_asset_tab_payload` : ajouter, à côté de `regime_shapes` (renommer implicitement en usage
"Jour"), un dict `scale_shapes` :

```python
scale_shapes = {}
for freq in ["Y", "Q", "M"]:
    seg = ra.regime_scale_segments(history, freq)
    scale_shapes[freq] = [
        {"x0": str(row["start"].date()), "x1": str(row["end"].date()), "regime": row["regime"]}
        for _, row in seg.iterrows()
    ]
```
… et l'inclure dans le dict retourné (`"scale_shapes": scale_shapes`).

### Changements JS (dans `build_multi_asset_html`)

- Déplacer le bloc `.scale-sel` (dans `_asset_panel_html`) de la carte composition vers la carte
  du graphique prix (juste au-dessus de `<div id="chart-price-{ticker}">`), avec les 4 boutons
  `Jour / Année / Trimestre / Mois` (`data-scale="jour"|"Y"|"Q"|"M"`). Supprimer le sélecteur
  dupliqué de la carte composition (un seul contrôle désormais).
- `buildShapes(tabId)` : actuellement construit toujours les rects depuis `d.regime_shapes`
  (quotidien). Le rendre sensible à `TABS[tabId].state.scale` :
  ```js
  function buildShapes(tabId) {
    const d = TAB_DATA[tabId], st = TABS[tabId].state, shapes = [];
    const regimeBoxes = (st.scale === 'jour') ? d.regime_shapes : d.scale_shapes[st.scale];
    regimeBoxes.forEach(s => {
      if (!st.regimes[s.regime]) return;
      shapes.push({type:'rect',xref:'x',yref:'paper',x0:s.x0,x1:s.x1,y0:0,y1:1,
        fillcolor:REGIME_BG[s.regime],line:{width:0},layer:'below'});
    });
    d.event_lines.forEach(e => { /* inchangé, cf. §2 */ });
    return shapes;
  }
  ```
- `TABS[a.ticker].state.scale` initialisé à `'jour'` (au lieu de `'full'`).
- Le handler de clic sur `.scale-btn-{ticker}` doit maintenant : mettre à jour `state.scale`,
  rappeler `refreshTab(tabId)` (pour re-dessiner les 3 fonds), **et** rappeler
  `renderComposition(tabId, btn.dataset.scale === 'jour' ? 'full' : btn.dataset.scale)` (la
  composition garde en interne les clés `full/Y/Q/M`, seul le libellé du bouton change).

---

## 2. Ne jamais cacher les événements de marché référencés

**Problème :** les checkboxes catégorie (Crypto/Macro/Monétaire/Géopolitique) et la checkbox
"Afficher les libellés" permettent de masquer des événements. Retour : c'est inutile, les
événements doivent être **toujours visibles**.

**Changements :**
- `_asset_panel_html` : supprimer les 4 checkboxes catégorie et la checkbox "Afficher les
  libellés" du bloc `.legend`. Ne garder que les 3 checkboxes régime (Calme/Tendanciel/Stress),
  qui elles restent utiles pour alléger le fond si besoin.
- `buildShapes(tabId)` : les lignes d'événements (`d.event_lines`) sont **toujours** ajoutées aux
  shapes, sans filtre de catégorie (supprimer le `if (!st.cats[e.cat]) return;`).
- `buildAnnotations(tabId)` : supprimer le flag manuel `showLabels`. Les libellés texte
  s'affichent **automatiquement** dès qu'il y a ≤ 8 événements dans la plage X visible (logique de
  décongestion déjà existante, on la garde), sinon ils restent masqués avec le message discret
  "Zoomez pour voir les libellés" (déjà en place). Seule différence : ce n'est plus une checkbox
  qui active/désactive ce comportement, c'est automatique et permanent.
- Supprimer `state.cats` et `state.showLabels` de l'objet `TABS[ticker].state` (n'existent plus).
- Supprimer les listeners JS correspondants (`cat-cb-*`, `labels-cb-*`) dans `initAssetTab`.

---

## 3. Palette du fond régime peu lisible

**Problème concret identifié :** dans `assets.py`, `_REGIME_HEX["trending"] = "#2980b9"` (bleu)
est **strictement identique** à `_EVENT_COLORS["monetaire"] = "#2980b9"`, et
`_REGIME_HEX["stress"] = "#e74c3c"` est **strictement identique** à `_EVENT_COLORS["macro"] =
"#e74c3c"`. En plus de cette collision de couleurs entre deux systèmes différents (fond de régime
vs catégorie d'événement), le bleu du régime `trending` en `rgba(41,128,185,0.18)` se distingue mal
du fond des cartes (`#16213e`, un bleu marine déjà proche de cette teinte) — d'où l'impression que
les couleurs de régime "ne se voient pas bien".

**Correction dans `assets.py`** — palette régime repensée en feu tricolore (3 teintes bien plus
séparées entre elles et du fond marine), et recoloration des catégories d'événements pour lever
toute collision :

```python
_REGIME_BG = {
    "calm":     "rgba(46,204,113,0.24)",   # vert
    "trending": "rgba(241,196,15,0.22)",   # ambre/or (remplace le bleu, trop proche du fond)
    "stress":   "rgba(231,76,60,0.26)",    # rouge
}
_REGIME_HEX = {
    "calm":     "#2ecc71",
    "trending": "#f1c40f",
    "stress":   "#e74c3c",
}

_EVENT_COLORS = {
    "crypto":       "#e67e22",  # orange (inchangé)
    "macro":        "#ff6ec7",  # rose/magenta (était #e74c3c -> collision avec stress)
    "monetaire":    "#2980b9",  # bleu (libéré, ne collisionne plus avec trending)
    "geopolitique": "#8e44ad",  # violet (inchangé)
}
```

Mettre à jour la légende HTML (`.dot` de la légende dans `_asset_panel_html`) avec ces mêmes
valeurs (actuellement les points de couleur de la légende sont écrits en dur dans le HTML,
ex. `style="background:#27ae60"` pour calme et `style="background:#2980b9"` pour tendanciel —
remplacer par les nouvelles valeurs de `_REGIME_HEX`/`_EVENT_COLORS`, idéalement en générant ces
`style` depuis les constantes Python plutôt qu'en dur, pour éviter que ça se re-désynchronise).

---

## 4. Corrélation inter-actifs : peu claire et pas assez rigoureuse

**Problèmes relevés :**
1. Le graphique "Corrélation glissante inter-actifs (63j)" trace 6 courbes nommées par codes
   (`BTC-ETH`, `SPX-TLT`, ...) sans rappel de ce que ces codes désignent, ni de ce qu'est
   exactement la corrélation affichée (corrélation de quoi, entre quoi).
2. Le fond rouge de ce même graphique représente "les périodes de stress BTC" — un choix arbitraire
   (pourquoi BTC comme référence de marché et pas un des 3 autres actifs, ou une définition
   combinée ?), incohérent avec la définition de "stress" utilisée juste en dessous dans le tableau
   stress/calme (qui, elle, utilise l'union des 4 actifs).
3. Le tableau "Corrélation moyenne inter-actifs — stress vs calme" : la case "calme" est en réalité
   définie comme *"aucun des 4 actifs n'est en stress"* — ce qui **inclut les jours où un ou
   plusieurs actifs sont en régime `trending`**. Ce n'est pas un vrai contraste "calme" au sens
   strict, ça affaiblit la lecture de l'hypothèse de contagion.

**Corrections :**

### 4.1 Définitions unifiées et strictes (`regime_analytics.py`)

```python
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
```

Modifier `stress_conditioned_correlation` pour prendre **deux** dicts de masks au lieu d'un, et
utiliser une intersection stricte pour le "calme" :

```python
def stress_conditioned_correlation(returns_by_asset: dict, stress_masks: dict, calm_masks: dict) -> dict:
    """
    stress_masks : {ticker: bool série, True si p_stress > 0.5}.
    calm_masks   : {ticker: bool série, True si regime == "calm"}.

    "stress" = au moins 1 actif sur 4 en stress (union stress_masks).
    "calm"   = LES 4 ACTIFS SIMULTANÉMENT en régime calme (intersection stricte calm_masks) —
               remplace l'ancienne définition "aucun stress" qui laissait passer les jours trending.

    Retourne {"stress": corr_matrix, "calm": corr_matrix, "stress_mask": pd.Series, "calm_mask": pd.Series}.
    """
```

### 4.2 `dashboard_builder.py`

Dans `compute_all_analytics` : construire aussi `calm_masks[short] = (history["regime"] ==
"calm").reindex(returns.index).fillna(False)` en plus de `stress_masks`, et les passer tous les
deux à `ra.stress_conditioned_correlation(returns_by_asset, stress_masks, calm_masks)`.

Dans `_comparison_payload` : remplacer le calcul actuel des `stress_bands` (basé uniquement sur
`results["BTC-USD"]["history"]`, régime `stress`) par le mask marché unifié :

```python
market_stress = analytics["comparison"]["stress_conditioned"]["stress_mask"]
stress_bands = [
    {"x0": str(s["start"].date()), "x1": str(s["end"].date())}
    for s in ra.segment_boolean_mask(market_stress)
]
```

### 4.3 Sous-titres explicites dans le HTML (`build_multi_asset_html`)

Ajouter un `<p class="chart-note">` (nouveau style CSS simple : `color:#7f8c8d;font-size:.72rem;
margin-bottom:8px`) sous chaque `.card-label` concerné :

- Sous le titre du graphique de corrélation glissante :
  > "Corrélation de Pearson glissante (fenêtre 63 jours, rendements journaliers). BTC = Bitcoin,
  > ETH = Ethereum, SPX = S&P 500 (SPY), TLT = US Treasury 20+ ans (TLT). Fond rouge = jours où au
  > moins un des 4 actifs est en régime de stress (p_stress > 0.5)."
- Sous le titre du tableau stress/calme :
  > "Corrélation moyenne des rendements journaliers entre chaque paire d'actifs, calculée
  > séparément sur deux sous-échantillons de jours : Stress = au moins 1 actif sur 4 en régime
  > stress ce jour-là. Calme = les 4 actifs simultanément en régime calme ce jour-là. Hypothèse
  > testée : la corrélation inter-actifs augmente en période de stress (contagion)."

---

## 5. Tests à mettre à jour — `test_regime_analytics.py`

Ajouter :
1. `test_regime_scale_segments_dominant_and_bounds` : sur un petit DataFrame synthétique, vérifier
   que le régime dominant par période correspond bien à celui attendu et que `start`/`end`
   couvrent exactement la période sans trou ni chevauchement entre segments consécutifs.
2. `test_segment_boolean_mask_basic` : séquence booléenne `[F,T,T,F,T]` → 2 segments (indices 1-2
   et 4), bornes correctes.
3. `test_market_mask_union_and_intersection` : sur 2-3 masks synthétiques simples, vérifier
   `market_mask_union` (OR) et `market_mask_intersection` (AND) donnent le résultat logique attendu.
4. `test_stress_conditioned_correlation_strict_calm` : construire un cas où un actif est
   `trending` (pas `stress`) un jour donné — vérifier que ce jour est bien **exclu** du bucket
   "calme" avec la nouvelle définition stricte (il aurait été inclus avec l'ancienne).
5. Mettre à jour toute assertion existante qui dépendrait de l'ancienne signature de
   `stress_conditioned_correlation` (un seul dict de masks) — elle prend maintenant 2 dicts.

Exécution : `pytest calibration/regime/test_regime_analytics.py -v` (doit rester 100 % vert).

---

## 6. Vérification finale

- `pytest calibration/regime/ -v` → tout vert (existant + nouveaux tests §5).
- Régénérer : `python -m calibration.regime.dashboard_builder`.
- Ouvrir `regime_dashboard.html` : sur chaque onglet actif, le sélecteur Jour/Année/Trimestre/Mois
  est désormais au-dessus du graphique prix et change bien le fond régime du graphique prix (et
  vol/BOCPD) en plus de la composition.
- Vérifier qu'aucune checkbox catégorie/labels n'est visible — les événements et leurs traits
  verticaux sont **toujours** affichés, sans possibilité de les cacher.
- Vérifier visuellement que le régime `trending` est maintenant bien distinct (ambre) du fond des
  cartes et du régime `stress` (rouge) et `calm` (vert).
- Sur l'onglet Comparaison, vérifier que le sous-titre sous le graphique de corrélation glissante
  et sous le tableau stress/calme sont bien présents et lisibles, et que le fond rouge du graphique
  de corrélation correspond maintenant au stress "marché" (union des 4 actifs), pas seulement BTC.
