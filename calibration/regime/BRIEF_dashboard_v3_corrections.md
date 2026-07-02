# Corrections v3 — Dashboard Multi-Actifs DEITA (retour du 02/07/2026, suite)

**Erreur à corriger :** dans le patch v2, le sélecteur Jour/Année/Trimestre/Mois a été implémenté
comme un changement de **résolution de coloriage du fond** (segments de régime plus "grossiers"),
en gardant tout l'historique affiché sur le graphique. **Ce n'est pas ce qui était demandé.** Le
besoin réel : ces 4 boutons doivent **zoomer** le graphique (rétrécir la fenêtre temporelle
affichée) pour voir un jour / un mois / un trimestre / une année *en grand*, avec un moyen de
choisir *quelle* période regarder. Et la composition des régimes doit rester un **donut/camembert**
en toutes circonstances (jamais un bar chart), recalculé pour la période actuellement affichée.

Ce fichier annule et remplace la section §1 de `BRIEF_dashboard_v2_corrections.md`. Les sections 2,
3 et 4 de ce patch v2 (événements toujours visibles, palette de couleurs, rigueur de la corrélation
inter-actifs) restent valables et ne sont pas concernées ici.

**Contrainte de données à connaître :** les données viennent de yfinance en **résolution
journalière uniquement** (une valeur de clôture par jour, pas de données intra-journalières). Le
mode "Jour" ne peut donc pas montrer des mouvements *dans* une journée (ça n'existe pas dans ces
données gratuites) — il consiste à **zoomer sur une fenêtre courte** (quelques semaines) pour que
chaque point quotidien soit visuellement distinct, au lieu d'être écrasé dans une vue de 8 ans.

---

## 1. Nouveau mécanisme : zoom + navigation par date, pas recoloriage du fond

### Suppression de ce qui a été fait en v2 pour ce point

- `regime_analytics.py` : supprimer entièrement `regime_scale_segments` (plus utilisée).
- `regime_analytics.py` : supprimer `regime_scale_means` si rien d'autre ne l'utilise après ce
  patch (vérifier — après ce patch, plus aucun appelant ne doit rester).
- `dashboard_builder.py` : supprimer `_scale_means_payload`, le bloc `scale_shapes` dans
  `_asset_tab_payload`, la clé `"scale_means"` dans `compute_all_analytics` (dict `per_asset`,
  actuellement inutilisée de toute façon — vérifié : `_asset_tab_payload` recalculait son propre
  `_scale_means_payload(history)` au lieu de lire `analytics_asset["scale_means"]`, donc cette
  clé était déjà morte).
- `test_regime_analytics.py` : supprimer `test_regime_scale_means_sums_to_one`,
  `test_regime_scale_means_full_matches_global_mean`, `test_regime_scale_segments_dominant_and_bounds`,
  et les imports correspondants (`regime_scale_means`, `regime_scale_segments`).

### Ce qu'il faut construire à la place

**Les 4 boutons deviennent des raccourcis de zoom temporel**, avec ces fenêtres (en jours
calendaires, centrées sur une date d'ancrage) :

| Bouton | `data-scale` | Largeur de fenêtre | Usage |
|---|---|---|---|
| Jour | `jour` | 60 jours | voir les mouvements quotidiens distinctement |
| Mois | `mois` | 365 jours | voir les cycles mensuels sur ~1 an |
| Trimestre | `trimestre` | 1095 jours (3 ans) | voir les cycles trimestriels |
| Année | `annee` | historique complet | vue d'ensemble (mode par défaut à l'ouverture) |

**Sélecteur de date** : un `<input type="date">` à côté des boutons permet de choisir la date sur
laquelle centrer le zoom (pertinent pour Jour/Mois/Trimestre ; sans effet sur Année qui montre
toujours tout — le désactiver visuellement quand Année est actif). Bornes `min`/`max` de l'input =
`first_date`/`last_date` de l'actif.

### `dashboard_builder.py` — `_asset_panel_html`

Signature modifiée pour recevoir les bornes de dates (disponibles dans `tabs_payload` construit
juste avant dans `build_multi_asset_html`) :

```python
asset_panels = "".join(
    _asset_panel_html(a, tabs_payload[a["ticker"]]["first_date"], tabs_payload[a["ticker"]]["last_date"])
    for a in ASSETS
)
```

```python
def _asset_panel_html(asset: dict, first_date: str, last_date: str) -> str:
    ticker = asset["ticker"]
    ...
    # dans le bloc .scale-sel, remplacer le contenu par :
    """
    <div class="scale-sel">
      <span class="scale-label">Centrer sur</span>
      <input type="date" id="datepick-{ticker}" class="date-pick" min="{first_date}" max="{last_date}" disabled>
      <button class="scale-btn scale-btn-{ticker}" data-scale="jour">Jour</button>
      <button class="scale-btn scale-btn-{ticker}" data-scale="mois">Mois</button>
      <button class="scale-btn scale-btn-{ticker}" data-scale="trimestre">Trimestre</button>
      <button class="scale-btn scale-btn-{ticker} active" data-scale="annee">Ann&#233;e</button>
    </div>
    """
```

Ajouter au CSS : `.scale-label{{font-size:.72rem;color:#7f8c8d;margin-right:2px}}` et
`.date-pick{{background:#0f0f1a;color:#ecf0f1;border:1px solid #1c2a3a;border-radius:4px;
padding:2px 6px;font-size:.72rem}} .date-pick:disabled{{opacity:.4}}`.

### `dashboard_builder.py` — `_asset_tab_payload`

Ajouter le tableau brut des régimes quotidiens (nécessaire pour calculer la composition
dynamiquement côté JS) :

```python
return {
    ...
    "dates": dates_str,
    "regimes": regimes,   # <- NOUVEAU : liste quotidienne, même longueur que "dates"
    ...
    # supprimer "scale_shapes" et "scale_means"
}
```

### JS — `build_multi_asset_html`

**État par onglet** : `scale` initialisé à `'annee'` (pas `'jour'`) — l'ouverture d'un onglet montre
tout l'historique par défaut, comme avant.

**`buildShapes(tabId)`** : revient à toujours utiliser le détail quotidien (plus de branche
`scale_shapes`) :

```js
function buildShapes(tabId) {
  const d = TAB_DATA[tabId], st = TABS[tabId].state, shapes = [];
  d.regime_shapes.forEach(s => {
    if (!st.regimes[s.regime]) return;
    shapes.push({type:'rect',xref:'x',yref:'paper',x0:s.x0,x1:s.x1,y0:0,y1:1,
      fillcolor:REGIME_BG[s.regime],line:{width:0},layer:'below'});
  });
  d.event_lines.forEach(e => {
    shapes.push({type:'line',xref:'x',yref:'paper',x0:e.x,x1:e.x,y0:0,y1:1,
      line:{color:e.color,width:1.2,dash:'dot'}});
  });
  return shapes;
}
```

**Nouvelle fonction `applyZoom`** (remplace la logique de clic actuelle sur `.scale-btn-*`) :

```js
const ZOOM_WINDOW_DAYS = {jour: 60, mois: 365, trimestre: 1095};

function clampDate(dateObj, minStr, maxStr) {
  const s = dateObj.toISOString().slice(0, 10);
  if (s < minStr) return minStr;
  if (s > maxStr) return maxStr;
  return s;
}

function applyZoom(tabId, scale, anchorDateStr) {
  const d = TAB_DATA[tabId];
  TABS[tabId].state.scale = scale;
  let r0, r1;
  if (scale === 'annee') {
    r0 = d.first_date; r1 = d.last_date;
  } else {
    const anchor = anchorDateStr ? new Date(anchorDateStr + 'T00:00:00') : new Date(d.last_date + 'T00:00:00');
    const half = ZOOM_WINDOW_DAYS[scale] / 2;
    const start = new Date(anchor); start.setDate(start.getDate() - half);
    const end = new Date(anchor); end.setDate(end.getDate() + half);
    r0 = clampDate(start, d.first_date, d.last_date);
    r1 = clampDate(end, d.first_date, d.last_date);
  }
  ['price', 'vol', 'cp'].forEach(k => Plotly.relayout(`chart-${k}-${tabId}`, {
    'xaxis.range[0]': r0, 'xaxis.range[1]': r1,
  }));
  TABS[tabId].currentXRange = [r0, r1];
  onRangeChange(tabId);

  const dp = document.getElementById(`datepick-${tabId}`);
  if (dp) dp.disabled = (scale === 'annee');
}
```

**Listeners** (remplacent le listener actuel sur `.scale-btn-{tabId}`) :

```js
document.querySelectorAll(`.scale-btn-${tabId}`).forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll(`.scale-btn-${tabId}`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const dp = document.getElementById(`datepick-${tabId}`);
    applyZoom(tabId, btn.dataset.scale, dp && dp.value ? dp.value : null);
  });
});
const dp = document.getElementById(`datepick-${tabId}`);
dp.addEventListener('change', () => {
  const activeBtn = document.querySelector(`.scale-btn-${tabId}.active`);
  applyZoom(tabId, activeBtn.dataset.scale, dp.value);
});
```

**Composition = toujours un donut, recalculé dynamiquement sur la fenêtre visible** (remplace
entièrement `renderComposition` et son mode bar chart) :

```js
function updateComposition(tabId) {
  const d = TAB_DATA[tabId], range = TABS[tabId].currentXRange;
  let nCalm = 0, nTrend = 0, nStress = 0, total = 0;
  for (let i = 0; i < d.dates.length; i++) {
    if (d.dates[i] >= range[0] && d.dates[i] <= range[1]) {
      total++;
      if (d.regimes[i] === 'calm') nCalm++;
      else if (d.regimes[i] === 'trending') nTrend++;
      else nStress++;
    }
  }
  const pct = v => total ? v / total * 100 : 0;
  const trace = {
    type: 'pie', labels: ['Calme', 'Tendanciel', 'Stress'],
    values: [pct(nCalm), pct(nTrend), pct(nStress)],
    marker: {colors: [REGIME_HEX.calm, REGIME_HEX.trending, REGIME_HEX.stress]},
    hole: 0.42, textinfo: 'label+percent', textfont: {size: 11, color: '#ecf0f1'}, showlegend: false,
  };
  Plotly.react(`chart-dist-${tabId}`, [trace], {
    paper_bgcolor: BG, plot_bgcolor: BG, font: FONT, margin: {l: 10, r: 10, t: 26, b: 10},
    title: {text: `${range[0]} &#8594; ${range[1]} &middot; ${total} j`, font: {size: 10, color: '#7f8c8d'}},
  }, {responsive: true, displayModeBar: false});
}
```

**Fusionner `updateLabelsOnly` avec la mise à jour de la composition**, pour que *tout zoom/pan*
(bouton, date picker, ou molette/glisser-déposer souris natif Plotly) mette à jour la composition
en même temps que les libellés :

```js
function onRangeChange(tabId) {
  const {annotations, overflow} = buildAnnotations(tabId);
  Plotly.relayout(`chart-price-${tabId}`, {annotations});
  const msg = document.getElementById(`declutter-${tabId}`);
  if (msg) msg.style.display = overflow ? 'block' : 'none';
  updateComposition(tabId);
}
```

Remplacer les deux appels existants à `updateLabelsOnly(tabId)` (dans `syncRange`) par
`onRangeChange(tabId)`, et supprimer `updateLabelsOnly`.

**Initialisation** (`initAssetTab`) : remplacer `renderComposition(tabId, 'full')` par
`updateComposition(tabId)` (qui lira `TABS[tabId].currentXRange`, déjà initialisé à
`[first_date, last_date]` — donc au premier rendu la composition couvre tout l'historique, cohérent
avec le bouton "Année" actif par défaut).

---

## 2. Ce qui NE change PAS dans ce patch

- Les 3 checkboxes régime (Calme/Tendanciel/Stress) restent telles quelles (v2 §2).
- Les événements de marché restent toujours visibles, sans checkbox catégorie (v2 §2).
- La palette de couleurs régime (vert/ambre/rouge) et catégories d'événements reste telle quelle
  (v2 §3) — ce n'est pas remise en cause par ce ticket, sauf indication contraire du tuteur.
- L'onglet Comparaison (largeur des régimes, corrélation décalée vol/régime, corrélation glissante
  inter-actifs, tableau stress/calme) reste inchangé (v2 §4).

---

## 3. Tests à ajuster — `test_regime_analytics.py`

- Supprimer les 3 tests et les 2 imports listés en §1 (fonctions supprimées).
- Aucun nouveau test Python n'est nécessaire ici : le zoom et le donut dynamique sont une pure
  logique JavaScript côté frontend (pas de nouvelle fonction dans `regime_analytics.py`). Un
  contrôle visuel manuel (§4) remplace un test automatisé pour cette partie.

---

## 4. Vérification finale

- `pytest calibration/regime/ -v` → tout vert (moins les 3 tests supprimés, plus aucune régression
  sur le reste).
- Régénérer : `python -m calibration.regime.dashboard_builder`.
- Sur chaque onglet actif :
  - À l'ouverture, bouton "Année" actif, vue complète, donut composition = statistiques sur tout
    l'historique (doit être identique à ce que montrait l'ancien mode "Total").
  - Cliquer "Mois" : le graphique prix (+ vol + BOCPD, synchronisés) zoome sur une fenêtre d'~1 an
    se terminant à la dernière date ; le donut composition se met à jour pour ne refléter que
    cette fenêtre visible.
  - Choisir une date dans le calendrier (ex. mars 2020) puis cliquer "Jour" : la fenêtre se
    recentre sur ~2 mois autour de mars 2020, on doit voir distinctement les points/jours
    individuels du crash COVID, et le donut composition doit montrer une forte proportion de
    stress sur cette fenêtre précise.
  - Zoomer/dézoomer à la souris directement sur le graphique (sans passer par les boutons) : le
    donut composition doit aussi se mettre à jour (pas seulement via les boutons).
  - Le calendrier doit se désactiver (grisé) quand "Année" est actif.
  - La composition reste un donut/anneau dans tous les cas — jamais de bar chart.
