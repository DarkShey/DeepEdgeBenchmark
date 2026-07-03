# Corrections v8 — Dashboard Multi-Actifs DEITA (retour du 02/07/2026, suite)

## 0. Sur le "rien ne s'affiche" du graphique v7

J'ai tracé toute la chaîne (payload Python → JSON embarqué → logique JS → appel Plotly) avec un
test automatisé (DOM simulé + Plotly stubé) : les 5 actifs ont des valeurs `avant`/`après` valides
(aucun `null`/`NaN`), `renderVolBeforeAfter('all')` s'exécute sans erreur, et l'appel
`Plotly.react('chart-vol-before-after', ...)` reçoit bien 2 traces avec les bonnes données. Le
code lui-même n'a pas de bug identifiable statiquement.

**Deux causes probables, à vérifier avant de considérer que c'est un bug de plus à corriger :**
1. **Cache navigateur** : si `output/regime_dashboard.html` a été régénéré mais que l'onglet du
   navigateur était déjà ouvert sur l'ancienne version, un simple rechargement (Ctrl+R / Cmd+R, ou
   fermer et rouvrir le fichier) peut suffire.
2. **Erreur silencieuse spécifique au vrai navigateur** (non reproductible dans un DOM simulé) :
   ouvrir la console développeur (F12 → onglet "Console") au moment où l'onglet "Comparaison" est
   cliqué, et regarder s'il y a un message en rouge. S'il y en a un, le copier-coller pour
   diagnostic précis plutôt que deviner.

**En parallèle**, ce patch remplace de toute façon le graphique par une version plus robuste et
plus professionnelle (§1) — donc même si la cause exacte du "rien ne s'affiche" reste incertaine,
la nouvelle version règle aussi le problème de fond design relevé (cf. §1, notamment le mélange
d'échelles très différentes entre crypto et obligations sur un même axe).

---

## 1. Refonte : présentation "professionnelle" du signal vol-avant/après

**Problème design de la v7** (au-delà du bug d'affichage) : comparer les valeurs absolues de σₜ de
BTC (~3-4%) et de ZN=F (~0.3%) sur le même axe Y écrase visuellement les obligations — leurs
barres seraient minuscules à côté de celles des cryptos, rendant la comparaison inter-actifs
inexploitable. Un vrai professionnel normaliserait plutôt sur la **variation relative** (%),
seule grandeur comparable entre classes d'actifs d'échelles de volatilité très différentes.

**Nouvelle structure, en 2 niveaux de lecture (pattern standard des notes de recherche) :**
1. **Un chiffre clé en tête** : la variation moyenne (tous actifs confondus) de la volatilité
   après vs avant un changement de régime, avec une phrase d'interprétation.
2. **Un graphique en barres divergentes, horizontal, trié par ampleur** : un seul chiffre par
   actif (le delta en %), pas deux barres à comparer mentalement — la lecture "quel actif réagit
   le plus" saute aux yeux immédiatement.

Le détail des valeurs brutes (avant/après en unités réelles) reste disponible mais démoté dans un
tableau repliable (`<details>`, même pattern que la table d'événements) pour ceux qui veulent
vérifier les chiffres sous-jacents — pas dans le graphique principal.

### Backend — aucun nouveau calcul, réutilise `vol_before_after` existant

`COMPARISON.vol_before_after[short][mode]` contient déjà `avant`, `apres`, `delta_pct` — rien à
changer côté `regime_analytics.py`/calcul. Seul le rendu JS change.

### HTML — remplace le bloc `chart-vol-before-after` de v7

```html
<div class="card">
  <div class="card-label">La volatilit&#233; augmente-t-elle avant ou apr&#232;s un changement de r&#233;gime&nbsp;?</div>
  <p class="chart-note">Variation de la volatilit&#233; moyenne entre les 10 jours qui pr&#233;c&#232;dent un
    changement de r&#233;gime et les 10 jours qui suivent (jour du changement inclus). Valeurs en
    variation relative (%) pour rester comparables entre crypto et obligations, dont les niveaux de
    volatilit&#233; absolus n'ont rien &#224; voir.</p>

  <div class="scale-sel">
    <button class="va-btn va-btn-active" data-mode="all">Tous les changements de r&#233;gime</button>
    <button class="va-btn" data-mode="into_stress">Uniquement vers le stress</button>
  </div>

  <div class="headline-stat">
    <div class="headline-value" id="vol-headline-value">&#8212;</div>
    <div class="headline-label" id="vol-headline-label"></div>
  </div>

  <div id="chart-vol-before-after" style="height:280px"></div>

  <details style="margin-top:8px">
    <summary>Voir les valeurs brutes (&#963;&#8339; moyen, avant/apr&#232;s, par actif)</summary>
    <div style="overflow-x:auto;margin-top:6px">
      <table><thead><tr><th>Actif</th><th>Avant</th><th>Apr&#232;s</th><th>Variation</th><th>n transitions</th></tr></thead>
      <tbody id="vol-before-after-table"></tbody></table>
    </div>
  </details>
</div>
```

CSS à ajouter :
```css
.headline-stat{ text-align:center; margin:10px 0 16px; }
.headline-value{ font-size:2.1rem; font-weight:700; }
.headline-value.up{ color:#e74c3c; }
.headline-value.down{ color:#2ecc71; }
.headline-label{ font-size:.78rem; color:#7f8c8d; margin-top:2px; }
```

### JS — remplace entièrement `renderVolBeforeAfter`

```js
function renderVolBeforeAfter(mode) {
  const shorts = ASSETS.map(a => a.short);
  let rows = shorts.map((s, i) => ({
    short: s,
    label: ASSETS[i].label,
    color: ASSETS[i].color,
    ...COMPARISON.vol_before_after[s][mode],
  }));

  // Trier par amplitude de variation, du plus fort au plus faible (barres divergentes lisibles)
  rows = rows.filter(r => r.delta_pct !== null).sort((a, b) => b.delta_pct - a.delta_pct);

  // ── Chiffre clé en tête : moyenne des deltas ────────────────────────────────
  const avgDelta = rows.reduce((s, r) => s + r.delta_pct, 0) / rows.length;
  const headlineEl = document.getElementById('vol-headline-value');
  const labelEl = document.getElementById('vol-headline-label');
  headlineEl.textContent = `${avgDelta >= 0 ? '+' : ''}${avgDelta.toFixed(1)}%`;
  headlineEl.className = 'headline-value ' + (avgDelta >= 0 ? 'up' : 'down');
  labelEl.textContent = avgDelta >= 0
    ? "volatilité moyenne plus élevée APRÈS un changement de régime — la vol suit le changement, elle ne l'annonce pas clairement en moyenne"
    : "volatilité moyenne plus élevée AVANT un changement de régime — signal avancé en moyenne sur l'ensemble des actifs";

  // ── Barres divergentes horizontales, une valeur par actif ───────────────────
  const trace = {
    type: 'bar', orientation: 'h',
    x: rows.map(r => r.delta_pct),
    y: rows.map(r => r.short),
    marker: {color: rows.map(r => r.delta_pct >= 0 ? '#e74c3c' : '#2ecc71')},
    text: rows.map(r => `${r.delta_pct >= 0 ? '+' : ''}${r.delta_pct.toFixed(1)}%`),
    textposition: 'outside',
    hovertemplate: '%{y} : %{x:.1f}%<extra></extra>',
  };
  Plotly.newPlot('chart-vol-before-after', [trace], Object.assign({}, baseLayout(), {
    margin: {l: 50, r: 40, t: 10, b: 34},
    xaxis: {title: 'Variation de volatilité, après vs avant (%)', gridcolor: GRID, zeroline: true, zerolinewidth: 2, zerolinecolor: '#566573'},
    yaxis: {type: 'category', gridcolor: GRID, categoryorder: 'array', categoryarray: rows.map(r => r.short).slice().reverse()},
    showlegend: false,
  }), {responsive: true, displayModeBar: false});

  // ── Détail brut dans la table repliable ──────────────────────────────────────
  const tbody = document.getElementById('vol-before-after-table');
  tbody.innerHTML = '';
  const nKey = mode === 'all' ? 'n_all' : 'n_stress';
  shorts.forEach(s => {
    const d = COMPARISON.vol_before_after[s][mode];
    const n = COMPARISON.vol_before_after[s][nKey];
    const tr = document.createElement('tr');
    const fmt = v => v !== null ? v.toFixed(3) : '—';
    tr.innerHTML = `<td>${s}</td><td>${fmt(d.avant)}</td><td>${fmt(d.apres)}</td>` +
      `<td>${d.delta_pct !== null ? (d.delta_pct>=0?'+':'')+d.delta_pct.toFixed(1)+'%' : '—'}</td><td>${n}</td>`;
    tbody.appendChild(tr);
  });
}
```

**Note d'implémentation** : utiliser `Plotly.newPlot` (pas `Plotly.react`) à chaque appel ici —
comme ce graphique change de forme selon le mode (nombre de barres identique mais tri/couleurs
recalculés), repartir d'un état propre à chaque fois est plus sûr et évite tout état résiduel
d'un rendu précédent.

Garder le listener existant sur `.va-btn` (aucun changement) — il appelle déjà
`renderVolBeforeAfter(btn.dataset.mode)`, qui pointe maintenant vers cette nouvelle implémentation.

Supprimer l'ancien `<p class="chart-note" id="vol-before-after-conclusion">` (remplacé par le
bloc `.headline-stat`, plus visuel qu'une phrase en petit texte sous le graphique).

---

## 2. Vérification finale

- Avant de considérer quoi que ce soit résolu, confirmer d'abord si le rechargement navigateur
  (§0) suffisait à faire apparaître l'ancien graphique v7 — si oui, c'était un problème de cache,
  pas de code, mais la refonte §1 reste appliquée car elle corrige un vrai problème de design
  (échelles incomparables) indépendant du bug d'affichage.
- Régénérer `output/regime_dashboard.html`, ouvrir l'onglet Comparaison : un chiffre clé en gros
  (ex. "+15%") avec une phrase d'interprétation, un graphique à barres horizontales triées (une
  barre par actif, rouge si hausse/vert si baisse de volatilité après transition), et un tableau
  de détail repliable avec les valeurs brutes.
- Basculer sur "Uniquement vers le stress" : le chiffre clé, le tri des barres et le tableau se
  mettent à jour.
- Vérifier avec la console développeur (F12) qu'aucune erreur n'apparaît en cliquant sur l'onglet
  Comparaison ni sur les 2 boutons de bascule.
