# BRIEF v11 — Question 1 : la volatilité précède-t-elle ou suit-elle le début d'un régime stress ?

## Objectif

Répondre visuellement à la question : "la volatilité augmente-t-elle avant, ou seulement après, le basculement vers le régime stress ?" — et si oui de combien de jours. Le résultat remplace, dans la page Comparaison, le bloc "vol avant/après" actuel (résumé à 2 chiffres, trop pauvre) par une vraie courbe d'étude d'événement.

**Périmètre de cette V1** : uniquement la question "volatilité vs début de régime". Les questions 2 (volume), 3 (cross-corrélation vol/volume) et 4 (Granger) suivront dans des briefs séparés une fois celui-ci validé.

**Décision de nettoyage de la page Comparaison, actée avant ce brief** : la page Comparaison est réduite à deux blocs seulement — le box plot "Largeur des régimes" (inchangé) et le nouveau graphique d'étude d'événement décrit ci-dessous. Les blocs "vol avant/après", "corrélation glissante inter-actifs" et "corrélation stress vs calme par paire" sont retirés de la page. Important : ne pas supprimer les fonctions `rolling_cross_correlation`, `pairwise_stress_calm_correlation`, `fisher_r_critical`, `correlation_significance` ni leurs tests dans `regime_analytics.py`/`test_regime_analytics.py` — elles restent du code valide et testé, seulement retiré de l'affichage du dashboard pour l'instant.

## Modifications à faire

### 1. `regime_analytics.py` — généraliser `regime_transition_vol_profile` à une colonne quelconque

Actuellement la fonction est câblée en dur sur `df["sigma_t"]`. Ajouter un paramètre `column: str = "sigma_t"` à la signature, et l'utiliser à la place de `df["sigma_t"]` dans le corps de la fonction (variable `sigma = df[column]`). Renommer en conséquence les colonnes de sortie de façon générique si besoin (`mean_value`, `std_value` au lieu de `mean_sigma`, `std_sigma`) — si ce renommage casse des appels existants ou les tests, garder `mean_sigma`/`std_sigma` comme noms de colonnes par défaut (valeur de `column` n'affecte que la source, pas le nom des colonnes de sortie) pour limiter le risque de régression. Comportement par défaut (`column="sigma_t"`) inchangé.

### 2. `dashboard_builder.py` — `compute_all_analytics()` : retirer le calcul cross-actifs, ajouter le profil volume

Retirer de cette fonction tout ce qui sert uniquement à `cross_correlation` / `pairwise_stress_calm` / `market_stress_majority`, qui ne sont plus utilisés une fois la page nettoyée : les dicts `returns_by_asset`, `sigma_by_asset`, `volume_by_asset`, `swing_by_asset`, `stress_masks`, `calm_masks`, le calcul de `cross_correlation`, `pairwise_stress_calm`, `stress_count`/`market_stress_majority`, et les clés correspondantes dans le dict retourné (`"comparison"` ne garde que `"combined_segments"`).

Vérifier avant de supprimer qu'aucune de ces variables n'est utilisée ailleurs dans la fonction (ex. `returns_by_asset` ne sert qu'à `cross_correlation` et `pairwise_stress_calm` — confirmer par une lecture rapide du corps de la fonction avant de couper).

Dans la boucle `for asset in ASSETS:`, à côté de la ligne existante :

```python
profile_into_stress = ra.regime_transition_vol_profile(
    history, window=10, alignment="start", only_into="stress"
)
```

ajouter :

```python
profile_into_stress_volume = ra.regime_transition_vol_profile(
    history, window=10, alignment="start", only_into="stress", column="volume_norm"
)
```

et l'ajouter dans le dict `per_asset[ticker]` sous la clé `"profile_into_stress_volume"` (elle ne sera pas utilisée dans ce brief-ci, mais sera nécessaire pour la question 2 — autant l'avoir déjà en place).

### 3. `dashboard_builder.py` — `_comparison_payload()` : construire le payload de l'étude d'événement

Retirer la construction de `vol_before_after`, `cross_correlation_payload`, `stress_bands`, `pairs_table` et les clés correspondantes du dict retourné.

Ajouter une nouvelle section qui construit, pour chaque actif, une série indexée et un jour de première réaction significative :

```python
event_study = {}
for asset in ASSETS:
    ticker = asset["ticker"]
    profile = analytics["per_asset"][ticker]["profile_into_stress"]  # colonnes: rel_day, mean_sigma, std_sigma, n_events
    if profile["n_events"].iloc[0] == 0 or profile["mean_sigma"].isna().all():
        event_study[asset["short"]] = {
            "label": asset["label"], "color": asset["color"],
            "rel_day": profile["rel_day"].tolist(), "sigma_index": [None] * len(profile),
            "n_events": 0, "first_reaction_day": None,
        }
        continue

    baseline_mask = profile["rel_day"].between(-10, -5)
    baseline = profile.loc[baseline_mask, "mean_sigma"].mean()
    n_events = int(profile["n_events"].iloc[0])

    sigma_index = ((profile["mean_sigma"] / baseline - 1.0) * 100.0)

    # Erreur standard de la moyenne à chaque jour relatif -> seuil de déviation "significative"
    se = profile["std_sigma"] / np.sqrt(n_events)
    deviation = (profile["mean_sigma"] - baseline).abs()
    significant = deviation > se
    first_reaction_day = None
    for rel_day, is_sig in zip(profile["rel_day"], significant):
        if is_sig:
            first_reaction_day = int(rel_day)
            break

    event_study[asset["short"]] = {
        "label": asset["label"], "color": asset["color"],
        "rel_day": profile["rel_day"].tolist(),
        "sigma_index": [_num(v) if not np.isnan(v) else None for v in sigma_index],
        "n_events": n_events,
        "first_reaction_day": first_reaction_day,
    }
```

Ajouter `"event_study": event_study` dans le dict retourné par `_comparison_payload()`.

Note de rigueur : le "premier jour de réaction significative" est défini comme le premier jour relatif (en parcourant de -10 à +10) où l'écart moyen à la baseline dépasse une erreur standard — c'est un seuil raisonnable mais pas un test formel ; le présenter dans l'UI comme indicatif, pas comme une significativité statistique au sens strict (celle-là est réservée au test de Granger prévu pour la question 4).

### 4. `dashboard_builder.py` — HTML : remplacer les 3 cartes retirées par une seule carte

Dans le bloc `<div class="tab-panel" data-tab="COMPARISON">`, garder la carte "Largeur des régimes" telle quelle. Retirer entièrement les 3 cartes suivantes : "La volatilité augmente-t-elle avant ou après un passage en stress ?" (contient `chart-vol-before-after`), "Corrélation glissante inter-actifs" (contient `chart-crosscorr`), et "Corrélation moyenne inter-actifs — stress vs calme" (contient `pairs-body`).

Les remplacer par une seule nouvelle carte :

```html
<div class="card">
  <div class="card-label">La volatilit&#233; annonce-t-elle ou confirme-t-elle un passage en stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
  <p class="chart-note">Volatilit&#233; moyenne autour de chaque entr&#233;e en r&#233;gime stress (jour 0 = jour du
    basculement), index&#233;e sur la p&#233;riode -10&#224;-5 jours = 0% pour rester comparable entre crypto et
    obligations. Un losange marque, pour chaque actif, le premier jour o&#249; l'&#233;cart &#224; la p&#233;riode
    pr&#233;-&#233;v&#233;nement d&#233;passe l'erreur standard &#8212; indicatif, pas un test statistique formel.</p>
  <div id="chart-event-study" style="height:380px"></div>
  <details style="margin-top:8px">
    <summary>Voir le d&#233;tail par actif (jour de premi&#232;re r&#233;action, nombre d'&#233;v&#233;nements)</summary>
    <div style="overflow-x:auto;margin-top:6px">
      <table><thead><tr><th>Actif</th><th>Premier jour de r&#233;action</th><th>n transitions vers stress</th></tr></thead>
      <tbody id="event-study-table"></tbody></table>
    </div>
  </details>
</div>
```

### 5. `dashboard_builder.py` — JS `initComparisonTab()` : retirer l'ancien code, ajouter le nouveau

Retirer entièrement `renderVolBeforeAfter()` et son appel, le bloc de corrélation glissante (`CC_PALETTE`, `CC_AXIS_TITLES`, `CC_INTRA_CLASS`, `buildCcTraces`, `stressShapes`, `ccLayout`, le `Plotly.newPlot('chart-crosscorr', ...)`, les listeners `.cc-btn`), et la boucle `pairs-body`. Garder le bloc `chart-box` (box plot) inchangé.

Ajouter :

```js
// ── Étude d'événement : volatilité indexée autour de l'entrée en stress ──────────
const es = COMPARISON.event_study;
const esTraces = Object.keys(es).map(short => {{
  const a = es[short];
  return {{
    type: 'scatter', mode: 'lines', name: short,
    x: a.rel_day, y: a.sigma_index,
    line: {{ color: a.color, width: 2 }},
    legendgroup: short,
    hovertemplate: `${{short}} : %{{y:.1f}}%<extra></extra>`,
  }};
}});
const esMarkers = Object.keys(es).map(short => {{
  const a = es[short];
  const y = a.rel_day.map(d => d === a.first_reaction_day ? a.sigma_index[a.rel_day.indexOf(d)] : null);
  return {{
    type: 'scatter', mode: 'markers', name: short, legendgroup: short, showlegend: false,
    x: a.rel_day, y: y,
    marker: {{ color: a.color, size: 11, symbol: 'diamond', line: {{ color: '#fff', width: 1 }} }},
    hovertemplate: `${{short}} : premi&#232;re r&#233;action au jour %{{x}}<extra></extra>`,
  }};
}});
Plotly.newPlot('chart-event-study', [...esTraces, ...esMarkers], Object.assign({{}}, baseLayout(), {{
  margin: {{ l: 55, r: 18, t: 10, b: 45 }},
  xaxis: {{ title: 'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)',
            gridcolor: GRID, dtick: 1, zeroline: false }},
  yaxis: {{ title: '&#201;cart de volatilit&#233; vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)',
            gridcolor: GRID, zeroline: true, zerolinewidth: 2, zerolinecolor: '#566573' }},
  shapes: [{{ type: 'line', x0: 0, x1: 0, xref: 'x', y0: 0, y1: 1, yref: 'paper',
             line: {{ color: '#7f8c8d', width: 1, dash: 'dash' }} }}],
}}), {{ responsive: true, displayModeBar: false }});

const esBody = document.getElementById('event-study-table');
Object.keys(es).forEach(short => {{
  const a = es[short];
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{short}}</td>` +
    `<td>${{a.first_reaction_day !== null ? (a.first_reaction_day >= 0 ? '+' : '') + a.first_reaction_day + ' j' : 'aucun &#233;cart significatif d&#233;tect&#233;'}}</td>` +
    `<td>${{a.n_events}}</td>`;
  esBody.appendChild(tr);
}});
```

## Vérifications demandées après implémentation

1. Le nouveau graphique s'affiche dans l'onglet Comparaison avec 5 courbes (une par actif) + les losanges de première réaction là où ils existent.
2. Le box plot "Largeur des régimes" est toujours présent et inchangé.
3. Les 3 anciens blocs (vol avant/après, corrélation glissante, tableau de paires) ont bien disparu de la page.
4. `regime_analytics.py` et `test_regime_analytics.py` n'ont pas perdu de fonctions ni de tests — seul `dashboard_builder.py` a changé.
5. Les onglets par actif (BTC/ETH/SPX/ZN/TLT), le panneau volume ajouté au brief précédent, et le zoom ne sont pas affectés par ce brief.
6. Vérifier dans la console navigateur qu'il n'y a pas d'erreur JS au chargement de l'onglet Comparaison.
