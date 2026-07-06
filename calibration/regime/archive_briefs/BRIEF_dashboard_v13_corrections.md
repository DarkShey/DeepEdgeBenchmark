# BRIEF v13 — Partie 2 : déclencheur de fin de régime (sortie du stress)

## Objectif

Symétrique de la Partie 1 (BRIEF v10/v11/v12), mais sur la fin du régime stress plutôt que son début : est-ce que la volatilité et/ou le volume redescendent avant la sortie du régime stress (signal précurseur de fin de crise), ou seulement après (confirmation tardive) ?

Décision actée avant ce brief : on étudie spécifiquement la sortie du régime stress (peu importe le régime suivant), pas toutes les fins de régime poolées ensemble. Ça demande un ajustement de `regime_transition_vol_profile()` : le paramètre existant `only_into` filtre sur le régime de **destination**, pas sur celui dont on sort — il faut un nouveau paramètre pour filtrer sur le régime de départ.

## Modifications à faire

### 1. `regime_analytics.py` — ajouter `only_from` à `regime_transition_vol_profile()`

Ajouter un paramètre `only_from: str | None = None` à la signature. Dans la boucle sur `candidate_positions`, ajouter un filtre supplémentaire (en plus du filtre `only_into` existant) :

```python
if only_from is not None and segments["regime"].iloc[i] != only_from:
    continue
```

Ce filtre porte sur `segments["regime"]` directement (le régime du segment qui se termine), pas sur `regime_at_event` (qui pour `alignment="end"` est déjà décalé vers le régime suivant — cf. code existant). `only_from` et `only_into` peuvent être combinés (ex. `only_from="stress", only_into="calm"` pour ne garder que les sorties de stress vers le calme spécifiquement) mais dans ce brief on n'utilise que `only_from="stress"` seul.

Mettre à jour la docstring pour expliquer ce nouveau paramètre, sur le même modèle que la documentation existante de `only_into`.

### 2. `dashboard_builder.py` — `compute_all_analytics()` : nouveaux profils

Dans la boucle `for asset in ASSETS:`, à côté des profils existants (`profile_all`, `profile_into_stress`, `profile_into_stress_volume`), ajouter :

```python
profile_out_of_stress = ra.regime_transition_vol_profile(
    history, window=10, alignment="end", only_from="stress"
)
profile_out_of_stress_volume = ra.regime_transition_vol_profile(
    history, window=10, alignment="end", only_from="stress", column="volume_norm"
)
```

Les ajouter dans `per_asset[ticker]` sous les clés `"profile_out_of_stress"` et `"profile_out_of_stress_volume"`.

### 3. `dashboard_builder.py` — `_comparison_payload()` : deux nouvelles séries indexées

Réutiliser `_event_study_series()` (déjà créée en v12) — aucune modification de cette fonction n'est nécessaire, elle est déjà assez générique (la baseline reste "-10 à -5 jours avant l'événement", ce qui correspond ici à une période encore bien à l'intérieur du régime stress, avant que la sortie n'approche — c'est la bonne référence pour détecter une décrue précoce).

```python
event_study_end = {
    asset["short"]: _event_study_series(analytics["per_asset"][asset["ticker"]]["profile_out_of_stress"], asset)
    for asset in ASSETS
}
event_study_end_volume = {
    asset["short"]: _event_study_series(analytics["per_asset"][asset["ticker"]]["profile_out_of_stress_volume"], asset)
    for asset in ASSETS
}
```

Ajouter `"event_study_end": event_study_end` et `"event_study_end_volume": event_study_end_volume` dans le dict retourné par `_comparison_payload()`.

### 4. `dashboard_builder.py` (HTML) — deux nouvelles cartes

Ajouter, après les deux cartes de la Partie 1 (début de régime), deux cartes symétriques pour la fin :

```html
<div class="card">
  <div class="card-label">La volatilit&#233; recule-t-elle avant ou apr&#232;s la sortie du r&#233;gime stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
  <p class="chart-note">Volatilit&#233; moyenne autour de chaque sortie du r&#233;gime stress (jour 0 = dernier jour
    avant la sortie, quel que soit le r&#233;gime suivant), index&#233;e sur la p&#233;riode -10&#224;-5 jours = 0%.
    Un recul qui commence AVANT le jour 0 sugg&#233;rerait un signal pr&#233;curseur de sortie de crise&nbsp;;
    un recul qui ne commence qu'APR&#200;S sugg&#233;rerait que la vol ne fait que confirmer, avec retard,
    une sortie de crise d&#233;j&#224; act&#233;e par le mod&#232;le.</p>
  <div id="chart-event-study-end" style="height:380px"></div>
  <details style="margin-top:8px">
    <summary>Voir le d&#233;tail par actif (jour de premi&#232;re r&#233;action, nombre d'&#233;v&#233;nements)</summary>
    <div style="overflow-x:auto;margin-top:6px">
      <table><thead><tr><th>Actif</th><th>Premier jour de r&#233;action</th><th>n sorties de stress</th></tr></thead>
      <tbody id="event-study-end-table"></tbody></table>
    </div>
  </details>
</div>

<div class="card">
  <div class="card-label">Le volume recule-t-il avant ou apr&#232;s la sortie du r&#233;gime stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
  <p class="chart-note">M&#234;me m&#233;thode que ci-dessus, appliqu&#233;e au volume normalis&#233;.</p>
  <div id="chart-event-study-end-volume" style="height:380px"></div>
  <details style="margin-top:8px">
    <summary>Voir le d&#233;tail par actif (jour de premi&#232;re r&#233;action, nombre d'&#233;v&#233;nements)</summary>
    <div style="overflow-x:auto;margin-top:6px">
      <table><thead><tr><th>Actif</th><th>Premier jour de r&#233;action</th><th>n sorties de stress</th></tr></thead>
      <tbody id="event-study-end-volume-table"></tbody></table>
    </div>
  </details>
</div>
```

### 5. `dashboard_builder.py` (JS) — étendre `renderEventStudy()` et l'appeler 2 fois de plus

`renderEventStudy()` (créée en v12) doit accepter un paramètre supplémentaire `xAxisTitle` (actuellement le texte de l'axe x est écrit en dur pour le cas "début de régime" — il faut le paramétrer, car le cas "fin de régime" a un jour 0 différent) :

```js
function renderEventStudy(data, chartId, tableId, xAxisTitle, yAxisTitle) {{
  // ... corps inchangé, sauf :
  xaxis: {{ title: xAxisTitle, gridcolor: GRID, dtick: 1, zeroline: false }},
  // ... reste inchangé
}}
```

Mettre à jour les 2 appels existants (v11/v12) pour passer explicitement leur `xAxisTitle` :

```js
renderEventStudy(COMPARISON.event_study, 'chart-event-study', 'event-study-table',
  'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)',
  '&#201;cart de volatilit&#233; vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)');
renderEventStudy(COMPARISON.event_study_volume, 'chart-event-study-volume', 'event-study-volume-table',
  'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)',
  '&#201;cart de volume vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)');
```

Et ajouter les 2 nouveaux appels :

```js
renderEventStudy(COMPARISON.event_study_end, 'chart-event-study-end', 'event-study-end-table',
  'Jours relatifs &#224; la sortie du r&#233;gime stress (0 = dernier jour avant la sortie)',
  '&#201;cart de volatilit&#233; vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)');
renderEventStudy(COMPARISON.event_study_end_volume, 'chart-event-study-end-volume', 'event-study-end-volume-table',
  'Jours relatifs &#224; la sortie du r&#233;gime stress (0 = dernier jour avant la sortie)',
  '&#201;cart de volume vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)');
```

## Vérifications demandées après implémentation

1. La page Comparaison affiche maintenant 4 graphiques d'étude d'événement (2 pour le début de régime stress, 2 pour la sortie), plus le box plot inchangé.
2. Les 2 graphiques existants (début de régime, v11/v12) sont visuellement identiques à avant ce brief.
3. `only_from` fonctionne correctement : vérifier que `profile_out_of_stress` ne contient que des segments dont le régime **de départ** est stress (pas seulement des segments qui aboutissent à un régime précis).
4. Aucune erreur JS en console sur l'onglet Comparaison.
5. Les tests existants de `regime_analytics.py` passent toujours (l'ajout de `only_from` avec valeur par défaut `None` ne doit rien casser).
