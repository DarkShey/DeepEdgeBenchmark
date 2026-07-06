# BRIEF v14 — Partie 2 : déclencheur de fin de régime (sortie du stress)

**Ce brief remplace `BRIEF_dashboard_v13_corrections.md`, qui n'a jamais été exécuté et dont certains détails ne correspondent plus au code actuel** (Claude Code a implémenté Q1/Q2/Q3 plus loin que prévu entre-temps — étude d'événement + test formel de causalité de Granger + cross-corrélation vol/volume, tous déjà en place et testés). Ce brief-ci est réécrit pour s'appuyer exactement sur les noms de fonctions et la structure réellement présents dans le code aujourd'hui (`_event_study_from_profile`, `granger_causality_*_to_stress`, `_comparison_payload`, `renderEventStudy`).

## Objectif

Symétrique de la Partie 1, mais sur la sortie du régime stress plutôt que son entrée : est-ce que la volatilité et/ou le volume redescendent avant la sortie du régime stress (signal précurseur de fin de crise), ou seulement après (confirmation tardive) ?

Décision actée avant ce brief : on étudie spécifiquement la sortie du régime stress (peu importe le régime suivant), pas toutes les fins de régime poolées ensemble.

**Ce que ce brief ajoute** : 2 nouveaux graphiques d'étude d'événement (vol, volume), sur le modèle exact de ceux de la Partie 1.

**Ce que ce brief n'ajoute PAS, et pourquoi** : pas de nouveau test de Granger pour la sortie de stress. Les tests de Granger existants (`granger_causality_vol_to_stress`/`_volume_to_stress`) tournent sur `p_stress` en continu sur tout l'historique, pas sur un sous-ensemble d'événements ponctuels — c'est une contrainte structurelle du test (la régression décalée exige une série temporelle continue, pas des fenêtres découpées autour d'événements). Ils captent donc déjà, de façon générale, la relation prédictive entre vol/volume et l'état de stress dans le temps — y compris implicitement autour des sorties de crise. L'étude d'événement (indicative, pas un test formel) reste le bon outil pour la question "à quoi ressemble le comportement spécifiquement autour des sorties de stress", exactement comme en Partie 1.

## Modifications à faire

### 1. `regime_analytics.py` — ajouter `only_from` à `regime_transition_vol_profile()`

Signature actuelle : `regime_transition_vol_profile(df, window=10, alignment="start", only_into=None, column="sigma_t")`. Ajouter le paramètre `only_from: str | None = None` juste après `only_into` dans la liste (avant `column`) — position purement pour la lisibilité : tous les appels existants dans le code et les tests utilisent des arguments nommés (aucun appel positionnel au-delà de `df`), donc l'ordre exact n'a aucune incidence sur la compatibilité, vérifié par lecture de tous les appels existants.

Dans la boucle sur `candidate_positions` (juste avant ou après le `if only_into is not None and regime_at_event.iloc[i] != only_into: continue` existant), ajouter :

```python
if only_from is not None and segments["regime"].iloc[i] != only_from:
    continue
```

Ce filtre porte sur `segments["regime"]` directement (le régime du segment qui se termine), pas sur `regime_at_event` (qui pour `alignment="end"` est déjà décalé vers le régime suivant). `only_from` et `only_into` peuvent être combinés, mais dans ce brief on n'utilise que `only_from="stress"` seul. Mettre à jour la docstring sur le modèle de la documentation existante de `only_into`.

Test unitaire à ajouter dans `test_regime_analytics.py`, sur le modèle exact de `test_regime_transition_vol_profile_peak_near_event` déjà présente :

```python
def test_regime_transition_vol_profile_only_from_filters_on_departing_regime():
    window = 5
    # 3 segments : calm(0-29), stress(30-59), calm(60-89) -> 2 transitions en alignment="end" :
    # fin du segment calm (pos 29, régime de départ = calm) et fin du segment stress (pos 59,
    # régime de départ = stress). only_from="stress" ne doit garder QUE la seconde, quel que
    # soit le régime suivant (ici calm) — à la différence de only_into, qui filtrerait sur le
    # régime suivant et non sur celui qui se termine.
    regimes = ["calm"] * 30 + ["stress"] * 30 + ["calm"] * 30
    df = _make_sigma_history(regimes, np.ones(len(regimes)))

    profile = regime_transition_vol_profile(df, window=window, alignment="end", only_from="stress")

    assert (profile["n_events"] == 1).all()
```

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

Réutiliser `_event_study_from_profile()` telle quelle (aucune modification nécessaire) :

```python
event_study_end = {}
event_study_end_volume = {}
for asset in ASSETS:
    ticker = asset["ticker"]
    event_study_end[asset["short"]] = _event_study_from_profile(
        analytics["per_asset"][ticker]["profile_out_of_stress"], asset["label"], asset["color"]
    )
    event_study_end_volume[asset["short"]] = _event_study_from_profile(
        analytics["per_asset"][ticker]["profile_out_of_stress_volume"], asset["label"], asset["color"]
    )
```

Ajouter `"event_study_end": event_study_end` et `"event_study_end_volume": event_study_end_volume` au dict retourné par `_comparison_payload()`.

### 4. `dashboard_builder.py` (HTML) — section "Partie 2" + deux nouvelles cartes

Ajouter, après la carte de cross-corrélation vol/volume (dernière carte actuelle de l'onglet Comparaison), d'abord un séparateur visuel marquant le passage à la Partie 2 (la page devient dense, mieux vaut la structurer clairement plutôt que d'enchaîner sans repère) :

```html
<h2 style="font-size:1rem;color:#95a5a6;border-top:1px solid #1c2a3a;padding-top:14px;margin-top:18px">
  Partie 2 &#8212; D&#233;clencheur de fin de r&#233;gime (sortie du stress)
</h2>
```

Puis les deux cartes, sur le modèle exact des cartes "annonce-t-elle ou confirme-t-elle un passage en stress" existantes :

```html
<div class="card">
  <div class="card-label">La volatilit&#233; recule-t-elle avant ou apr&#232;s la sortie du r&#233;gime stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
  <p class="chart-note">Volatilit&#233; moyenne autour de chaque sortie du r&#233;gime stress (jour 0 = dernier jour
    avant la sortie, quel que soit le r&#233;gime suivant), index&#233;e sur la p&#233;riode -10&#224;-5 jours = 0%
    &#8212; m&#234;me m&#233;thode que la Partie 1. Un losange marque le premier jour o&#249; l'&#233;cart d&#233;passe 1,96&#215;
    l'erreur standard de la p&#233;riode de r&#233;f&#233;rence (indicatif, pas un test statistique formel &#8212; cf.
    les tests de Granger de la Partie 1 pour la preuve formelle, qui couvrent d&#233;j&#224; la relation
    g&#233;n&#233;rale vol/volume vs stress dans le temps).</p>
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

### 5. `dashboard_builder.py` (JS) — paramétrer `renderEventStudy()` et l'appeler 2 fois de plus

`renderEventStudy(data, chartId, tableId)` a actuellement le texte de l'axe x écrit en dur ("Jours relatifs au début du régime stress..."). Lui ajouter un 4e paramètre `xAxisTitle` :

```js
function renderEventStudy(data, chartId, tableId, xAxisTitle) {{
  // ... corps inchangé, sauf la ligne :
  xaxis: {{ title: xAxisTitle, gridcolor: GRID, dtick: 1, zeroline: false }},
  // ... reste inchangé
}}
```

Mettre à jour les 2 appels existants pour passer explicitement leur titre actuel (aucun changement visuel) :

```js
renderEventStudy(COMPARISON.event_study, 'chart-event-study', 'event-study-table',
  'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)');
renderEventStudy(COMPARISON.event_study_volume, 'chart-event-study-volume', 'event-study-volume-table',
  'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)');
```

Et ajouter les 2 nouveaux appels :

```js
renderEventStudy(COMPARISON.event_study_end, 'chart-event-study-end', 'event-study-end-table',
  'Jours relatifs &#224; la sortie du r&#233;gime stress (0 = dernier jour avant la sortie)');
renderEventStudy(COMPARISON.event_study_end_volume, 'chart-event-study-end-volume', 'event-study-end-volume-table',
  'Jours relatifs &#224; la sortie du r&#233;gime stress (0 = dernier jour avant la sortie)');
```

## Vérifications demandées après implémentation

1. La page Comparaison affiche maintenant, sous un séparateur "Partie 2", les 2 nouveaux graphiques (vol, volume autour de la sortie du stress).
2. Les 5 blocs existants (box plot, Q1 événement + Granger, Q2 événement + Granger, Q3 CCF) sont visuellement identiques à avant ce brief.
3. `only_from` fonctionne correctement : vérifier avec un test unitaire ou un print de contrôle que `profile_out_of_stress` ne contient que des segments dont le régime **de départ** est stress, quel que soit le régime suivant (contrairement à `only_into` qui filtre sur le régime suivant).
4. Aucune erreur JS en console sur l'onglet Comparaison.
5. Les 25 tests de `test_regime_analytics.py` passent toujours, plus les nouveaux tests écrits pour `only_from` (au moins un test vérifiant qu'il filtre bien sur le régime de départ et pas sur le régime de destination, et un test vérifiant que `only_from` et `only_into` peuvent être combinés).
