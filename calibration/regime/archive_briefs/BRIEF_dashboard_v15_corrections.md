# BRIEF v15 — Extension Partie 1 : indicateurs de force/momentum (MACD, Momentum, RoC, RSI, Stochastique)

**Prérequis : `BRIEF_dashboard_v14_corrections.md` doit être exécuté avant celui-ci** — ce brief suppose que `renderEventStudy()` accepte déjà un 4e paramètre `xAxisTitle` (ajouté en v14) et que la Partie 2 (sortie de stress) est en place.

## Objectif

Étendre la Partie 1 (vol/volume vs début de régime stress) à 5 indicateurs de force/momentum supplémentaires : MACD (histogramme), Momentum (non borné), RoC, RSI, Stochastique (%K). Décision actée avant ce brief : ces indicateurs restent une couche de confirmation à côté du HMM (le modèle de classification des régimes n'est pas modifié), et on garde les 4 oscillateurs de momentum (au lieu d'en choisir un seul) pour comparer explicitement leur utilité.

Méthode : strictement la même que pour Q1/Q2 (étude d'événement indicative + test formel de causalité de Granger vers `p_stress`), appliquée à 5 nouvelles colonnes plutôt qu'à `sigma_t`/`volume_norm`. Aucune nouvelle méthode statistique à inventer — uniquement une extension et une correction de robustesse numérique (cf. §1c).

**Affichage : une seule section avec sélecteur, pas 10 nouvelles cartes.** Pour éviter de refaire le fouillis déjà nettoyé, ces 5 indicateurs partagent une seule carte "étude d'événement" et une seule carte "Granger", avec des boutons pour choisir quel indicateur afficher — pas 10 cartes permanentes.

## Modifications à faire

### 1a. `regime_analytics.py` — rendre `_granger_causality_column_to_stress` public

Renommer `_granger_causality_column_to_stress` en `granger_causality_to_stress` (retirer le underscore) : cette fonction va maintenant être appelée directement depuis `dashboard_builder.py` (un autre module), donc la convention de nommage "underscore = privé à ce module" ne s'applique plus. `granger_causality_vol_to_stress` et `granger_causality_volume_to_stress` restent inchangées (elles appellent juste la fonction renommée en interne) — aucun changement de comportement, aucun risque de régression sur les tests existants qui les utilisent.

### 1b. `dashboard_builder.py` — calcul des 5 indicateurs via `pandas_ta`

Ajouter `import pandas_ta as ta` en haut du fichier (déjà une dépendance du projet, utilisée dans `regime_hmm.py`).

Définir une constante, à côté de `GRANGER_BONFERRONI_ALPHA` :

```python
# Indicateurs de force/momentum (couche de confirmation, hors HMM) — clé = nom de colonne dans
# history_ext, valeur = libellé humain pour l'UI.
FORCE_SIGNALS = {
    "macd_hist": "MACD (histogramme)",
    "momentum": "Momentum (non borné)",
    "roc": "Rate of Change (RoC)",
    "rsi": "RSI",
    "stoch_k": "Stochastique (%K)",
}
```

Dans `compute_all_analytics()`, pour chaque actif, après avoir calculé `history` (déjà présent), construire `history_ext` avec les 5 colonnes ajoutées à partir de `prices` :

```python
history_ext = history.copy()
macd = ta.macd(prices["Close"], fast=12, slow=26, signal=9)
history_ext["macd_hist"] = macd["MACDh_12_26_9"].reindex(history_ext.index)
history_ext["momentum"] = ta.mom(prices["Close"], length=10).reindex(history_ext.index)
history_ext["roc"] = ta.roc(prices["Close"], length=10).reindex(history_ext.index)
history_ext["rsi"] = ta.rsi(prices["Close"], length=14).reindex(history_ext.index)
stoch = ta.stoch(prices["High"], prices["Low"], prices["Close"], k=14, d=3)
history_ext["stoch_k"] = stoch["STOCHk_14_3_3"].reindex(history_ext.index)
```

**Vérification obligatoire avant d'aller plus loin** : les noms de colonnes exacts retournés par `pandas_ta` (`MACDh_12_26_9`, `STOCHk_14_3_3`) doivent être confirmés en imprimant `macd.columns.tolist()` et `stoch.columns.tolist()` dans l'environnement réel — même prudence que celle déjà appliquée à `ADX_14`/`DMP_14`/`DMN_14` par le passé (la version de pandas_ta disponible ici n'a pas pu être testée dans mon bac à sable, incompatibilité de version Python). Vérifier aussi que `ta.mom()` et `ta.roc()` retournent bien une `pd.Series` directement assignable (pas un DataFrame à colonne unique nécessitant `.iloc[:, 0]`).

Ensuite, pour chaque signal de `FORCE_SIGNALS`, calculer le profil d'événement et le test de Granger, avec des clés composites dans `per_asset[ticker]` (pas de dict imbriqué, pour pouvoir réutiliser `_granger_table_payload()` telle quelle, cf. §2) :

```python
for signal_key in FORCE_SIGNALS:
    per_asset[ticker][f"profile_{signal_key}"] = ra.regime_transition_vol_profile(
        history_ext, window=10, alignment="start", only_into="stress", column=signal_key
    )
    per_asset[ticker][f"granger_{signal_key}"] = ra.granger_causality_to_stress(
        history_ext, signal_key, maxlag=10
    )
```

### 1c. `dashboard_builder.py` — corriger `_event_study_from_profile()` pour les signaux signés (point de rigueur important)

`_event_study_from_profile()` calcule actuellement l'indexation en pourcentage de la baseline : `(mean_sigma / baseline - 1) * 100`. Ça fonctionne pour `sigma_t`/`volume_norm` (toujours positifs, baseline jamais nulle), mais **casserait silencieusement** pour MACD/Momentum (qui oscillent autour de 0 — une baseline proche de 0 ou négative rend ce ratio instable ou trompeur) et n'a pas vraiment de sens non plus pour RoC (déjà un pourcentage) ou RSI/Stochastique (bornés 0-100, pas naturellement des "ratios à leur propre moyenne").

Correction : ajouter un paramètre `method: str = "pct"` à `_event_study_from_profile()`. Si `method == "zscore"`, remplacer le calcul de l'indexation par un score z standardisé sur l'écart-type de la période de référence (au lieu du pourcentage) — robuste au signe et à une baseline proche de 0, et rend les 5 actifs comparables sur un même graphique malgré des échelles de prix très différentes (BTC vs obligations). **Le calcul du seuil de significativité (`first_reaction_day`) ne change pas** : il est déjà en unités absolues (`(mean_sigma - baseline).abs() > 1.959964 * baseline_std_of_stat / sqrt(n_events)`), donc déjà robuste au signe — seule la valeur AFFICHÉE sur le graphique change de formule.

```python
def _event_study_from_profile(profile: pd.DataFrame, label: str, color: str, method: str = "pct") -> dict:
    if profile["n_events"].iloc[0] == 0 or profile["mean_sigma"].isna().all():
        return {
            "label": label, "color": color,
            "rel_day": profile["rel_day"].tolist(), "index_pct": [None] * len(profile),
            "n_events": 0, "first_reaction_day": None,
        }

    baseline_mask = profile["rel_day"].between(-10, -5)
    baseline_mean = profile.loc[baseline_mask, "mean_sigma"].mean()
    n_events = int(profile["n_events"].iloc[0])
    baseline_std_of_stat = profile.loc[baseline_mask, "std_sigma"].mean()

    if method == "zscore":
        baseline_std = profile.loc[baseline_mask, "mean_sigma"].std()
        if not baseline_std or np.isnan(baseline_std) or baseline_std == 0:
            return {
                "label": label, "color": color,
                "rel_day": profile["rel_day"].tolist(), "index_pct": [None] * len(profile),
                "n_events": n_events, "first_reaction_day": None,
            }
        index_values = (profile["mean_sigma"] - baseline_mean) / baseline_std
    else:
        index_values = (profile["mean_sigma"] / baseline_mean - 1.0) * 100.0

    threshold = 1.959964 * baseline_std_of_stat / np.sqrt(n_events)
    deviation = (profile["mean_sigma"] - baseline_mean).abs()
    significant = deviation > threshold
    first_reaction_day = None
    for rel_day, is_sig in zip(profile["rel_day"], significant):
        if is_sig:
            first_reaction_day = int(rel_day)
            break

    return {
        "label": label, "color": color,
        "rel_day": profile["rel_day"].tolist(),
        "index_pct": [_num(v) if not np.isnan(v) else None for v in index_values],
        "n_events": n_events,
        "first_reaction_day": first_reaction_day,
    }
```

Les appels existants (`event_study`, `event_study_volume`, `event_study_end`, `event_study_end_volume`) ne changent pas — ils gardent `method="pct"` par défaut, donc comportement strictement identique à avant ce brief.

### 2. `dashboard_builder.py` — `_comparison_payload()` : payload des 5 signaux

```python
force_event_study = {
    signal_key: {
        asset["short"]: _event_study_from_profile(
            analytics["per_asset"][asset["ticker"]][f"profile_{signal_key}"],
            asset["label"], asset["color"], method="zscore",
        )
        for asset in ASSETS
    }
    for signal_key in FORCE_SIGNALS
}
force_granger = {
    signal_key: _granger_table_payload(analytics, f"granger_{signal_key}", GRANGER_BONFERRONI_ALPHA)
    for signal_key in FORCE_SIGNALS
}
```

Ajouter `"force_event_study": force_event_study`, `"force_granger": force_granger`, et `"force_signal_labels": FORCE_SIGNALS` au dict retourné par `_comparison_payload()`.

### 3. `dashboard_builder.py` (HTML) — une section, deux cartes, un sélecteur

Ajouter, après la section "Partie 2" (v14), un nouveau séparateur et 2 cartes (étude d'événement + Granger), partageant le même sélecteur de signal :

```html
<h2 style="font-size:1rem;color:#95a5a6;border-top:1px solid #1c2a3a;padding-top:14px;margin-top:18px">
  Extension Partie 1 &#8212; indicateurs de force/momentum (couche de confirmation, hors HMM)
</h2>

<div class="scale-sel" style="justify-content:flex-start;margin-bottom:10px">
  <button class="force-btn force-btn-active" data-signal="rsi">RSI</button>
  <button class="force-btn" data-signal="macd_hist">MACD</button>
  <button class="force-btn" data-signal="momentum">Momentum</button>
  <button class="force-btn" data-signal="roc">RoC</button>
  <button class="force-btn" data-signal="stoch_k">Stochastique</button>
</div>

<div class="card">
  <div class="card-label">Le signal s&#233;lectionn&#233; pr&#233;c&#232;de-t-il ou confirme-t-il un passage en stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
  <p class="chart-note">M&#234;me m&#233;thode que la Partie 1 (indexation standardis&#233;e en &#233;carts-types de la
    p&#233;riode de r&#233;f&#233;rence, robuste aux signaux qui oscillent autour de 0 comme le MACD). On teste
    maintenant 7 signaux au total avec le vol et le volume &#8212; chaque test individuel reste correctement
    corrig&#233;, mais comparer "lequel ressort le mieux" parmi 7 comporte un risque suppl&#233;mentaire de
    surinterpr&#233;tation, &#224; garder en t&#234;te.</p>
  <div id="chart-force-event-study" style="height:380px"></div>
  <details style="margin-top:8px">
    <summary>Voir le d&#233;tail par actif (jour de premi&#232;re r&#233;action, nombre d'&#233;v&#233;nements)</summary>
    <div style="overflow-x:auto;margin-top:6px">
      <table><thead><tr><th>Actif</th><th>Premier jour de r&#233;action</th><th>n transitions vers stress</th></tr></thead>
      <tbody id="force-event-study-table"></tbody></table>
    </div>
  </details>
</div>

<div class="card">
  <div class="card-label">Test formel &#8212; le signal s&#233;lectionn&#233; pr&#233;dit-il le r&#233;gime stress futur&nbsp;? (causalit&#233; de Granger)</div>
  <p class="chart-note">M&#234;me test que pour vol/volume (ADF + Granger + Bonferroni, &#945;=0.005).</p>
  <div id="chart-force-granger" style="height:260px"></div>
  <p class="chart-note" id="force-granger-verdict"></p>
  <details style="margin-top:8px">
    <summary>Voir le d&#233;tail par actif (stationnarit&#233; ADF, p-value minimale)</summary>
    <div style="overflow-x:auto;margin-top:6px">
      <table><thead><tr><th>Actif</th><th>ADF signal (p)</th><th>ADF p_stress (p)</th><th>p-value min (lag)</th><th>Verdict</th></tr></thead>
      <tbody id="force-granger-table"></tbody></table>
    </div>
  </details>
</div>
```

### 4. `dashboard_builder.py` (JS) — sélecteur + réutilisation de `renderEventStudy`/`renderGrangerCard`

`renderEventStudy()` (déjà étendue en v14 avec un `xAxisTitle`) doit accepter un 5e paramètre `yAxisTitle` (le texte "Écart vs période pré-événement (%)" est actuellement en dur, faux pour le mode z-score) :

```js
function renderEventStudy(data, chartId, tableId, xAxisTitle, yAxisTitle) {{
  // ... corps inchangé, sauf :
  yaxis: {{ title: yAxisTitle, gridcolor: GRID, zeroline: true, zerolinewidth: 2, zerolinecolor: '#566573' }},
  // ... reste inchangé
}}
```

Mettre à jour les 4 appels existants (Q1/Q2 début + Q1/Q2 fin, v11/v12/v14) pour passer explicitement `'&#201;cart vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)'` en 5e argument (aucun changement visuel).

Ajouter :

```js
const FORCE_SIGNAL_LABELS = COMPARISON.force_signal_labels;
function renderForceSignal(signal) {{
  renderEventStudy(COMPARISON.force_event_study[signal], 'chart-force-event-study', 'force-event-study-table',
    'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)',
    '&#201;cart standardis&#233; vs p&#233;riode pr&#233;-&#233;v&#233;nement (&#233;carts-types)');
  renderGrangerCard(COMPARISON.force_granger[signal], COMPARISON.granger_alpha,
    'chart-force-granger', 'force-granger-table', 'force-granger-verdict', FORCE_SIGNAL_LABELS[signal]);
}}
renderForceSignal('rsi');
document.querySelectorAll('.force-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.force-btn').forEach(b => b.classList.remove('force-btn-active'));
    btn.classList.add('force-btn-active');
    renderForceSignal(btn.dataset.signal);
  }});
}});
```

## Vérifications demandées après implémentation

1. Les noms de colonnes `pandas_ta` (`MACDh_12_26_9`, `STOCHk_14_3_3`) sont bien ceux retournés dans l'environnement réel — sinon adapter et signaler l'écart.
2. Les 5 boutons changent bien le contenu des 2 graphiques (étude d'événement + Granger) sans recharger la page.
3. Les graphiques Q1/Q2/Partie 2 (vol/volume) restent visuellement identiques à avant ce brief (le paramètre `method="pct"` par défaut ne doit rien changer pour eux).
4. Pour le MACD/Momentum (signaux signés), vérifier que le graphique affiche des valeurs cohérentes (pas de NaN générralisé, pas de valeurs aberrantes signe d'une division par une baseline proche de 0 mal gérée).
5. Les tests existants passent toujours ; ajouter au moins un test unitaire pour `_event_study_from_profile(..., method="zscore")` sur un signal synthétique signé (ex. valeurs positives et négatives autour de 0), vérifiant que la fonction ne lève pas d'erreur et retourne des `index_pct` non nuls.
