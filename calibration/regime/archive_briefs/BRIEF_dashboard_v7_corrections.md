# Corrections v7 — Dashboard Multi-Actifs DEITA (retour du 02/07/2026, suite)

**Retour :** la carte "Volatilité autour des changements de régime (étude d'événement, ±10j)"
ajoutée en v6 est illisible — une grille de 5 mini-graphiques avec 2 courbes superposées, une
bande d'écart-type, un axe "jour relatif à l'événement" (concept abstrait de recherche
quantitative), sur des valeurs qui varient peu (~2.7 à ~3.7) donc visuellement plates. Personne
d'extérieur au sujet ne peut lire ça d'un coup d'œil.

**Diagnostic** : le problème n'est pas la méthode (l'étude d'événement reste la bonne approche
statistique, gardée telle quelle en interne), c'est la **restitution visuelle**, beaucoup trop
technique pour l'audience. Solution : garder le calcul, changer radicalement l'affichage.

Ce patch ne touche que `dashboard_builder.py` (uniquement la construction du payload de cette
carte + son rendu JS) — pas `regime_analytics.py` (la fonction `regime_transition_vol_profile`
reste inchangée, on ne fait qu'en dériver un résumé plus simple pour l'affichage), pas les
fichiers du moteur de régime.

---

## Nouvelle version : comparaison "avant / après" en barres, pas en courbes

**Principe** : remplacer les 21 points par transition (jour -10 à +10) par **2 chiffres** :
la volatilité moyenne des 10 jours **avant** la transition, et des 10 jours **après** (jour de
transition inclus). Un simple diagramme en barres à 2 barres par actif répond à la question posée
("la vol précède-t-elle le changement, ou vient-elle après ?") de façon immédiatement lisible —
pas besoin de comprendre ce qu'est un "jour relatif à l'événement" pour lire 2 barres côte à côte.

**Un seul graphique** (pas 5 mini-graphiques séparés) : axe X = les 5 actifs, 2 barres groupées par
actif ("Avant" en gris neutre, "Après" en orange), avec la valeur écrite directement au-dessus de
chaque barre et le delta en % affiché entre les deux. En dessous du graphique, une phrase de
conclusion en français simple, générée automatiquement à partir des chiffres.

Un bouton bascule optionnel permet de recalculer les mêmes 2 barres en isolant uniquement les
transitions **vers le régime stress** (le cas le plus parlant), sans ajouter de 2ᵉ série de
courbes à lire en parallèle — même graphique, mêmes 2 barres, juste d'autres chiffres derrière.

### `dashboard_builder.py` — payload

Dans `compute_all_analytics`, garder l'appel existant à `regime_transition_vol_profile` (aucun
changement côté calcul), mais dériver un résumé simple à partir du DataFrame retourné :

```python
def _vol_before_after(profile_df: pd.DataFrame) -> dict:
    """
    Réduit un profil d'étude d'événement (21 points, rel_day -10..10) à 2 chiffres :
    moyenne de mean_sigma sur rel_day < 0 ("avant") et sur rel_day >= 0 ("après", jour de
    transition inclus). Retourne aussi le delta en % et le nombre d'événements sous-jacents.
    """
    before = profile_df.loc[profile_df["rel_day"] < 0, "mean_sigma"]
    after = profile_df.loc[profile_df["rel_day"] >= 0, "mean_sigma"]
    avant = float(before.mean()) if before.notna().any() else None
    apres = float(after.mean()) if after.notna().any() else None
    delta_pct = ((apres - avant) / avant * 100) if (avant and apres and avant != 0) else None
    return {"avant": avant, "apres": apres, "delta_pct": delta_pct}
```

Dans `_comparison_payload`, remplacer le payload `vol_profiles` actuel (qui expose les séries
complètes rel_day/mean_sigma/std_sigma pour 5 mini-graphiques) par un payload résumé :

```python
vol_before_after = {}
for asset in ASSETS:
    ticker = asset["ticker"]
    profile_all = analytics["per_asset"][ticker]["profile_all"]
    profile_stress = analytics["per_asset"][ticker]["profile_into_stress"]
    vol_before_after[asset["short"]] = {
        "label": asset["label"],
        "all": _vol_before_after(profile_all),
        "into_stress": _vol_before_after(profile_stress),
        "n_all": int(profile_all["n_events"].iloc[0]) if len(profile_all) else 0,
        "n_stress": int(profile_stress["n_events"].iloc[0]) if len(profile_stress) else 0,
    }
```

(Adapter aux noms réels des variables déjà en place dans `compute_all_analytics`/`_comparison_payload`
pour `profile_all`/`profile_into_stress` — la logique de calcul de l'étude d'événement elle-même ne
change pas, seul ce résumé est nouveau.)

### HTML — remplacer la carte

```html
<div class="card">
  <div class="card-label">La volatilité augmente-t-elle avant ou après un changement de régime ?</div>
  <p class="chart-note">Comparaison de la volatilité moyenne dans les 10 jours qui précèdent un
    changement de régime, contre les 10 jours qui suivent (jour du changement inclus). Si "Après"
    est nettement plus haut que "Avant", la volatilité ne prévient pas le changement, elle
    l'accompagne ou le suit. Si "Avant" est plus haut, la volatilité est un signal avancé.</p>
  <div class="scale-sel">
    <button class="va-btn va-btn-active" data-mode="all">Tous les changements de r&#233;gime</button>
    <button class="va-btn" data-mode="into_stress">Uniquement vers le stress</button>
  </div>
  <div id="chart-vol-before-after" style="height:320px"></div>
  <p class="chart-note" id="vol-before-after-conclusion"></p>
</div>
```

Retirer entièrement l'ancien `<div class="grid2x2" id="vol-profile-grid">` et son bloc JS associé
(`profileBandTrace`, la boucle `ASSETS.forEach` qui créait 5 mini-graphiques).

### JS — nouveau rendu

```js
function renderVolBeforeAfter(mode) {
  const shorts = ASSETS.map(a => a.short);
  const data = shorts.map(s => COMPARISON.vol_before_after[s][mode]);

  const avantTrace = {
    type: 'bar', name: 'Avant (10j)', x: shorts, y: data.map(d => d.avant),
    marker: {color: '#7f8c8d'},
    text: data.map(d => d.avant !== null ? d.avant.toFixed(2) : '—'), textposition: 'outside',
  };
  const apresTrace = {
    type: 'bar', name: 'Après (10j)', x: shorts, y: data.map(d => d.apres),
    marker: {color: '#e67e22'},
    text: data.map(d => d.apres !== null ? d.apres.toFixed(2) : '—'), textposition: 'outside',
  };

  Plotly.react('chart-vol-before-after', [avantTrace, apresTrace], Object.assign({}, baseLayout(), {
    barmode: 'group', margin: {l: 50, r: 10, t: 10, b: 40},
    yaxis: {title: 'Volatilité moyenne σⱼ (%)', gridcolor: GRID},
  }), {responsive: true, displayModeBar: false});

  // Phrase de conclusion générée automatiquement
  const rises = shorts.filter((s, i) => data[i].delta_pct !== null && data[i].delta_pct > 10);
  const falls = shorts.filter((s, i) => data[i].delta_pct !== null && data[i].delta_pct < -10);
  let msg;
  if (rises.length >= shorts.length - 1) {
    msg = `Sur presque tous les actifs (${rises.join(', ')}), la volatilité est plus élevée APRÈS le changement de régime qu'avant — la vol suit le changement, elle ne l'annonce pas clairement.`;
  } else if (falls.length >= shorts.length - 1) {
    msg = `Sur presque tous les actifs, la volatilité était déjà plus élevée AVANT le changement de régime — la vol précède bien le changement, comme un signal avancé.`;
  } else {
    msg = `Résultat mixte selon l'actif : ${rises.length} actif(s) montrent une vol plus forte après (${rises.join(', ') || 'aucun'}), ${falls.length} avant (${falls.join(', ') || 'aucun'}).`;
  }
  document.getElementById('vol-before-after-conclusion').textContent = msg;
}

document.querySelectorAll('.va-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.va-btn').forEach(b => b.classList.remove('va-btn-active'));
    btn.classList.add('va-btn-active');
    renderVolBeforeAfter(btn.dataset.mode);
  });
});
renderVolBeforeAfter('all');   // appelé dans initComparisonTab(), à la place de l'ancienne boucle vol-profile-grid
```

CSS à ajouter (mêmes codes couleurs que `.scale-btn` pour cohérence visuelle, renommés pour ce
composant) :
```css
.va-btn{ background:#0f0f1a; color:#95a5a6; border:1px solid #1c2a3a; border-radius:4px;
  padding:3px 10px; font-size:.72rem; cursor:pointer; margin-right:6px; }
.va-btn-active{ background:#2980b9; color:#fff; }
```

---

## Vérification finale

- `pytest calibration/regime/ -v` → toujours vert (aucun test ne porte sur le rendu HTML de cette
  carte, seulement sur `regime_transition_vol_profile` qui n'est pas modifiée).
- Régénérer `output/regime_dashboard.html`, ouvrir l'onglet Comparaison : un seul graphique à
  barres (pas 5 petits graphiques), 2 barres par actif clairement étiquetées "Avant"/"Après" avec
  les valeurs affichées dessus, un bouton pour basculer sur "Uniquement vers le stress", et une
  phrase de conclusion en français simple sous le graphique qui change selon le mode sélectionné.
- Relire la phrase de conclusion générée : elle doit correspondre à ce qui est visuellement montré
  par les barres (vérifier qu'elle n'est pas générique/à côté de la plaque).
