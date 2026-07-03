# BRIEF v10 — Ajout d'un panneau volume (volume_norm) sur chaque onglet actif

## Objectif

Ajouter un panneau de volume normalisé (`volume_norm` = volume / moyenne mobile 30j du volume) sur **chaque onglet actif individuel** (BTC-USD, ETH-USD, SPY, ZN=F, TLT), positionné en panneau séparé, juste en dessous du panneau de volatilité existant.

**Hors périmètre** : la page "Comparaison" n'est pas concernée par ce brief. Ne pas toucher aux probabilités de régime (calm/bull/bear/stress) ni à leur logique de calcul.

## Pourquoi

Le volume est un indicateur de force du mouvement de volatilité. Avant de pouvoir tester une corrélation/déphasage entre pics de volume, pics de volatilité et débuts de régime, il faut d'abord visualiser le volume normalisé sur chaque actif, dans un panneau dédié (pas superposé à la volatilité, pour éviter les biais de lecture d'un double axe).

## Modifications à faire

### 1. `regime_agent.py` — exposer `volume_norm` dans l'historique

Dans `_predict_history()`, `features["volume_norm"]` est déjà calculé en interne mais n'est pas recopié dans le DataFrame retourné. Juste après la ligne :

```python
df["vol_of_vol"] = features["sigma_t"].rolling(20).std().values
```

ajouter :

```python
df["volume_norm"] = features["volume_norm"].values
```

### 2. `dashboard_builder.py` — `_asset_tab_payload()` : envoyer `volume_norm` au frontend

À côté des lignes existantes qui construisent `"sigma"` et `"vol_of_vol"`, ajouter :

```python
"volume_norm": [_num(v) if not np.isnan(v) else None for v in history["volume_norm"]],
```

### 3. `dashboard_builder.py` — `_asset_panel_html()` : nouvelle carte HTML

Insérer une nouvelle carte entre la carte "volatilité" (`chart-vol-{dom}`) et la carte BOCPD (`chart-cp-{dom}`), sur le même modèle que les cartes existantes (même classe `card`, même largeur, titre clair). Hauteur suggérée : 120px (cohérent avec `chart-cp` qui fait 120px, un peu moins que `chart-vol` à 140px car c'est un signal secondaire).

Titre de la carte : "Volume normalisé (ratio à la moyenne 30j)".

```html
<div class="card">
  <h3>Volume normalisé (ratio à la moyenne 30j)</h3>
  <div id="chart-volume-{dom}" style="height:120px"></div>
</div>
```

### 4. `dashboard_builder.py` — JS `initAssetTab(tabId)` : tracer le nouveau graphe

Ajouter un trace pour `chart-volume-{SHORT_OF[tabId]}`, dans le même bloc que les traces sigma/vol_of_vol existantes. Type de trace : bar (barres), plus lisible qu'une aire pour du volume. Ajouter une ligne de référence horizontale à `y=1` (car `volume_norm` est un ratio à sa propre moyenne 30j — au-dessus de 1 = volume supérieur à la normale, en dessous = inférieur).

```js
const volumeTrace = {{
  x: d.dates, y: d.volume_norm, type: 'bar', name: 'Volume (norm.)',
  marker: {{ color: '#7f8c8d' }}
}};
Plotly.newPlot(`chart-volume-${{SHORT_OF[tabId]}}`, [volumeTrace], {{
  margin: {{ t: 10, b: 20, l: 40, r: 10 }},
  yaxis: {{ title: 'x moyenne 30j' }},
  xaxis: {{ range: currentXRange, matches: undefined }},
  shapes: [{{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 1, y1: 1, yref: 'y',
             line: {{ color: '#2c3e50', width: 1, dash: 'dot' }} }}],
  showlegend: false
}}, {{ responsive: true }});
```

Respecter le style déjà utilisé pour `chart-vol`/`chart-cp` (marges, police, couleurs de fond) pour rester cohérent visuellement — s'inspirer du code existant plutôt que de réinventer un style.

### 5. `dashboard_builder.py` — synchronisation du zoom (2 points d'intégration, critiques)

Le dashboard utilise un mécanisme "single-source-of-truth" pour le zoom : seul `chart-price` écoute les événements de zoom/drag utilisateur (`plotly_relayout`), tous les autres panneaux (`chart-vol`, `chart-cp`) ne font que suivre via `Plotly.relayout()`. Ce point est **critique** : c'est ce qui évite une boucle infinie déjà corrigée par le passé (voir `CORRECTIF_dashboard_v4_boucle_infinie.md`). Le nouveau panneau `chart-volume` doit suivre exactement le même principe : uniquement suiveur, jamais émetteur.

**5a.** Dans `applyZoom(tabId, scale, anchorDateStr)`, la ligne :

```js
['price','vol','cp'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{...}}))
```

devient :

```js
['price','vol','cp','volume'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{...}}))
```

**5b.** Dans le listener `plotly_relayout` attaché à `chart-price` (bloc qui relaye déjà vers `chart-vol` et `chart-cp`), ajouter un appel équivalent pour `chart-volume` :

```js
Plotly.relayout(`chart-volume-${{SHORT_OF[tabId]}}`, {{'xaxis.range[0]': r0, 'xaxis.range[1]': r1}});
```

à ajouter juste à côté des deux appels existants pour `chart-vol` et `chart-cp` dans ce même bloc.

**5c.** Dans `refreshTab(tabId)`, la ligne qui applique les shapes (fonds colorés de régime) :

```js
['price','vol','cp'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{shapes}}));
```

devient également :

```js
['price','vol','cp','volume'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{shapes}}));
```

(pour que le panneau volume affiche aussi les fonds colorés de régime, comme les autres panneaux).

## Vérifications demandées après implémentation

1. Sur chaque onglet actif (BTC-USD, ETH-USD, SPY, ZN=F, TLT), le panneau volume s'affiche bien, avec des données réelles (pas de graphe vide).
2. Le zoom (boutons Jour/Mois/Trimestre/Année + calendrier) et le drag à la souris sur `chart-price` déplacent bien les 4 panneaux (price, vol, volume, cp) de façon synchronisée.
3. Les fonds colorés de régime apparaissent aussi sur le panneau volume.
4. La page "Comparaison" est inchangée.
5. Confirmer qu'aucune régression n'est introduite sur le mécanisme de zoom (pas de gel de page, pas de boucle infinie) — tester en zoomant/dézoomant plusieurs fois de suite sur chaque onglet.
