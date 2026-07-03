# Correctif v4 — Boucle infinie de synchronisation (dashboard gelait au clic)

**Date :** 02/07/2026
**Signalement :** "la page est trop complexe, Firefox ne la supporte pas. Je ne peux pas cliquer
sur les box ni entrer de date dans la partie des dates."
**Fichier modifié :** `dashboard_builder.py` uniquement (le mécanisme cassé est 100% côté JS
généré, aucun changement sur le moteur de régime ni sur les fichiers protégés).

---

## 1. Ce que le signalement laissait penser, et ce que c'était réellement

Le symptôme ("Firefox ne supporte pas la page") suggérait un problème de compatibilité
navigateur ou de performance. **Ce n'était pas le cas.** C'était un vrai bug logique : au premier
clic sur un bouton de zoom (Jour/Mois/Trimestre/Année) ou sur le sélecteur de date, l'onglet se
figeait complètement — plus aucune interaction possible, y compris sur des éléments sans rapport
(checkboxes de régime).

---

## 2. Méthode de diagnostic — ne pas deviner, reproduire

Plutôt que de supposer une cause, le bug a été **reproduit et confirmé mécaniquement** :

1. Installation de Firefox headless (`playwright install firefox`) pour rejouer le clic
   automatiquement et capturer les erreurs de la console JS.
2. Validation préalable que le JavaScript généré n'avait **aucune erreur de syntaxe**
   (parseur `esprima` sur le bloc `<script>` extrait — verdict : `PARSE OK`).
3. Simulation d'un clic réel sur le bouton "Mois" via Playwright : le clic n'a **jamais abouti**,
   avec un timeout de 30 secondes dépassé alors que Playwright confirmait pourtant l'élément
   "visible, enabled and stable".
4. Deuxième tentative en déclenchant le clic directement en JS (`element.click()` via
   `page.evaluate`) pour éliminer toute question de positionnement/visibilité de l'élément :
   **le process a dû être tué de force** (code de sortie 144 = SIGTERM) après blocage total,
   confirmant que la page elle-même se figeait — pas un souci de détection de clic par l'outil de
   test.

Ces deux tests convergent vers une seule explication possible : un handler d'évènement JS entre
dans une **boucle qui ne se termine jamais**, bloquant le thread principal du navigateur.

---

## 3. Root cause exacte

### Le mécanisme cassé (hérité de v1/v2, jamais revu depuis)

Les 3 graphiques temporels d'un onglet (prix / volatilité / BOCPD) étaient synchronisés au zoom
en s'écoutant **mutuellement** :

```js
// AVANT (buggé)
['price','vol','cp'].forEach(id => {
  document.getElementById(id).on('plotly_relayout', e => {
    syncRange(id, e['xaxis.range[0]'], e['xaxis.range[1]']);   // relaie aux 2 AUTRES graphiques
  });
});
```

Avec un garde-fou anti-réentrance (`_syncing`) remis à `false` **immédiatement après** l'appel à
`Plotly.relayout()` — donc de façon **synchrone**.

### Pourquoi ça bouclait

Le patch v3 a introduit `applyZoom()`, qui déclenche un `Plotly.relayout()` **programmatique**
sur les 3 graphiques à la fois (pas seulement sur celui qu'on vient de faire glisser à la souris).
Or `Plotly.relayout()` émet l'évènement `plotly_relayout` **de façon asynchrone** (après le
re-rendu, pas dans la même pile d'appels). Séquence observée :

1. Clic sur "Mois" → `applyZoom` relaie la nouvelle plage aux 3 graphiques.
2. Le graphique prix émet (de façon différée) son propre `plotly_relayout` → le listener relaie
   aux graphiques vol/BOCPD.
3. Mais entretemps, `_syncing` avait déjà été remis à `false` (remise à `false` synchrone, alors
   que l'évènement du graphique voisin arrive plus tard, de façon asynchrone) : le garde-fou ne
   protège donc plus rien.
4. Le graphique vol relaie à son tour vers prix + BOCPD, qui relaient de nouveau vers vol + BOCPD…
   **un cycle sans fin entre 3 nœuds qui s'écoutent tous mutuellement.**

Tant que le zoom ne venait que d'une seule interaction utilisateur isolée sur un seul graphique
(v1/v2), le risque de collision de timing restait faible en pratique. Le patch v3
(`applyZoom` déclenchant 3 relayouts d'un coup) l'a rendu systématique.

---

## 4. Correctif appliqué

### 4.1 Architecture "source unique" (élimine le cycle structurellement)

Seul le graphique **prix** écoute désormais `plotly_relayout`. Volatilité et BOCPD ne sont plus
que des suiveurs, sans aucun listener :

```js
document.getElementById(`chart-price-${tabId}`).on('plotly_relayout', e => {
  if (TABS[tabId]._programmatic) return;   // évite un aller-retour redondant (inoffensif, pas une boucle)
  if (e['xaxis.range[0]'] !== undefined) {
    const r0 = String(e['xaxis.range[0]']).slice(0,10);
    const r1 = String(e['xaxis.range[1]']).slice(0,10);
    Plotly.relayout(`chart-vol-${tabId}`, {'xaxis.range[0]':r0,'xaxis.range[1]':r1});
    Plotly.relayout(`chart-cp-${tabId}`,  {'xaxis.range[0]':r0,'xaxis.range[1]':r1});
    TABS[tabId].currentXRange = [r0, r1];
    onRangeChange(tabId);
  }
});
```

Avec seulement des arêtes `prix → vol` et `prix → BOCPD` (jamais l'inverse), un cycle est
**impossible par construction**, quel que soit le timing synchrone/asynchrone de Plotly — le
raisonnement ne dépend plus d'aucune hypothèse fragile sur l'ordre d'exécution.

### 4.2 Flag `_programmatic` (propreté, pas anti-boucle)

`applyZoom()` (boutons + sélecteur de date) pose `TABS[tabId]._programmatic = true` pendant ses
propres appels à `Plotly.relayout()`, pour éviter que son propre relayout sur le graphique prix
ne redéclenche une passe redondante (mais bornée — au pire un aller de plus, jamais un cycle).

### 4.3 Bug secondaire trouvé en testant : normalisation des dates

En testant le glisser-sélection natif à la souris (zoom Plotly standard), les bornes retournées
par l'évènement incluaient une heure précise (`2020-07-31 00:22:46.6081`) au lieu d'une date
simple (`2020-07-31`), ce qui décalait légèrement le comptage du donut de composition aux
frontières. Corrigé par troncature à la date (`.slice(0,10)`) dès la réception de l'évènement.

---

## 5. Vérification — reproduite automatiquement, pas seulement "relue"

Toutes les vérifications ci-dessous ont été exécutées dans un **vrai Firefox headless**
(Playwright), pas seulement en relisant le code :

| Test | Résultat |
|---|---|
| Clic sur les 4 boutons de zoom (Jour/Mois/Trimestre/Année), sur les 4 actifs (16 clics) | Page réactive après chaque clic, 0 erreur |
| Sélecteur de date (`#datepick-BTC-USD`) rempli avec une date (mars 2020, crash COVID) | Zoom correct (~2 mois autour de la date), donut recalculé (61 j), page réactive |
| Glisser-déposer natif à la souris directement sur le graphique prix | Vol + BOCPD synchronisés, donut recalculé, page réactive, dates normalisées |
| Checkbox de régime (Calme/Tendanciel/Stress) | Toggle fonctionne, page réactive |
| Onglet Comparaison (box plot + corrélation glissante) | Rendu correct, page réactive |
| Erreurs JS capturées (`pageerror` + console) sur l'ensemble des tests ci-dessus | **0 erreur** |

`pytest calibration/regime/ -v` → **15/15 verts**, aucune régression (ce correctif ne touche
que le JavaScript généré par `dashboard_builder.py`, aucun impact sur `regime_state.py`,
`regime_hmm.py`, `regime_bocpd.py` ni `test_regime_agent.py`).

---

## 6. Fichiers touchés

- `calibration/regime/dashboard_builder.py` — seul fichier modifié (mécanisme de synchronisation
  du zoom, ~30 lignes JS changées).
- `calibration/regime/output/regime_dashboard.html` — régénéré avec le correctif.
