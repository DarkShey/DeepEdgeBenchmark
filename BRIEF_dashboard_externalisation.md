# BRIEF — Externalisation du dashboard (2 étapes : artefact CI, puis lazy-load par date)

## 0. Contexte

`Run/dashboard.html` est un **fichier HTML auto-contenu** : la coquille (HTML/CSS/JS,
quelques dizaines de Ko, taille fixe) **plus toutes les données embarquées inline**.
Mécanique actuelle (`model_artifacts/generate_dashboard.py`) :

- `render_html()` (ligne ~232) construit le HTML ; `data_json = json.dumps(payload…)`
  (ligne ~253) sérialise **tout** le payload, puis le template remplace le marqueur
  `const DATA = __DATA_JSON__;` (ligne ~436) par ce blob.
- `main()` (ligne ~1646) écrit le tout : `out_path.write_text(html)` → `Run/dashboard.html`.
- Publication : `.github/workflows/deploy-pages.yml` **copie** le fichier committé vers
  `_site/index.html` et le déploie sur GitHub Pages.

**Problème.** Chaque run réécrit le blob `DATA` en entier avec ~60 combos de plus →
**+~1 Mo par run**. Historique : 2,4 Mo (2026-07-10) → **9,1 Mo (2026-07-16)**, croissance
non bornée. Conséquences : commits git lourds, chargement de page lent (le navigateur
télécharge tout avant d'afficher), GitHub ne prévisualise pas les fichiers > 1 Mo.

**Décision (opérateur).** Deux étapes séquentielles, **livrables indépendamment** :

- **Étape 1 — Option 3** : sortir le dashboard du repo. Il devient un **artefact de build**
  régénéré et déployé par la CI Pages. Le HTML n'est plus versionné → le repo cesse de
  grossir du fait du dashboard.
- **Étape 2 — Option 2** : **découper les données par date** + **lazy-load**. La coquille
  ne charge que l'index au démarrage puis récupère les données d'une date à la demande →
  affichage instantané même avec des centaines de dates.

> Faire l'étape 1 d'abord, la valider (URL Pages à jour, dashboard hors du repo), **puis**
> l'étape 2. L'étape 2 suppose l'étape 1 en place (déploiement CI de tout `_site/`).

---

## ÉTAPE 1 — Le dashboard devient un artefact de build (CI)

### 1.1 Objectif

`Run/dashboard.html` **n'est plus committé**. À chaque push d'un run (données `Run/`), le
workflow Pages **régénère** le dashboard depuis les artefacts committés et le déploie.
**L'URL Pages reste identique** : `https://darkshey.github.io/DeepEdgeBenchmark/`.

### 1.2 Garde-fous

- Branche dédiée : `maeva/dashboard-ci` (sans accent).
- **Ne pas modifier la logique** de `generate_dashboard.py` (rendu, métriques, contenu) —
  cette étape ne touche que le *packaging* (git + workflow).
- URL Pages **inchangée**, comportement visuel **inchangé**.
- Le dashboard reste **régénérable en local** pour prévisualiser
  (`python -m model_artifacts.generate_dashboard`), mais **ne jamais le committer**.

### 1.3 Modifications

**a) Retirer le dashboard du suivi git**
- Ajouter à `.gitignore` :
  ```
  Run/dashboard.html
  ```
- Le désindexer sans le supprimer localement :
  ```
  git rm --cached Run/dashboard.html
  ```
- Vérifier : `git ls-files Run/dashboard.html` ne renvoie plus rien.

**b) Adapter `.github/workflows/deploy-pages.yml`**

- **Déclencheur** : aujourd'hui `paths: ['Run/dashboard.html', …]`. Comme le fichier
  n'est plus committé, déclencher sur les **données** :
  ```yaml
  on:
    push:
      branches: [main]
      paths:
        - 'Run/**'                       # un nouveau run met à jour les données
        - 'validation/tracking.db'       # sim_trades lit la DB
        - 'model_artifacts/generate_dashboard.py'
        - '.github/workflows/deploy-pages.yml'
    workflow_dispatch:
  ```
  > Retirer `Run/dashboard.html` de la liste `paths`.

- **Étapes du job** : remplacer le `cp Run/dashboard.html _site/index.html` par une
  génération. Dépendances CI **minimales** (le dashboard n'importe **aucun** modèle lourd —
  ni tensorflow/torch/prophet/… ; l'import `calibration.regime.assets` est best-effort avec
  fallback, donc inutile en CI) :
  ```yaml
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install minimal deps
      run: pip install pandas pyarrow numpy
    - name: Generate dashboard
      run: |
        mkdir -p _site
        python -m model_artifacts.generate_dashboard --out _site/index.html
    - uses: actions/configure-pages@v5
    - uses: actions/upload-pages-artifact@v3
      with:
        path: _site
    - id: deployment
      uses: actions/deploy-pages@v4
  ```
  > `generate_dashboard.py` accepte déjà `--out` (défaut `<run-root>/dashboard.html`) — on
  > l'écrit directement dans `_site/index.html`, aucun `cp` intermédiaire.
  > `--run-root` défaut = `Run` (déjà correct après checkout).

### 1.4 Points d'attention

- **Sources en CI** : `generate_dashboard` lit `Run/*/*.parquet` + `Run/*/*.json` +
  `validation/tracking.db` — **tous committés**, donc présents après `actions/checkout`. Rien
  à télécharger.
- Le workflow **ne committe rien** : il génère dans `_site/` (éphémère) et déploie. Aucune
  écriture dans le repo.
- **Coût CI** : install `pandas/pyarrow/numpy` ~1 min, génération quelques secondes.
- Pour figer les versions et accélérer, on peut plus tard mettre en cache pip / épingler les
  versions ; hors périmètre de cette étape.

### 1.5 Critères d'acceptation

1. `git ls-files | grep dashboard.html` → **vide** (le fichier n'est plus versionné).
2. Un push touchant `Run/` déclenche le workflow → **run vert**, URL Pages **à jour** (même
   URL, `generated_at` récent, tous les combos présents).
3. Un `git log --stat` d'un nouveau run **ne contient plus** `Run/dashboard.html`.
4. `du -sh` du repo ne croît plus du fait du dashboard.

---

## ÉTAPE 2 — Découpage des données par date + lazy-load (Option 2)

### 2.1 Objectif

Ne plus embarquer **toutes** les séries dans la page. La coquille charge un **index léger**
au démarrage (rendu instantané), puis récupère par `fetch()` les données d'**une seule date**
quand l'utilisateur la sélectionne. Le poids initial devient **constant** (coquille + index),
indépendamment du nombre de dates.

### 2.2 Principe

`generate_dashboard.py` produit désormais, dans `_site/` :

```
_site/
  index.html            # coquille légère (HTML/CSS/JS + index inline OU petit index.json)
  data/
    20260707.json       # séries + records de cette date uniquement
    20260708.json
    …
    20260716.json
  data/index.json       # liste des dates + résumé léger nécessaire à la vue par défaut
```

- **Vue par défaut** : bâtie à partir de `data/index.json` (dates disponibles + agrégats
  légers : ex. dernière date, KPIs de tête). Aucun téléchargement des grosses séries.
- **Interaction** : au changement de date/combo, `fetch('data/<date>.json')`, mise en cache
  côté client (`Map`), puis rendu. Une date déjà chargée n'est pas re-téléchargée.

### 2.3 Modifications dans `generate_dashboard.py`

- **Séparer le `payload`** en deux niveaux :
  - *léger* (pour le premier rendu) : liste des dates, catalogue actifs/modèles, agrégats de
    tête → sérialisé dans `data/index.json` (ou inline dans la coquille, si < ~100 Ko).
  - *lourd, par date* : toutes les séries/records → un fichier `data/<date>.json` chacun.
- Remplacer l'injection unique `const DATA = __DATA_JSON__;` par :
  - écrire chaque `data/<date>.json` (une passe sur `payload` groupé par date) ;
  - écrire `data/index.json` ;
  - la coquille ne contient plus que l'**index** + le **JS de fetch** (pas les séries).
- Conserver un mode **mono-fichier local** optionnel (`--inline` / `--single-file`) pour
  prévisualiser hors-ligne sans serveur — utile en dev (un `file://` ne peut pas `fetch()`
  d'autres fichiers ; le mode inline reste pratique en local).

### 2.4 Côté coquille (JS)

- **Au chargement** : lire `data/index.json` (ou l'index inline), rendre la vue globale.
- **Au changement de date/combo** :
  ```js
  async function loadDate(date) {
    if (cache.has(date)) return cache.get(date);
    const res = await fetch(`data/${date}.json`);
    const d = await res.json();
    cache.set(date, d);
    return d;
  }
  ```
  Gérer les états **loading** et **erreur** (afficher un indicateur, message si 404/réseau).
- Les filtres (actif/modèle/horizon) opèrent sur la/les date(s) déjà chargées.

### 2.5 Intégration CI (dépend de l'étape 1)

- Le workflow déploie **tout `_site/`** (déjà le cas avec `upload-pages-artifact` sur `_site`)
  → `index.html` **et** `data/*.json` publiés ensemble, servis en **same-origin** (donc
  `fetch()` OK, pas de souci CORS).
- Rien de plus à committer : `data/*.json` sont des artefacts générés en CI, jamais versionnés
  (cohérent avec l'étape 1).

### 2.6 Critères d'acceptation

1. Poids du **premier chargement** (coquille + `data/index.json`) **borné** (cible : < ~500 Ko),
   **indépendant** du nombre de dates.
2. Affichage initial **< 1-2 s** même avec 100+ dates.
3. Sélection d'une date → **un seul** `fetch('data/<date>.json')` (vérifiable dans l'onglet
   réseau) ; re-sélection → **aucun** re-téléchargement (cache).
4. URL Pages **inchangée**, contenu visuel **identique** à l'actuel.
5. Mode `--inline` local produit toujours un mono-fichier ouvrable en `file://`.

---

## Ordre de livraison

1. **Étape 1** (rapide, faible risque) : gitignore + `git rm --cached` + workflow régénérant.
   → Valider les critères §1.5 avant de continuer.
2. **Étape 2** (plus de JS/refacto de `generate_dashboard`) : découpage par date + lazy-load.
   → Valider §2.6.

## Hors périmètre

- Refonte visuelle / nouvelles métriques du dashboard.
- Pagination/pruning de l'historique (archivage des vieilles dates) — pourra venir après si
  le volume total des `data/*.json` en CI devient lui-même un sujet.
- Migration du stockage des runs (`Run/*/parquet`) — inchangé.
