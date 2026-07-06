# Brief d'implémentation — Finalisation & tests des modèles IA (Maéva)

**Projet :** DeepEdgeBenchmark — couche `models/`
**Scope :** Finaliser les 4 forecasters (`arima`, `sarima`, `prophet`, `lstm`), vérifier leurs
dépendances, les rendre opérationnels, écrire les tests (smoke + unit) et générer les artefacts.
**Destinataire :** Claude Code (exécution dans le repo, en local, avec `.venv` activé).
**Ne PAS coder à l'aveugle :** ce brief décrit *quoi* faire et *comment valider*. Chaque étape a
des critères d'acceptation explicites. Tu exécutes, tu tests, tu me rends compte — dans cet ordre.

---

## 0. État constaté (audit statique préalable — déjà fait)

Vérifications faites en amont sur `models/` (compilation + inspection des imports) :

| Point | Constat |
|-------|---------|
| Compilation | Les 4 fichiers passent `python -m py_compile` sans erreur de syntaxe. |
| Interface | Interface **uniforme** sur les 4 : `fetch_data()`, `compute_metrics()`, `run_<model>()`, `next_step_<model>()`, `save_plot()`, `main()`. C'est un atout : les tests peuvent être factorisés. |
| CLI | Contrat commun `--ticker / --start / --end / --test-ratio / --next-step / --plot`. Spécifiques : `arima` a `--order`, `lstm` a `--epochs`. |
| `requirements.txt` | Couvre bien tous les imports (numpy, pandas, yfinance, statsmodels, arch, scikit-learn, matplotlib, prophet, tensorflow). Cohérent. |
| Orchestrateur | `benchmarks/run_benchmark.py` importe les 4 modules via `sys.path.insert(.../models)` — OK, chemin correct. |
| **Défaut à corriger** | Le tableau du `README.md` (racine) pointe vers `benchmarks/arima_model.py`, `benchmarks/sarima_model.py`, etc. **Les fichiers sont en réalité dans `models/`.** → liens cassés à réparer (voir Étape 4). |

**Sur la lib « à part » :** confirmé — les 3 modèles statistiques (ARIMA/GARCH via `arch`,
SARIMA via `statsmodels`, Prophet) forment un groupe ; le **LSTM est le seul sur TensorFlow/Keras**.
On **garde TensorFlow** (pas de migration PyTorch). Le seul point de vigilance TF est le
non-déterminisme (voir Étape 3, seed).

---

## 1. Objectif

Rendre les 4 modèles **reproductibles, testés et documentés par des artefacts**, sans toucher à
la logique de modélisation (ordres ARIMA, archi LSTM, saisonnalités Prophet restent tels quels).
On ne « réécrit » pas les modèles : on les **finalise** (robustesse d'exécution + tests + sorties).

---

## 2. Étape 1 — Vérification des imports & de l'environnement

**But :** garantir que `pip install -r requirements.txt` suffit à faire tourner les 4 modèles.

1. Créer/activer un venv propre :
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Vérifier que chaque module s'importe **sans réseau** (l'import ne doit pas télécharger de données) :
   ```bash
   cd models
   python -c "import arima_model, sarima_model, prophet_model, lstm_model; print('imports OK')"
   ```
3. Logger les versions effectives dans un fichier d'environnement (voir Étape 5, `REPORT.md`) :
   ```bash
   python -c "import numpy,pandas,statsmodels,sklearn,tensorflow,prophet,arch,yfinance; \
   print(numpy.__version__,pandas.__version__,statsmodels.__version__,sklearn.__version__, \
   tensorflow.__version__,prophet.__version__,arch.__version__,yfinance.__version__)"
   ```

**Critères d'acceptation :**
- [ ] Import des 4 modules OK, aucun `ModuleNotFoundError`.
- [ ] Aucun appel réseau déclenché au simple import.
- [ ] Versions installées ≥ celles épinglées dans `requirements.txt`.

---

## 3. Étape 2 — Smoke test (le modèle « tourne »)

**But :** confirmer qu'un run bout-en-bout produit des métriques finies, vite, sur peu de données.

Deux niveaux :

**2a. Smoke via CLI (mini-fenêtre, réseau requis).** Un run court par modèle :
```bash
cd models
python arima_model.py   --ticker SPY --start 2024-01-01 --end 2024-03-01 --test-ratio 0.2
python sarima_model.py  --ticker SPY --start 2024-01-01 --end 2024-03-01 --test-ratio 0.2
python prophet_model.py --ticker SPY --start 2024-01-01 --end 2024-03-01 --test-ratio 0.2
python lstm_model.py    --ticker SPY --start 2023-06-01 --end 2024-03-01 --test-ratio 0.2 --epochs 3
```
> LSTM : fenêtre plus longue (look-back = 30) + `--epochs 3` pour aller vite.

**2b. Smoke sans réseau (préféré pour la CI/reproductibilité).** yfinance est une source de
flakiness (réseau, rate-limit, tickers qui changent). Injecter une série **synthétique** et appeler
directement `run_<model>()` sans passer par `fetch_data()`. Exemple de fixture à réutiliser dans les
tests (Étape 3) :
```python
import numpy as np, pandas as pd
def synthetic_series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n)
    # marche aléatoire + légère tendance, prix strictement positifs
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.Series(prices, index=idx, name="Close")
```

**Critères d'acceptation (par modèle) :**
- [ ] Le run se termine sans exception.
- [ ] Le dict de résultat contient des `predicted` **finis** (pas de NaN/inf).
- [ ] `compute_metrics` renvoie RMSE, MAE, Dir. Acc, couverture PI 95 % — toutes finies.
- [ ] Runtime smoke < ~30 s (hors LSTM), LSTM < ~2 min avec `epochs=3`.

---

## 4. Étape 3 — Unit tests (pytest)

**But :** verrouiller le comportement des briques, **sans réseau**, de façon déterministe.

**Fichiers à créer :**
```
DeepEdgeBenchmark/
└── models/
    ├── conftest.py                 ← fixture synthetic_series + seeds
    ├── test_models_common.py       ← tests factorisés sur les 4 modèles
    └── test_metrics.py             ← tests unitaires de compute_metrics
```

**Contenu attendu :**

`test_metrics.py` — `compute_metrics` est identique/quasi-identique dans les 4 modules, on la teste à fond :
- prédiction parfaite (`predicted == actual`) → RMSE = 0, MAE = 0, Dir. Acc = 100 %.
- Directional Accuracy sur un cas connu construit à la main (signes de variations imposés).
- couverture PI : bornes `[lower, upper]` englobant tout l'actual → couverture = 100 %.
- robustesse aux entrées de tailles incohérentes → erreur claire (pas un crash opaque).

`test_models_common.py` — paramétré sur les 4 modules (`@pytest.mark.parametrize`) via l'interface commune :
- `fetch_data` est **monkeypatché** pour renvoyer `synthetic_series()` (zéro réseau).
- `run_<model>(train, test)` renvoie un dict avec les clés attendues (`predicted`, `pi_lower`, `pi_upper`, `metrics`…) — **normaliser le contrat de sortie** si divergence entre modèles.
- toutes les valeurs `predicted` sont finies et de longueur == `len(test)`.
- `next_step_<model>(series)` renvoie un point + un intervalle finis.
- **contrainte point-in-time** : vérifier que `run_*` n'utilise jamais un indice ≥ à la date prédite (pas de fuite du futur). Test léger : tronquer la série, re-prédire, comparer.

> Pour LSTM : fixer les seeds (`np.random.seed`, `tf.random.set_seed`, `PYTHONHASHSEED`) dans
> `conftest.py` et tolérer une marge (`pytest.approx`) plutôt qu'une égalité stricte — TF n'est pas
> bit-exact d'une machine à l'autre. `epochs` réduit (2–3) pour la vitesse.

**Commande :**
```bash
cd models && pytest -q
```

**Critères d'acceptation :**
- [ ] `pytest` vert, 0 échec, 0 test réseau (tourne hors-ligne).
- [ ] Couverture des 4 modèles + de `compute_metrics`.
- [ ] Durée totale de la suite < ~3 min.

---

## 5. Étape 4 — Correctifs de finalisation (légers)

Avant de générer les artefacts, appliquer ces corrections **ciblées** :

1. **README** : corriger le tableau — remplacer les liens `benchmarks/xxx_model.py` par
   `models/xxx_model.py`, et mettre à jour les blocs `cd benchmarks` → `cd models` là où ils
   pointent vers les modèles individuels (l'orchestrateur `run_benchmark.py` reste, lui, dans `benchmarks/`).
2. **Reproductibilité** : centraliser un seed global optionnel (`--seed`, défaut fixe) au moins pour
   le LSTM.
3. **Garde-fous d'exécution** : message d'erreur explicite si `fetch_data` renvoie une série vide
   (`No data returned for <ticker>`), et si `len(series)` < look-back pour le LSTM.

> Ne **pas** modifier les hyperparamètres de modélisation (ordres, unités, saisonnalités). C'est
> hors scope de cette finalisation.

---

## 6. Étape 5 — Artefacts à générer

**Recommandation quant senior — ce qui compte vraiment sur un benchmark de forecasting :**

Le vrai livrable d'un benchmark n'est pas le modèle sérialisé, c'est la **table de métriques
hors-échantillon reproductible**. Ordre de priorité :

**(A) Table de métriques comparative — LIVRABLE PRINCIPAL.**
`artifacts/metrics.csv` **et** `artifacts/metrics.json`, une ligne par (modèle × ticker) :
`model, ticker, start, end, test_ratio, n_test, RMSE, MAE, MAPE, DirAcc_%, PI_cov95_%, runtime_s, seed`.
C'est là-dessus qu'on juge et compare les 4 modèles. À produire via `run_benchmark.py` étendu
(il calcule déjà les métriques ; il faut juste les **exporter** en plus de les afficher).

**(B) Plots forecast — sanity visuel.**
- 1 PNG par modèle (`artifacts/forecast_<model>_<ticker>.png`) : actual vs predicted + bande PI 95 %.
- Le comparatif `benchmark_visual.png` (déjà produit par `run_benchmark.py`) → le déplacer/copier dans `artifacts/` pour cohérence.

**(C) Rapport de run daté — reproductibilité.**
`artifacts/REPORT.md` : date, tickers/fenêtres testés, seed, versions des libs (Étape 1),
résultats des tests (pass/fail + durée), et la table de métriques inline. C'est ce qui rend le
benchmark rejouable et auditable.

**(D) Sérialisation des modèles — OPTIONNEL, faible priorité.**
À expliciter parce que c'est contre-intuitif : ARIMA et SARIMA se **re-fittent à chaque pas** du
walk-forward, il n'existe donc pas « un » modèle figé à sauver — la sérialisation n'a pas de sens
ici. Prophet et LSTM sont fittés une fois et *pourraient* être sauvés (`.pkl` / `.keras`), mais
**seulement si l'objectif est de servir des prédictions en prod**, pas pour le benchmark. Tant qu'on
reste sur de l'évaluation comparative : **on ne sérialise pas.** À rediscuter si on passe en mode
« prédiction live ».

**Arborescence artefacts :**
```
DeepEdgeBenchmark/
└── artifacts/
    ├── metrics.csv
    ├── metrics.json
    ├── forecast_arima_SPY.png
    ├── forecast_sarima_SPY.png
    ├── forecast_prophet_SPY.png
    ├── forecast_lstm_SPY.png
    ├── benchmark_visual.png
    └── REPORT.md
```

**Commande de génération :**
```bash
cd benchmarks
python run_benchmark.py --ticker SPY --start 2023-01-01 --end 2024-12-31
# → doit écrire les métriques + plots dans ../artifacts/
```

**Critères d'acceptation :**
- [ ] `metrics.csv` + `metrics.json` produits, 4 lignes (1/modèle), valeurs finies.
- [ ] 4 PNG forecast + le comparatif présents dans `artifacts/`.
- [ ] `REPORT.md` daté, avec versions libs + résultats tests + table de métriques.
- [ ] Le tout rejouable : relancer la commande redonne des métriques identiques (à la tolérance LSTM près, seed fixé).

---

## 7. Ordre d'exécution pour Claude Code

1. Étape 1 — venv + imports (bloquant).
2. Étape 3 — écrire `conftest.py` + tests, faire passer `pytest` (hors-ligne).
3. Étape 2 — smoke CLI sur SPY (valide la chaîne réseau→métriques).
4. Étape 4 — correctifs README + seed + garde-fous.
5. Étape 5 — étendre `run_benchmark.py` pour exporter les artefacts, puis générer.
6. Rendre compte : coller la sortie de `pytest` + la table `metrics.csv` + confirmer les artefacts.

**Règle :** si une étape échoue, **s'arrêter et remonter l'erreur** (ne pas contourner en modifiant
la logique de modélisation). On ajuste ensemble.

---

## 8. Résumé des critères de « terminé »

- [ ] Les 4 modèles importent et tournent (smoke OK).
- [ ] `pytest` vert, hors-ligne, couvre les 4 modèles + `compute_metrics`.
- [ ] README corrigé (liens `models/`).
- [ ] Artefacts générés : `metrics.{csv,json}`, plots, `REPORT.md`.
- [ ] Run reproductible (seed fixé, versions loggées).
