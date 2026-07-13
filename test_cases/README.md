# `test_cases/` — Scénarios de test sur transitions de régime

Système **séparé** du pipeline de production (`model_artifacts/` → `Run/`) : au lieu de
prévoir hors-échantillon sur les données les plus récentes, ce module rejoue les 5
modèles (ARIMA-GARCH, SARIMA, Prophet, LSTM, TSDiff) sur des **moments historiques
précis et fixes** — des transitions de régime de marché (bull→stress, bear→calme,
bear→stress) — pour pouvoir comparer leurs performances sur des scénarios de référence,
année après année, sans que le référentiel ne bouge.

## Vue d'ensemble du pipeline

```
DONNEE~1.XLS (racine du repo)
        │  python -m test_cases.convert_source_data
        ▼
test_cases/data/<ticker>.csv        (Close + Regime quotidien par actif)
        │  python -m test_cases.run_test_cases
        ▼
test_cases/results/<tc_id>/<ticker>/<transition_date>/<model>.json
        │  python -m test_cases.generate_dashboard
        ▼
test_cases/dashboard.html            (à ouvrir dans un navigateur)
```

## Les trois tableaux (registres)

Toute la configuration du système passe par ces trois fichiers. Modifier le
comportement (ajouter un modèle/actif/test case) ne demande jamais de toucher au reste
du code (`transitions.py`, `run_test_cases.py`, `generate_dashboard.py`).

### 1. `registry_models.py` — les modèles

```python
MODELS = [
    {"id": "ARIMA-GARCH", "label": "ARIMA-GARCH", "folder": "ARIMA",
     "forecaster": "forecast_horizons_arima", "isolated_subprocess": False},
    ...
]
```

- `forecaster` : nom d'une fonction `forecast_horizons_<x>(train: pd.Series, horizons:
  list[int]) -> {h_days: (point, lo, hi)}` de `benchmarks/multi_horizon.py`. Fit une
  seule fois par appel, contrat déjà utilisé par `model_artifacts/pipeline.py`.
- `folder` : préfixe du dossier dans `Run/` (`Run/<date>-<folder>-<asset>-<horizon>/`),
  utilisé uniquement pour retrouver les derniers `hyperparams.json` en date (traçabilité,
  affiché sur le dashboard).
- `isolated_subprocess` : `True` uniquement pour LSTM (deadlock TensorFlow documenté si
  `statsmodels`/`arima_model` sont importés dans le même process — voir
  `model_artifacts/pipeline.py` et `models/conftest.py`).

**Pour ajouter un modèle** :
1. Écrire `forecast_horizons_<nom>(train, horizons)` dans `benchmarks/multi_horizon.py`
   si elle n'existe pas déjà (même contrat que les autres).
2. Ajouter une entrée à `MODELS` dans `registry_models.py`.
3. Si le modèle a des dépendances lourdes incompatibles avec TensorFlow dans le même
   process (rare), l'isoler comme LSTM (`isolated_subprocess: True` + un petit script
   `<nom>_subprocess_forecast.py` sur le modèle de `lstm_subprocess_forecast.py`).

### 2. `registry_assets.py` — les actifs

Réutilise `calibration/regime/assets.py::ASSETS` (source unique de vérité déjà utilisée
par tout le reste du repo — pas de duplication) et associe à chaque ticker son CSV
d'historique sous `data/`.

**Pour ajouter un actif** :
1. L'ajouter à `calibration/regime/assets.py::ASSETS` (ticker, label, couleur, classe
   d'actif) — `registry_assets.py` le reprend automatiquement.
2. Fournir son historique quotidien (mêmes colonnes que les CSV existants : `Date,
   Close, Sigma_t_pct, Vol_of_Vol, Volume_norm, Changepoint_prob, Regime`) sous
   `test_cases/data/<ticker>.csv` — soit en l'ajoutant comme feuille dans `DONNEE~1.XLS`
   (nom de feuille = le `label` exact) puis en relançant `convert_source_data.py`, soit
   en écrivant directement le CSV si la source n'est pas dans ce fichier Excel.

### 3. `registry_test_cases.py` — les test cases

```python
TEST_CASES = [
    {"id": "TC1.2", "name": "bull_stress", "label": "Bull → Stress",
     "regime_from": "bull", "regime_to": "stress",
     "extra_filter": "tc12_stress_filter", "horizons_days": [1, 7]},
    ...
]
MAX_OCCURRENCES_PER_ASSET = 3
```

Un test case est une **transition de régime** : `regime_from` (jour D-1) →
`regime_to` (jour D), détectée directement sur la colonne `Regime` des CSV (déjà
calculée par le moteur `calibration/regime`, aucun re-calcul de HMM nécessaire ici).

- `extra_filter` (optionnel) : nom d'une fonction de `transitions.py` qui affine la
  détection brute (voir TC1.2 ci-dessous). `None` = toute transition de régime brute
  compte.
- `horizons_days` : les horizons (en jours de bourse) prévus à chaque occurrence.
- `MAX_OCCURRENCES_PER_ASSET` : nombre d'occurrences historiques les plus récentes
  retenues par (actif × test case) — borne le coût de calcul (LSTM notamment).
  Configurable aussi via `--max-occurrences` en CLI.

**Pour ajouter un test case** : ajouter une entrée à `TEST_CASES`. Rien d'autre à
modifier — `transitions.py` et `run_test_cases.py` sont génériques sur `regime_from`/
`regime_to`.

#### La règle TC1.2 (bull→stress confirmé)

Le filtre `tc12_stress_filter` (dans `transitions.py`) ne garde, parmi les transitions
brutes bull→stress, que celles où le choc est déjà statistiquement significatif au jour
même de la transition :

```
PI_low(D) = Close(D-1) * (1 - 1.96 * Sigma_t_pct(D-1) / 100)
condition retenue : Close(D) < PI_low(D)
```

où `D-1` est le dernier jour encore "bull" (le cutoff) et `D` le premier jour "stress".
La bande de référence est construite avec les colonnes déjà présentes dans le fichier de
données (volatilité GARCH `Sigma_t_pct`), **pas** avec un des 5 modèles évalués — sinon
dépendance circulaire avant même d'avoir lancé les modèles. Cette formule est isolée
dans une seule fonction : si l'interprétation doit être corrigée, la modifier là ne
change rien d'autre à l'architecture. TC1.3 (bear→calme) et TC1.4 (bear→stress)
n'ont pas de filtre additionnel (transition de régime brute).

## Exécution

```bash
# 1. Convertir DONNEE~1.XLS en CSV par actif (une fois, ou si la source change)
python -m test_cases.convert_source_data

# 2. Lancer les modèles sur toutes les occurrences de tous les test cases
python -m test_cases.run_test_cases

# Sous-ensembles utiles pour itérer rapidement (pas de LSTM/Prophet, coûteux) :
python -m test_cases.run_test_cases --test-cases TC1.3 --assets BTC-USD --models ARIMA-GARCH,SARIMA
python -m test_cases.run_test_cases --max-occurrences 1 --epochs 10   # LSTM plus rapide

# 3. Générer la page HTML
python -m test_cases.generate_dashboard
# -> ouvrir test_cases/dashboard.html dans un navigateur
```

## Ce que fait `run_test_cases.py`

Pour chaque (test case × actif × occurrence retenue × modèle) :
- entraîne le modèle sur tout l'historique connu jusqu'au **cutoff** (dernier jour de
  l'ancien régime — jamais de fuite du futur) ;
- prévoit les horizons demandés (`forecast_horizons_<model>`, fit une seule fois) ;
- compare à la valeur réelle déjà connue (toutes les transitions étudiées sont passées,
  lue directement dans le CSV — pas de retéléchargement) : `in_interval`,
  `direction_correct`, `beats_naif` (mêmes définitions que
  `validation/verdict_rules.py` / `BRIEF_tracking_db.md`) ;
- retrouve le dernier `Run/<date>-<folder>-<asset>-D1/hyperparams.json` en date pour ce
  (modèle, actif) et l'attache au résultat avec sa date, à titre purement informatif —
  le fit lui-même est toujours refait à la date historique du cutoff (les
  hyperparamètres sont de toute façon des constantes de modèle, cf.
  `model_artifacts/pipeline.py` §12) ;
- écrit `results/<tc_id>/<ticker>/<transition_date>/<model_id>.json`.

## Dashboard (`generate_dashboard.py`)

Même charte visuelle que `Run/dashboard.html` (`model_artifacts/generate_dashboard.py`) :
- un onglet par **actif** ;
- un sous-onglet par **test case** ;
- un sélecteur d'**occurrence** (date de transition) si plusieurs sont disponibles ;
- pour l'occurrence choisie : un graphe (prix réel, fond coloré par régime, prévisions
  D+1/D+7 par modèle avec intervalle 95%) et un tableau détaillé par modèle.

Si un (actif × test case) n'a aucune occurrence historique (ex. TLT pour TC1.2, 0
transition bull→stress détectée), le dashboard l'indique explicitement plutôt que de
planter.
