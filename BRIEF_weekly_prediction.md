# BRIEF — Prédiction hebdomadaire (W+1 / W+2 / W+3) par modèles entraînés en weekly

## 0. Contexte

Aujourd'hui, l'horizon long (« D+7 ») est produit par des modèles **entraînés en
journalier** puis extrapolés sur plusieurs pas :

- **LSTM** — rollout **récursif** : `benchmarks/multi_horizon.py:forecast_from_fitted_lstm`
  ré-injecte ses propres prédictions (`buffer.append(p_scaled)  # recursif : jamais le vrai
  futur`) sur 5-7 pas. Les erreurs se **composent** (biais + variance qui s'accumulent).
- **ARIMA / SARIMA / Prophet** — multi-step natif, mais qui extrapole une dynamique
  **quotidienne** vers un horizon d'une semaine.
- Le backtest D+7 (`pipeline.py:_run_model_d7_rolling`) fait des **origines glissantes au pas
  quotidien**, chaque origine ré-appelant `forecast_horizons_<model>` (refit).

**Problème *by-design*.** Un modèle appris sur des rendements *quotidiens* n'a jamais vu la
loi à 1 semaine ; il la déduit en enchaînant des pas courts. Un modèle entraîné sur des
rendements **hebdomadaires** cible directement la distribution à 1 semaine : **un seul pas**,
pas de compounding, et la structure vol/autocorrélation hebdo est **apprise** et non extrapolée.

**Ancrages « daily-centric » du code** (surface à généraliser) :

| Couche | Point | Fichier |
|---|---|---|
| Données | fetch en barres journalières, aucun resample | `benchmarks/run_benchmark.py:download_full_data` |
| Horizons | codés en **jours de bourse** : `{"D1":1,"D7":5}`, business `(1,7)` | `pipeline.py:113,120` |
| Backtest | origines glissantes au **pas quotidien** | `pipeline.py:_run_model_d7_rolling` |
| Multi-step | rollout récursif (LSTM) | `multi_horizon.py:forecast_from_fitted_lstm` |
| tracking.db | colonne `horizon` = jours de bourse | `tracking_db.py` |
| Fenêtre | `WINDOW_YEARS = 3` → **~156 barres hebdo** | `pipeline.py:124` |

## 1. Objectif

Ouvrir une **voie hebdomadaire** : des modèles entraînés sur des séries **resamplées en
weekly**, produisant nativement **W+1, W+2, W+3**. But scientifique : vérifier que le
**weekly-natif** bat le **daily-multistep** aux horizons pluri-hebdomadaires (RMSE / CRPS /
couverture plus stables, sans erreur composée).

> **Livré en 2 temps** : d'abord un **prototype tête-à-tête** (Phase 0, ce brief + code joint)
> pour **valider ou infirmer l'hypothèse avec des chiffres** ; ensuite seulement, si le signal
> est là, la **généralisation** dans le pipeline (Phases 1-3).

## 2. Principe de conception : la fréquence devient une dimension

Aujourd'hui l'axe d'horizons est `{"D1":1,"D7":5}` (jours). On généralise en
**(fréquence, pas)** :

```python
HORIZONS = {"D1":("D",1), "D7":("D",5), "W1":("W",1), "W2":("W",2), "W3":("W",3)}
```

Un modèle weekly opère sur la série resamplée ; son « 1 pas » = **1 semaine**. La fréquence
est threadée à travers : données (resample) → fit → forecast (pas en fréquence native) →
backtest (walk-forward au pas d'1 semaine) → tracking.db (colonne `frequency`) → dossiers
`Run/` → dashboard.

### 2.1 Convention de resampling (à figer)

```python
weekly = daily.resample("W-FRI").last().dropna()   # clôture du vendredi
```

- **W-FRI** (clôture vendredi) comme ancre canonique.
- **Point-in-time** : une semaine n'est comptée que **complète**. Un run en milieu de semaine
  ne doit PAS créer une barre hebdo partielle du vendredi à venir (sinon fuite/instabilité).
- Jours fériés : `.last()` sur le resample gère l'absence du vendredi (prend le dernier jour
  coté de la semaine).

### 2.2 Longueur d'historique

3 ans = **~156 barres hebdo** : insuffisant pour SARIMA saisonnier (période 52) et marginal
pour LSTM/diffusion. → pour la voie weekly, viser **8-10 ans** (~400-520 barres). yfinance
fournit 10 ans de daily sans difficulté ; `WINDOW_YEARS` doit être **fréquence-dépendant**.

## 3. Phase 0 — Prototype tête-à-tête (TSDiff-W vs TSDiff daily-multistep)

**Objectif** : mesurer, sur les mêmes actifs / dates-cibles, l'écart entre :
- **TSDiff-W** : entraîné sur rendements **hebdo**, `horizon=3` → génère W+1/W+2/W+3 **en un
  seul tir** (le UNet1D produit tout le chemin d'un coup, **aucune récursion**) ;
- **TSDiff-D** : le port daily existant (`models/tsdiff_model.py`), `horizon` étendu à ~15
  jours de bourse, lu aux offsets correspondant aux vendredis cibles (multi-step daily natif).

**Pourquoi TSDiff d'abord** : son architecture génère déjà un **chemin d'horizon complet**
(pas de rollout récursif), c'est donc le cas où la comparaison isole proprement l'effet
« fréquence d'entraînement » (weekly vs daily) sans mélanger l'effet « récursif vs one-shot ».

**Code joint** (prototype, hors pipeline de prod) :
- `models/tsdiff_weekly.py` — `to_weekly()`, `forecast_weekly()` (fit hebdo + W+1/2/3),
  réutilise la classe `TSDiff` de `models/tsdiff_model.py` avec `horizon=3`.
- `experiments/weekly_headtohead_tsdiff.py` — walk-forward hebdo sur N origines : à chaque
  vendredi-origine, on entraîne les deux modèles sur les données ≤ origine et on prévoit les
  **mêmes dates-cibles** (vendredis t+1/t+2/t+3) ; on agrège **RMSE, couverture PI, CRPS** par
  horizon et par actif.

**Critère de décision** : si TSDiff-W ≤ TSDiff-D en RMSE/CRPS à W+2/W+3 (attendu) avec une
couverture PI au moins aussi bonne → on généralise (Phases 1-3). Sinon → on documente que
l'intuition ne se matérialise pas sur ces séries et on s'arrête.

## 4. Phases de généralisation (si Phase 0 concluante)

### Phase 1 — Couche données + axe d'horizons
- `to_weekly()` réutilisable ; `WINDOW_YEARS` fréquence-dépendant.
- Généraliser le mapping d'horizons en `(fréquence, pas)`.

### Phase 2 — Variantes de modèles weekly (patron TSDiff déjà en place)
- Enregistrer `TSDiff-W` (puis `LSTM-W`, `Prophet-W`, `Naive-W`, `ARIMA-W`) dans `MODELS`,
  `MODEL_FOLDER_NAME`, le dispatch (`_run_model_*`), `MODEL_ADAPTERS`, le dashboard.
- Chaque variante : resample train→weekly, fit, forecast W+1/2/3.
  - **Direct multi-horizon** privilégié (une sortie par horizon → zéro compounding) ; TSDiff-W
    le fait nativement (chemin généré d'un coup).

### Phase 3 — Backtest hebdo + persistance
- `_run_model_weekly()` : walk-forward **au pas d'1 semaine** (généralise
  `_run_model_d7_rolling`).
- `tracking.db` : colonne `frequency` (`D`/`W`) + `step` ; horizons `W1/W2/W3`. Permet la
  comparaison **côte à côte** daily-multistep vs weekly-natif dans le dashboard.
- Dossiers `Run/<date>-<modèle>-<actif>-W1/W2/W3`, facet weekly au dashboard.

## 5. Garde-fous

- Branche dédiée : `maeva/weekly-prediction` (sans accent) pour la généralisation ;
  le **prototype** (Phase 0) peut vivre sur une branche courte type `exp/weekly-tsdiff`.
- **Ne pas modifier** le pipeline de prod ni les modèles daily existants en Phase 0 — le
  prototype est **additif** (`models/tsdiff_weekly.py` réutilise `tsdiff_model.py` en lecture).
- **Point-in-time strict** : jamais de barre hebdo partielle ; le backtest ne voit jamais
  au-delà de l'origine.
- Reproductibilité : seeds fixées (`tsdiff_model.set_seed`).

## 6. Critères d'acceptation (Phase 0)

1. `experiments/weekly_headtohead_tsdiff.py` tourne de bout en bout sur ≥ 2 actifs et produit
   un tableau **RMSE + couverture + CRPS** par horizon (W+1/2/3) pour **TSDiff-W** et
   **TSDiff-D**.
2. Les deux modèles prévoient **exactement les mêmes dates-cibles** (vendredis t+k) à chaque
   origine (comparaison équitable).
3. Résultats sauvegardés (JSON/CSV) + lisibles, permettant de trancher le critère §3.

## 7. Hors périmètre (Phase 0)

- Intégration au pipeline de prod, tracking.db, dashboard (→ Phases 1-3).
- Variantes weekly des autres modèles (LSTM-W, etc.).
- Optimisation d'hyperparamètres weekly (lookback, horizon, epochs) — valeurs raisonnables
  par défaut, à affiner en Phase 2.
