# BRIEF — Brancher la calibration σ dynamique sur la prédiction **live** (prod) + vérification comportementale

> Fait suite à `HANDOFF_sigma_calibration_suivi.md` (commit `4888afe`). Les adoptions
> (skew-t ARIMA-GARCH, EWMA Naive/LSTM, correction multiplicative √EWMA(z²) SARIMA/Prophet,
> Prophet log-espace, CRPS two-piece) sont **en place dans `models/*.py`** — donc actives
> sur le **backtest/Gate2 D1** et le **dashboard**. **Elles n'atteignent PAS la prévision
> live** écrite dans `tracking.db`, qui passe par `benchmarks/multi_horizon.py` (non migré).
> Ce brief branche les adoptions sur la prod, **de façon causale et réversible** (mêmes
> garde-fous que `4888afe`), et exige une **vérification de comportement** — pas un simple
> re-run du pipeline du jour.

## 0. État des lieux (vérifié dans le code, ne pas re-supposer)

- **Adoptions présentes** (`models/*.py`, commit `4888afe`) : `arima_model.GARCH_DIST="skewt"`,
  `naive_model.SIGMA_MODE="ewma"`, `lstm_model.SIGMA_MODE="ewma"`,
  `sarima_model.CALIBRATE_SIGMA="ewma"`, `prophet_model.LOG_SPACE=True` +
  `CALIBRATE_SIGMA="ewma"`. Toutes causales, toutes avec switch de retour arrière.
- **Chemin OÙ elles agissent déjà** : `pipeline._run_model_d1` → `run_*` (métriques Gate2 D1)
  et le reporting (`prob_kpi_common.sample_parametric` two-piece,
  `honest_eval.metrics.crps_parametric`, libellés `generate_dashboard`). ✅
- **Le trou** : la prévision *live* (horizons business 1 et 7 jours de bourse, écrite dans
  `tracking.db`) passe par `pipeline._forecast_horizon` / `_forecast_all_horizons` →
  `benchmarks/multi_horizon.py` (importé `as mh`), **non migré** :
  - `fit_arima` : GARCH `dist="normal"` (≈ l.47) ; `forecast_from_fitted_arima` : bande
    **symétrique** `Z_95` (≈ l.68-69) → pas de skew-t.
  - `fit_prophet` / `forecast_from_fitted_prophet` : fit en **espace prix**, bornes natives
    `yhat_lower/upper` (≈ l.143) → ni log, ni EWMA.
  - `forecast_from_fitted_lstm` : `point ± 1.96·σ` figé (≈ l.222) ;
    `forecast_horizons_naive` : `± 1.96·σ·√h` figé (≈ l.244) ; SARIMA : `conf_int` brut
    (≈ l.98) → pas d'EWMA.
  - **Conséquence : aujourd'hui, aucune prédiction trackée (D+1 ni D+7) n'a les adoptions.**
- **Sur le §8.1 du handoff** : il parle de « brancher `next_step_prophet(sigma_scale=…)` ».
  Le *mécanisme* (`sigma_scale`) est le bon, mais `next_step_prophet` **n'est jamais appelé
  par la prod** (seulement `archives/` + CLI). Le vrai point d'entrée à migrer est **`mh`**.

## 1. Objectif

Faire que les prédictions **live** de `tracking.db` utilisent **les mêmes lois et le même σ
dynamique** que le backtest adopté — causal (aucune fuite d'information future) et réversible
(switch de retour arrière restaurant le comportement historique bit-for-bit).

## 2. Les trois chantiers

### Chantier 1 — Brancher les adoptions sur le chemin live (`mh` + `sigma_scale` depuis `tracking.db`) — **cœur du travail**

**1a. Aligner `benchmarks/multi_horizon.py` sur `models/*.py`** (réutiliser, ne pas
réimplémenter — importer les constantes/fonctions des modèles) :
- ARIMA-GARCH : `fit_arima` → `dist=arima_model.GARCH_DIST` ; `forecast_from_fitted_arima` →
  bornes via les **quantiles de la loi ajustée** (mirror `arima_model._std_quantiles` /
  `_dist_shape`), plus `Z_95`.
- Prophet : `fit_prophet` / `forecast_from_fitted_prophet` → fit sur **`log(price)`**, bornes
  exponentiées (même spécification que `prophet_model.run_prophet`, loi lognormale en prix).
- LSTM / Naive / SARIMA : accepter un **facteur de correction** `sigma_scale` et l'appliquer
  **multiplicativement autour du point** (mirror `run_*` :
  `lo = point − (point − lo)·corr`, `hi = point + (hi − point)·corr`).

**1b. Calculer `sigma_scale = √EWMA(z²)` depuis `tracking.db`** (nouvelle fonction utilitaire,
p.ex. `validation/sigma_scale.py`, réutilisée par `pipeline._forecast_horizon`) :
- `z = (y_true − y_pred) / σ_own`, `σ_own = (y_upper − y_lower)/(2·1.959963985)`, sur les
  prédictions **réalisées** récentes (`source IN ('oos','live')`, `daily_duplicate=0`,
  `y_true NOT NULL`) — cf. `prob_kpi_common.load_matrix_rows` pour le contrat exact.
- Par **(modèle × actif × horizon)**, λ=0.94, **causal** : n'utiliser que des origines
  `cutoff_date < date de prévision`. État EWMA initialisé à 1 (« faire confiance au σ du
  modèle jusqu'à preuve du contraire »), comme dans `run_sarima`/`run_prophet`.

**1c. Réversibilité** : un flag global (p.ex. `pipeline.USE_DYNAMIC_SIGMA` + CLI
`--calibrate-sigma off`) qui restaure exactement le comportement historique de `mh`.

**Périmètre D+1 vs D+7 — décision explicite (défaut recommandé) :**
- Les adoptions **distributionnelles** (skew-t, log-espace) sont **indépendantes de
  l'horizon** → appliquées à **tous les horizons** (D+1 et D+7).
- La **correction EWMA multiplicative** n'a été validée qu'en **1-pas (D+1)** (cf. HANDOFF
  §3-4). Défaut : **activée en D+1**, **désactivée en D+7** (flag OFF + `TODO`), pour ne pas
  extrapoler un réglage non testé sur la prod. → activer D+7 seulement après un backtest
  7-pas dédié.
- **MISE À JOUR 2026-07-31** : le backtest 7-pas dédié a été fait
  (`experiments/d7_sigma_scale_validation.py`, W1 × 5 actifs, correction causale avec lag
  de résolution 7 j). Verdict initial : D+7 activé pour SARIMA (MACE 6,3→4,7), Prophet
  (34,1→9,4) et Naive (7,3→4,5) ; LSTM d'abord exclu (dégradation sévère sur SPY 5,0→9,5).
- **MISE À JOUR 2026-07-31 (suite) — garde-fou λ et activation LSTM** : l'analyse de
  garde-fou (`experiments/d7_guard_analysis.py`) a identifié la cause de l'échec SPY :
  les origines quotidiennes d'horizon 7 j se **chevauchent de 6 j**, leurs z² sont
  autocorrélés, et λ=0,94 par origine sur-réagit. Le correctif structurel — **λ D+7 =
  0,94^(1/7)** (0,94 par observation indépendante) — répare SPY (9,6→1,4) et améliore
  aussi SARIMA (4,7→2,9) et Naive (4,5→2,8). **LSTM rejoint donc le D+7** (MACE moyenne
  5,4). Exception documentée : **Prophet garde λ=0,94 en D+7** (σ brut massivement
  décalibré sur ZN/TLT — la vitesse d'adaptation prime, 9,4 vs 12,7 en λ lent).
  Câblé dans `pipeline.py` (`SIGMA_SCALE_HORIZONS`, `_sigma_scale_lambda`), toujours
  réversible via `--calibrate-sigma off`.

### Chantier 2 — Tests

- Lancer la **suite complète dans le venv local** (`~/.venvs/DeepEdgeBenchmark`) :
  `models/`, `honest_eval/`, `model_artifacts/` — **dont `model_artifacts/test_pipeline.py`
  qui exige `pandas_ta`** (indisponible dans le sandbox du run initial), `experiments/`.
- **Nouveaux tests** pour le branchement live : `mh` adopté (asymétrie skew-t, bornes Prophet
  log **strictement positives**, causalité de `sigma_scale`), et **non-régression** :
  `--calibrate-sigma off` ⇒ bornes identiques à l'historique.

### Chantier 3 — Vérification comportementale (PAS un re-run du jour)

Prouver, **chiffres à l'appui, sur les données offline reproductibles**
(`experiments/offline_prices.py`, `DONNEE~1.XLS`), **sans réécrire `tracking.db`** :
- le **σ EWMA varie réellement** dans le temps (variance du chemin σ_t non nulle ; se
  resserre en régime calme, s'élargit en régime agité) ;
- le **skew-t est réellement asymétrique** (|q_lo| ≠ q_hi sur la loi ajustée) ;
- **Prophet log** ⇒ bornes strictement positives **et** couverture améliorée vs espace prix ;
- la **correction `sigma_scale`** ≠ 1 quand le modèle est mécalibré, ≈ 1 sur l'état EWMA
  initial (non-régression douce) ;
- **CRPS/two-piece** : se réduit **exactement** au gaussien pour des bornes symétriques
  (non-régression) et **diffère** pour des bornes asymétriques.

## 3. Garde-fous (non négociables)

- **Causalité** : `sigma_scale` n'utilise que des cibles réalisées d'origine **strictement
  antérieure** à la date de prévision. Aucune fuite.
- **Réversibilité** : le switch off restaure l'historique **bit-for-bit** (test dédié).
- **`tracking.db`** : passer par le chemin d'insertion existant (`insert_oos_predictions` /
  `tracking_db.compute_real_flag`), **jamais** d'`INSERT` maison ; **ne pas réécrire**
  l'historique existant.
- **Ne pas toucher au point forecast** — uniquement les bornes/σ (LSTM : réseau et prévision
  ponctuelle inchangés).
- **`mh` reste le seul point d'extraction** des modèles (cf. docstring `pipeline.py` ≈ l.9-10
  « ne pas modifier les modèles depuis la pipeline »).
- **Ne rien pousser** : commit local, revue humaine d'abord.

## 4. Critère de succès

- Une prédiction live générée après branchement montre, **en base**, des bornes issues des
  adoptions (skew-t asymétrique pour ARIMA-GARCH, log pour Prophet, σ élargi/resserré par
  EWMA en D+1) — vérifiable par une requête `tracking.db`.
- `--calibrate-sigma off` ⇒ bornes **identiques** à l'historique (non-régression).
- Suite de tests **verte**, y compris `test_pipeline.py` en local.
- Vérification comportementale du chantier 3 **documentée** (petit rapport chiffré).
- EWMA **D+7 non activée** tant que non validée par un backtest 7-pas.
  *(Fait le 2026-07-31 — validée puis activée pour SARIMA/Prophet/Naive, LSTM exclu ; cf.
  mise à jour du §Périmètre.)*

## 5. Ce que ce brief ne fait PAS

- Ne re-valide pas les adoptions elles-mêmes (déjà fait, HANDOFF §2-6).
- Ne change pas le point forecast ni l'architecture des modèles.
- N'active pas l'EWMA en D+7 par défaut (décision/validation séparée).
- Ne touche pas au dashboard (déjà à jour) ni à `archives/`.
- Ne push pas.

## 6. Références (à lire avant de coder)

| Quoi | Où |
|---|---|
| Adoptions de référence à **mirrorer** | `models/{arima,sarima,naive,lstm,prophet}_model.py` (`run_*`, `next_step_*`) |
| Point d'entrée live **à migrer** | `benchmarks/multi_horizon.py` (`fit_*`, `forecast_from_fitted_*`, `forecast_horizons_*`) |
| Chaînage prod | `model_artifacts/pipeline.py` (`_forecast_horizon` ≈ l.555, `_forecast_all_horizons`, `_save_business_predictions`, `_run_model_d1`) |
| Lecture des réalisés + schéma | `experiments/prob_kpi_common.py` (`load_matrix_rows`), `validation/tracking_db.py` |
| Reconstruction CRPS dist-aware | `experiments/prob_kpi_common.py` (`sample_parametric` two-piece), `honest_eval/metrics.py` (`crps_parametric`) |
| Données offline reproductibles | `experiments/offline_prices.py`, `DONNEE~1.XLS` |
| Détails scientifiques | `HANDOFF_sigma_calibration_suivi.md` (§5 adoptions, §8 reste-à-faire) |
