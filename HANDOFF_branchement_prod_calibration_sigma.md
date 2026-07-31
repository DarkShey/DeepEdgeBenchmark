# HANDOFF — Branchement prod de la calibration σ (mh.py, tracking.db) + vérification comportementale

**Date : 2026-07-31. Statut : terminé (branché, testé, vérifié), commit LOCAL uniquement.**
Suite directe de `BRIEF_branchement_prod_calibration_sigma.md`, lui-même suite de
`HANDOFF_sigma_calibration_suivi.md` (commit `4888afe`). Ce document couvre les trois
chantiers du brief : brancher les adoptions σ sur le chemin live/backtest D+7
(`benchmarks/multi_horizon.py`), les tests, et la vérification comportementale sur
données offline.

---

## 1. Le trou comblé

Avant ce travail : les adoptions σ (skew-t ARIMA-GARCH, EWMA Naive/LSTM, correction
multiplicative SARIMA/Prophet, Prophet log-espace) étaient actives dans `models/*.py`
(`run_*`, backtest Gate2 D1 + dashboard) mais **jamais exercées** par
`benchmarks/multi_horizon.py` (`mh`), le seul point d'extraction utilisé par :
- la **prévision live** écrite dans `tracking.db` (`model_artifacts/pipeline.py`
  `_forecast_all_horizons` → `_save_business_predictions`) ;
- le **backtest Gate2 D7** (`_run_model_d7_rolling` → `_forecast_horizon`).

`mh.fit_arima` codait `dist="normal"` en dur, Prophet fittait en espace prix, et
SARIMA/Naive/LSTM n'avaient aucun mécanisme de correction de σ. Résultat : aucune
prédiction trackée (D+1 ni D+7) n'avait les adoptions, malgré leur validation complète
dans `HANDOFF_sigma_calibration_suivi.md`.

## 2. Chantier 1 — Branchement

### 1a. `benchmarks/multi_horizon.py` aligné sur `models/*.py`

Chaque `fit_*`/`forecast_from_fitted_*` accepte maintenant les paramètres d'adoption,
tous **optionnels avec un défaut qui adopte** (`None` → défère au défaut de
`models/*.py`) :

| Modèle | Nouveau(x) paramètre(s) | Mécanisme |
|---|---|---|
| `fit_arima` / `forecast_from_fitted_arima` | `dist=None` → `arima_model.GARCH_DIST` ("skewt") | Bornes recalculées depuis les quantiles de la loi ajustée (mirror `arima_model._dist_shape`/`_std_quantiles`) au lieu du multiplicateur normal figé `Z_95`. `dist="normal"` reproduit **exactement** (bit-for-bit) l'ancien calcul symétrique — pas de passage par `_std_quantiles` dans ce cas, pour éviter toute dérive `1.96` vs `norm.ppf(0.975)=1.959964`. |
| `fit_prophet` / `forecast_from_fitted_prophet` | `log_space=None` → `prophet_model.LOG_SPACE` (True), `sigma_scale=None` | Fit sur `log(price)`, bornes exponentiées (lognormale en prix). `sigma_scale` appliqué **en espace log, avant l'exponentiation** (mirror `next_step_prophet`). |
| `forecast_from_fitted_sarima` / `forecast_horizons_sarima` | `sigma_scale=None` | Correction multiplicative `lo = point - (point-lo)*corr` via le nouveau helper `_scaled_bounds`. |
| `forecast_from_fitted_lstm` / `forecast_horizons_lstm` | `sigma_scale=None` | idem, mirror `run_lstm`. |
| `forecast_horizons_naive` | `sigma_scale=None` | idem, mirror `run_naive`. |

`sigma_scale` est un `dict {h_days: facteur}` : un horizon **absent du dict** (ou
`sigma_scale=None`) n'est **pas touché** — bornes brutes renvoyées telles quelles (le
helper `_scaled_bounds` retourne `lo_raw, hi_raw` sans aucune arithmétique dans ce cas,
pour garantir le bit-for-bit même sur une multiplication par 1.0 qui resterait exposée
à l'arrondi flottant d'une soustraction-addition).

### 1b. `validation/sigma_scale.py` (nouveau)

`sigma_scale(model, asset, horizon, as_of_cutoff_date, db_path)` : calcule
`sqrt(EWMA(z²))`, λ=0.94, sur les prédictions **réalisées** de `tracking.db`
(`source IN ('oos','live')`, `daily_duplicate=0`, `y_true NOT NULL`,
`frequence='daily'`, `horizon_type='daily'`), **strictement causal**
(`cutoff_date < as_of_cutoff_date`, ordre chronologique), état initial 1.0. Lecture
seule — aucun INSERT, `td.init_db(db_path)` appelé en tête (même garde-fou paresseux
que `tracking_db.save_prediction`, pour ne pas planter sur une base neuve).

### 1c. Réversibilité — `--calibrate-sigma off`

`model_artifacts/pipeline.py` :
- `DEFAULT_CALIBRATE_SIGMA = "on"`, CLI `--calibrate-sigma {on,off}`, threadé à travers
  `main → run_pipeline → process_asset_model → evaluate_gate2` / `_forecast_all_horizons`
  (et propagé aux 2 sous-processus modèles/LSTM).
- `_sigma_adoption_kwargs("off")` → `("normal", False)` (dist/log_space explicites,
  legacy) ; `"on"` → `(None, None)` (mh.py adopte ses propres défauts).
- `sigma_scale` calculé **une seule fois** par `process_asset_model` (pas de refetch
  par horizon), uniquement si `calibrate_sigma == "on"` et modèle dans
  `{SARIMA, Prophet, Naive, LSTM}` (jamais ARIMA-GARCH — sigma déjà dynamique via GARCH
  — ni TSDiff — échantillonnage natif) ; appliqué **uniquement** à l'horizon business
  D+1 effectif (`1 + business_lag`), jamais D+7 (cf. §3).
- `off` restaure `benchmarks/multi_horizon.py` bit-for-bit : `dist="normal"` +
  `log_space=False` + aucun `sigma_scale` transmis nulle part (live **et** Gate2 D7).

LSTM (subprocess isolé, `model_artifacts/lstm_worker.py`) : `sigma_scale_map` sérialisé
en JSON (`--sigma-scale-json`), appliqué uniquement à `--live-horizons` (jamais Gate2
D1/D7 du worker), même formule `_scaled_bounds` dupliquée localement (le worker
n'importe jamais `benchmarks.multi_horizon`, cf. docstring anti-deadlock TensorFlow).

### Périmètre D+1 vs D+7 (décision du brief, appliquée telle quelle)

- **Distributionnel** (skew-t ARIMA-GARCH, log-espace Prophet) : indépendant de
  l'horizon → actif sur **tous** les horizons dès que `calibrate_sigma="on"`, y compris
  le backtest Gate2 D7 (`_run_model_d7_rolling`, qui partage les mêmes fonctions `mh`
  — alignement voulu, pas un effet de bord : Gate2 D7 devient cohérent avec Gate2 D1,
  déjà migré en `models/*.py` depuis `4888afe`).
- **Correction EWMA `sigma_scale`** : **D+1 business uniquement**. Jamais transmis à
  `_run_model_d7_rolling` (Gate2 D7 backtest, h_days=5) ni à l'horizon business D+7
  (h_days=7+lag) — TODO explicite dans le code (`process_asset_model`) : activer D+7
  seulement après un backtest 7-pas dédié (cf. brief §2, non traité ici).

### Bug corrigé au passage

`_compute_metrics_for` (Gate2 D7 backtest) appelait
`arima_model.compute_metrics(actual, predicted, pi_lower=..., pi_upper=...)` — la
signature réelle est `compute_metrics(actual, predicted, pi_bounds={level: (lo, hi)})`
(seul module sur les 6 à avoir cette signature). Ça levait un `TypeError` **avant même
mes changements** (vérifié : `git show HEAD:model_artifacts/pipeline.py` a le même
appel), donc `[Gate2 FAIL] ARIMA-GARCH D7` en silence — jamais catché par la suite de
tests existante puisque `test_gate2_passes_and_metrics_have_expected_keys[D7-ARIMA-GARCH]`
existait mais passait avant l'exception (masqué par un test trop permissif, corrigé au
passage). Root cause hors périmètre du brief mais bloquait la vérification "Gate2 D7
verte" de ce chantier — corrigé par un cas spécial dans `_compute_metrics_for`.

## 3. Chantier 2 — Tests

**Nouveaux fichiers** :
- `benchmarks/test_multi_horizon.py` (18 tests) : `_scaled_bounds` (helper pur),
  ARIMA-GARCH (`dist="normal"` bit-for-bit vs formule legacy rejouée manuellement,
  défaut skew-t asymétrique — en prix **et** en log, cohérence fit/forecast du
  `dist` résolu), SARIMA/Naive/LSTM (`sigma_scale` absent = bit-for-bit,
  présent = élargit **uniquement** l'horizon ciblé — LSTM via un modèle/scaler
  duck-typés déterministes, aucun entraînement réel), Prophet (transform
  log/prix/`sigma_scale` via un faux modèle `.predict()` déterministe — cf. note
  méthodologique ci-dessous — + 2 tests `@pytest.mark.slow` avec fit réel sur
  `model.history["y"]` pour vérifier que `log_space` fitte bien sur `log(price)`).
- `validation/test_sigma_scale.py` (6 tests) : neutre à 1.0 sans historique, exclusion
  stricte des lignes à la date `as_of` ou postérieures (causalité), formule EWMA exacte
  (comparée à un calcul manuel), portée (modèle × actif × horizon) étanche,
  `daily_duplicate` ignoré, élargissement après une prédiction mécalibrée. Insertion
  **uniquement** via `tracking_db.save_prediction` + `evaluate_pending` (jamais de SQL
  d'écriture direct sur `predictions`).
- `benchmarks/conftest.py` : même garde-fou anti-deadlock TensorFlow que
  `models/conftest.py` (nécessaire dès que `mh.forecast_from_fitted_lstm` est exercé).
- `model_artifacts/test_pipeline.py` (+5 tests) : `_sigma_adoption_kwargs` (on/off),
  `--calibrate-sigma off` reproduit une bande ARIMA-GARCH live symétrique / `"on"` une
  bande asymétrique (bout en bout CLI → `process_asset_model` →
  `_forecast_all_horizons` → `mh`), `sigma_scale` mocké élargit **uniquement** la ligne
  `horizon=1` de `tracking.db` (jamais `horizon=7`) — vérifié par une requête SQL
  directe sur une base temporaire.

**Note méthodologique (Prophet)** : les tests de transformation
(`log_space`/`sigma_scale`) utilisent un faux modèle Prophet dont `.predict()` est
déterministe, pas un fit réel suivi d'un second `.predict()`. Découvert en écrivant ces
tests : `Prophet.predict()` tire ses bornes `yhat_lower`/`yhat_upper` par échantillonnage
Monte-Carlo interne **re-tiré à chaque appel** (RNG non fixée) — deux `.predict()`
consécutifs sur le **même** modèle déjà fitté donnent des bornes légèrement différentes.
Un premier essai de test (assertion bit-for-bit sur deux `.predict()` séparés) échouait
donc sporadiquement sur ce bruit, sans rapport avec la migration. Les tests actuels
isolent la logique de transformation (mon code) du générateur Monte-Carlo de Prophet
(pas mon code).

**Résultats (venv local `~/.venvs/DeepEdgeBenchmark`, `pandas_ta` disponible)** —
chaque dossier lancé séparément (`pytest <dossier>/`, pas tous ensemble : collision de
nom `conftest.py` entre `models/` et `experiments/` en mode d'import "prepend" sans
`__init__.py`, **pré-existante**, non liée à ce travail) :

| Dossier | Résultat |
|---|---|
| `models/` | 77 passed |
| `honest_eval/` | 14 passed |
| `experiments/` | 134 passed |
| `model_artifacts/` (dont `test_pipeline.py`, `pandas_ta` OK) | 39 passed |
| `validation/` + `benchmarks/test_multi_horizon.py` | 204 passed |

**Total : 468 passed, 0 failed.**

## 4. Chantier 3 — Vérification comportementale (`experiments/verify_live_sigma_calibration.py`)

Script autonome, données offline (`experiments/offline_prices.py` / `DONNEE~1.XLS`),
**aucun accès réseau**, **`tracking.db` jamais touché** (bases sqlite temporaires
créées/détruites par test, via `tracking_db.save_prediction` + `evaluate_pending` —
jamais d'INSERT/UPDATE maison). Rapport chiffré généré à chaque exécution :
`RAPPORT_verification_calibration_sigma.md` (racine du repo).

Les 5 preuves demandées par le brief, toutes vérifiées (voir le rapport pour les
chiffres complets) :

1. **σ_scale varie dans le temps et suit le régime** : simulé sur BTC-USD, deux
   fenêtres historiquement contrastées (crash Terra/Luna mai-juin 2022 vs été 2023
   calme), walk-forward Naive `sigma_mode="frozen"` rejoué jour par jour dans une base
   temporaire. `sigma_scale[0] == 1.0` exact (aucun historique), variance du chemin
   `> 0` dans les deux fenêtres, moyenne plus élevée en régime agité qu'en régime calme.
2. **Skew-t réellement asymétrique** : `arima_model._std_quantiles` sur BTC-USD et SPY
   (offline 2023-2024) donne `|q_lo| != q_hi` (asymétrie relative 12-16%), et
   l'asymétrie survit à la construction complète des bornes multi-horizon en prix
   (`forecast_horizons_arima`, D+1 et D+7).
3. **Prophet log-espace** : bornes strictement positives par construction (exponentiation
   d'un nombre réel), et couverture empirique supérieure à l'espace prix sur un
   walk-forward court (illustratif — la validation statistique rigoureuse à 3 fenêtres
   ×2 actifs reste celle de `HANDOFF_sigma_calibration_suivi.md` §4).
4. **σ_scale ≈ 1 à l'état initial** : couvert par le point 1 (`sigma_scale[0] == 1.0`
   exact dans les deux fenêtres).
5. **CRPS two-piece** : bornes symétriques → écart relatif au CRPS gaussien fermé
   < 0.15% (test de Kolmogorov-Smirnov non rejeté, p > 0.8) ; bornes asymétriques →
   écart de 43 à 117% (mesurablement différent).

**Incident résolu pendant la vérification (hors périmètre du code de prod, mais
documenté)** : `experiments/crps_metrics.crps_empirical` calcule le second terme du CRPS
via une matrice `n×n` de paires (`np.abs(x[:,None]-x[None,:])`) — avec
`n_samples=200_000` ça alloue littéralement `200000² × 8 octets = 320 Go`, ce qui a fait
planter (`SIGKILL`, RAM 8 Go) la machine de vérification. Le script utilise maintenant
`honest_eval.metrics._crps_empirical_sorted` (formule `O(n log n)` de Gneiting & Raftery
2007 eq. 20, déjà présente et testée dans le repo, même résultat mathématique) —
`crps_metrics.crps_empirical` lui-même n'a pas été touché (utilisé ailleurs avec des
`n_samples` bien plus petits, jamais un problème dans son usage actuel).

**Nettoyage disque effectué en cours de route** (bloquait aussi la suite pytest, avec
la même racine "disque plein" causant un `OSError: No space left on device` sur
`models/test_prophet_model.py`, sans rapport avec le code) : caches purs et
régénérables uniquement (`conda clean --packages --tarballs`, `~/.venvs`/`ShipIt`
VSCode+Claude, Firefox, Playwright, pip) — validé item par item avec l'utilisatrice
avant suppression, ~6-7 Go libérés, rien de personnel touché.

## 5. Garde-fous — respectés

- **Causalité** : `sigma_scale` n'utilise que `cutoff_date < as_of_cutoff_date`
  (test dédié `validation/test_sigma_scale.py`).
- **Réversibilité bit-for-bit** : `--calibrate-sigma off` (tests dédiés côté `mh.py` ET
  côté pipeline CLI).
- **`tracking.db`** : aucun INSERT/UPDATE fait main, partout via
  `tracking_db.save_prediction`/`evaluate_pending` (existants, inchangés) ; le script de
  vérification n'écrit jamais dans la vraie base.
- **Point forecast intact** : `sigma_scale`/`dist`/`log_space` ne touchent jamais le
  `point` retourné (vérifié dans tous les tests `_scaled_bounds`/transform).
- **`mh` seul point d'extraction** : aucune logique de modélisation dupliquée dans
  `pipeline.py` (hormis `lstm_worker.py`, déjà dupliqué pour l'isolation TensorFlow,
  cf. son propre docstring — la correction `sigma_scale` y est mirrorée à l'identique).
- **Rien poussé** : commit local uniquement (§6).

## 6. Commit

Un seul commit local, pas de push. Fichiers modifiés : `benchmarks/multi_horizon.py`,
`model_artifacts/pipeline.py`, `model_artifacts/lstm_worker.py`,
`model_artifacts/test_pipeline.py`. Fichiers nouveaux : `validation/sigma_scale.py`,
`validation/test_sigma_scale.py`, `benchmarks/conftest.py`,
`benchmarks/test_multi_horizon.py`, `experiments/verify_live_sigma_calibration.py`,
`RAPPORT_verification_calibration_sigma.md`.

## 7. Reste à faire (connu, non traité ici — hors périmètre du brief)

1. **D+7 EWMA** : désactivée par design (cf. §2 Périmètre) — activer seulement après un
   backtest 7-pas dédié (brief §2, jamais fait à ce jour).
2. Étendre la vérification comportementale Prophet (§4.3) à un walk-forward complet
   (15+ points, refit_freq=1) une fois sur une machine avec plus de RAM — le run réduit
   ici (9 points, refit_freq=3) reste illustratif, pas une nouvelle validation
   statistique.
3. La collision de nom `conftest.py` entre `models/` et `experiments/` (pré-existante,
   cf. §3) empêche de lancer plusieurs dossiers de tests en une seule invocation pytest
   — à corriger un jour via `--import-mode=importlib` ou des packages `__init__.py`,
   hors périmètre de ce brief.
4. Espace disque de la machine de dev : reste à 94-95% même après nettoyage (`Run/` =
   2.3 Go, `.venvs` = 2.7 Go, gros de l'occupation ailleurs sur la machine, ~175 Go
   hors-projet) — à surveiller si d'autres runs lourds (LSTM/Prophet) sont relancés.

## 8. Reproduction

```bash
# suite complète (séparément par dossier, cf. §3)
~/.venvs/DeepEdgeBenchmark/bin/python -m pytest models/ -q
~/.venvs/DeepEdgeBenchmark/bin/python -m pytest honest_eval/ -q
~/.venvs/DeepEdgeBenchmark/bin/python -m pytest experiments/ -q
~/.venvs/DeepEdgeBenchmark/bin/python -m pytest model_artifacts/ -q
~/.venvs/DeepEdgeBenchmark/bin/python -m pytest validation/ benchmarks/test_multi_horizon.py -q

# vérification comportementale (Chantier 3, régénère RAPPORT_verification_calibration_sigma.md)
~/.venvs/DeepEdgeBenchmark/bin/python experiments/verify_live_sigma_calibration.py

# pipeline réel, adoptions actives (défaut) vs comportement historique
python -m model_artifacts.pipeline --calibrate-sigma on   # défaut
python -m model_artifacts.pipeline --calibrate-sigma off  # bit-for-bit historique
```
