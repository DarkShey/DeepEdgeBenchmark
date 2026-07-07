# BRIEF — Artefacts modèles : 2 quality gates (training + validation) → Run/

## 0. Contexte

On aligne les artefacts produits par les modèles sur le document **« DEITA — Artifacts des Modèles »** (Kyrio). On n'a que 4 modèles réels (ARIMA, SARIMA, Prophet, LSTM) + Naive → on ne traite que ceux-là (les 4 autres du doc — RandomForest, XGBoost, TimesNet, NHITS, TFT — n'existent pas encore, c'est normal).

**Décision de conception assumée** : on introduit une étape d'**entraînement + sérialisation** des modèles. Le code actuel ne sérialisait pas (walk-forward, re-fit à chaque pas). C'est un ajout volontaire, validé côté tuteur.

## 1. Objectif

Créer un pipeline **« train + validate »** qui, pour chaque combinaison **(modèle × actif × horizon)**, passe **2 quality gates** et produit les fichiers d'artefacts conformes au doc, déposés dans `Run/`. Supprimer le dossier `artifacts/` (remplacé par `Run/`).

## 2. Les 2 quality gates

- **Gate 1 — Training.** Entraîne le modèle sur les **85 % du début** de la fenêtre. Contrôle qualité : l'entraînement s'est bien passé (pas de NaN/inf, convergence). Si OK → **sérialise** le modèle + ses dépendances.
- **Gate 2 — Validation.** Évalue le modèle entraîné sur les **15 % de fin** (hors-échantillon, jamais vus). Contrôle qualité : métriques calculables + critères (couverture d'intervalle, skill vs naïf). Si OK → **sauve** les métriques.

Chaque gate doit logguer PASS/FAIL ; un échec de gate sur une combinaison n'interrompt pas les autres (comme `run_benchmark.py`).

## 3. Split des données

- **Fréquence** : daily (1d). Le training se fait à la même fréquence que la prédiction visée.
- **Fenêtre** : les **3 dernières années glissantes** (run_date − 3 ans → run_date).
- **Train** = 85 % du début, **Validation** = 15 % de fin. Split **chronologique strict** (jamais de mélange → aucune fuite de données / look-ahead).

## 4. Combinaisons & structure de dossiers

- **Modèles** : ARIMA, SARIMA, Prophet, LSTM (+ Naive, sans sérialisation).
- **Actifs** : BTC-USD, ETH-USD, SPY, ZN=F, TLT (les 5 de `calibration/regime/assets.py`).
- **Horizons** : D+1, D+7.
- **Un sous-dossier par combinaison** :

```
Run/<YYYYMMDD>-<modèle>-<asset>-<horizon>/
ex : Run/20260707-ARIMA-BTC-USD-D1/
```

→ 4 modèles × 5 actifs × 2 horizons = 40 dossiers (+ 10 pour Naive).

## 5. Fichiers produits par dossier (les « 5 fichiers » du doc)

| # | Fichier | Produit par | Contenu |
|---|---|---|---|
| 1 | `model.<pkl/json/h5>` | Training | modèle entraîné sérialisé |
| 2 | `scaler.pkl` | Training | normalizer — **LSTM uniquement** |
| 3 | `hyperparams.json` | Training | configuration du run |
| 4 | `metrics.json` | Validation | métriques OOS sur les 15 % |
| 5 | `metadata.json` | (les deux) | asset, fréquence, fenêtre, dates, commit git |

**Détail exact par modèle :**

- **ARIMA / SARIMA** : `model.pkl` (pickle de l'objet `ARIMAResults`) · `hyperparams.json` (p,d,q [+P,D,Q,m] + AIC/BIC) · `residuals.parquet` (résidus du fit, demandé par le doc) · `metrics.json` · `metadata.json`.
- **Prophet** : `model.json` (via `prophet.serialize.model_to_json`) · `hyperparams.json` (changepoint_prior_scale, seasonality_prior…) · `metrics.json` · `metadata.json` · *(optionnel `components.parquet` : trend/saisonnalité)*.
- **LSTM** : `model.h5` (sauvegarde Keras — c'est du **TensorFlow**, pas PyTorch → `.h5`) · `scaler.pkl` (le MinMaxScaler) · `hyperparams.json` (seq_len, hidden_size, epochs, lr, batch) · `metrics.json` · `metadata.json`.
- **Naive** : `metrics.json` · `metadata.json` (pas de modèle à sauver).

## 6. Contenu de `metrics.json`

`RMSE`, `MAE`, `MAPE`, `directional_accuracy`, `pi_coverage_95`, `n_val`, `horizon`, `asset`, `model`.

**Ajout (dashboard v2)** : `pi_width_min`, `pi_width_mean`, `pi_width_max` (largeur `pi_upper - pi_lower`
de l'intervalle à 95%, agrégée sur les points de validation Gate 2) — alimente le breakdown
modèle × horizon du dashboard.

### 6bis. `predictions.parquet` et `prices.parquet` (dashboard v2)

Produits par Gate 2 (si succès) en plus de `metrics.json`, pour le graphe prix + prédictions
du dashboard (`model_artifacts/generate_dashboard.py`) :

- `predictions.parquet` : colonnes `date, actual, predicted, pi_lower, pi_upper` — un point par
  jour de validation (D+1, walk-forward 1-step) ou par origine glissante (D+7, cf. §12 de
  `pipeline.py`). Absent si Gate 2 échoue.
- `prices.parquet` : colonnes `date, close` — historique complet (train + validation), écrit dans
  chaque dossier de combinaison (même redondance assumée que `metadata.json`, cf. §12) pour que le
  dossier reste auto-suffisant. La coupure train/validation est déjà connue via
  `metadata.json.train_end`.

## 7. Contenu de `metadata.json`

`asset`, `asset_class`, `frequency` (`1d`), `window_start`, `window_end`, `train_end` (date de coupure 85 %), `run_date`, `git_commit`, `seed`, versions des libs clés.

## 8. Nettoyage

- **Supprimer le dossier `artifacts/`** (remplacé par `Run/`).
- Si `run_benchmark.py` écrit encore dans `artifacts/`, le rediriger vers `Run/` ou le laisser en l'état mais ne plus s'appuyer dessus.
- Mettre à jour `.gitignore` / `README.md` si nécessaire.
- *(Optionnel)* si on veut garder un visuel, ajouter `forecast.png` par combinaison — non demandé par le doc, à confirmer.

## 9. Réutiliser, ne pas réécrire

Réutiliser les modèles existants (`models/*.py`) et les adaptateurs (`benchmarks/multi_horizon.py`) pour l'entraînement et la prédiction. Le nouveau code = **l'orchestration train + validate + save**, pas la logique des modèles. Ne pas dupliquer la modélisation.

## 10. Tests (hors-ligne, déterministes)

Un test qui, pour une combinaison sur **données synthétiques** (mock, pas de réseau) :
- vérifie que les 2 gates tournent et logguent PASS ;
- vérifie que les 5 fichiers attendus sont créés ;
- **round-trip** : le `model.*` sauvegardé se **recharge** et redonne les mêmes prédictions (preuve que la sérialisation est exploitable pour le déploiement — c'est tout l'intérêt d'un artifact) ;
- vérifie que `metrics.json` contient les bonnes clés.

## 11. Branche

Nouvelle branche dédiée, ex. `maeva/model-artifacts` (pas sur `maeva/tracking-db` ni `integration/validation`). Confirme la branche au début et reste dessus.

## 12. À garder en tête (points signalés)

- La sérialisation **contredit le choix initial** « pas de sérialisation » (§6-D) — c'est assumé et validé.
- Pour ARIMA/SARIMA/Prophet/LSTM, le modèle entraîné est **le même quel que soit l'horizon** (fit une fois, forecast 1 ou 7 pas) → `model.*` sera **identique** dans les dossiers `...-D1` et `...-D7` d'une même paire modèle-actif. Redondant mais cohérent avec la structure demandée. *(Alternative possible : `<date>-<modèle>-<asset>/` avec le modèle une seule fois + `metrics_D1.json`/`metrics_D7.json`.)*

## 13. Livrables

1. Le module d'orchestration `train + validate` (nouveau).
2. Les dossiers `Run/<YYYYMMDD>-<modèle>-<asset>-<horizon>/` peuplés pour un run réel.
3. Le test hors-ligne (dont le round-trip de sérialisation).
4. `artifacts/` supprimé.
5. À la fin : `git branch --show-current`, `git status`, et l'arborescence `Run/`.
