# BRIEF — Unification de la base (`predictions` unique)

> **Statut** : à implémenter. Chantier 100 % dans `validation/` (mes test cases).
> **Objet** : supprimer la redondance `predictions` (live) / `daily_oos_log` (oos) — deux tables qui décrivent
> **la même entité** avec des noms de colonnes différents — en fusionnant tout dans **une seule table de brut,
> `predictions`**, distinguée par une colonne `source`. Puis brancher mes deux test cases (`bull_calm_d1`,
> `sideways_d1`) dessus.
> **Périmètre strict** : `validation/sim_trades.py`, `validation/tracking_db.py`, `validation/evaluate_daily.py`
> et leurs tests. **Ne toucher NI** au dossier `test_cases/` (autre stagiaire), **NI** aux modèles, **NI** à
> `model_artifacts/pipeline.py`.

---

## 0. Constat de départ (redondance à tuer)

| `predictions` (live) | `daily_oos_log` (oos) | Signification |
|---|---|---|
| `cutoff_date` | `d_date` | D (jour de décision) |
| `target_date` | `target_date` | D+1 |
| `last_close` | `reference_price` | P(D) |
| `y_pred` | `predicted` | PI_mid |
| `y_lower` | `pi_lower` | PI_low |
| `y_upper` | `pi_upper` | PI_high |
| `y_true` | `realized_price` | P(D+1) réalisé |
| `regime`, `model`, `asset`, `horizon` | idem | — |

Ce sont les **mêmes données**. En plus, `daily_oos_log` contient aujourd'hui ~100 lignes `source="live"` qui
**dupliquent** `predictions`. On garde `predictions` comme table unique de brut et on supprime `daily_oos_log`.

`sim_trades` (résultats calculés des test cases) **reste** : c'est une couche différente du brut, pas une redondance.

**Base finale visée** : `predictions` (brut : live + oos via `source`) + `sim_trades` (résultats) + `test_cases`
(métadonnées). Trois tables, chacune un rôle, zéro doublon.

---

## 1. Audit préalable (à documenter en clair AVANT de coder)

Compter et afficher, dans `validation/tracking.db` :
- `predictions` : nb de lignes, colonnes, nb en `horizon=1`, nb résolues (`y_true` non NULL), dates distinctes.
- `daily_oos_log` : nb de lignes par `source`.
- `sim_trades` : nb de lignes par `(source, rule_version)`.

Confirmer que le schéma de `predictions` couvre déjà tout ce dont l'OOS a besoin (cf. §0) — donc aucune colonne
métier nouvelle n'est nécessaire pour accueillir l'OOS.

---

## 2. Migration — une seule table de brut

1. **Colonne `source`** ajoutée à `predictions` : `TEXT DEFAULT 'live'`, via migration paresseuse
   (`PRAGMA table_info` puis `ALTER TABLE ADD COLUMN` si absente). Les lignes existantes deviennent `'live'`
   sans intervention ; le pipeline continue d'écrire du `'live'` par défaut, **sans modification**.
2. **Ingestion OOS → `predictions`** : garder la lecture des `Run/*-D1/predictions.parquet` comme *source*,
   mais **insérer désormais dans `predictions`** avec `source='oos'`. Mapping direct :
   `cutoff_date=d_date`, `target_date`, `last_close=reference_price`, `y_pred=predicted`, `y_lower=pi_lower`,
   `y_upper=pi_upper`, `y_true=realized_price`, `regime`, `model`, `asset`, `horizon=1`, `run_id` (dossier Run).
   Colonnes métier live (`tc_id`, `verdict_integrite`, `verdict_plausibilite`, `in_interval`, `abs_error`,
   `abs_error_naif`, `beats_naif`, `direction_correct`, `created_at`, `evaluated_at`) laissées **NULL** pour l'OOS.
   Idempotent (relançable sans doublon).
3. **Supprimer la table `daily_oos_log`** et tout code qui la crée/écrit. Purger ses lignes `live` en double.
4. **Lecture des test cases depuis `predictions`** : adapter `generate_sim_trades`, `kpi_report`,
   `sync_live_trades` et le lecteur du rapport Sideways pour lire `predictions` avec alias normalisés
   (`last_close AS reference_price`, `y_pred AS predicted`, `y_lower AS pi_lower`, `y_upper AS pi_upper`,
   `y_true AS realized_price`, `cutoff_date AS d_date`), filtré par `source` selon le besoin
   (`'oos'` pour le backtest, `'live'` pour le suivi). Une **vue SQL `all_predictions`** ou une petite fonction
   de lecture unique est acceptable, tant qu'il n'existe **qu'une seule table physique de brut**.
5. **`sim_trades` inchangé.**

---

## 3. Garde-fous — ne rien casser ailleurs (à vérifier explicitement)

- **Toutes les requêtes live** sur `predictions` doivent filtrer **`source='live'`** pour ignorer l'OOS :
  en particulier `tracking_db.evaluate_pending` (le cron résout les prédictions échues) et
  `fetch_predictions_for_run` / l'export `business_validation.json`. Un run OOS ne doit **jamais** être ramassé
  par le cron ni exporté comme prédiction business.
- **`tracking_db.save_prediction`** doit écrire `source='live'`.
- **`model_artifacts/pipeline.py` non modifié** : confirmer qu'il fonctionne toujours grâce au défaut `'live'`.
  Faire tourner un smoke test d'écriture + un `evaluate_daily` à vide (idempotent, ne casse rien).
- **`test_cases/` (autre stagiaire) n'utilise pas `tracking.db`** (il lit des CSV) : confirmer l'absence
  d'interférence, aucun fichier partagé.
- Vérifier que `dashboards/` / `model_artifacts/generate_dashboard.py` lisent les `metrics.json`/parquet et non
  `daily_oos_log` (donc non impactés) — à confirmer par un grep.

---

## 4. Non-régression obligatoire

Après migration, les résultats calculés depuis `predictions` doivent être **strictement identiques** à ceux
d'avant (calculés depuis `daily_oos_log`/parquet) :

- **Bull-Calm** : `6 613 signaux, counter −418`.
- **Sideways** (k=0,10) : `taux_signal 73 %, taux_justesse 96,4 %`.

Si un seul chiffre diffère, **s'arrêter et signaler** (c'est le signe qu'un mapping de colonne ou un filtre
`source` est faux).

---

## 5. Finalisation

- Mettre à jour les tests qui référençaient `daily_oos_log`.
- Ajouter un **test de coexistence** : live + oos dans `predictions` distingués par `source` ; une requête live
  (ex. `evaluate_pending`) ignore bien les lignes `oos`.
- `pytest validation/ -q` **vert**.
- Commit (`validation/tracking.db` est force-inclus via `!validation/tracking.db` dans `.gitignore`) + push.
- **Montrer le plan + l'audit (§1) avant d'écrire du code.**
