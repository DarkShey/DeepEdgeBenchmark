# BRIEF — Audit de la base de suivi (`validation/tracking.db`)

## 0. Contexte & périmètre

La table `predictions` de `validation/tracking.db` mélange deux régimes (`source='live'`
et `source='oos'`). En inspectant un export, on constate deux anomalies :

1. **Des doublons** : une même prédiction `(source, modèle, actif, horizon, date de coupure,
   date cible)` apparaît plusieurs fois (ex. observé sur un extrait : SARIMA OOS × 3 pour le
   2026-07-07 avec des valeurs quasi identiques, LSTM/Prophet/TSDiff × 2 ou × 3).
2. **Prophet incohérent en OOS** : les prédictions OOS de Prophet sont aberrantes
   (~+20 %, soit +13 000 à +19 000 points face au réel) alors que les prédictions **live**
   du même modèle sont correctes (±1 500).

> **Ce brief = AUDIT en LECTURE SEULE.** Objectif : diagnostiquer et chiffrer, **sans rien
> supprimer ni corriger**. La suppression des doublons et le correctif Prophet feront l'objet
> d'un **brief de correction séparé**, écrit à partir des conclusions de cet audit.

## 1. Objectif

Produire un **rapport d'audit** qui :
1. inventorie tout le contenu de `predictions` (volumes par `source`, modèle, actif, horizon,
   plage de dates, taux d'évaluation) ;
2. identifie et **chiffre tous les doublons** selon la règle « un seul enregistrement par
   prédiction métier, celui du dernier run » (§5) ;
3. produit la **liste des lignes à garder vs à supprimer** (dry-run, aucune suppression) ;
4. **diagnostique la cause racine** de l'anomalie Prophet OOS (sans la corriger).

## 2. Garde-fous (impératif)

- **Ne rien écrire dans `validation/tracking.db`** : aucun `DELETE`, `UPDATE`, `INSERT`,
  `VACUUM`, ni migration de schéma. Ouvrir la base en lecture seule, ou travailler sur une
  **copie** (`cp validation/tracking.db /tmp/audit.db` puis auditer la copie).
- **Ne pas modifier** `tracking_db.py`, `model_artifacts/pipeline.py`, `models/*.py`,
  `validation/sim_trades.py`. Audit uniquement.
- Travailler sur une **branche dédiée** : `maeva/audit-tracking-db` (sans accent).
- Tout le code d'audit va dans un dossier/fichier isolé, ex. `validation/audit/` ou
  `validation/audit_tracking_db.py`.

## 3. Schéma existant (rappel, à ne pas modifier)

Table `predictions` (cf. `validation/tracking_db.py`) avec notamment :
`id` (PK autoincrement), `run_id`, `tc_id` (NULL pour l'OOS), `model`, `asset`, `horizon`,
`cutoff_date`, `target_date`, `regime`, `last_close`, `y_pred`, `y_lower`, `y_upper`,
`y_true` (NULL tant que non évaluée), `evaluated_at`, `created_at`, `source`
(`'live'` par défaut, ou `'oos'`).

Deux contraintes d'unicité sont **déjà** en place :

- `UNIQUE (tc_id, model, cutoff_date)` → protège les lignes **live** (idempotence des
  re-runs). OK.
- Index partiel `idx_predictions_oos_unique ON (source, run_id, model, asset, horizon,
  cutoff_date) WHERE source='oos'`.

**⚠️ Cause racine des doublons OOS à confirmer :** l'index OOS **inclut `run_id`**. Deux
backtests joués à des moments différents produisent des `run_id` distincts → l'index ne les
considère PAS comme des doublons → les lignes s'**empilent** pour la même prédiction métier.
C'est l'hypothèse principale à valider dans l'audit.

## 4. Travail d'audit demandé

### A. Inventaire

Reporter, par `source` puis ventilé par `(model, asset, horizon)` :
nombre de lignes, plage `min/max(cutoff_date)`, nombre de `run_id` distincts,
nombre de lignes évaluées (`y_true IS NOT NULL`) vs en attente.

### B. Détection des doublons

Définir la **clé métier** d'une prédiction (§5), puis pour chaque clé comptant plus d'une
ligne, lister : la clé, le nombre de copies, les `run_id` concernés, les `y_pred` de chaque
copie et leur écart-type (pour distinguer les vrais re-runs identiques d'une divergence de
valeurs). Fournir le total de lignes en doublon par `source`.

Requête de départ (à adapter) :

```sql
SELECT source, model, asset, horizon, cutoff_date, target_date,
       COUNT(*)                AS n_copies,
       COUNT(DISTINCT run_id)  AS n_runs,
       GROUP_CONCAT(run_id)    AS run_ids,
       ROUND(MAX(y_pred)-MIN(y_pred), 2) AS spread_y_pred
FROM predictions
GROUP BY source, model, asset, horizon, cutoff_date, target_date
HAVING COUNT(*) > 1
ORDER BY n_copies DESC, cutoff_date DESC;
```

### C. Règle « ne garder que le dernier run » — voir §5, en DRY-RUN

Produire la liste `keep` / `drop` **sans supprimer**. Chaque ligne `drop` doit indiquer
`id`, la clé métier, le `run_id` écarté et le `run_id` gagnant (celui conservé).

### D. Diagnostic Prophet — voir §6.

## 5. Règle « dernier run par date »

Pour chaque **clé métier** = `(source, model, asset, horizon, cutoff_date, target_date)`,
ne conserver qu'**une** ligne : celle issue du **run le plus récent**. Ordre de récence à
déterminer et documenter :

- **live** : `run_id` de forme `run_YYYYMMDDThhmmss` → l'ordre lexicographique = ordre
  chronologique. Le plus grand gagne.
- **oos** : `run_id` de forme `YYYYMMDD-MODEL-ASSET-D1` → extraire le préfixe date `YYYYMMDD`,
  le plus récent gagne.
- **Départage** si `run_id` insuffisant : `created_at` le plus récent, puis `id` (MAX) en
  dernier recours.

L'audit doit **vérifier que cette règle est bien applicable** (formats de `run_id` réellement
présents) et signaler tout `run_id` qui ne suit aucun des deux formats.

## 6. Diagnostic de l'anomalie Prophet

Fait mesuré : Prophet **OOS** sur-estime d'environ +20 % (constaté sur extrait : réel ≈ 62–64 k,
prédiction ≈ 76–81 k), alors que Prophet **live** est correct sur les mêmes dates. Même modèle,
comportements opposés → la divergence vient du **chemin de génération/ingestion**, pas de la
formule seule.

Les deux chemins à comparer :
- **live** : `model_artifacts/pipeline.py` → `save_prediction()` → table.
- **oos** : `validation/sim_trades.py` ingère `Run/*-Prophet-*-D1/predictions.parquet`.

À investiguer (lecture seule ; **ne rien corriger**, juste identifier et prouver la cause) :

1. Ouvrir un `Run/*-Prophet-*-D1/predictions.parquet` aberrant et comparer ses colonnes à ce
   que `sim_trades.py` lit réellement : **quelle colonne est prise comme `y_pred` ?**
   (`yhat` vs `yhat_upper`/`trend`/une valeur en échelle log ou non inversée ?).
2. Vérifier l'échelle : les valeurs du parquet sont-elles déjà en niveau de prix, ou en
   log/rendement non ré-exponentié ? Un facteur ~1,2 constant oriente vers une transformation
   (log, cap logistique, saturation de tendance) mal inversée côté OOS.
3. Comparer les réglages Prophet entre le chemin live et le backtest OOS
   (`growth='linear'/'logistic'`, `cap`, saisonnalité, historique d'entraînement utilisé).
4. Confirmer que l'anomalie est **systématique** (toutes les lignes Prophet OOS) et **propre à
   Prophet** (les autres modèles OOS sont sains).

Livrable de cette section : **la cause racine identifiée, avec la preuve** (extrait de parquet,
ligne de code de lecture, ou différence de config), + une recommandation de correctif à traiter
dans le brief suivant.

## 7. Livrables attendus

1. `validation/audit/audit_tracking_db.md` — le rapport : inventaire (§4A), tableau des
   doublons (§4B), conclusion sur la cause racine des doublons (§3), diagnostic Prophet (§6).
2. `validation/audit/audit_keep_drop.csv` — une ligne par enregistrement avec une colonne
   `decision` (`keep`/`drop`), la clé métier, `run_id`, et `run_id_gagnant` pour les `drop`.
3. Un **résumé chiffré** en tête de rapport : total lignes, nb de doublons live, nb de doublons
   oos, nb de lignes qui seraient supprimées, nb de lignes Prophet OOS impactées.

## 8. Hors périmètre (→ brief de correction à venir)

- Suppression effective des doublons.
- Correctif du code Prophet / de l'ingestion `sim_trades.py`.
- Ajout/modification de contrainte d'unicité (ex. retirer `run_id` de l'index OOS) pour empêcher
  la réapparition des doublons.

Ces points seront traités **après validation du rapport d'audit**.
