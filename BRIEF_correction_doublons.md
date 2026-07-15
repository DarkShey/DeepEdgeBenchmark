# BRIEF — Correction : dédoublonnage OOS par flag `daily_duplicate`

## 0. Contexte

L'audit (`validation/audit/audit_tracking_db.md`, branche `maeva/audit-tracking-db`) a établi :
- **14 174 lignes** dans `predictions` : `live=200`, `oos=13 974`.
- **Doublons live : 0** (déjà protégés par `UNIQUE(tc_id, model, cutoff_date)`).
- **Doublons oos : 9 875 lignes excédentaires** sur **3 998 groupes**.
- **Cause** : l'index `idx_predictions_oos_unique` inclut `run_id`. Chaque rejeu de backtest
  reçoit un `run_id` différent, donc les lignes s'empilent au lieu d'être écrasées.

**Décision (tuteur) :** on **ne supprime rien**. On **marque** les doublons avec un flag et on
**conserve le dernier lancement** de chaque prédiction. Approche réversible et auditable.

> Le correctif **Prophet** (absence de walk-forward) est **hors périmètre** de ce brief : il fera
> l'objet d'un brief séparé. Ici, on ne fait QUE le dédoublonnage par flag.

## 1. Objectif

Ajouter une colonne `daily_duplicate` à `predictions` et, pour chaque prédiction en doublon,
la mettre à `1` sur toutes les copies **sauf la plus récente** (qui reste à `0`). Aucune ligne
supprimée. Opération **idempotente** (rejouable) et **réversible** (remise à `0` possible).

## 2. Règle de dédoublonnage

**Groupe de doublons (clé métier)** — deux lignes sont « la même prédiction » si elles ont le
même :
`(source, model, asset, horizon, cutoff_date, target_date)`.

> Exemple : « ARIMA / BTC-USD / horizon 1 / cutoff 2026-07-07 / target 2026-07-08 » lancé 3 fois
> = 1 groupe de 3 lignes.

**Ligne conservée (`daily_duplicate = 0`)** = le **dernier lancement** du groupe, déterminé par
ordre de priorité :
1. `run_id` le plus récent (départage principal) ;
2. à égalité, `created_at` le plus récent ;
3. à égalité, `id` le plus grand (l'`id` autoincrement = ordre d'insertion réel = dernier
   inséré en base ; c'est le juge de paix ultime, il est toujours unique).

**Toutes les autres lignes du groupe** → `daily_duplicate = 1`.

**Périmètre d'application :** uniquement `source = 'oos'`. Les lignes `live` (0 doublon) doivent
rester à `daily_duplicate = 0`.

## 3. Garde-fous

- Branche dédiée : `maeva/correction-doublons` (sans accent).
- **Sauvegarde obligatoire avant toute écriture** : `cp validation/tracking.db
  validation/tracking.db.bak_YYYYMMDD` (le garder hors du commit / dans `.gitignore` si besoin).
- Toute écriture dans **une seule transaction** (`BEGIN … COMMIT`), rollback si un contrôle échoue.
- **Ne pas modifier** `model_artifacts/pipeline.py`, `models/*.py`.
- On **ne touche pas** aux index / contraintes existants dans ce brief (voir §7 pour la suite).

## 4. Modifications à faire

### 4.1 Schéma — nouvelle colonne (dans `validation/tracking_db.py`, `init_db()`)

Ajouter à la table `predictions` :

```sql
daily_duplicate INTEGER NOT NULL DEFAULT 0
```

- Ajout **idempotent** : vérifier via `PRAGMA table_info(predictions)` si la colonne existe
  déjà avant `ALTER TABLE predictions ADD COLUMN daily_duplicate INTEGER NOT NULL DEFAULT 0`.
- Répercuter la colonne dans le bloc `CREATE TABLE` (et dans `predictions_new` de la fonction
  de migration `_migrate_predictions_add_source`) pour qu'une base créée à neuf l'ait aussi.

### 4.2 Fonction de flag (nouvelle fonction dans `tracking_db.py`)

```python
def flag_daily_duplicates(db_path=DEFAULT_DB_PATH) -> int:
    """Marque daily_duplicate=1 sur toutes les copies OOS sauf le dernier lancement
    de chaque prédiction (source, model, asset, horizon, cutoff_date, target_date).
    Idempotent. Retourne le nombre de lignes passées à 1."""
```

Logique SQL (repartir de 0 à chaque exécution → idempotence) :

```sql
-- 1) reset (pour rejouabilité)
UPDATE predictions SET daily_duplicate = 0 WHERE source = 'oos';

-- 2) élire le survivant de chaque groupe = dernier lancement
--    puis flaguer tout le reste
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY source, model, asset, horizon, cutoff_date, target_date
               ORDER BY run_id DESC, created_at DESC, id DESC
           ) AS rn
    FROM predictions
    WHERE source = 'oos'
)
UPDATE predictions
SET daily_duplicate = 1
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
```

### 4.3 Requêtes de lecture — ignorer les doublons flaggés

Les fonctions qui agrègent ou exportent l'OOS doivent exclure les lignes flaggées, pour que les
analyses ne comptent chaque prédiction qu'une fois. Ajouter `AND daily_duplicate = 0` (ou
`WHERE daily_duplicate = 0`) là où des lignes OOS sont lues à des fins de statistiques/export :
notamment `export_csv()` quand `source != 'live'`, et toute future agrégation OOS.
Ne **pas** filtrer dans les fonctions purement `source='live'` (elles n'ont pas de doublon).

## 5. Exécution (script one-shot)

Fournir `validation/correction/apply_daily_duplicate_flag.py` qui :
1. fait la sauvegarde `.bak` ;
2. appelle `init_db()` (crée la colonne) puis `flag_daily_duplicates()` ;
3. affiche le rapport de contrôle (§6).

Le script ne doit **rien supprimer**.

## 6. Contrôles de non-régression (à afficher en fin d'exécution)

- Nombre de lignes total **inchangé** : `= 14 174` avant et après (aucune suppression).
- `daily_duplicate = 1` attendu : **9 875** ; `daily_duplicate = 0` : **4 299**
  (= 4 299 − 200 live = 4 099 groupes OOS uniques + 200 live ; vérifier la cohérence avec
  les 3 998 groupes en doublon du rapport d'audit).
- **Chaque groupe métier OOS a exactement un survivant** :
  ```sql
  SELECT COUNT(*) FROM (
    SELECT 1 FROM predictions WHERE source='oos' AND daily_duplicate=0
    GROUP BY source, model, asset, horizon, cutoff_date, target_date
    HAVING COUNT(*) <> 1
  );  -- doit renvoyer 0
  ```
- Les 200 lignes `live` sont toutes à `daily_duplicate = 0`.
- Rejouer le script une 2ᵉ fois ne change aucun compte (idempotence).
- Recouper la liste des `id` flaggés avec `audit_keep_drop.csv` (colonne `decision='drop'`) :
  ils doivent correspondre.

## 7. Tests

Ajouter à `validation/test_tracking_db.py` :
- un cas où 3 lignes OOS de même clé métier et de `run_id` croissants → seule celle du `run_id`
  max reste à 0 ;
- un cas d'égalité de `run_id` départagé par `id` max ;
- vérif qu'une ligne `live` n'est jamais flaggée ;
- vérif d'idempotence (2 appels → même résultat).

## 8. Suite (hors périmètre, pour mémoire)

- **Prévention durable** : retirer `run_id` de `idx_predictions_oos_unique` pour que les rejeux
  s'écrasent nativement — à décider dans un brief ultérieur (change le comportement d'ingestion).
- **Correctif Prophet** (walk-forward) : brief séparé.
