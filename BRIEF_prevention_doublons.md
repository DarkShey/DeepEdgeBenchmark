# BRIEF — Prévention durable : zéro doublon OOS, garanti par contrainte

## 0. Contexte & décision

Les doublons OOS existants ont été flaggés (`daily_duplicate=1`, 9 875 lignes), et `sim_trades`
a été réconcilié. Décision maintenant : **plus aucun doublon, ni dans l'historique, ni à
l'avenir.** On passe donc de l'approche « flag réversible » à une **garantie dure** :

1. **supprimer physiquement** les 9 875 lignes flaggées (table propre) ;
2. **remplacer l'index d'unicité OOS** pour qu'il porte sur la clé métier **sans `run_id`** →
   un doublon devient impossible à insérer ;
3. **passer l'ingestion en « garde le dernier run »** (upsert) au lieu d'empiler.

> ⚠️ Le tuteur avait initialement demandé de *flaguer* (ne pas supprimer). Ce brief **supprime**
> les lignes flaggées. Réversible via `validation/tracking.db.bak_*`. À signaler au tuteur.

## 1. Cause à éliminer

Index actuel (`tracking_db.py`, `init_db`, l.104-108) :

```sql
CREATE UNIQUE INDEX idx_predictions_oos_unique
ON predictions (source, run_id, model, asset, horizon, cutoff_date)
WHERE source = 'oos';
```

`run_id` est dans la clé → deux backtests (donc deux `run_id`) sur les mêmes dates ne collisionnent
jamais et s'empilent. **Retirer `run_id`** rend la même prédiction unique quel que soit le run.

## 2. Garde-fous

- Branche dédiée : `maeva/prevention-doublons` (sans accent).
- **Nouvelle sauvegarde** `validation/tracking.db.bak_YYYYMMDD` avant écriture (gitignorée).
- Tout en **une transaction**, rollback si un contrôle échoue.
- **Live jamais touché** (aucun doublon, hors du périmètre de l'index OOS).
- Modifs de code limitées à `validation/tracking_db.py` (index) et
  `validation/sim_trades.py` (ingestion + rebuild OOS). Ne pas toucher aux modèles ni au pipeline.

## 3. Étape 1 — Nettoyer la table (one-shot)

Sur la base existante, **avant** de poser le nouvel index (SQLite refuse un index UNIQUE si des
doublons subsistent) :

```sql
-- 3.1 supprimer les doublons flaggés
DELETE FROM predictions WHERE source='oos' AND daily_duplicate=1;   -- attendu : 9 875

-- 3.2 CONTRÔLE bloquant : plus aucun groupe métier en double
SELECT COUNT(*) FROM (
    SELECT 1 FROM predictions WHERE source='oos'
    GROUP BY source, model, asset, horizon, cutoff_date
    HAVING COUNT(*) > 1
);   -- DOIT valoir 0, sinon rollback
```

`sim_trades` OOS a déjà été réconcilié (brief précédent : signaux liés aux flaggés supprimés), donc
la suppression de ces lignes `predictions` ne laisse aucun `sim_trade` orphelin. Vérifier quand
même (§6).

## 4. Étape 2 — Poser la contrainte dure (`tracking_db.py`)

Remplacer l'index OOS. Dans `init_db()` :

```sql
DROP INDEX IF EXISTS idx_predictions_oos_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_oos_unique
ON predictions (source, model, asset, horizon, cutoff_date)     -- run_id retiré
WHERE source = 'oos';
```

- Mettre à jour le **commentaire/docstring** de `init_db` et l'entête du module qui expliquent
  l'ancienne clé (l.23-24, l.101-108) : la clé métier OOS est désormais
  `(source, model, asset, horizon, cutoff_date)`, `run_id` n'est plus qu'une métadonnée de
  provenance.
- Idempotent : `DROP … IF EXISTS` puis `CREATE … IF NOT EXISTS` se rejoue sans erreur.

## 5. Étape 3 — Ingestion « garde le dernier run » (`sim_trades.py`)

`insert_oos_predictions` (l.399-421) fait aujourd'hui `INSERT OR IGNORE` → avec le nouvel index,
un ré-run serait **ignoré** (l'ancien resterait), ce qui garderait le *premier* run, pas le
dernier. Passer en **upsert « garde le dernier »** :

```sql
INSERT INTO predictions (run_id, model, asset, horizon, regime, cutoff_date,
                         target_date, last_close, y_pred, y_lower, y_upper, y_true, source)
VALUES (:run_id, :model, :asset, :horizon, :regime, :cutoff_date,
        :target_date, :last_close, :y_pred, :y_lower, :y_upper, :y_true, :source)
ON CONFLICT (source, model, asset, horizon, cutoff_date) WHERE source='oos'
DO UPDATE SET
    run_id      = excluded.run_id,       -- provenance = le dernier run
    target_date = excluded.target_date,
    last_close  = excluded.last_close,
    y_pred      = excluded.y_pred,
    y_lower     = excluded.y_lower,
    y_upper     = excluded.y_upper,
    y_true      = excluded.y_true,
    regime      = excluded.regime;
```

- La cible `ON CONFLICT (...) WHERE source='oos'` doit reprendre **exactement** le prédicat de
  l'index partiel (exigence SQLite).
- `ingest_oos` parcourt `sorted(Path.glob("*-D1"))` : l'ordre lexicographique = ordre
  chronologique (préfixe `YYYYMMDD` du nom de dossier), donc le dernier run traité pour une date
  donnée est bien le plus récent → « garde le dernier » respecté. Le documenter dans la docstring.

## 6. Étape 4 — Cohérence `sim_trades` OOS après ré-ingestion

Comme l'upsert peut changer `run_id`/`y_pred` d'un survivant, et que `sim_trades` est indexé sur
`run_id`, le plus simple et le plus sûr est de **reconstruire les `sim_trades` OOS** après chaque
ingestion (l'OOS est déterministe et entièrement résolu) :

```python
# à la fin de ingest_oos(), après insertion :
#   DELETE FROM sim_trades WHERE source='oos';
#   pour chaque rule_version présente en OOS : generate_sim_trades(source='oos', rule_version=rv)
```

Le **live n'est jamais reconstruit** (`WHERE source='oos'`). La fonction `reconcile_oos_sim_trades`
basée sur le flag devient inutile pour le go-forward (plus de lignes flaggées) : la conserver mais
noter qu'elle est superflue une fois la table propre, ou la remplacer par ce rebuild.

## 7. Sort de la colonne `daily_duplicate`

Une fois les flaggés supprimés, toutes les lignes restantes sont à `0`. La colonne devient
**vestigiale** :
- **Recommandé** : la garder (toujours `0`) pour ne pas casser la vue `all_predictions`
  (`AND daily_duplicate = 0`, devenu un no-op inoffensif) ni les requêtes existantes. Zéro risque.
- Option : la retirer partout (colonne + filtre de la vue + tests) pour une base plus nette — plus
  de churn, à faire seulement si tu veux vraiment nettoyer.

Choisir « garder » sauf demande contraire.

## 8. Contrôles de non-régression

- `DELETE` a retiré **exactement 9 875** lignes ; total `predictions` : 14 174 → 4 299.
- Groupes métier OOS en double : **0** (requête §3.2).
- Le nouvel index existe et **bloque réellement** un doublon : tenter d'insérer deux fois la même
  clé OOS avec des `run_id` différents → une seule ligne, valeurs = celles du dernier insert.
- `sim_trades` live : inchangé. `sim_trades` OOS : cohérent (0 orphelin).
- **Test anti-régression du pipeline** : rejouer `ingest_oos` sur les `Run/*-D1` existants n'ajoute
  aucune ligne `predictions` (upsert idempotent : mêmes données → mêmes valeurs) et ne crée aucun
  doublon.
- `kpi_report(source="oos")` : `n_total` cohérent avec 4 099 survivants (filtré horizon=1).

## 9. Tests

- `test_tracking_db.py` : le nouvel index rejette/écrase un doublon `(source, model, asset,
  horizon, cutoff_date)` ; deux `run_id` différents → 1 seule ligne ; le contenu est celui du
  dernier inséré (keep-latest).
- `test_sim_trades.py` : `insert_oos_predictions` en upsert garde le dernier run ; ré-ingestion
  idempotente ; rebuild OOS ne touche pas le live.

## 10. Ordre d'exécution (script one-shot `validation/correction/enforce_no_duplicates.py`)

1. backup `.bak` ;
2. `DELETE` des flaggés (§3) + contrôle bloquant ;
3. swap de l'index (§4) ;
4. (le code d'ingestion upsert §5 et le rebuild §6 s'appliquent aux runs futurs) ;
5. rebuild `sim_trades` OOS une fois pour repartir propre ;
6. afficher tous les contrôles §8.
