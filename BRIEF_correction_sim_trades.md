# BRIEF — Correction : propager `daily_duplicate=0` à `sim_trades.py`

## 0. Contexte

Le flag `daily_duplicate` a été posé dans `predictions` (brief précédent : 9 875 lignes OOS
marquées, base intacte). **Mais la simulation de trades ne l'utilise pas encore** : toutes les
lectures OOS de `sim_trades.py` passent par la vue `all_predictions`, qui filtre `horizon = 1`
mais **pas** `daily_duplicate = 0`. Conséquence :

- les **KPIs OOS** (`kpi_report`, `naive_*_report`, `daily_detail`, balayage `k`) comptent encore
  chaque prédiction rejouée plusieurs fois → `n_total`, taux et ROI faussés ;
- la table **`sim_trades`** contient déjà des signaux OOS générés à partir des doublons (un
  `run_id` distinct par rejeu → un `sim_trade` distinct), donc les signaux comptés sont eux aussi
  gonflés.

Objectif : ne compter chaque prédiction OOS **qu'une fois** (le survivant, `daily_duplicate=0`),
sans toucher au live (0 doublon) ni aux règles de trading.

## 1. Deux volets

1. **La vue** `all_predictions` : ajouter `AND daily_duplicate = 0`. Corrige d'un coup toutes les
   lectures « log » (n_total, générations futures, rapports naïfs, daily_detail, k-sweep).
2. **La table** `sim_trades` : réconcilier les signaux OOS déjà calculés pour qu'ils ne
   reflètent plus que les survivants.

## 2. Garde-fous

- Branche dédiée : `maeva/correction-sim-trades` (sans accent).
- **Sauvegarde** `validation/tracking.db.bak_YYYYMMDD` avant toute écriture (ignorée par git).
- Écritures en **transaction**, rollback si un contrôle échoue.
- **Ne pas modifier** les règles (`bull_calm_d1`, `sideways_d1`, etc.), ni `tracking_db.py`, ni
  `pipeline.py`/`models/*`. On ne touche qu'à la vue et à la table `sim_trades` (OOS uniquement).
- Le **live n'est jamais touché** : il n'a aucun doublon et toutes ses lignes sont déjà à
  `daily_duplicate = 0`.

## 3. Volet 1 — La vue `all_predictions`

Dans `validation/sim_trades.py`, `init_db()` (bloc `CREATE VIEW all_predictions`, ~ligne 143),
ajouter le filtre :

```sql
CREATE VIEW all_predictions AS
SELECT
    id, run_id, model, asset, horizon, regime,
    cutoff_date AS d_date, target_date,
    last_close  AS reference_price, y_pred AS predicted,
    y_lower     AS pi_lower,        y_upper AS pi_upper,
    y_true      AS realized_price,  source
FROM predictions
WHERE horizon = 1
  AND daily_duplicate = 0        -- << AJOUT : ignorer les doublons OOS flaggés
```

La vue étant recréée à chaque `init_db()` (`DROP VIEW` + `CREATE`), aucune migration : le
prochain appel suffit. Le live reste inclus (ses lignes sont toutes à 0).

> Vérifier au passage qu'aucune lecture OOS ne court-circuite la vue en lisant `predictions`
> en direct. (Revue faite côté brief : dans `sim_trades.py`, toutes les lectures de stats passent
> par `all_predictions` ; seules les écritures d'ingestion touchent `predictions` directement.)

## 4. Volet 2 — Réconcilier `sim_trades` (OOS)

`sim_trades` est une table **de résultats calculés**, entièrement redérivable depuis
`predictions` + les règles. Chaque doublon flaggé a un `run_id` propre et a pu produire son
propre `sim_trade`. Il faut retirer les `sim_trades` OOS issus de prédictions flaggées.

**Règle :** un `sim_trade` OOS ne doit subsister que si sa prédiction source survit
(`daily_duplicate = 0`). Le rapprochement se fait sur
`(source, run_id, model, asset, horizon, d_date = cutoff_date)`.

Deux façons de faire — **choisir la 4.A** (chirurgicale, préserve les survivants existants) :

### 4.A — Suppression ciblée (recommandé)

```sql
DELETE FROM sim_trades
WHERE source = 'oos'
  AND id IN (
      SELECT s.id
      FROM sim_trades s
      JOIN predictions p
        ON p.source = 'oos' AND p.run_id = s.run_id AND p.model = s.model
       AND p.asset = s.asset AND p.horizon = s.horizon AND p.cutoff_date = s.d_date
      WHERE p.daily_duplicate = 1
  );
```

Puis re-générer (idempotent, ne recrée rien s'il ne manque rien) pour garantir que chaque
survivant a bien son signal, sur **chaque `rule_version` réellement présente en OOS** :

```python
for rv in rule_versions_oos:          # distinct rule_version des sim_trades OOS existants
    generate_sim_trades(db_path=..., rule_version=rv, source="oos")
```

### 4.B — Reconstruction complète OOS (alternative, plus simple à raisonner)

`DELETE FROM sim_trades WHERE source='oos';` puis re-générer toutes les `rule_versions` OOS via
la vue (désormais filtrée). Correct car l'OOS est déterministe et entièrement résolu, mais perd
les `created_at` d'origine — sans importance pour un backtest. Ne pas utiliser si on veut
conserver l'historique exact des lignes survivantes.

> Dans les deux cas, **le live n'est jamais supprimé ni régénéré** (`WHERE source='oos'`).

## 5. Script d'exécution

`validation/correction/apply_sim_trades_dedup.py` :
1. sauvegarde `.bak` ;
2. `init_db()` (recrée la vue filtrée) ;
3. relève les `rule_version` distinctes en OOS ;
4. applique la suppression ciblée (4.A) puis re-génère ;
5. affiche les contrôles (§6).

## 6. Contrôles de non-régression

- **Live intact** : nombre de `sim_trades` `source='live'` inchangé avant/après ; aucune ligne
  live supprimée.
- **Plus aucun signal OOS issu d'un doublon** :
  ```sql
  SELECT COUNT(*) FROM sim_trades s
  JOIN predictions p
    ON p.source='oos' AND p.run_id=s.run_id AND p.model=s.model AND p.asset=s.asset
   AND p.horizon=s.horizon AND p.cutoff_date=s.d_date
  WHERE s.source='oos' AND p.daily_duplicate=1;   -- doit renvoyer 0
  ```
- **`n_total` OOS des KPIs a baissé** : comparer `kpi_report(source="oos", group_by=())` avant/
  après — le `n_total` global doit passer du volume gonflé au nombre de prédictions OOS uniques
  (cohérent avec les ~4 099 survivants OOS du rapport d'audit, filtré horizon=1).
- **Un seul signal par prédiction survivante** : plus aucun couple de `sim_trades` OOS ne partage
  la même clé métier `(model, asset, horizon, d_date)` via des `run_id` différents.
- **Idempotence** : relancer le script ne change aucun compteur.

## 7. Tests (`validation/test_sim_trades.py`)

- la vue `all_predictions` n'expose pas les lignes OOS `daily_duplicate=1` (et expose bien le
  survivant) ;
- après réconciliation, `kpi_report(source="oos")` ne compte que les survivants (monter un jeu
  avec 1 survivant + 2 doublons → `n_total` = 1 pour ce combo) ;
- une ligne `live` reste toujours visible et son `sim_trade` intact ;
- idempotence de `apply_sim_trades_dedup`.

## 8. Hors périmètre

- Le correctif **Prophet** (walk-forward) — brief suivant.
- Le retrait de `run_id` de l'index d'unicité OOS (prévention native des doublons) — à décider
  plus tard ; tant qu'on garde l'approche par flag, ce n'est pas requis.
