# RUNBOOK — régénération ensemble 5 graines × 200 tirages (NsDiff + TSDiff)

*Pour le tuteur. Compute lourd (entraînements diffusion), volontairement PAS lancé
sur la machine de l'étudiante — cf. `BRIEF_dashboard_multiseed_200.md`.*

> **État au 2026-08-10 — étapes 2, 3 et 5 FAITES pour NsDiff, rien à relancer.**
>
> - Étape 2 : `nsdiff_daily_weekly_multiseed.py` exécuté (8,4 min, 25 checkpoints
>   sous `checkpoints_nsdiff_multiseed_200/`). `nsdiff_daily_weekly_multiseed.json`
>   est désormais à `n_samples: 200`, 5 graines. Le bandeau du dashboard affiche
>   « config de production (tâche 6) », `all_target_config: true`.
> - Étape 3 : les 2 700 lignes `oos` NsDiff/weekly avaient déjà été réécrites en
>   ensemble 5×200 par `repoint_oos_to_ensemble.py` (`run_id`
>   `20260808-oos-repoint-ensemble`) — voie équivalente à
>   `oos_ensemble_nsdiff_daily_weekly.py`, même agrégation par concaténation des
>   nuages. Ne pas rejouer l'étape 3 : elle réécrirait les mêmes lignes.
> - **TSDiff : hors verdict, mais rien n'est supprimé.** Le modèle est en veille
>   (`BRIEF_tsdiff_veille.md`) et retiré du registre (`benchmark_registry.RETIRED`).
>   Il reste dans `MULTISEED_MODELS` — donc ses artefacts sont toujours chargés et
>   ses badges toujours affichés partout où ils existent, notamment le dashboard
>   MENSUEL. Seul `all_target_config` est désormais calculé sur
>   `ACTIVE_MULTISEED_MODELS` (les modèles encore en course), sinon l'artefact
>   weekly jamais généré d'un modèle en veille bloquait le bandeau indéfiniment.
>   Conséquence pratique : les ~1h30-2h de régénération TSDiff annoncées plus bas
>   ne sont plus un prérequis. Les commandes restent ci-dessous si le modèle est
>   réactivé (la mise en veille est réversible par construction).
> - **Hors périmètre, assumé :** les 1 200 lignes `oos` NsDiff à l'horizon MENSUEL
>   (`run_id 20260805-nsdiff-monthly-oos`) restent en mono-graine 42 / n=50. Elles
>   n'apparaissent pas dans le dashboard, borné à `horizon_unit = W+1`. Les aligner
>   demanderait un re-run de `nsdiff_monthly_multiseed.py` à n=200 ET un script de
>   repointage mensuel qui n'existe pas encore.

## Portée

Seuls **NsDiff** et **TSDiff** sont concernés (les 2 seuls modèles échantillonnés du
panel de 7 — cf. l'audit en tête de `NOTE_dashboard_multiseed_200.md`). Les 5 autres
(Naive, ARIMA-GARCH, SARIMA, Prophet, LSTM) sont analytiques/déterministes : rien à
régénérer pour eux.

## Ordre des opérations

Pour **chaque** modèle (NsDiff d'abord — infra déjà en place et éprouvée, puis
TSDiff) :

### 1. Smoke-test (obligatoire avant le run complet)

```bash
# NsDiff (script existant, mécanisme déjà utilisé en prod à n_samples=50)
python experiments/oos_nsdiff_daily_weekly.py --smoke --dry-run

# TSDiff (nouveau script, jamais exécuté — smoke-test avant tout)
python experiments/oos_tsdiff_daily_weekly.py --smoke --dry-run
```

Vérifier que ça tourne sans erreur (1 actif SPY, epochs=2, quelques échantillons) et
que les origines s'alignent (pas de `SystemExit` sur un mismatch de `target_date`).

### 2. Artefact JSON multiseed à 200 (CV-table, source du badge) — n'écrit JAMAIS dans `tracking.db`

```bash
python experiments/nsdiff_daily_weekly_multiseed.py     # -> nsdiff_daily_weekly_multiseed.json (écrasé en place)
python experiments/tsdiff_daily_weekly_multiseed.py     # -> tsdiff_daily_weekly_multiseed.json (nouveau fichier)
```

- Reprenable : checkpoints sous `experiments/checkpoints_{ns,ts}diff_multiseed_200/`
  (un fichier par `(seed, asset)`, ré-exécuter le script reprend là où il s'est arrêté).
- Durée attendue : le run n=50 historique de `nsdiff_daily_weekly_multiseed.py` (5
  graines × 5 actifs × 2 régimes) a pris **~4 min** sur la machine de l'étudiante
  (25 checkpoints, 14:59→15:03 le 2026-08-04) — le fit domine le coût, pas
  l'échantillonnage, donc n=200 ne devrait pas être beaucoup plus long. Le run TSDiff
  n90/n_samples=50 (2 régimes × 5 actifs, 1 graine) a pris ~948s (~16 min) — × 5
  graines ⇒ prévoir **1h30-2h** pour TSDiff, sensiblement plus long que NsDiff (epochs
  par actif plus élevés, jusqu'à 80).

### 3. Ligne `oos` ENSEMBLE (concaténation 5×200 → 1000, upsert 1 ligne/origine)

**Impératif : lancer l'étape 2 du même modèle en premier** (ou laisser ce script la
déclencher lui-même — il réutilise `load_price_data`/`run_seed`, donc régénère les
checkpoints manquants) puisque l'ensemble writer réutilise les *mêmes* checkpoints
sans refit.

```bash
# Vérifier d'abord SANS écrire (vérifie l'alignement 1:1 avec les origines existantes) :
python experiments/oos_ensemble_nsdiff_daily_weekly.py --dry-run
python experiments/oos_ensemble_tsdiff_daily_weekly.py --dry-run

# Puis pour de vrai (upsert + reset + backfill_eval_metrics.py automatique) :
python experiments/oos_ensemble_nsdiff_daily_weekly.py
python experiments/oos_ensemble_tsdiff_daily_weekly.py
```

Chaque script :
1. vérifie que les origines construites tombent EXACTEMENT sur les origines déjà en
   base (`model='{NsDiff,TSDiff}' AND horizon_type='weekly' AND source='oos'`) —
   abandonne avec une erreur claire sinon (jamais de régénération "à peu près") ;
2. upsert via `insert_oos_predictions` (remplace la ligne single-seed existante,
   n'en ajoute pas) ;
3. remet à NULL `abs_error`/`in_interval`/etc. sur exactement ce périmètre (l'upsert
   ne les touche pas, ils resteraient périmés sinon — piège documenté dans le script) ;
4. relance `backfill_eval_metrics.py` pour les recalculer depuis les nouvelles bornes.

### 4. Contrôle de comptage (non-négociable, cf. brief §7)

```bash
sqlite3 validation/tracking.db "SELECT source, COUNT(*) FROM predictions GROUP BY source;"
```

- `oos` : nombre de lignes **identique** à avant (on remplace, jamais on n'ajoute).
- `live` / `backtest_rolling_*` : **inchangés**.
- Comparer au comptage AVANT déjà consigné dans `NOTE_dashboard_multiseed_200.md`.

### 5. Régénérer le dashboard (code déjà en place, aucune modif nécessaire)

```bash
python experiments/dashboard_d7_w1.py
```

Le label de config et le badge de robustesse se mettent à jour automatiquement : dès
que `nsdiff_daily_weekly_multiseed.json`/`tsdiff_daily_weekly_multiseed.json`
affichent `n_samples: 200`, le bandeau de config du dashboard passe de « single-seed
/ 50 tirages, en attente » à l'état ensemble — **aucun changement de code requis**,
c'est lu dynamiquement depuis les artefacts.

### 6. pytest (vert avant ET après, cf. brief)

```bash
~/.venvs/DeepEdgeBenchmark/bin/python -m pytest experiments/ validation/ models/ -q
```

## Pièges déjà rencontrés (évités dans les scripts fournis, ne pas les réintroduire)

- **Concaténer, pas moyenner** les 5 nuages avant de lire les quantiles (fait dans
  `build_ensemble_rows` des deux scripts d'ensemble).
- **Une seule ligne par origine** dans `oos` — jamais 5 (une par graine).
- **Origines lues verbatim** depuis la DB (`load_baseline_triplets*`), jamais
  re-dérivées d'un `three_way_split` sur des données re-téléchargées (risque de
  dérive yfinance — documenté dans `oos_tsdiff_daily_weekly.py`, c'est pour ça qu'un
  nouveau script a été écrit plutôt que de relancer `weekly_headtohead_v2.py`).
- **`abs_error` périmé** après l'upsert (l'`ON CONFLICT DO UPDATE` de
  `insert_oos_predictions` ne le touche pas) — géré par le reset explicite avant
  `backfill_eval_metrics.py`.
