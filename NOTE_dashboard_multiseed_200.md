# NOTE — Dashboard D7/W1 : préparation ensemble 5×200 + badge de robustesse

*Suite de `BRIEF_dashboard_multiseed_200.md`. Priorité corrigée en cours de route
(cf. échange) : le compute lourd (entraînements diffusion) tourne chez le tuteur, pas
sur cette machine. Ce tour ne contient donc **aucun entraînement exécuté** — seulement
l'audit, les scripts de régénération (prêts, non lancés) et les changements dashboard
(déployés, visibles dès maintenant sur les données actuelles).*

## 1. Audit §3 — échantillonné vs analytique (grep + lecture, rien supposé)

Sources lues : `benchmarks/multi_horizon.py` (les 7 `forecast_horizons_*`),
`models/{tsdiff,nsdiff,prophet,lstm}_model.py`.

| Modèle | Bandes | Verdict |
|---|---|---|
| **TSDiff** | `np.mean`/`np.quantile([.025,.975])` sur un nuage de `N_SAMPLES=50` tirages, `seed` + `n_samples` contrôlés par notre code (`set_seed`, `forecast_horizons_tsdiff`) | **Échantillonné → à régénérer** |
| **NsDiff** | idem, ligne à ligne (`forecast_horizons_nsdiff`) | **Échantillonné → à régénérer** |
| Naive | `last_price ± Z95·σ·√h`, formule fermée, aucune graine | Analytique |
| ARIMA-GARCH | quantiles de la loi GARCH ajustée (skew-t/normal), aucun tirage, aucune graine | Analytique |
| SARIMA | `conf_int()` gaussien fermé de `statsmodels`, aucune graine | Analytique |
| LSTM | point = rollout déterministe (seed d'ENTRAÎNEMENT fixe), bande = `point ± 1.96·std·√h` (formule fermée, pas un nuage relu en quantiles) | Analytique (malgré la graine) |
| Prophet | Facebook Prophet simule en interne (Monte-Carlo trend/bruit) mais `fit_prophet`/`forecast_horizons_prophet` n'exposent ni `seed` ni `n_samples` — rien à régénérer avec le pipeline 5×200 tel quel | **Dette déclarée, non régénéré** (voir §4) |

Seuls **NsDiff** et **TSDiff** sont concernés par le budget 5×200.

## 2. Scripts de régénération — prêts, **non exécutés**

| Script | Rôle | Statut |
|---|---|---|
| `oos_nsdiff_daily_weekly.py` | `generate_nsdiff_asset(..., collect_samples=True)` (ajout additif) | Modifié, testé (imports + pytest -k nsdiff) |
| `nsdiff_daily_weekly_multiseed.py` | CV-table JSON à `n_samples=200`, nouveau dossier de checkpoints (`checkpoints_nsdiff_multiseed_200/`), **n'écrit jamais `oos`** | Modifié, non exécuté |
| `oos_ensemble_nsdiff_daily_weekly.py` | Concatène 5×200→1000, upsert `oos` (1 ligne/origine), vérifie les origines 1:1, reset+`backfill_eval_metrics.py` | Nouveau, non exécuté |
| `oos_tsdiff_daily_weekly.py` | Mirroir NsDiff pour TSDiff (`generate_tsdiff_asset`, origines lues verbatim) | Nouveau, non exécuté |
| `tsdiff_daily_weekly_multiseed.py` | CV-table JSON TSDiff à `n_samples=200` | Nouveau, non exécuté |
| `oos_ensemble_tsdiff_daily_weekly.py` | Ensemble writer TSDiff (mirroir NsDiff) | Nouveau, non exécuté |

Tous importés et smoke-testés (`python -c "import ..."`) — zéro erreur de syntaxe/nom,
mais **aucun run réel** (aucun fit, aucun checkpoint, aucune ligne `oos` réécrite).
Procédure complète pour le tuteur : `RUNBOOK_regeneration_multiseed_200.md`.

## 3. Changements dashboard (déployés, effectifs dès maintenant)

`experiments/dashboard_d7_w1.py` + `dashboard_d7_w1_template.py` :

- **Bandeau de config honnête**, calculé depuis les artefacts JSON multiseed réels
  (jamais un texte figé). Il affichait *« Données actuelles (NsDiff : 5 graine(s) x
  50 tirages ; TSDiff : artefact absent) — cible : ensemble 5 graines (42-46) x 200
  tirages, régénération en attente côté tuteur »*. **Depuis le 2026-08-10** il
  affiche *« Données (modèles échantillonnés NsDiff) (TSDiff en veille, hors
  verdict) : ensemble 5 graines (42-46) x 200 tirages — config de production
  (tâche 6) »* (`all_target_config: true`) : le JSON NsDiff a été régénéré à
  `n_samples=200`, et TSDiff — en veille, cf. `BRIEF_tsdiff_veille.md` — ne
  conditionne plus le verdict (`all_target_config` est calculé sur
  `ACTIVE_MULTISEED_MODELS`). Il reste dans `MULTISEED_MODELS` : ses artefacts sont
  toujours chargés et ses badges toujours affichés là où ils existent, notamment le
  dashboard mensuel. Le mécanisme reste dynamique — aucun texte figé n'a été
  introduit.
- **Badge de robustesse par cellule** (colonne « Robustesse inter-graines », visible
  par défaut) : lu depuis `{model}_daily_weekly_multiseed.json` → `cv_table[asset]`
  (`verdict_stable`, `cv_winkler_daily/weekly`). Dégradation gracieuse vérifiée : les
  35 cellules non-NsDiff affichent `—`, les 5 cellules NsDiff affichent leur badge réel
  (ex. `4/5 instable · CV(W) 11.8%/2.9%` sur BTC-USD, `5/5 stable · CV(W) 11.6%/5.1%`
  sur ETH-USD) — capture d'écran vérifiée, aucune erreur console.
- **Traçabilité JSON** : `payload["data_config"]` (cible, artefacts consommés + leur
  `n_samples`/`effective_n`/chemin, modèles analytiques déclarés, note sur
  l'indépendance badge/lignes `oos`).
- **Non touché** (vérifié par relecture) : `winkler_score`, `build_daily_weekly_pairs`,
  `run_pooled_test`, `comparison_3_daily_vs_weekly`, `paired_test.py`.

## 4. Contrôle `predictions` — inchangé (aucune écriture cette session)

```
source                    | AVANT   | APRÈS
backtest_rolling_nsdiff   | 1080    | 1080
live                      | 1510    | 1510
oos                       | 27524   | 27524
```

`tracking.db` vérifié **binaire identique** (`diff` contre la sauvegarde faite en
tout début de session) — le dashboard est en lecture seule, aucun script de
régénération n'a été lancé.

## 5. pytest

Avant et après ce lot de changements (dashboard + scripts prêts) :
`experiments/` + `validation/` + `models/` → **421 passed** dans les deux cas, zéro
régression.

## 6. Dettes déclarées

- **Prophet** : non concerné par le pipeline 5×200 (bandes issues d'une simulation
  interne à la librairie, non paramétrable ici) — cf. §1.
- **TSDiff** régénéré via un **nouveau** mirroir (`oos_tsdiff_daily_weekly.py`) plutôt
  qu'en rejouant `weekly_headtohead_v2.py` (comment les lignes `oos` actuelles ont
  historiquement été produites) : `run_pair_v2` re-dérive ses origines d'un
  `three_way_split` sur des données re-téléchargées, ce qui risque de diverger des
  origines déjà en base (dérive yfinance, déjà documentée pour TLT dans
  `oos_nsdiff_daily_weekly.py`). Le nouveau script lit les origines verbatim depuis la
  DB (même mécanisme que NsDiff) — **non exécuté, donc non recoupé empiriquement avec
  les origines existantes** ; le premier `--dry-run` du tuteur (§4 du RUNBOOK) fera
  cette vérification avant tout upsert.
- **Recoupement avec les notes du duel** (`NOTE_duel_nsdiff_vs_tsdiff_budget_egal.md`,
  `NOTE_nsdiff_consolidation…` tâche 6) : différé — n'a de sens qu'une fois les lignes
  `oos` ensemble réellement écrites (actuellement toujours single-seed/50).
