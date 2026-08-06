# BRIEF — Passage du dashboard D7/W1 de single-seed/50 tirages à ensemble multi-graines/200 tirages

*Créé le 2026-08-06. Suite de `BRIEF_nsdiff_dashboard_daily_oos.md` (NsDiff sur le dashboard) et de
`NOTE_nsdiff_consolidation_daily_vs_weekly.md` §8 (« Non fait, volontairement : régénérer TOUS les
modèles au même budget »). Objet : rendre le dashboard `oos` représentatif de la config de production
— l'**ensemble 5 graines × 200 tirages** (tâche 6, meilleur rapport qualité/prix) — au lieu du
single-seed 42 / 50 tirages actuel, non représentatif.*

> **Contexte d'exécution.** Le compute lourd (régénération des modèles) tourne **en local** sur la
> machine du tuteur, pas dans le cloud. Ce brief est écrit pour **Claude Code** : il faut lancer les
> runs, écrire dans `tracking.db`, régénérer le HTML et faire tourner `pytest`. Décisions de design
> déjà actées avec l'étudiante (voir §1) — ne pas les rediscuter, les exécuter.

---

## 0. Objectif (une phrase)

Faire que `experiments/dashboard_d7_w1.py` affiche, en `source='oos'`, les métriques de l'**ensemble
5 graines (42–46) × 200 tirages** de chaque modèle échantillonné (au lieu de single-seed 42 / 50
tirages), avec un **badge de robustesse inter-graines** par cellule, sans casser le pipeline ni le
schéma, et à origines strictement identiques à l'existant.

---

## 1. Décisions de design déjà actées (ne pas rediscuter)

1. **Le dashboard montre l'ENSEMBLE**, pas les graines séparées : pour chaque origine, on concatène les
   5 nuages de 200 tirages en un nuage de 1000 et on lit point + bandes dessus (tâche 6). → **une seule
   ligne `oos` par origine**, comme aujourd'hui.
2. **Badge de robustesse** en plus : chaque cellule (modèle × actif) affiche la stabilité inter-graines
   du verdict et le CV(Winkler), **lus depuis les artefacts JSON multiseed** (jamais depuis `oos`).
3. **Zéro refonte de schéma** : pas de colonne `seed` ni `n_samples` ajoutée à `predictions`. La
   provenance (config = « ensemble 42–46 × 200 ») est portée par un label de config + le `run_id`, pas
   par le schéma.

Ce qui est explicitement **écarté** : stocker 5 lignes par origine (une par graine) dans `oos` — ça
casserait l'appariement daily/weekly et le pooling (double comptage), et ce n'est pas la config de
production. Le niveau « poolé multi-graines / expected-run » de la note reste dans les artefacts
isolés, pas sur le dashboard.

---

## 2. Ce qui est déjà vrai (vérifié dans le code, ne pas re-supposer)

- **Le dashboard n'a AUCUNE hypothèse single-seed/50.** Le `seed` qu'il manipule (`--seed 42`,
  `CELL_TEST_SEED=0`) est celui du **bootstrap de test**, pas de la génération. Il lit `y_pred`,
  `y_lower`, `y_upper` depuis `predictions WHERE source='oos'` et les teste. → **rien à changer dans sa
  logique de test** ; il « marche » sur de meilleures données.
- **Table `predictions`** : stocke **point + quantiles** (`y_pred`, `y_lower`, `y_upper`), pas le nuage.
  Colonnes clés : `model, asset, cutoff_date, target_date, frequence, horizon_type, horizon_unit,
  y_pred, y_lower, y_upper, y_true, source`. **Pas de colonne `seed` ni `n_samples`.** `source` ∈
  {`oos` (27 524 lignes), `live`, `backtest_rolling_nsdiff`}.
- **Insertion `oos`** : `validation/sim_trades.py:insert_oos_predictions` — upsert idempotent
  `ON CONFLICT (source, model, asset, horizon, frequence, horizon_type, cutoff_date) WHERE
  source='oos'`. **La clé ne contient pas la graine** → une ligne « ensemble » **écrase proprement** la
  ligne single-seed existante pour la même origine. C'est ce qui rend la migration sans risque de
  schéma.
- **Convention du point** (à conserver telle quelle sur l'ensemble) : `point = float(np.mean(s))` sur
  le nuage ; `lo, hi = np.quantile(s, [0.025, 0.975])`. Réf. `oos_nsdiff_daily_weekly.py:287,304`.
- **Appariement / tests importés** de `matrice_paired_tests.py`
  (`build_daily_weekly_pairs`, `comparison_3_daily_vs_weekly`) et bootstrap de
  `paired_test.py` — jamais réimplémentés. Ne pas y toucher.
- **Artefacts multiseed** existants (ex. `nsdiff_daily_weekly_multiseed.json`) : structure
  `cv_table[asset] = {verdicts_by_seed, p_values_by_seed, verdict_stable, effective_n,
  cv_rmse_daily/weekly, cv_winkler_daily/weekly, *_mean}`. **Actuellement à n_samples=50** → à
  régénérer à 200 (voir §4), sinon le badge afficherait un CV périmé.

---

## 3. Étape 0 — Audit : quels modèles sont échantillonnés ? (obligatoire, ne pas supposer)

Tous les modèles ne sont pas concernés par le budget de tirages. Avant tout run :

- **Grep + lecture** des scripts de génération `oos` de chaque modèle (`weekly_multimodel.py`,
  `weekly_headtohead*.py`, `oos_nsdiff_*.py`, et le producteur des baselines) pour classer chaque
  modèle des 7 (`Naive, ARIMA-GARCH, SARIMA, Prophet, LSTM, TSDiff, NsDiff`) en :
  - **échantillonné** (ses bandes viennent d'un nuage fini de `n_samples` tirages, donc affecté par le
    biais du quantile à 50 et par la graine) → **à régénérer en ensemble 5×200** ;
  - **analytique / déterministe** (bandes fermées ou bootstrap déterministe, pas de graine, pas de
    `n_samples`) → **inchangé**, mais **le déclarer** dans la note de sortie (ne pas le régénérer « pour
    faire pareil » : ça ne veut rien dire pour lui).
- Écrire ce classement en tête de la note de résultats. **Le non-négociable de comparabilité
  (`NOTE_nsdiff_consolidation…` §8) porte sur les modèles échantillonnés** : ceux-là doivent tous
  passer au même budget 5×200, sinon leurs chiffres ne sont plus comparables entre eux.

---

## 4. Le contrat de données — ce que la régénération doit produire

Pour **chaque modèle échantillonné**, **chaque actif**, **chaque origine** (`cutoff_date`,
`target_date`), **chaque horizon** (W+1 au moins ; idéalement W+1/W+2/W+3), **chaque régime**
(B = daily, C = weekly) :

1. Générer **5 nuages de 200 tirages** (graines 42, 43, 44, 45, 46), en **conservant les tirages
   bruts** (`collect_samples=True` / équivalent — pas seulement les quantiles).
2. **Concaténer** les 5 × 200 = **1000 tirages** en un seul nuage (mélange des 5 prédictives).
   ⚠️ **Concaténer, PAS moyenner les bornes** (moyenner lisserait ; concaténer élargit quand les graines
   divergent — c'est l'effet recherché de la tâche 6).
3. Calculer sur les 1000 tirages : `y_pred = np.mean(cloud)`, `y_lower, y_upper =
   np.quantile(cloud, [0.025, 0.975])` — **même convention qu'aujourd'hui**, juste sur 1000 au lieu de
   50 tirages.
4. **Upsert** cette **unique** ligne dans `predictions` via `insert_oos_predictions`
   (`source='oos'`), sur **l'origine exacte** de la ligne existante → elle remplace la ligne
   single-seed/50. Utiliser un `run_id` explicite (ex. `oos_ensemble_5s200_<horodatage>`) pour la
   traçabilité.
5. En parallèle (jamais dans `oos`), produire/rafraîchir l'**artefact JSON multiseed** du modèle
   (même format que `nsdiff_daily_weekly_multiseed.json`, mais à **n_samples=200**) : `cv_table` par
   actif avec `verdict_stable`, `p_values_by_seed`, `cv_winkler_daily/weekly`. C'est la source du badge
   (§5). Réutiliser `nsdiff_daily_weekly_multiseed.py` comme patron (il boucle déjà les graines, garde
   des checkpoints reprenables, et n'écrit jamais dans la DB).

**Réutiliser, ne pas réimplémenter** : `generate_nsdiff_asset` (avec `n_samples=200` +
`collect_samples`), la logique d'ensemble de la tâche 6 (`nsdiff_seed_ensemble.py` si présent, sinon la
concaténation ci-dessus), `insert_oos_predictions`, le backfill des métriques dérivées
(`backfill_eval_metrics.py` sur `WHERE abs_error IS NULL`). Origines **lues verbatim**
(`load_baseline_triplets` / la DB), jamais régénérées « à peu près ».

---

## 5. Réusinage du dashboard (`experiments/dashboard_d7_w1.py`)

Changements **additifs**, la logique de test inchangée :

1. **Label de config** (traçabilité honnête). Aujourd'hui le dashboard n'affiche ni « 50/single » ni
   « 200/ensemble ». Ajouter un libellé de config visible (en-tête + pied de page + `payload` JSON),
   ex. « **Données : ensemble 5 graines (42–46) × 200 tirages** — config de production (tâche 6) ».
   Le porter par une constante/flag, pas par le schéma. Retirer toute formulation laissant croire à du
   single-seed/50.
2. **Badge de robustesse par cellule.** Charger, en entrée optionnelle, les artefacts JSON multiseed
   par modèle (`--multiseed-json model=path ...` ou un dossier conventionné). Pour chaque cellule
   (modèle × actif), afficher :
   - **stabilité du verdict** : `verdict_stable` sur les 5 graines (ex. badge « 5/5 stable » vs
     « instable — verdict change selon la graine ») ;
   - **CV(Winkler)** daily & weekly (ex. « CV 4 % / 11 % »).
   Dégradation gracieuse : si l'artefact d'un modèle manque, pas de badge pour ce modèle (jamais
   d'erreur bloquante).
3. **Traçabilité** dans le JSON de sortie : ajouter `data_config: "ensemble_5seed_200"`, la liste des
   artefacts multiseed consommés, et leurs `effective_n`. Garder `seed_pooled` / `seed_cell_tests`
   (seeds de test, distincts — le pied de page doit continuer à expliquer la distinction).
4. **Ne pas** modifier `winkler_score`, `build_daily_weekly_pairs`, `run_pooled_test`,
   `comparison_3_daily_vs_weekly`, `paired_test.py`. Le badge lit des JSON déjà calculés ; il n'ajoute
   aucun test.

---

## 6. Non-négociables

- [ ] **Comparabilité** : tous les modèles échantillonnés régénérés au **même budget 5×200**. Un
      modèle laissé à 50 casse la comparabilité — soit tous, soit c'est déclaré comme dette.
- [ ] **Origines identiques** aux lignes existantes (bloquant pour l'upsert et l'appariement) : vérifier
      après coup que chaque ligne ensemble tombe sur un `(model, asset, frequence, horizon_type,
      horizon_unit, cutoff_date)` déjà présent.
- [ ] **`oos` = ensemble uniquement** (1 ligne/origine). Les runs **par graine** restent dans des
      artefacts JSON isolés — **jamais** écrits dans `predictions`.
- [ ] **Lignes `live` et `backtest_rolling_*` intactes** : comptage de contrôle avant/après par
      `source`.
- [ ] **Point-in-time préservé** : origines inchangées, `mu`/`sd` sur le train seul (rien à retoucher
      si on réutilise les producteurs existants).
- [ ] **Aucun fichier source existant cassé** ; changements dashboard **additifs** ; **`pytest` vert
      AVANT et APRÈS** (`experiments/`, `validation/`, `models/`).
- [ ] **Convention du point conservée** : `np.mean` du nuage de 1000, quantiles 2,5/97,5 — pas la
      médiane.

---

## 7. Vérification finale (le test « c'est fini »)

- [ ] `select source, count(*) from predictions group by source` : `oos` inchangé en **nombre de
      lignes** (on remplace, on n'ajoute pas), `live` / `backtest_rolling_*` **identiques** au comptage
      d'avant.
- [ ] Contrôle de cohérence des chiffres : sur au moins une cellule connue, la Cov95 ensemble doit
      **monter** vs l'ancienne valeur à 50 tirages (le biais du quantile se comble), et recouper les
      notes (`NOTE_duel_nsdiff_vs_tsdiff_budget_egal.md` §1, `NOTE_nsdiff_consolidation…` tâche 6) à
      l'échantillonnage près.
- [ ] `python experiments/dashboard_d7_w1.py` régénère le HTML **sans erreur**, en-tête affichant
      « ensemble 5 graines × 200 », **badge de robustesse présent** sur les cellules des modèles dont
      l'artefact multiseed existe.
- [ ] Page ouvrable en `file://` **sans gel**.
- [ ] `pytest` : vert avant ET après ; nombre de tests ≥ l'existant (aucune régression). Déclarer tout
      skip pré-existant (ex. `properscoring not installed`).

---

## 8. Livrables

1. Le(s) script(s) de régénération ensemble (réutilisant l'existant), paramétrables (modèles, actifs,
   origines, graines, `n_samples`), + les artefacts JSON multiseed par modèle à 200 tirages.
2. Lignes `oos` de `predictions` mises à jour (ensemble 5×200) pour tous les modèles échantillonnés, sur
   les origines existantes.
3. `dashboard_d7_w1.py` réusiné (label de config + badge robustesse + traçabilité JSON) et
   `dashboard_d7_w1.html` régénéré.
4. Courte note `NOTE_dashboard_multiseed_200.md` : audit échantillonné/analytique (§3), modèles
   régénérés, `run_id`, contrôle de comptage `source` avant/après, recoupement avec les notes du duel,
   dettes déclarées (modèles analytiques laissés tels quels, horizons non couverts le cas échéant).

---

## 9. Pièges à éviter

- **Ne pas** moyenner les bornes des 5 graines — **concaténer** les nuages (1000 tirages) avant de lire
  les quantiles.
- **Ne pas** écrire les lignes par graine dans `oos` — seulement l'ensemble (1 ligne/origine). Sinon
  `build_daily_weekly_pairs` double-compte et le pooling est faussé.
- **Ne pas** régénérer d'origines « à peu près » — lecture verbatim, sinon l'upsert crée des doublons au
  lieu de remplacer, et l'appariement daily/weekly casse.
- **Ne pas** prendre la médiane : le point est la **moyenne** du nuage (convention existante).
- **Ne pas** régénérer un modèle analytique « pour faire pareil » : sans nuage fini, le budget de
  tirages ne s'applique pas — le classer à l'étape 0 et le laisser.
- **Ne pas** oublier de rafraîchir les artefacts multiseed à **200** : les JSON actuels sont à 50, le
  badge afficherait un CV périmé.
- **Ne pas** toucher aux fonctions de test importées (`winkler_score`, `build_daily_weekly_pairs`,
  `run_pooled_test`, `comparison_3_daily_vs_weekly`, `paired_test.py`).
