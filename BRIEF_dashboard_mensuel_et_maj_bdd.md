# BRIEF — Dashboard externe : brancher le mensuel + garantir la fraîcheur base

*Cible : `model_artifacts/generate_dashboard.py` (dashboard externe de Kyrio, déployé sur GitHub Pages
par `.github/workflows/deploy-pages.yml`). Deux objectifs indépendants, livrables séparément :
**A —** faire apparaître les données mensuelles (aujourd'hui simple emplacement réservé) ;
**B —** rendre visible/vérifiable que le dashboard est à jour vis-à-vis de `validation/tracking.db`.*

---

## 0. Objectif (deux phrases)

Le dashboard externe affiche le **daily** (depuis `Run/**`) et le **weekly** (depuis `tracking.db`), mais le
**monthly** n'est qu'un bouton grisé (`SIM_TRADES_PIPELINES = [..., "monthly"]`) sans aucun collecteur — alors
que la base contient déjà **2 400 lignes mensuelles** (`horizon_type='monthly'`, M+1/2/3, NsDiff & TSDiff,
5 actifs, 40 origines par actif, 2023-01-31 → 2026-04-30). On **branche le mensuel en calquant strictement le
weekly**, et on **stampe la provenance** (dernière origine + `run_id`) par piste pour que « à jour sur la base »
soit un fait vérifiable, pas une supposition.

---

## 1. État actuel (vérifié, pas supposé)

- **Daily** : `collect_run_data()` lit `Run/<date>-<modèle>-<actif>-<horizon>/` (D+1, D+7). Hors périmètre de ce brief.
- **Weekly** : `collect_weekly_kpis()` (~ligne 297) lit
  `frequence='weekly' AND horizon_type='weekly' AND source IN ('oos','live')`, restreint à
  `WEEKLY_KPI_MODELS = ["TSDiff","NsDiff"]`. Rendu dans le payload + badge de robustesse via
  `collect_weekly_multiseed()` (réutilise `experiments/multiseed_badge_common.py`, **stdlib seule**).
- **Monthly** : **rien**. `SIM_TRADES_PIPELINES` (ligne 65) et le label JS `pipelineLabel` (~ligne 1599) listent
  `"monthly"`, mais aucun `tc_id` ni collecteur n'y est rattaché → bouton grisé, aucune donnée lue.
- **Données mensuelles en base** (`source='oos'`) : deux régimes présents —
  **régime C** (`frequence='monthly'`, mensuel natif) et **régime B** (`frequence='daily'`, daily poussé à la
  fin de mois), horizons M+1/M+2/M+3, modèles **NsDiff & TSDiff** uniquement, 40 origines/actif.

---

## 2. Non-négociables (garde-fous)

- **`tracking.db` en lecture seule.** Le dashboard est une couche d'affichage — aucune écriture, aucun recalcul
  de prédictions.
- **Aucun fichier source modifié hors ajouts.** On **ajoute** `collect_monthly_kpis()` en **miroir** de
  `collect_weekly_kpis()` (comme `oos_nsdiff_daily_weekly.load_baseline_triplets_daily` a mirroré son homologue
  plutôt que d'éditer le partagé). On ne touche pas à la logique weekly/daily existante.
- **Environnement CI minimal.** Le job Pages n'installe que `pandas pyarrow numpy scipy`. Le collecteur mensuel
  **ne doit importer aucune dépendance lourde** (pas de `arima_model`, `statsmodels`, `arch`, `yfinance`) —
  mêmes règles que le weekly. Réutiliser `prob_kpi_common` (déjà importé) et, si besoin d'un nuage,
  `_weekly_row_samples` (généralisé, voir 3.2) — jamais réimplémenter le bootstrap ou le CRPS.
- **Dégradation gracieuse.** Comme `collect_weekly_kpis`, le collecteur mensuel renvoie un dict vide en cas de
  base absente/vide (try/except), sans faire échouer le reste du dashboard.
- **Honnêteté statistique (cf. §7).** Le mensuel est à **faible puissance** (effective_n ≈ 10). Aucun verdict
  d'équivalence ; les libellés restent factuels.

---

## 3. Objectif A — brancher le mensuel

### 3.1 Régime affiché
Calquer le weekly : afficher le **régime C (mensuel natif)** comme surface principale
(`frequence='monthly' AND horizon_type='monthly'`), M+1/M+2/M+3. Le régime B (daily→fin de mois) reste **hors
affichage** dans cette première version (comme le weekly n'affiche que le régime C) — le mentionner comme
extension possible, pas l'implémenter.

### 3.2 Nouvelles constantes et collecteur
- Ajouter `MONTHLY_KPI_MODELS = ["TSDiff", "NsDiff"]` (mêmes 2 modèles échantillonnés que le weekly ; ce sont les
  seuls présents en mensuel de toute façon).
- Ajouter `collect_monthly_kpis(db_path, n_samples=500, seed=42)` : **copie exacte de `collect_weekly_kpis`**, à
  trois substitutions près —
  1. filtre SQL `frequence='monthly' AND horizon_type='monthly'` ;
  2. `MONTHLY_KPI_MODELS` au lieu de `WEEKLY_KPI_MODELS` ;
  3. le groupe par `horizon_unit` renverra `M+1/M+2/M+3` (aucune hypothèse « W » codée en dur ailleurs à
     vérifier).
- **Nuage d'échantillons** : `_weekly_row_samples` est agnostique à la fréquence (il ne dépend que de
  `y_pred/y_lower/y_upper/last_close`). Le réutiliser tel quel ; si son nom gêne, l'aliaser localement
  (`_row_samples = _weekly_row_samples`) — **ne pas le dupliquer**. Cov95 exacte depuis les bornes (comme
  weekly), CRPS/cov50-80 en approximation gaussienne, `crps_is_approx=True` pour les deux modèles.

### 3.3 Payload + rendu
- Dans `main()`/`render_html()`, ajouter la sortie de `collect_monthly_kpis()` au payload à côté des KPI weekly,
  et **rendre la section mensuelle en miroir de la section weekly** (mêmes colonnes : point, IC95, Cov95 exacte,
  cov50/80 gaussiennes, CRPS approx, n_total/n_realized).
- Activer le bouton **« Monthly »** du sélecteur de pipeline : il n'est plus grisé dès qu'une donnée mensuelle
  existe pour l'actif courant (même condition d'activation que weekly).
- Badge de robustesse : réutiliser `collect_weekly_multiseed` en le généralisant aux modèles mensuels **si** un
  artefact multiseed mensuel existe (`nsdiff_monthly_multiseed.json` / `tsdiff_monthly_multiseed.json` sont
  présents dans `experiments/`) ; sinon dégradation gracieuse (pas de badge), jamais d'erreur.

### 3.4 Ce qu'on n'ajoute PAS
- **Aucune règle de trading mensuelle** (pas de nouveaux `TC*` dans `TC_PIPELINE`). Le mensuel est une surface
  **KPI**, pas un pipeline de signaux — cohérent avec le fait que `sim_trades` n'a de règles qu'à D+1.

---

## 4. Objectif B — à jour sur la base (fraîcheur vérifiable)

Le générateur lit déjà la base en direct : « à jour » n'est donc pas un bug de code mais un **problème de
régénération et de visibilité**. Deux livrables :

- **B1 — Stamp de provenance par piste.** Ajouter au bandeau, pour chaque piste lue en base (weekly, monthly),
  un libellé factuel : `dernière origine = MAX(cutoff_date)`, `n origines`, et le `run_id` le plus récent
  (`MAX(created_at)`). Objectif : lire d'un coup d'œil jusqu'où va la donnée affichée. Reprendre l'esprit du
  bandeau « honnête » déjà en place côté multiseed (`build_data_config`) — libellé factuel, jamais figé.
- **B2 — Régénération contrôlée.** La CI régénère à chaque push touchant `Run/**`, `tracking.db` ou le
  générateur. Après toute mise à jour de la base (ex. bascule ensemble 5×200 / `repoint-ensemble`), s'assurer
  que le push déclenche bien le workflow et que le stamp B1 reflète le nouveau `run_id`. Documenter la commande
  de régénération locale de contrôle : `python -m model_artifacts.generate_dashboard --inline` (prévisualisation
  `file://`).

> Note de synchronisation : sur le dépôt courant, aucune trace du run `repoint-ensemble` (08-08) — runs `oos`
> les plus récents `20260805` (mensuel) / `20260804` (daily-weekly). Exécuter ce brief **sur l'environnement où
> `tracking.db` est à jour** (ou après synchronisation), sinon le stamp B1 affichera fidèlement… un état périmé.

---

## 5. Tests unitaires (≥ 4)

1. `collect_monthly_kpis` sur une base jouet contenant des lignes M+1/2/3 → renvoie les bonnes cellules
   (NsDiff/TSDiff × 5 actifs), Cov95 exacte cohérente avec des cas construits.
2. Base sans mensuel → renvoie `{}` (dégradation gracieuse, pas d'exception).
3. Le mensuel **ne lit pas** le régime B (aucune ligne `frequence='daily'` ne remonte dans la surface mensuelle).
4. `_weekly_row_samples` réutilisé (pas dupliqué) : un test d'identité montre que la fonction sert bien les deux
   fréquences.
5. B1 : le stamp de provenance renvoie `MAX(cutoff_date)` et le `run_id` attendus sur la base jouet.

Contrainte : la suite existante reste verte, et les tests mensuels **n'importent aucune dépendance lourde**
(exécutables dans l'environnement CI minimal).

---

## 6. Vérification / régénération

- Régénérer (`--inline`) et vérifier de visu : l'onglet **Monthly** s'active, affiche NsDiff & TSDiff × 5 actifs,
  M+1/2/3, avec Cov95 et la mention « CRPS approx (gaussien) ».
- Rapprocher les comptes affichés de la base : `40 origines/actif`, fenêtre `2023-01 → 2026-04` — tout écart
  documenté dans le résumé de run, pas silencieux.
- Vérifier le stamp B1 (dernière origine + `run_id`) sur weekly **et** monthly.

---

## 7. Limite déclarée — faible puissance du mensuel

~130-140 clôtures mensuelles depuis 2015 ; cibles M+1/M+2/M+3 recouvrantes (corrélées) → **effective_n ≈ 10**.
La plupart des cellules mensuelles ressortiront « indistinguishable » : c'est l'effet **attendu** d'un faible
échantillon, **jamais** une équivalence démontrée. Le rendu doit l'afficher explicitement (note de bas de
section), exactement comme le weekly refuse de dire « plus précis ».

---

## Livrable

`generate_dashboard.py` augmenté de `collect_monthly_kpis` (+ constante `MONTHLY_KPI_MODELS`), du rendu mensuel
en miroir du weekly, de l'activation du bouton Monthly, et du stamp de provenance B1 — plus les tests. Dashboard
régénéré, onglet Monthly vivant, fraîcheur base lisible d'un coup d'œil. Aucun fichier partagé modifié hors
ajouts ; `tracking.db` intacte.
