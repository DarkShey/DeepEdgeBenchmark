# BRIEF — Introduire le régime **weekly** comme pipeline de premier plan (surface probabiliste + génération récurrente TSDiff)

> Fait suite au test rigoureux `experiments/METHODOLOGIE_weekly_vs_daily.md` /
> `experiments/weekly_vs_daily_pooled.json`. **Verdict qui cadre ce brief** : le weekly
> ne bat significativement le daily sur la précision (CRPS) pour aucun modèle, mais
> **corrige significativement la sur-confiance de TSDiff** (couverture 95 % 0,55 → 0,78,
> p<0,001 après Holm). L'intégration est donc justifiée sur le terrain de la
> **calibration de l'incertitude**, pas de la précision — et ce brief doit refléter
> cette honnêteté partout (étiquetage, doc, dashboard).

## 0. État des lieux (vérifié dans le code, ne pas re-supposer)

La pipeline **daily** est une chaîne complète : génération → `validation/tracking.db`
(`real_flag='live'`) → `validation/sim_trades.py` (trades simulés) → dashboard
(`model_artifacts/generate_dashboard.py`). Le **weekly** n'a que le premier maillon :

- **Données présentes** : `tracking.db` contient déjà 5 400 lignes `horizon_type='weekly'`
  (2 700 régime B daily→hebdo + 2 700 régime C hebdo natif), 6 modèles × 5 actifs ×
  W+1/W+2/W+3, toutes évaluées. Colonnes `frequence` / `horizon_type` / `horizon_unit`
  déjà en schéma.
- **Trou 1 — couche trading** : `sim_trades.py` filtre en dur `horizon_type='daily'`
  (≈ lignes 188 et 687). Le weekly est invisible pour la couche trades + « Vraies/Fausses
  prédictions ».
- **Trou 2 — dashboard** : `SIM_TRADES_PIPELINES = ["daily","weekly","monthly"]` et
  `pipelineLabel = {daily,weekly,monthly}` existent, mais l'onglet weekly est un
  **placeholder vide**.
- **Trou 3 — génération récurrente** : rien ne produit du weekly *en avant*. La
  génération natif hebdo existe en one-shot dans `experiments/weekly_multiasset.py`
  (`--phase final`, TSDiff-W + RandomWalk), jamais wirée en production récurrente.

## 1. Objectif

Faire du **régime C (hebdo natif)** un régime de premier plan, **visible et alimenté en
continu**, sur le terrain où il a une valeur démontrée (KPI probabilistes), sans sur-vendre.

Concrètement, deux livrables :

1. **Surface** : le weekly devient un régime sélectionnable dans le dashboard, affichant
   les prédictions weekly et leurs **KPI probabilistes** (point, intervalle, CRPS,
   couverture 50/80/95) pour les modèles disposant de données weekly, avec **TSDiff mis
   en avant** comme le modèle qui bénéficie (calibration).
2. **Génération récurrente** : un job hebdomadaire qui produit du régime C **en avant**,
   stocké en `real_flag='live'`, **pour TSDiff** (architecture extensible aux autres,
   mais on ne génère que TSDiff par défaut, cf. verdict).

## 2. Décision de conception assumée (périmètre)

- On **n'introduit PAS** de scénarios de trade weekly. Les scénarios de
  `sim_trades.py` sont spécifiques au D+1 (`bull_calm_d1`, `sideways_d1`, …). Concevoir
  leur équivalent weekly (horizon 1–3 semaines, seuils, frais) est une **décision de
  design séparée**, hors périmètre. → Le weekly est surfacé sur les **KPI probabilistes**,
  pas sur la couche trades simulés.
- On **ne touche pas** au régime B dans la génération récurrente (le verdict ne lui
  donne pas d'intérêt ; il reste en base pour la comparaison). La génération récurrente
  ne concerne que le régime C.
- **TSDiff uniquement** pour la génération récurrente. La *surface* dashboard peut
  afficher les 6 modèles (leurs données weekly existent déjà), mais l'étiquette doit
  dire clairement que seul TSDiff a un gain mesuré.

## 3. Garde-fous (honnêteté — non négociables)

- **Étiquetage** : partout où le weekly TSDiff est présenté, indiquer que le gain est
  **de calibration, pas de précision** (renvoyer à `METHODOLOGIE_weekly_vs_daily.md`).
  Ne jamais afficher/écrire « weekly plus précis ».
- **Pas de fuite** : la génération récurrente ne conditionne que sur du passé ≤ origine
  (fenêtre réalisée), `mu`/`sd` figés, mêmes garde-fous que `weekly_multiasset.py`.
- **`real_flag`** recalculé via `tracking_db.compute_real_flag` (ne pas le poser à la
  main) ; passer par `sim_trades.insert_oos_predictions` (ou le chemin live équivalent),
  pas d'`INSERT` maison, pour respecter la contrainte d'upsert
  `(source, model, asset, horizon, frequence, horizon_type, cutoff_date)`.
- **LSTM d'abord** (dépendance externe à ce brief, à signaler) : le LSTM weekly est
  significativement dégradé (CRPS +0,74) — probablement un bug de données manquantes, pas
  un vrai effet. **Ne pas afficher le LSTM weekly comme un résultat valide** tant que ce
  n'est pas diagnostiqué (cf. §6). Le masquer ou le marquer « suspect » dans la surface.
- **Tests** : chaque couche modifiée a son test (le dépôt teste `sim_trades`,
  `generate_dashboard`, `pipeline`). Ne pas régresser les filtres daily existants.

## 4. Plan d'implémentation

### Couche A — Surface dashboard (weekly en KPI probabilistes)
1. Ajouter un sélecteur de **fréquence/régime** (daily ↔ weekly) dans
   `model_artifacts/generate_dashboard.py`, réutilisant la structure
   `SIM_TRADES_PIPELINES`/`pipelineLabel` déjà réservée.
2. Pour la vue weekly, lire les lignes `horizon_type='weekly'`, `frequence='weekly'`
   (régime C) et afficher par (modèle × actif × W+1/2/3) : point, intervalle,
   **couverture 50/80/95, CRPS** (réutiliser `experiments/prob_kpi_common.py` /
   `crps_metrics.py`, ne pas réimplémenter).
3. Mettre TSDiff en tête avec le badge calibration (0,55 → 0,78). Marquer LSTM weekly
   « suspect » (cf. §3).

### Couche B — Génération récurrente TSDiff weekly (régime C)
4. Extraire de `experiments/weekly_multiasset.py` (`--phase final`) la logique de
   génération natif hebdo TSDiff-W en une fonction de **production** réutilisable
   (fit ≤ origine, resample `W-FRI`, échantillons → point + quantiles → W+1/2/3).
5. L'ordonnancer en **hebdomadaire** (vendredi soir, après clôture) : générer les
   prédictions W+1/2/3 pour les 5 actifs, stocker via `insert_oos_predictions` avec
   `frequence='weekly'`, `horizon_type='weekly'`, `real_flag` recalculé.
6. À maturité des cibles (semaine écoulée), le backfill d'évaluation existant
   (`experiments/backfill_eval_metrics.py`) renseigne `y_true`/métriques.

### Couche C — (Optionnel, si demandé) filtre sim_trades
7. **Hors périmètre par défaut.** Si le tuteur veut la couche trades, ouvrir un brief
   séparé « scénarios de trade weekly » — ne pas bricoler `_d1` en weekly ici.

## 5. Critère de succès

- Le dashboard a un onglet **weekly fonctionnel** montrant, par actif, les prédictions
  régime C + CRPS + couverture, TSDiff en avant avec l'étiquette calibration.
- Un run du **job hebdo** produit et stocke de nouvelles lignes TSDiff weekly `live`,
  vérifiables en base.
- Aucun test daily existant ne régresse.
- Nulle part il n'est écrit/affiché que « le weekly est plus précis ».

## 6. Dépendance à traiter en parallèle (à ne pas ignorer)

**Diagnostiquer le LSTM weekly** avant de le montrer : CRPS +0,74 vs daily, couverture
qui baisse — signature d'un bug (données manquantes / resample), cohérent avec l'anomalie
LSTM déjà observée. Tant que non résolu, LSTM weekly est masqué/marqué suspect (§3).

## 7. Ce que ce brief ne fait PAS

- Pas de scénarios de trade weekly / pas de couche `sim_trades` weekly (couche C = brief
  séparé).
- Pas de génération récurrente pour les 5 autres modèles (seul TSDiff est justifié).
- Pas de monthly.
- Pas de revendication de gain de **précision** — uniquement **calibration**, TSDiff.
- Pas de re-test statistique (déjà fait : `METHODOLOGIE_weekly_vs_daily.md`).

## 8. Références (fichiers à lire avant de coder)

| Quoi | Où |
|---|---|
| Verdict + méthode qui cadre ce brief | `experiments/METHODOLOGIE_weekly_vs_daily.md`, `experiments/weekly_vs_daily_pooled.json` |
| Générateur natif hebdo (à extraire en prod) | `experiments/weekly_multiasset.py` (`--phase final`) |
| Chemin d'insertion + contrainte upsert | `validation/sim_trades.py` (`insert_oos_predictions`) |
| `real_flag` | `validation/tracking_db.py` (`compute_real_flag`) |
| Dashboard + slots weekly réservés | `model_artifacts/generate_dashboard.py` |
| KPI probabilistes (réutiliser) | `experiments/prob_kpi_common.py`, `experiments/crps_metrics.py` |
