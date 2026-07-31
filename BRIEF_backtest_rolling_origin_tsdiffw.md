# BRIEF — Backtest rolling-origin de TSDiff-W natif (Étape 1)

*Créé le 2026-07-27 — rattaché à `methodologie_diffusion_vs_classiques.md`, étape 1.*

---

## Pourquoi (le problème qu'on résout)

Le TSDiff weekly évalué jusqu'ici n'est **pas natif** : c'est le daily poussé en multi-pas → bruit de diffusion mal propagé → intervalles trop étroits → calibration weekly effondrée (16 % de couverture observée pour une cible de 50 %). C'est un **artefact de protocole (sous-problème P1)**, pas un échec du modèle.

Le TSDiff-W natif (`experiments/weekly_tsdiff_production.py`) est en prod depuis le 2026-07-21 mais **une seule vague de prédictions live** a été générée, non encore résolue. Attendre le live pour conclure = une poignée de points sur plusieurs mois → **puissance statistique insuffisante**, et risque de raisonnement motivé.

**Objectif de ce brief :** trancher P1 **maintenant** en backtestant TSDiff-W natif en **rolling-origin sur l'historique**, ce qui donne des centaines de points d'évaluation immédiatement, avec une comparaison à armes égales contre les baselines sur les mêmes origines.

**Question à laquelle le backtest doit répondre :**
> Une fois entraîné/prédit nativement en hebdo (et sans fuite temporelle), la calibration et le CRPS weekly de TSDiff se corrigent-ils, ou le déficit persiste-t-il ?

- Si **corrigés** → P1 confirmé et levé ; on peut ré-analyser proprement.
- Si le déficit **persiste** → ce n'était pas qu'un artefact ; il faut regarder P4 (méthodo) et P5 (cause de fond).

---

## Périmètre

- **Modèle central :** TSDiff-W natif (`experiments/weekly_tsdiff_production.py`).
- **Horizons :** W1, W2, W3 (comme la prod).
- **Actifs :** les 5 actifs de la matrice weekly actuelle (mêmes que le live pour comparabilité).
- **Baselines à rejouer sur les MÊMES origines :** ARIMA-GARCH, SARIMA, Naive, LSTM(-W), Prophet — celles déjà dans la matrice CRPS.

---

## Protocole rolling-origin (spécification)

1. **Fenêtre de backtest :** définir une période historique couvrant au moins plusieurs dizaines de cutoffs hebdo (viser **≥ 100–200 triplets (actif, cutoff, horizon) résolus** au total pour avoir de la puissance). Documenter début/fin.
2. **Boucle rolling-origin :** pour chaque cutoff hebdo `t` (pas = 1 semaine, origine glissante ou expansive — choisir et **documenter** le choix, cohérent avec le protocole d'entraînement TSDiff) :
   - le modèle ne voit **que** les données ≤ `t` (voir « Anti-fuite » ci-dessous) ;
   - générer les prévisions W1/W2/W3 **avec nuage d'échantillons** (TSDiff échantillonne nativement) ;
   - même **nombre d'échantillons** que le protocole live (vérifier dans le code prod) — assez pour un CRPS stable ;
   - récupérer `y_true` depuis les prix réels aux `target_date` correspondantes (`validation/price_fetcher.py`).
3. **Baselines sur origines identiques :** générer les prévisions des 5 baselines aux **mêmes (actif, cutoff, target)** — sinon la comparaison n'est pas appariée. Réutiliser `experiments/generate_samples_parametric.py` pour leurs nuages d'échantillons.
4. **Stockage isolé :** écrire les résultats **sans polluer le live**. Option propre : mêmes tables mais `source='backtest_rolling'` (la colonne `source` existe déjà dans `predictions`), ou une DB dédiée. Ne pas écraser les lignes `source='live'`.

## Anti-fuite temporelle (LE point critique du backtest)

Un backtest de modèle de diffusion est trompeur si le modèle a été entraîné sur tout l'historique puis « rejoué » dans le passé → fuite (lookahead). Pour chaque cutoff `t` :
- l'entraînement/fit ne doit utiliser **aucune** donnée postérieure à `t` ;
- si TSDiff-W est ré-entraîné par fenêtre, ré-entraîner (ou charger un checkpoint entraîné uniquement sur ≤ `t`) ;
- si c'est trop coûteux en compute, **documenter explicitement** l'approximation retenue et son biais.
- Vérifier qu'aucune normalisation/scaler n'est ajustée sur l'ensemble complet.

---

## Métriques (identiques au protocole live, pour comparabilité)

Réutiliser l'outillage existant — **ne pas réimplémenter** :
- **CRPS empirique** sur nuage d'échantillons : `experiments/crps_metrics.py::crps_empirical` (Gneiting & Raftery 2007, éq. 20).
- **Calibration / couverture** aux niveaux 50 / 80 / 95 % : `prob_kpi_common.py::coverage_flag`.
- **Sharpness, Winkler, PIT :** `prob_kpi_common.py` (`sharpness`, `winkler_score`, `pit_value`, `row_kpis`).
- **Point :** MAE, `beats_naif`, `direction_correct` (colonnes déjà calculées à l'éval).
- **Tests appariés** TSDiff-W vs chaque baseline sur les triplets communs (cf. `experiments/paired_test.py` / `matrice_paired_tests.py`).

---

## Livrables attendus

1. **Script de backtest** rolling-origin (nouveau fichier `experiments/backtest_rolling_tsdiffw.py`), paramétrable (fenêtre, actifs, horizons, nb d'échantillons).
2. **Données de backtest** stockées et taguées `source='backtest_rolling'` (ou DB dédiée).
3. **JSON de résultats** au format de `kpi_probabilistes.json` : CRPS normalisé + couverture 50/80/95 par horizon et par modèle, sur les origines du backtest.
4. **Note de synthèse courte** (1 page md) répondant à la question centrale : la calibration/CRPS weekly de TSDiff-W se corrige-t-elle ? Avec le **nombre de points résolus** (puissance) et les tests appariés.
5. **Journal des choix de protocole** : fenêtre, glissant vs expansif, stratégie anti-fuite, nb d'échantillons.

---

## Critères d'acceptation

- [ ] Anti-fuite temporelle explicitement gérée et documentée (bloquant).
- [ ] TSDiff-W et les 5 baselines évalués sur des triplets **(actif, cutoff, target) strictement identiques**.
- [ ] ≥ ~100–200 triplets résolus (sinon, dire pourquoi et quelle puissance on a).
- [ ] CRPS/couverture calculés avec l'outillage existant, pas une réimplémentation ad hoc.
- [ ] Le live (`source='live'`) est intact.
- [ ] La note conclut clairement : P1 levé (corrigé) / partiellement / non.

---

## Pièges à éviter

- **Ne pas** conclure sur le live weekly (trop peu de points) — c'est justement ce qu'on remplace.
- **Ne pas** comparer TSDiff-W à des baselines calculées sur d'autres cutoffs.
- **Ne pas** laisser une fuite temporelle transformer un mauvais modèle en bon (ou l'inverse).
- **Ne pas** oublier que le **daily**, lui, perd déjà en natif — ce backtest ne concerne que P1/weekly, il ne "sauve" pas l'hypothèse globale.
