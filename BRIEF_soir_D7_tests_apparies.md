# BRIEF (ce soir) — Corriger D+7 + tests appariés sur toute la matrice

> Deux tâches à finir **ce soir** avant de refaire la synthèse. Le reste du plan (décision de
> protocole, réglage hebdo des modèles classiques) reste pour plus tard.

## 0. Contexte

La matrice est complète (240/240, 30 origines) et le rapport de synthèse est produit — mais deux
verrous empêchent d'en faire des résultats fiables : (1) un bug de données sur D+7, (2) l'absence
de test de significativité sur la matrice complète. On règle les deux, puis on régénère la
synthèse.

## 1. Tâche 1 — Corriger le bug D+7 (cause racine + ré-ingestion)

**Symptôme** : sur les 1 809 lignes D+7, `cutoff_date` et `last_close` sont faux à 100 % (et
donc `abs_error_naif`/`beats_naif`/`direction_correct` en cascade). Les prévisions elles-mêmes
(`y_pred`, intervalles, `y_true`, `target_date`) sont correctes → RMSE et couverture ne sont pas
affectés, mais l'exactitude directionnelle et tout ce qui dérive du prix d'origine le sont.

**Cause** : le script d'ingestion (`build_oos_prediction_rows()`) suppose « ligne précédente =
la veille » — vrai pour le D+1 dense, faux pour le D+7 à 10 origines espacées de ~17 jours.

**À faire** :
1. **Corriger la cause racine** dans `build_oos_prediction_rows()` : lire le vrai couple
   (origine, cible) depuis la logique Gate2 du pipeline, ne plus déduire l'origine de la ligne
   précédente.
2. **Supprimer et ré-ingérer** les 1 809 lignes D+7 proprement.
3. **Vérifier le même bug sur les lignes hebdomadaires** (W+1/2/3) : leurs origines sont aussi
   espacées (~1/semaine), donc si elles ont été ingérées via la même logique, `cutoff_date`/
   `last_close` sont peut-être faux aussi — invisible sur RMSE/couverture. Contrôler et corriger
   si besoin.
4. **Recouper** : après correction, `cutoff_date`/`last_close` doivent correspondre au recalcul
   depuis les prix réels (`cutoff = target − 7 j calendaires`, même jour de semaine ;
   `last_close` = clôture réelle à cette date). Doit matcher.
5. Au passage, **unifier la définition D+7 = cutoff + 7 jours calendaires** (même jour de
   semaine, repli sur jour de bourse le plus proche si férié) et supprimer la variante
   « 7 jours de bourse ».

**Sortie attendue** : nb de lignes D+7 corrigées, si le weekly était touché (et combien),
résultat du recoupement.

## 2. Tâche 2 — Tests appariés sur toute la matrice

Réutiliser `experiments/paired_test.py` (déjà écrit et testé) pour remplacer les classements par
estimation ponctuelle par des comparaisons **testées**.

**Règle du bootstrap (important — chevauchement des origines)** : les 30 origines se chevauchent
(cf. horizons W+1/2/3 depuis des origines espacées d'une semaine) → leurs erreurs sont corrélées.
Utiliser un **bootstrap par blocs** (rééchantillonner des paquets d'origines consécutives), pas
un tirage indépendant, sinon les p-values sont trop optimistes. Afficher aussi la taille
d'échantillon effective (~10-15, pas 30).

**Comparaisons à produire** (toutes intra-actif, RMSE jamais comparé entre actifs) :

1. **Classement des modèles par horizon × actif** : pour chaque (actif, horizon), quels modèles
   sont significativement meilleurs, lesquels sont indistinguables (différence de RMSE et/ou
   CRPS testée par paires).
2. **Calibration par horizon** : comparer les couvertures (déjà mesurées) — surtout marquer qui
   atteint ~0,95 et qui sous-couvre significativement.
3. **Daily vs weekly par modèle** : régime B (daily→weekly) vs régime C (weekly natif) pour
   chaque modèle. *Comparaison propre* (protocole constant à l'intérieur d'un modèle).
4. **D+7 vs W+1 par modèle × actif**, alignés sur **origines-vendredi** (D+7 depuis un vendredi
   cible le vendredi suivant = cible de W+1). *Comparaison propre*.

**Règle de conclusion** : ne déclarer « X meilleur que Y » **que si l'écart est significatif**
(p < 0,05 après bootstrap par blocs). Sinon → « indistinguable ».

**Caveat à porter dans les résultats** : les comparaisons **inter-modèles** (point 1) restent
entachées de l'asymétrie de protocole (TSDiff figé vs les autres ré-entraînés) → à interpréter
avec prudence tant que le protocole n'est pas unifié. Les points 3 et 4 (intra-modèle) ne sont
pas concernés → ce sont les conclusions les plus solides.

## 3. Livrables

- Base `tracking.db` corrigée (D+7 et, si besoin, weekly).
- `experiments/matrice_paired_tests.json` : pour chaque comparaison, différence moyenne,
  intervalle bootstrap par blocs, p-value, verdict (significatif / indistinguable), taille
  effective.
- Un point d'étape court : lignes corrigées + résumé des comparaisons significatives.

## 4. Ensuite

Une fois ces deux tâches finies et vérifiées, on **régénère la synthèse** (le PDF) en
remplaçant les classements ponctuels par les verdicts testés. Le reste du plan (unification du
protocole, réglage hebdo des classiques) reste pour plus tard.

## 5. Vérification

- Aucune ligne D+7 avec `cutoff ≠ target − 7 j calendaires` après correction.
- Recoupement `cutoff`/`last_close` vs prix réels : 0 écart.
- Bootstrap par blocs bien utilisé (pas de tirage indépendant sur des origines corrélées).
- Relire les p-values en gardant la taille effective (~10-15) en tête.
