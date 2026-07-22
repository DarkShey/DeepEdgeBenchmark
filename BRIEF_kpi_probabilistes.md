# BRIEF — KPI probabilistes : réévaluer la matrice avec des métriques adaptées (diffusion)

## 0. Contexte

La synthèse actuelle classe les modèles au **RMSE**, qui ne juge que la prévision **centrale**
(la moyenne). C'est inadapté aux modèles **probabilistes / de diffusion** (TSDiff) qui produisent
une **distribution entière** de scénarios : le RMSE ignore la forme de la distribution, les
queues et l'incertitude — donc les résultats sont peu exploitables pour ces modèles. On refait
l'évaluation avec des KPI adaptés, puis on refait la synthèse.

## 1. Objectif

Réévaluer les **240 combinaisons** (6 modèles × 5 actifs × 5 horizons) sur des **KPI
probabilistes**, rejouer les tests statistiques rigoureux sur ces KPI, et produire une nouvelle
synthèse comparative exploitable.

## 2. Le jeu de KPI

- **CRPS empirique** (métrique principale) — proper scoring rule qui note la distribution
  complète (précision + incertitude), calculée sur le **nuage d'échantillons**. Remplace le RMSE
  comme métrique reine.
- **Calibration multi-niveaux + PIT** — couverture testée à **50 % / 80 % / 95 %** (pas juste
  95 %), + histogramme PIT pour vérifier que la distribution est juste à **tous** les niveaux.
- **Sharpness** — largeur moyenne des intervalles, lue **à couverture égale**.
- **Score de Winkler / d'intervalle** — couverture + finesse en un chiffre par niveau.
- **MASE** — vue « point » sans échelle, conservée mais **secondaire** (plus la métrique
  principale pour les modèles probabilistes).
- **Energy score** *(bonus, optionnel)* — juge le **chemin conjoint W1→W2→W3** (la diffusion
  génère des trajectoires cohérentes). À faire seulement si le reste tient.

## 3. Le prérequis : persister les échantillons

La base ne stocke aujourd'hui que **point + intervalle 95 %**. Le CRPS empirique et le PIT
exigent les **échantillons bruts**. Il faut donc les régénérer et les sauvegarder, **pour les 6
modèles**, à **N échantillons identique** (ex. N = 500) pour l'équité :

- **TSDiff** : échantillons **natifs** (ré-échantillonnage depuis le modèle déjà fitté, sans
  ré-entraîner).
- **ARIMA-GARCH / SARIMA / Prophet / Naive** : tirer N échantillons de leur **distribution
  prédictive** (paramétrique) à chaque origine/horizon.
- **LSTM** : selon l'implémentation (MC-dropout ou tirage depuis sa bande prédictive) — à
  documenter.

**Décision de conception assumée** : on **ne ré-entraîne pas** — on ré-échantillonne / on tire
depuis les modèles/distributions déjà en place, aux **mêmes origines et dates-cibles** que la
matrice actuelle (sinon les KPI ne seront pas comparables à l'existant).

## 4. Étapes

1. **Générer + persister les échantillons** (N=500) pour les 6 modèles, sur les 240 cellules,
   mêmes origines/dates-cibles. Stockage dédié (pas de surcharge de `tracking.db` avec des
   nuages ; plutôt un artefact `experiments/samples/…` ou colonnes/table dédiée).
2. **Calculer les KPI** (CRPS empirique, couverture 50/80/95, PIT, sharpness, Winkler, MASE) par
   cellule → `experiments/kpi_probabilistes.json`.
3. **Rejouer les tests** : pooling par modèle + **DM-HAC / bootstrap par blocs** + tailles
   d'effet et IC, mais **sur CRPS et calibration** au lieu du RMSE. Correction multiple, clusters
   par classe d'actif (comme l'analyse poolée existante).
4. **Refaire la synthèse** (PDF) avec ces KPI — TSDiff enfin jugé sur ce qu'il fait vraiment.

## 5. Garde-fous

- **N échantillons identique** pour les 6 modèles (équité stricte du CRPS/PIT).
- **CRPS empirique cohérent** (même estimateur pour tous).
- **PIT sur la distribution complète**, pas juste l'intervalle 95 %.
- **Pas de lookahead** : ré-échantillonnage conditionné sur du passé ≤ origine uniquement.
- **Puissance toujours affichée** (origines chevauchantes → effectif ~n/3) ; ne rien déclarer
  significatif sans le test.
- Rappel **hors périmètre** (inchangé) : le classement des modèles sur la précision brute reste
  biaisé par l'asymétrie de protocole (TSDiff figé vs autres réentraînés) — mais les KPI
  probabilistes par modèle et les questions intra-modèle n'en souffrent pas.

## 6. Ce que ce brief ne fait PAS

- Pas de nouveaux modèles / actifs / horizons — on reste sur les 240 cellules.
- Pas de ré-entraînement — uniquement ré-échantillonnage depuis les modèles fittés.
- Pas de conclusion sans test statistique.

## 7. Livrables

- Échantillons persistés (N=500, 6 modèles, 240 cellules).
- `experiments/kpi_probabilistes.json` (KPI par cellule) + tests poolés (CRPS/calibration).
- Nouvelle synthèse PDF centrée sur les KPI probabilistes.
