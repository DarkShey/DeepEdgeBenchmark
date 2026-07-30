# BRIEF — Analyse poolée : extraire le maximum de signal des données actuelles

> Aucune nouvelle donnée / aucun nouvel entraînement. On exploite mieux les 240 cellules déjà en
> base pour transformer des « non concluants » en verdicts fermes là où c'est possible.

## 0. Constat

Les tests par cellule (240 comparaisons éparpillées) manquent de puissance : la plupart sont
non concluants sur 30 origines chevauchantes. Mais on peut **regrouper** l'évidence et raisonner
en **taille d'effet + intervalle de confiance**, ce qui répond souvent à la question sans
nouvelle donnée. C'est le levier au meilleur rapport résultat/effort.

## 1. Objectif

Répondre fermement à deux questions **propres** (non confondues par le protocole) :
1. **La fréquence d'entraînement change-t-elle quelque chose ?** (régime B vs C, par modèle, poolé)
2. **Quels modèles sont fiablement (mal) calibrés ?** (par modèle, poolé)

…en fournissant pour chacune une **taille d'effet, un intervalle de confiance à 95 %, et une
p-value robuste**, avec correction pour comparaisons multiples. Un intervalle serré autour de
zéro doit être présenté comme une conclusion (« pas de différence exploitable »), pas comme un échec.

## 2. Décisions méthodo (à appliquer)

### 2.1 Métrique sans échelle (indispensable au pooling)
Le RMSE brut n'est pas comparable entre actifs. Construire un **différentiel de perte par
origine, sans échelle** :
- Point : erreur **scalée façon MASE** (erreur absolue / erreur absolue moyenne d'un naïf
  in-sample sur le même actif) **ou** skill vs naïf. À justifier, mais rester cohérent.
- Distribution : **CRPS normalisé** (CRPS / échelle de l'actif) pour la partie incertitude.
Le différentiel poolé doit donc être adimensionnel et comparable BTC ↔ ZN=F.

### 2.2 Test approprié
- **Diebold-Mariano à variance HAC** (Newey-West, lag ≈ horizon) sur la série des différentiels
  de perte par origine — le standard pour comparer deux prévisions à pertes autocorrélées.
- **Bootstrap par blocs** (déjà écrit) en robustesse ; les deux doivent concorder.

### 2.3 Pooling + dépendance
- **Un test par modèle**, poolé sur toutes ses cellules (actifs × horizons), au lieu de 15 tests.
- **Corréler = ne pas double-compter** : ZN=F et TLT (même sous-jacent taux) et les deux cryptos
  sont corrélés → clusteriser par **classe d'actif** (erreurs-types cluster-robustes, ou
  agrégation hiérarchique), pas traiter les 5 actifs comme indépendants.
- **Comparaisons multiples** : correction Holm (ou Benjamini-Hochberg) sur les 6 modèles.

### 2.4 Taille d'effet + IC partout
Rapporter systématiquement : différentiel moyen (en unités de skill), **IC 95 %**, p-value
(DM-HAC et bootstrap). Verdict par cellule/modèle : « significatif (sens) » / « pas de
différence exploitable — IC contenu dans ±X ».

## 3. Portée (et ce qu'on NE fait PAS)

- **Question fréquence (B vs C)** : poolée par modèle → verdict ferme par modèle.
- **Question D+7 vs W+1** : même pooling par modèle (en assumant la faible puissance, l'afficher).
- **Calibration** : par modèle, **un test de couverture poolé** (couverture observée vs 0,95,
  type Kupiec / binomial, avec IC sur l'écart de couverture) → un seul chiffre par modèle qui
  confirme/renforce « ARIMA-GARCH le plus fiable, TSDiff le pire ».
- **On NE poole PAS le classement inter-modèles** : il porte le biais de protocole (TSDiff figé
  vs autres réentraînés). Aucun test ne corrige un biais → à traiter par le re-run (autre brief),
  pas ici. Le noter explicitement.

## 4. Livrables

- `experiments/pooled_analysis.json` : par question et par modèle → métrique utilisée,
  différentiel moyen, IC 95 %, p-value DM-HAC, p-value bootstrap, verdict, taille effective.
- Un point d'étape court : pour chaque modèle, la fréquence change-t-elle quelque chose (oui /
  non / IC), et le verdict de calibration poolé.

## 5. Vérification

- Métrique bien adimensionnelle (résultats cohérents entre actifs d'échelles différentes).
- DM-HAC et bootstrap par blocs concordent ; sinon investiguer.
- Clustering par classe d'actif effectif (ZN=F/TLT et les cryptos non double-comptés).
- Correction multiple appliquée ; IC rapportés partout (pas que des p-values).

## 6. Ensuite

Une fois `pooled_analysis.json` produit, on régénère la synthèse : les « non concluants » par
cellule deviennent, là où c'est possible, des verdicts fermes par modèle (avec IC). Le reste
(classement inter-modèles, plus de puissance) reste pour les briefs suivants.
