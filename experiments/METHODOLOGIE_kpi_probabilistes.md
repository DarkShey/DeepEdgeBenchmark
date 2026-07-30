# Ce qui a été testé — KPI probabilistes (BRIEF_kpi_probabilistes.md)

Ce document explique, étape par étape et sans code, tout ce qui a été fait pour produire
`experiments/rapport_kpi_probabilistes.pdf`. Objectif : que ce ne soit plus une boîte noire.

## 0. Le problème de départ

La synthèse précédente jugeait les 6 modèles au **RMSE** (erreur quadratique moyenne sur la
prédiction ponctuelle). Ça marche pour un modèle qui sort UN chiffre, mais TSDiff sort une
**distribution** (un nuage de scénarios) — le RMSE ne regarde que le centre de ce nuage, jamais
sa forme ni sa largeur. Autrement dit : un modèle peut avoir un bon RMSE et être complètement
faux sur son incertitude (trop confiant ou pas assez), et le RMSE ne le verra jamais.

## 1. Le périmètre : les "240 combinaisons"

Ce nombre vient de : **6 modèles × 5 actifs × 8 "cellules de régime"**.

Les 5 actifs : SPY, BTC-USD, ETH-USD, ZN=F (obligations), TLT (obligations).

Les 8 cellules de régime, c'est la combinaison de 3 choses :
- la **fréquence d'entraînement** du modèle (quotidien ou hebdomadaire),
- le **type d'horizon** (quotidien D+1/D+7, ou hebdomadaire W+1/W+2/W+3),
- l'**horizon précis**.

| Cellule | Fréquence | Horizon |
|---|---|---|
| A | quotidien | D+1 |
| A | quotidien | D+7 |
| B | quotidien (entraîné) → cible hebdo | W+1, W+2, W+3 |
| C | hebdomadaire (natif) | W+1, W+2, W+3 |

Ce ne sont **pas** de nouvelles données : ce sont les prédictions déjà stockées dans
`validation/tracking.db` (table `predictions`), celles qui alimentent le dashboard actuel.
Je n'ai rien régénéré côté "quelle date prédire quoi" — j'ai pris exactement les origines
(`cutoff_date`) et cibles (`target_date`) déjà en base, filtrées comme le dashboard le fait
(`daily_duplicate=0`, lignes déjà évaluées c'est-à-dire avec un `y_true` connu).

**Résultat réel obtenu :** 150 cellules sur 240 avaient effectivement des données (voir
§6 "trous connus" — ce n'est pas un raté de ma part, c'est l'état réel de la base).

## 2. Le problème technique : il n'existe aucun nuage de points nulle part

`tracking.db` ne stocke que **3 nombres** par prédiction : le point (`y_pred`) et l'intervalle
à 95 % (`y_lower`, `y_upper`). Jamais les tirages individuels qui ont servi à les calculer.
Pour calculer un CRPS empirique ou une calibration à plusieurs niveaux, il faut un **nuage
d'échantillons** (j'ai choisi N=500 par ligne, identique pour les 6 modèles — condition
d'équité explicite du brief).

Il a donc fallu **régénérer 500 échantillons par ligne**, pour environ 9000 lignes. Deux
méthodes complètement différentes selon les modèles, expliquées ci-dessous.

## 3. Pour 5 modèles sur 6 : aucun nouveau calcul, juste retrouver leur loi

**ARIMA-GARCH, SARIMA, Prophet, Naive, LSTM.**

J'ai lu le code source de chacun (`models/*.py`) pour voir *comment* il construit son
intervalle à 95 %. Résultat : les 5 le font tous de la même manière — un centre `y_pred`
plus ou moins `1,96 × sigma` (c'est la définition d'un intervalle de confiance gaussien à
95 %). ARIMA-GARCH le fait en espace "rendement logarithmique" (donc log-normal en prix),
les 4 autres directement en prix.

Conséquence : si je connais `y_pred`, `y_lower` et `y_upper` (déjà en base), je peux
**retrouver `sigma` par un simple calcul algébrique** (`sigma = (y_upper - y_pred) / 1,96`),
puis tirer 500 nombres aléatoires dans cette loi (gaussienne ou log-normale) avec ce
`sigma`. C'est tirer dans **la loi que le modèle avait déjà implicitement choisie** — pas une
approximation inventée pour l'occasion.

**Coût : zéro.** Pas de ré-entraînement, pas de rechargement de modèle, juste du calcul.
Vérifié : j'ai regénéré les échantillons puis recalculé les quantiles à 2,5 % et 97,5 % dessus
— ils retombent à moins de 0,15 % de l'intervalle original stocké en base (cohérence numérique).

*Limite honnête à garder en tête : ce nuage n'a jamais existé "en vrai" pour ces 5 modèles —
c'est la loi théorique qu'ils utilisaient déjà pour fabriquer leur intervalle, pas un
échantillonnage indépendant du modèle réel.*

## 4. TSDiff : le seul modèle qui a dû être re-calculé

TSDiff est un modèle de diffusion : il génère nativement un nuage de scénarios, contrairement
aux 5 autres. Mais j'ai vérifié directement dans le code (`model_artifacts/pipeline.py`) une
chose importante : **aucun modèle TSDiff entraîné n'est jamais sauvegardé sur disque**. Chaque
valeur actuellement en base pour TSDiff a été produite par un entraînement qui a eu lieu une
fois, dont le nuage de 50 échantillons a été utilisé pour calculer moyenne + quantiles, puis
**jeté**. Il n'y a donc rien à "recharger" : impossible de faire du "zéro calcul" comme pour les
5 autres.

**J'ai donc dû relancer un vrai entraînement**, mais en suivant strictement le protocole déjà
utilisé (même code, même seed=42, mêmes hyperparamètres déjà choisis avant ce travail par
`epoch_sweep.py`), en gardant cette fois les 500 échantillons au lieu de les jeter. Ce n'est
pas une nouvelle méthode inventée — c'est exactement ce que le pipeline fait déjà à chaque
prédiction, sauf qu'on garde le résultat intermédiaire.

Détail des 3 entraînements distincts (un par cellule de régime, car chacune utilise une
fenêtre de données et des hyperparamètres différents) :

| Cellule | Ce qui est entraîné | Fenêtre d'entraînement | Epochs |
|---|---|---|---|
| A (D+1/D+7) | 1 modèle par actif | 700 derniers jours avant la 1ère origine testée | 40 (défaut du pipeline) |
| B (W+1/2/3, entraîné quotidien) | 1 modèle par actif | tout l'historique dispo (2015→) | 20 à 30 selon l'actif (déjà choisi par `epoch_sweep.py`) |
| C (W+1/2/3, natif hebdo) | 1 modèle par actif | tout l'historique dispo (2015→) | 30 à 80 selon l'actif (déjà choisi par `weekly_multiasset.py`) |

Soit **13 entraînements TSDiff au total** (5 actifs × cellule A, + 5 × cellule B, + 3 × cellule C
qui avait des données pour seulement 3 actifs — voir §6). Chaque entraînement prend entre 20
secondes (cellule C) et ~2-3 minutes (cellule A/B) de calcul pur, plus le temps de générer les
500 échantillons à chaque origine testée (le plus long : jusqu'à 15-20 minutes par actif pour
la cellule A qui a le plus d'origines).

### Un bug trouvé et corrigé pendant le travail

Ma première tentative (cellule A) a produit un nuage **quasi figé** (écart-type de 0,20 $ sur un
titre à 700 $ — alors que le titre bouge de plusieurs dollars par jour). Vérification : les
points centraux (la moyenne du nuage) étaient corrects, seule la largeur était absurdement
petite. J'ai identifié la cause : j'entraînais sur tout l'historique depuis 2015 (~2800 jours)
alors que le pipeline de production n'a jamais utilisé plus de ~700 jours pour cette cellule
(vérifié dans les métadonnées des runs passés). Avec 4× plus de données au même nombre
d'epochs, le réseau reçoit ~4× plus de gradient-steps et s'effondre (un problème déjà connu et
documenté ailleurs dans ce dépôt pour TSDiff). Corrigé en bornant la fenêtre à 700 jours pour
cette cellule uniquement (B et C utilisaient déjà, par construction, la bonne fenêtre longue).
Vérifié après correction : écart-type revenu à 9-15 $, cohérent avec la largeur des intervalles
déjà stockés en base.

*Limite honnête : ce nouvel entraînement suit le même protocole mais n'est pas
"bit-identique" à ce qui a produit les valeurs originales en base (corrélation ~0,97 entre mes
points centraux et les `y_pred` stockés pour la cellule A — proche, pas identique).*

## 5. Les KPI calculés, sur chaque ligne (chaque prédiction × 500 échantillons)

- **CRPS empirique** — la métrique principale. Note la distribution *entière* contre la valeur
  réalisée : plus bas = meilleur. Contrairement au RMSE, un modèle est pénalisé s'il est trop
  confiant (nuage trop étroit) OU pas assez (nuage trop large), pas seulement s'il se trompe sur
  le centre.
- **Couverture à 50 %, 80 %, 95 %** — sur 100 prédictions, si le modèle dit "80 % de chances que
  ça tombe dans cette fourchette", est-ce que ça tombe dedans ~80 fois sur 100 dans la réalité ?
- **Finesse (sharpness)** — largeur de l'intervalle, lue au même niveau de couverture (comparer
  des intervalles de même couverture, pas des largeurs brutes).
- **Score de Winkler** — combine couverture et finesse en un seul chiffre par niveau (pénalise à
  la fois un intervalle trop large ET une valeur qui tombe dehors).
- **PIT** (probability integral transform) — où tombe la valeur réelle dans le nuage trié (en
  fraction). Si le modèle est bien calibré, ces valeurs doivent être uniformément réparties
  entre 0 et 1 sur beaucoup de prédictions.
- **MASE** — secondaire, gardé pour comparaison avec l'ancienne synthèse. Erreur du modèle
  divisée par l'erreur de Naive sur les *mêmes* origines.

Tout ça est stocké ligne par ligne (une ligne = un modèle × un actif × une origine × un
horizon) et agrégé par cellule dans `experiments/kpi_probabilistes.json`.

## 6. Trous de couverture trouvés (pas des bugs — vérifiés directement en base)

- **Régime C (natif hebdomadaire) quasi vide pour TSDiff** : sur ~90 lignes possibles par actif,
  89-90 sont marquées `daily_duplicate=1` (donc exclues du "tableau actuel", comme le fait déjà
  le dashboard). Vérifié directement en base — pas un artefact de mon travail. Résultat : le
  test "fréquence B vs C" (§7) n'a jamais assez de données pour conclure, pour aucun des 6
  modèles.
- **TSDiff régime B, horizon W+3, sur BTC-USD/ETH-USD (crypto)** : le modèle de cette cellule est
  entraîné pour prédire au maximum 15 pas en avant (calibré pour des marchés à 5 jours/semaine).
  Or crypto trade 7j/7 : W+3 y correspond à 21 pas, au-delà de la capacité du modèle. 30 lignes
  manquent pour cette raison précise — un plafond hérité du protocole d'origine, que je n'ai pas
  cherché à contourner (ça demanderait un nouvel entraînement avec un horizon différent, donc une
  nouvelle décision de conception, hors périmètre du brief).

## 7. Les tests statistiques poolés (la partie "on ne conclut rien sans test")

Même méthode que l'analyse poolée déjà existante dans ce dépôt (`experiments/pooled_analysis.py`),
réutilisée telle quelle — je n'ai rien réinventé côté statistique, seulement changé la métrique
d'entrée (CRPS empirique au lieu du RMSE, calibration à 3 niveaux au lieu d'1 seul).

**Pourquoi ces précautions :**
- **Bootstrap par blocs** (pas un bootstrap naïf) : les origines se chevauchent dans le temps
  (une prédiction W+2 partage sa cible avec la prédiction W+1 de l'origine suivante) — un
  bootstrap naïf sous-estimerait l'incertitude en traitant ces points comme indépendants.
- **DM-HAC** (Diebold-Mariano avec variance corrigée) : un second test indépendant, croisé avec
  le bootstrap — si les deux ne sont pas d'accord sur la significativité, c'est signalé, pas
  caché.
- **Regroupement par classe d'actif** : ZN=F et TLT (deux façons de parier sur les taux) sont
  moyennés en une série "obligations" avant le test ; pareil pour BTC-USD/ETH-USD en "crypto".
  Sinon on compterait deux fois la même information (actifs corrélés = pas deux preuves
  indépendantes).
- **Correction de Holm** : quand on teste les 6 modèles sur la même question, le seuil de
  significativité est resserré pour ne pas déclarer "significatif" par pur hasard de multiplicité.
- **N effectif toujours affiché** : à côté de "n=1520 lignes", le vrai nombre de blocs
  indépendants (`n_eff`, ex. ~270-300) — c'est CE nombre qui reflète la vraie puissance
  statistique, pas le compte brut de lignes.

**Les 3 questions posées, avec leurs résultats :**

1. **Régime B vs régime C** (entraîné quotidien vs natif hebdomadaire, à horizon égal) : "données
   insuffisantes" pour les 6 modèles — régime C trop vide (§6).
2. **D+7 vs W+1** (même horizon réel, deux façons différentes d'y arriver) : pas de différence
   exploitable pour ARIMA-GARCH/SARIMA/Naive ; D+7 significativement meilleur pour Prophet et
   LSTM ; données insuffisantes pour TSDiff. Puissance faible partout (n_eff ~11-13) — à lire
   avec prudence, c'est signalé dans le rapport.
3. **Calibration à 50/80/95 %** (est-ce que le modèle est significativement mal calibré ?) :
   détail complet dans le rapport PDF §5.1 — c'est le résultat le plus riche, avec TSDiff qui
   ressort comme le seul bien calibré à 50 % mais significativement sous-calibré à 95 %.

## 8. Ce qui n'a PAS été testé (hors périmètre, assumé)

- **Le classement brut entre les 6 modèles n'est pas poolé statistiquement.** TSDiff a un
  protocole d'entraînement différent des 5 autres (entraîné une fois par cellule vs les 5 autres
  qui n'ont subi aucun nouvel entraînement ici) — comparer leurs CRPS directement resterait biaisé
  par cette asymétrie, et aucun test ne corrige ça. C'est écrit noir sur blanc dans le rapport.
- **Pas de nouveaux modèles, actifs ou horizons** — je suis resté strictement sur les cellules
  déjà couvertes par la base actuelle.
- **Pas de ré-entraînement au-delà de ce qui était strictement nécessaire pour TSDiff** — les 5
  autres modèles n'ont subi aucun calcul lourd.

## 9. Où retrouver chaque chose

| Quoi | Où |
|---|---|
| Échantillons bruts (500 par ligne) | `experiments/samples/<ACTIF>_{parametric,tsdiff}.{index.parquet,samples.npz}` |
| KPI par ligne + par cellule, 5 actifs | `experiments/kpi_probabilistes.json` |
| Tests statistiques poolés | `experiments/pooled_analysis_prob.json` |
| Synthèse lisible (PDF) | `experiments/rapport_kpi_probabilistes.pdf` |
| Ce document | `experiments/METHODOLOGIE_kpi_probabilistes.md` |
| Code : échantillonnage paramétrique (5 modèles) | `experiments/generate_samples_parametric.py` |
| Code : échantillonnage natif TSDiff | `experiments/generate_samples_tsdiff.py` |
| Code : calcul des KPI | `experiments/prob_kpi_common.py`, `experiments/compute_prob_kpi_pilot.py` |
| Code : construction de la matrice complète | `experiments/build_kpi_probabilistes.py` |
| Code : tests statistiques poolés | `experiments/pooled_analysis_prob.py` |
