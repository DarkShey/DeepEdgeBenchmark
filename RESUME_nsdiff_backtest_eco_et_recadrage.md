# RÉSUMÉ — NsDiff : valeur économique, re-cadrage 200 tirages, cadrage monthly

*2026-08-07. Synthèse du chantier répondant au brief « NsDiff : valeur économique,
re-cadrage 200 tirages, cadrage monthly ». Ce document est une **vue** de
`NOTE_nsdiff_backtest_eco_et_recadrage.md`, qui reste la source de vérité et
contient le détail complet, les protocoles et les limites.*

---

## 1. Le résultat, en une phrase

**La configuration qu'on déploierait n'est plus battue par ARIMA-GARCH — et elle
ne le bat pas non plus, ni en calibration, ni en argent, ni sur les scénarios de
trajectoires. Le programme se termine sur une parité, pas sur une victoire.**

| Chantier | Question | Réponse |
|---|---|---|
| **A1** re-cadrage | la piste `oos` est-elle encore comparable ? | non — **1 seul modèle sur 7** était concerné (NsDiff) ; base repointée sur 200 tirages |
| **A2** config production | la config qu'on déploierait bat-elle GARCH ? | **elle fait jeu égal** : 0 défaite sur 12 tests poolés, contre 6 sur 12 pour une graine unique |
| **A3-ii** cadence de refit | l'asymétrie de protocole expliquait-elle l'écart ? | **non** — ×24,6 en coût de calcul ne déplace aucun verdict |
| **B** valeur économique | NsDiff apporte-t-il de l'argent au-delà de GARCH ? | **non** — 0 résultat survit à Holm dans les 9 familles de décision |
| **B3** scénarios | la diffusion se différencie-t-elle par la trajectoire ? | **non** — et sur le pricing d'option, GARCH gagne 7 cellules sur 10 |
| **C** monthly | le régime mensuel est-il faisable ? | **NO-GO** pour les 3 voies — mais à une observation près (§4) |

**Ce qui a changé en base** : 2 700 lignes NsDiff repointées de 50 à 200 tirages
(couverture 0,909 → 0,9315), sauvegarde horodatée prise avant écriture. TSDiff
retiré du benchmark. **453 tests passent** (365 avant le chantier, +88 neufs).

---

## 2. Zoom n° 1 — les deux cas qui « ne survivent pas à Holm »

### 2.1 Pourquoi une correction, et laquelle

Le chantier B teste beaucoup : 3 stratégies × 3 horizons × 2 régimes × 3 niveaux
de coût, et pour chacun plusieurs bras. À un seuil de 5 % sans correction, on
attend **un faux positif toutes les 20 cases** — c'est-à-dire, sur une famille de
6 tests, une chance sur quatre d'en voir au moins un « significatif » alors que
rien n'est vrai.

La correction retenue est **Holm-Bonferroni**, déclarée avant les runs. Elle
fonctionne en **descente** : on trie les p-values croissantes, on compare la
plus petite au seuil le plus strict α/m, la deuxième à α/(m−1), etc., et **on
s'arrête au premier échec** — tout ce qui suit tombe avec lui, même si sa propre
p-value passerait son propre seuil.

**Famille déclarée** : les 6 tests poolés globaux (3 horizons × 2 régimes) d'un
même match et d'une même métrique. Seuil le plus strict : 0,05 / 6 = **0,00833**.

Deux familles du chantier B ont produit des résultats bruts significatifs. Ni
l'une ni l'autre ne survit — et les deux échouent **dès le premier rang**.

### 2.2 Cas n° 1 — le bootstrap apparié sur `var_limit`

Stratégie de limite de risque, bras « graine tirée au hasard » contre GARCH.
Deux cellules sur six ressortent significatives en brut.

| Rang | Cellule | p brute | Seuil de son rang | Verdict brut | p ajustée | Verdict Holm |
|---|---|---|---|---|---|---|
| 1 | weekly W+3 | **0,0354** | 0,00833 | NsDiff meilleur | 0,2124 | indistinguable |
| 2 | weekly W+2 | **0,0482** | 0,01000 | NsDiff meilleur | 0,2410 | indistinguable |
| 3 | weekly W+1 | 0,0844 | 0,01250 | indistinguable | 0,3376 | indistinguable |
| 4 | daily W+1 | 0,2834 | 0,01667 | indistinguable | 0,8502 | indistinguable |
| 5 | daily W+2 | 1,0000 | 0,02500 | indistinguable | 1,0000 | indistinguable |
| 6 | daily W+3 | 1,0000 | 0,05000 | indistinguable | 1,0000 | indistinguable |

**Lecture** : la plus petite p-value de la famille vaut 0,0354, contre un seuil
de 0,00833 — un facteur **4,2** au-dessus. La procédure s'arrête au rang 1 et
rien ne passe. La plus petite p ajustée de la famille est **0,21**.

**Et l'ordre de grandeur confirme** : sur les deux cellules significatives,
l'écart moyen de PnL vaut **+3,3 et +4,9 points de base par origine** en faveur
de NsDiff (+2,4 sur la troisième cellule weekly, non significative). Au niveau de coût central, un
aller-retour coûte 10 bps sur SPY et **60 bps sur la crypto**. L'avantage
détecté, à supposer qu'il soit réel, est un **ordre de grandeur sous les frais**.

### 2.3 Cas n° 2 — le test SPA de Hansen sur `var_limit`

Le SPA (`mcs.spa_test`, déjà dans le repo) est **unilatéral** et recentré : il est
plus puissant qu'un bootstrap bilatéral pour détecter un petit avantage
systématique. Il trouve, lui, **trois** cellules — toutes en weekly.

| Rang | Cellule | p SPA brute | Seuil de son rang | p ajustée | Gain moyen par origine |
|---|---|---|---|---|---|
| 1 | weekly W+1 | **0,0332** | 0,00833 | 0,1992 | +2,2 bps |
| 2 | weekly W+2 | **0,0406** | 0,01000 | 0,2030 | +3,2 bps |
| 3 | weekly W+3 | **0,0482** | 0,01250 | 0,2030 | +4,3 bps |
| 4 | daily W+2 | 0,3678 | 0,01667 | 1,0000 | +0,9 bps |
| 5 | daily W+3 | 0,3962 | 0,02500 | 1,0000 | +0,7 bps |
| 6 | daily W+1 | 1,0000 | 0,05000 | 1,0000 | −1,3 bps |

Même mécanique : plus petite p = 0,0332 contre un seuil de 0,00833, arrêt au
rang 1, **0 survivant sur 6**.

**Le SPA a été soumis à Holm comme le reste, et c'est un choix.** L'en exempter
au motif qu'il est « le test le plus adapté » reviendrait à se réserver le test
le plus favorable après avoir vu les résultats — exactement ce que la correction
existe pour empêcher.

### 2.4 Ce que « ne survit pas à Holm » veut dire, et ne veut pas dire

- **Ce que ça veut dire** : le motif observé est compatible avec le hasard, une
  fois pris en compte le nombre de tests effectués. Il ne fonde pas une décision.
- **Ce que ça ne veut PAS dire** : que l'effet est nul. À `effective_n` = 30, la
  puissance est faible ; un vrai avantage de 3 bps par origine serait indétectable
  ici quoi qu'il arrive.
- **Ce qui tranche vraiment, dans ce cas précis** : la **taille de l'effet**, pas
  la p-value. 2 à 5 bps par origine face à 10–60 bps de frais aller-retour : le
  signal, réel ou non, n'est pas exploitable.

**Bilan des 9 familles de décision du chantier B** : 5 rejets bruts au total,
**0 survivant**.

---

## 3. Zoom n° 2 — « ne bat pas acheter-et-garder » : ce que ça signifie exactement

### 3.1 Ce qui a été testé

24 tests appariés (2 stratégies actives × 3 horizons × 2 régimes × 2 modèles),
bootstrap par blocs sur les différences de PnL par origine. **Aucun ne rejette
l'égalité** : p de **0,25 à 1,00**. « Ne bat pas » signifie donc littéralement
*« aucun test ne permet de conclure à une différence de PnL »* — et non
*« fait moins bien »*. La nuance est importante, parce que le tableau descriptif
raconte autre chose.

### 3.2 Le tableau descriptif — et il n'est pas défavorable

Niveau de coût central, bras NsDiff-ensemble contre acheter-et-garder :

| Cellule | Stratégie | PnL stratégie | PnL B&H | Sharpe stratégie | Sharpe B&H | Drawdown strat. | Drawdown B&H |
|---|---|---|---|---|---|---|---|
| weekly W+1 | `var_limit` | −0,047 | −0,230 | −0,40 | −0,49 | **−0,103** | −0,503 |
| daily W+1 | `var_limit` | −0,067 | −0,230 | −0,48 | −0,49 | **−0,133** | −0,503 |
| weekly W+2 | `var_limit` | **+0,012** | −0,073 | **+0,06** | −0,07 | **−0,127** | −0,796 |
| daily W+2 | `var_limit` | −0,002 | −0,073 | −0,01 | −0,07 | **−0,142** | −0,796 |
| weekly W+3 | `var_limit` | +0,056 | +0,105 | **+0,23** | +0,07 | **−0,147** | −1,067 |
| daily W+3 | `var_limit` | +0,032 | +0,105 | **+0,11** | +0,07 | **−0,175** | −1,067 |

Sur le papier, `var_limit` a un **meilleur Sharpe que B&H dans les 6 cellules**
et un **drawdown 3,8 à 7,3 fois plus faible**. Pourquoi n'est-ce pas présenté
comme une victoire ?

### 3.3 Les trois raisons de ne pas surinterpréter ce tableau

**(a) La réduction de drawdown est en grande partie mécanique.** `var_limit`
dimensionne sa position pour que la VaR à 2,5 % vaille 3 % du capital : elle
n'est donc **pas pleinement investie**. Exposition moyenne |w| :

| Cellule | `var_limit` | `inverse_width` | acheter-et-garder |
|---|---|---|---|
| W+1 | 0,60 – 0,66 | 0,84 – 0,87 | 1,00 |
| W+2 | 0,49 – 0,55 | 0,85 – 0,88 | 1,00 |
| W+3 | 0,44 – 0,49 | 0,85 – 0,88 | 1,00 |

Et le détail par actif est encore plus net (`var_limit`, weekly W+1) : sur la
crypto, l'exposition tombe à **0,16 sur ETH** et **0,23 sur BTC**, contre 0,81
sur SPY et 1,00 sur ZN=F. Or c'est précisément la crypto qui a chuté sur la
période (B&H : −0,46 sur BTC, −0,55 sur ETH). **Être trois quarts hors du marché
sur l'actif qui s'effondre réduit mécaniquement la perte et le drawdown** — ce
n'est pas la preuve d'un signal, c'est la conséquence arithmétique d'une
exposition plus faible.

**(b) Le Sharpe, lui, est sans échelle — mais il n'est pas la quantité testée.**
Le Sharpe corrige de l'exposition, et il reste favorable à `var_limit`. C'est le
point le plus solide en faveur de la stratégie. Mais le test porte sur les
**différences de PnL par origine**, et à `effective_n` = 30, il n'a pas la
puissance de certifier un écart de Sharpe de cet ordre. Affirmer « `var_limit`
bat B&H » sur la base d'un Sharpe descriptif non testé serait exactement le genre
de raccourci que le reste du programme s'interdit.

**(c) `inverse_width` ne bénéficie d'aucune de ces excuses — et détruit de la
valeur.** Elle est investie à ~85 %, donc comparable à B&H, et malgré cela :
**11 séries de PnL négatives sur 12**, **8 pires que B&H sur 12**, un taux de
gain de 42–47 % (soit sous le pile ou face), et un Sharpe négatif dans 6 cellules
sur 6. Le proxy de confiance « 1/largeur du PI » n'est informatif **ni pour
NsDiff ni pour GARCH**. C'est le résultat le plus net du chantier B, et il est
négatif pour les deux modèles.

### 3.4 La formulation juste

> Sur la période et le panier testés, aucune des stratégies dérivées des
> intervalles ne dégage un PnL statistiquement distinguable de celui d'acheter et
> garder. `var_limit` présente un profil risque/rendement descriptivement
> meilleur, largement attribuable à une exposition moyenne de 44–66 % ;
> `inverse_width` détruit de la valeur à exposition comparable. **Aucun des deux
> modèles ne se distingue de l'autre sur aucune de ces stratégies.**

### 3.5 Le cas à part : la famille 3 ne prend jamais position

La troisième famille du brief — « prendre position seulement si le PI exclut le
rendement nul » — donne un résultat en apparence dégénéré et en réalité très
informatif : sur **2 700 origines**, pour **les deux modèles**, aux trois
horizons, **zéro position ouverte**. Un intervalle à 95 % contient *toujours* le
prix courant à ces horizons.

La question du brief — « des intervalles honnêtes filtrent-ils mieux les faux
signaux ? » — reçoit donc une réponse nette : **à 95 %, aucun des deux modèles
n'émet jamais de signal**. Rejouer à un niveau plus étroit exigerait de choisir
un seuil *après* avoir vu ce résultat ; non fait, délibérément.

---

## 4. Zoom n° 3 — le NO-GO monthly, critère par critère

### 4.1 Les deux critères, déclarés avant tout résultat

1. **Couverture** : la couverture observée à 95 % doit tomber dans
   **[0,90 ; 0,98]** — à **chacun** des trois horizons, pas en moyenne ;
2. **Winkler** : le score composite ne doit pas être **significativement pire**
   que celui de la baseline `garch_monthly` (bootstrap par blocs, 5 %).

Une voie qui échoue à l'un des deux est NO-GO. Les deux critères ont été fixés
dans le module avant le premier fit.

### 4.2 Le problème de fond : le nombre de fenêtres, pas d'observations

| Actif | Quotidiennes | Hebdo | **Mensuelles** | Fenêtres d'entraînement (seq_len = 30) |
|---|---|---|---|---|
| SPY | 2 905 | 604 | 139 | **70** (×8,0 de moins qu'en hebdo) |
| BTC-USD | 4 222 | 604 | 139 | 70 (×8,0) |
| ZN=F | 2 905 | 604 | 139 | 70 (×8,0) |
| ETH-USD | 3 179 | 455 | 105 | **36** (×11,4) |
| TLT | 2 121 | 441 | 102 | **33** (×12,0) |

À `seq_len` = 30, le lookback consomme 30 des ~100 rendements disponibles. Sur
ETH et TLT, un modèle d'incertitude voit littéralement **une trentaine
d'exemples**. Le budget d'entraînement devient le levier dominant — et le sweep
le confirme brutalement : à 40 époques, la voie native produit une bande de
**81 % du prix** et couvre 100 % ; à 160 époques, **32 %** et 94 %. Sous-entraîné,
le modèle ne sait rien faire d'autre qu'élargir.

*Note de protocole* : un premier passage sur (20, 40, 80) époques a placé
l'optimum **au bord** de la grille. Une grille dont l'argmin est sur sa borne est
une troncature, pas une sélection : elle a été élargie **une fois** à
(20, 40, 80, 160, 320), sur la **validation**, avant tout regard sur le test.

### 4.3 Le verdict, voie par voie

| Voie | Config retenue | Cov95 | Largeur (% prix) | Winkler | RMSE | CRPS | Verdict |
|---|---|---|---|---|---|---|---|
| `garch_monthly` (baseline) | refit / origine | 0,944 | 22,00 | **152,7** | **31,17** | — | — |
| `monthly_native` | seq_len 12, 160 ép. | 0,926 | 22,67 | 154,5 | 31,34 | **17,73** | **NO-GO** |
| `weekly_propagated` | seq_len 30, 160 ép. | 0,898 | 21,79 | 170,9 | 31,42 | 18,83 | **NO-GO** |
| `synthetic_augmented` | seq_len 30, 40 ép. | 0,852 | 20,57 | 214,0 | 34,48 | 19,27 | **NO-GO** |

Détail horizon par horizon (✓ = critère tenu, ✗ = échoué) :

| Voie | M+1 | M+2 | M+3 |
|---|---|---|---|
| `monthly_native` | cov 0,917 ✓ — Winkler ✓ | cov **0,889 ✗** — Winkler ✓ | cov 0,972 ✓ — Winkler ✓ |
| `weekly_propagated` | cov **0,861 ✗** — Winkler ✓ | cov **0,861 ✗** — Winkler ✓ | cov 0,972 ✓ — Winkler ✓ |
| `synthetic_augmented` | cov **0,861 ✗** — Winkler ✓ | cov **0,861 ✗** — Winkler **✗** (p = 0,0074) | cov **0,833 ✗** — Winkler **✗** (p = 0,0006) |

### 4.4 Les trois NO-GO ne se valent pas du tout

**`monthly_native` échoue d'une seule observation.** Sa seule case hors bande est
M+2 à 0,889 contre une borne à 0,90. Sur 36 origines, cela fait **32 succès au
lieu de 33** : une observation. Sur tous les autres axes, la voie native fait jeu
égal avec GARCH-monthly — Winkler 154,5 contre 152,7, RMSE 31,34 contre 31,17, et
**jamais significativement pire à aucun horizon** (p = 0,899 / 0,412 / 1,000).

Le critère a été fixé a priori et il est appliqué tel quel — mais présenter ce
NO-GO comme un rejet net serait malhonnête. **L'énoncé exact est** : *à
`effective_n` = 12, le pilote ne permet pas de conclure au go, et il n'en est pas
loin.*

**`weekly_propagated` échoue vraiment.** Propager le nuage hebdomadaire jusqu'à la
cible mensuelle sous-couvre franchement à M+1 et M+2 (0,861 des deux côtés) et
paie **12 % de Winkler de plus** que la voie native. Le modèle hebdomadaire est
entraîné à un horizon de 3 semaines et se retrouve étiré à 13 pas : il ne
transporte pas correctement l'incertitude sur cette distance. Cette voie est
**dominée par la voie native sur les quatre métriques** — à abandonner.

**`synthetic_augmented` est la seule à dégrader activement.** Elle est
significativement **pire** que GARCH-monthly à M+2 et M+3, sous-couvre partout
(0,833 – 0,861) et a le plus mauvais RMSE du lot. L'explication est structurelle,
et elle était déclarée dans le module **avant** le run : une série KernelSynth est
tirée d'un processus gaussien à covariance fixe. Elle apporte de la **diversité
de formes** — tendances, cycles, ruptures de corrélation — mais elle est
**homoscédastique par construction** : ni queues lourdes, ni clustering de
volatilité, ni asymétrie. Entraîner un modèle d'*incertitude* sur 5 fenêtres
synthétiques homoscédastiques pour 1 fenêtre réelle **lui apprend à ne pas voir
la volatilité changer**.

La condition posée par le brief (« si (i) sous-entraîne ») était bien remplie ;
**le remède proposé ne marche pas** — et il échoue pour une raison qu'on peut
nommer, pas par accident.

### 4.5 Conséquence et suite

Le brief déclarait un critère d'arrêt : *« si B conclut "aucune valeur économique
ajoutée, aucune différenciation scénarios", le chantier C perd sa justification »*.
**B conclut exactement cela.** Le chantier C a néanmoins été mené jusqu'à son
verdict — il tournait en parallèle, et une étude de faisabilité arrêtée avant sa
conclusion ne vaut rien. Les deux convergent : **le mensuel sort du périmètre
NsDiff.**

**Si on y revenait un jour**, par rendement attendu décroissant :

- refaire le pilote sur les **5 actifs** — le NO-GO tient à une observation sur un
  seul actif, cinq pilotes donneraient une base de décision réelle ;
- élargir encore la grille d'époques vers le haut : la courbe de CRPS n'a été
  explorée qu'une fois au-delà de 80 ;
- abandonner `weekly_propagated` (dominée) et remplacer KernelSynth par un
  générateur à **volatilité stochastique**, seule façon de corriger le défaut
  identifié en §4.4.

---

## 5. Ce qu'il faut retenir

1. **Le re-cadrage était nécessaire mais petit** : un modèle sur sept, pas sept.
   Le chiffrer avant de l'exécuter a évité un chantier inutile.
2. **L'ensemble multi-graines est la seule amélioration gratuite du programme** :
   il fait passer NsDiff de 6 défaites sur 12 à 0 sur 12 contre GARCH, sans un
   seul fit supplémentaire.
3. **Les explications alternatives ont été éliminées une à une** : ni la fraîcheur
   du fit (A3-ii, ×24,6 en coût pour rien), ni la structure jointe des
   trajectoires (B3) n'expliquent l'absence d'avantage.
4. **La valeur économique différentielle n'existe pas à ce jour** — et le peu qui
   ressort en brut est un ordre de grandeur sous les frais de transaction.
5. **Le mensuel est hors périmètre**, mais la voie native mérite d'être rejugée
   sur cinq actifs avant d'être classée définitivement.

---

*Source de vérité : `NOTE_nsdiff_backtest_eco_et_recadrage.md`. Artefacts JSON
reproductibles dans `experiments/`. Tout chiffre de ce résumé a été revérifié
contre les JSON produits par les runs.*
