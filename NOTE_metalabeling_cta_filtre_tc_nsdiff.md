# NOTE — Méta-labeling artisanal : le CTA filtré par la taxonomie TC de NsDiff

2026-08-11. Réponse au `BRIEF_metalabeling_cta_filtre_tc_nsdiff.md`.

**Verdict : le critère d'arrêt 1 est atteint, sur ses deux branches. Le méta-labeling
artisanal est clos, et la version apprise est interdite en suite directe.**

- Famille de Holm (m = 2) : **0 rejet**. F1 vs C0 p = 0,308 (ajusté 0,616) ; F2 vs C0
  p = 0,563 (ajusté 0,616).
- Placebo à couverture égale : **F2 ne le bat pas** — 40 % des vetos aléatoires font
  mieux que F2.

L'hypothèse primaire était conjonctive (F2 bat C0 **et** F2 bat le placebo). Aucune des
deux conditions n'est remplie.

## 1. Ce que l'analyse de puissance disait avant les runs

Le brief demandait de chiffrer l'écart détectable **avant** de lancer, et de le dire si
la couverture du veto était trop faible. Elle l'était — c'est le fait central de ce
chantier, et il vient d'une propriété de la géométrie, pas d'un choix d'implémentation.

**La taxonomie TC s'effondre sur un seul état à l'horizon hebdomadaire.** Sur les 2 380
cellules (340 origines × 7 actifs) :

| État TC | NsDiff | ARIMA-GARCH |
|---|---|---|
| Sideways | **2 322 (97,6 %)** | **2 348 (98,7 %)** |
| Bull-Calm | 43 | 14 |
| Bear-Calm | 15 | 18 |
| Bull-Stress / Bear-Stress | **0** | **0** |

Zéro état Stress, comme pré-déclaré (`drift ≪ largeur`). Mais l'attente disait « le
filtre vivra surtout sur Calm et Sideways » : en réalité il vit **presque uniquement sur
Sideways**, parce que `sideways_d1` exige `|predicted − ref| ≤ 0,10 · W`, et qu'à une
semaine la dérive franchit ce dixième de largeur dans 2,4 % des cas seulement. C'est le
même fait mécanique que la famille 3 close, mesuré ici sous un autre angle.

Conséquences directes, calculées avant lecture des verdicts :

| Bras | Origines modifiées vs C0 | Écart observé | MDE (80 %, effective_n = 113) | Détectable ? |
|---|---|---|---|---|
| F1 | 29 / 340 | +0,65 bps | 3,2 bps | **non** |
| F2 | 340 / 340 | −7,44 bps | 67,6 bps | **non** |
| G1 | 12 / 340 | −0,49 bps | 2,5 bps | non |
| G2 | 340 / 340 | −7,59 bps | 67,4 bps | non |

Les deux filtres sont hors de portée, mais **pour deux raisons opposées**, et c'est
important de ne pas les confondre :

- **F1 ne fait presque rien.** Il ne vétote que 33 cellules sur 2 380 (1,4 %) et ne
  modifie le portefeuille que sur 29 origines sur 340. Son écart (+0,65 bps) est à
  1 écart-type ; il faudrait 3,2 bps pour trancher.
- **F2 ne fait presque plus rien du tout.** Il ne prend que **25 trades sur 2 380**
  (1,1 %). Ce n'est pas « le CTA moins ses mauvais trades », c'est une abstention quasi
  totale — un portefeuille en liquidité 98,9 % du temps. Sa variance par origine explose
  (MDE 67,6 bps) alors que le PnL total de C0 vaut 7,4 bps : un ordre de grandeur au-delà
  de ce que la fenêtre peut résoudre.

## 2. Les bras, net de frais (niveau central, roulement inclus)

| Bras | PnL moyen / origine | Trades pris | Sharpe annualisé |
|---|---|---|---|
| C0 (CTA seul) | +7,41 bps | 2 380 / 2 380 | +0,208 |
| **F1** (filtre faible) | +8,06 bps | 2 347 (98,6 %) | +0,226 |
| **F2** (filtre strict) | **−0,03 bps** | **25 (1,1 %)** | −0,025 |
| G1 (contrôle GARCH) | +6,92 bps | 2 367 (99,5 %) | +0,194 |
| G2 (contrôle GARCH) | −0,17 bps | 19 (0,8 %) | −0,034 |
| Buy & Hold | +33,19 bps | — | — |

Le CTA reste très en dessous de B&H (−25,78 bps/origine, p = 0,18) — résultat connu,
rappelé ici seulement comme toile de fond.

**Attribution : ce n'est pas NsDiff.** Le brief avait pré-déclaré que si le veto Sideways
était le seul mécanisme actif, l'attribution se jouerait sur G2. Elle s'y joue, et la
réponse est nette : la géométrie GARCH produit 98,7 % de Sideways contre 97,6 % pour
NsDiff, G2 prend 19 trades là où F2 en prend 25, et les deux bras finissent à ≈ 0. **La
largeur de n'importe quel intervalle fait la même chose.** Rien dans ce filtre n'est
propre à la variance apprise de NsDiff.

## 3. Lecture sélective — le résultat le plus intéressant, et il est descriptif

C'est la lecture qui distingue « l'idée est fausse » de « l'idée n'est pas testable à
cette couverture ». Elle penche pour la seconde.

**F1 retire bel et bien de mauvais trades.** Ses 33 trades vetoés (contradiction
Bull × short ou Bear × long) valent en moyenne **−46,58 bps**, contre **+8,17 bps** pour
les 2 347 pris ; leur hit-rate est de 42,4 % contre 50,0 %. Le signe est celui qu'espère
le méta-labeling : quand NsDiff contredit franchement le CTA, le trade CTA est en moyenne
mauvais. Par état : les 25 contradictions Bull-Calm valent −50,78 bps, les 8
contradictions Bear-Calm −33,45 bps.

**Mais 33 trades sur 2 380 ne déplacent pas un portefeuille.** L'effet agrégé est
+0,65 bps par origine, p = 0,308, très en dessous du seuil détectable. Le couple
(couverture, qualité) est donc : *qualité plausible, couverture dérisoire*. Rapporter la
qualité seule aurait donné l'illusion d'un filtre qui marche.

**F2 vétote exactement ce qu'il ne faut pas.** Ses 2 355 trades vetoés valent en moyenne
**+7,52 bps** — c'est le gros du PnL du CTA — et les 25 trades qu'il retient valent
**−3,10 bps** malgré un hit-rate de 60 % (quelques pertes lourdes, dont 7 concordances
Bear-Calm à −82,14 bps). En exigeant la concordance, F2 jette le Sideways, où le CTA
gagne l'essentiel du peu qu'il gagne.

Par régime, F1 suit C0 de près (2020 : +25,1 vs +23,1 ; 2022 : −23,3 vs −23,5 ;
2024-2026 : +12,3 vs +11,3) — cohérent avec 29 origines modifiées. F2 est plat partout
(−0,7 / +0,1 / −0,2), ce qui est la signature d'un bras qui ne trade pas.

## 4. Trois choix d'exécution, déclarés

**Quel « CTA corrigé, direction Hull » ?** Le brief demande « direction Hull,
v2-2026-08-08 ». Aucun artefact ne porte exactement cette étiquette : la version corrigée
du moteur (`deita_cta_signal_v4/`) retient la **conviction**, pas la direction Hull.
J'ai retenu `deita_cta_signal_own/`, et c'est bien le signal corrigé pour la direction
Hull : le bug 1 (conviction-carré) vit dans `_conviction_level`, que `trend_direction` —
univariée — ne traverse jamais ; le bug 2 (calendrier ffill) est précisément ce
qu'évite `calendar="own"`, comme le code le dit en toutes lettres ; le bug 3 était un
look-ahead du harnais de validation DEITA, pas du signal. Sur les 7 actifs de la grille
oos2020, c'est l'artefact exact.

**Précédence des états TC — lacune du brief, comblée avant les runs.** Le brief liste les
états comme exclusifs. Ils ne le sont pas : Stress et Calm sont étanches par construction
(garde-fous explicites dans `bull_calm_d1`/`bear_calm_d1`), **mais Sideways recouvre
Calm**. Précédence retenue : **Stress > Sideways > Calm**, au motif que Sideways signifie
« dérive négligeable devant la largeur », donc « NsDiff n'exprime pas de direction » — le
lire comme Bull/Bear serait prendre du bruit pour une direction. L'ordre inverse aurait
vidé la ligne Sideways de la matrice et rendu F1 ≈ F2, donc le brief sans objet. Ce choix
est favorable à l'hypothèse testée, pas défavorable : il donne au filtre sa seule
matière.

**Placebo.** 100 tirages, graine 20260810, veto sans remise portant sur **exactement** le
même nombre de cellules que F2, actif par actif. F2 = −0,03 bps ; médiane des placebos
−0,21 bps, intervalle [p05 −1,10 ; p95 +1,26]. 40 % des tirages aléatoires font mieux.

## 5. Limites

- **Fenêtre inchangée** (2020-2026, 340 origines, effective_n = 113) — l'interdit tient.
- Taille fixe |w| = 1 : le sizing n'était pas la question, et il n'a pas été touché.
- La lecture sélective est **descriptive** : les sous-ensembles pris/vetoés n'ont ni la
  même taille ni les mêmes jours, aucun test n'y est appliqué.
- G1/G2, le par-actif et le par-état sont **exploratoires**, étiquetés tels quels, hors
  famille de Holm.
- `fee_bps` de la géométrie TC n'intervient pas ici : les états sont dérivés de la seule
  géométrie (ref, predicted, pi_low, pi_high), et le PnL passe par `real_fees` + roulement.

## 6. Ce que ce résultat ferme

Critère d'arrêt 1, sur ses deux branches (Holm sans rejet **et** placebo non battu) :
**le méta-labeling artisanal est clos**. La version apprise — modèle secondaire entraîné,
Lopez de Prado complet — est **explicitement interdite en suite directe**, comme le brief
l'engageait : elle aurait moins de données par état (43 Bull-Calm et 15 Bear-Calm sur
6,5 ans) et plus de paramètres ; ce serait chercher dans un sous-espace plus riche ce que
le sous-espace simple vient de refuser.

Aucun nouveau filtre, seuil ou matrice n'a été essayé après lecture des résultats : les
deux filtres de ce brief sont les seuls qui aient tourné.

La raison de fond, en une phrase : à l'horizon d'une semaine, la dérive prédite est si
petite devant la largeur de l'intervalle que la taxonomie TC ne distingue presque rien —
97,6 % des cellules tombent dans le même état, et un intervalle GARCH fait exactement
pareil. Ce n'est pas un échec de NsDiff, c'est l'absence d'information exploitable dans
la géométrie prévision/PI à cet horizon.

## 7. Reproduire

```bash
python experiments/metalabel_cta_tc.py     # lecture seule sur tracking.db
```

Sortie : `experiments/metalabel_cta_tc.json`. `tracking.db` est ouvert en mode
`file:...?mode=ro` — aucune écriture possible. Signal gelé `deita_cta_signal_own/`,
grille `oos2020` W+1, frais `real_fees` niveau central avec roulement, bootstrap par
blocs de longueur 3, graine 42.

pytest : 795 passed, 1 skipped avant ce chantier → 823 passed, 1 skipped après (28
nouveaux tests, dont la matrice de décision case par case et le placebo à couverture
exactement égale).
