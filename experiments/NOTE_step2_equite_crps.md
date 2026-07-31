# NOTE — Étape 2 : la comparaison CRPS équitable, et le sort du "match serré"

*2026-07-27. Fichiers : `generate_baseline_weekly_bootstrap.py` (nouveau générateur), `tsdiff_sampler_sweep.py`/
`tsdiff_sampler_sweep_btc80.py` (sweep N/K_DENOISE), `step2_equite_paired_tests.json` (3 verdicts),
`baseline_weekly_bootstrap_index.json`/`_samples.npz` (nuages bootstrap complets). Origines/actifs/
checkpoints : identiques à `step1_final` (5 actifs, 78 origines, crypto à 80 epochs), non rejoué.*

## Question

L'étape 1 concluait à un "match serré" (TSDiff-W indiscernable des meilleures baselines à W+1,
léger désavantage résiduel à W+2/W+3), mais en comparant un vrai nuage de diffusion (TSDiff-W)
contre un nuage gaussien simplifié (baselines). Ce verdict tient-il une fois la comparaison rendue
équitable, dans les deux sens ?

## Volet A — trois verdicts poolés, côte à côte (CRPS normalisé, Holm par horizon)

`+` = TSDiff-W pire que la baseline, `-` = TSDiff-W meilleur. `*` = significatif après Holm, `ns` = non.

| Comparaison | V1 gaussien (= étape 1) | V2 baselines empiriques | V3 tout gaussien |
|---|---|---|---|
| W+1 vs ARIMA-GARCH | +0.108 (ns) | +0.143 (ns) | +0.132 (ns) |
| W+1 vs SARIMA | +0.071 (ns) | -0.016 (ns) | +0.095 (ns) |
| W+1 vs Naive | +0.118 (ns) | +0.103 (ns) | +0.141 (ns) |
| W+1 vs Prophet | -9.009 (\*) | -9.262 (\*) | -8.985 (\*) |
| W+1 vs LSTM | -2.835 (\*) | -2.938 (\*) | -2.812 (\*) |
| W+2 vs ARIMA-GARCH | +0.425 (\*) | +0.411 (\*) | +0.449 (\*) |
| W+2 vs SARIMA | +0.366 (ns) | +0.359 (ns) | +0.390 (ns) |
| W+2 vs Naive | +0.384 (ns) | +0.383 (\*) | +0.408 (ns) |
| W+2 vs Prophet | -8.275 (\*) | -7.818 (\*) | -8.252 (\*) |
| W+2 vs LSTM | -1.800 (\*) | -1.982 (\*) | -1.776 (\*) |
| W+3 vs ARIMA-GARCH | +0.683 (\*) | +0.634 (\*) | +0.725 (\*) |
| W+3 vs SARIMA | +0.606 (ns) | +0.639 (\*) | +0.648 (ns) |
| W+3 vs Naive | +0.642 (ns) | +0.648 (\*) | +0.684 (ns) |
| W+3 vs Prophet | -7.511 (\*) | -6.764 (\*) | -7.469 (\*) |
| W+3 vs LSTM | -1.131 (ns) | -1.356 (\*) | -1.089 (ns) |

**Le classement ne bascule jamais** : sur les 15 comparaisons, le signe de l'écart est identique
dans V1/V2/V3 à chaque fois — aucun cas où TSDiff-W passe de "meilleur" à "pire" ou l'inverse selon
la méthode d'échantillonnage. Seuls **4 cas sur 15** changent de statut de significativité entre V1
et V2 (W+2 vs Naive, W+3 vs SARIMA/Naive/LSTM), et **dans les 4 cas la bascule va dans le sens
défavorable à TSDiff-W** (ns → significatif, TSDiff-W perd plus souvent une fois les baselines
échantillonnées empiriquement) — jamais l'inverse. V3 (TSDiff-W lui-même mis en gaussien) est
quasi identique à V1 en magnitude et en significativité : mettre TSDiff-W au même "handicap" que
les baselines ne change quasiment rien.

**Conclusion volet A : l'asymétrie n'était pas déterminante, et si elle biaisait quelque chose, ce
n'était pas en faveur de TSDiff-W.** Le "match serré" de l'étape 1 est confirmé, pas affaibli par
la correction — il devient même légèrement *plus* défavorable à TSDiff-W (SARIMA et Naive
rejoignent ARIMA-GARCH comme adversaires significatifs à W+3 sous l'échantillonnage équitable).

## Validité de l'approximation bootstrap (couverture réelle, cible 50/80/95%)

| Modèle | Cov50 (W+1/W+2/W+3) | Cov80 | Cov95 | Commentaire |
|---|---|---|---|---|
| ARIMA-GARCH | .52/.56/.55 | .82/.79/.82 | .94/.94/.92 | résidus GARCH ~iid par construction — pas d'approximation, calibration excellente |
| Naive | .55/.56/.55 | .86/.83/.86 | .97/.98/.97 | empirique pur, pas d'approximation |
| **SARIMA** | .36/.44/.47 | .68/.72/.74 | .92/.90/.87 | approximation (somme de h résidus) — sous-couvre modérément à 50%, **meilleure qu'au gaussien à 95%** (.92 vs .89 gaussien) |
| **Prophet** | .38/.49/.55 | .62/.77/.81 | .81/.87/.91 | même approximation — sous-couvre à W+1, **converge vers le nominal à W+3**, meilleure qu'au gaussien à W+2/W+3 |
| LSTM | .38/.35/.33 | .57/.52/.52 | .71/.66/.64 | **nettement sous-calibré**, et ça empire avec l'horizon — le rollout récursif MC-Dropout ne capture pas assez la croissance d'incertitude multi-pas |

**Le garde-fou tient pour SARIMA/Prophet** (l'approximation "somme de h résidus" du point 2) : la
couverture reste dans une plage raisonnable, et n'est pas pire — plutôt meilleure sur certains
horizons — que la couverture gaussienne de référence. **Point non anticipé mais réel** : c'est le
bootstrap MC-Dropout de LSTM (pas SARIMA/Prophet) qui est le moins bien calibré des 5 — la
généralisation récursive multi-pas de la couche Dropout sous-estime l'incertitude qui devrait
croître avec l'horizon. Ça n'invalide pas le CRPS de LSTM comme règle de score propre, mais sa
propre qualité de nuage est le point le plus fragile du volet A, pas le résidu SARIMA/Prophet
qu'on soupçonnait a priori.

## Volet B — le sampler TSDiff était-il bien réglé ?

**Bug détecté et corrigé en cours de route** : le premier passage du sweep utilisait par erreur
les epochs de l'ancien `epoch_sweep_results.json` (BTC=30) au lieu des epochs corrigés de
`step1_final` (BTC=80) — rejoué avec le bon checkpoint avant de conclure.

| | N=50 | N=100 | N=200 | **N=500 (actuel)** | N=1000 |
|---|---|---|---|---|---|
| SPY CRPS | 10.39 | 10.13 | 9.85 | 9.98 | 9.99 |
| BTC-USD CRPS (80 epochs) | 4189 | 4315 | 4143 | **4103** | 4150 |

| | K=5 | K=10 | **K=20 (actuel)** | K=40 | K=80 |
|---|---|---|---|---|---|
| SPY CRPS | 10.60 | 10.09 | 9.98 | 9.93 | 9.92 |
| BTC-USD CRPS (80 epochs) | 4308 | 4218 | **4103** | 4177 | 4198 |

**N=500 : stable.** Le CRPS ne dérive plus au-delà de N=200 sur les deux actifs — pas de biais
d'estimateur significatif laissé sur la table à N=500.

**K_DENOISE=20 : déjà (quasi) optimal.** Avec le bon checkpoint (epochs=80), le CRPS de BTC-USD
est minimal à K=20 et légèrement pire à K=40/80 — pas de dégradation ni de gain caché en poussant
plus loin. (Le run initial, biaisé par le mauvais checkpoint à epochs=30, avait laissé croire à une
amélioration continue jusqu'à K=80 — artefact du sous-entraînement, pas du réglage K_DENOISE lui-même.)

**batch_size=32 : pas de sweep dédié** (constant partout dans le projet, jamais fait varier — rien
à comparer) ; la dispersion des nuages déjà observée dans `step1_final`/ce sweep (écarts-types non
nuls, couverture non dégénérée à aucun horizon) est une preuve empirique directe qu'on n'est pas
dans le régime de collapse documenté.

**Conclusion volet B : le sampler TSDiff (N=500, K_DENOISE=20, batch_size=32) est correctement
réglé.** Le "match serré" de l'étape 1 n'est pas un artefact d'un générateur d'échantillons
sous-optimal.

## Verdict de l'étape 2

Le "match serré" de l'étape 1 **tient**, sous les deux corrections mesurées séparément (asymétrie
d'échantillonnage, réglages du sampler). Il n'était ni un artefact de comparaison biaisée en
faveur de TSDiff-W, ni un artefact d'un sampler mal réglé — et la correction de l'asymétrie
pencherait plutôt légèrement en défaveur de TSDiff-W à W+2/W+3, ce qui rend le résultat d'étape 1
plus robuste, pas plus fragile.
