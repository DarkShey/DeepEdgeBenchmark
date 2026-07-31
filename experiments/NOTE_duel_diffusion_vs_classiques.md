# NOTE — Le duel diffusion vs classiques (BRIEF_unification_protocole_duel.md + BRIEF_baselines_fortes.md + BRIEF_multigraines.md)

*2026-07-30. Fichiers : `duel_origins.py` (§2.1), `duel_sampling_adapters.py` (§2.2, GARCH-Student/
GJR ajouté par BRIEF_baselines_fortes.md), `crps_metrics.py::crps_fair` (§2.3), `duel_pairwise_tests.py`
(§2.4), `mcs.py` (§2.5), `duel_backtest.py` (assemblage + sélection du spec GARCH), `duel_multiseed.py`
+ `duel_global_training.py` (boucle multi-graines + entraînement global, BRIEF_multigraines.md,
exécutés ici) + `duel_backtest.json` (le livrable brut, désormais organisé par graine + agrégats).
Run réel : 5 actifs (SPY, BTC-USD, ETH-USD, ZN=F, TLT), **S = 5 graines (42, 43, 44, 45, 46)**,
`n_val=12`, `n_test=30`, `m=500`, `k_denoise=20`, mêmes origines pour toutes les graines
(`--end 2026-07-29` explicite). Comparaison entraînement global menée sur un **sous-ensemble déclaré
de 2 graines (42, 43)** — coût du entraînement pooled + évaluation sur les 5 actifs jugé prohibitif
au-delà pour ce chantier, cf. §5 du brief.*

## Grille de conformité — protocole du duel (§4 de BRIEF_unification_protocole_duel.md)

| Point | Critère (niveau 1) | Statut |
|---|---|---|
| Estimateur CRPS | fair CRPS (Ferro 2014), `m` identique ; pas d'approximation par quantiles | **conforme** |
| Nombre de tirages `m` | 500, strictement identique pour les 6 modèles à chaque origine | **conforme** |
| Splits temporels | rolling origin (`three_way_split`) + embargo=2 semaines | **conforme** |
| Ré-estimation | identique entre les 6 modèles : gelé à T0 pour tous | **conforme** |
| Échantillonnage classiques | GARCH-Student/GJR simulé, SARIMAX `.simulate()`, Prophet `predictive_samples()`, LSTM MC-Dropout, Naive `random_walk_samples` | **conforme** |
| Tests | `dm_hac_test` + bootstrap par blocs + Clark-West + Holm | **conforme** |
| Verdict multi-modèles | MCS (Hansen-Lunde-Nason 2011) + SPA vs GARCH(1,1) | **conforme** |
| Agrégation inter-actifs | échelle MASE + fusion bonds/crypto avant pooling | **conforme** |

## Grille de conformité — baselines fortes (§4 de BRIEF_baselines_fortes.md)

| Point | Critère (niveau 1) | Statut |
|---|---|---|
| GARCH innovations | Student (`dist="t"`), `nu` estimé | **conforme** |
| Variante asymétrique | GJR-GARCH (`o=1`, `gamma` estimé) | **conforme** |
| Échantillonnage | vraies trajectoires simulées, aucune reconstruction gaussienne | **conforme** |
| Sélection de spéc. | validation uniquement, jamais le test | **conforme** |
| Protocole du duel | origines / gel T0 / CRPS / tests / MCS strictement inchangés | **conforme** |
| Traçabilité | bloc avant/après documenté | **conforme** |

## Grille de conformité — multi-graines (§4 de BRIEF_multigraines.md)

| Point | Critère (niveau 1) | Statut |
|---|---|---|
| Nombre de graines | **S = 5** (42, 43, 44, 45, 46), jamais 1 | **conforme** |
| Portée de la graine | chaque graine ré-arme init + échantillonnage TSDiff, MC-Dropout LSTM, tirages classiques (`duel_multiseed.py`, un seul `args.seed` threadé partout) | **conforme** |
| Dispersion rapportée | CRPS moyen ± écart-type par case et par modèle sur les 5 graines (`aggregate_crps_dispersion`) | **conforme** |
| Stabilité du verdict | fraction de graines où MCS / cases Holm / poolé tiennent, cases instables nommées (`aggregate_mcs_stability`/`_holm_stability`/`_pooled_stability`) | **conforme** |
| Entraînement global | mode global testé vs par-actif, sous-ensemble de 2 graines déclaré pour le coût | **conforme** |
| Budgets déclarés | époques / données / `m` / HPO / graines documentés par modèle (§ ci-dessous) | **conforme** |
| Protocole du duel | origines / gel T0 / CRPS / tests / MCS strictement inchangés (vérifié : mêmes origines, mêmes candidats d'époques sur les 5 graines) | **conforme** |

## Robustesse multi-graines (BRIEF_multigraines.md §2.2 — le vrai livrable)

### Dispersion du CRPS (coefficient de variation inter-graines, écart-type / moyenne, en %)

| Actif | Horizon | ARIMA-GARCH | SARIMA | Prophet | Naive | LSTM | TSDiff |
|---|---|---|---|---|---|---|---|
| BTC-USD | W1/W2/W3 | 1.7/1.0/0.5 | 0.9/0.8/0.7 | 0.0/0.0/0.0 | 1.0/0.2/0.3 | 4.6/9.1/12.4 | **4.4/10.3/12.0** |
| ETH-USD | W1/W2/W3 | 1.0/0.5/0.4 | 1.1/1.2/1.0 | 0.1/0.1/0.1 | 1.0/0.9/0.4 | 13.0/21.3/25.9 | **20.8/33.9/30.8** |
| SPY | W1/W2/W3 | 0.9/0.6/0.4 | 0.5/0.6/0.4 | 0.2/0.1/0.3 | 1.0/0.4/0.8 | 7.6/6.5/7.0 | 3.2/3.4/4.8 |
| TLT | W1/W2/W3 | 1.0/1.1/0.6 | 1.1/0.4/1.2 | 0.4/0.4/0.2 | 0.5/0.7/1.0 | 1.4/2.3/5.6 | 3.4/3.8/4.9 |
| ZN=F | W1/W2/W3 | 0.8/0.9/0.7 | 1.0/0.4/0.9 | 0.7/0.8/0.3 | 0.8/0.9/0.7 | 3.3/6.9/10.3 | 2.1/3.5/4.2 |

**Les modèles entraînés/stochastiques (TSDiff, LSTM) ont une variance inter-graines un ordre de
grandeur au-dessus des modèles classiques ajustés par MLE (ARIMA-GARCH, SARIMA : CV < 2% partout)**
— exactement le phénomène que Bouthillier et al. (2021) documentent, et la raison d'être de ce
chantier. Le pire cas est ETH-USD/TSDiff à W2 : CV = 34%, c'est-à-dire que le CRPS moyen d'une
graine à l'autre varie de plus d'un tiers de sa propre valeur — un verdict basé sur une seule
graine sur cet actif n'aurait aucune valeur probante.

### Stabilité du Model Confidence Set (fraction des 5 graines où chaque modèle est dans le MCS)

| Case | TSDiff | ARIMA-GARCH | SARIMA | Prophet | LSTM | Naive |
|---|---|---|---|---|---|---|
| BTC-USD\|W1 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| BTC-USD\|W2 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| BTC-USD\|W3 | **0.8** | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| ETH-USD\|W1 | **0.8** | 1.0 | 1.0 | 0.0 | 0.6 | 1.0 |
| ETH-USD\|W2 | **0.6** | 1.0 | 1.0 | 0.0 | 0.6 | 1.0 |
| ETH-USD\|W3 | **0.6** | 1.0 | 1.0 | 0.0 | 0.6 | 0.8 |
| SPY (3 horizons) | 1.0 | 1.0 | 1.0 | 0/1/1 | 0.0/0.6/0.8 | 1.0 |
| TLT (3 horizons) | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| ZN=F (3 horizons) | 1.0 | 1.0 | 1.0 | 0/0.6/1.0 | 0.6/1.0/1.0 | 1.0 |

**Cases qui basculent** (fraction ∉ {0, 1}, nommées comme l'exige le brief) : **BTC-USD\|W3**
(TSDiff exclu du MCS sur 1 graine/5), **ETH-USD\|W1/W2/W3** (TSDiff exclu sur 1 à 2 graines/5) —
c'est précisément sur crypto et aux horizons longs que l'appartenance de TSDiff au MCS n'est **pas**
garantie graine par graine. ARIMA-GARCH, SARIMA et Naive sont, eux, stables à 1.0 sur (presque)
toutes les cases (Naive descend une fois à 0.8 sur ETH-USD-W3) — cohérent avec leur CV quasi-nul
ci-dessus. LSTM est la famille la moins stable dans le MCS (0.0 à 1.0 selon la case).

### Stabilité des tests appariés TSDiff vs chaque classique (Holm, grille 5×3×5 = 75 cases)

**21 des 75 cases (28%) sont instables** (la conclusion significatif/non-significatif change selon
la graine) — jamais masqué : les 21 cases (toutes `TSDiff vs {ARIMA-GARCH, SARIMA, Naive, LSTM}`,
principalement sur ETH-USD, quelques-unes sur BTC-USD/TLT/ZN=F) impliquent presque toutes la graine
**43** comme graine "significative isolée" (ETH-USD y a un TSDiff nettement moins bon que sur les 4
autres graines). **Seules 9 cases sont significatives sur les 5 graines** (`fraction=1.0`, jamais
masquées) : les 3 horizons de `TSDiff vs Prophet` sur BTC-USD, ETH-USD-W1, SPY-W1, TLT-W1/W2,
ZN=F-W1, plus `TSDiff vs LSTM` sur SPY-W1 — **c'est le SEUL sous-ensemble de comparaisons dont le
signe et la significativité sont robustes graine par graine.**

### Stabilité du verdict poolé (MASE + classes d'actifs, 5 paires × 3 horizons = 15 cases) — LE RENVERSEMENT

| Paire | W1 | W2 | W3 |
|---|---|---|---|
| TSDiff vs ARIMA-GARCH | ns 1/5 (43) | **sig 4/5 (43,44,45,46)** | **sig 4/5 (43,44,45,46)** |
| TSDiff vs SARIMA | sig 1/5 (43) | sig 2/5 (43,44) | sig 2/5 (43,44) |
| TSDiff vs Naive | sig 2/5 (43,44) | sig 3/5 (43,44,45) | sig 3/5 (43,44,45) |
| TSDiff vs Prophet | **sig 5/5** | **sig 5/5** | **sig 5/5** |
| TSDiff vs LSTM | **sig 5/5** | sig 1/5 (45) | sig 0/5 |

**Sur les 4 comparaisons contre ARIMA-GARCH/SARIMA/Naive/LSTM (hors Prophet), quand une case
devient significative sur plusieurs graines, `mean_diff` (TSDiff − classique, échelle MASE) est
TOUJOURS POSITIF : TSDiff est significativement MOINS BON, jamais meilleur.** Exemple concret --
`TSDiff vs ARIMA-GARCH` à W2/W3 : `mean_diff` = +0.08/+0.04 (graine 42, non significatif) mais
+1.17/+1.54 (graine 43), +0.51/+0.79 (graine 44), +0.29/+0.57 (graine 45), +0.29/+0.32 (graine 46)
— **la graine 42 (celle du rapport initial BRIEF_baselines_fortes.md) était la PLUS FAVORABLE à
TSDiff des 5 graines testées, pas une graine représentative.** Sur 4 graines sur 5, poolé sur les 5
actifs, TSDiff est démontré significativement moins bon qu'ARIMA-GARCH à W2/W3.

### Le rejet SPA vs GARCH(1,1) — seul résultat parfaitement stable

**0 rejet sur les 15 cases × 5 graines = 75 tests : aucun modèle, sur aucune graine, ne bat
GARCH(1,1) de façon significative.** C'est le seul résultat de ce duel qui est identique, sans
la moindre exception, quelle que soit la graine.

## Entraînement global vs par actif (BRIEF_multigraines.md §2.3)

TSDiff entraîné **une seule fois sur les 5 actifs poolés** (fenêtres z-scorées par actif,
concaténées, un seul jeu de poids partagé), époques sélectionnées par validation poolée (jamais le
test), comparé au TSDiff par-actif de la **même graine** (mêmes classiques, mêmes origines) :

| | Époques poolées sélectionnées | Delta CRPS global vs par-actif (moyenne des 15 cases) | Cases où le MCS change |
|---|---|---|---|
| Graine 42 | 40 | **+25.8% (systématiquement pire)** | 9/15 |
| Graine 43 | 40 | +6.6% (pire en général, sauf ETH-USD) | 7/15 |

**L'entraînement global dégrade le CRPS de TSDiff dans 27 des 30 cases testées** (2 graines × 15
cases), de +11% à +43% selon la case — sauf sur **ETH-USD, où l'entraînement global améliore
nettement le CRPS de la graine 43** (-20% à -35%), précisément la graine/l'actif où le TSDiff
par-actif était le moins stable (cf. le CV de 34% ci-dessus et les 21 cases Holm instables
concentrées sur ETH-USD-graine-43). Le standing de TSDiff dans le MCS se dégrade dans la majorité
des cas où l'entraînement change (16 cellules changées sur 30, presque toutes dans le sens
"le global sort du MCS où le par-actif y était"), avec 2 exceptions inverses sur BTC-USD-W3 et
ETH-USD-W3 en graine 43 où c'est le par-actif qui était hors MCS.

**Conclusion de cette brique : l'entraînement par actif reste le choix par défaut défendable** —
le pooling multi-actifs n'apporte un gain que lorsque l'entraînement par-actif est lui-même
instable sur un actif donné (le cas ETH-USD-graine-43), ce qui suggère que le pooling agit comme
une régularisation de dernier recours plutôt qu'une amélioration générale.

## Budgets déclarés par modèle (BRIEF_multigraines.md §2.4)

| Modèle | Époques / spec | Données (fenêtre) | `m` | HPO | Graines |
|---|---|---|---|---|---|
| TSDiff (par actif) | candidats 40/60/80, sélection validation **par actif** (80 dans 23/25 cas graine×actif, 40 ou 60 sinon — cf. tableau ci-dessous) | 2015-01-01 → 2026-07-29 | 500 | époques (validation uniquement) | 5 |
| TSDiff (global) | candidats 40/60/80, sélection validation **poolée** (40 retenu sur les 2 graines testées) | 2015-01-01 → 2026-07-29 | 500 | époques (validation pooled uniquement) | 2 (sous-ensemble déclaré) |
| ARIMA-GARCH | spec normal/t/gjr-t, sélection validation par actif (mix des 3 selon graine/actif, cf. tableau) | 2015-01-01 → 2026-07-29 | 500 | spec d'innovation (validation uniquement) | 5 |
| SARIMA | ordre fixe (`sarima_model.ORDER`/`SEASONAL_ORDER`) | 2015-01-01 → 2026-07-29 | 500 | aucun | 5 |
| Prophet | par défaut (`interval_width=0.95`) | 2015-01-01 → 2026-07-29 | 500 | aucun | 5 |
| LSTM | époques fixes (`lstm_model.EPOCHS`), MC-Dropout ré-armé par graine | 2015-01-01 → 2026-07-29 | 500 | aucun | 5 |
| Naive | — | 2015-01-01 → 2026-07-29 | 500 | aucun | 5 |

Sélections effectives par graine × actif (traçabilité complète, aucune moyenne masquant la
variabilité) :

| Graine | SPY | BTC-USD | ETH-USD | ZN=F | TLT |
|---|---|---|---|---|---|
| 42 | 80ep / gjr-t | 40ep / gjr-t | 60ep / normal | 80ep / normal | 80ep / t |
| 43 | 80ep / gjr-t | 80ep / t | 80ep / t | 80ep / normal | 80ep / normal |
| 44 | 80ep / gjr-t | 80ep / gjr-t | 80ep / t | 80ep / t | 80ep / t |
| 45 | 80ep / gjr-t | 80ep / normal | 80ep / t | 80ep / t | 80ep / t |
| 46 | 80ep / normal | 80ep / normal | 80ep / normal | 80ep / t | 80ep / t |

## Lecture (rappel des briefs précédents, contre N3)

Ne pas lire : « TSDiff est dans le MCS 15/15 » ou « rien ne bat le naïf » comme des faits acquis
sur la seule graine 42. Lire : *sur 5 graines, le rejet SPA vs GARCH(1,1) et la supériorité de
TSDiff sur Prophet sont robustes ; l'appartenance de TSDiff au MCS et sa parité avec
ARIMA-GARCH/SARIMA/Naive ne le sont PAS sur crypto (BTC-USD-W3, ETH-USD) et se dégradent aux
horizons longs sur 4 graines/5 poolées ; l'entraînement par actif reste préférable au global sur
la quasi-totalité des cases testées.*

## Phrase de conclusion franche (BRIEF_multigraines.md §3)

**Non, le verdict n'est pas stable d'une graine à l'autre.** La graine 42 (celle utilisée dans
BRIEF_baselines_fortes.md) était la plus favorable à TSDiff des 5 testées : sur les 4 autres
graines, poolé sur les 5 actifs, TSDiff devient significativement moins bon qu'ARIMA-GARCH à W2/W3
(4 graines sur 5), et son appartenance au Model Confidence Set devient incertaine sur BTC-USD-W3 et
sur les 3 horizons d'ETH-USD (0.6 à 0.8 des graines seulement). Seuls deux résultats résistent aux
5 graines sans exception : **TSDiff bat significativement Prophet** (tous actifs, tous horizons) et
**aucun modèle ne bat significativement GARCH(1,1)** (SPA, 0 rejet sur 75 tests). Le reste du
verdict — TSDiff à égalité avec le naïf et les classiques sérieux — était un artefact partiel de la
graine 42, pas une conclusion robuste.

## Limites déclarées (pas masquées)

- `tsdiff_hp_samples=100` (et non 500) pour la sélection d'époques sur le bloc de validation
  uniquement — sans effet sur le score final (`m=500` partout dans le duel scoré).
- `n_boot=2000` pour MCS/SPA par graine (contre 10000 pour les bootstrap par paire de
  `paired_test.py`, déjà en place).
- Entraînement global testé sur **2 graines seulement** (sous-ensemble déclaré, brief §2.3 l'autorise
  explicitement pour raison de coût) — les graines 44/45/46 n'ont pas été testées en entraînement
  global ; l'ampleur exacte de la dégradation pourrait varier sur ces graines non testées.
- Bruit numérique run-to-run (<1%) sur SARIMA/Prophet déjà documenté dans la version précédente de
  cette NOTE, sans effet sur les conclusions ci-dessus (les écarts inter-graines rapportés ici sont
  d'un ordre de grandeur supérieur à ce bruit).
- Sélection du spec GARCH et des époques TSDiff-W répétée indépendamment par graine (jamais sur le
  test) — la variabilité de ces choix d'une graine à l'autre (tableau ci-dessus) fait partie de la
  variance inter-graines mesurée, pas un artefact de méthode.
