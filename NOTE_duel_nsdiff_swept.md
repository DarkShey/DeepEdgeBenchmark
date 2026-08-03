# NOTE — NsDiff-W avec sweep d'époques validation-only (miroir de NOTE_duel_nsdiff.md)

*2026-08-03. Suite de `BRIEF_nsdiff_epoch_sweep.md`, faisant suite à
`BRIEF_integration_nsdiff.md` + `NOTE_duel_nsdiff.md` (run à époques fixes,
`NSDIFF_EPOCHS_W=40`, jamais sélectionnées sur la validation). Objectif :
donner à NsDiff-W le même sweep d'époques (candidats 40/60/80, sélection
val-only, verrou E1) que TSDiff-W possède déjà, pour neutraliser la réserve
« CV pas toutes choses égales par ailleurs » de la note précédente — **pas**
une optimisation pour faire gagner NsDiff. Fichiers :
`experiments/epoch_sweep.py::fit_checkpoints_nsdiff`/`_sweep_one_model_nsdiff`
(nouvelles fonctions sœurs, la branche TSDiff est inchangée), `duel_backtest.py
::run_nsdiff_records` (routage sweep/fixe), `duel_multiseed.py`
(`--nsdiff-epoch-candidates`/`--nsdiff-hp-samples`/`--nsdiff-fixed-epochs`).
Run réel : 5 actifs, S = 5 graines (42-46), `n_val=12`, `n_test=30`, `m=500`,
`--nsdiff-epoch-candidates 40 60 80`, `--nsdiff-hp-samples 100`,
`--end 2026-07-29` (identique aux deux runs précédents), `--skip-global`.*

## Protocole et isolation des artefacts

- **Checkpoints TSDiff/classiques relus tels quels** depuis
  `experiments/checkpoints/` (25 fichiers, inchangés — vérifié : même compte
  de fichiers, `git diff` vide sur `experiments/duel_backtest.json` après le
  run, checksums identiques avant/après).
- **NsDiff recalculé à neuf** avec le sweep d'époques, checkpointé dans un
  **nouveau** sous-dossier isolé (`experiments/checkpoints_nsdiff_swept/`,
  distinct de `experiments/checkpoints_nsdiff/` du run à époques fixes),
  résultat écrit dans `experiments/duel_backtest_nsdiff_swept.json` (nouveau
  fichier — `experiments/duel_backtest_nsdiff.json`, le run à époques fixes,
  vérifié inchangé lui aussi, checksum identique).
- **Sweep incrémental, pas des fits indépendants** (décision déclarée, brief
  §3.3) : `nsdiff_model.NsDiff.train()` ne recrée jamais son optimiseur entre
  deux appels (`self.opt` construit une seule fois dans `__init__`), donc
  `epoch_sweep.fit_checkpoints_nsdiff` reprend l'entraînement là où il s'est
  arrêté à chaque palier (40 → +20 → 60 → +20 → 80), exactement comme
  `fit_checkpoints` le fait déjà pour TSDiff (sans EMA à réconcilier ici,
  NsDiff n'en a pas). Un seul entraînement par (graine, actif), pas trois.
- **Sélection validation-only, verrou E1** : `epoch_sweep._sweep_one_model_
  nsdiff` ne reçoit que `val_pos` (jamais `test_pos`), `hp_samples=100`
  (distinct de `m=500` utilisé pour le scoring final, comme TSDiff-W avec
  `--tsdiff-hp-samples` vs `--m-samples`). Prouvé par test
  (`experiments/test_epoch_sweep.py::test_sweep_one_model_nsdiff_never_reads_
  test_block`) : tronquer la série hebdo pour supprimer tout le bloc de test
  produit des scores CRPS_val **bit-identiques** — si le sweep avait lu ne
  serait-ce qu'une ligne du bloc de test, ça aurait planté ou changé le
  résultat. `select_epochs` réutilisé sans aucune modification.
- `--include-nsdiff` sans les nouveaux flags reste possible via l'échappatoire
  `--nsdiff-fixed-epochs N` (reproduit le comportement pré-sweep) ; le duel
  6-modèles sans `--include-nsdiff` n'a pas été touché du tout. Suite pytest
  entièrement verte (497 tests + 20 nouveaux : 11 dans `test_epoch_sweep.py`,
  9 dans `test_duel_multiseed.py`).
- Coût CPU : 144.5s pour les 25 combinaisons graine×actif (vs 95s pour le run
  à époques fixes) — le sweep coûte ~1.5× le run fixe (3 candidats au lieu
  d'1, mais entraînement incrémental, pas 3×), toujours de l'ordre de la
  minute pour l'ensemble, sans commune mesure avec le coût TSDiff.

## Le tableau des époques sélectionnées (traçabilité, §2 du brief)

| Graine | BTC-USD | ETH-USD | SPY | TLT | ZN=F |
|---|---|---|---|---|---|
| 42 | 40 | 40 | 40 | 40 | **60** |
| 43 | 40 | 40 | 40 | 40 | 40 |
| 44 | **60** | 40 | 40 | 40 | **60** |
| 45 | 40 | 40 | 40 | 40 | **60** |
| 46 | 40 | 40 | **80** | 40 | **60** |

**Distribution sur les 25 combinaisons graine×actif : 40 époques dans 19 cas
(76%), 60 dans 5 cas (20%, concentrés sur ZN=F — 4 des 5 graines), 80 dans 1
cas isolé (SPY, graine 46).** Le sweep **confirme majoritairement** le choix
figé de 40 (76% des cas), mais **pas partout** — ZN=F penche vers 60 sur 4
graines sur 5, ce qui n'aurait pas été visible avec le budget figé. Aucune
moyenne ne masque cette variabilité : c'est une donnée du résultat, pas un
artefact.

## LE livrable : le CV inter-graines de NsDiff survit-il à l'égalisation ?

**Oui, l'écart de stabilité survit largement — légèrement réduit, jamais
inversé.**

| Actif | Horizon | TSDiff (CV%) | NsDiff-fixe (CV%) | NsDiff-swept (CV%) | Δ(swept−fixe) |
|---|---|---|---|---|---|
| BTC-USD | W1 | 4.4 | 3.5 | 3.9 | +0.4 |
| BTC-USD | W2 | 10.3 | 5.0 | 6.5 | +1.5 |
| BTC-USD | W3 | 12.0 | 5.4 | 7.8 | +2.5 |
| ETH-USD | W1 | 20.8 | 0.8 | 0.8 | +0.0 |
| ETH-USD | W2 | 33.9 | 1.5 | 1.5 | +0.0 |
| ETH-USD | W3 | 30.8 | 2.0 | 2.0 | +0.0 |
| SPY | W1 | 3.2 | 1.8 | 1.8 | −0.0 |
| SPY | W2 | 3.4 | 2.5 | 2.5 | −0.0 |
| SPY | W3 | 4.8 | 2.2 | 2.8 | +0.6 |
| TLT | W1 | 3.4 | 1.5 | 1.5 | −0.0 |
| TLT | W2 | 3.8 | 2.3 | 2.3 | +0.0 |
| TLT | W3 | 4.9 | 2.0 | 2.0 | +0.0 |
| ZN=F | W1 | 2.1 | 1.4 | 2.0 | +0.6 |
| ZN=F | W2 | 3.5 | 2.0 | 1.7 | −0.2 |
| ZN=F | W3 | 4.2 | 2.0 | 0.7 | −1.3 |

**Moyenne des 15 cases : TSDiff = 9.7% (inchangé, checkpoints relus tels
quels), NsDiff-fixe = 2.4%, NsDiff-swept = 2.7%.** Le ratio de stabilité
passe de **4.05×** (fixe) à **3.65×** (swept) — NsDiff reste très nettement
plus reproductible d'une graine à l'autre que TSDiff, l'écart se réduit
d'environ 10% mais **ne disparaît pas et ne s'inverse nulle part**. La
hausse de CV vient presque exclusivement de BTC-USD (+0.4/+1.5/+2.5 points)
et ZN=F/W1 (+0.6) — précisément les actifs où le sweep a sélectionné 60
époques sur 1 à 4 graines sur 5 (cf. tableau ci-dessus) : le choix discret
d'époques devient lui-même une source de variance inter-graines
supplémentaire, exactement le mécanisme que ce chantier visait à mesurer
honnêtement (et pas à cacher). ZN=F/W2/W3 baissent même légèrement (−0.2/
−1.3) : le sweep n'est pas strictement défavorable partout non plus.

**Conclusion honnête** : l'avantage de stabilité de NsDiff n'était donc
**pas principalement un artefact du budget d'époques figé** — il survit,
quasi intact, à l'égalisation exacte du protocole de sélection.

## Ré-réponses aux 3 questions (confirmé / infirmé vs le run figé)

### 1) NsDiff vs GARCH(1,1) sous SPA

**Inchangé : toujours 0 rejet sur 75 tests.** Signal directionnel quasi
identique : gain positif sur 28/75 cases (identique au run figé, ex-aequo
avec SARIMA), meilleur candidat de la case sur 15/75 (contre 16/75 figé —
un seul basculement de cellule, du bruit, pas un changement de tendance).
**Confirmé, rien ne bouge.**

### 2) NsDiff vs TSDiff en CRPS

**Confirmé, quasiment au chiffre près.** Verdict poolé (MASE) identique au
run figé : W1 sig 1/5 (graine 43), **W2 sig 4/5 (43,44,45,46)**, **W3 sig
4/5 (43,44,45,46)**. Signe stable sur 14/15 combinaisons graine×horizon
(la même exception minuscule et non significative, graine 42/W3 :
mean_diff = −0.030 swept vs −0.022 fixe). **NsDiff bat TSDiff en CRPS moyen
sur 15/15 cases, sans exception**, contre 15/15 déjà dans le run figé — le
sweep ne change même pas un seul gagnant de case, seulement l'ampleur (delta
vs fixe de −0.3% à +2.0% selon la case, cf. tableau détaillé ci-dessous).
**Confirmé, robuste au sweep.**

### 3) MCS : NsDiff plus souvent que TSDiff dans le MCS ?

**Inchangé cellule par cellule, à la décimale près.** Les fractions MCS de
NsDiff sont **identiques** entre le run figé et le run swept sur les 15
cases (BTC-USD W1=0.6/W2=0.8/W3=0.8 ; ETH-USD 1.0 partout ; SPY/TLT/ZN=F 1.0
partout) — le choix d'époques (40 vs 60 vs 80 selon la case) n'a fait
basculer **aucune** graine d'entrée/sortie du MCS. Moyenne inchangée :
NsDiff = 0.947, TSDiff = 0.920. Le constat de `NOTE_duel_nsdiff.md` tient
mot pour mot : NsDiff nettement plus stable que TSDiff sur ETH-USD (là où
TSDiff décrochait), légèrement moins stable sur BTC-USD à W1/W2 — **rien
n'a changé, confirmé.**

## Comparaison explicite swept vs fixe (le vrai delta, case par case)

| Actif | Horizon | CRPS NsDiff-fixe | CRPS NsDiff-swept | Δ CRPS | Époques (cas majoritaire) |
|---|---|---|---|---|---|
| BTC-USD | W1 | 2537.27 | 2542.77 | +0.2% | 40 (4/5), 60 (1/5) |
| BTC-USD | W2 | 4131.06 | 4160.71 | +0.7% | 40 (4/5), 60 (1/5) |
| BTC-USD | W3 | 5485.35 | 5555.56 | +1.3% | 40 (4/5), 60 (1/5) |
| ETH-USD | W1-W3 | (inchangé) | (inchangé) | +0.0% | 40 (5/5) |
| SPY | W1 | 7.13 | 7.15 | +0.3% | 40 (4/5), 80 (1/5) |
| SPY | W2-W3 | ~inchangé | ~inchangé | 0.0/−0.3% | 40 (4/5), 80 (1/5) |
| TLT | W1-W3 | (inchangé) | (inchangé) | ~0.0% | 40 (5/5) |
| ZN=F | W1 | 0.385 | 0.391 | +1.6% | 40 (1/5), 60 (4/5) |
| ZN=F | W2 | 0.530 | 0.537 | +1.4% | 40 (1/5), 60 (4/5) |
| ZN=F | W3 | 0.666 | 0.680 | +2.0% | 40 (1/5), 60 (4/5) |

**Le sweep dégrade très légèrement le CRPS moyen de NsDiff sur BTC-USD et
ZN=F (+0.2% à +2.0%), inchangé sur ETH-USD/TLT, quasi-nul sur SPY.** C'est
cohérent : là où le sweep préfère 60 (voire 80) à 40 sur *certaines* graines
seulement, la moyenne des 5 graines mélange une minorité de graines avec
plus d'époques (potentiellement plus proches du sur-apprentissage sur cet
actif précis) et une majorité restée à 40 — d'où une moyenne légèrement
moins bonne que le run entièrement figé à 40, ET une variance inter-graines
plus élevée sur ces mêmes cases (cf. tableau CV ci-dessus, mêmes actifs).
**Aucune case ne bascule de gagnant contre TSDiff** (NsDiff reste devant
TSDiff sur toutes les 15 cases dans les deux runs) — la dégradation, quand
elle existe, reste largement dans la marge où NsDiff bat encore largement
TSDiff (ex. BTC-USD W3 : TSDiff=6131, NsDiff-fixe=5485, NsDiff-swept=5556 —
NsDiff-swept reste ~575 CRPS meilleur que TSDiff malgré le delta de +71 vs
NsDiff-fixe).

## Stabilité des tests appariés (Holm) et verdict poolé — inchangés

**Total de cases instables identique : 19/90 (21%) dans les deux runs**, la
même décomposition par paire (`NsDiff vs ARIMA-GARCH` 8 instables, `NsDiff vs
LSTM` 6, `TSDiff vs NsDiff` 3 — toutes sur ETH-USD, `NsDiff vs Prophet` 1,
`NsDiff vs Naive` 1, `NsDiff vs SARIMA` 0). Le verdict poolé (MASE) par paire
et par horizon est **identique au chiffre près** au run figé pour les 6
paires impliquant NsDiff (`NsDiff vs Prophet`/`LSTM` : sig 5/5 ou 4/5,
robuste ; `NsDiff vs ARIMA-GARCH`/`SARIMA`/`Naive` : jamais stable-
significatif, signe qui change de graine). **Rien de neuf ici — le sweep
n'a strictement rien changé au verdict statistique par paire.**

## Résumé factuel (à ne pas sur-interpréter)

- **Le CV de NsDiff survit à l'égalisation du budget d'époques** : 2.4%→2.7%
  en moyenne (contre 9.7% pour TSDiff, inchangé) — ratio de stabilité
  4.05×→3.65×. L'avantage se réduit légèrement mais reste massif.
- Sweep très majoritairement stable : 76% des combinaisons graine×actif
  sélectionnent 40 (comme le run figé), 20% sélectionnent 60 (concentré sur
  ZN=F), 1 cas isolé à 80 (SPY, graine 46).
- Les 3 questions (SPA/GARCH, NsDiff vs TSDiff CRPS, MCS) donnent des
  réponses **identiques** au run figé — confirmé, pas supposé.
- Le sweep dégrade très légèrement le CRPS moyen de NsDiff sur BTC-USD/ZN=F
  (+0.2% à +2.0%), sans jamais faire perdre à NsDiff sa place devant TSDiff
  sur aucune des 15 cases.
- Rien, dans les tests appariés Holm ou le verdict poolé, ne bouge d'une
  case à l'autre entre le run figé et le run swept.

## Limites déclarées (mise à jour)

- **Résolue** : la réserve « le CV 4× plus faible de NsDiff n'est pas toutes
  choses égales par ailleurs » (budget d'époques figé vs sweepé) de
  `NOTE_duel_nsdiff.md` est neutralisée par ce chantier — le résultat de
  cette note en tient compte.
- **Sweep incrémental (déclaré, §3.3)** : comme pour TSDiff-W, chaque
  candidat d'époques réutilise les poids/l'état de l'optimiseur du candidat
  précédent plutôt que de repartir de zéro — un fit par (graine, actif), pas
  trois indépendants. C'est une approximation du "vrai" sweep indépendant
  (les poids à 60 époques dépendent légèrement d'avoir d'abord entraîné 40
  époques), exactement la même approximation que celle déjà acceptée pour
  TSDiff-W depuis `BRIEF_weekly_prediction_v2.md` — pas une nouvelle
  incertitude introduite par ce chantier.
- `--nsdiff-hp-samples=100` (comme `--tsdiff-hp-samples`) pour la sélection
  d'époques uniquement — sans effet sur le score final (`m=500` partout dans
  le duel scoré).
- Comparaison strictement à 5 graines (42-46), comme les deux runs
  précédents — pas de garantie hors de cet échantillon de graines.
- `--skip-global` toujours en vigueur : la comparaison entraînement global
  vs par-actif reste hors périmètre pour NsDiff.
- Le tableau des époques sélectionnées (§ ci-dessus) montre une vraie
  variabilité graine par graine sur BTC-USD/ZN=F/SPY — un sweep mené sur
  d'autres graines pourrait sélectionner d'autres valeurs sur ces mêmes
  actifs ; ce n'est pas masqué, c'est le résultat.

## Reproduire

```
python experiments/duel_multiseed.py --seeds 42 43 44 45 46 --m-samples 500 \
    --skip-global --include-nsdiff \
    --nsdiff-epoch-candidates 40 60 80 --nsdiff-hp-samples 100 \
    --end 2026-07-29 --out experiments/duel_backtest_nsdiff_swept.json
```

Avec le code tel qu'il est sur `feat/nsdiff-epoch-sweep` : les checkpoints
TSDiff/classiques existants (`experiments/checkpoints/`) sont relus tels
quels si présents ; les checkpoints NsDiff-swept sont écrits dans
`experiments/checkpoints/seed{N}_{actif}_nsdiff.json` par défaut (nom de
fichier distinct des checkpoints classiques ET du run NsDiff à époques fixes
— cf. `duel_multiseed.checkpoint_path_nsdiff`, inchangé depuis
`NOTE_duel_nsdiff.md`). Ce run a, comme le précédent, isolé ses checkpoints
dans un sous-dossier séparé (`experiments/checkpoints_nsdiff_swept/`) par
précaution supplémentaire, via un monkeypatch runtime documenté dans
`NOTE_duel_nsdiff.md`.

Pour reproduire le run à époques **figées** (`NOTE_duel_nsdiff.md`) à la
place, ajouter `--nsdiff-fixed-epochs 40` (bypass le sweep entièrement).
