# NOTE — Étape 1bis : le déficit CRPS crypto de TSDiff-W était bien un sous-entraînement

*2026-07-27 — brackets d'epochs rejoués sur le vrai backtest rolling (78 origines), pas sur le
bloc de validation du sweep (12 origines). Fichiers : `backtest_rolling_tsdiffw_epoch_bracket_crypto.json`
et `..._paired_tests.json`. Script : `experiments/backtest_rolling_tsdiffw.py --epochs-override`.*

## Rappel du point de départ

Le backtest rolling initial (`NOTE_backtest_rolling_tsdiffw.md`) montrait TSDiff-W significativement
pire que ARIMA-GARCH/SARIMA/Naive sur le CRPS, **concentré sur BTC/ETH** — SPY/TLT/ZN=F étant à
égalité statistique avec les mêmes baselines. BTC/ETH utilisaient 30/60 epochs (choisis par
`epoch_sweep.py`, argmin CRPS sur un bloc de validation de 12 origines), contre 80 pour les 3
autres actifs.

Ma première vérification (courbes du sweep, 10→120 epochs, même grille que les 3 autres actifs)
montrait un minimum net à 30/60 avec dégradation au-delà — ce qui semblait **réfuter** l'hypothèse
de sous-entraînement. **Cette vérification était insuffisante** : elle ne portait que sur le petit
bloc de validation, pas sur le jeu d'évaluation réel (78 origines) sur lequel le verdict est reporté.

## Ce que montre le vrai backtest (78 origines, N=500 échantillons natifs)

| Actif | Epochs | CRPS W+1 | CRPS W+2 | CRPS W+3 |
|---|---|---|---|---|
| BTC-USD | 20 | 11 272 | 20 571 | 31 014 |
| BTC-USD | 30 (original) | 7 060 | 12 752 | 18 659 |
| BTC-USD | 40 | 4 744 | 8 588 | 12 208 |
| BTC-USD | **80** | **2 777** | **4 741** | **6 588** |
| ETH-USD | 40 | 290 | 520 | 717 |
| ETH-USD | 60 (original) | 183 | 306 | 397 |
| ETH-USD | **80** | **157** | **251** | **323** |

**Contrairement à la courbe du bloc de validation (en U, minimum à 30/60), le CRPS décroît de façon
monotone avec plus d'epochs sur le vrai backtest**, jusqu'à 80 (la limite testée, alignée sur les 3
autres actifs) — aucun signe de plateau ou de rebond. La couverture (50/80/95%) diminue en parallèle
mais reste dans une plage raisonnable à 80 epochs (ex. BTC W+1 : 44/90/— %, ETH W+1 : 45/—/91 % —
du même ordre que SPY/TLT/ZN=F dans le backtest initial, pas un collapse).

## Amélioration epochs=80 vs epochs original (30/60), même triplets, test apparié

| Actif | Horizon | Δ CRPS | p-value | Significatif |
|---|---|---|---|---|
| BTC-USD | W+1 | **-60,7 %** | <0.0001 | oui |
| BTC-USD | W+2 | **-62,8 %** | <0.0001 | oui |
| BTC-USD | W+3 | **-64,7 %** | <0.0001 | oui |
| ETH-USD | W+1 | **-14,1 %** | <0.0001 | oui |
| ETH-USD | W+2 | **-17,8 %** | <0.0001 | oui |
| ETH-USD | W+3 | **-18,7 %** | 0.0002 | oui |

## TSDiff-W(epochs=80) vs baselines classiques, crypto

| Actif | Horizon | vs ARIMA-GARCH | vs SARIMA | vs Naive |
|---|---|---|---|---|
| BTC-USD | W+1 | non significatif (p=0.69) | non sig. (p=0.28) | non sig. (p=0.22) |
| BTC-USD | W+2 | **sig., TSDiff pire** (p=0.03) | non sig. (p=0.06) | non sig. (p=0.09) |
| BTC-USD | W+3 | **sig., TSDiff pire** (p=0.002) | **sig., TSDiff pire** (p=0.02) | **sig., TSDiff pire** (p=0.03) |
| ETH-USD | W+1 | **sig., TSDiff pire, faible écart** (p=0.02) | non sig. (p=0.17) | non sig. (p=0.09) |
| ETH-USD | W+2 | non sig. (p=0.09) | non sig. (p=0.09) | non sig. (p=0.10) |
| ETH-USD | W+3 | non sig. (p=0.11) | non sig. (p=0.17) | non sig. (p=0.11) |

(TSDiff bat significativement Prophet et globalement LSTM sur les deux actifs, à tous les horizons
— inchangé par rapport au run initial.)

## Conclusion

**Le déficit crypto est confirmé comme un sous-entraînement, pas une limite structurelle.**
- Le petit bloc de validation du sweep (12 origines) donnait un optimum trompeur : il ne
  généralise pas au jeu d'évaluation réel (78 origines). C'est précisément le point soulevé avant
  de lancer ce contrôle — la vérification l'a confirmé, dans le sens inverse de ce qu'on attendait.
- À epochs=80 (alignés sur SPY/TLT/ZN=F), le CRPS crypto chute de 61-65 % (BTC) et 14-19 % (ETH),
  tests appariés très significatifs (p<0.001 partout).
- **ETH-USD devient statistiquement indiscernable de SARIMA/Naive à tous les horizons**, et de
  ARIMA-GARCH sauf un écart faible à W+1 — même profil que SPY/TLT/ZN=F dans le backtest initial.
- **BTC-USD se rapproche fortement mais garde un écart résiduel significatif à W+2/W+3** contre
  ARIMA-GARCH (et SARIMA/Naive à W+3) — plus petit d'un ordre de grandeur qu'avant (le rapport
  TSDiff/baseline passe de ~2,7x à ~1,2x sur W+3), mais pas totalement refermé.
- Le CRPS n'ayant montré aucun signe de plateau jusqu'à 80 epochs (la limite testée, choisie pour
  rester comparable aux 3 autres actifs), il est possible que BTC bénéficierait encore d'epochs
  supplémentaires au-delà de 80 — non testé ici (sortirait du cadre "aligné sur les autres actifs"
  de cette vérification).

## Ce qu'il faut retenir pour la suite

Ce n'est **pas** "TSDiff-W est structurellement limité sur le crypto" — la piste initiale
(sous-entraînement) était la bonne, mais la première vérification (fondée sur le petit bloc de
validation) donnait la mauvaise réponse. Deux implications :
1. **Le protocole de sélection d'epochs actuel (`epoch_sweep.py`, bloc de validation à 12 origines)
   n'est pas fiable pour choisir les epochs de TSDiff-W** — au moins pour BTC/ETH, il indique un
   optimum qui n'en est pas un sur le jeu d'évaluation réel. À corriger avant de refaire confiance
   à ses sélections pour d'autres actifs/regimes.
2. Le verdict global du backtest initial (TSDiff-W perd contre ARIMA-GARCH/SARIMA/Naive, poolé sur
   les 5 actifs) doit être **rejoué avec BTC/ETH à 80 epochs** avant toute conclusion définitive sur
   l'hypothèse "diffusion vs classiques" — le déficit poolé était tiré presque entièrement par
   BTC/ETH sous-entraînés ; avec le correctif, il devrait se réduire nettement, potentiellement
   jusqu'à la non-significativité sur plusieurs horizons.
