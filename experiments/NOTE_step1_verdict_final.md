# NOTE — Clôture de l'étape 1 : le vrai verdict poolé (5 actifs, crypto corrigé)

*2026-07-27 — Fichiers : `backtest_rolling_tsdiffw_step1_final.json` (KPI, format
`kpi_probabilistes.json`), `backtest_rolling_tsdiffw_step1_final_paired_tests.json` (avant/après).
SPY/TLT/ZN=F : mêmes epochs (80), même protocole, même seed=42 que le run initial — recalculés à
l'identique (déterministe) car la base ne stocke pas le nuage d'échantillons brut. BTC-USD/ETH-USD :
80 epochs (justifié dans `NOTE_epoch_bracket_crypto.md`), contre 30/60 dans le run initial.*

## Rappel : pourquoi ce run existe

Le verdict poolé initial (`NOTE_backtest_rolling_tsdiffw.md`) disait « TSDiff-W perd
significativement contre ARIMA-GARCH/SARIMA/Naive aux 3 horizons ». L'étape 1bis a montré que ce
verdict était tiré par BTC/ETH sous-entraînés (30/60 epochs au lieu de 80 pour les 3 autres
actifs) — sur le vrai backtest, les pousser à 80 epochs referme -61 à -65 % du CRPS pour BTC et
-14 à -19 % pour ETH. Ce run rejoue le poolé complet avec ce correctif, et **rien d'autre**.

## Le tableau avant/après (CRPS normalisé poolé, DM-HAC + bootstrap par blocs, Holm sur les 5
comparaisons, séparément par horizon)

| Horizon | Comparaison | AVANT (crypto sous-entraîné) | APRÈS (crypto corrigé, 80 epochs) |
|---|---|---|---|
| **W+1** | vs ARIMA-GARCH | +1.81, **p<0.0001, significatif (TSDiff pire)** | +0.11, p=0.45, **non significatif** |
| | vs SARIMA | +1.78, **sig.** | +0.07, **non sig.** |
| | vs Naive | +1.82, **sig.** | +0.12, **non sig.** |
| | vs Prophet | -7.30, sig. (TSDiff meilleur) | -9.01, sig. (TSDiff meilleur) |
| | vs LSTM | -1.13, sig. (TSDiff meilleur) | -2.84, sig. (TSDiff meilleur) |
| **W+2** | vs ARIMA-GARCH | +3.64, **sig.** | +0.42, **sig., mais 8,6x plus petit** |
| | vs SARIMA | +3.58, **sig.** | +0.37, **non sig.** |
| | vs Naive | +3.60, **sig.** | +0.38, **non sig.** |
| | vs Prophet | -5.06, sig. (TSDiff meilleur) | -8.28, sig. (TSDiff meilleur) |
| | vs LSTM | +1.42, non sig. | -1.80, sig. (TSDiff meilleur) |
| **W+3** | vs ARIMA-GARCH | +5.50, **sig.** | +0.68, **sig., mais 8,1x plus petit** |
| | vs SARIMA | +5.42, **sig.** | +0.61, **non sig.** |
| | vs Naive | +5.46, **sig.** | +0.64, **non sig.** |
| | vs Prophet | -2.70, non sig. | -7.51, sig. (TSDiff meilleur) |
| | vs LSTM | +3.68, **sig.** | -1.13, non sig. |

(Valeurs = différentiel de CRPS normalisé poolé, positif = TSDiff-W pire. Holm appliqué séparément
par horizon sur les 5 comparaisons.)

## Ratio de CRPS brut, par actif (corrigé, indicatif — pas le test formel ci-dessus)

| | vs ARIMA-GARCH | vs SARIMA | vs Naive |
|---|---|---|---|
| SPY | 0.99–1.04x | 0.97–0.98x | 0.98–0.99x |
| TLT | 1.01–1.15x | 0.96–1.08x | 0.96–1.06x |
| ZN=F | 0.99–1.06x | 0.98–1.05x | 0.99–1.05x |
| BTC-USD | 1.01–1.18x | 1.07–1.24x | 1.09–1.23x |
| ETH-USD | 1.14–1.17x | 1.08–1.15x | 1.13–1.16x |

Tous les actifs sont maintenant dans une fourchette de ±25 % des meilleures baselines — contre des
écarts de 2 à 4x pour BTC/ETH dans le run initial.

## Le vrai verdict de l'étape 1

**À W+1 : TSDiff-W est statistiquement indiscernable des 3 meilleures baselines classiques**
(ARIMA-GARCH, SARIMA, Naive) — aucun écart significatif après correction Holm. Il bat
significativement Prophet et LSTM.

**À W+2 et W+3 : TSDiff-W perd encore contre ARIMA-GARCH, mais l'écart s'est réduit d'un facteur
~8x** (poolé, normalisé) par rapport au run initial, et **n'est plus significatif contre SARIMA ni
Naive**. Il continue de battre Prophet partout, et LSTM à W+2 (plus de différence significative à
W+3).

**Ce qui n'a pas changé** : TSDiff-W ne bat jamais ARIMA-GARCH de façon significative à aucun
horizon (l'écart W+1 s'efface mais ne s'inverse pas) ; il reste nettement meilleur que les deux
baselines les plus faibles (Prophet, LSTM) partout.

## Cadrage — ce que ce résultat ne dit PAS

- **Périmètre strictement weekly.** Le daily perd déjà en natif sur la précision (analyse
  antérieure, inchangée par ce travail) — ce verdict ne concerne que W+1/W+2/W+3.
- **La comparaison CRPS reste asymétrique** : TSDiff-W tire de vrais échantillons de diffusion, les
  5 baselines tirent des échantillons gaussiens depuis leur PI stockée (protocole de production).
  Si cette asymétrie favorise un camp, c'est plutôt TSDiff-W (distribution potentiellement plus
  riche) — donc le fait qu'il n'ait toujours pas d'avantage significatif à W+2/W+3 est plutôt
  conservateur pour lui, pas un artefact en sa défaveur. Traiter cette asymétrie de façon rigoureuse
  (échantillonnage empirique/bootstrap pour les baselines) est l'objet d'une étape suivante — **non
  traité ici**.
- Le protocole de sélection d'epochs (`epoch_sweep.py`) reste à corriger structurellement pour les
  futurs actifs/régimes — cf. `NOTE_epoch_bracket_crypto.md`, point 1.

## Conclusion

L'étape 1, correctement close : **l'hypothèse « la diffusion bat les modèles classiques » n'est ni
confirmée ni clairement infirmée en weekly** — c'est un résultat beaucoup plus nuancé que le
verdict initial (« TSDiff-W perd significativement partout »), qui était un artefact de
sous-entraînement crypto. À protocole propre, TSDiff-W est à égalité avec les meilleures baselines
classiques à W+1, proche (écart réduit ~8x, non significatif contre 2 des 3) à W+2/W+3, et bat
nettement les baselines faibles partout.
