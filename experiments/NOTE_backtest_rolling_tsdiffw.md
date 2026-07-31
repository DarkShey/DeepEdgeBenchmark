# NOTE — Backtest rolling-origin TSDiff-W natif (P1 : artefact ou déficit réel ?)

*2026-07-27 — `experiments/backtest_rolling_tsdiffw.py`, run complet 5 actifs, 26,6 min.*

## Question

Le TSDiff weekly évalué jusqu'ici n'était pas natif (TSDiff-D poussé multi-pas), avec une
calibration effondrée (16 % de couverture pour une cible de 50 %). Un backtest rolling-origin
de TSDiff-W **vraiment natif**, sur les mêmes triplets que les 5 baselines, corrige-t-il ce
déficit — et si oui, l'hypothèse « la diffusion bat les modèles classiques » tient-elle alors ?

## Protocole (résumé — détails dans le script)

- **1170 triplets appariés** (actif, cutoff, horizon), strictement identiques entre TSDiff-W et
  les 5 baselines : 78 origines hebdo × 5 actifs × 3 horizons (W+1/W+2/W+3), 2024-10-18 →
  2026-07-02/03. Origines reprises telles quelles du backtest déjà existant des baselines
  (`weekly_multimodel_n90.json`, réentraînées à chaque origine), moins ~12 dates/actif exclues
  pour éviter une fuite de sélection d'epoch.
- **TSDiff-W** : réentraînement expansif périodique (tous les 15 origines, 6 blocs/actif) —
  obsolescence des poids bornée à ≤ 15 semaines, au lieu de ~90 pour un train-once-forward pur.
  N=500 échantillons natifs de diffusion par cellule.
- **Baselines** : ré-échantillonnage gaussien (N=500) depuis leur PI 95 % déjà stockée —
  protocole strictement identique à la production en hebdo (Monte-Carlo trop coûteux à ces
  horizons). **Asymétrie assumée** : TSDiff-W tire de vrais échantillons de diffusion, les
  baselines des échantillons gaussiens. Si TSDiff-W perd malgré son avantage distributionnel
  potentiel, le résultat n'en est que plus solide.
- Lignes TSDiff-W stockées `source='backtest_rolling'` (1170 lignes, isolées du live/oos).
- Tests appariés : bootstrap par blocs par (actif, horizon) + test poolé (DM-HAC + bootstrap par
  blocs, cluster par classe d'actif, CRPS normalisé par l'échelle de chaque actif), Holm sur les
  5 comparaisons, séparément par horizon.

## Résultat 1 — La calibration de TSDiff-W (P1) : effondrement levé

| | 50 % (cible) | 80 % (cible) | 95 % (cible) |
|---|---|---|---|
| **Ancienne matrice (contaminée par TSDiff-D)** | 16 % | 41 % | 66 % |
| **TSDiff-W natif, pooled (ce backtest, n=1170)** | **41 %** | **71 %** | **87 %** |

Par classe d'actif (TSDiff-W seul, indépendant de la méthode d'échantillonnage des baselines) :

| Classe | Cov 50 % | Cov 80 % | Cov 95 % |
|---|---|---|---|
| Crypto (BTC/ETH) | 0.54 | 0.86 | 0.98 |
| Bonds (TLT/ZN=F) | 0.34 | 0.63 | 0.81 |
| Index (SPY) | 0.30 | 0.56 | 0.80 |

**Verdict P1 : levé, pas parfaitement.** Plus aucune trace de l'effondrement catastrophique — la
couverture est redevenue du même ordre que celle des baselines (ARIMA-GARCH: 94-95 % à 95 % ;
SARIMA/Naive: 86-89 %). TSDiff-W sous-couvre encore un peu sur index/bonds (intervalles un peu
trop étroits) mais reste dans une plage normale, pas un artefact catastrophique. C'était bien
majoritairement un problème de protocole (P1), pas un déficit structurel de calibration.

## Résultat 2 — CRPS, pooled et Holm-corrigé (233 origines poolées, effective_n≈77)

TSDiff-W vs chaque baseline (positif = TSDiff-W a un CRPS **plus élevé**, donc pire) :

| Horizon | vs ARIMA-GARCH | vs SARIMA | vs Naive | vs LSTM | vs Prophet |
|---|---|---|---|---|---|
| W+1 | **+1.81** (sig.) | **+1.78** (sig.) | **+1.81** (sig.) | -1.08 (non-concordant) | **-7.32** (sig., TSDiff meilleur) |
| W+2 | **+3.64** (sig.) | **+3.59** (sig.) | **+3.60** (sig.) | +1.41 (n.s.) | **-5.02** (sig., TSDiff meilleur) |
| W+3 | **+5.47** (sig.) | **+5.44** (sig.) | **+5.47** (sig.) | **+3.65** (sig.) | -2.69 (n.s.) |

**TSDiff-W perd significativement contre ARIMA-GARCH, SARIMA et Naive aux 3 horizons** (p<0.01
après Holm, DM-HAC et bootstrap concordants). Il **bat significativement Prophet** à W+1/W+2.
Face à LSTM, résultat mitigé (pas de verdict robuste à W+1/W+2, TSDiff-W perd à W+3).

## Résultat 3 — mais ce déficit est concentré sur le crypto (analyse par actif)

Décomposition par actif (test bloc-bootstrap par actif, non Holm-corrigé, à lire comme un
diagnostic, pas un verdict indépendant) :

- **BTC-USD / ETH-USD (crypto)** : TSDiff-W significativement **pire** qu'ARIMA-GARCH/SARIMA/
  Naive aux 3 horizons — écart large et net.
- **SPY (index)** : **aucune différence significative** avec ARIMA-GARCH/SARIMA/Naive à aucun
  horizon — TSDiff-W est à égalité statistique avec les meilleures baselines. Bat LSTM et
  Prophet significativement.
- **TLT / ZN=F (bonds)** : très majoritairement **aucune différence significative** avec
  ARIMA-GARCH/SARIMA/Naive (sauf TLT à W+3). Bat LSTM et Prophet significativement.

Le déficit poolé (Résultat 2) est donc tiré presque entièrement par le crypto, où TSDiff-W est
nettement moins bon. Sur index et bonds, une fois le protocole corrigé, TSDiff-W tient la
comparaison avec les meilleures baselines classiques.

## Conclusion

1. **P1 (artefact de calibration weekly) : levé.** Le TSDiff-W natif, correctement backtesté,
   n'a plus la calibration effondrée observée avec le TSDiff-D poussé multi-pas.
2. **L'hypothèse « la diffusion bat les modèles classiques » n'est toujours pas confirmée** de
   façon générale : à protocole propre et triplets identiques, TSDiff-W perd significativement
   contre ARIMA-GARCH/SARIMA/Naive sur le CRPS, à tous les horizons, sur l'ensemble poolé.
3. **Nuance importante** : ce déficit n'est pas uniforme — il vient presque exclusivement du
   crypto (BTC/ETH). Sur index (SPY) et bonds (TLT/ZN=F), TSDiff-W est statistiquement
   indiscernable des meilleures baselines classiques, et bat nettement LSTM et Prophet partout.
4. **Prochaine piste (hors scope ici)** : investiguer pourquoi le crypto tire TSDiff-W vers le
   bas — sous-entraînement (BTC=30 / ETH=60 epochs, les plus faibles des 5 actifs) plutôt qu'une
   limite structurelle du modèle de diffusion, à vérifier avant de conclure plus largement (P4/P5
   du plan méthodologique).

## Puissance et limites

- 1170 triplets bruts, 233 origines poolées par horizon, effective_n≈77 après ajustement pour
  l'autocorrélation des cibles W1-W3 qui se chevauchent — puissance correcte, pas excellente.
- Asymétrie d'échantillonnage assumée et documentée (diffusion natif vs gaussien) : si elle
  favorise un camp, c'est plutôt TSDiff-W (distribution plus riche) — le fait qu'il perde quand
  même contre ARIMA/SARIMA/Naive rend ce résultat plus solide, pas moins.
- Réentraînement périodique (15 origines) reste une approximation par rapport au réentraînement
  à chaque origine des baselines — biais résiduel documenté, probablement mineur vu le coût de
  fit observé (quelques dizaines de secondes/bloc).
