# Note — contrôle qualité NsDiff vs TSDiff (1 actif × 1 graine)

**Date : 2026-08-03.** Contrôle qualité demandé par `BRIEF_integration_nsdiff.md` §6/§7.7 :
comparer calibration (PI Cov 95 %) et CRPS empirique de NsDiff vs TSDiff sur **1 actif ×
1 graine**, plus le budget temps CPU (`python models/nsdiff_model.py --ticker SPY`).

## Protocole

- Actif : `SPY`, `2021-01-01` → `2024-12-31` (yfinance, Close auto-adjust) — 853 jours
  train, 151 jours test (`test_ratio=0.15`, split identique aux deux modèles).
- Graine : `42`, seedée via `set_seed` (chaque modèle threade sa propre fonction —
  `tsdiff_model.set_seed` / `nsdiff_model.set_seed`, même contrat).
- `run_tsdiff(train, test, keep_samples=True)` / `run_nsdiff(train, test,
  keep_samples=True)`, **hyperparamètres par défaut des deux fichiers** (aucun réglage
  spécial pour favoriser l'un ou l'autre) : TSDiff `hidden=64, depth=2, epochs=40,
  n_samples=50, k_denoise=20` ; NsDiff `hidden_mean=hidden_sigma=hidden_denoise=32,
  epochs=40, n_samples=50, k_denoise=20` (= nombre de pas de diffusion, cf. docstring
  `nsdiff_model.py` sur le choix d'échantillonnage ancestral complet plutôt que DDIM).
- CRPS : `experiments/crps_metrics.crps_fair` (Ferro 2014, non biaisé taille finie) sur le
  nuage `ensemble` (n_samples=50/pas) déjà produit par le walk-forward — jamais l'O(n²)
  `crps_empirical` en boucle sur un grand nuage (cf. mémoire disque/RAM machine).

## Résultats

| Métrique               | TSDiff   | NsDiff   |
|-------------------------|----------|----------|
| RMSE                    | 5.0759   | **4.6950** |
| MAE                     | 3.8421   | **3.3998** |
| Dir. Acc (%)            | **55.33**| 52.00    |
| PI Cov 95 % (%)         | 90.73    | **92.05** |
| CRPS fair (moyenne)     | 2.6848   | **2.4778** |
| Temps total (walk-forward, 151 pas) | 59.9 s | **2.6 s** |

## Lecture

- **Calibration** (la thèse du brief) : NsDiff est **au moins comparable, légèrement
  meilleur** que TSDiff ici — couverture PI 95 % plus proche de la cible (92.05 % vs
  90.73 %, cible 95 %) et CRPS fair inférieur (2.478 vs 2.685). Cohérent avec l'attendu :
  le couple `g_psi`/UANS conditionne explicitement la largeur de l'intervalle sur
  l'historique récent, là où TSDiff (bruit à variance constante) ne le fait pas.
- **Point forecast** (RMSE/MAE) : NsDiff légèrement meilleur aussi, mais ce n'est pas la
  métrique qui compte pour ce modèle (cf. docstring `nsdiff_model.py` — le backbone de
  moyenne `f_phi` est volontairement dégraissé, la contribution du modèle est sur la
  variance). Le Dir. Acc légèrement inférieur (52.0 vs 55.33) reste dans le bruit attendu
  à faible SNR sur un seul actif/une seule graine — pas une conclusion à tirer d'un point
  de mesure unique (cf. duel multi-graines/multi-actifs pour un verdict statistique, une
  fois lancé avec `--include-nsdiff`).
- **Budget CPU** : NsDiff est ~23× plus rapide sur ce run (2.6 s vs 59.9 s pour 151 pas de
  walk-forward). Attendu : le backbone de moyenne dégraissé (MLP) et le dénoiseur
  (`_Denoiser`, MLPs conditionnés par pas de diffusion) sont beaucoup moins chers qu'un
  U-Net 1D + attention (TSDiff), et les deux modèles utilisent le même nombre de pas de
  diffusion à l'inférence (20). Marge confortable pour durcir les hyperparamètres NsDiff
  (hidden dims, epochs) si un futur duel multi-actifs montre que ça vaut le coût CPU
  additionnel.

## Limites de ce contrôle

Un seul actif, une seule graine — un signal directionnel, pas une preuve statistique.
Le duel rigoureux (rolling-origin, MCS/SPA, Holm, multi-graines) existe déjà et couvre
NsDiff dès qu'on l'active : `python experiments/duel_backtest.py --include-nsdiff` (et
`duel_multiseed.py --include-nsdiff` pour la robustesse multi-graines) — non lancé ici
(coût CPU/temps hors budget de ce contrôle ponctuel), mais câblé et testé (cf. la suite
`experiments/` verte après le commit de câblage du duel).

## Reproduire

```
cd models
python -c "
import tsdiff_model as td, nsdiff_model as nd
prices = td.fetch_data('SPY', '2021-01-01', '2024-12-31')
split = int(len(prices) * 0.85)
train, test = prices.iloc[:split], prices.iloc[split:]
td.set_seed(42); print(td.run_tsdiff(train, test))
nd.set_seed(42); print(nd.run_nsdiff(train, test))
"
```
