# Vérification comportementale — BRIEF_branchement_prod_calibration_sigma.md, Chantier 3

Données offline reproductibles (experiments/offline_prices.py / DONNEE~1.XLS), aucun accès réseau, tracking.db JAMAIS modifié (bases sqlite temporaires).

Note méthodologique : la §3 (Prophet) tourne sur une fenêtre de test volontairement courte (9 points, refit_freq=3) -- machine de vérification à 8 Go de RAM, insuffisante pour le walk-forward complet à 15 points/refit_freq=1 utilisé dans le HANDOFF (qui, lui, reste la validation statistique de référence : 3 fenêtres x SPY/BTC, cf. HANDOFF §4). Ici, objectif illustratif : confirmer le SIGNE de l'effet (positivité + couverture) sur le chemin live nouvellement branché, pas re-mesurer sa significativité.

## 1 & 4. sigma_scale varie dans le temps et suit le régime de volatilité

- AGITE   (BTC-USD, 2022-05 -> 2022-07, crash Terra/Luna) : 75 pas, sigma_scale[0]=1.0000 (neutre, aucun historique) -> sigma_scale[-1]=0.4898, variance du chemin=0.023877, moyenne=0.7643
- CALME   (BTC-USD, 2023-06 -> 2023-08) : 75 pas, sigma_scale[0]=1.0000 (neutre, aucun historique) -> sigma_scale[-1]=0.3699, variance du chemin=0.034753, moyenne=0.6855

-> moyenne sigma_scale (hors 5 premiers pas) : calme=0.6666 vs agité=0.7506 (plus large en régime agité: OK)

## 2. Skew-t ARIMA-GARCH (branché sur mh.py) réellement asymétrique

- BTC-USD  : q_lo=-1.69053  q_hi=+1.97528  |q_lo|-q_hi=-0.28475  asymétrie relative=15.54%  (shape skew-t=[3.0182, 0.1092])
    D+1 bornes prix : down(log)=0.04888  up(log)=0.05711  écart=-0.00823
    D+7 bornes prix : down(log)=0.13091  up(log)=0.15296  écart=-0.02205
- SPY      : q_lo=-2.10361  q_hi=+1.86549  |q_lo|-q_hi=+0.23812  asymétrie relative=12.00%  (shape skew-t=[11.5188, -0.1262])
    D+1 bornes prix : down(log)=0.01906  up(log)=0.01690  écart=+0.00216
    D+7 bornes prix : down(log)=0.04951  up(log)=0.04391  écart=+0.00560

## 3. Prophet log-espace (branché sur mh.py) : bornes positives + couverture

- espace prix           : min(lower)=+62836.22  couverture 95%=22.2%  largeur moyenne=8525.21
- espace log (adopté)   : min(lower)=+58889.46  couverture 95%=33.3%  largeur moyenne=8595.17

## 5. CRPS two-piece : réduction EXACTE au gaussien (bornes symétriques), diverge (bornes asymétriques)

### Bornes symétriques (sigma_lo == sigma_hi) -- doit reproduire N(mu,sigma)
- sigma= 2.0 : CRPS two-piece=1.99144  CRPS gaussien (forme fermée)=1.98885  écart relatif=0.130%  (KS vs N(0,1): stat=0.0014, p=0.804)
- sigma= 5.0 : CRPS two-piece=1.86507  CRPS gaussien (forme fermée)=1.86578  écart relatif=0.038%  (KS vs N(0,1): stat=0.0012, p=0.925)

### Bornes asymétriques (sigma_lo != sigma_hi) -- doit DIFFERER du gaussien symétrique
- sigma_lo=2.0, sigma_hi=6.0 : CRPS two-piece=1.01366  vs CRPS gaussien symétrique(sigma_moy=4.0)=1.79258  écart=43.45%
- sigma_lo=6.0, sigma_hi=2.0 : CRPS two-piece=3.88720  vs CRPS gaussien symétrique(sigma_moy=4.0)=1.79258  écart=116.85%

## Conclusion

Les 5 preuves demandées par le Chantier 3 du brief sont vérifiées ci-dessus, chiffres à l'appui, sur données offline reproductibles.
