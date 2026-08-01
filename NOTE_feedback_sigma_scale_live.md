# Note — biais de rétroaction dans la boucle sigma_scale live (pour Maeva)

**Date : 2026-07-31. Contexte** : en câblant l'alimentation du chemin hebdomadaire
(`weekly_multimodel.run_model_asset`), j'ai dû choisir sur quelle bande mesurer le z
de l'EWMA — et ce choix a révélé un biais dans la boucle live D+1/D+7 existante.
Rien d'urgent (le sens de la correction reste bon), mais à corriger proprement.

## Le mécanisme

`validation/sigma_scale.py` mesure z = (y_true − y_pred)/σ_stocké, où σ_stocké vient
des bornes **écrites en base** — qui sont les bornes **déjà corrigées** par le facteur
appliqué ce jour-là. La boucle se mesure donc elle-même :

- facteur appliqué au jour t : s_t = √(s2_t), avec σ_stocké = s_t · σ_brut ;
- z observé : z_t = z_brut_t / s_t → mise à jour s2_{t+1} = λ·s2_t + (1−λ)·z_brut²/s2_t ;
- point fixe : s2* = √(E[z_brut²]), donc **facteur appliqué à l'équilibre =
  (E[z_brut²])^¼ au lieu de (E[z_brut²])^½**.

Concrètement : si les bandes brutes d'un modèle ont besoin d'un ×2, la boucle live
converge vers ×1,41 (~84 % de couverture au lieu de 95 pour une gaussienne). Mieux que
rien, mais sous-correction structurelle. Les backtests de validation (D+1, D+7, hebdo)
ne voient pas ce biais : ils mesurent z sur la bande **brute** (état indépendant de la
correction) — c'est pour ça que leurs chiffres sont meilleurs que ce que le live
atteindra à l'équilibre. Les tests de `test_sigma_scale.py` ne le voient pas non plus :
ils insèrent des bornes synthétiques non corrigées.

## Correctif proposé (léger)

Le champ `sigma_scale` appliqué est déjà tracé dans les records hebdo (nouveau champ,
cf. `run_model_asset`). Pour le live D+1/D+7 :

1. ajouter une colonne `sigma_scale_applied REAL DEFAULT 1.0` à `predictions`
   (migration idempotente comme `frequence`/`horizon_type`) et la remplir dans
   `_save_business_predictions` avec le facteur du jour ;
2. dans `sigma_scale.py`, dé-biaiser : z_brut = z_observé × sigma_scale_applied
   (une ligne dans la boucle EWMA ; les lignes historiques valent 1.0 → aucun
   changement rétroactif).

Alternative sans migration : reconstruire σ_brut = σ_stocké / facteur_du_jour en
rejouant la chaîne des facteurs — fragile, je la déconseille.

## En attendant

Le monitoring de couverture du dashboard (carte « Couverture réalisée ») verra de
toute façon la sous-correction résiduelle si elle devient matérielle : c'est le
garde-fou. La correction du biais rendra simplement l'équilibre exact.
