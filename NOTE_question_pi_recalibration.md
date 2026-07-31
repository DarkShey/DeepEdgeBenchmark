# Question — `calibration/pi_recalibration.py`, encore utile ?

Posée le 31 juillet 2026, pas de canal direct pour demander autrement — laissée ici
pour quiconque passe par le dépôt.

## Ce qu'on a trouvé

`calibration/pi_recalibration.py` est présent en local, **jamais committé** (untracked
depuis avant le début de nos échanges sur la calibration des PI). C'est un diagnostic en
lecture seule qui recalibre les PI **déjà stockées dans `tracking.db`** (les prédictions
live, pas un backtest) via :
- deux corrections de forme fittées sur les résidus OOS glissants (Student-t / quantiles
  empiriques, walk-forward, sans lookahead) — même esprit que l'option 1 de
  `HANDOFF_dist_options_comparison.md`, mais un peu plus tôt et sur les prédictions
  réelles plutôt qu'un backtest ;
- une largeur réactive (`--reactive`) qui fait varier σ par une EWMA glissante des
  résidus au carré au lieu de l'écart-type sur tout l'historique.

Son propre docstring dit explicitement que c'est un diagnostic, et que « brancher la
méthode qui a l'air la meilleure dans le pipeline live est une étape séparée, plus
tard » — jamais fait, apparemment laissé en plan.

## La question

Ça recoupe très fortement ce qui vient d'être adopté dans `models/*.py`
(`HANDOFF_sigma_calibration_suivi.md`, σ dynamique par EWMA causale) — mais appliqué à
un endroit différent (post-hoc sur `tracking.db`, vs. dans le walk-forward du backtest
de chaque modèle).

- Est-ce un brouillon antérieur, devenu obsolète maintenant que l'EWMA est adoptée
  directement dans les modèles ?
- Ou est-ce la brique qui manque pour le point ouvert §8.1 de
  `HANDOFF_sigma_calibration_suivi.md` (« alimenter `next_step_prophet(sigma_scale=...)`
  depuis `tracking.db` ») — auquel cas il vaudrait le coup de le reprendre plutôt que
  d'en réécrire un ?

Je n'ai pas touché au fichier en attendant une réponse, pour éviter tout travail en
double.
