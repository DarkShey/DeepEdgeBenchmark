# Réponse — `calibration/pi_recalibration.py` : partiellement dépassé, à garder en diagnostic

**Date : 2026-07-31. Répond à `NOTE_question_pi_recalibration.md`** (le fichier n'étant
que sur ta machine, je réponds d'après ta description — précise si le script fait plus
que ce qui y est décrit).

## Réponse courte

Les deux hypothèses de ta note sont vraies chacune à moitié :

1. **Sa largeur réactive (EWMA) est dépassée** — pas par l'adoption dans `models/*.py`,
   mais par la brique live que tu cherchais : le point §8.1 du HANDOFF **a été
   implémenté depuis** par Maeva (`validation/sigma_scale.py` + câblage
   `pipeline._build_sigma_scale_map`, commit `e0ee61c`), puis étendu au D+7 avec un λ
   ajusté au recouvrement (`9b99783`) et au walk-forward hebdo (`85a59c1`). C'est
   exactement « une EWMA glissante des résidus au carré depuis tracking.db », causale
   (prédictions résolues uniquement), avec flag de réversibilité. Le réécrire depuis
   ton script n'a plus d'objet.

2. **Ses corrections de forme restent la partie intéressante** — et ne sont couvertes
   par rien de ce qui est adopté. Nous n'avons adopté que des corrections de **niveau**
   (EWMA), plus le skew-t natif d'ARIMA-GARCH. Or il reste un vrai problème de **forme
   centrale** : après log+EWMA, Prophet sur ZN/TLT tient son 95 % mais sous-couvre à
   50 % (cov ≈ 28-32 %, cf. `NOTE_reponse_prophet_zn_tlt.md`) — un centre trop étroit
   qu'aucun facteur multiplicatif ne peut corriger (il bouge les trois niveaux
   ensemble). Tes fits Student-t / quantiles empiriques **glissants sur les résidus OOS
   réels** sont précisément l'outil pour explorer ça — et c'est différent de l'option 1
   du comparatif (forme statique fittée une fois, qui ne généralisait pas d'une fenêtre
   à l'autre) : une forme *glissante* pourrait généraliser là où la statique échoue.
   Personne n'a testé cette combinaison forme-glissante × EWMA-niveau.

## Recommandation concrète

- **Committe-le** (en l'état, comme diagnostic — il est untracked, c'est le seul
  exemplaire) plutôt que de le laisser en plan.
- **Ne le branche pas en live** : la boucle live existe (`sigma_scale.py`) ; s'il faut
  un jour brancher une correction de forme, elle passera par le même chemin
  (`pipeline._build_sigma_scale_map` généralisé à des quantiles par niveau).
- **Deux mises à niveau nécessaires avant de s'en servir** comme diagnostic :
  1. mesurer les z sur la bande **brute** : depuis l'activation live, les bornes
     stockées sont déjà corrigées — un diagnostic qui lit les bornes stockées se mesure
     lui-même (biais de point fixe en racine quatrième, détaillé dans
     `NOTE_feedback_sigma_scale_live.md`). Divise par `sigma_scale_applied` (colonne en
     cours d'ajout, cf. la note) pour retrouver la bande brute ;
  2. scorer en MACE stricte 50/80/95 (pas seulement la couverture 95) — c'est au
     niveau 50 que le problème de forme restant est visible.
- **Cas d'usage cible** : Prophet ZN/TLT (et éventuellement les z du LSTM), forme
  glissante par-dessus l'EWMA de niveau — si ça ferme l'écart cov50 sans casser le 95,
  c'est un candidat d'adoption, à valider multi-fenêtres comme le reste.
