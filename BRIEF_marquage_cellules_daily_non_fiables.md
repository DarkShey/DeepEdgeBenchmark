# Mini-brief Claude Code — Marquage des cellules à défaut permanent de couverture (option 2, DECISION_derive_couverture_daily)

2026-08-10. Implémente l'option 2 actée dans DECISION_derive_couverture_daily.md — la seule recommandation validée encore non exécutée. Objectif : protection du consommateur — personne ne doit lire une cellule affichée « 95 % » dont la couverture réelle est 28,9 % (Prophet/BTC daily) sans avertissement. L'option 3 (diagnostic sigma hebdo → régime B multi-pas, lecture seule) reste hors de ce brief ; l'option 4 (régénérer la piste daily) reste disqualifiée.

## Principe : un marquage dérivé, jamais une liste codée en dur

Le critère est repris **tel quel** de la note de décision : une cellule (modèle, actif, régime, horizon) est à **défaut permanent** si sa couverture sur le **plein échantillon** de la piste est elle aussi hors de la bande [0,88 ; 0,99] — par opposition à une dérive, où seule la fenêtre glissante H3 sort de la bande. Le marquage est recalculé à chaque génération du dashboard depuis tracking.db (lecture seule) : si la piste évolue, le marquage suit. La liste des ~35 cellules attendues sert de **vérification**, pas de source.

Sous-catégoriser à l'affichage : sous-couverture (le cas dangereux — intervalle trop étroit, risque sous-estimé) vs sur-couverture (intervalle inutilement large — coûteux, pas dangereux). Les deux sont marquées, avec des libellés distincts.

## Ce qui est livré

1. **Brique de calcul** (`experiments/` ou module dashboard existant) : fonction `permanent_defect_cells(source, band=(0.88, 0.99))` → liste de cellules avec couverture plein échantillon, n origines, sens du défaut. Réutiliser la machinerie de coverage_monitor (mêmes lectures, bande déclarée au même endroit — une seule constante partagée, pas deux).
2. **Dashboard D7/W1** : badge visuel sur chaque cellule marquée (ex. « ⚠ non fiable — couvre X % sur tout l'échantillon » en sous-couverture ; « ◇ sur-couvert » sinon), entrée de légende, et compteur dans le bandeau de config (à côté de sampling_reference) : « N cellules marquées non fiables (critère DECISION_derive_couverture_daily) ». Le marquage s'applique aux deux pistes si le dashboard en affiche plusieurs ; à défaut, à la piste oos affichée.
3. **Tests unitaires** (minimum quatre) : (i) Prophet/BTC daily est détectée (le cas qui justifie la brique) ; (ii) une cellule saine connue n'est pas marquée ; (iii) la symétrie sous/sur-couverture est correcte sur cas construits ; (iv) la bande vit en un seul endroit — un test échoue si coverage_monitor et le marquage divergent sur sa valeur.
4. **Régénération du dashboard** et vérification : le compte des cellules marquées est rapproché des 35 attendues par la note de décision (écart toléré si la piste a bougé depuis, à documenter dans le JSON d'artefact) ; le HTML contient les badges ; `generated_at` mis à jour.
5. **Documentation** : une ligne dans DECISION_derive_couverture_daily.md — « option 2 : implémentée le … , N cellules marquées » (patch par contenu, standard habituel) — et mention dans le README du dashboard si existant.

## Garde-fous

- tracking.db en lecture seule — le marquage est une couche d'affichage, aucune ligne modifiée, aucune colonne ajoutée.
- Le badge n'est pas un verdict statistique : libellé factuel (« couverture observée X % sur N origines »), pas de p-value inventée. Le test formel reste Kupiec ; la bande est un déclencheur, comme pour H3.
- Ne pas marquer les cellules en simple dérive H3 (fenêtre glissante seule) : ce serait mélanger les deux problèmes que la note de décision a précisément séparés. La dérive reste du ressort du résumé de job quotidien.
- pytest vert avant/après (777 passed, 1 skipped actuel côté benchmark).
