# Brief Claude Code — Couplage CTA (DEITA) × sizing NsDiff : la direction au trend, la taille à la fourchette

2026-08-08. **Nouveau programme**, pas une réouverture : le volet « NsDiff prédit-il ? » est clos définitivement (SYNTHESE_finale_programme_nsdiff.md) et le reste. La question posée ici est différente et n'a jamais été testée : *une échelle de risque calibrée (la largeur prédictive de NsDiff) améliore-t-elle le sizing d'un signal directionnel externe (le CTA de DEITA), net de frais, par rapport au sizing standard par volatilité réalisée ?* Littérature d'appui : Moskowitz-Ooi-Pedersen 2012 (time-series momentum), Moreira-Muir 2017 / Harvey et al. 2018 (volatility targeting).

## Les trois garde-fous, traduits en contraintes de design

1. **Ratio paramètres/données (la leçon TsDiff)** : ce programme n'entraîne AUCUN nouveau modèle. Le couplage est au niveau décision, pas au niveau modèle — les nuages NsDiff (ensemble 5×200, grille oos2020) existent sur disque, le signal CTA vient de DEITA tel quel. Le conditionnement exogène de NsDiff (régime en entrée du réseau) est explicitement **hors périmètre** ; il ne serait envisagé que si ce programme-ci conclut positivement, dans un brief futur distinct.
2. **Décroissance du momentum post-2010** : le signal CTA doit prouver qu'il vit encore sur cette fenêtre avant tout couplage. C'est le chantier 0, bloquant — pas de sizing d'un signal mort.
3. **Programme neuf, hypothèse primaire pré-déclarée** : la comparaison n'est plus NsDiff vs GARCH mais sizing-NsDiff vs sizing-naïf, à signal directionnel identique. Hypothèse primaire, familles de Holm et critères d'arrêt déclarés ci-dessous, avant tout run.

## Chantier 0 — Porte d'entrée : le signal CTA-DEITA a-t-il un edge sur la fenêtre ? (bloquant)

- **Interface DEITA** : importer le signal directionnel (signe et éventuellement force du trend, et/ou état de régime) comme série gelée point-in-time, un fichier par actif, hash enregistré. Vérification bloquante d'absence de look-ahead : le signal à l'origine t ne dépend que de données ≤ t (rejouer 5 origines à la main depuis les données brutes et comparer).
- **Test** : CTA seul, taille fixe (|w| = 1), sur la grille oos2020 (340 origines, 7 actifs, weekly), PnL net de frais (real_fees) vs B&H, bootstrap par blocs. En complément descriptif : hit-rate, Sharpe, comportement 2020 et 2022 (le trend doit montrer son « crisis alpha » là ou nulle part).
- **Critère de porte, déclaré** : au moins un actif avec PnL net positif et p < 0,05 brut (pas de Holm ici — c'est une porte, pas une conclusion), OU un Sharpe poolé > 0 avec direction cohérente sur les classes d'actifs. Sinon : le momentum est mort sur cette fenêtre pour ce panel, le programme s'arrête avant tout couplage, et c'est la conclusion.

## Chantier 1 — Les quatre bras de sizing, même signal partout

Le signe de la position est identique dans les quatre bras (le CTA-DEITA). Seule la **taille** change. |w| ≤ 1, sans levier, rebalancement hebdomadaire, horizon de détention W+1 (les horizons W+2/W+3 en exploratoire).

| Bras | Taille de position | Ce qu'il teste |
|---|---|---|
| A (candidat) | budget de risque / largeur prédictive NsDiff (ensemble 1000, échelle interquantile 10-90) | l'échelle calibrée apprise |
| B (baseline principale) | budget de risque / vol réalisée EWMA (même lambda que la brique sigma existante) | le vol targeting standard — c'est LUI qu'il faut battre |
| C (contrôle bas) | taille fixe | le CTA brut — mesure ce que tout sizing apporte |
| D (contrôle d'attribution) | budget de risque / largeur GARCH | « est-ce NsDiff, ou n'importe quelle vol conditionnelle ? » |

Budget de risque identique pour A, B, D (calibré pour une vol cible commune, déclarée, ex. 10 % annualisée), afin que les expositions moyennes soient comparables — sinon on re-mesure l'exposition, pas l'échelle (la leçon var_limit vs B&H du chantier éco).

## Hypothèse primaire et familles

- **Hypothèse primaire** : bras A bat bras B en PnL net par origine — SPY-ES, weekly, W+1, coût central. Test apparié bootstrap par blocs, unité = origine, poolé multi-graines côté NsDiff (le nuage ensemble est déjà multi-graines).
- **Famille de Holm primaire (m = 2)** : A vs B sur {SPY-ES, ZN-FUT} × {W+1}, les deux instruments à frais bas. *(m = 2, corrigé le 2026-08-08 avant tout run — le brief annonçait m = 4 pour une famille qui en compte 2 ; cf. NOTE §3 et PATCH_gate0_branche2_et_holm_m2.md. La famille n'est pas étendue à W+2/W+3 pour justifier le 4 : W+1 est l'horizon de détention déclaré.)* Tout le reste — autres actifs, W+2/W+3, A vs C, A vs D, Sharpe, drawdown — est exploratoire et étiqueté tel quel.
- **Lecture de D, déclarée d'avance** : si A bat B mais pas D (ou D ≈ A), la conclusion est « la vol conditionnelle améliore le sizing, NsDiff n'y ajoute rien de propre » — résultat utile, pas un échec du protocole.
- **Analyse de puissance avant les runs** (méthode du chantier 2 existant, SE bootstrap en 1/√n) : chiffrer l'écart A−B détectable à effective_n ≈ 118. Si l'écart plausible (quelques bps) est indétectable, le dire avant de lancer et dimensionner les attentes — le Sharpe descriptif portera alors l'essentiel de la lecture, et la note devra le dire.

## Chantier 2 — Exécution et réalisme

- Grille oos2020 uniquement (340 origines, prix prices_v3 gelés), frais real_fees par instrument aux trois niveaux, coût de roulement H2 inclus pour les futures. Le turnover diffère entre bras (une taille qui bouge plus paie plus) : le moteur econ_backtest le facture déjà, le vérifier par un test unitaire dédié (position constante vs oscillante).
- Monitoring H3 branché sur les bras (couverture n'a pas de sens ici, mais PnL glissant 26 origines par bras — même mécanique, métrique adaptée) pour la lecture par régime (2020 / 2022 / 2024-2026).
- Aucune écriture dans tracking.db : artefacts JSON isolés, comme les chantiers précédents.

## Critères d'arrêt, déclarés

1. Chantier 0 négatif → arrêt complet, conclusion « pas de signal CTA exploitable sur la fenêtre ».
2. Hypothèse primaire négative sous Holm → le couplage sizing-NsDiff est clos ; pas de re-test à d'autres niveaux, budgets ou fenêtres. Les exploratoires sont rapportés mais ne rouvrent rien.
3. Hypothèse primaire positive → un seul brief suivant est autorisé : réplication sur la seconde moitié temporelle de la grille + extension aux autres instruments, avant toute conclusion de déploiement.
4. Dans tous les cas, le conditionnement exogène de NsDiff (garde-fou 1) reste fermé tant que 3 n'a pas abouti.

## Non-négociables (inchangés)

Conventions descriptif / poolé ; familles de Holm et tous seuils/budgets/lambdas déclarés avant les runs (ceux de ce brief font foi) ; briques réutilisées (econ_backtest, paired_test, real_fees + roll, multiple_testing, nsdiff_production_spec, machinerie sigma EWMA) ; code neuf couvert de tests unitaires ; tracking.db lecture seule ; prix gelés prices_v3 partagés ; signal DEITA gelé et hashé ; pytest vert avant/après (729 passed, 1 skipped actuel).
