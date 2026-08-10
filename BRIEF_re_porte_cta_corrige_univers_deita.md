# Brief Claude Code — Porte 0-bis : le CTA corrigé, jugé dans son habitat (univers DEITA complet)

2026-08-08. **Réouverture explicite et décidée** — elle ne contredit pas le critère d'arrêt de NOTE_couplage_cta_deita_sizing_nsdiff.md, elle s'appuie sur trois faits nouveaux, tous acquis pour des raisons mécaniques et non ajustés sur des résultats : (i) le signal a été réparé (bug 1 conviction-singleton, bug 2 calendrier ffill — TICKET_DEITA_cta_quant_engine_RESOLUTION.md), (ii) l'instrument a été réparé (branche 2 sur l'excès, PATCH_gate0_branche2_et_holm_m2.md), (iii) le périmètre s'élargit sur une limite **déclarée avant le verdict** : « un panel de 7 actifs est petit pour un CTA — le trend vit sur des dizaines de marchés, la diversification fait l'essentiel du Sharpe ». Le verdict de la porte 0 d'origine reste clos et n'est pas réécrit ; ceci est la porte 0-bis, un programme distinct.

**Interdit inchangé : la fenêtre.** 2020-01 → 2026-07, mêmes origines que la grille oos2020. Élargir l'univers sur une justification pré-existante est légitime ; changer de fenêtre après avoir vu les résultats ne l'est pas.

## Chantier 0-bis-A — Données : un socle de prix unique pour le signal ET le PnL

La leçon de la porte 0 : la base de prix DEITA et prices_v3 divergent (2,6e-03 sur SPY) — un signal produit sur une série et tradé sur une autre est interdit.

- Construire **prices_v4** : l'univers DEITA complet (16 actifs) + le panel benchmark, téléchargé par le même pipeline que prices_v3 (source unique, calendriers de cotation propres), gelé, hashé. Départ : le plus long historique commun disponible par actif, borné à 2011-05 comme prices_v3.
- Substitutions à acter et documenter : si la base DEITA utilise des futures (GC=F, CL=F) là où le benchmark a des ETF (GLD, USO), l'univers 0-bis prend **une seule** convention par exposition, déclarée dans le brief d'exécution avant tout calcul — pas les deux.
- Le signal CTA (moteur corrigé : bugs 1 et 2 actifs, drapeaux par défaut) est **recalculé sur prices_v4**, gelé, hashé ; vérification de look-ahead identique à la porte 0 (recalcul tronqué sur 5 origines, égalité exacte, contre-épreuve par fuite injectée).
- Frais : étendre real_fees aux nouveaux instruments, ordres de grandeur déclarés par familles (ETF actions/obligations 2-5 bps, futures 1-2 bps + roulement H2, crypto 10-60 bps), un seul fichier, avant les runs.

## Chantier 0-bis-B — La porte, avec l'instrument corrigé

CTA seul, taille fixe |w| = 1, weekly, W+1, net de frais au niveau central. Comparaison appariée vs B&H, bootstrap par blocs, sur les ~340 origines.

**Critère de porte, déclaré (deux branches, l'une OU l'autre) :**

1. **Branche 1 (inchangée)** : au moins un actif à PnL net positif ET p < 0,05 brut vs B&H.
2. **Branche 2 (version corrigée uniquement)** : Sharpe poolé de l'**excès** (PnL signal − PnL B&H par origine, portefeuille équipondéré) > 0, ET excès positif sur ≥ 3 classes d'actifs sur 4 (classes déclarées dans le brief d'exécution : actions / taux / matières premières–or / crypto ; le mapping des 16 actifs est figé avant les runs).
3. Lecture complémentaire déclarée, non décisionnelle : le **portefeuille diversifié** est la vraie unité d'un CTA — Sharpe de l'excès du portefeuille équipondéré rapporté avec son IC bootstrap, et comportement 2020/2022 par tranche (le crisis alpha restauré par le correctif calendrier doit se voir en mars 2020, sinon nulle part).

**Si la porte échoue : clôture définitive du dossier trading, toute la ligne** (recommandation R5 actée) — signal corrigé, instrument corrigé, habitat naturel, fenêtre de 6,5 ans traversant trois régimes : il n'y aura pas de porte 0-ter. La pile DEITA reste son CTA corrigé, sans couplage.

## Chantier 1 (conditionnel) — Le sizing 4 bras, inchangé, sur le panel

Si la porte passe, le chantier 1 du BRIEF_couplage_cta_deita_sizing_nsdiff.md s'applique **tel quel** — bras A/B/C/D, budget de risque commun à vol cible 10 %, hypothèse primaire A vs B sur {SPY-ES, ZN-FUT} × {W+1}, famille de Holm m = 2, analyse de puissance préalable, critères d'arrêt 2-4 inchangés. Une seule adaptation : le signal directionnel est celui du moteur corrigé sur prices_v4, restreint aux instruments du panel (les nuages NsDiff n'existent que là — aucun nouveau modèle n'est entraîné, garde-fou 1 inchangé).

Lucidité à écrire dans la note d'exécution : sur la grille longue, NsDiff est le modèle le moins bien couvert du benchmark ; l'issue « D ≈ A, la vol conditionnelle suffit, NsDiff n'ajoute rien » est plausible et serait une conclusion propre. Dans ce cas la pile de production recommandée est CTA corrigé + vol targeting EWMA/GARCH, et NsDiff sort définitivement du périmètre trading.

## Ce que ce brief ne rouvre pas

Le duel NsDiff vs GARCH (clos), la famille directionnelle F3 (close), le mensuel (hors périmètre), le conditionnement exogène de NsDiff (fermé tant que le critère 3 du brief couplage n'a pas abouti), et la fenêtre temporelle. Les bordereaux de frais réels ne deviennent exigibles que si le chantier 1 tourne et qu'un edge net doit être certifié.

## Non-négociables (inchangés)

Conventions descriptif / poolé ; seuils, classes, mapping d'univers et grille de frais déclarés avant les runs ; briques réutilisées (cta_gate0 corrigé, econ_backtest, paired_test, real_fees + roulement, multiple_testing) ; code neuf couvert de tests unitaires ; tracking.db lecture seule ; prix gelés prices_v4 partagés par le signal et le PnL ; signal et moteur hashés (manifest) ; pytest vert avant/après (759 passed, 1 skipped côté benchmark ; 173 dans le périmètre CTA côté DEITA).
