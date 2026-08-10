# SYNTHÈSE DE CLÔTURE — Dossier trading DeepEdgeBenchmark × DEITA

2026-08-08. Document chapeau : il clôt l'ensemble du dossier trading ouvert par le programme NsDiff et fermé par la porte 0-bis. Chaque verdict ci-dessous a sa note source ; ce document n'ajoute aucun chiffre nouveau, il assemble et fixe l'état final.

## Le verdict global, en une phrase

Sur la fenêtre 2020-2026, aux horizons 1-3 semaines, sur un univers allant de 5 à 18 actifs, **aucun des candidats testés — diffusion (NsDiff), volatilité classique (GARCH), trend-following (CTA-DEITA corrigé) — ne bat le simple fait de détenir le marché, net de frais** ; les quatre programmes qui l'établissent ont chacun été fermés par un critère d'arrêt déclaré avant les runs, et la dernière fermeture est une preuve contre (IC entièrement négatif), pas une absence de preuve.

## Les quatre programmes, et comment chacun s'est fermé

**1. NsDiff comme modèle de prévision** (consolidation daily/weekly → duel TsDiff → grille longue). Question : la diffusion apporte-t-elle une meilleure prévision probabiliste ? Réponse : NsDiff est la meilleure diffusion testée (TsDiff dominé 29/0 au Winkler, retiré — sur-paramétrisation structurelle, 525 paramètres par fenêtre weekly), mais son seul avantage défendable, la calibration, s'est rétréci à chaque gain de rigueur : le Winkler était un artefact de n_samples=50 ; la couverture s'est renversée sur la grille longue (Cov95 0,929, dernier des six modèles, contre 0,947 pour GARCH). Fermeture : hypothèse primaire 0/4 même non corrigée, sur effective_n≈118, contre un champion vérifié à sa meilleure loi (gaussien ≡ skew-t). *NsDiff archivé comme référence de recherche.*

**2. La valeur économique des intervalles** (backtest éco → edge vs frais → régénération). Question : des intervalles calibrés rapportent-ils de l'argent ? Réponse : non — 0 survivant à Holm sur 54 familles, edge non répliqué entre grilles (+1,33 → +0,33 bps ; l'edge crypto daily change de signe), frais ≈ 2 % de l'edge différentiel (l'argument des coûts ne tenait pas la question de déploiement — c'est la puissance qui tranche, et elle a tranché contre). Les 4 seuls survivants du dossier disaient « B&H bat la stratégie ». Fermeture : critère d'arrêt du brief régénération, atteint et confirmé.

**3. Le signal directionnel dans les intervalles** (famille 3 → F3 reposée). Question : la fourchette contient-elle une direction ? Réponse : non, par fait mécanique — à 1-3 semaines, le drift franchit si rarement un quart de la largeur interquantile (0,4 % d'émissions) qu'aucune stratégie ne peut s'y construire. Vrai pour NsDiff comme pour GARCH. Fermeture : critère de clôture déclaré, sans autre reformulation autorisée.

**4. Le couplage CTA × sizing** (porte 0 → corrections → porte 0-bis). Question : un signal directionnel externe, dimensionné par une échelle de risque calibrée, bat-il le vol targeting standard ? Réponse : la question n'a jamais été atteinte — le signal lui-même n'a pas passé sa porte, deux fois. Porte 0 : signal dégénéré (conviction-carré) et branche 2 défectueuse. Porte 0-bis, tout réparé, univers complet de 18 actifs : excès du portefeuille diversifié −51,77 bps/origine, IC95 [−1,79 ; −0,22], 1 % des tirages positifs ; crisis alpha absent en 2020 (pire tranche, −116 bps) comme en 2022 (+10,6 seulement) ; Sharpe OOS causal du CTA ≈ 0. Fermeture : critère de porte, sur les deux branches, avec engagement écrit qu'il n'y aura pas de porte 0-ter.

## Les trois bugs de production — la vraie valeur du dossier

Découverts parce que la discipline du benchmark a été appliquée à du code qu'elle n'auditait pas :

1. **Conviction-carré** (`_conviction_level`) : pour tout actif seul dans son sous-secteur, la conviction ne changeait jamais de signe — 0,06 % de SELL sur les jours-actifs, la vente n'était pas rare, elle était absente. Le pipeline amplifiait les achats et neutralisait les ventes ; la décision BUY/HOLD/SELL change sur 35 % des jours-actifs après correction.
2. **Calendrier ffill** : les actifs 5 j/7 héritaient deux jours à rendement nul par semaine dans la moyenne de Hull — trend dilué et retardé précisément dans les krachs (crisis alpha mars 2020 : +84 bps → −2 selon le calendrier).
3. **Look-ahead d'un jour dans le harnais de validation DEITA** : poids de t × rendement de t. Le Sharpe OOS « historique » de 0,83 devient −0,01 une fois causal. Trouvé parce qu'un Sharpe de 3,95 est invraisemblable et qu'on l'a vérifié au lieu de le publier.

Conséquence : tous les backtests historiques DEITA antérieurs aux correctifs sont caducs, et la validation qui les jugeait aussi. Recommandation la plus urgente du dossier, hors backtest : **si le CTA v1 tourne en production, l'arrêter** ; l'adoption du v2 (`CTA_SIGNAL_VERSION = "v2-2026-08-08"`) est une décision métier, en sachant que son edge démontré est nul — honnêtement nul, ce qui est un progrès sur un edge fictif de 0,83.

## Ce que le dossier laisse en état de marche

ARIMA-GARCH (gaussien, config actée) comme référence d'intervalles du benchmark ; le dashboard sur la config production ensemble 5×200 (Cov95 0,9452) ; le monitoring de couverture glissante H3 en routine quotidienne, avec la caractérisation des 35 défauts permanents daily des modèles de référence (décision de marquage prise, option 4 disqualifiée) ; les socles de prix gelés prices_v3/prices_v4 (18 actifs) et la grille oos2020 à effective_n≈118 ; le moteur CTA corrigé et hashé ; 777 tests côté benchmark, 101 dans le périmètre CTA côté DEITA. L'infrastructure survit aux verdicts — c'est elle qui permettra de juger vite le prochain candidat.

## Ce qui rouvrirait quelque chose — et ce qui ne rouvre rien

Aucune réouverture n'est autorisée dans les périmètres testés : ni fenêtre, ni niveau, ni budget, ni reformulation des familles closes. Serait un **programme nouveau**, avec sa porte et ses critères propres : d'autres horizons (intraday, où la microstructure change la question ; pluriannuel, où la littérature situe la vraie prédictibilité), une autre classe de stratégies (portage, valeur, cross-sectionnel — rien à voir avec les intervalles), ou le conditionnement exogène de NsDiff — resté verrouillé par son garde-fou, faute de couplage positif. Les bordereaux de frais réels ne sont exigibles qu'au service d'un tel programme. Le prior de chacun se discute à la lumière de Welch-Goyal et de M6 : faible.

## Les leçons, confirmées sur quatre programmes

1. Les verdicts single-seed et petits échantillons ne survivent pas — tout ce qui a été conclu à n=50 tirages ou une graine a été renversé par la mesure propre.
2. La puissance ne crée pas d'avantage, elle révèle ceux qui existent — ici celui de GARCH, puis celui de B&H.
3. Les critères d'arrêt déclarés avant les runs sont ce qui permet de conclure — quatre fermetures, zéro prolongation négociée après coup.
4. Vérifier l'équivalence plutôt que l'affirmer trouve des bugs — trois bugs de production, tous découverts par des contrôles que rien n'obligeait à faire.
5. Un chiffre invraisemblable est une alarme, pas un résultat — le Sharpe 3,95 a déclenché la vérification qui a trouvé le look-ahead.

## Registre des documents

| Programme | Source de vérité |
|---|---|
| NsDiff prévision/calibration | SYNTHESE_finale_programme_nsdiff.md ← NOTE_nsdiff_consolidation…, NOTE_duel_nsdiff_vs_tsdiff…, NOTE_nsdiff_regeneration_oos_et_famille3.md |
| Valeur économique | NOTE_nsdiff_backtest_eco_et_recadrage.md, NOTE_nsdiff_edge_vs_frais.md |
| Signal directionnel (F3) | NOTE_nsdiff_regeneration_oos_et_famille3.md §6 |
| Couplage CTA × sizing | NOTE_couplage_cta_deita_sizing_nsdiff.md, porte 0-bis (note d'exécution), PATCH_gate0_branche2_et_holm_m2.md |
| Bugs DEITA | TICKET_DEITA_cta_quant_engine.md + _RESOLUTION.md, R1_revue_diff_et_revalidation.md |
| Entretien benchmark | DECISION_derive_couverture_daily.md, runs 20260806-oos-repoint-m200 et 20260808-oos-repoint-ensemble |
