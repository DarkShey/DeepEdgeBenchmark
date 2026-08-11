# Brief Claude Code — Méta-labeling artisanal : le CTA filtré par la taxonomie TC de NsDiff

2026-08-10. **Nouveau programme, décidé explicitement** — il ne rejoue ni la porte 0-bis (le CTA seul, clos) ni le sizing 4 bras (jamais atteint) : la question est celle du méta-labeling (Lopez de Prado 2018) dans sa version deux-étages complète — *un filtre dérivé de la géométrie prévision/PI de NsDiff (la taxonomie TC1.1-1.5) améliore-t-il la politique CTA en supprimant les faux signaux ?* Le méta-labeler ne prédit pas le marché ; il prédit quand le primaire est fiable. Cadre théorique : prédiction sélective (El-Yaniv-Wiener — discipline risque-couverture) + méta-labeling. **Fenêtre inchangée** (2020-2026, grille oos2020 weekly, 340 origines) — l'interdit tient toujours.

**Zéro apprentissage, zéro génération** (garde-fou paramètres/données) : le filtre est un jeu de règles fixé ci-dessous ; les états TC de NsDiff se dérivent des lignes weekly existantes en base (last_close, y_pred, y_lower, y_upper de l'ensemble 5×200) ; le signal CTA v2 gelé et hashé existe. Ce chantier est une jointure et un backtest, pas un modèle.

## 1. Les deux étages, déclarés

**Primaire** : CTA corrigé (direction Hull, v2-2026-08-08), signe long/short, cadence hebdomadaire, détention W+1.

**Secondaire (méta-filtre)** : l'état TC de NsDiff à la même origine, calculé en appliquant les règles de `sim_trades.py` **telles quelles** à la géométrie weekly W+1 (ref = last_close, predicted = médiane ensemble, pi_low/pi_high = quantiles 2,5/97,5) — les règles sont des fonctions de (ref, predicted, pi_low, pi_high), agnostiques à l'horizon. États possibles : Bull-Calm, Bull-Stress, Bear-Calm, Bear-Stress, Sideways, aucun.

**Matrice de décision, fixée a priori — deux filtres, pas de balayage :**

| État NsDiff \ Signal CTA | CTA long | CTA short |
|---|---|---|
| Bull (calm/stress) | prendre | **F1 : prendre · F2 : veto** (contradiction) |
| Bear (calm/stress) | **F1 : prendre · F2 : veto** (contradiction) | prendre |
| Sideways | F1 : prendre · F2 : veto | F1 : prendre · F2 : veto |
| Aucun état | prendre (le filtre ne sait pas → ne bloque pas) | prendre |

Correction de lecture : **F1 = veto sur contradiction seule** (Bull×short, Bear×long) ; **F2 = agir uniquement sur concordance** (Bull×long, Bear×short) — sideways et contradiction vetoés. F1 est le filtre faible, F2 le strict. Taille fixe |w| = 1 sur les trades pris — le sizing n'est pas la question de ce brief, on isole le filtre.

## 2. Les bras du backtest (politique complète, weekly, net de frais real_fees + roulement)

| Bras | Définition | Rôle |
|---|---|---|
| C0 | CTA seul, tous signaux pris | baseline (résultat connu : négatif vs B&H) |
| F1 | CTA filtré faible (NsDiff) | le candidat conservateur |
| F2 | CTA filtré strict (NsDiff) | le candidat méta-labeling |
| G1/G2 | mêmes filtres construits sur la géométrie **GARCH** | attribution : « est-ce NsDiff, ou n'importe quel PI ? » |
| R | filtre **placebo** : veto aléatoire à couverture égale à F2 (graine fixée, 100 tirages) | le contrôle canonique du méta-labeling |

Le placebo est décisif : un filtre qui retire des trades au hasard améliore parfois le PnL par chance. F2 ne « marche » que s'il bat R **à couverture égale** — la discipline risque-couverture d'El-Yaniv appliquée au trading.

## 3. Les deux lectures, chacune avec son instrument (la leçon de la session)

- **Lecture sélective (TC-style, descriptive)** : qualité des trades *pris* vs *vetoés* — le filtre retire-t-il réellement des mauvais trades ? Hit-rate et PnL moyen des deux sous-ensembles, par état TC et par régime de marché (2020/2022/2024-26). Toujours rapportée en couple (couverture, qualité), jamais l'un sans l'autre.
- **Lecture politique complète (décisionnelle)** : PnL net par origine du portefeuille équipondéré, apparié contre C0 et contre B&H, bootstrap par blocs, l'abstention payant son coût d'opportunité.

## 4. Hypothèse primaire, famille, puissance

- **Hypothèse primaire** : F2 bat C0 en PnL net par origine, portefeuille équipondéré weekly W+1, coût central — ET F2 bat le placebo R à couverture égale (les deux conditions, conjonctives).
- **Famille de Holm (m = 2)** : {F1 vs C0, F2 vs C0}, portefeuille. G1/G2, par-actif, par-état : exploratoires, étiquetés tels quels.
- **Analyse de puissance avant les runs** (SE bootstrap existants) : chiffrer l'écart détectable à effective_n ≈ 118, en tenant compte du fait que F2 ne diffère de C0 que sur les origines vetoées — la puissance effective porte sur ce sous-ensemble ; si la couverture du veto est trop faible pour détecter quoi que ce soit, le dire avant de lancer.
- Attentes pré-déclarées : les états Stress seront quasi absents (drift ≪ largeur — connu) ; le filtre vivra donc surtout sur Calm et Sideways. Si le veto Sideways de F2 est le seul actif, la conclusion d'attribution se joue sur G2 (la largeur GARCH fait-elle pareil ?).

## 5. Critères d'arrêt, déclarés

1. Ni F1 ni F2 ne survit à Holm, OU F2 ne bat pas le placebo à couverture égale → **le méta-labeling artisanal est clos**, et la version apprise (modèle secondaire entraîné, Lopez de Prado complet) est **explicitement interdite en suite directe** — elle aurait moins de données par état et plus de paramètres ; ce serait chercher dans un sous-espace plus riche ce que le sous-espace simple vient de refuser.
2. Hypothèse primaire positive → un seul brief suivant autorisé : réplication sur la seconde moitié temporelle de la grille + G-contrôles complets, avant toute conclusion.
3. Dans tous les cas : pas de nouveau filtre, seuil ou matrice après lecture des résultats ; les deux filtres de ce brief sont les seuls.

## Non-négociables (inchangés)

Règles TC importées de `sim_trades.py` sans modification ; conventions descriptif / poolé ; tous seuils et matrices fixés dans ce brief ; briques réutilisées (econ_backtest, paired_test, real_fees + roulement, multiple_testing, signal CTA v2 hashé, lignes NsDiff/GARCH en base — lecture seule) ; code neuf couvert de tests unitaires (dont : la matrice de décision sur cas construits, et le placebo à couverture exactement égale) ; tracking.db lecture seule ; pytest vert avant/après (777 passed, 1 skipped côté benchmark).
