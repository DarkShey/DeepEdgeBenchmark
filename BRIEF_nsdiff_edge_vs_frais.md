# Brief Claude Code — NsDiff : rouvrir la question économique par le rapport edge/frais

2026-08-07. Fait suite à NOTE_nsdiff_backtest_eco_et_recadrage.md (NO-GO trading : edge de +2 à +5 bps/origine, sous les frais testés de 10-60 bps) et à la Note de décision du 7/8. Le NO-GO est « pas prouvé rentable », pas « prouvé inutile » : ce brief attaque les trois leviers qui peuvent le faire évoluer, par rendement attendu décroissant. Référence actée : n_samples=200, multi-graines 42-46, config production = ensemble 5×200.

## Point 0 — hygiène documentaire (préalable, 10 minutes)

L'état du dashboard a été vérifié en base le 7/8 : la bascule 50→200 **a été exécutée** le 6/8 (`repoint_oos_to_m200.json` : applied=true, run `20260806-oos-repoint-m200`, sauvegarde `tracking.db.bak_repoint_m200_2026-08-06T064808`, couverture 0,909→0,9315). La Note de décision dit encore « single-seed/50, scripts non exécutés » — la corriger : le dashboard est en graine 42 × 200 tirages ; seule l'intégration de l'ensemble 5×200 au dashboard reste ouverte. Harmoniser aussi effective_n mensuel (12 vs 13) entre les deux documents, et ajouter à la note les résultats B3 (pricing d'option : GARCH 7/10) et A3-ii (refit ×24,6 : aucun verdict déplacé).

## Chantier 1 — Frais réels et instruments à bas coûts (le seul levier qui peut inverser le NO-GO)

L'edge mesuré (+2 à +5 bps/origine, le plus net en weekly avec var_limit) n'est pas exploitable à 10-60 bps de frais forfaitaires. La question devient : existe-t-il un univers d'exécution où frais < edge ?

### 1.1 Grille de frais réels par instrument

Remplacer les 3 niveaux forfaitaires par des frais réalistes par actif, documentés et sourcés dans la note : futures (ES pour l'exposition actions, ZN pour les taux : de l'ordre de 1-2 bps l'aller-retour, spread inclus), ETF au comptant (SPY/TLT : ~2-5 bps selon le courtier), crypto (BTC/ETH spot : 10-60 bps selon la plateforme, à garder comme borne haute). Chaque hypothèse de frais est un paramètre déclaré du run, jamais ajusté après lecture des résultats.

### 1.2 Rejouer le chantier B sur cet univers

- Mêmes 3 stratégies, mêmes horizons, config production (ensemble 5×200) côté NsDiff, GARCH refit par origine — protocoles naturels, asymétrie déclarée.
- Même discipline : familles Holm déclarées avant les runs, bootstrap par blocs + SPA, tous soumis à la correction.
- Question posée au run, à écrire telle quelle dans la note : « au niveau de frais réel de chaque instrument, l'edge net par origine devient-il positif et le PnL distinguable de B&H / de GARCH ? »
- Sortie attendue : matrice instrument × stratégie × horizon avec edge brut, frais, edge net — la décision se lit sur l'edge net, la p-value vient ensuite.

### 1.3 Re-test déclaré de la famille 3 (PI exclut zéro) à niveau plus étroit

À 95 %, zéro signal sur 2 700 origines (résultat structurel : un PI à 95 % contient toujours le prix courant à ces horizons). Le rejeu à niveau plus étroit avait été volontairement non fait pour ne pas choisir un seuil après coup. Le déclarer maintenant, a priori, dans ce brief : **niveau 80 %** (et lui seul), justification : c'est le premier niveau standard où la largeur des PI observés (médiane ~3-6 % du prix en weekly) laisse mécaniquement la possibilité d'exclure zéro. Un seul niveau, pas de balayage — si 80 % ne produit pas de signaux exploitables nets de frais, la famille est close.

## Chantier 2 — Puissance : plus de données

À effective_n=30, un vrai edge de 3 bps est indétectable et « indistinguable » ne conclut rien. Objectif : faire passer les verdicts de « pas prouvé » à « tranché », dans un sens ou l'autre.

- **Plus d'origines** : étendre la grille oos vers le passé autant que l'historique le permet (la limite actuelle de 90 origines vient de la fenêtre 2024-2026 ; chiffrer ce que donnerait 2020-2026 en origines et en effective_n, régimes de marché traversés à documenter — un backtest qui inclut 2022 ne raconte pas la même histoire que 2024-2026).
- **Plus d'actifs** : ajouter 3-5 instruments liquides à frais bas alignés avec le chantier 1 (ex. QQQ ou ES, GLD, un deuxième point de courbe de taux). Critère d'inclusion déclaré : liquidité, historique ≥ à celui des actifs actuels, frais réels ≤ 5 bps. L'entraînement suit le protocole standard (train-once-forward, 5 graines, 200 tirages).
- **Cible de puissance** : avant de lancer, calculer l'effective_n nécessaire pour détecter un edge de 3 bps à 80 % de puissance (analyse de puissance sur la variance observée des différences de PnL par origine, brique bootstrap existante). Si la cible est inatteignable même avec l'extension, le dire — c'est un résultat.
- Toute conclusion reste au standard du programme : poolé multi-graines, familles Holm déclarées.

## Chantier 3 — Balayage du budget de tirages par horizon (appoint, ferme la réserve « optimal »)

La Note de décision assume : « 200 est un point de fonctionnement raisonné, pas un optimum certifié — seul un balayage le prouverait ». Le faire, c'est peu coûteux (aucun refit, les nuages se rééchantillonnent depuis les fits existants ou se régénèrent en minutes).

- **Balayage** : n_samples ∈ {50, 100, 200, 400, 800} × {daily, weekly} × 3 horizons, 5 graines, sur les nuages NsDiff. Métriques : couverture 95 %, largeur, Winkler ; courbes de convergence par régime.
- **Hypothèse à trancher** (déclarée dans la consolidation §1, jamais testée) : le nuage daily, propagé sur ~5 pas et à queues plus lourdes, converge plus lentement — il gagnait +4,2 pts de couverture en passant 50→200 contre +2,4 au weekly. Si confirmé, acter des budgets différenciés N_daily > N_weekly (ex. 400/200) comme réglage de production, en gardant le budget strictement égal entre modèles au sein de tout match comparatif (non-négociable inchangé).
- **Critère d'arrêt** : le budget retenu par régime est le plus petit N où la couverture et le Winkler sont à moins d'un demi-écart-type bootstrap de leur valeur à 800. Au-delà, on paie du calcul pour rien.
- Si le balayage valide 200 partout : la réserve « pas prouvé optimal » tombe, à documenter dans la note et le bandeau du dashboard.

## Ordre, dépendances, critère d'arrêt global

Point 0 immédiat. Chantier 3 d'abord (court, fige le budget par régime que les chantiers 1-2 utiliseront). Puis chantier 1 sur les actifs existants (rapide, tranche déjà la question frais). Le chantier 2 (entraînements neufs) n'est lancé que si le chantier 1 montre au moins une cellule à edge net positif — sinon le programme se clôt sur un NO-GO définitif à l'univers de frais réel près, et c'est une conclusion défendable. Ce critère d'arrêt est déclaré ici, avant tout run.

## Non-négociables (inchangés)

- Multi-graines 42-46, conventions descriptif / graine fixe / poolé ; familles Holm déclarées avant les runs ; aucun seuil, marge, niveau ou grille choisi après lecture des résultats (les choix de ce brief — 80 %, grille de N, critères d'inclusion — sont les déclarations a priori).
- Briques réutilisées (paired_test, mcs.spa_test, diffusion_headtohead, moteur de backtest du chantier B) ; code neuf couvert de tests unitaires.
- tracking.db : lecture seule ; toute écriture éventuelle passe par un script dédié avec dry-run par défaut, sauvegarde horodatée et --apply explicite (standard repoint_oos_to_m200).
- Prix gelés partagés ; budget d'échantillonnage strictement égal entre modèles comparés.
- pytest vert avant/après (453 passed actuel).
