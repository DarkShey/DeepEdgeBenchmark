# Brief Claude Code — Re-test de NsDiff seul sur les test cases TC1.1-TC1.5b (sim_trades, oos)

2026-08-10. Constat vérifié en base : les 6 modèles de référence ont chacun 1 000-2 000 lignes oos à horizon 1 dans `sim_trades` (grille : 5 actifs × ~167 dates quotidiennes, 2026-01-22 → 2026-07-09), mais **NsDiff n'a que 218 lignes `live` et zéro ligne oos** — il n'est jamais passé par les TC1.1-1.5b en oos, faute de prévisions D+1 sur cette grille (sa piste oos est en horizons hebdo). Ce brief comble ce trou.

**Statut déclaré d'emblée** : ce re-test est **descriptif et comparatif** — taux d'utilisation et qualité des signaux par règle, unité = jour-signal. Ce n'est PAS une réouverture du volet économique clos (walk-forward apparié par origine, familles de Holm) : les TC mesurent autre chose — l'utilisabilité opérationnelle des signaux de DEB — et le verdict se lira dans les KPI du dashboard sim_trades, pas en p-values contre B&H.

## 1. Générer les prévisions NsDiff D+1 manquantes (aucun refit)

- **Grille** : reprendre **exactement** les clés (actif, d_date) des lignes oos existantes d'ARIMA-GARCH (le modèle le plus complet, 1 998 lignes) — comparabilité 1:1, vérification bloquante des clés avant toute écriture (le standard qui a déjà sauvé 28 560 lignes).
- **Prévision** : fits daily NsDiff existants (train-once-forward, 5 graines 42-46), **forecast seul à horizon 1 jour** — même mécanique que nsdiff_multiseed_v2 (aucun réentraînement, quelques minutes). Config référence : ensemble 5×200 = 1000 tirages, `predicted` = médiane, `pi_low/pi_high` = quantiles empiriques 2,5/97,5. Déclarer la distance train→test (fits gelés, historique récent vu uniquement par la fenêtre d'entrée seq_len=30).
- **ref et realized : lus depuis les lignes existantes** des autres modèles pour les mêmes clés — pas re-téléchargés, pas recalculés. C'est la garantie que seuls `predicted/pi_low/pi_high` diffèrent entre modèles ; toute divergence de ref/realized est un bug, pas une donnée.
- Look-ahead : vérification par recalcul tronqué sur 5 dates (première et dernière incluses), égalité exacte, contre-épreuve par fuite injectée — le mécanisme existant.

## 2. Passer les 6 règles telles quelles

- `RULES` de `validation/sim_trades.py` appelé sans aucune modification : bull_calm_d1 (TC1.1), pi95_conf (TC1.2), bear_calm_d1 (TC1.3), bear_stress_d1 (TC1.4), sideways_d1 (TC1.5), sideways_gated_d1 (TC1.5b).
- `fee_bps` : la même valeur que celle utilisée pour les lignes oos des autres modèles (à lire dans la base/le code, pas à choisir).
- `vol_bucket`/`stress_score` pour TC1.5b : le proxy terciles par (actif × modèle) existant, appliqué aux largeurs NsDiff — aucun paramètre nouveau.
- **Écriture** : script d'upsert standard (dry-run par défaut, sauvegarde horodatée, `--apply`, vérif 1:1, empreinte des lignes non-NsDiff avant/après — elles ne doivent pas bouger d'un octet). run_id dédié `20260810-NsDiff-oos-D1-simtrades`.

## 3. Lecture — les KPI du dashboard sim_trades, plus trois attentes pré-déclarées

Mêmes agrégats que pour les autres modèles (dashboard sim_trades + taux d'utilisation) : part de jours avec signal par TC, distribution des branches/counters, ROI par signal (TC1.1-1.4), in_band (TC1.5), gated_out (TC1.5b), degenerate_pi. Comparaison aux 6 autres modèles sur la même grille.

Trois attentes écrites avant le run, pour discipliner la lecture :

1. **TC1.2 et TC1.4 (stress : `pi_low > ref` / `pi_high < ref`) devraient quasi jamais émettre** si les PI de NsDiff sont honnêtes — c'est la famille 3 en version D+1, et la leçon drift ≪ largeur s'applique encore plus à un jour. Un taux d'émission stress élevé serait un **drapeau rouge de PI trop étroits**, à croiser avec la couverture daily (NsDiff sous-couvre la crypto).
2. **TC1.1/TC1.3 (calm)** : l'émission dépend du signe de la médiane vs ref — attendre ~50-60 % de jours émetteurs, comme les autres modèles ; l'information est dans les counters (branches 1-2 vs 3-4), pas dans le taux.
3. **TC1.5/1.5b (sideways)** : c'est le terrain où un modèle à variance apprise peut se distinguer — le signal exige |predicted − ref| ≤ k·W, donc dépend de la largeur. Comparer le taux de signaux plats et leur in_band à ceux de GARCH est la lecture la plus intéressante du re-test.

## 4. Limites à déclarer dans la note

Fenêtre courte (~167 jours, janv.-juil. 2026, un seul régime de marché) — aucune conclusion de robustesse, c'est un instantané comparatif ; les signaux sont conditionnels au modèle (jours d'émission différents entre modèles) — les comparaisons sont par agrégat, pas appariées jour à jour sauf sur l'intersection ; la piste daily des modèles de référence porte 35 cellules à défaut permanent de couverture (les ROI oos des autres modèles se lisent avec cette réserve) ; et le verdict du programme NsDiff (clos) n'est ni rejoué ni affecté.

## Non-négociables (inchangés)

Règles TC intouchées (aucune modification de sim_trades.py hors ajout éventuel de plomberie opt-in) ; tracking.db en écriture uniquement via le script d'upsert dédié ; prix/ref/realized hérités des lignes existantes ; ensemble multi-graines ; tests unitaires sur le générateur D+1 (dont le test de look-ahead) ; pytest vert avant/après (777 passed, 1 skipped actuel).
