# Brief Claude Code — NsDiff : extension de données, puissance, dashboard, re-jugement mensuel

2026-08-08. Fait suite à NOTE_nsdiff_edge_vs_frais.md (frais ≈ 2 % de l'edge différentiel ; mur GARCH tenu par la puissance ; cible 3 bps inatteignable, effets observés atteignables à 150-270 origines ; balayage de N tranché). TsDiff est hors périmètre — retiré du benchmark, ne réapparaît dans aucun chantier de ce brief.

## Décisions actées en entrée (pas à re-discuter, à implémenter)

1. **Pas de budget de tirages différencié par régime.** L'hypothèse « daily converge plus lentement » est réfutée (nsamples_sweep, sous-échantillonnage emboîté). Un seul budget.
2. **Config production = ensemble 5×200 = 1000 tirages** : convergée 6/6 cellules, rien à changer.
3. **Panel weekly déséquilibré déclaré** (validé) : chaque actif à son historique maximal propre, pas de troncature au plus court.
4. **weekly_propagated abandonnée** (dominée sur les 4 métriques) — acter dans la note, ne plus jamais la rejouer.

## Chantier A — Extension des données : panel déséquilibré, weekly 800 + daily aligné

### A1. Nouvelles séries de prix

Un seul départ commun là où l'historique le permet : **2011-05** (= 800 semaines à date). Source unique (yfinance), prix re-gelés dans un nouveau dossier `prices_v3/` — les anciens `diffusion_multiseed_v2/prices/` sont conservés intacts, aucun chiffre historique n'est recalculé sur les nouvelles séries sans le déclarer.

| Actif | Départ | Obs weekly attendues | Obs daily attendues | Fenêtres train daily (origines 2020-01, seq_len=30) |
|---|---|---|---|---|
| SPY | 2011-05 | ~800 | ~3 830 | ~2 235 |
| ZN=F | 2011-05 | ~800 | ~3 830 | ~2 235 |
| TLT | 2011-05 | ~800 | ~3 830 | ~2 235 |
| BTC-USD | 2014-09 (début source) | ~620 | ~4 340 | ~1 930 |
| ETH-USD | 2017-11 (début source) | ~455 | ~3 190 | ~760 |

- Cible daily déclarée : **≥ 2 000 fenêtres d'entraînement** par actif là où l'historique le permet (niveau du train actuel validé, ratio NsDiff ≤ 4 params/fenêtre). BTC et ETH n'y arrivent pas : les déclarer cellules faibles du panel, pas les tronquer ni chercher de source alternative (homogénéité de source prioritaire).
- Ajouter **GLD et USO** au panier (chantier 2 de la note : seuls candidats à exposition réellement nouvelle ; QQQ/EFA/IEF exclus, corrélation 0,8-0,9 avec l'existant — critère déclaré). Départ 2011-05, frais réels 4-5 bps dans `real_fees.py`.
- Documenter dans la note les régimes traversés par tranche d'historique (2011-2015 : QE/taux zéro ; 2015-2020 ; 2020 : COVID ; 2022 : bear taux ; 2024-2026) — le gain est de validité externe autant que de puissance.

### A2. Vérifications bloquantes avant tout run

Recouvrement exact des séries nouvelles et anciennes sur la période commune (mêmes cotes à la tolérance yfinance près, ~2e-7 relatif) ; sinon, stop et documenter. Comptages du tableau A1 vérifiés et écrits dans le JSON d'artefact.

## Chantier B — Régénération de la grille oos à départ 2020-01 et hypothèse pré-déclarée

C'est le « chantier en soi » que la note a chiffré sans l'exécuter. Grille d'origines : **2020-01 → 2026, 340 origines, effective_n ≈ 113**, pour **tous** les modèles de référence restants (la comparabilité mutuelle l'exige).

- NsDiff : train-once-forward sur données < 2020-01 (261 semaines weekly ; ~2 235 fenêtres daily pour les actifs longs), 5 graines, 200 tirages/graine, config production = ensemble. GARCH : refit par origine, son protocole naturel. Asymétrie déclarée comme toujours.
- **Hypothèse primaire pré-déclarée** (réponse à la limite « edge non corrigé pour la sélection d'instrument ») : `var_limit` sur SPY (ES et ETF), W+2 et W+3 weekly, vs GARCH — n requis sous Holm 231-270 contre 340 disponibles. C'est LE test confirmatoire du programme économique. Tout le reste (autres instruments, GLD/USO, daily) est exploratoire et étiqueté tel quel dans la note.
- Rejouer aussi : les 4 survivants « B&H bat la stratégie » (SPY W+1 weekly, ~−12 bps) — se répliquent-ils sur 340 origines ? — et le match calibration/Winkler NsDiff-ensemble vs GARCH au nouveau départ.
- Familles de Holm déclarées avant les runs, identiques en structure à celles du chantier précédent.
- **Critère d'arrêt global, déclaré ici** : si l'hypothèse primaire ne survit pas à Holm sur 340 origines, le volet économique du programme se clôt définitivement — il n'y aura pas de troisième univers de test.

## Chantier C — Dashboard : bascule sur l'ensemble (court, à faire en premier)

Le balayage a montré que la piste single-seed 42×200 du dashboard n'est pas convergée (weekly exige 800 par graine). Deux options, décision recommandée = la première :

- **Basculer le dashboard sur l'ensemble 5×200 = 1000** (la config production, convergée) : script dédié type `repoint_oos_to_m200` — dry-run par défaut, sauvegarde horodatée, `--apply` explicite, vérification 1:1 des clés, backfill des colonnes dérivées, bandeau de config mis à jour.
- À défaut (si le dashboard doit rester single-seed pour une raison à documenter) : monter son budget à 800 tirages — pas 400, insuffisant en weekly.

## Chantier D — Mensuel : re-jugement sur données étendues

Le NO-GO mensuel tenait à une observation, et son problème de fond (33-70 fenêtres d'entraînement) est précisément ce que le chantier A corrige : SPY/ZN/TLT passent à ~183 mois d'historique (~130+ fenêtres train à seq_len=12), soit environ le double. Ordre des opérations :

- **D1 (= point ii, demandé a minima)** : élargir la grille d'époques vers le haut — candidats (80, 160, 320, 640), sélection sur validation par argmin CRPS, règle existante « argmin au bord → élargir une fois » maintenue. À faire même si le reste du chantier D est reporté.
- **D2 (= point i)** : rejouer le pilote monthly_native sur **les 5 actifs** (+ GLD/USO si A est fait), sur les données étendues, avec la grille D1. Critères go/no-go inchangés et déclarés : couverture 95 % dans [0,90 ; 0,98] à chaque horizon, Winkler pas significativement pire que garch_monthly (régénéré sur le même historique étendu). effective_n par cellule à recalculer et déclarer.
- **D3 (= point iii, conditionnel)** : le générateur à volatilité stochastique (remplaçant de KernelSynth) n'est lancé **que si** D2 échoue encore par sous-entraînement manifeste (PI sur-larges + couverture ~100 % sur les actifs courts). Si D2 passe ou échoue pour une autre raison, D3 est classé sans suite — la R&D d'augmentation synthétique n'a de justification que le déficit de données qu'A résorbe en partie.
- weekly_propagated : mention unique dans la note comme voie abandonnée (décision actée n° 4), aucun re-run.

## Ordre et dépendances

C d'abord (court, indépendant, ferme la réserve dashboard). Puis A (données) → B et D2/D3 dessus. D1 peut tourner à tout moment. B est le chantier lourd : chiffrer le coût de régénération complet avant de lancer, et le découper si nécessaire (weekly d'abord — l'hypothèse primaire est weekly).

## Non-négociables (inchangés)

- Multi-graines 42-46 ; conventions descriptif / graine fixe / poolé ; familles Holm et tous seuils/grilles/critères déclarés avant les runs (ceux de ce brief font foi).
- Briques réutilisées : econ_backtest, paired_test, mcs.spa_test, multiple_testing, nsdiff_production_spec, real_fees, calibration_tests, machinerie monthly existante. Code neuf couvert de tests unitaires.
- tracking.db : lecture seule, sauf chantier C via script dédié (dry-run/sauvegarde/--apply).
- Prix gelés partagés par tous les bras d'un même match ; budget d'échantillonnage strictement égal entre modèles comparés.
- pytest vert avant/après (573 passed, 1 skipped actuel).
