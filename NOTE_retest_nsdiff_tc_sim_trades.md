# NOTE — Re-test de NsDiff sur les test cases TC1.1-TC1.5b (sim_trades, oos)

2026-08-11. Réponse au `BRIEF_retest_nsdiff_tc_sim_trades.md`.

**Statut, redit d'emblée** : ce re-test est **descriptif et comparatif**. Il mesure
l'utilisabilité opérationnelle des signaux (taux d'utilisation, qualité par règle,
unité = jour-signal). Ce n'est **pas** une réouverture du volet économique clos :
aucune p-value contre B&H, aucune famille de Holm, aucun walk-forward apparié par
origine. Le verdict du programme NsDiff n'est ni rejoué ni affecté.

## 0. Ce qui a été fait

Le trou constaté par le brief est confirmé : avant ce run, NsDiff avait 218 lignes
`live` et **zéro** ligne `oos` dans `sim_trades`, faute de prévisions D+1 — sa piste
oos est en horizons hebdomadaires.

`experiments/oos_nsdiff_d1_simtrades.py` produit les 666 lignes manquantes
(`run_id = 20260810-NsDiff-oos-D1-simtrades`) :

- **Grille reprise verbatim** des lignes oos d'ARIMA-GARCH — 666 clés (actif, d_date),
  167 dates, 5 actifs, 2026-01-21 → 2026-07-09. Comparabilité 1:1.
- **`ref`/`realized` hérités**, jamais recalculés. `reference_price` sert aussi
  d'ancrage à la dé-standardisation : seuls `predicted`/`pi_lower`/`pi_upper` diffèrent
  entre modèles, par construction.
- **Aucun réentraînement par origine** : un fit par (graine, actif) sur l'historique
  strictement antérieur à la première origine, puis forecast seul. Distance train→test :
  fits gelés sur l'avant-grille, historique récent vu uniquement par la fenêtre d'entrée
  `seq_len = 30`.
- **Ensemble 5 graines (42-46) × 200 tirages = 1000**, nuages concaténés ;
  `predicted` = médiane, PI = quantiles empiriques 2,5 / 97,5 %.
- **Anti-look-ahead** vérifié par recalcul tronqué sur 5 dates par actif (première et
  dernière incluses), égalité exacte exigée, plus une contre-épreuve par fuite injectée
  en test unitaire — sans quoi le contrôle pourrait passer sans rien discriminer.
- **Écriture** : dry-run par défaut, sauvegarde horodatée, `--apply` explicite, et
  empreinte SHA-256 de **tout le reste de la base** (pas seulement des lignes
  non-NsDiff : les lignes oos hebdomadaires de NsDiff, l'ensemble 5×200 repointé, sont
  le voisinage le plus exposé à un upsert trop large). Empreinte **inchangée** après
  écriture.

Les 6 règles ont été passées via `validation.sim_trades.generate_sim_trades`, **sans
aucune modification de `sim_trades.py`**. `fee_bps = 0.0` — lu, pas choisi : c'est le
défaut avec lequel `sim_trades.main()` a produit les lignes oos des autres modèles,
revérifié sur une ligne réelle où `roi == (realized − ref) / ref` exactement.
`vol_bucket` de TC1.5b : proxy terciles par (actif × modèle) existant, appliqué tel quel
aux largeurs NsDiff. `degenerate_pi = 0` sur les 666 lignes.

## 1. Les trois attentes pré-déclarées, confrontées

### Attente 1 — les règles de stress ne devraient quasi jamais émettre : **VÉRIFIÉE, mais elle ne prouve pas ce qu'on croit**

NsDiff émet **exactement zéro** signal sur TC1.2 (`pi95_conf`) et TC1.4
(`bear_stress_d1`) — le plus strict des 7 modèles (Prophet 5,5 % / 8,8 %, LSTM 3,1 % /
0,7 %, les autres zéro aussi). Aucun drapeau rouge de PI trop étroits **par ce test**.

Mais le croisement avec la couverture, que le brief demandait explicitement, donne la
lecture inverse — et c'est le point le plus important de cette note. Les règles de
stress exigent que l'intervalle **entier** passe au-dessus (`pi_low > ref`) ou au-dessous
(`pi_high < ref`) de la référence. Elles testent donc le rapport **dérive / largeur**,
pas l'honnêteté des bandes. À D+1 la dérive médiane de NsDiff est quasi nulle, si bien
que l'intervalle enjambe toujours `ref` — **même avec les bandes les plus étroites du
panel**. Zéro émission de stress est ici la signature d'une dérive minuscule, pas d'un
intervalle prudent. Conclure « PI honnêtes » de ce seul chiffre serait une erreur : c'est
la couverture (§2) qui répond, et elle dit sous-couverture.

### Attente 2 — TC1.1/TC1.3 (calm) autour de 50-60 % de jours émetteurs : **VÉRIFIÉE**

NsDiff émet sur **51,4 %** des jours en `bull_calm_d1` (342/666) et **48,5 %** en
`bear_calm_d1` (323/666) — dans la bande annoncée, au milieu du panel (ARIMA-GARCH
67,9 % / 32,1 %, SARIMA 56,6 % / 43,4 %, LSTM 53,2 % / 42,9 %, TSDiff 33,4 % / 66,6 %).
Les deux règles se partagent 665 des 666 jours : la médiane NsDiff tombe presque toujours
d'un côté ou de l'autre de `ref`, jamais dessus.

Comme annoncé, l'information est dans les counters, pas dans le taux :

| TC1.1 `bull_calm_d1` | branche 1 | 2 | 3 | 4 | direction_ok | ROI moyen/signal |
|---|---|---|---|---|---|---|
| NsDiff | 10 | 142 | 167 | 23 | **44,4 %** | −0,179 % |
| ARIMA-GARCH | 13 | 208 | 206 | 25 | 48,9 % | −0,170 % |

NsDiff a la plus faible justesse directionnelle du panel en `bull_calm_d1` (44,4 %,
contre 46,4-52,7 % pour les autres) : sur les jours où il émet, il se trompe de sens
plus d'une fois sur deux. En `bear_calm_d1` il est à 46,4 % (ROI moyen −0,072 %). Ces
ROI se lisent avec la réserve du §3 : tous les modèles sont entre −0,28 % et +0,09 % par
signal, la dispersion inter-modèles est du même ordre que le bruit d'une fenêtre de
6 mois, et `fee_bps = 0` ignore les frais.

### Attente 3 — TC1.5/1.5b, le terrain où une variance apprise peut se distinguer : **VÉRIFIÉE — NsDiff se distingue, dans le mauvais sens**

C'était la lecture annoncée comme la plus intéressante. Elle l'est.

| Modèle | n | Largeur moyenne (% du prix) | Couverture réalisée (`in_band`) | Écart au nominal 95 % |
|---|---|---|---|---|
| **NsDiff** | 664 | **6,27** | **90,2 %** | **−4,8** |
| ARIMA-GARCH | 666 | 7,23 | 92,3 % | −2,7 |
| SARIMA | 666 | 8,67 | 96,4 % | +1,4 |
| Naive | 662 | 8,70 | 96,1 % | +1,1 |
| TSDiff | 564 | 8,92 | 96,1 % | +1,1 |
| LSTM | 299 | 15,99 | 100,0 % | +5,0 |
| Prophet | 171 | 17,66 | 98,8 % | +3,8 |

`in_band` **est** la couverture du PI à 95 % sur les jours émis : le comparer entre
modèles sans regarder la largeur n'a pas de sens, une bande large l'obtient
gratuitement. Les deux colonnes ensemble sont sans ambiguïté : **NsDiff produit les
intervalles les plus étroits des 7 modèles et la couverture la plus basse**. Le
classement est quasi monotone en largeur, et NsDiff est à l'extrémité.

Le détail par actif, apparié sur la même grille, montre que le déficit vient de la
crypto — ce qui **confirme à D+1 l'avertissement « NsDiff sous-couvre la crypto »** cité
par le brief :

| Actif | Largeur NsDiff / GARCH | Couverture NsDiff / GARCH |
|---|---|---|
| BTC-USD | 8,61 % / 9,28 % | **89,6 % / 93,9 %** |
| ETH-USD | 12,18 % / 15,33 % | **90,9 % / 95,2 %** |
| SPY | 3,34 % / 3,37 % | 90,2 % / 88,4 % |
| TLT | 2,36 % / 2,40 % | 91,1 % / 92,9 % |
| ZN=F | 1,08 % / 1,03 % | 89,4 % / 89,4 % |

Sur les actions et les taux, NsDiff fait jeu égal avec GARCH à largeur quasi identique
(et le bat même légèrement sur SPY). Sur la crypto, il resserre nettement les bandes —
de 3,2 points de pourcentage de prix sur ETH — et paie ce resserrement de 4 points de
couverture. La variance apprise se distingue donc bien de GARCH ; elle est simplement
trop confiante là où la volatilité est la plus forte.

**Effet du gate (TC1.5b), contre-intuitif et symétrique.** Le gate retient 444 signaux
sur 664 (220 `gated_out`) et fait **baisser** la couverture : 90,2 % → **87,6 %** pour
NsDiff, 92,3 % → 90,6 % pour ARIMA-GARCH. Il conserve les jours du tercile de vol
anticipée le plus bas, c'est-à-dire ceux où les bandes sont les plus étroites — donc les
plus exposés à la sous-couverture. Le gate n'améliore pas la qualité du signal plat sur
cette fenêtre, pour aucun des deux modèles.

## 2. Un bug de données rencontré en chemin (TLT)

La vérification bloquante des prix a fait remonter une incohérence **préexistante**, non
introduite par ce run. Les `reference_price` de TLT en base mélangent deux millésimes,
séparés par un facteur constant de **+0,3933 %** — signature d'un réajustement de
dividendes (`auto_adjust`) : yfinance réécrit tout l'historique à chaque distribution, et
les modèles ont été ingérés à des dates différentes. Concrètement, ARIMA-GARCH est sur le
millésime récent pour 109 de ses 112 dates TLT et sur l'ancien pour 3 (2026-01-22,
2026-01-26, 2026-01-27) ; TSDiff est lui aussi panaché. Au total 21 clés TLT et 1 clé SPY
portent des `ref`/`realized` qui diffèrent d'un modèle à l'autre.

Conformément au brief, c'est un bug, pas une donnée. Il est **sans effet sur ce re-test** :
les prix récupérés ne servent qu'à la fenêtre de conditionnement, faite de
**log-rendements**, rigoureusement invariants par multiplication de toute la série par une
constante ; et l'ancrage est le `reference_price` hérité, jamais celui de la série. Le
script sélectionne pour TLT la source qui colle le mieux (cache gelé `DONNEE~1.XLS` :
3 écarts sur 112, contre 112/112 pour yfinance), journalise les dates concernées dans
`experiments/oos_nsdiff_d1_simtrades.json`, et bloque au-delà de 5 % de niveaux
divergents. **Il reste à corriger à la source**, hors périmètre de ce brief.

## 3. Limites — à lire avant toute conclusion

- **Fenêtre courte et régime unique.** ~167 jours, janvier-juillet 2026. C'est un
  instantané comparatif, aucune conclusion de robustesse n'en sort.
- **Signaux conditionnels au modèle.** Les jours d'émission diffèrent d'un modèle à
  l'autre : les comparaisons sont par agrégat, pas appariées jour à jour — sauf le
  tableau par actif du §1, restreint à la grille commune.
- **Défaut de couverture permanent de la piste daily.** Les 35 cellules de référence en
  portent un ; les ROI oos des autres modèles se lisent avec cette réserve, et les
  couvertures de 88-90 % observées sur SPY et ZN=F pour *tous* les modèles en sont la
  manifestation directe.
- **`fee_bps = 0`.** Les ROI par signal sont bruts de frais, par cohérence avec les
  lignes existantes — pas parce que c'est réaliste.
- **Volet économique clos, non rouvert.** Rien ici ne se lit comme une p-value.

## 4. Reproduire

```bash
python experiments/oos_nsdiff_d1_simtrades.py            # dry-run, aucune écriture
python experiments/oos_nsdiff_d1_simtrades.py --apply    # sauvegarde + écriture + 6 règles
python -m validation.generate_sim_trades_dashboard       # dashboard sim_trades
python -m model_artifacts.generate_taux_utilisation      # taux d'utilisation
```

Reprenable : un checkpoint par (graine, actif) sous
`experiments/checkpoints_nsdiff_d1_simtrades/` (gitignoré, régénérable en ~8 min).
Plan et journal du run : `experiments/oos_nsdiff_d1_simtrades.json`.

`NsDiff` a été ajouté à `SIM_TRADES_MODELS` (`model_artifacts/generate_dashboard.py`) :
il en était absent parce qu'il n'avait aucune ligne oos à horizon 1. Sans cet ajout, le
re-test resterait invisible dans le taux d'utilisation, là où le brief dit que le verdict
se lit.

pytest : **783 passed, 1 skipped** avant (le brief annonçait 777 — l'écart vient des
tests de dashboard ajoutés entre-temps), **795 passed, 1 skipped** après, les 12 nouveaux
étant ceux du générateur.
