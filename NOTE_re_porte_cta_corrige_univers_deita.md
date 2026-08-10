# NOTE — Porte 0-bis : le CTA corrigé, jugé dans son habitat. Elle échoue, et le dossier trading se clôt.

*2026-08-08. Réponse au brief « Porte 0-bis : le CTA corrigé, jugé dans son habitat
(univers DEITA complet) ». Précédé de R1, traité côté DEITA
(`R1_revue_diff_et_revalidation.md`).*

| Artefact | Chantier | Contenu |
|---|---|---|
| `experiments/prices_v4.py` → `prices_v4/` | 0-bis-A | 18 actifs, socle unique signal + PnL, gelé, hashé, recouvrement bloquant |
| `experiments/deita_cta_signal.py --universe v4` → `deita_cta_signal_v4/` | 0-bis-A | signal du moteur corrigé, gelé, look-ahead vérifié |
| `experiments/real_fees.py` | 0-bis-A | 11 instruments ajoutés, ordres de grandeur déclarés par famille |
| `experiments/cta_gate0bis.py` → `.json` | 0-bis-B | la porte, avec l'instrument corrigé |
| `experiments/test_cta_gate0bis.py` | 0-bis | 16 tests (déclarations d'univers, grille, IC bootstrap) |

---

## 0. Le résultat

**La porte 0-bis échoue sur les deux branches. Le dossier trading se clôt
définitivement, toute la ligne** — c'est le critère d'arrêt du brief, et il avait
été écrit pour cette éventualité : « signal corrigé, instrument corrigé, habitat
naturel, fenêtre de 6,5 ans traversant trois régimes : il n'y aura pas de porte
0-ter ».

Le chiffre central : sur un portefeuille équipondéré de 17 instruments,
**l'excès contre acheter-et-garder vaut −51,77 bps par origine, Sharpe −0,97,
IC95 [−1,79 ; −0,22]** — 1 % seulement des tirages bootstrap ressortent positifs.
Ce n'est pas un résultat indéterminé, c'est un résultat négatif net.

Trois précisions qui comptent pour la lecture :

1. **Le CTA corrigé ne perd pas d'argent** : le PnL brut du portefeuille est de
   +0,42 bps par origine, Sharpe 0,01. Il ne bat simplement pas le fait de
   détenir, sur une fenêtre où presque tout le panel a monté.
2. **Les deux seules cellules significatives disent que B&H gagne** — GLD-ETF
   (−55,4 bps, p = 0,0004) et SI-FUT (−79,8 bps, p = 0,009). Aucune ne va dans
   l'autre sens.
3. **Le crisis alpha attendu n'est pas là.** Le correctif de calendrier restaure
   bien la réactivité aux krachs — c'était le motif encourageant du brief — mais
   2020 reste la pire tranche de la fenêtre (excès −116,47 bps, Sharpe −1,69).
   Seule 2022 est positive, et faiblement (+10,61 bps, Sharpe 0,18).

Ce verdict est cohérent avec R1, obtenu par un chemin entièrement différent : sur
le harnais walk-forward de DEITA, une fois son look-ahead retiré, le signal
corrigé a un Sharpe OOS de +0,028, IC [−0,579 ; 0,642]. Deux mesures
indépendantes, même conclusion.

---

## 1. Chantier 0-bis-A — le socle de prix, et la déclaration d'univers

### 1.1 Pourquoi prices_v4 existe

La porte 0 s'était rabattue sur 7 actifs pour une raison unique : la base de prix
de DEITA et `prices_v3` divergent de 2,6e-03 sur SPY, et un signal produit sur une
série puis tradé sur une autre est interdit. `prices_v4` supprime l'obstacle à la
racine — les 18 actifs sont téléchargés par le **même pipeline** que `prices_v3`
(même source, même fenêtre, calendriers de cotation propres), gelés et hashés
ensemble. Le signal et le PnL lisent désormais le même socle.

### 1.2 L'univers, déclaré avant tout calcul

Union de l'univers DEITA (16) et du panel du benchmark (7), avec la règle de
substitution que le brief exige — une seule convention par exposition :

> **Quand une exposition existe des deux côtés sous deux véhicules différents,
> c'est l'instrument du benchmark qui est retenu.**
> or : GC=F → GLD ; pétrole : CL=F → USO.

Motif, antérieur à tout résultat : `prices_v4` doit servir le signal *et* le PnL,
et le chantier 1 conditionnel ne peut trader que les instruments du panel — les
nuages NsDiff n'existent que là. Retenir GLD/USO fait du panel un **sous-ensemble
strict** de l'univers 0-bis (vérifié par test), donc aucune exposition n'apparaît
deux fois et le chantier 1 se brancherait sans re-gel. Prix à payer, déclaré : USO
est un ETF à roulement mensuel, proxy moins pur que CL=F.

**18 actifs = 16 (DEITA) − 2 (substitués) + 4 (ZN=F, TLT, GLD, USO).**

**Les quatre classes de la branche 2**, figées avant les runs : actions (5) /
taux (2) / matières-or (5) / crypto (5). **VXX est déclaré hors classes** : ETN de
volatilité court terme à décroissance structurelle de roulement, il n'appartient à
aucune des quatre et l'y forcer contaminerait le verdict de cette classe. Il reste
dans l'univers de conviction (moteur DEITA tel quel) et il est rapporté à part.
Au passage : DEITA le range sous le secteur « Crypto », ce qui est une erreur
d'étiquetage de son côté — signalée, hors périmètre du ticket, non corrigée ici.

### 1.3 La vérification bloquante

Sur les 7 actifs partagés avec `prices_v3`, le nouveau gel doit **reproduire**
l'ancien — mêmes tolérances qu'au chantier R1. Résultat : écart maximal de
log-rendement **2,1e-06** (TLT), zéro pour cinq des sept. Sans cela, la grille
0-bis ne se comparerait plus à `oos2020` et le chantier 1 trouverait sous les
nuages NsDiff des prix qui ne les ont pas produits.

### 1.4 Le signal retenu change, et c'est la conséquence directe du correctif

À la porte 0, la conviction hiérarchique avait été écartée : elle dégénérait en
« toujours long » sur 4 actifs sur 7. **Le bug 1 étant corrigé et l'univers
élargi, cette raison a disparu** — la conviction est de nouveau directionnelle
(44 à 58 % de positions longues, 274 à 532 changements de signe selon l'actif).
C'est donc elle, le CTA complet de DEITA, qui est jugée ici. La direction Hull est
gelée à côté, en lecture secondaire.

Vérification d'absence de look-ahead, identique à la porte 0 : recalcul complet du
signal sur historique tronqué à 5 origines réparties sur la grille — **écart
0,00e+00**, tolérance 1e-12. La contre-épreuve par fuite injectée reste en place
dans les tests.

### 1.5 Les frais

11 instruments ajoutés, ordres de grandeur déclarés par famille comme l'exige le
brief : ETF actions 2-5 bps (majorés à 5,5 pour EEM, 5 pour EFA — sous-jacents
décalés, spread plus large), futures 1-2 bps + roulement H2 (SI 2,5 ; HG 3 ;
ZC 4 — carnets plus minces que ES/ZN), crypto 10-60 bps (40-50 au central pour
SOL/BNB/LINK, hors des deux plus grosses capitalisations), VXX 10 bps (ETN).
Deux réserves écrites dans le fichier : le maïs roule ~5 fois par an et non 4, le
modèle H2 sous-estime donc son roulement ; VXX porte une décroissance de contango
qui n'est pas un frais de transaction et n'est pas modélisée.

## 2. Chantier 0-bis-B — la porte

### 2.1 Le montage

CTA seul, taille fixe |w| = 1, weekly, W+1, net de frais au niveau central,
roulement inclus. 354 origines, 18 actifs, prix `prices_v4`. Comparaison appariée
contre acheter-et-garder, bootstrap par blocs. **La fenêtre est inchangée** —
2020-01 → 2026-07, mêmes origines que `oos2020`.

### 2.2 Par instrument (W+1, triés par excès)

| Instrument | classe | long | PnL net | vs B&H | Sharpe | p |
|---|---|---:|---:|---:|---:|---:|
| VXX-ETN | *(hors)* | 42 % | +37,41 | +117,07 | 0,33 | 0,102 |
| TLT-ETF | taux | 50 % | −2,81 | +7,48 | −0,10 | 0,672 |
| ZN-FUT | taux | 48 % | −4,69 | +1,63 | −0,42 | 0,835 |
| ZC-FUT | matières | 54 % | +8,06 | +1,44 | 0,15 | 0,910 |
| EEM-ETF | actions | 54 % | +9,91 | −4,33 | 0,25 | 0,839 |
| HG-FUT | matières | 52 % | +14,99 | −11,99 | 0,30 | 0,657 |
| EFA-ETF | actions | 56 % | +1,30 | −15,07 | 0,03 | 0,494 |
| IWM-ETF | actions | 57 % | −3,40 | −25,36 | −0,07 | 0,330 |
| QQQ-ETF | actions | 57 % | +11,35 | −27,13 | 0,26 | 0,300 |
| SPY-ETF / SPY-ES | actions | 58 % | −0,70 / +1,19 | −28,53 | ≈0 | 0,201 |
| USO-ETF | matières | 57 % | −31,78 | −48,38 | −0,36 | 0,199 |
| GLD-ETF | matières | 54 % | −28,71 | **−55,44** | −0,88 | **0,0004** |
| BNB-SPOT | crypto | 48 % | +85,17 | −60,66 | 0,47 | 0,351 |
| SI-FUT | matières | 57 % | −34,12 | **−79,77** | −0,48 | **0,009** |
| LINK-SPOT | crypto | 49 % | −10,70 | −96,08 | −0,06 | 0,305 |
| BTC-SPOT | crypto | 47 % | −41,84 | −111,29 | −0,38 | 0,073 |
| ETH-SPOT | crypto | 50 % | −16,46 | −124,76 | −0,11 | 0,103 |
| SOL-SPOT | crypto | 56 % | +62,76 | −160,05 | 0,27 | 0,116 |

Seuls trois instruments battent B&H, tous de très peu (+1,4 à +7,5 bps) et aucun
significativement. Les deux seules cellules significatives vont dans l'autre sens.

### 2.3 Le critère, appliqué tel qu'il est déclaré

- **Branche 1** — « au moins un actif à PnL net positif ET p < 0,05 brut vs B&H » :
  **échec**. Sept instruments ont un PnL positif, aucun n'est significatif (p de
  0,073 à 0,910) ; les deux p significatifs portent sur des PnL négatifs. La
  lecture concurrente (p contre zéro) **échoue également**.
- **Branche 2** — « Sharpe poolé de l'excès > 0 ET ≥ 3 classes sur 4 à excès
  positif » : **échec**, et largement. Sharpe de l'excès **−0,97**, et **1 classe
  sur 4** positive :

| Classe | excès moyen |
|---|---:|
| taux | **+5,15 bps** |
| actions | −14,94 bps |
| matières | −48,69 bps |
| crypto | −114,47 bps |

**PORTE 0-bis : ÉCHEC.**

### 2.4 La lecture complémentaire, déclarée non décisionnelle

Le portefeuille diversifié est la vraie unité d'un CTA, et c'était l'argument
central du brief pour élargir l'univers : la diversification fait l'essentiel du
Sharpe. Elle a été mesurée, et elle ne sauve rien.

| | valeur |
|---|---|
| excès moyen | −51,77 bps / origine |
| Sharpe de l'excès | **−0,97**, IC95 [−1,79 ; −0,22] |
| tirages bootstrap positifs | **1 %** |
| PnL brut du portefeuille | +0,42 bps / origine, Sharpe 0,01 |

**Par tranche de marché** (excès du portefeuille) :

| Tranche | n | excès | Sharpe |
|---|---:|---:|---:|
| 2020 (COVID) | 52 | −116,47 bps | −1,69 |
| 2021 | 53 | −31,28 bps | −0,55 |
| 2022 (bear taux) | 52 | **+10,61 bps** | 0,18 |
| 2023 | 52 | −77,17 bps | −1,91 |
| 2024-2026 | 117 | −48,75 bps | −1,08 |

Le brief attendait le crisis alpha en mars 2020, restauré par le correctif de
calendrier. **Il n'y est pas.** Le correctif a bien rendu le signal réactif — c'est
mesuré au ticket, amplitude × 1,15 à × 1,32 sur la fenêtre de choc — mais réagir
vite à un krach ne suffit pas quand le rebond qui suit est plus rapide encore : un
trend-follower vend le creux et rachète plus haut. 2022, marché baissier lent, est
la seule tranche où il gagne, et de 10 bps.

**Hors classes** : VXX-ETN affiche le meilleur excès de tout l'univers (+117 bps),
et c'est précisément pourquoi il a été exclu avant les runs. Un CTA short sur un
ETN à décroissance structurelle de roulement gagne le contango, pas la tendance —
l'inclure aurait fait passer la classe qui l'aurait accueilli, pour une raison qui
n'a rien à voir avec le trend. La déclaration préalable a servi.

## 3. Chantier 1 (sizing 4 bras) — non exécuté

Conditionnel à la porte. Elle échoue, il ne tourne pas. Ne sont donc pas écrits :
les bras A/B/C/D, le budget de risque à vol cible 10 %, l'hypothèse primaire A vs B
sur {SPY-ES, ZN-FUT} × {W+1}, la famille de Holm m = 2, l'analyse de puissance.

Les bordereaux de frais réels ne deviennent pas exigibles non plus : le brief les
conditionne au chantier 1, et R3 le disait — « ils ne deviennent décisionnels que
si un edge net doit être certifié ». Il n'y en a pas à certifier.

## 4. Le point de sortie (R5), atteint

**Le dossier trading est clos sur toute la ligne.** Pas de signal exploitable à ces
horizons, sur ce panel élargi, sur cette fenêtre ; pas de rôle trading pour NsDiff,
faute de signal directionnel à dimensionner. DEITA garde son CTA corrigé pour ce
qu'il est — un signal réparé, sans edge démontré ni ici ni sur son propre harnais
walk-forward.

Ce que le cycle laisse, et qui n'est pas rien : **trois programmes fermés par des
critères déclarés avant les runs**, et **trois bugs de production trouvés et
corrigés en chemin** — la conviction-carré, le calendrier dilué, et le look-ahead
du harnais de validation. Le troisième a été trouvé parce qu'un Sharpe de 3,95 est
invraisemblable et qu'on l'a vérifié au lieu de le publier.

## 5. Non-négociables — statut

| Non-négociable | Statut |
|---|---|
| Conventions descriptif / poolé | tenu — §2.4 explicitement descriptif, la porte seule décide |
| Seuils, classes, mapping d'univers, grille de frais déclarés avant les runs | tenu — univers, substitutions, 4 classes et exclusion de VXX figés dans `prices_v4.py` et couverts par 6 tests de déclaration |
| Briques réutilisées | tenu — `cta_gate0` corrigé (`evaluate`, `branch2_verdict`, `by_period`, `attach_signal`), `econ_backtest`, `paired_test`, `real_fees` + roulement, `prices_v3` (pipeline et vérification de recouvrement) |
| Code neuf couvert de tests unitaires | tenu — 16 tests (`test_cta_gate0bis.py`) + 3 ajoutés à `test_real_fees_roll.py` |
| `tracking.db` en lecture seule | tenu — aucun accès |
| Prix gelés partagés par le signal et le PnL | tenu — `prices_v4` unique des deux côtés, c'est l'objet du chantier |
| Signal et moteur hashés | tenu — SHA-256 du moteur, des prix et de chaque série dans `manifest.json` |
| Fenêtre inchangée | tenu — 2020-01 → 2026-07, mêmes origines que `oos2020` |
| pytest vert avant/après | tenu — 759 passed / 1 skipped avant, **775 passed / 1 skipped** après (benchmark) ; **24 passed** côté DEITA, 173 dans le périmètre CTA |

## 6. Limites déclarées

- **Le signal jugé ici n'est pas celui de la porte 0.** C'est la conviction
  hiérarchique du moteur corrigé, pas la direction Hull. Le changement est motivé
  (§1.4) mais il est réel : les deux portes ne mesurent pas le même objet, et le
  verdict de la porte 0 reste clos sans être réécrit.
- **USO au lieu de CL=F** pour l'exposition pétrole (règle de substitution
  déclarée). USO porte un roulement mensuel qui dégrade son rendement propre ;
  l'excès vs B&H en est peu affecté (les deux bras le subissent), le PnL absolu
  davantage.
- **L'univers de conviction inclut VXX**, exclu des classes mais présent dans la
  hiérarchie du moteur — c'est DEITA tel quel. Il influence donc marginalement le
  niveau marché des autres actifs.
- **Un seul horizon.** W+1 est l'horizon de détention déclaré par le brief ; W+2 et
  W+3 n'ont pas été calculés, contrairement à la porte 0 où ils étaient descriptifs.
- **Les frais restent des hypothèses** déclarées par famille, jamais des
  bordereaux. À ces niveaux d'écart — excès de −160 à +117 bps pour des frais de
  1,6 à 50 bps — ils ne portent pas la conclusion.
- **Le bootstrap du Sharpe est par blocs de 3**, convention du dépôt. À W+1 les
  sleeves ne se chevauchent pas ; le bloc couvre l'autocorrélation des rendements
  eux-mêmes, pas celle des positions.
