# NOTE — Couplage CTA (DEITA) × sizing NsDiff : le programme s'arrête à sa porte d'entrée

*2026-08-08. Réponse au brief « Couplage CTA (DEITA) × sizing NsDiff : la
direction au trend, la taille à la fourchette ». Programme neuf, distinct du
volet « NsDiff prédit-il ? » clos par `SYNTHESE_finale_programme_nsdiff.md`.*

**Artefacts produits** :

| Fichier | Chantier | Contenu |
|---|---|---|
| `experiments/deita_cta_signal.py` → `deita_cta_signal/` | 0a | signal DEITA gelé point-in-time, hashé, vérification d'absence de look-ahead |
| `experiments/cta_gate0.py` → `.json`, `_own_calendar.json`, `_conviction.json` | 0b | porte d'entrée : le CTA seul a-t-il un edge net ? (3 variantes) |
| `experiments/test_cta_gate0.py` | 0 | 15 tests unitaires (causalité, calendrier, lecture du signal, moteur d'évaluation) |

---

## 0. Le résultat, d'emblée

**La porte d'entrée du chantier 0 échoue. Le programme s'arrête avant tout
couplage, et c'est la conclusion — pas un incident de parcours.** C'est le critère
d'arrêt n° 1 du brief, appliqué tel qu'il est écrit : « Chantier 0 négatif → arrêt
complet, conclusion "pas de signal CTA exploitable sur la fenêtre" ». Les
chantiers 1 (les quatre bras de sizing) et 2 (exécution et réalisme) ne sont donc
**pas** implémentés.

Deux choses ont été trouvées en chemin, et elles comptent au moins autant que le
verdict :

1. **Le signal de conviction hiérarchique de DEITA n'est pas un signal
   directionnel sur ce panel.** Pour tout actif seul dans son sous-secteur — dont
   SPY et ZN=F, les deux instruments de l'hypothèse primaire — la conviction se
   réduit algébriquement à un **carré** et ne change plus jamais de signe : 99,3 à
   100 % de positions longues. Ce n'est pas un artefact du panel du benchmark : le
   défaut est présent dans l'univers de 16 actifs de DEITA lui-même.
2. **La branche 2 du critère de porte ne teste pas ce qu'elle croit tester.**
   Rejouée avec le signal dégénéré — c'est-à-dire avec un acheter-et-garder
   déguisé — elle **passe** (Sharpe poolé 0,77, 3 classes sur 4 positives), alors
   que l'edge vs B&H vaut exactement 0,00 bps et p = 1,000. Sur un panel haussier
   2020-2026, un Sharpe poolé positif mesure le marché, pas le signal. La
   branche 1, elle, rejette correctement.

Le verdict d'arrêt ne dépend donc pas de la décision d'interface du §1 : avec le
signal retenu les deux branches échouent, et avec le signal rejeté la seule
branche qui « passe » est celle qu'on vient de montrer défaillante.

---

## 1. Chantier 0a — l'interface DEITA, et une décision imposée par la mesure

### 1.1 Ce que DEITA expose

`cta_quant_engine.py` (portage Python du CTA MATLAB `aistis02_clean.m`, hashé dans
le manifeste) produit deux objets :

- **la direction de tendance** — moyenne mobile de Hull sur 20 jours des
  rendements quotidiens, dont on prend le signe. C'est le cœur trend-following du
  système, et c'est ce que DEITA utilise tel quel dans
  `compute_cta_signal(pure_trend_mode=True)` ;
- **la conviction hiérarchique** — cette direction multipliée par une agrégation à
  trois niveaux (marché / sous-secteur / secteur), chaque niveau pondéré par une
  corrélation glissante de 63 jours en « leave-one-out ».

Aucun paramètre n'a été retouché dans un cas comme dans l'autre.

### 1.2 Pourquoi la conviction est écartée — le mécanisme, puis la mesure

`_conviction_level` renvoie, pour un actif **seul dans son groupe**, sa propre
série lissée. Un actif seul dans son sous-secteur *et* dans son secteur voit donc
deux des trois niveaux valoir `smooth`, et sa conviction devient

```
conv = smooth × (mkt + 2·smooth)/3 = (2/3)·smooth² + smooth·mkt/3
```

dont le terme dominant est un carré. Le signe ne peut plus changer. Mesure sur la
grille (340 origines, départ 2020-01) :

| Actif | sous-secteur | conviction : part longue | direction Hull : part longue |
|---|---|---:|---:|
| SPY | singleton | 100,0 % | 60,3 % |
| BTC-USD | singleton | 100,0 % | 53,9 % |
| ZN=F | singleton | 99,4 % | 47,3 % |
| TLT | singleton | 99,9 % | 49,9 % |
| ETH-USD | 2 membres | 100,0 % † | 54,9 % |
| GLD | 3 membres † | 99,9 % † | 56,2 % |
| USO | 2 membres † | 100,0 % † | 57,3 % |

† sur le panel du benchmark seul, tous les sous-secteurs sont des singletons ; les
tailles de groupe indiquées sont celles de l'**univers unifié** (les 16 actifs de
DEITA + le panel, calculé depuis la base locale de DEITA, sans réseau), où ETH,
GLD et USO retrouvent des pairs — et où leur conviction redevient effectivement
directionnelle (48,9 %, 55,6 %, 55,0 % de positions longues). La règle se lit sans
ambiguïté : **la dégénérescence frappe exactement les sous-secteurs à un membre**.

### 1.3 Pourquoi l'univers élargi ne sauve pas non plus

Deux raisons, chacune suffisante :

- **SPY et ZN=F restent dégénérés** dans l'univers unifié (100,0 % et 99,3 %
  longs) — or ce sont précisément les deux instruments de l'hypothèse primaire du
  brief. Le programme aurait testé un sizing appliqué à un signal constant.
- **Les prix ne se recollent pas.** Sur SPY, seul actif vérifiable dans les deux
  sources, les log-rendements de la base DEITA et de `prices_v3` s'écartent
  jusqu'à **2,6e-03** — deux ordres de grandeur au-dessus de la tolérance de gel
  du benchmark (1e-5). Faire produire le signal par une série et le trader sur une
  autre réintroduirait exactement ce que le gel des prix interdit. La base DEITA
  ne contient d'ailleurs ni TLT ni ZN=F, et remplace GLD/USO par leurs futures
  GC=F/CL=F — des instruments voisins, pas les mêmes.

### 1.4 La convention de calendrier — un écart qui n'est pas cosmétique

`compute_cta_signal` fait `prices[[asset]].ffill()` **avant** de calculer les
rendements. Sur un panel mixte — crypto 7 j/7, actions 5 j/7, exactement ce que
DEITA lui passe en production — les actions héritent du cours de vendredi le
samedi et le dimanche : deux jours à rendement nul entrent dans la moyenne de Hull
chaque semaine. Sur le calendrier propre à chaque actif, ces jours n'existent pas.

L'écart retourne le signe sur **13 à 15 %** des observations quotidiennes des
actifs à 5 jours (0 % en crypto, que le `ffill` ne touche pas) et sur **7,3 %** des
cellules (origine × actif) de la grille.

**Convention retenue : celle de DEITA** (`ffill`), parce que le brief dit « le
signal CTA vient de DEITA tel quel » et que c'est ce que DEITA fait. Le signal
gelé vaut alors **exactement** le `hull_slope` que publie `compute_cta_signal` :
vérifié sur 28 comparaisons (7 actifs × 4 dates), **0 écart**, et figé par un test
unitaire. La convention « calendrier propre » est calculée à côté et la porte est
jugée sous les deux (§2.4).

### 1.5 Ce qui est retenu, et ce que ça change

**Signal retenu : la direction Hull**, calculée entièrement sur `prices_v3`. Elle
est univariée — elle ne dépend d'aucun univers, seulement de la série de l'actif —
donc le non-négociable « prix gelés partagés » est respecté sans compromis, et
aucun appel réseau n'intervient.

La question du brief est posée à l'identique : *une échelle de risque calibrée
améliore-t-elle le sizing d'un signal directionnel **externe** ?* Le signal reste
celui de DEITA, exogène, non entraîné ici, et le signe resterait commun aux quatre
bras. Ce qui est perdu est la pondération par conviction — laquelle, sur 4 actifs
sur 7, ne pondère rien puisqu'elle ne change jamais de signe. La conviction est
néanmoins calculée et archivée (`conviction.parquet`) pour que la décision reste
vérifiable, et c'est elle qui sert le contrôle de robustesse du §2.4.

### 1.6 La vérification d'absence de look-ahead

Bloquante, et pas déclarative. Pour 5 origines réparties sur la grille — la
première et la dernière comprises — le signal est **entièrement recalculé sur
l'historique tronqué** à cette date, et comparé à la valeur gelée calculée sur
l'historique complet.

| Origine | observations tronquées | écart max |
|---|---:|---:|
| 2020-01-03 | 2 785 | 0,00e+00 |
| 2021-08-13 | 3 373 | 0,00e+00 |
| 2023-03-31 | 3 968 | 0,00e+00 |
| 2024-11-15 | 4 563 | 0,00e+00 |
| 2026-07-02 | 5 157 | 0,00e+00 |

Égalité exacte, tolérance 1e-12. Le test unitaire `test_lookahead_check_detecte_une_fuite_injectee`
vérifie la contre-épreuve : en injectant une fuite artificielle, le mécanisme la
voit — la vérification prouve donc quelque chose, elle ne passe pas par
construction.

Signal, prix et moteur sont hashés (SHA-256) dans `manifest.json`.

## 2. Chantier 0b — la porte : le trend ne vit plus sur cette fenêtre

### 2.1 Le montage

CTA seul, taille fixe |w| = 1, signe donné par le signal gelé. Grille `oos2020` :
354 origines, 7 actifs, régime weekly, prix `prices_v3`. Frais `real_fees` au
niveau central, **coût de roulement H2 inclus** pour les futures. Horizon de
détention W+1 ; W+2 et W+3 en descriptif. Comparaison : acheter-et-garder, mêmes
origines, mêmes frais.

### 2.2 Ce que ça donne (W+1)

| Instrument | part longue | PnL net | vs B&H | hit | Sharpe | p (vs B&H) |
|---|---:|---:|---:|---:|---:|---:|
| ETH-SPOT | 56 % | +101,04 | −7,27 | 51,5 % | 0,66 | 0,936 |
| USO-ETF | 56 % | −2,13 | −18,73 | 50,6 % | −0,02 | 0,706 |
| GLD-ETF | 58 % | −3,04 | −29,76 | 47,9 % | −0,09 | 0,073 |
| ZN-FUT | 48 % | −3,70 | +2,62 | 50,3 % | −0,33 | 0,717 |
| TLT-ETF | 52 % | −9,97 | +0,31 | 47,4 % | −0,34 | 1,000 |
| SPY-ES | 59 % | −12,22 | −41,94 | 50,3 % | −0,33 | **0,044** |
| SPY-ETF | 59 % | −14,11 | −41,94 | 50,3 % | −0,38 | **0,038** |
| BTC-SPOT | 56 % | −24,63 | −94,08 | 47,9 % | −0,21 | 0,124 |

*(PnL en bps par origine.)* Un seul actif dégage un PnL positif — ETH-SPOT, et son
acheter-et-garder fait mieux. **Les deux seules cellules significatives sont SPY,
et elles disent que l'acheter-et-garder bat le CTA** (−41,94 bps, p = 0,038-0,044).

Portefeuille équipondéré, un instrument par actif : **+6,21 bps par origine,
Sharpe 0,17**, tiré par la seule crypto. Par classe : Crypto +38,20, Commodity
−2,58, Bond −6,84, Equity −14,11 → **1 classe positive sur 4**.

### 2.3 Le critère de porte, appliqué tel qu'il est déclaré

- **Branche 1** — « au moins un actif à PnL net positif et p < 0,05 brut » :
  **échec**. Le seul instrument à PnL positif (ETH-SPOT) a p = 0,936 ; les deux
  seuls p significatifs (SPY, 0,038-0,044) portent sur des PnL négatifs, donc en
  faveur de l'acheter-et-garder. Le brief énonce ce critère juste après avoir
  décrit un test *vs B&H* ; « p » pouvant aussi se lire *contre zéro*, la lecture
  concurrente a été calculée et **échoue également** (ETH-SPOT au mieux à 0,091).
  Le verdict ne dépend pas de la lecture.
- **Branche 2** — « Sharpe poolé > 0 avec direction cohérente sur les classes » :
  **échec**. Le Sharpe est positif (0,17) mais 1 seule classe sur 4 l'est, contre
  les 3 exigées par l'opérationnalisation déclarée avant le run.

**Porte 0 : ÉCHEC.**

### 2.4 Le contrôle qui rend l'arrêt robuste — et qui casse la branche 2

La porte a été rejouée sous les deux variantes écartées — le calendrier propre
(§1.4) et la conviction hiérarchique (§1.2) — pour vérifier que l'arrêt ne dépend
d'aucune des deux décisions d'interface :

| | Hull, calendrier DEITA *(retenu)* | Hull, calendrier propre | conviction (rejetée) |
|---|---|---|---|
| part longue | 48-59 % | 48-60 % | 99-100 % |
| edge vs B&H (W+1) | −94 à +3 bps | −94 à +2 bps | **exactement 0,00 bps, p = 1,000** |
| PnL poolé / Sharpe | +6,21 bps / 0,17 | +7,41 bps / 0,21 | +33,31 bps / 0,77 |
| classes positives | 1/4 | 2/4 | 3/4 |
| branche 1 | échec | échec | échec |
| branche 2 | échec | échec | **passe** |
| **porte** | **échec** | **échec** | passe |

L'edge vs B&H exactement nul est la preuve numérique que la conviction dégénérée
**est** l'acheter-et-garder, et non un signal qui lui ressemblerait. Et pourtant
elle franchit la branche 2. La conclusion à retenir dépasse ce programme :

> Sur un panel haussier, un Sharpe poolé positif mesure le bêta du panel, pas
> l'edge du signal. La branche 2 est franchissable par n'importe quel « signal »
> toujours long. Seule la branche 1 — la comparaison appariée contre
> acheter-et-garder — teste ce que la porte prétend tester.

**Réparé depuis** (`PATCH_gate0_branche2_et_holm_m2.md`, P2) : la branche 2 porte
désormais sur l'**excès** — PnL du signal moins PnL d'acheter-et-garder, apparié
par origine — au lieu du PnL brut. Un signal constant a un excès identiquement
nul, donc l'échec devient mécanique quelle que soit la pente du marché. La
correction est **postérieure au verdict et ne l'affecte pas**, et ce n'est pas une
affirmation : les trois variantes ont été rejouées sous la branche réparée et
restent en échec. Le cas historique bascule comme attendu — la conviction
dégénérée passait la branche 2 d'origine (Sharpe 0,77 ; 3/4 classes), elle y
échoue désormais (1/4 classe à excès positif, excès moyen +0,12 bps). Les deux
formulations restent rapportées dans `cta_gate0.json` pour que la comparaison
soit vérifiable.

Un test unitaire fige ce contrôle
(`test_un_signal_toujours_long_est_exactement_acheter_et_garder`) : l'edge doit
valoir zéro **exactement**, pas approximativement.

### 2.5 Le descriptif, pour ce qu'il vaut

**Par tranche de marché** (SPY-ETF, W+1) — le trend devait montrer son « crisis
alpha » en 2020 et 2022 ou nulle part :

| Tranche | n | PnL/origine | Sharpe |
|---|---:|---:|---:|
| 2020 (COVID) | 53 | −2,02 bps | −0,03 |
| 2021 | 52 | −48,63 bps | −2,09 |
| 2022 (bear taux) | 52 | −41,74 bps | **−0,94** |
| 2023 | 52 | +9,41 bps | +0,35 |
| 2024-2026 | 131 | −3,73 bps | −0,13 |

**Le crisis alpha n'apparaît nulle part** : ni en 2020, ni en 2022 — les deux
années où un CTA doit précisément justifier son existence. Une seule tranche sur
cinq est positive. C'est le motif de fond derrière l'échec de la porte, et il est
cohérent avec la décroissance du momentum que le garde-fou n° 2 du brief
anticipait.

*(Sur le calendrier propre, le même découpage donnait +83,99 bps et Sharpe 1,41 en
2020 : le `ffill` des week-ends efface le crisis alpha de mars 2020 en diluant les
rendements de krach dans des jours nuls. C'est une observation intéressante sur la
convention de DEITA — pas un argument pour en changer après coup.)*

**Autres horizons** (descriptif, W+2 / W+3, PnL net en bps) : les signes se
déplacent sans se stabiliser — SPY-ETF −22,2 puis −2,0 ; BTC-SPOT −0,6 puis +16,9 ;
USO-ETF −44,0 puis −5,8. ETH-SPOT domine partout (+166 puis +214) mais son
acheter-et-garder domine autant : c'est le rendement de l'actif, pas celui du
signal.

## 3. Ce qui n'a pas été fait, et pourquoi

Les chantiers 1 et 2 ne sont pas implémentés. Ce n'est pas une réduction de
périmètre décidée en cours de route : c'est le critère d'arrêt n° 1, déclaré dans
le brief avant tout run, et la porte a été construite précisément pour être
capable d'échouer.

Concrètement, ne sont donc pas écrits : les quatre bras de sizing (A largeur
NsDiff / B vol EWMA / C fixe / D largeur GARCH), le budget de risque commun à vol
cible, l'hypothèse primaire A vs B et sa famille de Holm, l'analyse de puissance
préalable, le test unitaire de turnover et le PnL glissant par bras. Le brief est
sans ambiguïté : *« pas de sizing d'un signal mort »*.

**La famille de Holm déclarée a été corrigée, avant tout run** : le brief
annonçait m = 4 pour « A vs B sur {SPY-ES, ZN-FUT} × {W+1} », qui compte 2 tests.
Le brief porte désormais **m = 2**, avec la trace de la correction. Elle reste
une déclaration *a priori* — les chantiers 1-2 n'ont jamais tourné — et l'erreur
allait dans le sens conservateur (m surdéclaré = seuils plus stricts). La famille
n'a pas été étendue à W+2/W+3 pour justifier le 4 : W+1 est l'horizon de
détention déclaré.

## 4. Ce qui rouvrirait la question, et à quelles conditions

Le critère d'arrêt ferme ce programme-ci. Il restait trois pistes ; la deuxième —
« corriger la branche 2 » — est **traitée** par
`PATCH_gate0_branche2_et_holm_m2.md` et sort donc de cette liste (cf. §2.4). Les
deux autres subsistent, et aucune n'est autorisée par le brief actuel — chacune
demanderait une décision explicite :

1. **Réparer le signal de DEITA, chez DEITA.** La dégénérescence des sous-secteurs
   à un membre est un défaut du moteur, pas de ce programme. Un
   `_conviction_level` qui, pour un groupe d'un seul membre, renverrait la
   conviction du niveau supérieur au lieu de la série de l'actif supprimerait le
   carré. C'est une correction à faire dans `cta_quant_engine.py`, à valider chez
   DEITA — et elle changerait les signaux de production de DEITA, ce qui dépasse
   très largement ce brief.
2. **Changer de fenêtre ou de panel.** Interdit ici, et à raison : ce serait
   chercher la fenêtre où le momentum vit encore après avoir vu qu'il ne vit pas
   sur celle-ci.

Le garde-fou n° 1 du brief — le conditionnement exogène de NsDiff — reste fermé,
comme prévu : il n'était envisageable que si ce programme concluait positivement.

## 5. Non-négociables — statut

| Non-négociable | Statut |
|---|---|
| Conventions descriptif / poolé | tenu — tout le §2.5 est étiqueté descriptif, la porte seule décide |
| Seuils et critères déclarés avant les runs | tenu — critère de porte, opérationnalisation de « direction cohérente » (≥ 3 classes sur 4) et lecture concurrente de « p » déclarés dans la docstring avant exécution |
| Briques réutilisées | tenu — `econ_backtest`, `paired_test`, `real_fees` + roulement H2, `prices_v3`, grille `grid2020` |
| Code neuf couvert de tests unitaires | tenu — 15 tests (`test_cta_gate0.py`), dont la contre-épreuve de la vérification de causalité et l'équivalence avec `compute_cta_signal` |
| `tracking.db` en lecture seule | tenu — aucune écriture, aucun accès |
| Prix gelés `prices_v3` partagés | tenu — et c'est ce qui a écarté la base de prix de DEITA |
| Signal DEITA gelé et hashé | tenu — SHA-256 du moteur, des prix et de chaque série de signal dans `manifest.json` |
| Aucun modèle entraîné | tenu — aucun entraînement, le couplage était de toute façon au niveau décision |
| pytest vert avant/après | tenu — 729 passed / 1 skipped avant, **744 passed / 1 skipped** après |

## 6. Limites déclarées

- **Le signal testé n'est pas la conviction hiérarchique de DEITA**, mais sa
  direction Hull — calculée par le code de DEITA, avec sa convention de
  calendrier, et vérifiée égale à son `compute_cta_signal(pure_trend_mode=True)`. La justification est mesurée et détaillée au §1.2-1.4 ; elle
  reste une décision d'interface, et quelqu'un qui la refuserait devrait expliquer
  comment tester un sizing sur un signal qui ne change jamais de signe sur les
  deux instruments de l'hypothèse primaire.
- **L'univers de conviction du contrôle §1.2** mélange la base de prix de DEITA et
  `prices_v3`. C'est acceptable pour un *descriptif* dont la seule fonction est de
  montrer que l'élargissement ne lève pas la dégénérescence sur SPY et ZN=F ; ce
  serait inacceptable pour un résultat, et ça n'en fonde aucun.
- **La porte est jugée à l'horizon W+1 seul.** W+2 et W+3 sont rapportés en
  descriptif et ne changent pas la lecture, mais ils n'ont pas été soumis au
  critère — le brief fixe W+1 comme horizon de détention.
- **Les frais restent des hypothèses déclarées** (`real_fees`), au niveau central.
  Aucun bordereau réel n'est versé au dépôt. À ces niveaux d'écart — le PnL des
  cellules va de −25 à +101 bps pour des frais de 1,6 à 30 bps — la conclusion
  n'en dépend pas.
- **Un panel de 7 actifs est petit pour un CTA.** Le trend-following vit
  habituellement sur des dizaines de marchés, où la diversification fait
  l'essentiel du Sharpe. Ce que la porte établit vaut pour *ce* panel sur *cette*
  fenêtre — ce qui est exactement la question posée, mais ne se généralise pas à
  « le momentum est mort ».
