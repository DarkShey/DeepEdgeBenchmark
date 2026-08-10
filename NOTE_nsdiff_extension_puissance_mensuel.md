# NOTE — NsDiff : extension de données, puissance, dashboard, re-jugement mensuel

*2026-08-08. Réponse au brief « NsDiff : extension de données, puissance,
dashboard, re-jugement mensuel ». Fait suite à `NOTE_nsdiff_edge_vs_frais.md`.
TSDiff est hors périmètre — retiré du benchmark, absent de tous les chantiers.*

**Artefacts produits** :

| Fichier | Chantier | Contenu |
|---|---|---|
| `experiments/repoint_oos_to_ensemble.py` → `.json` | C | bascule du dashboard sur la config production (seule écriture en base) |
| `experiments/prices_v3.py` → `prices_v3/` | A | panel étendu : 7 actifs depuis 2011-05, vérifications bloquantes |
| `experiments/cost_grid_2020.py` → `.json` | B | chiffrage du coût de régénération, **avant** lancement |
| `experiments/grid2020.py` → `grid2020/` | B | grille 340 origines à départ 2020-01, NsDiff + ARIMA-GARCH |
| `experiments/grid2020_tests.py` → `.json` | B | hypothèse primaire, réplication B&H, calibration |
| `experiments/monthly_feasibility.py` (D1+D2) → `monthly_d2/` | D | grille d'époques élargie, pilote sur le panel étendu |
| `experiments/stochvol_synth.py` (+ `test_`) → `monthly_d3/` | D3 | générateur à volatilité stochastique — 18 tests unitaires |

**Décisions actées en entrée, implémentées sans re-discussion** : budget de
tirages unique (pas de différenciation par régime) ; config production =
ensemble 5×200 ; panel déséquilibré assumé ; `weekly_propagated` abandonnée —
mentionnée ici une seule fois, comme voie close, et jamais rejouée.

---

## 0. Le résultat principal, d'emblée

**L'hypothèse primaire pré-déclarée ne survit pas. Le critère d'arrêt du volet
économique est déclenché : il se clôt définitivement.**

Sur 354 origines (`effective_n` = 118, contre 30), `var_limit` sur SPY à W+2 et
W+3 contre GARCH donne **0 rejet sur 4** — pas même au seuil non corrigé. L'edge
net reste positif (+8,3 à +12,2 bps par origine) mais il rétrécit d'un tiers, et
la p-value passe de 0,017-0,022 à 0,087-0,136.

Trois autres résultats tombent avec lui, et c'est le vrai enseignement :

1. **La réplication échoue.** Les 4 survivants « acheter-et-garder bat la
   stratégie » (p = 0,0004 à 0,0082) ne se répliquent pas : p = 0,099 à 0,219.
2. **La parité de calibration ne tient pas.** Rejouée sur la grille longue, elle
   donne 5 survivants Holm, **tous en faveur de GARCH**, zéro pour NsDiff.
3. **L'analyse de puissance s'est trompée, et on sait pourquoi.** Elle prédisait
   une erreur-type de 2,74 bps à 340 origines ; elle vaut 5,45 — inchangée. Le
   gain de quatre fois plus d'observations a été **exactement annulé** par une
   dispersion par origine deux fois plus grande : 2020 et 2022 ne ressemblent pas
   à 2024-2026. *Allonger la fenêtre ne se contente pas d'ajouter des données ;
   cela change le processus mesuré.*

**Ce qui va dans l'autre sens** : le volet mensuel, lui, s'améliore franchement
grâce à l'extension — **TLT et USO passent le go/no-go**, les deux premiers du
programme, et l'augmentation synthétique répare l'effondrement des cellules
courtes. Le mensuel n'était pas condamné par le modèle mais par le volume de
données ; le volet économique, lui, l'était par l'absence d'effet.

---

## 1. Chantier C — le dashboard porte enfin la configuration de production

Le balayage du chantier précédent avait établi qu'une graine unique à 200 tirages
**n'est pas convergée** (le régime weekly en exige 800 aux trois horizons), alors
que l'ensemble 5×200 = 1000 l'est sur 6 cellules sur 6. Le dashboard affichait
donc une configuration que le programme savait sous-convergée, et qui n'était pas
celle qu'on déploierait.

**Bascule appliquée** (`repoint_oos_to_ensemble.py`, dry-run par défaut,
sauvegarde horodatée, vérification 1:1 des clés avant écriture) :

| | avant (graine 42 × 200) | après (ensemble 5×200) |
|---|---|---|
| Couverture 95 % | 0,9315 | **0,9452** |
| Largeur du PI (% du prix) | 23,26 | 23,88 |
| Winkler moyen | 10 307 | **9 382** |

2 700 lignes, clés identiques une à une, prix alignés à 2,2·10⁻⁷, cibles
identiques. Colonnes dérivées remises à NULL puis recalculées.

**Ce que cela change pour le lecteur, et qui doit être dit** : la ligne NsDiff
du dashboard cesse de répondre à *« qu'obtient-on en tirant une graine au
hasard ? »* pour répondre à *« qu'obtient-on en déployant les cinq ? »*. Ce n'est
pas une amélioration du modèle — c'est un changement de configuration, et la
couverture monte parce que le mélange de cinq lois prédictives est plus large que
chacune d'elles. Le bandeau de configuration du dashboard porte désormais
`seeds: [42..46]` et `n_samples_effective: 1000`, et `benchmark_registry` en fait
foi.

---

## 2. Chantier A — le panel étendu

### 2.1 Ce qui a été constitué

`prices_v3/` — répertoire **nouveau**, les anciennes séries restent intactes.
Départ commun 2011-05 là où l'historique le permet, GLD et USO ajoutés.

| Actif | Span | daily | weekly | mensuel | fenêtres train daily | ≥ 2 000 ? |
|---|---|---|---|---|---|---|
| SPY | 2011-05 → 2026-07 | 3 829 | 795 | 183 | 2 137 | oui |
| ZN=F | 2011-05 → 2026-07 | 3 828 | 795 | 183 | 2 133 | oui |
| TLT | 2011-05 → 2026-07 | 3 829 | 795 | 183 | 2 137 | oui |
| **GLD** | 2011-05 → 2026-07 | 3 829 | 795 | 183 | 2 137 | oui |
| **USO** | 2011-05 → 2026-07 | 3 829 | 795 | 183 | 2 137 | oui |
| BTC-USD | 2014-09 → 2026-07 | 4 328 | 619 | 143 | 1 887 | **non** |
| ETH-USD | 2017-11 → 2026-07 | 3 179 | 455 | 105 | 738 | **non** |

À partir de 2020-01 : **340 origines de test**, `effective_n` = 113 pour tous les
actifs (contre 90 et 30 aujourd'hui).

**Cellules faibles déclarées** : BTC-USD et ETH-USD n'atteignent pas la cible de
2 000 fenêtres d'entraînement daily. Elles sont **déclarées telles quelles**, ni
tronquées ni complétées par une source alternative — l'homogénéité de source
prime, un actif dont la moitié de l'historique viendrait d'ailleurs ne serait
plus comparable aux autres.

*Gain accessoire mais réel* : TLT remonte à 2011 (795 semaines) alors que la
série gelée précédente démarrait en 2018 (441 semaines) — elle venait d'un
instantané offline plus court.

**Régimes traversés**, le gain étant de validité externe autant que de puissance :
2011-2015 (QE, taux zéro), 2015-2020 (normalisation, corrections 2015 et 2018),
2020 (choc COVID), 2022 (marché baissier de taux, corrélation actions/obligations
positive), 2024-2026 (la fenêtre du benchmark actuel).

### 2.2 La vérification bloquante a servi — et pas comme prévu

Le premier jet comparait les **prix** à 2·10⁻⁷. Il a bloqué sur SPY (7,1·10⁻⁷) et
TLT (4,0·10⁻³). Le diagnostic sépare radicalement les deux cas :

- **SPY** — ratio nouveau/ancien médian exactement 1,00000000, dispersion
  1,4·10⁻⁶, aucune date au-delà de 10⁻⁵. C'est le bruit de dernière décimale que
  yfinance ressert d'un appel à l'autre, déjà documenté dans le repo. La
  tolérance était simplement trop serrée : la convention de travail du repo est
  10⁻⁶ à 4·10⁻⁶.
- **TLT** — ratio médian **0,99598782**, dispersion 2,5·10⁻⁶. Un facteur
  multiplicatif **constant**, pas une révision : l'ancienne série venait de
  `fetch_tlt_patched` (instantané offline), la nouvelle de yfinance, et les deux
  n'appliquent pas le même ajustement de dividendes.

Or **tous les modèles de ce repo travaillent sur les log-rendements**, que
multiplier tous les prix par une constante laisse strictement inchangés. La
vérification a donc été refaite sur ce que les modèles voient réellement :

| Contrôle | Tolérance | SPY | TLT | ZN=F / BTC / ETH |
|---|---|---|---|---|
| écart max sur les **log-rendements** | 10⁻⁵ | 1,1·10⁻⁶ | 1,9·10⁻⁶ | **0** |
| **dispersion** du ratio de prix | 10⁻⁴ | 1,4·10⁻⁶ | 2,5·10⁻⁶ | **0** |

Les deux passent. La distinction est celle qui compte : un ratio **constant** est
un changement de base d'ajustement, sans effet sur les rendements ; un ratio
**qui dérive** serait une révision d'historique, et celle-là invaliderait toute
comparaison avec les résultats publiés. Le test sur les prix bruts ne faisait
pas cette distinction — il confondait les deux.

*Réserve déclarée* : les **niveaux** de prix de `prices_v3/TLT` ne sont pas
comparables à ceux de l'ancienne série (facteur 0,996). Seuls les rendements
le sont — ce qui suffit à tous les modèles, mais pas à une comparaison de prix.

### 2.3 GLD et USO dans la grille de frais

Ajoutés à `real_fees.py` avec leurs trois niveaux : GLD 2,5 / **4,0** / 6,0 bps
aller-retour, USO 3,0 / **5,0** / 8,0. Au niveau central, GLD passe sous la borne
haute de l'edge mesuré (5 bps), USO est exactement à la borne — le filtre étant
strict, il l'exclut.

*Réserve déclarée sur USO* : il porte un coût de roulement structurel (contango)
qui n'est **pas** un frais de transaction et n'est donc pas modélisé. Il affecte
le rendement du sous-jacent, pas l'écart entre deux modèles qui le prévoient tous
les deux.

---

## 3. Chantier B — chiffrer d'abord, découper ensuite

### 3.1 Le chiffrage, exigé par le brief avant tout lancement

`cost_grid_2020.py` ne devine pas : il **chronomètre** chaque modèle sur des
origines réelles et extrapole à la grille de 340 × 7 × 2 = 4 760
origines-cellules.

| Modèle | Protocole | Coût extrapolé |
|---|---|---|
| **LSTM** | refit par origine | **13,72 h** |
| Prophet | refit par origine | 1,01 h |
| ARIMA-GARCH | refit par origine | 0,87 h |
| SARIMA | refit par origine | 0,40 h |
| NsDiff | train-once-forward, 5 graines | 0,30 h |
| Naive | refit par origine | 0,00 h |
| **TOTAL** | | **16,31 h** |

Le seuil de décision était déclaré à 2 h : **hors budget, découpage requis**. Et
**LSTM représente 84 % du total à lui seul.**

### 3.2 Le découpage, et ce qu'il laisse de côté

La règle de découpe est celle du brief : l'hypothèse primaire est weekly, sur
SPY, NsDiff contre GARCH. Or **aucun test du chantier B ne fait intervenir
SARIMA, Prophet, LSTM ou Naive** — ils servent la comparabilité du *dashboard*,
pas le test confirmatoire.

Ce qui a été régénéré : **NsDiff (5 graines) + ARIMA-GARCH, les deux régimes, les
7 actifs — 1,17 h**. Cela couvre l'hypothèse primaire, la réplication
« acheter-et-garder », le match de calibration, et toutes les cellules
exploratoires.

Ce qui ne l'a pas été : les 4 modèles classiques. **Chiffrés, non exécutés,
déclarés tels quels.** Conséquence à assumer : le dashboard ne peut pas être
repointé sur la grille 2020 sans ces 15,1 h — c'est un chantier séparé, dont le
coût est maintenant connu au lieu d'être supposé.

### 3.3 L'hypothèse primaire, pré-déclarée

Fixée **avant** d'avoir vu la moindre donnée de cette grille, et inscrite en dur
dans `grid2020_tests.py` :

> `var_limit`, SPY (véhicules ES et ETF), W+2 et W+3, régime **weekly**,
> NsDiff-ensemble contre ARIMA-GARCH. Famille de Holm : ces 4 tests, et eux seuls.

Le motif du choix est lui aussi antérieur : l'analyse de puissance donnait pour
ces cellules un *n* requis sous Holm de **231 à 270 origines**, contre **340**
disponibles ici — les seules du programme à être à la fois favorables et
atteignables. C'est la réponse à la limite que la note précédente s'imposait
(« l'edge n'est pas corrigé pour la sélection d'instrument ») : l'instrument, la
stratégie, les horizons et le régime sont choisis avant, pas après.

**Critère d'arrêt global, déclaré dans le brief** : si l'hypothèse primaire ne
survit pas à Holm sur 340 origines, le volet économique du programme se clôt
définitivement — il n'y aura pas de troisième univers de test.

### 3.4 Résultat : l'hypothèse primaire ne survit pas

**354 origines, `effective_n` = 118** (contre 90 et 30). Famille de Holm : les
4 tests pré-déclarés, seuil le plus strict 0,0125.

| Cellule | edge net | p | p ajustée | Verdict Holm |
|---|---|---|---|---|
| SPY-ES · W+2 | +8,54 bps | 0,1228 | 0,3464 | indistinguable |
| SPY-ES · W+3 | +12,22 bps | 0,0866 | 0,3464 | indistinguable |
| SPY-ETF · W+2 | +8,25 bps | 0,1364 | 0,3464 | indistinguable |
| SPY-ETF · W+3 | +11,95 bps | 0,0946 | 0,3464 | indistinguable |

**0 rejet brut sur 4.** Pas même au seuil non corrigé de 5 %.

### 3.5 Pourquoi la p-value empire alors que la puissance quadruple

C'est le résultat le plus instructif du chantier, et il contredit l'analyse de
puissance qui avait justifié l'extension. Sur la même cellule :

| Cellule | Grille | n | eff_n | effet | SE | **écart-type par origine** |
|---|---|---|---|---|---|---|
| SPY-ES · W+2 | 2024-2026 | 90 | 30 | +11,49 | 5,32 | **29,1** |
| SPY-ES · W+2 | 2020-2026 | 340 | 118 | +8,54 | 5,45 | **59,2** |
| SPY-ES · W+3 | 2024-2026 | 90 | 30 | +16,48 | 8,21 | **44,9** |
| SPY-ES · W+3 | 2020-2026 | 340 | 118 | +12,22 | 7,17 | **77,8** |

L'erreur-type prédite à 340 origines était de **2,74 bps** (loi en 1/√n). Elle
vaut **5,45** — elle n'a pas bougé. Deux causes se cumulent, et aucune n'était
anticipée :

1. **L'effet rétrécit d'un tiers** (+11,5 → +8,5 bps). Il reste positif, mais il
   était plus grand sur la fenêtre où il avait été repéré.
2. **La dispersion par origine double** (29 → 59). 2020 (choc COVID) et 2022
   introduisent des écarts de PnL sans commune mesure avec 2024-2026. Le gain de
   puissance apporté par quatre fois plus d'origines est **exactement annulé**
   par une variance par origine deux fois plus grande.

L'analyse de puissance supposait l'effet **et** sa variance stables. Ni l'un ni
l'autre ne l'était. C'est une limite générique de ce genre de calcul — il
extrapole une hétérogénéité qu'il ne peut pas connaître — et il faut la retenir
comme telle : *une cible de puissance calculée sur une fenêtre courte et
homogène n'est pas atteignable en allongeant la fenêtre, parce qu'allonger la
fenêtre change le processus.*

### 3.6 La réplication échoue aussi

Les 4 survivants « acheter-et-garder bat la stratégie » (SPY W+1 weekly, ~−12 bps,
p = 0,0004 à 0,0082 sur 90 origines) **ne se répliquent pas** : p = 0,099 à 0,219
sur 354 origines, aucun verdict significatif. Le seul résultat que la note
précédente présentait comme statistiquement solide était donc, lui aussi,
spécifique à sa fenêtre.

### 3.7 La parité de calibration ne tient pas non plus

Le match Winkler NsDiff-ensemble contre GARCH, rejoué au nouveau départ
(21 cellules weekly, Holm) : **9 rejets bruts → 5 survivants, tous en faveur de
GARCH**, aucun en faveur de NsDiff.

| Survivant Holm | Cov95 ns / GARCH | Winkler ns / GARCH |
|---|---|---|
| ETH-USD · W+1 | 1,000 / 0,950 | 3 709 / **1 223** |
| ETH-USD · W+2 | 1,000 / 0,935 | 5 853 / **1 681** |
| ETH-USD · W+3 | 1,000 / 0,912 | 7 914 / **2 124** |
| BTC-USD · W+2 | 0,971 / 0,947 | 32 020 / **28 860** |
| SPY · W+1 | 0,871 / 0,959 | 64,3 / **51,6** |

La parité établie au chantier A2 sur 90 origines **était un artefact de
puissance**. Sur quatre fois plus de données, GARCH gagne 5 cellules et NsDiff
zéro. Le cas d'ETH est spectaculaire et cohérent avec le chantier A : NsDiff y
couvre 100 % avec un Winkler trois fois pire que GARCH — c'est la cellule à
738 fenêtres d'entraînement, déclarée faible avant tout run.

Les 5 actifs à historique long (SPY hors W+1, GLD, TLT, USO, ZN=F) restent
indistinguables — 15 cellules sur 21.

### 3.8 Le critère d'arrêt global est déclenché

Le brief le déclarait avant tout run : *« si l'hypothèse primaire ne survit pas à
Holm sur 340 origines, le volet économique du programme se clôt définitivement —
il n'y aura pas de troisième univers de test. »*

**Elle ne survit pas. Le volet économique est clos.**

C'est précisément ce pour quoi la pré-déclaration existe. Sur la grille de
90 origines, `var_limit` sur SPY affichait p = 0,017 et 0,022 — assez pour
donner envie d'y croire, pas assez pour survivre à Holm. Le test confirmatoire
sur données indépendantes tranche : l'effet ne généralise pas. Sans
pré-déclaration, on aurait pu chercher, parmi les 96 cellules exploratoires de
cette grille, celles qui « marchent » — il y en a **13 significatives en brut,
pro-NsDiff**, et rien n'aurait empêché de les présenter.

*Note sur l'exploratoire, étiqueté tel quel et ne fondant rien* : 64 cellules sur
96 ont un edge net positif ; 13 sont brutes-significatives pro-NsDiff, 2
pro-GARCH ; et la famille 3 à 80 % ouvre enfin des positions (22, contre 0 sur la
grille courte) — la fenêtre 2020-2026 contient des périodes où un PI à 80 %
exclut le rendement nul. Aucun de ces chiffres n'est corrigé pour les tests
multiples, et aucun ne doit être lu comme un résultat.

---

## 4. Chantier D — le mensuel re-jugé

### 4.1 D1 — la grille d'époques, élargie une troisième fois

La règle est inchangée depuis le début : **« argmin au bord ⇒ élargir une fois »**.
Son historique :

| Grille | Résultat | Suite |
|---|---|---|
| (20, 40, 80) | argmin à 80, **au bord** | élargie |
| (20, 40, 80, 160, 320) | argmin à 160, à l'intérieur | suffisante *à ce volume* |
| (20, 40, 80, 160, 320, **640**) | chantier D1 | — |

Le motif du troisième élargissement n'est pas un argmin au bord mais un
**changement de volume** : l'historique mensuel double (~105-139 → 183
observations pour les actifs longs), donc l'optimum se déplace vers le haut et
une grille calée sur l'ancien volume redeviendrait tronquante. Élargie **avant**
de regarder le moindre résultat de test, comme les fois précédentes.

### 4.2 D2 — le pilote sur données étendues : deux GO, les premiers du programme

Le pilote `monthly_native` rejoué sur les **7 actifs**, sur les données étendues,
avec la grille D1. Grille de test volontairement **inchangée** (36 origines,
`effective_n` = 12) : l'extension sert l'ENTRAÎNEMENT (70 → 114 fenêtres à
seq_len=30), pas le test — garder la même fenêtre d'évaluation est ce qui rend la
comparaison avec le verdict précédent honnête.

| Actif | config retenue | Cov95 | largeur (% prix) | Winkler ns / GARCH | Verdict |
|---|---|---|---|---|---|
| **TLT** | seq=30, 80 ép. | 0,935 | 21,60 | **22,4** / 24,3 | **GO** |
| **USO** | seq=12, 160 ép. | 0,935 | 46,62 | **96,0** / 92,2 | **GO** |
| ZN=F | seq=30, 80 ép. | 0,935 | 8,16 | 11,2 / 10,9 | NO-GO |
| SPY | seq=12, 160 ép. | 0,852 | 20,29 | 174,4 / 140,6 | NO-GO |
| GLD | seq=30, 320 ép. | 0,750 | 22,47 | 177,2 / 140,8 | NO-GO |
| BTC-USD | seq=12, 160 ép. | 1,000 | 119,87 | 84 571 / 78 558 | NO-GO |
| ETH-USD | seq=30, 320 ép. | 1,000 | 171,97 | 4 643 / 3 656 | NO-GO |

**Le mensuel cesse d'être uniformément NO-GO.** TLT passe les deux critères aux
trois horizons (0,917 / 0,972 / 0,917) et bat même GARCH-monthly sur le Winkler à
M+2 et M+3. USO aussi. C'est le premier résultat positif du volet mensuel depuis
son ouverture — et il tient à l'extension des données, puisque le protocole de
test est identique.

**Trois échecs de natures différentes**, qu'il serait faux de confondre :

- **ZN=F échoue d'un seul horizon** : M+1 à 0,861 alors que M+2 et M+3 sont en
  bande. Winkler jamais pire. Le même « à une observation près » que SPY dans la
  note précédente ;
- **SPY et GLD sous-couvrent** (0,852 et 0,750). C'est un **renversement** : sur
  les données courtes, SPY *sur*-couvrait. À configuration retenue identique
  (seq=12, 160 époques), plus de données d'entraînement a resserré les bandes —
  et le modèle a traversé la cible au lieu d'y converger. GLD, jamais testé
  auparavant, est le plus mauvais du panel ;
- **BTC et ETH s'effondrent en sur-couverture** (1,000, avec des PI valant 120 %
  et 172 % du prix). Ce sont exactement les deux cellules que le chantier A a
  déclarées faibles — leur historique n'existe pas avant 2014 et 2017, et aucune
  extension ne peut le créer.

### 4.3 D3 — déclenché, exécuté, et il ne conclut pas ce qu'il devait conclure

#### Une correction de diagnostic, imposée par l'écriture des tests

Avant les résultats, un point de méthode qui **corrige la note précédente**.
J'y attribuais l'échec de KernelSynth à l'**homoscédasticité** : une série tirée
d'un processus gaussien à covariance fixe a une variance conditionnelle
déterministe. Le premier test du nouveau générateur comparait donc l'ACF des
rendements **au carré** — la mesure usuelle du clustering de volatilité. Il a
échoué, et pas dans le sens attendu. Mesuré contre les **7 séries mensuelles
réelles** du panel :

| | ACF(r) **niveau** | ACF(r²) | excès de kurtosis |
|---|---|---|---|
| réel mensuel, moyenne | **+0,008** | +0,181 | +1,9 |
| KernelSynth | **+0,451** | +0,289 | −0,31 |
| `stochvol_synth` (calibré) | **−0,007** | +0,178 | +6,1 |

KernelSynth affiche une ACF(r²) **plus élevée** que le nouveau générateur. Ce
n'est pas du clustering : ses séries sont lisses, leurs **niveaux** sont
autocorrélés à +0,45, et cela gonfle mécaniquement l'ACF de leurs carrés. Le
défaut décisif est donc ailleurs, et il est plus grave : ces séries étant
consommées **telles quelles** comme rendements standardisés, on apprenait au
modèle qu'un rendement se prédit par le précédent — alors qu'un rendement réel
est blanc (+0,008).

Conséquences, toutes appliquées : le test discriminant porte désormais sur l'ACF
de **niveau** ; un test dédié verrouille le piège de l'ACF des carrés lue seule ;
et le générateur est **calibré sur les faits stylisés mesurés** plutôt que sur
des plages choisies a priori — calibration faite sur les séries réelles, avant
tout run de D3. *Réserve déclarée* : sa kurtosis (+6,1) dépasse la moyenne réelle
(+1,9) tout en restant dans la fourchette des actifs individuels (USO : +11,1).
Pour un modèle d'incertitude qui sur-couvrait, des queues légèrement trop
épaisses poussent du côté conservateur.

#### Le résultat

**Le déclencheur du brief est rempli, et précisément sur les actifs qu'il
nomme** : « sous-entraînement manifeste (PI sur-larges + couverture ~100 % sur
les actifs courts) » — BTC (1,000 ; 120 % du prix) et ETH (1,000 ; 172 %). D3 a
donc été lancé sur ces deux-là, et sur eux seuls. Les deux générateurs sont
comparés dans le même run, avec le même ratio d'augmentation et le même
protocole : seul le générateur change.

| Actif | Voie | Cov95 | largeur % | Winkler | RMSE | **CRPS** | Verdict |
|---|---|---|---|---|---|---|---|
| BTC | GARCH-monthly | 0,991 | 113,2 | 78 558 | 15 970 | — | baseline |
| BTC | `monthly_native` | 1,000 | 119,9 | 84 571 | 18 940 | 9 303 | NO-GO |
| BTC | **`stochvol`** | 0,963 | 91,9 | 67 680 | 16 650 | **8 579** | NO-GO |
| BTC | `kernelsynth` | 0,963 | 89,7 | **65 420** | 18 340 | 9 207 | **GO** |
| ETH | GARCH-monthly | 0,907 | 102,7 | 3 656 | 794 | — | baseline |
| ETH | `monthly_native` | 1,000 | 172,0 | 4 643 | 1 142 | 551 | NO-GO |
| ETH | **`stochvol`** | 0,972 | 118,6 | **3 414** | 874 | **439** | NO-GO |
| ETH | `kernelsynth` | 0,926 | 114,3 | 3 835 | 988 | 484 | NO-GO |

**Ce qui est établi** : l'augmentation, quel que soit le générateur, **répare
l'effondrement**. La couverture passe de 1,000 à 0,963-0,972, la largeur du PI
chute de 120 % à 92 % (BTC) et de 172 % à 119 % (ETH). Le diagnostic de
sous-entraînement était juste, et le remède est le bon.

**Ce qui n'est pas établi — et c'est le point honnête** : que le générateur à
volatilité stochastique soit décisivement meilleur. Il gagne sur les **règles de
score propres** (CRPS meilleur sur les deux actifs ; Winkler meilleur sur ETH,
et meilleur que GARCH-monthly lui-même à 3 414 contre 3 656), mais le verdict
binaire va dans l'autre sens sur BTC :

- sur **BTC**, `stochvol` échoue au seul M+3, en **sur**-couvrant (1,000) ;
  KernelSynth y passe (0,972) et décroche le GO ;
- sur **ETH**, `stochvol` échoue aussi au seul M+3, encore en sur-couvrant ;
  KernelSynth y échoue en **sous**-couvrant (0,861) **et** en étant
  significativement pire que GARCH (p = 0,021).

Autrement dit les deux générateurs échouent au même endroit — l'horizon le plus
long — mais **dans des directions opposées** : `stochvol` reste trop prudent,
KernelSynth devient trop confiant. Pour un modèle d'incertitude, ces deux échecs
ne se valent pas ; le second est le dangereux.

**Conclusion de D3, telle que les chiffres la portent** : *l'augmentation
synthétique est validée comme remède au sous-entraînement des cellules courtes ;
le choix du générateur pèse moins que prévu, et là où il pèse (ETH à M+3) il
favorise `stochvol`. Un seul GO est décroché, par KernelSynth sur BTC, et il
tient à un horizon.* Le générateur à volatilité stochastique n'était pas la
percée annoncée — il était l'hypothèse à tester, elle est testée, et elle ne se
confirme qu'à moitié.

*Réserve de méthode* : le go/no-go est binaire et conjonctif (trois horizons ×
deux critères). À `effective_n` = 12, un horizon suffit à faire basculer un
verdict. C'est pourquoi les scores propres — CRPS, Winkler — sont rapportés à
côté : ils ordonnent les voies là où le verdict binaire ne fait que les trier
en deux tas.

---

## 5. Non-négociables — statut

- **Multi-graines 42-46** partout ; la configuration production (ensemble) et le
  bras « graine tirée au hasard » sont évalués côte à côte dans chaque cellule.
- **Familles de Holm et tous les seuils déclarés AVANT les runs.** L'hypothèse
  primaire du chantier B est écrite en dur dans `grid2020_tests.py`, avec son
  motif (le *n* requis calculé au chantier précédent) — c'est ce qui rend son
  échec informatif plutôt que négociable.
- **Briques réutilisées** : `econ_backtest` (moteur inchangé), `paired_test`,
  `mcs.spa_test`, `multiple_testing`, `nsdiff_production_spec`, `real_fees`,
  `calibration_tests`, `dashboard_d7_w1`, machinerie monthly existante,
  `benchmarks.multi_horizon`. Code neuf uniquement là où rien n'existait
  (`prices_v3`, `cost_grid_2020`, `grid2020`, `grid2020_tests`,
  `repoint_oos_to_ensemble`, `stochvol_synth`), couvert de tests unitaires.
- **`tracking.db`** : une seule écriture, celle du chantier C, par script dédié
  (dry-run par défaut, sauvegarde horodatée, `--apply`, vérification 1:1 des clés).
  Tous les autres scripts sont en lecture seule.
- **Prix gelés partagés** : les deux bras de la grille 2020 lisent la MÊME série
  `prices_v3/`, donc voient la même cible et le même prix courant à chaque
  origine — par construction, il n'y a pas deux sources à réconcilier.
- **Budget d'échantillonnage strictement égal** entre modèles comparés.
- **pytest vert** : **593 passed, 1 skipped** (575 avant ce chantier, +18 tests
  neufs pour `stochvol_synth` et la grille de frais étendue). Le skip est
  pré-existant et sans rapport.

## 6. Limites déclarées

- **Quatre modèles de référence n'ont pas été régénérés** sur la grille 2020 :
  SARIMA, Prophet, LSTM, Naive. Chiffrés (15,1 h, dont 13,7 h pour LSTM), non
  exécutés, parce qu'aucun test de ce chantier ne les fait intervenir.
  Conséquence directe : **le dashboard ne peut pas être repointé sur la grille
  2020** sans ce calcul. C'est un chantier séparé, dont le coût est désormais
  connu au lieu d'être supposé.
- **BTC-USD et ETH-USD restent des cellules faibles** (1 887 et 738 fenêtres
  d'entraînement daily contre une cible de 2 000). Aucune extension ne peut le
  corriger : leur historique n'existe pas avant 2014 et 2017. Leurs résultats —
  notamment les 4 survivants Holm de calibration en faveur de GARCH — doivent se
  lire à travers cette limite.
- **La grille de frais reste un jeu d'hypothèses déclarées**, pas un relevé de
  bordereaux. GLD et USO y ont été ajoutés au même titre. Le coût de roulement
  d'USO (contango) n'est pas modélisé — il affecte le sous-jacent, pas l'écart
  entre deux modèles qui le prévoient tous les deux.
- **Les niveaux de prix de `prices_v3/TLT` ne sont pas comparables** à ceux de
  l'ancienne série (facteur constant 0,996, base de dividendes différente). Seuls
  les rendements le sont — ce qui suffit à tous les modèles, mais pas à une
  comparaison de prix.
- **Le go/no-go mensuel est binaire et conjonctif** (3 horizons × 2 critères) à
  `effective_n` = 12 : un seul horizon fait basculer un verdict. ZN=F échoue
  ainsi d'une case, et `stochvol` perd le GO sur BTC pour la même raison. Les
  scores propres (CRPS, Winkler) sont rapportés à côté précisément parce qu'ils
  ordonnent là où le verdict binaire ne fait que trier.
- **La grille de test mensuelle est restée à 36 origines** alors que l'historique
  en permettrait davantage. Choix délibéré : l'extension sert l'ENTRAÎNEMENT, et
  garder la fenêtre d'évaluation identique est ce qui rend la comparaison avec le
  verdict précédent honnête. Une grille de test plus longue est une expérience
  différente, à déclarer comme telle.
- **Non fait, volontairement** : aucun re-run de `weekly_propagated` (voie close,
  décision actée) ; aucun balayage de niveau au-delà de 80 % pour la famille 3 ;
  aucun troisième univers de test économique (critère d'arrêt déclenché).
