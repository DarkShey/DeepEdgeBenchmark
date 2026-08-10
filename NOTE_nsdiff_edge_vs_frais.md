# NOTE — NsDiff : l'edge face aux frais réels, budget de tirages, puissance

*2026-08-07. Réponse au brief « NsDiff : rouvrir la question économique par le
rapport edge/frais ». Fait suite à `NOTE_nsdiff_backtest_eco_et_recadrage.md`
(NO-GO trading) et à la Note de décision du 7/8.*

**Artefacts produits** :

| Fichier | Chantier | Contenu |
|---|---|---|
| `experiments/patch_note_decision.py` → `.json` | Point 0 | correction de la Note de décision (dry-run par défaut, sauvegarde, idempotent) |
| `experiments/nsamples_sweep.py` → `.json` | 3 | balayage du budget de tirages, par graine **et** sur la config production |
| `experiments/diffusion_multiseed_n800/` | 3 | artefact NsDiff à 800 tirages (5 graines × 5 actifs) |
| `experiments/real_fees.py` (+ `test_`) | 1.1 | grille de frais réels par instrument — 14 tests unitaires |
| `experiments/garch_pi80.py` → `garch_pi80/` | 1.3 | bandes ARIMA-GARCH à 80 %, dérivées et validées |
| `experiments/nsdiff_edge_vs_fees.py` → `.json` | 1.2 / 1.3 | matrice instrument × stratégie × horizon : edge brut, frais, edge net |
| `experiments/power_analysis.py` → `.json` | 2 | effective_n requis, origines disponibles, candidats d'extension |

**Modification de fichier partagé, déclarée** : `benchmarks/multi_horizon.py`
gagne un kwarg `level` (opt-in, défaut 0.95 = comportement historique
bit-for-bit) sur `forecast_from_fitted_arima` / `forecast_horizons_arima`.
Nécessaire au chantier 1.3 : reconstruire la bande à 80 % à partir de la bande à
95 % aurait supposé une loi gaussienne, alors que le modèle expose sa propre
fonction quantile — on la lui demande.

---

## 0. Le résultat principal, d'emblée

**L'argument qui fondait le NO-GO ne s'applique pas à la question posée.**

Le NO-GO reposait sur une comparaison : *« l'avantage vaut +3 à +5 bps par
transaction, alors que passer un ordre coûte 10 bps sur les actions et 60 sur la
crypto — le gain est sous son propre coût. »* Cette phrase compare un **écart
différentiel** (NsDiff moins GARCH) à un **coût absolu**. Elle n'est valide que
si l'alternative est de *ne pas trader*.

Or la question de production est autre : *« pour dimensionner mes positions,
j'utilise NsDiff ou GARCH ? »* Dans ce cadre, **les deux bras paient des frais,
sur des expositions quasi identiques, et ces frais s'annulent** :

| | médiane | maximum |
|---|---|---|
| \|écart de frais entre les deux bras\| | **0,095 bps** | 1,84 bps |
| \|edge brut\| | **6,43 bps** | 240,9 bps |
| poids des frais dans l'edge différentiel | **2,1 %** | — |

Trois conséquences, dans l'ordre :

1. **L'edge par instrument est bien plus grand que la moyenne qui fondait le
   NO-GO.** +3 à +5 bps était l'edge *poolé entre actifs* — une moyenne
   d'écarts de signes opposés. Par instrument, il monte à **+16,5 bps** par
   origine (SPY, `var_limit`, W+3 weekly). Les deux chiffres sont corrects ;
   ils ne mesurent pas la même chose, et c'est le second qui gouverne une
   décision d'allocation par instrument.
2. **Ce qui ne change pas** : rien ne survit à Holm face à GARCH. 30 rejets
   bruts sur 54 familles, **0 survivant** sur les comparaisons vs GARCH. Le mur
   GARCH tient — mais il tient sur la **puissance**, pas sur les frais.
3. **Ce qui devient mesurable** : l'analyse de puissance montre que la cible
   déclarée par le brief (3 bps) est **inatteignable** (2 200 à 19 000 origines
   requises contre 601 disponibles au maximum absolu), alors que les effets
   réellement observés par instrument le sont — **150 à 270 origines** pour les
   meilleures cellules, contre 90 aujourd'hui et **340 disponibles** en
   remontant la grille à 2020.

Et deux dossiers se ferment définitivement : la famille 3 (§3.3) et la réserve
« 200 tirages pas prouvé optimal » (§2).

---

## 1. Point 0 — la Note de décision est corrigée

Cinq corrections, appliquées par `experiments/patch_note_decision.py`
(dry-run par défaut, sauvegarde horodatée, `--apply` explicite, **idempotent** —
rejouable sans effet, et rejouable depuis la sauvegarde) :

1. **§3, « Réserves et suite »** — la note affirmait que « le dashboard D7/W1
   tourne encore en single-seed/50, scripts prêts et smoke-testés, non
   exécutés ». C'est faux depuis le 6/8 : `repoint_oos_to_m200.json` porte
   `applied=true`, run `20260806-oos-repoint-m200`, 2 700 lignes,
   couverture 0,909 → 0,9315. Remplacé par l'état réel, et par ce qui reste
   effectivement ouvert : l'intégration de l'**ensemble 5×200** au dashboard.
2. **Tableau de synthèse, ligne « Cadrage 200 tirages »** — même correction.
3. **§4, contrainte de puissance** — `effective_n` mensuel 12 vs 13 n'était pas
   une contradiction mais **deux grilles de test** : 40 origines → 13 (piste
   daily-poussé vs natif), 36 origines → 12 (faisabilité du chantier C). Rendu
   explicite des deux côtés, même formule, mêmes blocs de 3.
4. **Tableau de synthèse, ligne « Cadrage mensuel »** — même harmonisation.
5. **§2, nouveau paragraphe** — ajout des résultats **B3** (pricing d'option :
   GARCH gagne 7 cellules sur 10 ; fonctionnelles de trajectoire
   indistinguables) et **A3-ii** (refit ×24,6 : aucun verdict déplacé). Les deux
   ferment des explications alternatives et renforcent la recommandation.

*Note de méthode* : le premier jet du patch repérait les paragraphes par
**index**. L'insertion de la correction 5 décalait tous les suivants, rendant le
script non rejouable — il se croyait devant un paragraphe inattendu et refusait
de tourner. Corrigé : le repérage se fait par **contenu**.

---

## 2. Chantier 3 — le budget de tirages, tranché

### 2.1 La méthode : sous-échantillonnage emboîté, pas cinq runs

Cinq runs indépendants à budgets différents tirent cinq séquences aléatoires
différentes : leur écart mélange l'effet de *N* et le bruit de tirage. C'est
précisément ce piège qui avait produit l'hypothèse à tester (§2.3).

Ici, **un seul artefact à 800 tirages** (mêmes graines, mêmes époques, mêmes prix
gelés, même boucle — seul `n_samples` change), puis chaque budget *N* est évalué
en découpant chaque nuage en `800 // N` blocs **disjoints** et en moyennant :
16 blocs à N=50, 1 à N=800. Tous les budgets sont lus **sur le même matériel**,
et aucun tirage n'est réutilisé.

Repère de lecture, calculable sans rien simuler : la couverture d'un modèle
*parfait* lue sur *N* tirages vaut 91,27 % à N=50, 94,05 % à 200, 94,76 % à 800.

### 2.2 Par graine, 200 n'est pas convergé

Critère d'arrêt déclaré par le brief : le plus petit *N* dont la couverture **et**
le Winkler sont à moins d'un demi écart-type bootstrap de leur valeur à N=800.

| Régime | W+1 | W+2 | W+3 |
|---|---|---|---|
| weekly | 800 | 800 | 800 |
| daily | **400** | 800 | **400** |

La couverture monte encore de +0,4 à +0,8 point entre 200 et 800. **La réserve
« 200 est un point de fonctionnement raisonné, pas un optimum certifié » est donc
confirmée** pour un run à graine unique.

### 2.3 Mais la configuration production, elle, est convergée

Le balayage par graine ne mesure pas ce qu'on déploie. La configuration
production concatène 5 nuages de 200 en un nuage de **1000**, et lit les
quantiles dessus. Rejoué sur cet objet — les 5 graines **entrelacées**, de sorte
que tout bloc contigu soit un mélange stratifié et non une graine isolée :

| Cellule | plus petit budget convergé | production (1000) convergée ? |
|---|---|---|
| weekly W+1 | 400 | **oui** |
| weekly W+2 | 1000 | **oui** |
| weekly W+3 | 400 | **oui** |
| daily W+1 | 400 | **oui** |
| daily W+2 | 400 | **oui** |
| daily W+3 | 200 | **oui** |

**6 cellules sur 6.** La réserve tombe *pour la configuration de production* —
elle reste vraie pour un run à graine unique, ce qui concerne la piste `oos` du
dashboard (graine 42 × 200). Recommandation : si le dashboard doit rester en
single-seed, monter son budget à **400 minimum** ; s'il bascule sur l'ensemble
5×200, il n'y a rien à faire.

### 2.4 L'hypothèse déclarée est réfutée

`NOTE_nsdiff_consolidation_daily_vs_weekly.md` §1 avançait que le nuage daily,
propagé sur ~5 pas et à queues plus lourdes, convergerait plus lentement — il
gagnait +4,2 points de couverture en passant de 50 à 200 tirages contre +2,4 au
weekly. Mesuré proprement :

| Horizon | gain 50→200 daily / weekly | gain 200→800 daily / weekly | plus petit N convergé |
|---|---|---|---|
| W+1 | +2,03 / +1,57 pts | +0,63 / +0,53 | daily **400**, weekly 800 |
| W+2 | +2,30 / +2,16 pts | +0,71 / +0,79 | daily 800, weekly 800 |
| W+3 | +1,58 / +1,55 pts | +0,38 / +0,56 | daily **400**, weekly 800 |

**Non confirmée aux trois horizons** — et plutôt l'inverse : le daily converge à
400 là où le weekly a besoin de 800. L'écart de +4,2 contre +2,4 qui avait
motivé l'hypothèse comparait deux runs *indépendants* ; il était donc gonflé par
le bruit de tirage que le sous-échantillonnage emboîté élimine. Sur le même
matériel, l'écart tombe à +2,03 contre +1,57.

**Conséquence** : pas de budget différencié par régime. Un seul budget suffit,
et la configuration production le dépasse déjà.

---

## 3. Chantier 1 — l'edge face aux frais réels

### 3.1 La grille de frais, déclarée avant les runs

`experiments/real_fees.py`. Frais **aller-retour** tout compris (commission +
traversée du spread à l'entrée et à la sortie), trois niveaux par instrument.

| Instrument | Véhicule | bas | **central** | haut | sous l'edge de 5 bps ? |
|---|---|---|---|---|---|
| SPY-ES | future E-mini S&P 500 | 1,0 | **1,5** | 2,0 | **oui** |
| ZN-FUT | future 10-Year T-Note | 1,0 | **1,5** | 2,0 | **oui** |
| SPY-ETF | ETF au comptant | 2,0 | **3,5** | 5,0 | **oui** |
| TLT-ETF | ETF au comptant | 2,0 | **3,5** | 5,0 | **oui** |
| BTC-SPOT | crypto au comptant | 10 | **30** | 60 | non |
| ETH-SPOT | crypto au comptant | 10 | **30** | 60 | non |

**4 instruments sur 6 passent sous la borne haute de l'edge mesuré.** Seule la
crypto est exclue — et c'est elle qui condamnait l'exercice au chantier B.

L'exposition actions est évaluée sous **deux véhicules** partageant exactement
les mêmes prévisions (SPY-ETF et SPY-ES) : le même edge, exécuté autrement.

*Statut des chiffres, à citer* : ce sont des **hypothèses déclarées**, aux ordres
de grandeur fournis par le brief, pas un relevé de bordereaux de courtage. Elles
vivent toutes dans un seul fichier ; les remplacer par une grille tarifaire réelle
est un travail de dix minutes qui ne touche que lui. *Réserve supplémentaire* :
pour SPY-ES, la base ES/SPY et le roulement trimestriel ne sont pas modélisés —
simplification **favorable au future**, déclarée.

Le piège sémantique de ce chantier est l'unité : la grille est en aller-retour,
le moteur attend un coût unidirectionnel qu'il double lui-même. Confondre les
deux doublerait toute la grille sans qu'aucun calcul ne tombe. Un test unitaire
vérifie qu'une position pleine à rendement nul coûte **exactement** le frais
aller-retour déclaré, pour les 6 instruments × 3 niveaux.

### 3.2 La matrice edge brut / frais / edge net

Extrait au niveau de coût central, `var_limit`, régime weekly (bps par origine) :

| Instrument | A/R | Cellule | edge brut | surcoût | **EDGE NET** | PnL net NsDiff | PnL net B&H | p vs GARCH |
|---|---|---|---|---|---|---|---|---|
| SPY-ES | 1,5 | W+1 | +6,02 | −0,00 | **+6,03** | +19,31 | +31,58 | 0,070 |
| SPY-ES | 1,5 | W+2 | +11,47 | −0,02 | **+11,49** | +31,67 | +63,38 | 0,017 |
| SPY-ES | 1,5 | W+3 | +16,49 | +0,00 | **+16,48** | +43,90 | +96,46 | 0,022 |
| SPY-ETF | 3,5 | W+3 | +16,49 | +0,01 | **+16,48** | +42,90 | +94,46 | 0,021 |
| ZN-FUT | 1,5 | W+1 | −1,48 | −0,35 | −1,13 | −3,18 | −5,40 | 0,427 |

**La colonne « surcoût » est la conclusion du chantier.** Elle vaut ±0,35 bps là
où l'edge brut vaut ±16 bps : **les frais ne pèsent rien sur l'écart entre les
deux modèles**, parce que les deux bras trament des expositions quasi identiques
et paient donc presque le même montant. `edge_net ≈ edge_brut` partout.

Sur les 72 cellules (hors famille 3) au coût central, **37 ont un edge net
positif**. *À lire avec prudence* : 37 sur 72 est exactement ce qu'un tirage à
pile ou face produit. Ce comptage ne démontre rien — il sert seulement à
constater que le critère d'ouverture du chantier 2 posé par le brief (« au moins
une cellule à edge net positif ») est rempli, et ce critère est faible. Ce qui
justifie réellement de poursuivre est ailleurs : §4.

### 3.3 Ce qui survit à Holm — et ce n'est pas ce qu'on espérait

Familles déclarées avant le run, transposées du chantier B : pour un
(stratégie, comparaison, **instrument**) donné, la famille est ses 6 cellules
(3 horizons × 2 régimes). L'instrument est l'unité de décision — c'est toute la
question du chantier — donc on corrige à l'intérieur de chacun, sans pooler
entre eux.

**54 familles, 30 rejets bruts, 4 survivants.** Et les 4 survivants vont tous
dans le même sens, qui n'est pas celui de NsDiff :

| Famille | Cellule | Verdict | p | p ajustée | écart |
|---|---|---|---|---|---|
| `inverse_width` vs B&H, SPY-ES | W+1 weekly | **buy-and-hold meilleur** | 0,0004 | 0,0024 | −12,81 bps |
| `inverse_width` vs B&H, SPY-ETF | W+1 weekly | **buy-and-hold meilleur** | 0,0004 | 0,0024 | −12,54 bps |
| `var_limit` vs B&H, SPY-ES | W+1 weekly | **buy-and-hold meilleur** | 0,0060 | 0,0360 | −12,27 bps |
| `var_limit` vs B&H, SPY-ETF | W+1 weekly | **buy-and-hold meilleur** | 0,0082 | 0,0492 | −11,90 bps |

**Contre GARCH : zéro survivant.** `var_limit` sur SPY produit 3 rejets bruts
(p = 0,017 / 0,021 / 0,043) mais la plus petite p-value de sa famille de 6 vaut
0,017 contre un seuil de rang 1 à 0,00833 — la procédure s'arrête au premier
rang, comme au chantier B.

Le seul résultat statistiquement solide du chantier est donc : **à W+1 weekly sur
SPY, acheter et garder bat les deux stratégies dérivées des intervalles**, de
~12 bps par origine.

### 3.4 Famille 3 à 80 % — close, définitivement

Le brief déclare le niveau **a priori** : 80 %, et lui seul, pas de balayage —
« si 80 % ne produit pas de signaux exploitables nets de frais, la famille est
close ».

Côté NsDiff, les quantiles 10/90 se lisent sur le nuage production. Côté GARCH,
`tracking.db` ne stocke que la bande à 95 %. Deux constats ont guidé la
reconstruction :

- **la loi d'innovation, établie par mesure** : `arima_model` déclare
  aujourd'hui `GARCH_DIST = "skewt"`, mais les lignes `oos` datent du run
  `20260717`. Les reproduire avec chaque loi candidate départage sans
  ambiguïté — `normal` : **1,45·10⁻⁶** ; `t` : 2,7·10⁻³ ; `ged` : 2,8·10⁻³ ;
  `skewt` : 2,3·10⁻². Le bras GARCH du benchmark est la variante **gaussienne**.
  *(Cela valide a posteriori la reconstruction gaussienne des trajectoires du
  chantier B3 : ce n'était pas une approximation commode mais la loi réellement
  ajustée.)*
- **la bande est exactement log-symétrique** autour du point : écart maximal
  1,8·10⁻¹⁵ sur **100 %** des 2 700 lignes.

La bande à 80 % est donc **dérivée** de la bande publiée —
`lo80 = y_pred·(lo95/y_pred)^r` — et non régénérée. Motif : la régénération
intégrale reproduit la base à 3·10⁻⁷ en médiane mais dérive jusqu'à 1,6·10⁻³ sur
certaines origines, le **point** bougeant aussi — un optimum local différent de
l'optimiseur ARIMA/GARCH sur des fenêtres à volatilité extrême. Dériver depuis la
base garde le bras GARCH bit-pour-bit celui du benchmark.

**Deux garde-fous, et le second a servi.** Le test évident — comparer la bande
dérivée à une bande régénérée — est **aveugle là où l'optimiseur dérive
systématiquement** : il ne couvrait aucune ligne de `TLT|weekly`. Un test immune
à cette dérive a donc été ajouté : mesurer directement le **ratio de niveau** sur
les sorties régénérées, `log(hi80/point) / log(hi95/point)`. Si le fit dérive,
les trois quantités bougent ensemble et le ratio ne bouge pas ; le test couvre
donc 100 % des lignes.

Il a immédiatement attrapé une erreur réelle. Le ratio mesuré déviait de
**1,2·10⁻⁵** de la valeur attendue — parce que la bande à 95 % en base est bâtie
avec `Z_95 = 1.96`, l'arrondi documenté du repo, et non `norm.ppf(0,975) =
1,959964`. Le bon dénominateur est donc `Z_95`, et **r = 0,653853** (au lieu de
0,653865). L'écart déplace la bande de ~10⁻⁶ du prix et ne change aucune
conclusion — mais une dérivation dite exacte qui ne l'est qu'à 10⁻⁵ près n'est
pas exacte, et desserrer la tolérance aurait détruit la capacité du test à
détecter un vrai changement de loi. Corrigé, le ratio est vérifié à
**1,25·10⁻¹⁴** — la précision machine — sur 100 % des lignes, `TLT|weekly`
compris.

**Résultat, aux deux niveaux : `0` position ouverte sur 3 240 origines-instruments.**
Un intervalle à 80 % — largeur médiane 8,0 % du prix — contient encore toujours
le prix courant à ces horizons. La famille est close.

---

## 4. Chantier 2 — de combien d'origines a-t-on besoin ?

Le brief pose la question dans le bon ordre : *« avant de lancer, calculer
l'effective_n nécessaire. Si la cible est inatteignable même avec l'extension,
le dire — c'est un résultat. »*

Méthode sans nouvelle hypothèse : l'erreur-type du test est **déjà** estimée,
blocs compris, par le bootstrap existant, et décroît en 1/√n :
`n_requis = 90 × [SE_observé × (z_{1−α/2} + z_{1−β}) / δ]²`.

### 4.1 La cible déclarée est hors de portée ; les effets observés ne le sont pas

| Cellule (coût central) | edge net | SE | n pour **3 bps** | n pour l'effet observé | n sous Holm |
|---|---|---|---|---|---|
| SPY-ES `var_limit` W+2 weekly | +11,49 | 5,32 | 2 218 | 151 | **233** |
| SPY-ETF `var_limit` W+2 weekly | +11,52 | 5,31 | 2 211 | 150 | **231** |
| SPY-ETF `var_limit` W+3 weekly | +16,48 | 8,19 | 5 259 | 174 | **269** |
| SPY-ES `var_limit` W+3 weekly | +16,48 | 8,21 | 5 286 | 175 | **270** |
| SPY-ES `var_limit` W+3 daily | +11,40 | 6,19 | 3 007 | 208 | **321** |
| BTC-SPOT `inverse_width` W+1 weekly | +20,56 | 15,13 | 17 976 | 383 | 590 |
| ETH-SPOT `inverse_width` W+1 weekly | +30,46 | 108,81 | inatteignable | 9 012 | 13 904 |

**La cible de 3 bps demande 2 200 à 19 000 origines** — contre 601 disponibles
au maximum absolu. Elle est inatteignable, et c'est un résultat : elle vient de
l'edge *poolé entre actifs*, une moyenne qui écrase les écarts de signes opposés.

**Les effets réellement observés par instrument demandent 150 à 270 origines**
sous correction de Holm, pour les meilleures cellules SPY.

### 4.2 Ce que l'extension peut fournir — et ce qu'elle coûte

Comptage réel sur les séries gelées. Avancer la première origine de test prend
les données à l'entraînement : le tableau donne les deux faces.

| Départ | origines de test | effective_n | semaines d'entraînement restantes | cellules détectables sous Holm |
|---|---|---|---|---|
| 2015-01 | 601 | 200 | **0** | 13 |
| 2018-01 | 444 | 148 | 157 | 11 |
| **2020-01** | **340** | **113** | **261** | **6** |
| 2022-01 | 235 | 78 | 366 | 2 |
| 2024-10 (actuel) | 90 | 30 | 511 | 0 |

Un départ en 2015 donnerait 601 origines mais **zéro semaine d'entraînement** :
infaisable. Un départ en 2018 laisse 157 semaines, soit un tiers du budget
actuel — le modèle ne serait plus le même.

**Recommandation : départ en 2020-01.** 340 origines, `effective_n` = 113 (contre
30), et **261 semaines d'entraînement**, soit la moitié de l'actuel — assez pour
que NsDiff reste comparable. Six cellules deviennent détectables sous Holm. Le
backtest traverserait alors 2020 (choc COVID), 2022 (marché baissier de taux) et
2024-2026 : trois régimes au lieu d'un, ce qui est un gain de validité externe
autant que de puissance.

### 4.3 Candidats d'extension du panier

Critères d'inclusion déclarés dans le brief : liquidité, historique ≥ aux actifs
actuels, frais réels ≤ 5 bps aller-retour. Les cinq candidats les remplissent :

| Actif | Exposition | Frais A/R | Corrélé à |
|---|---|---|---|
| **GLD** | or | 4,0 bps | *aucun actif du panier* |
| **USO** | pétrole | 5,0 bps | *aucun actif du panier* |
| QQQ | actions technologiques | 3,5 bps | SPY (~0,9) |
| EFA | actions hors Amérique du Nord | 5,0 bps | SPY (~0,8) |
| IEF | taux 7-10 ans | 4,0 bps | TLT, ZN=F (~0,9) |

**GLD et USO sont les seuls à apporter une exposition réellement nouvelle.** Les
trois autres sont corrélés à 0,8-0,9 avec un actif déjà présent : ils ajoutent des
lignes mais peu d'information indépendante — et le programme dédoublonne déjà
ZN=F et TLT en une seule contribution « taux » précisément pour cette raison.

---

## 5. Non-négociables — statut

- **Multi-graines 42-46** : la configuration production (ensemble) et le bras
  « graine tirée au hasard » (PnL moyenné par origine) sont évalués côte à côte
  dans chaque cellule, et les 5 graines individuelles servent de contrôle.
- **Familles de Holm déclarées avant les runs** ; aucun seuil, marge, niveau ou
  grille choisi après lecture des résultats. Les choix de ce chantier — niveau
  80 %, grille de budgets, grille de frais, critères d'inclusion, cible de 3 bps,
  puissance 80 % — sont ceux du brief ou sont déclarés dans les modules avant
  exécution.
- **Briques réutilisées** : `paired_test`, `mcs.spa_test`, `econ_backtest`
  (moteur du chantier B, inchangé), `multiple_testing`, `nsdiff_production_spec`,
  `dashboard_d7_w1`, `benchmarks.multi_horizon`, `oos_reference_audit`. Code neuf
  uniquement là où rien n'existait, couvert de tests unitaires.
- **`tracking.db` : lecture seule.** Aucune écriture dans ce chantier.
- **Prix gelés partagés** (`diffusion_multiseed_v2/prices/`), aucun appel réseau,
  y compris pour l'artefact 800 tirages et la régénération GARCH.
- **Budget d'échantillonnage strictement égal** entre modèles comparés.
- **pytest vert** : `python -m pytest experiments validation benchmarks models -q`
  → **573 passed, 1 skipped**. Le skip est pré-existant et sans rapport
  (`test_crps_metrics.py:61 — properscoring not installed`).

## 6. Limites déclarées

- **Les frais sont des hypothèses, pas des relevés.** Ordres de grandeur fournis
  par le brief. Toute la grille vit dans `real_fees.py` et se remplace en un
  point. La conclusion §0 — « les frais pèsent 2 % de l'edge différentiel » — est
  robuste à ces valeurs, puisqu'elle repose sur le fait que **les deux bras les
  paient également** ; elle ne dépend donc pas du niveau retenu.
- **SPY-ES ne modélise ni la base ES/SPY ni le roulement.** Simplification
  favorable au future, déclarée. Elle n'affecte pas la conclusion, puisque
  SPY-ES et SPY-ETF donnent le même edge net à 0,1 bps près.
- **Puissance actuelle inchangée** : `effective_n` = 30. Tous les
  « indistinguable » de cette note restent des absences de preuve. C'est
  précisément ce que le chantier 2 quantifie.
- **L'edge par instrument n'est pas corrigé pour la sélection d'instrument.**
  Les familles de Holm corrigent à l'intérieur d'un instrument, pas entre les
  six. Choisir *a posteriori* l'instrument le plus favorable rouvrirait le
  problème de tests multiples d'un cran — et c'est exactement ce qu'un run
  d'extension devrait pré-déclarer.
- **Le critère d'ouverture du chantier 2 est faible.** « Au moins une cellule à
  edge net positif » est rempli par 37 cellules sur 72, soit ce qu'un tirage à
  pile ou face donnerait. Ce qui justifie de poursuivre est l'analyse de
  puissance (§4), pas ce comptage.
- **Non fait, volontairement** : l'extension des origines et l'ajout d'actifs
  ne sont que **chiffrés**, pas exécutés — ils demandent de régénérer la grille
  `oos` complète pour tous les modèles de référence, ce qui est un chantier en
  soi. Aucun balayage de niveau au-delà de 80 % pour la famille 3 (le brief
  l'interdit explicitement, et le résultat à 80 % la ferme).
