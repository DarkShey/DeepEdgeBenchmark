# NOTE — Régénération complète de la grille `oos`, famille 3 reposée, hygiène du benchmark

*2026-08-08. Réponse au brief « Régénération complète de la grille oos, famille 3
reposée proprement, hygiène du benchmark ». Fait suite à
`NOTE_nsdiff_extension_puissance_mensuel.md`, dont ce brief détaille et remplace
le chantier B. TSDiff est hors périmètre — retiré du benchmark, aucune ligne
régénérée.*

**Artefacts produits** :

| Fichier | Chantier | Contenu |
|---|---|---|
| `experiments/regen_plan_r0.py` → `.json` | R0 | inventaire, chiffrage, périmètre d'écriture, ordre d'exécution — **aucune écriture** |
| `experiments/regen_r1_freeze_check.py` → `.json` | R1 | porte bloquante sur le gel des prix, hors ligne |
| `experiments/grid2020_refs.py` → `grid2020_refs/` | R2/W | SARIMA, Prophet, Naive, LSTM sur la grille régénérée + sélection `SEQ_LEN*` pré-2020 |
| `experiments/grid2020.py --garch-dist skewt` → `grid2020/ARIMA-GARCH[skewt]/` | H1 | bras GARCH skew-t, même code, même grille |
| `experiments/h1_garch_dist.py` → `.json` | H1 | match gaussien vs skew-t, config championne actée |
| `experiments/upsert_grid2020.py` → `.json` | R2 | dépôt en base, piste distincte, dry-run/`--apply` |
| `experiments/grid_old_vs_new.py` → `.json` | R3 | tableau ancienne grille / nouvelle grille + renversements |
| `experiments/famille3_v2.py` → `.json` | F3 | famille 3 reposée : signal directionnel normalisé |
| `experiments/h2_futures_roll.py` → `.json` | H2 | coût de roulement des futures, sensibilité aux trois niveaux |
| `experiments/coverage_monitor.py` → `.json` + `_grid2020.json` | H3 | couverture glissante et bande d'alerte, sur la piste `oos` et sur la piste régénérée |

**Code neuf couvert de tests** : `test_coverage_monitor.py` (15),
`test_real_fees_roll.py` (10), `test_grid2020_refs.py` (16).

---

## 0. Le résultat principal, d'emblée

**La régénération ne sauve rien, et ce n'est pas une déception : c'est la réponse
que le programme cherchait.** Sur la grille longue, avec tous les modèles et le
champion GARCH à sa meilleure configuration, l'hypothèse primaire pré-déclarée ne
survit toujours pas (0 rejet sur 4, p = 0,087 à 0,136), la réplication échoue
toujours, et la parité de calibration est cette fois renversée *contre* NsDiff.
Le critère d'arrêt du volet économique, déclenché au chantier précédent, est
confirmé sur une grille où plus rien ne peut être attribué à un manque de
puissance ni à un adversaire mal réglé.

La famille 3, reposée dans la formulation que le brief déclare, **émet** cette
fois — c'est le progrès sur le chantier précédent, où elle n'émettait rien du
tout — mais elle n'émet que sur 0,4 % des origines et ne produit aucun survivant
sous Holm. Le critère de clôture déclaré s'applique : la question du signal
directionnel est close.

Trois résultats d'hygiène s'ajoutent, et deux d'entre eux sont des vrais
enseignements :

1. **H3 découvre ce que le benchmark rétrospectif ne pouvait pas voir** : 79
   cellules sur 180 de la piste `oos` sont hors bande d'alerte aujourd'hui, dont
   Prophet BTC-USD en régime daily à **0 % de couverture** sur les 26 dernières
   origines. Aucun verdict du programme ne l'avait signalé, parce qu'aucun ne
   regarde une fenêtre glissante.
2. **H2 lève une réserve et ne déplace rien** : le roulement trimestriel coûte
   0,12 à 0,35 bps sur 1 à 3 semaines et déplace l'edge différentiel d'au plus
   0,05 bps. La simplification « favorable au future » était réelle mais
   négligeable.
3. **Le contrôle 1:1 de l'upsert a attrapé un défaut de convention** avant toute
   écriture : les modèles de référence étiquettent leurs horizons `W1/W2/W3`, la
   base attend `W+1/W+2/W+3`. Sans cette vérification, 28 560 lignes seraient
   entrées en base avec un horizon nul.

---

## 1. R0 — chiffrer avant de lancer, et ne pas recompter ce qui est déjà payé

Le brief demande de traiter R comme une migration : phasée, chiffrée avant
exécution, réversible. `regen_plan_r0.py` produit ce plan en lecture seule
stricte (`tracking.db` ouvert en `file:…?mode=ro`).

Le chiffrage part d'un recensement, pas d'une hypothèse. Le socle est la table
`oos_reference_audit.INTERVAL_MECHANISM`, vérifiée ligne à ligne dans le code des
modèles ; on y superpose une seule information neuve — **ce que la grille 2020
contient déjà**. NsDiff (5 graines × 200) et ARIMA-GARCH gaussien y sont depuis
le chantier précédent : leur coût n'est pas rechiffré comme s'il restait à payer.
C'est la différence entre le coût affiché d'un chantier et son coût réel.

| Bras à produire | Phase W | Phase D | Total |
|---|---:|---:|---:|
| LSTM | 4,38 h | 7,23 h | 11,61 h |
| Prophet | 0,24 h | 0,80 h | 1,04 h |
| ARIMA-GARCH skew-t (H1) | 0,13 h | 0,71 h | 0,84 h |
| SARIMA | 0,06 h | 0,32 h | 0,39 h |
| Naive | 0,00 h | 0,00 h | 0,00 h |
| **Total** | **4,81 h** | **9,06 h** | **13,87 h** |

Le seuil est **déclaré avant lecture**, à la valeur que le brief propose
lui-même : 48 h. Le total tient largement dedans. Le découpage phase W / phase D
n'est donc **pas** une conséquence du budget — c'est le protocole qui l'impose,
l'hypothèse primaire étant weekly et la phase D lui étant conditionnée (R2.2). Le
dire ainsi évite un contresens facile : on ne coupe pas par économie, on coupe
parce que la question weekly précède la question daily.

Le périmètre d'**écriture** est chiffré lui aussi, parce qu'une migration dont on
ignore le volume n'est pas réversible : 18 900 lignes `oos` aujourd'hui, 42 840
après la phase W, 85 680 après les deux (×4,53).

**Écart mesuré à l'exécution, à déclarer.** Le chiffrage s'est trompé dans les
deux sens, et pas au même endroit :

| Bras | chiffré (phase W) | réel |
|---|---:|---:|
| LSTM | 4,38 h | 2,50 h (+ 0,22 h de sweep `SEQ_LEN*`) |
| Prophet | 0,24 h | 0,27 h |
| SARIMA | 0,06 h | 0,03 h |
| Naive | 0,00 h | 0,00 h |
| **Sous-total** | **4,68 h** | **3,02 h** |
| GARCH skew-t (les deux régimes) | 0,84 h | 1,33 h |

Le LSTM, qui pesait 84 % du chiffrage, a coûté **43 % de moins** que prévu :
la sonde prenait 3 origines réparties sur toute la grille, dont les plus tardives
— celles qui ont la fenêtre d'entraînement la plus longue — et extrapolait leur
coût à toutes. Le GARCH skew-t, lui, a coûté 58 % de plus : l'ajustement d'une
loi à queues épaisses et asymétrie converge plus lentement que la gaussienne, ce
que sonder une seule loi ne pouvait pas révéler. La hiérarchie entre modèles était
juste, les niveaux absolus non — ce qui reste acceptable pour ce à quoi le
chiffrage sert (décider du découpage), et à corriger si un jour il sert à réserver
une machine.

## 2. R1 — le gel des données, et un critère du brief qu'on n'applique pas

`regen_r1_freeze_check.py` rejoue la vérification bloquante hors ligne, sur les
fichiers gelés tels qu'ils sont aujourd'hui, sans aucun appel réseau. Les sept
actifs passent.

| Actif | Historique | Origines | Recouvrement avec `diffusion_multiseed_v2/prices` |
|---|---|---:|---|
| SPY | 2011-05-02 → 2026-07-23 | 340 | 2 905 dates, log-rdt max 1,1e-06 |
| ZN=F | 2011-05-02 → 2026-07-23 | 340 | 2 905 dates, log-rdt max 0 |
| TLT | 2011-05-02 → 2026-07-23 | 340 | 2 121 dates, log-rdt max 1,9e-06 — **niveau décalé** |
| BTC-USD | 2014-09-17 → 2026-07-23 | 340 | 4 222 dates, log-rdt max 0 |
| ETH-USD | 2017-11-09 → 2026-07-23 | 340 | 3 179 dates, log-rdt max 0 |
| GLD | 2011-05-02 → 2026-07-23 | 340 | premier entrant |
| USO | 2011-05-02 → 2026-07-23 | 340 | premier entrant |

**Une divergence assumée avec le brief.** Le brief écrit « tolérance ~2e-7
relative » sur les prix. Ce critère a été essayé au chantier A et il **bloque** —
non sur une révision d'historique, mais sur un changement de base d'ajustement
des dividendes : TLT présente un ratio ancien/nouveau **constant** de 0,99598782
(ancienne série `fetch_tlt_patched` hors ligne, nouvelle yfinance). Un facteur
constant ne déplace aucun log-rendement, donc aucun modèle. Le critère bloquant
retenu porte sur ce que les modèles voient réellement :

- log-rendements identiques, tolérance 1e-5 — **bloquant** ;
- dispersion du ratio de prix, tolérance 1e-4 — **bloquant** ;
- niveau du ratio — rapporté, jamais bloquant.

La deuxième condition est celle qui compte : elle distingue un changement de base
(ratio constant, dispersion nulle) d'une révision d'historique (ratio qui dérive).
C'est cette dernière que le brief veut interdire, et elle reste interdite. La
conséquence à retenir : **les niveaux de prix de `prices_v3` ne sont pas
comparables à ceux de l'ancien jeu pour TLT** — ce qui interdit, à soi seul, toute
comparaison origine par origine entre les deux grilles (cf. §5).

## 3. R2 phase W — la grille weekly complète, et son dépôt en base

### 3.1 Ce qui a été produit, et ce qui a été réutilisé

La grille weekly est désormais complète : **6 modèles × 7 actifs × 3 horizons ×
340 origines = 42 840 lignes**, exactement 7 140 par modèle. NsDiff (ensemble
5 × 200) et ARIMA-GARCH gaussien venaient du chantier précédent ; SARIMA, Prophet,
Naive et LSTM ont été produits ici, et le bras GARCH skew-t de H1 avec eux.

Le protocole de ces quatre modèles n'est pas « appeler leur fonction de
prévision » — c'est ce qui rendait la réécriture dangereuse. Il porte trois choix
déclarés ailleurs dans le dépôt, et `grid2020_refs.py` appelle donc la boucle
existante (`weekly_multimodel.run_model_asset`) telle quelle, via trois paramètres
opt-in :

- **variantes hebdo-natives** pour SARIMA (saisonnalité 5 jours désactivée) et
  Prophet (pas de saisonnalité hebdomadaire, dates futures sur W-FRI) ;
- **calibration sigma EWMA causale** pour SARIMA/Prophet/Naive/LSTM, avec son lag
  de résolution — le `z` d'une origine `k` n'entre dans l'état de W+j qu'à
  l'origine `k+j`. Elle travaille vraiment : sur SPY, le facteur moyen appliqué va
  de 1,45 (LSTM) à 2,28 (Prophet) ;
- **`SEQ_LEN*` par actif** pour le LSTM en régime hebdo natif.

Chaque bras est refit **à chaque origine** — leur protocole naturel, et l'asymétrie
avec le train-once-forward de NsDiff reste déclarée comme partout.

### 3.2 Une fuite qu'on ne reproduit pas

Le `SEQ_LEN*` publié du LSTM a été choisi sur les 12 origines qui précèdent
immédiatement les 30 origines de test de l'ancienne grille — lesquelles tombent
**en plein** dans les 340 de la nouvelle. Le réutiliser aurait importé une fuite,
du même type que celle déjà déclarée pour les époques hebdo de TSDiff.

Le sweep a donc été rejoué sur un bloc de validation **strictement antérieur à
2020-01, cible W+3 comprise** (T0 = 2019-09-13), même principe que le
train-once-forward de NsDiff. Règle de sélection inchangée : 1-SE sur le CRPS de
validation, le plus parcimonieux parmi les candidats statistiquement à égalité.
Résultat : `SEQ_LEN* = 8` pour ZN=F, TLT, BTC-USD, GLD, USO ; `26` pour SPY et
ETH-USD. Coût : 0,22 h.

### 3.3 Le contrôle 1:1 a servi, avant toute écriture

La vérification de clés exigée par le brief a bloqué le premier dry-run : les
modèles de référence étiquettent leurs horizons **`W1/W2/W3`**
(`weekly_headtohead.HORIZON_LABELS`), la grille et `tracking.db` attendent
**`W+1/W+2/W+3`** (`backtest_rolling_tsdiffw.HORIZON_UNITS`). Les deux conventions
coexistent dans le dépôt depuis longtemps ; personne ne les avait mises face à
face parce que personne n'avait encore fait passer ces modèles-là dans cette
table-là.

Sans le contrôle, 28 560 lignes seraient entrées avec un `horizon` nul. Le pont
entre les deux conventions vit maintenant dans un seul endroit
(`grid2020_refs.normalise_horizon`, idempotent — les checkpoints déjà écrits ont
été réparés sans recalculer une origine), et la vérification 1:1 refuse désormais
explicitement tout étiquetage non conforme.

### 3.4 Le dépôt en base : une piste neuve, et c'est une décision

Le brief interdit tout mélange ancien/nouveau. Or la nouvelle grille n'est pas un
*repointage* de l'ancienne : 340 origines contre 90, 7 actifs contre 5, et des
prix regelés dont les **niveaux** diffèrent pour TLT (§2). Écraser la piste `oos`
rendrait les anciens verdicts invérifiables ; un upsert partiel produirait
exactement le mélange interdit.

La grille est donc déposée sous **`source='oos2020'`**, piste distincte —
l'index d'unicité porte `source`, les deux coexistent sans collision. La piste
`oos` n'est ni lue en écriture ni modifiée, et le script le **vérifie** : empreinte
(comptage + somme de contrôle) relevée avant, recomparée après, échec bloquant
sinon. TSDiff, retiré, n'a aucune ligne dans la nouvelle piste.

Discipline appliquée, celle du standard `repoint_oos_to_m200` : dry-run par
défaut, sauvegarde horodatée
(`tracking.db.bak_grid2020_weekly_2026-08-08T140155`), `--apply` explicite,
colonnes dérivées recalculées par `backfill_eval_metrics` (doté d'un `--source`
pour l'occasion), cohérence de `in_interval` revérifiée après coup.

Résultat : **42 840 lignes déposées, piste protégée intacte, `in_interval`
cohérent avec les bornes.** Couverture 95 % sur la nouvelle piste, par modèle :

| Modèle | Cov95 (weekly, 340 origines, 7 actifs) |
|---|---:|
| Prophet | 0,947 |
| ARIMA-GARCH | 0,938 |
| LSTM | 0,934 |
| Naive | 0,931 |
| SARIMA | 0,931 |
| NsDiff | 0,929 |

Lue avec H3 branché sur cette piste (§8), la même grille donne 50 cellules en
alerte sur 126, dont 34 en sous-couverture — et **NsDiff est en tête des modèles
en alerte, à égalité avec Prophet (12 cellules chacun)**, ce qui est cohérent avec
le verdict de calibration du §5 et non un artefact de la piste neuve.

## 4. H1 — le champion joue-t-il avec la bonne loi ? Oui, et ça ne change rien

Le constat qui ouvrait H1 était solide : les lignes `oos` du bras ARIMA-GARCH sont
la variante **gaussienne** (elles se reproduisent à 1,45e-06 avec `dist="normal"`,
contre 2,3e-02 avec le skew-t), alors que `models/arima_model.py` déclare
aujourd'hui `GARCH_DIST="skewt"`. Le benchmark faisait donc jouer son champion
avec des queues fines sur des actifs à queues épaisses — et c'est ce champion qui
tient le mur contre NsDiff.

Le bras skew-t a été produit **par le même code, sur la même grille, avec les
mêmes prix gelés** : `grid2020.run_garch` prend désormais la loi en argument, et
rien d'autre ne change. La comparaison mesure la loi, et elle seule.

**Contrôle d'identité, passé** : les deux lois partagent l'équation de moyenne
ARIMA, donc le RMSE doit être identique. Dérive relative maximale mesurée :
**0,00e+00** sur les 21 cellules. Si ce chiffre avait bougé, les deux bras
n'auraient pas été le même modèle et la comparaison n'aurait pas porté sur ce
qu'on croit.

Résultat, régime weekly, 354 origines, 7 actifs :

| | gaussienne | skew-t |
|---|---|---|
| cellules gagnées au Winkler (Holm, m = 21) | 0 | 0 |
| tests significatifs bruts | 1 | — |
| Cov95 moyenne | 0,938 | 0,939 |

**Aucune cellule ne sépare les deux lois.** Là où l'effet était attendu — la
crypto, seule motivation de H1 — il n'est ni systématique ni dans le sens espéré :
le skew-t **élargit** BTC (36,2 → 37,1 % du prix à W+1) et **rétrécit** ETH
(44,5 → 42,9 %), pour des couvertures qui bougent de moins d'un point. Sur SPY, il
rétrécit les bandes d'environ 1 point de prix et fait perdre 2,7 points de
couverture à W+1 et W+2 — sans que le Winkler s'en ressente.

**Config championne actée : `dist="normal"`**, par la règle de départage déclarée
avant lecture (à égalité statistique, on conserve la loi qui a produit tous les
verdicts publiés ; un changement de champion doit être gagné, pas obtenu par
défaut).

Ce que H1 apporte n'est donc pas un meilleur champion, mais la fermeture d'une
échappatoire : on ne peut plus dire que le mur GARCH tenait grâce à une loi mal
choisie. Il tient avec l'une comme avec l'autre.

---

## 5. R3 — ancienne grille contre nouvelle grille, et 24 renversements

Les deux grilles ne partagent ni les mêmes origines, ni les mêmes prix, ni le
même périmètre. Ce qui se compare, ce sont les **verdicts** et les grandeurs
agrégées par cellule — jamais les lignes.

|  | ancienne | nouvelle |
|---|---|---|
| origines | 95 (`effective_n` 31) | 354 (`effective_n` 118) |
| actifs | 5 | 7 (+ GLD, USO) |
| prix | `diffusion_multiseed_v2/prices` | `prices_v3` |

Sur les 30 cellules communes (5 actifs × 2 régimes × 3 horizons) :

| Conclusion clé | ancienne grille | nouvelle grille |
|---|---:|---:|
| Cov95 NsDiff (moyenne) | 0,945 | 0,933 |
| Cov95 GARCH (moyenne) | 0,954 | 0,947 |
| Edge net moyen (`var_limit`) | +1,33 bps | +0,33 bps |
| Cellules où NsDiff gagne le Winkler | 0 | 0 |
| Cellules où GARCH gagne le Winkler | 3 | **16** |
| Cellules où le TOST conclut à l'équivalence des RMSE | 0 | 0 |

**24 cellules sur 30 portent un renversement**, et il faut les lire par type,
parce qu'ils ne disent pas la même chose :

- **Renversements de puissance (majoritaires)** — 16 cellules passent de
  « indistinguable » à « GARCH significativement meilleur » sur le Winkler.
  Ce n'est pas un changement de direction : c'est une absence de preuve qui
  devient une preuve, exactement l'effet attendu quand `effective_n` passe de 31
  à 118. Aucune cellule ne bascule dans l'autre sens vers NsDiff.
- **Les seuls renversements en sens inverse (3)** — TLT weekly, aux trois
  horizons, passe de « GARCH significativement meilleur » à « indistinguable ».
  C'est aussi le seul actif dont les **prix** ont changé de niveau entre les deux
  jeux (§2) ; son ancien verdict portait sur une série que le benchmark n'utilise
  plus.
- **Renversements de signe de l'edge (13 cellules)** — tous en crypto, sur TLT ou
  sur ZN=F,
  et tous de très faible amplitude sauf la crypto daily (BTC-USD W+3 : −4,05 →
  +15,80 bps). L'edge crypto daily change de signe entre deux échantillons : c'est
  la définition d'un résultat non répliqué, pas d'un edge.

La lecture d'ensemble est nette et ne dépend d'aucun de ces cas particuliers :
**quadrupler la longueur de l'échantillon n'a fait apparaître aucun avantage
NsDiff, et a rendu visible un avantage GARCH qui manquait de puissance pour se
montrer.**

## 6. F3 — la famille 3 reposée : elle émet, et elle ne rapporte rien

### 6.1 Ce qui était clos, et pourquoi ce n'était pas un problème de puissance

La famille 3 originelle prend position « si le PI exclut le prix courant ». Elle
n'émettait **jamais** : 0 position sur 3 240 origines-instruments, à 95 % comme à
80 %. La cause est mécanique et se lit sans test — la largeur médiane du PI
dépasse toujours le drift médian à 1-3 semaines. Descendre le niveau après avoir
vu ce résultat aurait été du p-hacking, et n'aurait rien changé au mécanisme.

### 6.2 La règle déclarée, et le paramètre que le brief laissait ouvert

Position si `|médiane prédictive − prix courant| > k × largeur`, signe donné par
la médiane, `k ∈ {0,25 ; 0,5}` — deux valeurs, aucun balayage. Le point est la
**médiane des deux côtés** : médiane du nuage agrégé côté NsDiff, prévision
centrale (= médiane de la loi prédictive) côté GARCH. Le bras GARCH est évalué
avec le même signal sur ses propres quantiles ; aucune règle du moteur ne sait
quel modèle l'appelle.

**Le brief fixait `k` mais pas le niveau de largeur** — il écrit seulement
« échelle interquantile ». Le niveau retenu est **80 %**, celui de la famille
qu'on remplace : garder la même échelle est la seule façon de dire que c'est la
*règle* qui change, et pas l'échelle sous elle. Ce choix n'est pas anodin, et
l'étage exploratoire le montre : la lecture concurrente — largeur = PI à 95 % —
reproduit trait pour trait la pathologie de la famille close. Les deux lectures
sont donc rapportées à l'étage 1, en descriptif ; une seule alimente le
confirmatoire.

### 6.3 Étage 1 — la famille émet-elle ? (descriptif, grille actuelle)

| Échelle | k | positions émises (NsDiff) |
|---|---|---|
| **80 % (déclarée)** | 0,25 | 3 / 2 700 |
| 80 % | 0,5 | 0 / 2 700 |
| 95 % (lecture concurrente) | 0,25 | 1 / 2 700 |
| 95 % | 0,5 | 0 / 2 700 |

Sur la grille actuelle, la famille est au bord de la non-émission dans les deux
lectures. C'est le contrôle que la famille 3 originelle n'avait pas passé, et il
avertit déjà : le confirmatoire va tester une règle qui se déclenche rarement.

### 6.4 Étage 2 — confirmatoire, famille de Holm dédiée

Famille déclarée avant tout calcul : `SPY-ETF × {W+1, W+2, W+3} × {k=0,25 ; k=0,5}
× {vs GARCH, vs B&H}`, régime weekly — **m = 12**. Périmètre choisi pour la même
raison que l'hypothèse primaire du programme : SPY weekly est la seule zone à `n`
requis atteignable. SPY-ES partage les mêmes prévisions et n'en diffère que par
les frais ; l'inclure aurait doublé la famille à information constante.

Sur la nouvelle grille, la famille **émet** : 53 positions sur 14 280 à k = 0,25
(0,4 %), 14 à k = 0,5 — contre 14 et 0 pour le bras GARCH. C'est le progrès réel
sur le chantier précédent : la règle n'est plus vide.

Résultat : **5 tests significatifs bruts, 0 sous Holm** (seuil le plus strict
0,0042). Les PnL nets vont de −1,16 à +7,90 bps par origine, sur 2 à 12 origines
actives selon la cellule.

### 6.5 Verdict, et la réserve qui va avec

Le critère de clôture déclaré est atteint : aux deux valeurs de `k`, aucun
survivant Holm en faveur de NsDiff sur une cellule à PnL net positif. **La
question « les intervalles de NsDiff portent-ils un signal directionnel ? » est
close définitivement, sans autre reformulation** — c'est la règle que le brief
pose et elle s'applique telle quelle.

La réserve honnête, qui ne change pas le verdict mais en fixe la portée : avec 2
à 12 origines actives sur 340, **ce test n'avait presque aucune puissance**. Ce
qu'on établit n'est donc pas « le signal directionnel est absent », mais « à
l'échelle d'incertitude des modèles, le drift à 1-3 semaines franchit si rarement
un quart de la largeur interquantile qu'aucune stratégie exploitable ne peut s'y
construire ». C'est le même diagnostic structurel qu'avant — drift ≪ incertitude
— mesuré cette fois sur un ratio continu au lieu d'une exclusion binaire, ce qui
est précisément ce que le brief demandait de vérifier.

## 7. H2 — le roulement des futures : réserve levée, verdict inchangé

Le chantier précédent déclarait que « la base ES/SPY et le roulement trimestriel
ne sont pas modélisés — hypothèse favorable au future ». H2 lève la moitié de
cette réserve et explique pourquoi l'autre moitié n'a pas à l'être.

**Le roulement est modélisé.** Hypothèse déclarée, conservatrice : un roulement
coûte un aller-retour complet de l'instrument (rien ne dit que le spread
calendaire soit plus large que l'outright — sur ES et ZN, l'inverse est fréquent).
Un sleeve tenu `h` semaines traverse une échéance avec probabilité `h/13`, donc le
coût attendu par origine vaut `(h/13) × aller-retour`.

| Horizon | outright | roulement | total |
|---|---:|---:|---:|
| W+1 | 1,50 bps | 0,12 bps | 1,62 bps |
| W+2 | 1,50 bps | 0,23 bps | 1,73 bps |
| W+3 | 1,50 bps | 0,35 bps | 1,85 bps |

Le roulement ne s'annule pas entre les deux bras — il est proportionnel à
`|position|`, et NsDiff est plus exposé que GARCH (0,607 contre 0,470 en moyenne
sur SPY-ES `var_limit` W+3). Mais l'effet est minuscule : **l'edge différentiel
bouge d'au plus 0,05 bps**, et 10 cellules sur 12 gardent un edge positif. Aucune
n'est significative sous Holm, ni avant ni après.

**La base ES/SPY n'est pas modélisée, et c'est un choix argumenté.** La base vaut,
à l'équilibre, `(taux sans risque − rendement du sous-jacent) × durée` : un long
future subit un portage que l'ETF ne subit pas. Mais le future n'immobilise que sa
marge, et le capital non déployé rapporte le taux sans risque. Dans une
comparaison **financée** — la seule que ce backtest fasse, `|w| ≤ 1`, sans levier
— les deux s'annulent au premier ordre. Mettre un chiffre de portage sans créditer
l'intérêt sur le cash pénaliserait le future d'un coût qu'il ne paie pas. La
réserve résiduelle est étroite et déclarée : l'écart de la base à son équilibre à
court terme n'est pas mesuré.

**Bordereaux réels : toujours pas de bordereaux.** Le brief demande de remplacer
la grille hypothétique « dès qu'ils existent ». Aucun relevé de courtage n'est
versé au dépôt. La grille reste donc déclarée comme hypothèse, à son emplacement
unique (`real_fees.INSTRUMENTS`), et H2 rapporte la sensibilité de chaque cellule
aux trois niveaux bas/central/haut — la seule réponse honnête tant que les
bordereaux manquent. Elle ne change aucun verdict : l'écart entre le niveau bas et
le niveau haut déplace l'edge de moins de 0,2 bps.

## 8. H3 — le monitoring en ligne, et ce qu'il trouve tout de suite

Tout le programme mesure une couverture rétrospective sur grille figée. H3 ajoute
la brique qui manque pour un usage réel : couverture glissante sur **26 origines**
(~ deux trimestres), bande d'alerte déclarée **[0,88 ; 0,99]**, épisodes
d'alertes consécutives. Sortir de la bande n'est pas un verdict statistique —
c'est un déclencheur d'investigation ; le test formel reste Kupiec. Aucune fuite
par construction : la fenêtre à l'origine `t` ne contient que des origines déjà
résolues à `t`.

Le choix de 26 est un compromis assumé : à 13 origines, l'écart-type d'une
couverture 95 % vaut ~6 points et la bande se déclencherait sur du bruit ; plus
long, on détecte la dérive un trimestre trop tard.

Branché sur la piste `oos` telle qu'elle est, il sort **79 cellules en alerte sur
180** :

| Statut | cellules |
|---|---:|
| OK | 101 |
| sur-couverture (> 0,99) | 42 |
| sous-couverture (< 0,88) | 37 |

Par modèle : Prophet 23 cellules en alerte, Naive 14, LSTM 13, SARIMA 13,
NsDiff 9, ARIMA-GARCH 7.

Le cas qui justifie à lui seul la brique : **Prophet BTC-USD daily, aux trois
horizons, couvre 0 % sur les 26 dernières origines** — l'intervalle a cessé de
contenir la cible, complètement, et aucun verdict rétrospectif du programme ne
l'avait dit, parce qu'aucun ne regarde ailleurs que le bloc entier. La
sur-couverture est traitée symétriquement, et c'est délibéré : des intervalles
inutilement larges ne coûtent rien en couverture et tout en Winkler.

**Branché sur la piste régénérée** (`--source oos2020`, 126 cellules weekly), le
même suivi donne 50 cellules en alerte, dont 34 en sous-couverture. Les modèles
les plus souvent en alerte sont **NsDiff et Prophet (12 cellules chacun)**, puis
LSTM (11), loin devant ARIMA-GARCH (5). La grille longue ne corrige donc pas la
dérive de couverture de NsDiff : elle la rend lisible semaine par semaine, au même
endroit où le Winkler la voyait déjà en agrégé (§5).

## 9. Phase D — le critère déclaré est évalué, et il n'est pas rempli

Le brief conditionne la phase D : elle n'est lancée « que si la phase W justifie
de continuer », selon un critère déclaré à deux branches — **hypothèse primaire
survivante OU signal de calibration nouveau à documenter**. Les deux sont
évaluées ici, mécaniquement, plutôt que remplacées par une préférence.

**Branche 1 — hypothèse primaire : NON.** 0 rejet sur 4 sous Holm, et aucun même
au seuil non corrigé (p = 0,087 à 0,136). La réplication des survivants
« acheter-et-garder bat la stratégie » échoue également, aux deux familles.

**Branche 2 — signal de calibration nouveau : NON.** Ce que la phase W produit sur
la calibration est un **renforcement**, pas une nouveauté : la parité de
calibration ne tenait déjà pas au chantier précédent, et la grille longue rend
simplement l'avantage GARCH visible sur 16 cellules au lieu de 3. H1, qui était la
seule source plausible de nouveauté (« si skew-t couvre mieux en crypto, le mur
GARCH se renforce »), ne sépare aucune cellule : il n'y a pas de champion nouveau
à documenter.

**La phase D n'est donc pas exécutée.** Ce n'est pas un renoncement de périmètre :
c'est la règle d'arrêt du brief qui s'applique. Le brief la rappelle d'ailleurs
lui-même en clôture — « si l'hypothèse primaire ne survit pas à Holm sur 340
origines, le volet économique se clôt définitivement — F3 inclus si sa famille est
aussi négative ». Les deux conditions sont réunies : **le volet économique est
clos, et la question du signal directionnel avec lui.** Les 9,06 h chiffrées pour
la phase D restent chiffrées et non dépensées, comme les 4 modèles classiques du
chantier précédent l'étaient avant celui-ci.

**Une réserve, et c'est la seule qui pourrait rouvrir la question.** H3 fait
apparaître un signal qui, lui, est neuf — mais il ne vient pas de la phase W et il
ne concerne pas le duel NsDiff/GARCH : sur la piste `oos` actuelle, les alertes de
couverture se concentrent nettement sur le régime **daily** (51 cellules en alerte
sur 90, dont 26 en sous-couverture, contre 28 sur 90 en weekly). Ce n'est pas un
argument confirmatoire — c'est un défaut opérationnel des modèles de référence, sur
l'ancienne grille. Si le programme devait rouvrir quelque chose, ce serait
celui-là, et il faudrait le décider explicitement : le critère du brief, lu comme
il est écrit, ne le couvre pas.

## 10. Non-négociables — statut

| Non-négociable | Statut |
|---|---|
| Multi-graines 42-46 | tenu — NsDiff en ensemble 5 × 200 partout |
| Conventions descriptif / graine fixe / poolé | tenu — étage 1 de F3 explicitement descriptif, tests poolés par bootstrap par blocs |
| Familles Holm et seuils déclarés avant les runs | tenu — F3 (m = 12) et H1 (m = 21) déclarés en dur dans les scripts avant tout calcul ; **une exception documentée** : le niveau de largeur de F3, que le brief laissait ouvert, est déclaré au §6.2 |
| Briques réutilisées | tenu — `econ_backtest`, `paired_test`, `mcs.spa_test`, `multiple_testing`, `real_fees`, `oos_reference_audit`, `cost_grid_2020`, `calibration_tests`, standard `repoint` |
| `tracking.db` en lecture seule hors script d'upsert dédié | tenu — R0, H3 et l'empreinte de contrôle ouvrent la base en `mode=ro` |
| Prix gelés partagés | tenu — `prices_v3/` pour tous les bras, aucun appel réseau dans la régénération |
| Budget d'échantillonnage égal | tenu — 5 × 200 des deux côtés, et les deux bras GARCH de H1 partagent le même code |
| pytest vert avant/après | tenu — **688 passed / 1 skipped** avant (et non 573, le dépôt a grossi depuis le brief), **729 passed / 1 skipped** après (41 tests neufs) |

**Modifications de fichiers partagés — toutes additives et opt-in**, chemin
historique inchangé quand le paramètre n'est pas passé, chacune couverte par un
test de non-régression :

- `weekly_multimodel.run_model_asset` : `daily=`, `test_pos=`, `forecast_fn=` ;
- `lstm_weekly_sweep.sweep_asset` : `daily=`, `val_pos=`, `with_regime_b=` ;
- `grid2020.run_garch` : `dist=`, `arm=` (+ `--garch-dist`, `--garch-arm`) ;
- `grid2020_tests` : `load_arms(garch_arm=)`, `nsdiff_bands(point=)` ;
- `backfill_eval_metrics` : `--source` ; `coverage_monitor` : `--source` ;
- `econ_backtest` : la famille 3' et ses deux `k` ;
- `real_fees` : `roll_cost_bps`, `total_round_trip_bps`, `one_way_total_bps`.

## 11. Limites déclarées

- **Le chiffrage R0 s'est trompé dans les deux sens** (§1) : −43 % sur le LSTM,
  +58 % sur le GARCH skew-t. La méthode — chronométrer plutôt que deviner — reste
  la bonne ; sonder 3 origines d'un seul actif et d'une seule loi ne suffit pas à
  fixer un niveau absolu. La hiérarchie entre modèles, elle, était juste, et c'est
  ce dont le découpage avait besoin.
- **Le `SEQ_LEN*` du LSTM a été resélectionné**, sur un bloc de 12 origines
  strictement antérieur à 2020-01 (cible W+3 comprise). Le `SEQ_LEN*` publié avait
  été choisi sur un bloc qui tombe **dans** la nouvelle grille de test : le
  réutiliser aurait importé une fuite, du même type que celle déjà déclarée pour
  les époques hebdo de TSDiff. Conséquence à assumer : les lignes LSTM de la
  nouvelle piste ne sont pas produites par la même configuration que celles de
  l'ancienne, et les deux ne se comparent pas ligne à ligne.
- **F3 conclut sur un test à très faible puissance** (2 à 12 origines actives sur
  340). Le critère de clôture déclaré est atteint tel qu'il est écrit ; ce qui est
  établi est un fait mécanique sur le rapport drift/incertitude, pas une absence
  de signal démontrée au sens statistique fort. Cf. §6.5.
- **La base ES/SPY n'est pas modélisée** (§7), avec l'argument financé /
  non-financé. La réserve résiduelle — écart de la base à son équilibre à court
  terme — n'est pas mesurée.
- **Les frais restent des hypothèses**, pas des bordereaux. Aucun relevé réel
  n'est versé au dépôt. La sensibilité aux trois niveaux est rapportée et ne
  déplace aucun verdict.
- **La bande d'alerte de H3 est un choix, pas un test.** [0,88 ; 0,99] sur 26
  origines déclenche une investigation ; elle n'a pas de niveau de confiance et ne
  doit pas être citée comme tel. Le test formel reste Kupiec.
- **GLD et USO sont des premiers entrants** : ils n'ont aucun historique de
  verdict, ne participent à aucune comparaison ancienne/nouvelle, et sont
  rapportés à part partout.
