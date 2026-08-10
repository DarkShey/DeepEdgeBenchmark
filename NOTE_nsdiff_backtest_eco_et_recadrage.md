# NOTE — NsDiff : valeur économique, re-cadrage 200 tirages, cadrage monthly

*2026-08-07. Réponse au brief « NsDiff : valeur économique, re-cadrage 200
tirages, cadrage monthly ». Fait suite à `NOTE_nsdiff_consolidation_daily_vs_weekly.md`
(2026-08-05) et `NOTE_duel_nsdiff_vs_tsdiff_budget_egal.md` (2026-08-06).*

**Supersède**, sur les points où elles se contredisent : la référence
d'échantillonnage de la piste `oos` (`n_samples=50` → **200**, §1) ; le verdict
« ARIMA-GARCH bat NsDiff » des deux notes précédentes, qui ne vaut **que** pour
un run à graine unique et **plus** pour la configuration production (§2.2) ;
et le statut de TSDiff, désormais **retiré du benchmark** (§3.1).

**Artefacts produits** :

| Fichier | Chantier | Contenu |
|---|---|---|
| `experiments/multiple_testing.py` (+ `test_`) | A3-iii | Holm-Bonferroni dans la machinerie partagée — 18 tests unitaires |
| `experiments/benchmark_registry.py` | A1/A3 | qui est une référence vivante, à quelle référence d'échantillonnage |
| `experiments/oos_reference_audit.py` → `.json` | A1 | quels modèles sont sensibles au biais de quantile, et de combien |
| `experiments/repoint_oos_to_m200.py` → `.json` | A1 | la bascule de `tracking.db` sur la référence 200 (seule écriture en base de ce chantier) |
| `experiments/nsdiff_production_spec.py` (+ `test_`) | A2 | la config candidate production comme **spec exécutable** — 15 tests |
| `experiments/nsdiff_ensemble_vs_garch.py` → `.json` | A2 | le match « la config qu'on déploierait vs GARCH » |
| `experiments/nsdiff_refit_cadence.py` → `.json` | A3-ii | coût et effet d'un refit périodique |
| `experiments/econ_backtest.py` (+ `test_`) | B1 | moteur de backtest économique — 31 tests |
| `experiments/nsdiff_vs_garch_econ.py` → `.json` | B2 | le juge de paix : matrice actif × stratégie × régime |
| `experiments/nsdiff_scenarios.py` → `.json` | B3 | le cas d'usage différenciant (trajectoires complètes) |
| `experiments/kernelsynth.py` (+ `test_`) | C | générateur de séries synthétiques (recette Chronos) — 24 tests |
| `experiments/monthly_feasibility.py` → `.json` | C | inventaire, 3 voies sur pilote, baseline GARCH-monthly, go/no-go |

---

## 0. Le résultat principal, d'emblée

**La configuration qu'on déploierait n'est plus battue par ARIMA-GARCH. Elle ne
le bat pas non plus — ni en calibration, ni en argent, ni sur les scénarios.
Le programme se termine sur une parité, pas sur une victoire.**

Trois énoncés, dans l'ordre où ils ont été établis :

1. **Re-cadrage (A).** Le « NsDiff perd les 6 tests poolés contre GARCH » des
   notes précédentes était énoncé pour *une graine tirée au hasard*. Rejoué sur
   la **configuration production** (ensemble 5 graines × 200 tirages), GARCH ne
   gagne plus **aucun** des 6 tests poolés globaux — il en gagnait 6 sur 12
   (RMSE + Winkler) contre la graine unique. L'ensemble referme l'écart de
   calibration **à coût de calcul nul**.
2. **Économie (B).** Sur 90 cellules testées au niveau de coût central
   (5 actifs × 2 régimes × 3 horizons × 3 stratégies), **2** sont significatives
   — toutes deux en faveur de NsDiff, aucune ne survit à Holm au niveau poolé.
   Aucune stratégie, chez aucun des deux modèles, ne bat *acheter et garder*.
   La troisième famille du brief ne prend **aucune position** : un PI à 95 %
   n'exclut jamais le rendement nul, sur 2 700 origines et pour les deux modèles.
3. **Scénarios (B3).** Sur trois fonctionnelles que les quantiles marginaux de
   GARCH ne donnent pas (minimum de parcours, put ATM, digital à barrière), les
   deux modèles sont indistinguables sur 4 des 6 tests poolés. Sur les deux
   fonctionnelles franchement dépendantes du chemin, ils sont indistinguables
   **partout**. Sur la troisième — le pricing du put — GARCH gagne **7 cellules
   sur 10**. Il n'y a pas de différenciation par la trajectoire, et là où il y a
   une différence par cellule, elle joue contre NsDiff.
4. **L'asymétrie de protocole n'était pas l'explication (A3-ii).** Refit
   trimestriel (×7,4 en coût de calcul) ou mensuel (×24,6) : le verdict vs GARCH
   est **identique dans les trois bras**, y compris sur la seule cellule que
   GARCH gagne (W+1 daily). Le caveat « GARCH est plus frais » peut cesser
   d'être invoqué comme explication alternative.

Ce que cela veut dire en pratique : *la calibration statistique était le bon
combat, il est gagné (parité) ; la valeur économique différentielle n'existe
pas à ce jour, et elle ne se cache ni dans la fraîcheur du fit, ni dans la
structure jointe des trajectoires.* Le brief prévoyait ce cas comme critère
d'arrêt — il est atteint, et le chantier C en tire la conséquence (§6.5).

---

## 1. Chantier A1 — la piste `oos` a basculé sur la référence 200 tirages

### 1.1 Combien de modèles étaient réellement concernés : deux, pas sept

Le brief demandait de régénérer « tous les modèles échantillonnés ». Le premier
travail a été d'établir cette liste au lieu de la supposer — parce que la réponse
change le coût du chantier d'un ordre de grandeur.
`experiments/oos_reference_audit.py` la produit, preuve dans le code à l'appui :

| Modèle | Bornes | Sensible à *m* ? |
|---|---|---|
| ARIMA-GARCH | analytiques (`last_price·exp(μ + σ·ppf)`) | **non**, aucun tirage |
| SARIMA | analytiques (`conf_int()` de statsmodels) | **non** |
| Naive | analytiques (`prev ± 1,96·σ_t`, σ EWMA causale) | **non** |
| LSTM | analytiques (`pred ± 1,96·σ_t`) | **non** (le MC-Dropout existe mais vaut 0 par défaut) |
| Prophet | quantiles sur ses 1 000 tirages internes | marginalement (−0,19 point) |
| **NsDiff** | quantiles empiriques 2.5/97.5 sur *m* = 50 | **oui** (−3,73 points) |
| **TSDiff** | idem, *m* = 50 | **oui** |

Le biais est purement mécanique et se calcule sans rien simuler : `np.quantile`
vise le rang \(r = (m-1)q+1\), dont le niveau estimé vaut en espérance
\(r/(m+1)\).

| *m* | niveau estimé à q=0.975 | couverture réelle d'un PI « 95 % » | déficit |
|---|---|---|---|
| 50 | 0,9564 | **91,27 %** | −3,73 pts |
| 200 | 0,9703 | 94,05 % | −0,94 pt |
| 500 | 0,9731 | 94,62 % | −0,38 pt |
| 1 000 | 0,9740 | 94,81 % | −0,19 pt |

### 1.2 La décision, et pourquoi elle n'est pas un entre-deux

**Actée : le dashboard bascule sur la référence 200 tirages.** Portée réelle,
une fois l'inventaire fait : **NsDiff seul est repointé**. Les quatre modèles
analytiques n'ont rien à régénérer ; Prophet, à 1 000 tirages internes, est à
0,19 point de la cible — sous la résolution de tout ce que le programme mesure.
TSDiff est **retiré du benchmark** (§3.1) et n'est donc pas repointé.

Après bascule, **tout modèle encore affiché est soit analytique, soit lu sur au
moins 200 tirages**. La comparabilité mutuelle — la raison d'être de la
contrainte « ne pas régénérer NsDiff seul » — est rétablie. Ce n'est pas un
entre-deux : c'est le constat qu'il n'y avait qu'un modèle à bouger.

### 1.3 Ce qui a bougé en base

Vérifications exigées **avant** toute écriture, toutes passées : origines
identiques une à une (2 700 lignes), `y_true` identique, `last_close` aligné à
2,2·10⁻⁷ près. Sauvegarde horodatée de `tracking.db` prise automatiquement.

| Cellule | Cov95 avant (m=50) | après (m=200) | largeur (% du prix) |
|---|---|---|---|
| daily / W+1 | 0,867 | **0,909** | 14,72 → 15,55 |
| daily / W+2 | 0,891 | 0,911 | 22,42 → 23,41 |
| daily / W+3 | 0,893 | 0,907 | 24,38 → 25,53 |
| weekly / W+1 | 0,931 | **0,956** | 17,06 → 17,95 |
| weekly / W+2 | 0,927 | **0,960** | 23,99 → 25,51 |
| weekly / W+3 | 0,944 | 0,947 | 30,29 → 31,62 |

Global : 0,909 → **0,9315**. La largeur monte de ~5 % partout, la couverture avec
elle : la signature exacte du biais de quantile, et rien d'autre. Les colonnes
dérivées (`abs_error`, `in_interval`, …) ont été remises à NULL puis recalculées
par `backfill_eval_metrics.py` — sinon elles seraient restées calées sur les
anciennes bornes, exactement l'incohérence silencieuse que ce chantier corrige.
Le dashboard a été régénéré : 30 cellules (6 modèles × 5 actifs).

### 1.4 Pourquoi TSDiff n'a *pas* été repointé — la donnée, pas l'opinion

L'audit compare aussi TSDiff, et le refuse sur preuve : sa largeur de PI
**diminue** en passant de 50 à 200 tirages (daily W+1 : 5,58 % → 3,87 % du prix),
alors que davantage de tirages ne peut qu'élargir. C'est impossible à budget
d'entraînement constant — l'artefact 200 tirages utilise un budget d'époques
daily sélectionné par validation depuis. Le repointer aurait été une
**substitution de modèle**, pas un re-cadrage. Le script le détecte et refuse.

---

## 2. Chantier A2 — la question de production, posée pour la première fois

### 2.1 La spec, en code

`experiments/nsdiff_production_spec.py` porte désormais la configuration
candidate production, et `nsdiff_seed_ensemble.build_ensemble_rows` l'appelle :
la formule n'existe qu'à un seul endroit, et ce qui est testé est littéralement
ce qui serait déployé.

- 5 graines (42–46), budget plat 40 époques, seq_len=30, k_denoise=20 ;
- 200 tirages par graine → **1 000 tirages** ;
- **concaténation** des nuages, jamais moyenne des bornes. Concaténer lit les
  quantiles du *mélange* des 5 lois : la bande s'élargit quand les graines sont
  en désaccord, ce qui est l'effet recherché. Moyenner les bornes l'aurait lissé.
  Un test unitaire verrouille précisément cet invariant (le mélange doit être
  **plus large** que la moyenne des bandes quand les graines divergent) ;
- point = moyenne, bandes = quantiles 2.5/97.5 du nuage concaténé ;
- **aucun refit** ;
- biais résiduel déclaré : 94,81 % de couverture réelle pour une étiquette 95 %.

*Effet de bord du passage par la spec, déclaré* : la concaténation se fait
désormais en float64 au lieu de float32. Les chiffres de la tâche 6 bougent au
9ᵉ chiffre significatif (écart relatif max 7,3·10⁻⁴, porté par la granularité du
bootstrap) et **aucun des 228 verdicts ne change**.

### 2.2 Le résultat : GARCH ne bat plus la config production

| Cellule poolée (globale) | tâche 7 — « une graine au hasard » | **A2 — la config production** |
|---|---|---|
| W+1 weekly | **GARCH+** (RMSE, p=0,0298) / = | = (0,145) / = (0,374) |
| W+1 daily | **GARCH+** (p<0,0001) / **GARCH+** (p=0,0194) | = (0,154) / = (0,411) |
| W+2 weekly | = (0,066) / = | = (0,156) / = (0,401) |
| W+2 daily | **GARCH+** (p=0,0080) / = | = (0,495) / = (0,954) |
| W+3 weekly | **GARCH+** (p=0,0300) / = | = (0,096) / = (1,000) |
| W+3 daily | = / **GARCH+** (p=0,0484) | = (0,660) / = (0,142) |

*(RMSE / Winkler ; « = » = indistinguable ; les p-values ne sont pas comparables
une à une entre les deux colonnes — le bras A a changé — mais les verdicts le sont.)*

**6 défaites sur 12 deviennent 0 sur 12.** Aucune victoire non plus : l'ensemble
amène NsDiff à parité avec GARCH, il ne le fait pas passer devant.

Correction de Holm, familles déclarées a priori (décision = les 6 tests poolés
globaux par métrique ; étendue = les 24, tous groupes) : **rien ne survit** dans
la famille de décision (0 rejet brut). Dans la famille étendue, une seule case
tient — Winkler, weekly W+1, obligations, en faveur de GARCH (p=0,0004,
p ajustée 0,0096).

---

## 3. Chantier A3 — les nettoyages de protocole

### 3.1 TSDiff est retiré du benchmark

La fuite de sélection d'époques hebdomadaire (validation à l'intérieur de la
grille de test, ~13 % des origines) n'est pas corrigée : **TSDiff est retiré**.
Deux motifs cumulatifs, consignés dans `benchmark_registry.py` :

- la fuite joue **en sa faveur** et il perd quand même les 6 tests poolés weekly
  du duel à budget égal ;
- sa couverture observée sur la grille `oos` va de **0,53 à 0,85** pour un PI
  étiqueté 95 %. Ce n'est plus un modèle sous-calibré.

Re-sweeper le bras hebdomadaire aurait coûté un chantier complet pour réparer un
modèle déjà battu. Ses lignes restent en base (historique vérifiable) ; le
dashboard ne l'affiche plus par défaut (`--exclude-models`, opt-in, additif).

### 3.2 La politique de tests multiples est passée en machinerie

`experiments/multiple_testing.py` — Holm-Bonferroni, appliqué dans A2, B2, B3 et
A3-ii. Choix déclarés : Holm plutôt que Bonferroni (il le domine uniformément au
même coût d'hypothèse), plutôt que Benjamini-Hochberg (BH contrôle le FDR ; ici
la question est décisionnelle et binaire, une seule case fausse suffit à fonder
une mauvaise décision de production → on contrôle le FWER). Holm ne suppose
aucune structure de dépendance, ce qui compte quand les tests d'une famille
partagent les mêmes origines.

**Définition de famille, appliquée partout** : une famille = les tests **poolés**
d'un même match et d'une même métrique. Les tests **par cellule** ne sont pas
corrigés — exploratoires par construction, et aucune conclusion du programme ne
s'y appuie. Les verdicts bruts ne sont jamais écrasés : les deux coexistent, et
`family_summary` rapporte systématiquement **ce que la correction coûte**.

### 3.3 Cadence de refit — l'asymétrie de protocole, enfin chiffrée

Depuis le début du programme, tout classement inter-modèles traîne le même
caveat : ARIMA-GARCH est refit **à chaque origine**, NsDiff est entraîné **une
fois** et roule 90 origines (~21 mois) dessus. Quand GARCH gagne, on ne sait pas
s'il gagne parce qu'il est meilleur ou parce qu'il est plus frais. Le caveat
était déclaré partout et n'avait jamais été mesuré.

`experiments/nsdiff_refit_cadence.py` le mesure sur **SPY et BTC-USD** (le mieux
couvert et le plus volatil), 5 graines, trois cadences : train-once,
trimestrielle (13 semaines), mensuelle (4 semaines). Fenêtre expansive dans tous
les cas — changer aussi la forme de la fenêtre mélangerait deux effets. Les bras
sont **appariés au niveau du générateur aléatoire** : `set_seed(seed + k)` à
chaque origine dans tous les bras, donc l'écart mesuré est l'effet du refit, pas
du bruit Monte-Carlo.

**Le coût :**

| Cadence | fits par (actif, graine) | temps de fit | multiplicateur |
|---|---|---|---|
| train-once | 1 | 19 s | ×1 |
| trimestrielle | 7 | 142 s | **×7,4** |
| mensuelle | 23 | 470 s | **×24,6** |

**Ce que ça achète :** presque rien. Sur 60 cellules (2 actifs × 2 régimes × 3
horizons × 5 graines) et après correction de Holm :

| Cadence | RMSE : bruts → Holm | Winkler : bruts → Holm |
|---|---|---|
| trimestrielle | 13 → **4** | 2 → **0** |
| mensuelle | 13 → **3** | 4 → **1** |

Toutes les cellules survivantes sont **BTC-USD**, et toutes vont dans le sens du
refit — cohérent : c'est l'actif dont la loi bouge le plus, donc celui où un fit
vieux de 21 mois coûte le plus cher. Sur SPY, **aucune cellule ne survit à Holm,
à aucune cadence**. Et les verdicts bruts vont dans les deux sens (trimestriel,
RMSE : 10 en faveur du refit, 3 en faveur du train-once).

**Et surtout — le verdict vs GARCH ne bouge pas :**

| Cellule poolée | train-once | trimestriel | mensuel |
|---|---|---|---|
| W+1 weekly | = / = | = / = | = / = |
| **W+1 daily** | **GARCH+** (RMSE, p<0,0001) | **GARCH+** (p<0,0001) | **GARCH+** (p<0,0001) |
| W+2 weekly | = / = | = / = | = / = |
| W+2 daily | = / = | = / = | = / = |
| W+3 weekly | = / = | = / = | = / = |
| W+3 daily | = / = | = / = | = / = |

**Réponse :** l'asymétrie de protocole **n'explique pas** l'avantage de GARCH.
Rafraîchir NsDiff jusqu'à 24 fois plus souvent ne déplace aucun verdict, y compris
celui — W+1 daily — que la fraîcheur était censée expliquer. Le caveat peut
cesser d'être invoqué comme explication alternative ; il reste vrai comme
description du protocole.

**Recommandation opérationnelle de cadence** : train-once-forward, et refit
**trimestriel sur les actifs crypto uniquement**. C'est le seul endroit où le
refit paie (4 cellules BTC survivent à Holm), au multiplicateur de coût le plus
faible ; le mensuel coûte 3,3 fois plus cher que le trimestriel pour un cellule
de moins.

---

## 4. Chantier B — le juge de paix

### 4.1 Ce qui a été construit

`experiments/econ_backtest.py` : moteur pur (aucune E/S, aucune connaissance de
la base ni des modèles), 31 tests unitaires. **Aucune fonction de décision ne
sait quel modèle l'appelle** — elles ne reçoivent que
(`last_close`, `y_pred`, `y_lower`, `y_upper`), ce que les deux modèles publient.

Le test unitaire qui compte : **causalité de bout en bout**. On saccage tout ce
qui suit l'instant *t* et on vérifie que la position en *t* ne bouge pas, pour
les trois familles. Une stratégie qui normaliserait par la médiane de la fenêtre
de test entière — la fuite classique qui rend un backtest joli — tomberait
immédiatement.

**Trois familles, déclarées a priori** (aucun seuil choisi après lecture d'un
résultat) :

1. `inverse_width` — position ∝ 1/largeur du PI, normalisée par la **médiane
   expansive** des largeurs du modèle lui-même (donc : sa confiance *relative*,
   pas son niveau absolu, qui diffère d'un modèle à l'autre) ; signe donné par
   le point prédictif vs le prix courant ; 8 origines de chauffe ;
2. `var_limit` — sleeve long dimensionné pour que sa VaR 2,5 % vaille un budget
   de risque de 3 % ; long-only volontairement, pour isoler la qualité du
   quantile bas de la question directionnelle ;
3. `filtered_direction` — position prise **seulement si le PI exclut le
   rendement nul**.

**Coûts** : trois niveaux, par classe d'actif — de SPY à ETH, un chiffre unique
serait faux partout. Aller-retour = 2 × bps × |position|.

| Niveau | actions / obligations | crypto |
|---|---|---|
| faible | 1 bp | 10 bps |
| **central** (décision) | 5 bps | 30 bps |
| élevé | 10 bps | 60 bps |

**Recouvrement, déclaré** : à l'horizon *h*, chaque origine ouvre un sleeve tenu
*h* semaines — les PnL par origine se chevauchent pour *h*>1. Traité par le
bootstrap **par blocs** (block_length=3), jamais par un test i.i.d. ; Sharpe
annualisé par √(52/h), pas √52.

**Référence de contexte** : acheter et garder. Sans elle, « NsDiff bat GARCH de
40 bps » ne dit pas si l'un ou l'autre valait mieux que ne rien faire.

### 4.2 La famille 3 ne prend jamais position — et c'est un résultat

Sur **2 700 origines**, pour **les deux modèles**, aux trois horizons :
**0 position ouverte**. Un intervalle à 95 % contient toujours le prix courant à
ces horizons. La question du brief — « des intervalles honnêtes filtrent-ils
mieux les faux signaux ? » — reçoit une réponse nette : **à 95 %, aucun des deux
modèles n'émet jamais de signal**. Tester à un niveau plus étroit exigerait de
choisir un seuil après avoir vu ce résultat ; non fait, délibérément.

### 4.3 Verdicts poolés, niveau de coût central

PnL total sur 90 origines, par unité de capital déployée par origine
(ensemble / GARCH / acheter-et-garder) :

| Cellule | `inverse_width` | `var_limit` |
|---|---|---|
| weekly W+1 | −0,177 / −0,287 / −0,230 | −0,047 / −0,067 / −0,230 |
| daily W+1 | −0,204 / −0,186 / −0,230 | −0,067 / −0,055 / −0,230 |
| weekly W+2 | −0,451 / −0,191 / −0,073 | +0,012 / −0,017 / −0,073 |
| daily W+2 | −0,524 / +0,094 / −0,073 | −0,002 / −0,010 / −0,073 |
| weekly W+3 | −0,346 / −0,367 / +0,105 | +0,056 / +0,017 / +0,105 |
| daily W+3 | −0,334 / −0,174 / +0,105 | +0,032 / +0,026 / +0,105 |

**Les 12 verdicts ensemble-vs-GARCH sont indistinguables.** Aucun des deux
modèles ne bat acheter-et-garder sur aucune cellule (12 tests, tous
indistinguables, p de 0,25 à 1,00). Sur les 90 cellules par actif au coût
central, **2 sont significatives** — `var_limit` sur SPY weekly à W+2 et W+3, en
faveur de NsDiff. Deux cellules sur 90 est ce qu'un tirage à 5 % produit en
moyenne (4,5), et rien ne survit à Holm au niveau poolé.

`inverse_width` **détruit de la valeur** : sur les 12 séries de PnL poolées
(2 modèles × 6 cellules), **11 sont négatives** et **8 font pire
qu'acheter-et-garder**. Le proxy de confiance « 1/largeur » n'est informatif ni
pour NsDiff ni pour GARCH. `var_limit` est nettement plus sain (7 séries
négatives sur 12, 4 pires que B&H) mais ne dégage pas d'avantage différentiel.

Robustesse aux coûts : les verdicts sont **identiques** aux trois niveaux
(un seul basculement, `var_limit` au niveau *élevé*, vers NsDiff).

### 4.4 VaR — la queue gauche du régime daily n'est pas fiable

Sur 30 cellules (5 actifs × 2 régimes × 3 horizons), taux de violation observé
contre une cible de 2,5 %, test de Kupiec : NsDiff-ensemble est rejeté sur
3 cellules (toutes en **sous**-couverture, crypto/daily aux horizons longs :
BTC daily W+2 6,7 %, BTC et ETH daily W+3 7,8 %) ; GARCH sur 2 (ETH **weekly**
W+2 6,7 % et W+3 8,9 %). Sur le **coût des dépassements**, l'asymétrie est nette
et va dans un seul sens : en **daily**, NsDiff paie plus cher que GARCH sur
**15 cellules sur 15** (ETH W+3 : 0,535 contre 0,198) ; en **weekly**, il paie
moins cher sur 8 cellules sur 15 seulement — un avantage réel mais non
systématique, porté par la crypto (ETH W+3 : 0,106 contre 0,405). Autrement dit
le régime daily de NsDiff a une queue gauche franchement moins fiable, et son
régime weekly ne fait que jeu égal.

### 4.5 SPA de Hansen — la seule tension du chantier, et sa résolution

Le test SPA (`mcs.spa_test`, déjà dans le repo, GARCH en référence, perte = −PnL)
ne dit pas tout à fait la même chose que le bootstrap apparié. Sur `var_limit`,
**régime weekly**, il rejette « aucun modèle ne bat GARCH » aux trois horizons :

| Cellule | p SPA brute | p ajustée (Holm) | gain moyen par origine |
|---|---|---|---|
| weekly W+1 | 0,0332 | 0,199 | +0,00022 |
| weekly W+2 | 0,0406 | 0,203 | +0,00032 |
| weekly W+3 | 0,0482 | 0,203 | +0,00043 |
| daily W+1/2/3 | 1,000 / 0,368 / 0,396 | 1,000 | −0,00013 / +0,00009 / +0,00007 |

C'est cohérent, pas contradictoire : SPA est **unilatéral** et recentré, donc plus
puissant que le bootstrap bilatéral pour détecter un petit avantage systématique.
Mais **le SPA passe sous Holm comme le reste** — l'exempter reviendrait à se
réserver le test le plus favorable. Famille de 6, seuil le plus strict 0,0083 :
**0 rejet sur 6 survit**, la plus petite p ajustée valant 0,199.

Et surtout, l'ordre de grandeur tranche : **+2 à +4 points de base par origine**,
avant même de considérer qu'un aller-retour crypto coûte 60 bps au niveau
central. L'avantage détecté par SPA, à supposer qu'il soit réel, est un ordre de
grandeur sous les frais.

Les mêmes familles pour les tests appariés : `var_limit|pooled_vs_garch` a 2
rejets bruts, **0 après Holm** ; tout le reste a 0 rejet brut. **Aucune des 9
familles de décision du chantier B ne produit un seul résultat qui survive.**

---

## 5. Chantier B3 — pas de différenciation par la trajectoire

### 5.1 Comment le bras GARCH a été construit sans l'affaiblir

Le piège de ce test est de comparer un nuage de trajectoires à un homme de
paille. On ne prête à GARCH aucune faiblesse qu'il n'a pas : on reconstruit le
générateur de scénarios que **ses propres sorties** définissent. Deux
vérifications empiriques avant d'écrire une ligne :

- `log(y_lower) + log(y_upper) = 2·log(y_pred)` à la précision machine sur
  **100 %** des lignes → l'intervalle est exactement symétrique en log, donc
  (μ_h, σ_h) se récupèrent **sans approximation** ;
- σ_h croît strictement avec *h* sur **100 %** des lignes → les variances
  d'incréments sont toutes positives, la reconstruction est toujours définie.

Les trajectoires GARCH sont des marches aléatoires gaussiennes calées sur
(μ_h, σ_h) : elles **reproduisent exactement** ses trois intervalles marginaux.
La seule différence entre les bras est la **structure de la loi jointe**.
Budget d'échantillonnage strictement égal : 1 000 trajectoires de chaque côté.

*Ce que cette construction concède à GARCH, à citer* : ses incréments sont
gaussiens et indépendants par construction. Si le modèle avait une opinion sur
l'asymétrie ou le clustering intra-fenêtre, elle n'apparaît pas dans ses trois
intervalles publiés. L'écart mesuré porte donc sur **ce que le format de sortie
de GARCH permet**, pas sur ce que le modèle GARCH sait — c'est la comparaison
pertinente pour une décision de production, où l'on consomme des sorties.

Côté NsDiff, les trajectoires sont réelles : les trois horizons d'une (graine,
origine) partagent l'index de tirage — un seul appel à `sample_paths`, dont
`forecast_from_fitted` lit les sommes cumulées. Corrélation vérifiée entre
tirage *i* à W+1 et à W+2 : **0,76** (un tirage indépendant donnerait 0).

### 5.2 Le verdict

| Régime | fonctionnelle | NsDiff | GARCH | verdict |
|---|---|---|---|---|
| weekly | CRPS min de parcours | 0,03403 | 0,03430 | indistinguable (p=0,586) |
| weekly | PnL vendeur de put ATM | −0,00007 | −0,00294 | **NsDiff+** (p<0,0001) |
| weekly | Brier digital à barrière | 0,1485 | 0,1502 | indistinguable (p=0,569) |
| daily | CRPS min de parcours | 0,03475 | 0,03456 | indistinguable (p=0,769) |
| daily | PnL vendeur de put ATM | −0,00480 | +0,00223 | **GARCH+** (p<0,0001) |
| daily | Brier digital à barrière | 0,1514 | 0,1499 | indistinguable (p=0,713) |

Les deux seules cases significatives survivent à Holm (m=6) — **et elles pointent
en sens opposés selon le régime**.

**Sur les deux fonctionnelles franchement dépendantes du chemin**, le résultat est
net : minimum de parcours et digital à barrière sont **indistinguables en poolé
dans les deux régimes**, et sur les 20 cellules par actif, seules 3 sont
significatives (SPY weekly / CRPS en faveur de NsDiff ; TLT weekly / CRPS et
TLT weekly / Brier en faveur de GARCH) — moins que ce qu'un tirage à 5 % sur
20 cellules produirait en moyenne, et dans les deux directions. **La structure
jointe de la diffusion n'apporte rien de mesurable là où on l'attendait le plus.**

**Sur le put ATM, il faut être précis, car le poolé et les cellules ne disent pas
la même chose.** Par cellule, GARCH gagne **7 fois sur 10** et NsDiff 2 (ETH
weekly, TLT weekly). Le poolé weekly bascule pourtant en faveur de NsDiff : la
moyenne cross-sectionnelle du PnL est dominée par l'écart ETH-weekly, très large
en valeur absolue. C'est une limite réelle du pooling sur une quantité non
normalisée — que le skill-score évite ailleurs dans le programme, et qui n'a pas
d'équivalent naturel pour un PnL. **Lecture retenue : par cellule, GARCH price
mieux le put ; le poolé weekly ne doit pas être lu comme l'inverse.** Le
diagnostic de stress (taux de dépassement du quantile 5 % du minimum de parcours,
cible 5 %) confirme sans départager : NsDiff est meilleur en weekly (ETH 0,056 vs
0,133), GARCH en daily (BTC 0,133 vs 0,056).

**Réponse à la question du brief** : non, NsDiff ne se différencie pas sur les
scénarios complets — et sur le seul axe où une différence par cellule est nette
(le pricing d'option), elle joue **contre** lui. La conclusion du programme est
complète.

---

## 6. Chantier C — cadrage monthly : NO-GO, mais de justesse

### 6.1 Inventaire — le problème est le nombre de FENÊTRES, pas d'observations

| Actif | quotidiennes | hebdo | **mensuelles** | fenêtres d'entraînement à seq_len=30 |
|---|---|---|---|---|
| SPY | 2 905 | 604 | 139 (2015-01 → 2026-07) | **70** (×8,0 de moins qu'en hebdo) |
| BTC-USD | 4 222 | 604 | 139 | 70 (×8,0) |
| ZN=F | 2 905 | 604 | 139 | 70 (×8,0) |
| ETH-USD | 3 179 | 455 | 105 (2017-11 →) | **36** (×11,4) |
| TLT | 2 121 | 441 | 102 (2018-02 →) | **33** (×12,0) |

Ce que devient `seq_len=30` à ce volume : il consomme 30 des ~100 rendements
d'entraînement disponibles, laissant **33 à 70 fenêtres**. Sur ETH et TLT, un
modèle à 30 pas de lookback voit littéralement une trentaine d'exemples.
Le budget d'époques devient donc le levier dominant — exactement ce que le
programme identifie depuis l'étape 1 bis.

**Confirmé par le sweep** : un premier passage sur (20, 40, 80) époques a placé
l'argmin **au bord** de la grille, CRPS de validation strictement décroissant. Une
grille dont l'optimum est sur sa borne n'est pas une sélection, c'est une
troncature — la grille a été élargie **une fois**, à (20, 40, 80, 160, 320),
avant de regarder le moindre résultat de test. L'effet est massif : à 40 époques,
`monthly_native` produit une bande de **81 % du prix** et couvre 100 % ; à 160
époques, 32 % et 94 %. Sous-entraîné, le modèle ne fait qu'élargir.

### 6.2 Protocole

Le **mois** est défini comme la dernière observation hebdomadaire (grille W-FRI)
de chaque mois calendaire. Conséquence voulue : la grille mensuelle est un
**sous-ensemble** de l'hebdomadaire, donc les trois voies partagent exactement
les mêmes origines et les mêmes cibles. Définir le mois sur la dernière séance du
mois aurait rendu la voie (ii) incomparable aux deux autres.

Pilote **SPY**. 36 origines de test (2023-05 → 2026-04), 12 de validation
strictement antérieures (2022-05 → 2023-04). `effective_n` = **12** — trois fois
moins que le corps du programme. Baseline `garch_monthly` refit à chaque origine.
`seq_len` et époques choisis par argmin CRPS sur la validation, jamais sur le test.

### 6.3 Résultats

| Voie | config retenue | Cov95 | largeur (% prix) | Winkler | RMSE | CRPS | verdict |
|---|---|---|---|---|---|---|---|
| `garch_monthly` (baseline) | refit/origine | 0,944 | 22,00 | **152,7** | **31,17** | n/a | — |
| `monthly_native` | seq_len=12, 160 ép. | **0,926** | 22,67 | 154,5 | 31,34 | **17,73** | **NO-GO** |
| `weekly_propagated` | seq_len=30, 160 ép. | 0,898 | 21,79 | 170,9 | 31,42 | 18,83 | **NO-GO** |
| `synthetic_augmented` | seq_len=30, 40 ép. | 0,852 | 20,57 | 214,0 | 34,48 | 19,27 | **NO-GO** |

Détail des deux critères, horizon par horizon :

| Voie | M+1 | M+2 | M+3 |
|---|---|---|---|
| `monthly_native` | cov 0,917 ✓ / Winkler non pire | cov **0,889 ✗** / non pire | cov 0,972 ✓ / non pire |
| `weekly_propagated` | cov **0,861 ✗** / non pire | cov **0,861 ✗** / non pire | cov 0,972 ✓ / non pire |
| `synthetic_augmented` | cov **0,861 ✗** / non pire | cov **0,861 ✗** / **PIRE** (p=0,0074) | cov **0,833 ✗** / **PIRE** (p=0,0006) |

**Verdict : NO-GO.** Aucune voie ne passe les deux critères déclarés.

### 6.4 Ce qu'il faut lire, et ce qu'il ne faut pas

- **`monthly_native` échoue d'une seule observation.** Sa seule case hors bande
  est M+2 à 0,889 contre une borne à 0,90 : sur 36 origines, c'est **32 succès au
  lieu de 33**. Le critère a été fixé a priori et il est appliqué tel quel — mais
  présenter ce NO-GO comme un rejet net serait malhonnête. Le bon énoncé est :
  *à `effective_n`=12, le pilote ne permet pas de conclure au go, et il n'est
  pas loin.* Sur les deux autres axes, `monthly_native` fait jeu égal avec
  GARCH-monthly (Winkler 154,5 vs 152,7 ; RMSE 31,34 vs 31,17 ; jamais
  significativement pire à aucun horizon).
- **La voie (ii) `weekly_propagated` est la plus décevante.** Propager le nuage
  hebdomadaire jusqu'à la cible mensuelle sous-couvre franchement à M+1 et M+2
  (0,861) et paie 12 % de Winkler de plus que la voie native. Le modèle
  hebdomadaire, entraîné à un horizon de 3 semaines, est étiré à 13 pas : il ne
  transporte pas correctement l'incertitude sur cette distance.
- **La voie (iii) `synthetic_augmented` est la seule à DÉGRADER.** Elle est
  significativement pire que GARCH-monthly à M+2 et M+3, sous-couvre partout
  (0,833–0,861) et a le pire RMSE. L'augmentation KernelSynth ne compense pas le
  manque de données ici : elle apporte de la **diversité de formes** (tendances,
  cycles, ruptures de corrélation), pas les propriétés stylisées qui manquent
  — queues lourdes, clustering de volatilité, asymétrie. Un processus gaussien à
  covariance fixe est homoscédastique par construction. Entraîner un modèle
  d'incertitude sur 5 fenêtres synthétiques homoscédastiques pour 1 fenêtre réelle
  lui apprend à ne pas voir la volatilité changer. **La condition du brief
  (« si (i) sous-entraîne ») était bien remplie ; le remède proposé ne marche
  pas.**

### 6.5 Conséquence, avec le critère d'arrêt du brief

Le brief déclare : *« Si B conclut "aucune valeur économique ajoutée, aucune
différenciation scénarios", le chantier C perd sa justification pour NsDiff — le
documenter comme critère d'arrêt. »*

**B conclut exactement cela** (§4, §5). Le chantier C a néanmoins été mené
jusqu'au bout — il était lancé en parallèle, comme l'ordonnancement du brief le
prévoyait, et une étude de faisabilité qui s'arrête avant son verdict ne vaut
rien. Son résultat converge avec le critère d'arrêt : **le mensuel sort du
périmètre NsDiff**, et c'est documenté.

**Ce qui resterait à faire si on y revenait un jour**, dans l'ordre de rendement
attendu : (1) refaire le pilote sur les 5 actifs — le NO-GO tient à une
observation sur un seul actif, et 5 pilotes donneraient une base de décision ;
(2) élargir encore la grille d'époques vers le haut, la courbe de CRPS n'ayant
été explorée qu'une fois au-delà de 80 ; (3) abandonner la voie (iii) telle
quelle, ou remplacer KernelSynth par un générateur à volatilité stochastique.
La voie (ii) est à abandonner : elle est dominée par (i) sur les quatre métriques.

---

## 7. Non-négociables — statut

- **Multi-graines 42-46, conventions descriptif / graine fixe / poolé.** Aucune
  conclusion de cette note ne repose sur une graine unique. Le chantier A2
  introduit un quatrième objet — la **configuration production** (l'ensemble des
  5 graines) — qui n'est ni un run à graine unique ni un poolage de métriques :
  c'est une configuration à part entière, et c'est déclaré partout où elle sert.
  Le chantier B évalue systématiquement les 5 graines individuellement à côté du
  poolé et de l'ensemble (`n_seeds_beating_garch` / `n_seeds_losing_to_garch`
  dans chaque cellule).
- **Briques réutilisées, pas réimplémentées** : `paired_test`,
  `dashboard_d7_w1` (Winkler, skill-score, pooling par classe),
  `calibration_tests` (Kupiec, écart de couverture par blocs),
  `diffusion_headtohead` (machinerie de match), `mcs.spa_test`,
  `crps_metrics.crps_empirical`, `matrice_paired_tests`,
  `benchmarks.multi_horizon.forecast_horizons_arima`,
  `validation.sim_trades.insert_oos_predictions`,
  `backfill_eval_metrics.py` — tous **importés et appelés**.
  Code neuf uniquement là où rien n'existait : Holm (`multiple_testing`), moteur
  de backtest économique (`econ_backtest`), générateur synthétique
  (`kernelsynth`), spec production (`nsdiff_production_spec`), registre
  (`benchmark_registry`). **88 tests unitaires neufs**, cas calculés à la main,
  entrées dégénérées, invariants — dont un test de **causalité de bout en bout**
  sur les trois familles de stratégies.
- **`tracking.db`** : une seule écriture, celle de la décision A1 explicitement
  actée (2 700 lignes NsDiff repointées), précédée d'une sauvegarde horodatée et
  de trois vérifications d'alignement bloquantes. Tous les autres scripts sont en
  lecture seule.
- **Deux modifications de fichiers partagés**, déclarées ici plutôt que
  découvertes dans un diff :
  1. `dashboard_d7_w1.py` (+33/−3) — ajout de `--exclude-models` et d'un bloc de
     provenance `sampling_reference` dans le payload. **Le comportement par
     défaut change** : TSDiff n'est plus affiché. C'est voulu (c'est la décision
     A3.1) et non additif — `--exclude-models` sans argument restaure
     l'affichage complet, y compris les modèles retirés.
  2. `nsdiff_seed_ensemble.build_ensemble_rows` — passe désormais par
     `nsdiff_production_spec.production_forecast` au lieu de recalculer la
     formule. Conséquence numérique déclarée en §2.1 (float64 au lieu de
     float32 ; 9ᵉ chiffre significatif ; **0 verdict sur 228 ne change**).
- **Prix gelés partagés** entre tous les bras d'un même match
  (`diffusion_multiseed_v2/prices/*.parquet`, aucun appel réseau) ; **budget
  d'échantillonnage strictement égal** : 1 000 trajectoires de chaque côté en
  §5, 200 tirages partout ailleurs.
- **Tout seuil déclaré a priori** : marges de coût par classe d'actif, budget de
  VaR (3 %), plafond de position (1×), chauffe (8 origines), barrière (0,95),
  quantile de stress (5 %), bande de couverture go/no-go ([0,90 ; 0,98]), familles
  de Holm. **Une seule exception, déclarée et datée** : la grille d'époques du
  chantier C a été élargie une fois (§6.1) parce que son argmin était sur la
  borne — élargissement décidé sur la **validation**, avant tout regard sur le test.
- **pytest vert** : `python -m pytest experiments validation -q` →
  **453 passed, 1 skipped** (contre 365 passed, 1 skipped avant ce chantier).
  Le skip est pré-existant et sans rapport (`test_crps_metrics.py:61 —
  properscoring not installed`).

## 8. Limites déclarées

- **Puissance.** `effective_n` = 30 pour les chantiers A et B (90 origines,
  blocs de 3) ; **12** pour le chantier C (36 origines). Aucun « indistinguable »
  de cette note ne démontre une absence d'effet. C'est particulièrement vrai du
  résultat central §2.2 : « GARCH ne bat plus la config production » signifie
  *l'écart n'est plus détectable à cette puissance*, pas *l'écart est nul*.
- **Tests multiples : corrigés là où ils portent une décision, pas ailleurs.**
  Les tests par cellule (90 en B, 20 en B3, 60 en A3-ii) ne sont pas corrigés et
  sont déclarés exploratoires. Les comptages de cellules significatives donnés
  dans cette note sont des comptages **bruts** et doivent être lus comme tels.
- **Pooling d'un PnL non normalisé** (§5.2) : la moyenne cross-sectionnelle d'un
  PnL peut être dominée par un seul actif. C'est arrivé une fois (put ATM,
  weekly) et c'est signalé à l'endroit exact où ça change la lecture. Le reste du
  programme évite ce piège par les skill-scores ; il n'existe pas d'équivalent
  naturel pour un PnL.
- **Drawdown maximal** (`econ_backtest.max_drawdown`) : proxy déclaré. À
  l'horizon *h* > 1 les sleeves se chevauchent, donc la courbe cumulée n'est pas
  l'equity d'un compte réel. Comparable entre modèles (même convention des deux
  côtés), pas interprétable comme un drawdown de compte.
- **Bras GARCH du §5** : ses incréments sont gaussiens et indépendants **par
  construction de la reconstruction**. L'écart mesuré porte sur ce que le format
  de sortie de GARCH permet, pas sur ce que le modèle GARCH sait.
- **Chantier A3-ii** : mesuré sur **2 actifs sur 5** (SPY, BTC-USD), comme le
  brief l'autorisait (« sur au moins un actif »). L'extrapolation aux trois autres
  est une inférence, pas une mesure — mais elle est appuyée par le fait que les
  seules cellules survivantes sont celles de l'actif le plus volatil.
- **Famille 3 du chantier B** : la réponse « aucune position jamais prise » est
  spécifique au niveau 95 %. Tester à un niveau plus étroit exigerait de choisir
  un seuil après avoir vu ce résultat ; non fait, délibérément.
- **KernelSynth** : une série tirée d'un GP à covariance fixe n'a ni queues
  lourdes, ni clustering de volatilité, ni asymétrie. C'est la limite qui explique
  §6.4 et elle était déclarée dans le module avant le run, pas après.
- **Non fait, volontairement** : aucun re-sweep du bras hebdomadaire de TSDiff
  (le modèle est retiré, §3.1) ; aucun repointage des lignes TSDiff en base
  (§1.4) ; aucune reprise des 5 modèles à bornes analytiques (rien à reprendre,
  §1.1) ; aucun pilote monthly hors SPY (§6.5).
