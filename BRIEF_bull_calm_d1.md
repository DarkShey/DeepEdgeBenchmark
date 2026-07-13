# BRIEF — Bull-Calm : Test Case Bull à D+1 (`bull_calm_d1`)

> **Statut** : spec validée (2026-07-12), à implémenter.
> **Objet** : mesurer l'utilisabilité *opérationnelle* d'une prédiction Price Interval 95 % à D+1
> via une stratégie simple et falsifiable — *long today, take profit tomorrow*.
> **Rule version** : `bull_calm_d1`.
> **Principe directeur** : on ne juge plus le modèle sur des métriques abstraites (RMSE, couverture)
> mais sur le **P&L d'une règle de trading explicite**. Un modèle « statistiquement bon » qui perd de
> l'argent sur Bull-Calm est inutilisable ; c'est exactement ce qu'on veut détecter.

---

## 0. Contexte & état du repo (ce qui existe déjà)

Les briques de données sont en place ; Bull-Calm ne rajoute qu'une **couche de simulation** par-dessus.

| Brique existante | Emplacement | Contenu utile pour Bull-Calm |
|---|---|---|
| Log OOS par combo | `Run/<run>/predictions.parquet` | `date, actual, predicted, pi_lower, pi_upper` (~113 pts OOS/combo, walk-forward one-step-ahead) |
| Prédictions live | `validation/tracking.db` → table `predictions` | `last_close, y_pred, y_lower, y_upper, target_date, y_true, direction_correct, in_interval, regime, …` |
| Résolution quotidienne | `validation/evaluate_daily.py` (`evaluate_pending`) | remplit `y_true`, `direction_correct`, `in_interval` quand la `target_date` est échue |
| Métriques modèle | `Run/<run>/metrics.json` | `pi_coverage_95`, `directional_accuracy`, `skill_vs_naive`, … |
| Verdicts pré-résolution | `validation/verdict_rules.py` | `verdict_integrite`, `verdict_plausibilite` |

**Ce que Bull-Calm ajoute et qui n'existe pas encore** : la table `daily_oos_log` (vue normalisée du log OOS
alimentant la simulation), la table `sim_trades` (un trade simulé par signal, avec `counter` et `roi`),
la règle `bull_calm_d1`, et les KPIs agrégés.

---

## 1. Vocabulaire & correspondance des noms

| Nom conceptuel (note manuscrite) | Champ opératif | Définition |
|---|---|---|
| `P_market(D)` | `reference_price` | dernier close **connu à D** (= `actual[t-1]` en OOS, `last_close` en live) |
| `PI_middle` | `predicted` / `y_pred` | prévision ponctuelle du modèle pour D+1 (entre PI_low et PI_high) |
| `PI_high` | `pi_upper` / `y_upper` | borne haute de l'intervalle conformal 95 % pour D+1 |
| `PI_low` | `pi_lower` / `y_lower` | borne basse de l'intervalle conformal 95 % pour D+1 |
| `P_market(D+1)` | `realized_price` | vrai close réalisé à D+1 (= `actual[t]` en OOS, `y_true` en live) |

---

## 2. Alignement temporel — **LE point critique** (anti look-ahead)

Dans `predictions.parquet`, la ligne `t` porte la prévision **one-step-ahead** faite pour `date[t]`
avec l'information disponible jusqu'à `date[t-1]` inclus (walk-forward pur, cf. `model_artifacts/pipeline.py`).
Le mapping D / D+1 **doit** donc être :

```
Pour chaque ligne t (t >= 1) du log OOS trié par date croissante :
    D              = date[t-1]
    D+1            = date[t]
    reference_price(D)  = actual[t-1]      # connu à D, AUCUN futur
    predicted(D+1)      = predicted[t]     # sortie modèle à D pour D+1
    PI_low(D+1)         = pi_lower[t]
    PI_high(D+1)        = pi_upper[t]
    realized_price(D+1) = actual[t]        # révélé seulement à D+1
```

> ⚠️ **Piège** : utiliser `actual[t]` comme prix de référence (au lieu de `actual[t-1]`) injecte du futur
> et fait exploser artificiellement les KPIs. Un test unitaire dédié doit verrouiller cet alignement
> (cf. §8, `test_no_lookahead`).

En **live** (`tracking.db`), l'alignement est déjà propre par construction : `last_close` = prix à D,
`y_true` = prix résolu à `target_date`. Aucun décalage à gérer.

---

## 3. Signal — décision à D

```
signal_valid  ⇔  predicted(D+1) > reference_price(D)  ET  reference_price(D) ≥ PI_low(D+1)
```

- `signal_valid = True`  → **valid buy signal** : on prend une position **longue** à D (au close de D, à `reference_price`).
- `signal_valid = False` → **flat** : aucun trade, la ligne ne génère pas de `sim_trade` (mais reste comptée dans `N_total`).

**Garde-fou d'étanchéité `reference_price ≥ PI_low` (taxonomie TC1.1–TC1.5).** Sans lui, `predicted > ref`
inclut les journées où `ref < PI_low`, qui relèvent en fait de **TC1.2 (bull stress)** : TC1.1 et TC1.2
compteraient alors deux fois les mêmes journées. Chaque jour doit tomber dans **une seule** case (cf. §3bis).
Impact mesuré sur le backtest OOS : 6 896 → **6 613 signaux** (les 283 journées retirées = candidats bull stress).

**Variante plus stricte `pi95_conf` = TC1.2 (bull stress).** Signal `pi_lower(D+1) > reference_price(D)`,
c.-à-d. `P(D) < PI_low` : toute la bande prédite est au-dessus d'aujourd'hui, hausse quasi-certaine même
au pire bas de l'intervalle. Déjà implémentée comme règle sœur dans `sim_trades.py` ; reste à la promouvoir
en test case à part entière avec son propre reporting.

---

## 3bis. Taxonomie des test cases TC1.1–TC1.5 (position de P(D) vs bande prédite)

Les cinq test cases partitionnent **une seule** grandeur : où se situe `P(D)` par rapport à
`[PI_low, PI_high]` et à `PI_mid = predicted`. Plus `P(D)` est bas sous la bande, plus c'est haussier ;
plus il est haut au-dessus, plus c'est baissier ; au centre, c'est plat. Partition étanche et exhaustive :

| Test case | Condition sur P(D) (connue à D) | Stratégie | Statut |
|---|---|---|---|
| **TC1.2 Bull stress** | `P(D) < PI_low` | long forte conviction | règle `pi95_conf` codée, à promouvoir |
| **TC1.1 Bull calm** | `PI_low ≤ P(D) < PI_mid` | long léger | ✅ implémenté (`bull_calm_d1`) |
| **TC1.5 Sideways** | `P(D) ≈ PI_mid` (`|PI_mid − P(D)| < ε`) | range / justesse | à concevoir (cf. note ROI) |
| **TC1.3 Bear calm** | `PI_mid < P(D) ≤ PI_high` | short léger | à faire (miroir de TC1.1) |
| **TC1.4 Bear stress** | `P(D) > PI_high` | short forte conviction | à faire (miroir de TC1.2) |

**Bear = miroir du bull** : même mécanique de counter, signes inversés (ROI = `(P(D) − realized)/P(D)`,
+2 si `realized < PI_low`, +1 si `realized < P(D)`, −1 si `P(D) ≤ realized ≤ PI_high`, −2 si `realized > PI_high`).

**Sideways = cas non directionnel.** Sur actions/sous-jacents (pas d'options dans ce benchmark), une prédiction
« pas de mouvement » n'a pas de P&L directionnel exploitable, et le mean-reversion intra-bande exigerait de
l'OHLC intraday (indisponible, close quotidien seul). Décision de design ouverte : **justesse pure**
(counter seulement, `realized` proche de `P(D)` / resté dans la bande, sans ROI) ou **ROI « coût d'opportunité
évité »**. Reco : justesse pure en v1.

---

## 4. Résolution — décision à D+1 (branches évaluées **dans l'ordre**)

Pour chaque signal valide, une fois `realized_price(D+1)` connu, on évalue les branches **de haut en bas**
et on s'arrête à la première vraie. `ref` = `reference_price(D)`.

| # | Condition (dans l'ordre) | Issue | `counter` | `roi` (ex-post) |
|---|---|---|---|---|
| 1 | `realized > PI_high` | take profit **plafonné à PI_high** | **+2** | `(PI_high − ref) / ref` |
| 2 | `realized > ref` | take profit | **+1** | `(realized − ref) / ref` |
| 3 | `PI_low ≤ realized ≤ ref` | take loss (perte contenue dans le PI) | **−1** | `(realized − ref) / ref` |
| 4 | `realized < PI_low` | stop-loss **à PI_low** (voir raffinement OHLC) | **−2** | `(realized − ref) / ref` en daily close |

**Exhaustivité** : les 4 branches couvrent tout l'axe réel sans trou ni recouvrement
(`realized>PIhigh` | `ref<realized≤PIhigh` | `PIlow≤realized≤ref` | `realized<PIlow`),
sachant que sur un signal valide `predicted>ref` mais **rien ne garantit** `PI_high>ref`
(cas dégénéré traité en §6).

### Logique du barème (pourquoi ces points)

Le `counter` note **conjointement la direction ET la qualité de l'intervalle** :

- **+2** : le marché a dépassé notre propre borne haute → le PI a *sous-promis*, la hausse était réelle et forte. Meilleur cas.
- **+1** : hausse dans l'intervalle, exactement le scénario visé. Bon.
- **−1** : baisse, **mais** le PI l'avait « couverte » (`realized` reste dans `[PI_low, PI_high]`) → mauvaise direction, intervalle honnête.
- **−2** : le prix casse **sous** la borne basse → l'intervalle 95 % a échoué *et* on perd. Pire cas.

### ROI ex-post — conventions

- Position **long only**, taille 1 unité de notionnel, pas de levier.
- Branche 1 : on suppose un ordre **take-profit** posé à `PI_high` → on capture au plus `PI_high`, d'où le plafond.
- Branche 4 : on suppose un **stop-loss** à `PI_low`. En données *daily close* on ne connaît que le close ;
  par prudence le ROI est calculé sur `realized` (close) au *cut-off*. **Raffinement OHLC** (optionnel, si low/high dispo) :
  si `low(D+1) ≤ PI_low`, le stop est réputé rempli à `PI_low` → `roi = (PI_low − ref)/ref` (meilleure fidélité).
- **Coûts de transaction** : à `0` en v1 (à paramétrer `fee_bps` plus tard ; l'ignorer surestime le ROI).
- `roi` est un **rendement simple par trade** ; le cumul se fait en somme (approx.) et en composé (`Π(1+roi) − 1`) — les deux reportés.

---

## 5. Exemple chiffré réel (Prophet · SPY · D1, extrait OOS)

Vérifié sur `Run/…-Prophet-SPY-D1/predictions.parquet` (règle appliquée telle quelle) :

| D+1 | ref=P(D) | predicted | PI_low | PI_high | realized | branche | counter | ROI % |
|---|---|---|---|---|---|---|---|---|
| 2026-02-02 | 688.31 | 691.94 | 677.28 | 706.21 | 691.73 | 2 | +1 | +0.497 |
| 2026-02-03 | 691.73 | 692.63 | 677.77 | 706.85 | 685.89 | 3 | −1 | −0.846 |
| 2026-02-05 | 682.56 | 693.90 | 679.78 | 708.40 | 674.04 | 4 | −2 | −1.249 |
| 2026-02-06 | 674.04 | 694.69 | 679.65 | 710.26 | 686.97 | 2 | +1 | +1.918 |

**Agrégat du combo** : 59 signaux valides / 112 jours · counter Σ = −14 (moy. −0,24) ·
taux de réalisation 49,2 % · ROI cumulé +2,78 % · distribution {+1: 29, −1: 17, −2: 13}.
Résultat cohérent avec `metrics.json` (`skill_vs_naive = "worse than naive"`) : **la règle
sert précisément à révéler ce genre de combo peu tradable**.

---

## 6. Edge cases & règles de robustesse

1. **Première ligne OOS (`t=0`)** : pas de `t-1`, donc pas de `reference_price` → ignorée (jamais de trade).
2. **`predicted > ref` mais `PI_high ≤ ref`** (intervalle incohérent avec la prévision ponctuelle) :
   la prévision est signalée `verdict_integrite = 0` en amont ; en Bull-Calm on **flag** `degenerate_pi = True`,
   on exécute quand même la règle (les branches restent exhaustives) mais on **exclut** ces trades des KPIs par défaut.
3. **NaN / bornes cassées** (`pi_lower > pi_upper`, valeurs non finies) : ligne exclue, comptée dans `n_dropped`.
4. **`realized` exactement sur une frontière** : inégalités choisies pour ne jamais laisser de trou —
   `> PI_high` strict (sinon branche 2), `≥ PI_low` inclusif en branche 3 (sinon branche 4). À tester explicitement.
5. **Jour non résolu en live** (`y_true IS NULL`) : le trade reste `status = "open"`, non compté dans les KPIs
   tant que `evaluate_daily.py` ne l'a pas résolu. Idempotent.
6. **Gap de calendrier (week-end/fériés)** : « D+1 » = jour de *trading* suivant, pas J+1 calendaire — respecté
   automatiquement car on itère sur les lignes du log (déjà en jours de bourse).

---

## 7. Modèle de données à implémenter

### 7.1 `daily_oos_log` (vue normalisée alimentant la simulation)

Une ligne = un couple (D → D+1) prêt à simuler, dérivé de `predictions.parquet` avec l'alignement du §2.

| Colonne | Type | Source |
|---|---|---|
| `run_id` | TEXT | dossier `Run/` |
| `model`, `asset`, `horizon` | TEXT/TEXT/INT | `metadata.json` |
| `regime` | TEXT | `business_validation.json` (`calm/bull/bear/stress`) |
| `d_date` | TEXT | `date[t-1]` |
| `target_date` | TEXT | `date[t]` |
| `reference_price` | REAL | `actual[t-1]` |
| `predicted` | REAL | `predicted[t]` |
| `pi_lower`, `pi_upper` | REAL | `pi_lower[t]`, `pi_upper[t]` |
| `realized_price` | REAL | `actual[t]` (NULL si non encore connu en live) |
| `source` | TEXT | `"oos"` ou `"live"` |

### 7.2 `sim_trades` (un trade simulé par signal valide)

| Colonne | Type | Définition |
|---|---|---|
| `id` | INTEGER PK | |
| `rule_version` | TEXT | `"bull_calm_d1"` |
| `run_id`, `model`, `asset`, `horizon`, `regime`, `source` | — | hérités du log |
| `d_date`, `target_date` | TEXT | |
| `reference_price`, `predicted`, `pi_lower`, `pi_upper`, `realized_price` | REAL | snapshot au moment du signal |
| `signal_valid` | INT | 1 (par déf. on ne stocke que les signaux valides ; les flats vivent dans `daily_oos_log`) |
| `direction_ok` | INT | `realized_price > reference_price` |
| `branch` | INT | 1..4 |
| `counter` | INT | −2 / −1 / +1 / +2 |
| `roi` | REAL | rendement simple ex-post |
| `degenerate_pi` | INT | 1 si `pi_upper ≤ reference_price` (exclu des KPIs par défaut) |
| `status` | TEXT | `"open"` (live non résolu) / `"closed"` |
| `created_at`, `evaluated_at` | TEXT | horodatage |

> Séparer strictement `source = "oos"` (validation statistique) et `source = "live"` (forward-test honnête)
> dans **tous** les KPIs. Ne jamais les agréger ensemble.

---

## 8. Pseudo-code de la règle (`bull_calm_d1`)

```python
def bull_calm_d1(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0):
    """Retourne (signal_valid, branch, counter, roi, degenerate_pi) pour un couple D->D+1.
    Toutes les entrées sont des prix au close ; realized peut être None (live non résolu)."""
    degenerate_pi = int(pi_high <= ref)

    # --- Signal à D ---
    signal_valid = predicted > ref
    if not signal_valid:
        return (False, None, 0, 0.0, degenerate_pi)   # flat

    if realized is None:
        return (True, None, None, None, degenerate_pi) # signal ouvert, non résolu

    # --- Résolution à D+1 : branches DANS L'ORDRE ---
    if realized > pi_high:                 # 1
        branch, counter, exit_px = 1, +2, pi_high
    elif realized > ref:                   # 2
        branch, counter, exit_px = 2, +1, realized
    elif realized >= pi_low:               # 3   (PI_low <= realized <= ref)
        branch, counter, exit_px = 3, -1, realized
    else:                                  # 4   (realized < PI_low) -> stop à PI_low
        branch, counter, exit_px = 4, -2, pi_low   # (v1 daily-close: exit_px=realized ; OHLC: pi_low)

    roi = (exit_px - ref) / ref - fee_bps / 1e4
    return (True, branch, counter, roi, degenerate_pi)
```

> Décision v1 assumée : en branche 4, `exit_px = realized` (daily close, prudent) tant que l'OHLC n'est pas
> câblé ; passer à `pi_low` uniquement avec le raffinement intraday. À trancher avec le tuteur (§11).

---

## 9. KPIs (par `asset × model × regime`, et agrégé — séparément OOS / live)

Tous calculés sur les **signaux valides non dégénérés** sauf mention contraire.

1. **N signaux / N flat / N total** — volume d'activité de la règle ; `taux_signal = N_signaux / N_total`.
2. **Précision de direction** — `mean(direction_ok)` = % des signaux où `realized > ref` (recoupe `direction_correct`).
3. **Taux de réalisation** — `mean(counter >= +1)` : la prédiction 95 % a-t-elle *effectivement* permis la stratégie Bull.
4. **Counter TC1** — `sum(counter)` et `mean(counter)` ; distribution des 4 branches (mix +2/+1/−1/−2).
5. **Counter ROI** — ROI cumulé simple `sum(roi)`, ROI composé `Π(1+roi) − 1`, ROI moyen/trade `mean(roi)`.
6. **Qualité risque-ajustée** — hit ratio, ROI médian, pire trade (min roi), et si assez de points : Sharpe ex-post
   `mean(roi)/std(roi) * sqrt(252)` (indicatif, à interpréter avec prudence sur petits N).
7. **Couverture PI** (réutilise l'existant `pi_coverage_95`) — % de `realized ∈ [pi_low, pi_high]` : sert de garde-fou
   (un counter −2 fréquent doit coïncider avec une couverture < 95 %).
8. **Benchmark naïf** — même règle appliquée à un prédicteur naïf (`predicted = ref`, PI = bandes vol √t) →
   Bull-Calm n'a de valeur que s'il **bat le naïf** (cohérent avec `skill_vs_naive` de `metrics.json`).

**Garde-fou d'interprétation** : sur un combo `worse than naive`, on *attend* un counter moyen négatif.
Un counter fortement positif sur un tel combo = suspicion de bug (probable look-ahead) → investiguer avant de célébrer.

---

## 10. Plan de tests unitaires (bloquants avant merge)

| Test | Ce qu'il verrouille |
|---|---|
| `test_branches_exhaustives` | les 4 branches couvrent tout l'axe, une seule vraie par cas |
| `test_frontieres` | `realized == PI_high`, `== ref`, `== PI_low` tombent dans la bonne branche |
| `test_counter_values` | mapping strict branche→counter (+2/+1/−1/−2) |
| `test_roi_formulas` | ROI exact pour chaque branche (dont plafond PI_high et stop PI_low) |
| `test_no_lookahead` | `reference_price == actual[t-1]` ; un swap sur `actual[t]` doit faire échouer le test |
| `test_signal_flat` | `predicted <= ref` → pas de trade, compté en N_total |
| `test_degenerate_pi` | `pi_high <= ref` → flag levé, exclu des KPIs par défaut |
| `test_live_open` | `realized is None` → `status="open"`, non compté dans les KPIs |
| `test_idempotent_resolution` | rejouer `evaluate_daily` ne double pas les trades |
| `test_vs_naive` | la règle sur prédicteur naïf produit un baseline reproductible |

Reproductibilité : `seed` fixe (déjà 42 dans `metadata.json`), snapshot des prix dans `sim_trades` (pas de
re-téléchargement à la volée pour les KPIs OOS).

---

## 11. Points ouverts à trancher avec le tuteur

1. **Branche 4 en v1** : ROI au close réalisé (prudent) *ou* au stop `PI_low` (optimiste sans OHLC) ?
   Reco : close réalisé en v1, `PI_low` seulement avec OHLC intraday.
2. **Coûts de transaction** : `fee_bps` = 0 en v1 ? Reco : exposer le paramètre dès maintenant, défaut 0, tester à 5–10 bps.
3. **`pi95_conf`** : la livrer en règle sœur dès la v1 pour comparaison, ou plus tard ? Reco : dès la v1 (coût marginal ~nul).
4. **Périmètre live vs OOS** : confirmer qu'on garde les deux séparés et qu'on **valide sur OOS avant** de communiquer
   des chiffres live (n encore trop faible aujourd'hui).
5. **Pooling** : agréger les KPIs sur tous les `asset × model` (puissance stat) tout en gardant le détail par combo ?
   Reco : oui, reporter les deux niveaux.

---

## 12. Livrables d'implémentation attendus

- `validation/sim_trades.py` (ou module dédié) : construction `daily_oos_log`, application `bull_calm_d1`, écriture `sim_trades`.
- Migration `tracking.db` : tables `daily_oos_log` et `sim_trades`.
- Branchement dans `evaluate_daily.py` : résoudre les `sim_trades` `open` en même temps que les prédictions.
- Rapport KPIs (réutiliser le style `rapport_validation` / dashboard existant), séparé OOS / live.
- Suite de tests du §10, verte, dans la CI (`.github/workflows`).
