# BRIEF — Sideways v2 : régime, volatilité et P&L short-vol (`sideways_gated_d1`)

> **Statut** : implémenté (130 tests verts) — ce brief reste la spec de référence. Extension de `sideways_d1`
> (cf. `BRIEF_sideways_d1.md`), pas un remplacement.
> **Objet** : passer d'un **test de justesse pur** (v1, `roi = NULL`) à un **signal de volatilité actionnable** —
> conditionné au régime/volatilité et assorti d'un **P&L short-vol honnête** — pour répondre à la question
> « peut-on *trader* une journée plate ? ».
> **Rule versions livrées** : `sideways_gated_d1` (nouvelle) ; `sideways_d1` reste **inchangée** (compat & baseline).
> **Décision de conception (quant senior)** : cf. §0. Elle est le cœur de ce brief — la lire avant de coder.

---

## 0. La décision de conception (à valider avec le tuteur)

Le brief v1 (§11.4) a écarté le ROI en disant : « pas d'options → une prédiction *ça ne bouge pas* n'a aucun P&L
directionnel exploitable ». C'est **vrai directionnellement, mais incomplet**. Une prédiction Sideways n'est pas
une absence de trade : c'est une **thèse short-volatilité / mean-reversion**. Le trade canonique du jour plat est
le **short straddle** (vendre le mouvement). Un quant senior ne dit donc pas « pas de trade » ; il dit trois choses :

**(A) On ne fabrique pas un ROI directionnel.** Le repo a une charte d'honnêteté explicite (`honest_benchmark.py`,
`honest_eval/`, verrous anti-look-ahead, « test de justesse pur »). Inventer un range-trade *intraday* sur des
barres *daily* — fade `pi_low`/`pi_high` puis sortie `pi_mid` — supposerait une granularité **que le benchmark n'a
pas** (un seul pas D→D+1, et au moment du signal `P(D)` est *déjà* au centre de la bande : aucune dislocation à
fader à l'entrée). **Rejeté** : ce serait un P&L fictif.

**(B) On monétise la seule chose réellement présente : le mouvement réalisé vs la bande.** Le P&L honnête d'un jour
plat est le **payoff d'un short straddle dont les breakevens sont les bornes du PI**. C'est calculable exactement
avec ce qu'on a déjà (`ref`, `W`, `realized`), c'est borné, moyennable, Sharpe-able, et **explicitement étiqueté
comme proxy d'évaluation** (« ce qu'aurait rapporté un straddle vendu sur la bande »), pas comme un rendement
actions exécuté. C'est le sens honnête de « trader le flat day ». Voir §3.

**(C) La vraie valeur quant est dans le conditionnement au régime.** Le défaut n°1 de la v1 est de traiter *toutes*
les prédictions plates à l'identique. Or une bande étroite annoncée en régime **stress / haute vol** est un
*calm-before-the-storm* : la prédiction de calme est la moins fiable là où elle coûte le plus cher. On enrichit
donc chaque signal de son `RegimeState` (déjà produit par `calibration/regime/`) et on **gâche le signal** quand le
régime le contredit. Voir §2 et §4.

> **En une phrase** : Sideways v2 = *signal plat gâté par le régime/volatilité* + *P&L short-vol honnête et borné* +
> *KPI de justesse tranchées par `vol_bucket × regime`*. On rend le flat day **actionnable et mesurable
> économiquement**, sans jamais fabriquer un rendement qu'on n'aurait pas pu capturer.

> **⚠️ Constat post-implémentation (backtest OOS)** : le proxy de vol OOS (terciles de `W`, §1) s'avère **confondu
> avec la largeur de bande** — les jours écartés par le gate (bandes larges) sur-couvrent trivialement et affichent
> une justesse/P&L *supérieurs* aux jours tradés. Le gate OOS n'ajoute donc **pas** de valeur sur backtest : il
> retire les jours faciles, pas les risqués. **Ne pas figer `vb_max`/`stress_max` sur l'OOS.** L'hypothèse du gate
> ne se teste et ne se calibre qu'avec le vrai `RegimeState` (vol GARCH *réalisée*, live). Cf. `sideways_v2_recap.pdf`.

---

## 1. Vocabulaire & données (rappels + ajouts)

Réutilise `BRIEF_sideways_d1.md` §1. Rappels : `ref = P(D)`, `predicted = PI_mid`, `[pi_low, pi_high]` PI 95 %,
`realized = P(D+1)`, `W = pi_high − pi_low` (proxy de vol anticipée), alignement anti look-ahead `ref = actual[t-1]`.

**Ajouts v2 — le contexte de régime, connu à D (jamais à D+1) :**

- `vol_bucket ∈ {0,1,2}` — 0 vol faible, 2 vol élevée. **Source selon `source`** :
  - `source="live"` : lu depuis le `RegimeState` (`calibration/regime/regime_state.py`, terciles de σ_t GARCH),
    déjà point-in-time par construction.
  - `source="oos"` : le `RegimeState` n'est pas câblé sur l'historique du backtest (`regime="unknown"` en OOS).
    **Proxy sans look-ahead** : tercile de `W` calculé sur l'échantillon OOS du groupe (`asset × model`). `W` est
    la vol *anticipée par le modèle*, connue à D → aucun look-ahead. **Limite connue** (cf. §0) : ce proxy est
    confondu avec la largeur de bande et ne remplace pas la vol réalisée — usage diagnostique, pas de calibration.
- `stress_score ∈ [0,1]` — `= probs["stress"]` du `RegimeState`. **Live uniquement** ; `NULL` en OOS (pas de proxy
  fiable). Le gating OOS repose alors sur `vol_bucket` seul (cf. §2).
- `regime_label ∈ {calm,bull,bear,stress}` — `argmax(probs)` du `RegimeState`. Live uniquement ; `"unknown"` en OOS.

Ces trois champs sont **capturés à la génération** du sim_trade (valeur à D) et **figés** — pas recalculés à la
résolution. En live, ils passent par un point d'injection `regime_lookup(asset, d_date) -> (vol_bucket, stress_score)`
plutôt qu'un appel direct à `calibration.regime` (qui refit un HMM sur OHLCV téléchargé, incompatible avec la
contrainte « stdlib+pandas, hors-ligne » des tests). Le câblage production réel se fait côté `evaluate_daily.py`.

---

## 2. Signal v2 — jour plat *validé par le régime* (`sideways_gated_d1`)

On part du signal v1 et on ajoute un **filtre de régime** (le gate). Deux paramètres nouveaux : `vb_max`
(vol_bucket maximal accepté) et `stress_max` (stress_score maximal accepté, live).

```
signal_v1        ⇔  (pi_low ≤ P(D) ≤ pi_high)  AND  (|predicted − P(D)| ≤ k·W)     # identique v1
gate_regime      ⇔  (vol_bucket ≤ vb_max)  AND  (source != "live"  OR  stress_score ≤ stress_max)
signal_sideways  ⇔  signal_v1  AND  gate_regime
```

Défauts : `k = 0.10` (repris v1), `vb_max = 1` (on refuse la vol élevée), `stress_max = 0.30`.
**Rappel §0** : ces défauts ne sont pas à figer sur l'OOS (proxy W confondu) — calibration sur live uniquement.

Lecture : on ne compte/monétise un jour plat **que si le marché est effectivement calme** (vol basse *et*, en live,
peu de probabilité de stress). Un jour où `signal_v1=True` mais `gate_regime=False` est un **« flat suspect »** :
on l'exclut des signaux tradables, mais on le **journalise** (colonne `gated_out`, cf. §6) pour mesurer combien de
faux-plats le régime nous a évités — c'est un KPI à part entière (§8, « valeur ajoutée du gate »).

> **Étanchéité / comparabilité** : `sideways_gated_d1` et `sideways_d1` partagent *exactement* le même `signal_v1`.
> La seule différence est le gate. Générer les deux permet de mesurer directement l'effet du gate à `k` constant.

---

## 3. Résolution v2 — counter de justesse **+** P&L short-vol borné

On garde **intégralement** le counter symétrique v1 (branches 1-4, +2/+1/−1/−2 ; cf. `BRIEF_sideways_d1.md` §3) :
c'est la mesure de justesse, elle ne change pas. On **ajoute** un champ `pnl_shortvol` : le payoff, par unité de
notionnel, d'un **short straddle dont les breakevens sont les bornes du PI**.

Soit `move = |realized − ref|` et la demi-bande `Hb = W/2`. Le straddle vendu encaisse une prime ≈ `Hb` (mouvement
anticipé) et perd le mouvement réalisé au-delà :

```
pnl_shortvol_raw = (Hb − move) / Hb            # = 1 − |realized − ref| / (W/2)
pnl_shortvol     = max(pnl_shortvol_raw, -1)   # risque défini (proxy iron condor) → borné [-1, +1]
```

- `+1` : immobile parfait (`realized == ref`).
- `0` : `realized` pile sur une borne du PI (breakeven).
- `< 0` : breakout ; plancher `−1` (perte maximale = prime, version defined-risk).

**Cohérence avec les branches** (par construction, `pnl_shortvol` est monotone décroissant en `move`) :

| Branch | Condition | `move` | `counter` | `pnl_shortvol` |
|---|---|---|---|---|
| 1 | immobile | `≤ W/4` | +2 | `[+0.5, +1]` |
| 2 | in-band | `(W/4, W/2]` | +1 | `[0, +0.5)` |
| 3 | petit breakout | `(W/2, W]` | −1 | `[−1, 0)` |
| 4 | gros breakout | `> W` | −2 | `−1` (clippé) |

> `pnl_shortvol` est **stocké dans la colonne `roi`** (elle existe déjà, nullable) *uniquement* pour
> `sideways_gated_d1`. **`sideways_d1` garde `roi = NULL`** (v1 = justesse pure, on ne la contamine pas). On
> documente partout que ce `roi` est un **proxy d'évaluation short-vol**, pas un rendement actions exécuté
> (docstring + libellé KPI `pnl_shortvol_*`, jamais `roi_*` dans le reporting sideways v2).
> `direction_ok = NULL` (toujours non directionnel).

**Live non résolu** (`realized is None`) : `status="open"`, `pnl_shortvol = NULL`, hors KPIs. Idempotent.

---

## 4. Exemple mental (v2)

`P(D)=100`, bande `[96,104]` → `W=8`, `Hb=4`, `PI_mid=100.3`, `ε=0.8`. Contexte régime à D : `vol_bucket=1`,
`stress_score=0.18`. Gate : `1 ≤ vb_max=1` OK, `0.18 ≤ 0.30` OK → **signal tradable**.
- `realized=100.5` → `move=0.5`, branch 1, counter +2, `pnl_shortvol = 1 − 0.5/4 = +0.875`.
- `realized=103` → `move=3`, branch 2, counter +1, `pnl_shortvol = 1 − 3/4 = +0.25`.
- `realized=106` → `move=6`, branch 3, counter −1, `pnl_shortvol = 1 − 6/4 = −0.5`.
- `realized=111` → `move=11`, branch 4, counter −2, `pnl_shortvol = max(1 − 11/4, −1) = −1`.

Même jour mais `vol_bucket=2` (vol élevée) → `gate_regime=False` → `gated_out=1`, ligne journalisée mais **hors
signaux tradables** : le régime nous dit que ce « plat » n'est pas fiable.

---

## 5. Edge cases (en plus de ceux de v1 §5)

1. **`vol_bucket` indisponible** (proxy OOS impossible car groupe trop petit pour des terciles, < 3 lignes) :
   `gate_regime` dégénère en « pass » (on ne gâche pas faute de donnée) mais on flag `vol_bucket = NULL` et la ligne
   est comptée dans `n_gate_undefined` (transparence, ne pas maquiller en signal validé).
2. **`stress_score` NULL en OOS** : le terme stress du gate est **neutralisé** (seul `vol_bucket ≤ vb_max` s'applique).
   Documenté : le gate OOS est *plus permissif* que le gate live — à garder en tête pour comparer OOS vs live.
3. **`W ≤ 0` (PI dégénéré)** : `pnl_shortvol` non défini (division par `Hb=0`) → ligne exclue, `degenerate_pi=1`,
   `pnl_shortvol=None`, comptée dans `n_dropped` (comme v1).
4. **`pnl_shortvol` et clip** : le clip à `−1` est **intentionnel** (defined-risk). Un test verrouille que la branche 4
   donne toujours exactement `−1` quelle que soit l'ampleur du breakout.
5. **Terciles de `W` (proxy vol OOS)** : calculés **par groupe `asset × model`**, pas globalement (une bande « large »
   sur BTC n'est pas « large » sur une obligation). Recalcul en mémoire à la génération, jamais persisté comme seuil.
   **Confond avec la largeur de bande** (§0) : diagnostique uniquement, ne jamais calibrer le gate dessus.

---

## 6. Modèle de données

Réutilise `sim_trades`. `rule_version = "sideways_gated_d1"`. Colonnes **déjà présentes** réutilisées :
`roi` (← `pnl_shortvol`), `in_band`, `branch`, `counter`, `regime`, `degenerate_pi`. Colonnes **ajoutées** (migration
lazy, comme le reste du repo) :

- `vol_bucket   INTEGER` — 0/1/2 ou NULL (indisponible). Valeur à D, figée.
- `stress_score REAL`    — [0,1] ou NULL (OOS). Valeur à D, figée.
- `gated_out    INTEGER` — 1 si `signal_v1=True` mais `gate_regime=False` (flat suspect écarté), sinon 0.

Les lignes « flat suspect » (`gated_out=1`) sont **persistées** avec `signal_valid=0` — contrairement aux flats
ordinaires (`signal_v1=False`, non persistés) — car le KPI « valeur ajoutée du gate » a besoin de les compter.
Contrainte d'unicité inchangée (`rule_version, source, run_id, model, asset, horizon, d_date`) → idempotence.
`direction_ok = NULL`. `sideways_d1` (v1) **n'est pas touchée** par la migration (ses colonnes restent NULL).

---

## 7. Pseudo-code de la règle (`sideways_gated_d1`)

```python
def sideways_gated_d1(ref, predicted, pi_low, pi_high, realized,
                      vol_bucket=None, stress_score=None, source="oos",
                      k=0.10, vb_max=1, stress_max=0.30, m_frac=0.25, h_frac=0.50):
    """TC1.5b Sideways gaté — signal plat validé par le régime + P&L short-vol borné.
    Retourne (signal_sideways, branch, counter, pnl_shortvol, in_band, gated_out, degenerate_pi).
    counter = justesse v1 (inchangé). pnl_shortvol dans [-1, +1] (proxy short-straddle), None si non résolu
    ou PI dégénéré. direction_ok géré à NULL par l'adaptateur."""
    W = pi_high - pi_low
    degenerate_pi = int(W <= 0)

    # --- Signal v1 (identique à sideways_d1) ---
    eps = k * W
    signal_v1 = (pi_low <= ref <= pi_high) and (abs(predicted - ref) <= eps)
    if not signal_v1:
        return False, None, 0, None, None, 0, degenerate_pi

    # --- Gate régime/volatilité ---
    gate_vol = (vol_bucket is None) or (vol_bucket <= vb_max)         # None => permissif (n_gate_undefined)
    gate_stress = (source != "live") or (stress_score is None) or (stress_score <= stress_max)
    if not (gate_vol and gate_stress):
        return False, None, 0, None, None, 1, degenerate_pi          # flat suspect écarté

    if realized is None or degenerate_pi:
        return True, None, None, None, None, 0, degenerate_pi        # non résolu / PI dégénéré

    # --- Résolution : counter v1 + pnl short-vol ---
    in_band = int(pi_low <= realized <= pi_high)
    m, h = m_frac * W, h_frac * W
    move = abs(realized - ref)
    if in_band and move <= m:
        branch, counter = 1, 2
    elif in_band:
        branch, counter = 2, 1
    else:
        dist = (pi_low - realized) if realized < pi_low else (realized - pi_high)
        branch, counter = (3, -1) if dist <= h else (4, -2)

    pnl_shortvol = max(1.0 - move / (W / 2.0), -1.0)                  # borné [-1, +1]
    return True, branch, counter, pnl_shortvol, in_band, 0, degenerate_pi
```

Branchement dans `RULES` via un adaptateur `_adapt_sideways_gated` alignant la signature normalisée
`(signal_valid, branch, counter, roi, direction_ok, in_band, degenerate_pi)` : `roi ← pnl_shortvol`,
`direction_ok ← None`. `generate_sim_trades`/`sync_live_trades` transmettent `vol_bucket`, `stress_score`, `source`
et écrivent aussi `gated_out`.

---

## 8. KPIs (par `asset × model × regime × vol_bucket`, OOS / live séparés)

La v2 ajoute deux axes : la **coupe par volatilité/régime** et le **P&L short-vol**. Fonction dédiée
`_summarize_group_sideways_gated` (dérivée de `_summarize_group_sideways`).

1. **Justesse v1** (inchangée) : `taux_justesse = mean(counter≥+1)`, `taux_immobile = mean(counter==+2)`,
   `taux_breakout` (décomposé haussier/baissier), `in_band_coverage`, distribution des 4 branches, `counter_sum/mean`.
2. **P&L short-vol** : `pnl_shortvol_mean`, `_median`, `_sum`, `_min`, et **`sharpe_shortvol`** (`mean/std·√252`).
   → répond quantitativement à « le flat day est-il monétisable, et à quel risque ? ».
3. **Surface de fiabilité** : toutes les métriques 1-2 **tranchées par `vol_bucket`** (et par `regime_label` en live).
4. **Sharpness de bande (garde-fou anti-triche)** : `rel_width_mean = mean(W / ref)` et
   `move_ratio_mean = mean(|realized − ref| / (W/2))` sur les jours sideways. Une bande large rend la justesse
   triviale : `taux_justesse` ne se lit **jamais seul**, toujours conjointement à `rel_width_mean` (comparer les
   modèles à largeur comparable). `move_ratio_mean < 1` = la bande a capté la prime de vol ; c'est l'exact
   complément de `pnl_shortvol_mean`. Pendant « sharpness » de la calibration probabiliste.
5. **Risque de queue du P&L short-vol** : le short-vol est à **skew négatif**, le Sharpe le flatte. Reporter
   `cvar_5_shortvol` (expected shortfall à 5 %, moyenne des 5 % pires jours), `pnl_skew`, `freq_floor = mean(pnl==−1)`
   (fréquence du « rouleau compresseur »), et `calmar_shortvol = pnl_shortvol_mean / |max_drawdown|` (max drawdown de
   la série cumulée de `pnl_shortvol`). Décision d'allocation sur la queue, jamais sur la seule moyenne/Sharpe.
6. **Valeur ajoutée du gate** : comparer `sideways_gated_d1` vs `sideways_d1` à `k` égal —
   `Δ taux_justesse`, `Δ pnl_shortvol_mean`, et `n_gated_out / n_signal_v1` (part de faux-plats évités).
   Un gate utile *augmente* la justesse et le Sharpe au prix d'un volume de signaux plus faible.
   **Résultat OOS observé** : le gate n'augmente PAS ces métriques (§0, proxy W confondu) — à réévaluer sur live.
7. **Volume** : `n_total`, `n_signal_tradable`, `n_gated_out`, `n_gate_undefined`, `n_open`, `taux_signal`.
8. **Sensibilité** : réutiliser le balayage `k` existant ; **ajouter un balayage `(vb_max, stress_max)`** pour figer
   le gate (viser un compromis Sharpe/volume). Reporter la courbe `sharpe_shortvol` vs `vb_max`.

> Reporting : réutiliser le style existant (`generate_sim_trades_dashboard.py`, PDF recap). **Ne jamais** libeller le
> P&L short-vol comme `roi`/`rendement` dans les sorties : toujours `pnl_shortvol` + note « proxy short-straddle,
> non exécuté ». OOS et live séparés (le gate live est plus strict, cf. §5.2).

---

## 9. Plan de tests unitaires (bloquants, TDD — rouge d'abord)

Réutilise la suite v1 pour la partie counter (elle doit rester verte via `sideways_d1`). Nouveaux tests ciblant v2 :

| Test | Ce qu'il verrouille |
|---|---|
| `test_gated_signal_passes_when_calm` | `vol_bucket ≤ vb_max` (+ stress ok en live) ⇒ signal tradable |
| `test_gated_signal_blocked_high_vol` | `vol_bucket > vb_max` ⇒ `signal_valid=0`, `gated_out=1` (pas un simple flat) |
| `test_gated_signal_blocked_high_stress_live_only` | `stress_score > stress_max` bloque en live, **ignoré** en OOS |
| `test_gate_undefined_when_vol_bucket_none` | `vol_bucket None` ⇒ permissif + comptage `n_gate_undefined` |
| `test_counter_identical_to_v1` | à signal validé, `branch/counter` == `sideways_d1` (le gate ne change pas la justesse) |
| `test_pnl_shortvol_monotone` | `pnl_shortvol` décroît quand `move` croît ; `+1` si immobile, `0` au breakeven |
| `test_pnl_shortvol_clip_branch4` | branche 4 ⇒ `pnl_shortvol == −1` exactement, quelle que soit l'ampleur |
| `test_pnl_shortvol_none_when_open` | `realized None` ⇒ `pnl_shortvol None`, `status="open"` |
| `test_v1_roi_still_null` | `sideways_d1` conserve `roi=NULL` (non contaminée par v2) |
| `test_direction_ok_null` | `direction_ok` toujours NULL |
| `test_degenerate_pi_excluded` | `W ≤ 0` ⇒ flag + exclusion, `pnl_shortvol None`, pas de division par zéro |
| `test_vol_tercile_proxy_per_group` | terciles de `W` calculés par `asset×model`, sans look-ahead |
| `test_gate_sweep_monotone` | durcir `vb_max`/`stress_max` réduit (monotone) `n_signal_tradable` |
| `test_sharpness_flags_wide_band` | à justesse égale, une bande large donne un `rel_width_mean` plus élevé (garde-fou lisible) |
| `test_cvar_and_freq_floor` | `cvar_5_shortvol ≤ pnl_mean` ; `freq_floor == part des branches 4` ; `pnl_skew` calculé |
| `test_calmar_uses_cumulative_drawdown` | `calmar_shortvol` basé sur le max drawdown de la série cumulée, pas sur `pnl_min` |
| `test_no_lookahead` | `ref==actual[t-1]` et `vol_bucket/stress` datés à D (jamais D+1) |

---

## 10. Livrables d'implémentation

- `validation/sim_trades.py` : `sideways_gated_d1(...)` + `_adapt_sideways_gated` + entrée dans `RULES` ;
  transmission `vol_bucket/stress_score/source` et écriture `gated_out` dans `generate_sim_trades`/`sync_live_trades` ;
  calcul du proxy `vol_bucket` par terciles de `W` (OOS, `_vol_bucket_terciles`/`_vol_bucket_proxy_for_rows`) et
  point d'injection `regime_lookup` (live).
- Migration lazy `sim_trades` : colonnes `vol_bucket`, `stress_score`, `gated_out`.
- `_summarize_group_sideways_gated` + `_gate_sweep` + branchement dans `kpi_report` (`gate_values=...`) : KPIs §8,
  coupe `by_vol_bucket`, P&L short-vol, sharpness de bande, risque de queue, balayage `(vb_max, stress_max)`.
- `validation/test_sim_trades.py` : suite §9, verte, écrite **avant** l'implémentation.
- Reporting : coupe `vol_bucket × regime`, libellé `pnl_shortvol` (jamais `roi`), OOS/live séparés
  (cf. `sideways_v2_recap.pdf`).
- `sideways_d1` (v1) **strictement inchangée** : sert de baseline pour le KPI « valeur ajoutée du gate ».

---

## 11. Points ouverts à trancher (avec le tuteur)

1. **Validation de principe du P&L short-vol** : accepter un proxy short-straddle *étiqueté comme tel* (non exécuté),
   ou rester en justesse pure ? *Reco* : l'accepter — c'est la seule lecture honnête de « trader le flat day » sans
   options ni intraday, et il reste séparable de la justesse. Les KPI de queue (§8.5) exposent correctement le risque
   short-vol que le Sharpe masque.
2. **Câblage du régime réalisé** : le gate ne se calibre **que sur live** (proxy W OOS confondu, §0). Prochaine étape :
   câbler `RegimeState` (vol GARCH réalisée + `stress_score`) via `regime_lookup` dans `evaluate_daily.py`, puis figer
   `vb_max`/`stress_max` (défauts `1`/`0.30`) sur données live en visant un `sharpe_shortvol` élevé à volume raisonnable.
3. **Clip du P&L** : plancher `−1` (defined-risk / iron condor). Variante non clippée (short straddle nu) possible si
   on veut voir la queue de perte brute — à exposer en paramètre, défaut clippé.
4. **Étanchéité 5 cases** : la zone morte `ε` reste à durcir sur Bull-Calm/Bear-Calm (hérité de v1 §11.3), hors
   périmètre v2.
