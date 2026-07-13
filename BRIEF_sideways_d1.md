# BRIEF — Sideways à D+1 (`sideways_d1`)

> **Statut** : spec à implémenter. 5ᵉ test case de la famille TC1 (position de `P(D)` vs bande prédite à D+1).
> **Objet** : mesurer si une prédiction Price Interval 95 % sait **identifier une journée calme** — c.-à-d. un jour
> où le modèle annonce « pas de mouvement » et où le marché lui donne raison.
> **Rule version** : `sideways_d1`.
> **Différence clé avec Bull-Calm / Bear** : le Sideways n'est **pas directionnel**. On ne prend pas de position
> longue ou courte. Sur un sous-jacent (actions, pas d'options dans ce benchmark), une prédiction « ça ne bouge
> pas » n'a aucun P&L directionnel exploitable → **pas de ROI**. C'est un **test de justesse pur** : le counter
> note uniquement la qualité de la prédiction de stabilité. Décision validée avec le tuteur.

---

## 0. Place dans la taxonomie TC1

Les 5 test cases partitionnent une seule grandeur : où se situe `P(D)` par rapport à `[PI_low, PI_high]` et à
`PI_mid = predicted`. Sideways est la **tranche centrale** : le prix d'aujourd'hui est au milieu de la bande, le
modèle prédit un mouvement négligeable.

| Test case | Condition sur P(D) | Nature | ROI |
|---|---|---|---|
| Bull-Stress | `P(D) < PI_low` | long forte conviction | oui |
| Bull-Calm | `PI_low ≤ P(D) < PI_mid − ε` | long léger | oui |
| **Sideways** | **`|PI_mid − P(D)| ≤ ε` (et P(D) dans la bande)** | **range / stabilité** | **non** |
| Bear-Calm | `PI_mid + ε < P(D) ≤ PI_high` | short léger | oui |
| Bear-Stress | `P(D) > PI_high` | short forte conviction | oui |

> **Note d'étanchéité** : `ε` est la **zone morte** qui sépare Sideways de Bull-Calm et Bear-Calm. Pour une
> partition parfaite quand les 5 cases coexisteront, il faudra durcir Bull-Calm en `predicted > P(D) + ε`
> (et Bear-Calm en `predicted < P(D) − ε`) avec **le même ε**. À faire lors de l'intégration des 5 cases
> (cf. §11). Aujourd'hui Bull-Calm utilise `predicted > P(D)` : léger recouvrement avec Sideways sur la bordure,
> à résorber à ce moment-là.

---

## 1. Vocabulaire & données

Réutilise l'existant (cf. `BRIEF_bull_calm_d1.md` §1-2). Rappels :

- `reference_price` = `P(D)` = dernier close connu à D (`actual[t-1]` en OOS, `last_close` en live).
- `predicted` = `PI_mid`, `pi_lower` = `PI_low`, `pi_upper` = `PI_high` (bornes conformal 95 % pour D+1).
- `realized_price` = `P(D+1)` = vrai close à D+1 (`actual[t]` en OOS, `y_true` en live).
- **`W = pi_upper − pi_lower`** = largeur de bande (proxy de la volatilité anticipée par le modèle).

**Alignement anti look-ahead identique** au Bull-Calm : `reference_price = actual[t-1]`, jamais `actual[t]`.

---

## 2. Signal — décision à D (« journée plate anticipée »)

```
ε = k · W                         # k = 0.10 par défaut (hyperparamètre, cf. §9 sensibilité)

signal_sideways  ⇔  (pi_lower ≤ P(D) ≤ pi_upper)      # P(D) dans la bande (sinon = cas stress)
                AND  (|predicted − P(D)| ≤ ε)          # mouvement prédit négligeable
```

- `signal_sideways = True` → le modèle annonce une **journée calme**. On **ne prend aucune position** ; on
  enregistre un « trade » de simulation *sans exposition* dont le seul rôle est de porter le `counter` de justesse.
- `signal_sideways = False` → la ligne n'est pas un jour Sideways (elle relève d'un autre test case). Comptée
  dans `N_total`, pas dans les signaux Sideways.

`ε` est **relatif à la largeur de bande** : le seuil se recalibre par actif et par régime (bande large en crypto,
étroite en obligation). Un seul paramètre `k` pilote toute la sélectivité du test.

---

## 3. Résolution — décision à D+1 (counter symétrique, **pas de ROI**)

La thèse du Sideways est « ça reste stable ». Le succès = le prix **reste dans la bande** et, idéalement, **proche
de `P(D)`**. L'échec = un **breakout** (dans un sens *ou* l'autre — le counter est **symétrique**). On évalue dans
l'ordre :

| # | Condition à D+1 (`m = W/4`, `h = W/2`) | Interprétation | `counter` | `roi` |
|---|---|---|---|---|
| 1 | `realized ∈ [PI_low, PI_high]` **et** `|realized − P(D)| ≤ m` | quasi immobile — prédiction de stabilité excellente | **+2** | `NULL` |
| 2 | `realized ∈ [PI_low, PI_high]` (hors zone 1) | resté dans la bande 95 % (l'IC a tenu) | **+1** | `NULL` |
| 3 | `realized` hors bande, à ≤ `h` de la borne la plus proche | petit breakout | **−1** | `NULL` |
| 4 | `realized` hors bande, à > `h` de la borne la plus proche | gros breakout (l'IC a explosé) | **−2** | `NULL` |

**Exhaustivité** : zones 1-2 couvrent tout `[PI_low, PI_high]` ; zones 3-4 couvrent tout l'extérieur. Aucun trou,
aucun recouvrement. **Symétrie** : un breakout haussier et un breakout baissier de même ampleur donnent le même
counter (contrairement au Bull-Calm, orienté). `m = W/4` (cœur de bande) et `h = W/2` (marge de breakout) sont des
paramètres exposés, valeurs par défaut ci-dessus.

> `roi` est **toujours `NULL`** pour `sideways_d1`. `direction_ok` n'a pas de sens ici (pas de direction pariée) →
> stocker `NULL` également. Le champ `in_band = int(pi_low ≤ realized ≤ pi_upper)` est ajouté/réutilisé comme
> mesure principale.

---

## 4. Exemple mental

`P(D) = 100`, bande `[96, 104]` → `W = 8`, `PI_mid = 100.3`, `ε = 0.10·8 = 0.8`. Comme `|100.3 − 100| = 0.3 ≤ 0.8`
et `P(D)` est dans la bande → **jour Sideways**. À D+1 : `m = 2`, `h = 4`.
- `realized = 100.5` → dans `[96,104]` et `|0.5| ≤ 2` → **+2** (immobile).
- `realized = 103` → dans la bande mais hors cœur → **+1**.
- `realized = 106` → hors bande, à `2` de la borne `104` (≤ `h=4`) → **−1**.
- `realized = 111` → hors bande, à `7` de `104` (> `4`) → **−2**.

---

## 5. Edge cases & robustesse

1. **PI dégénéré** (`W ≤ 0`, `pi_lower > pi_upper`, valeurs non finies) : ligne exclue, comptée dans `n_dropped`.
2. **`P(D)` hors bande** : par construction ce n'est pas un jour Sideways (garde-fou `pi_low ≤ P(D) ≤ pi_upper`),
   même si `|predicted − P(D)|` était petit — cohérent, car P(D) hors bande = cas stress.
3. **`realized` exactement sur une borne** : inclusif → compté *dans* la bande (zone 1/2).
4. **Live non résolu** (`y_true IS NULL`) : `status = "open"`, hors KPIs jusqu'à résolution. Idempotent.
5. **Première ligne OOS (`t=0`)** : ignorée (pas de `t-1`).
6. **`k` trop grand** : le bucket Sideways avale des jours franchement directionnels → surveiller `taux_signal`
   (cf. §9). `k` trop petit : bucket quasi vide, stats non significatives.

---

## 6. Modèle de données

Réutilise `daily_oos_log` et `sim_trades` (cf. `BRIEF_bull_calm_d1.md` §7) avec `rule_version = "sideways_d1"`.
Adaptations :

- `roi` : **NULL** pour toutes les lignes Sideways (colonne déjà nullable).
- `direction_ok` : **NULL** (non pertinent).
- `in_band` : `INTEGER` (0/1) — ajouter la colonne si absente, sinon réutiliser. Mesure clé du test.
- `branch` ∈ {1,2,3,4}, `counter` ∈ {+2,+1,−1,−2} comme d'habitude.
- Contrainte d'unicité inchangée (`rule_version, source, run_id, model, asset, horizon, d_date`) → idempotence.

Aucune migration de la vraie `tracking.db` requise (pas de données live persistées sous ce nom).

---

## 7. Pseudo-code de la règle (`sideways_d1`)

```python
def sideways_d1(ref, predicted, pi_low, pi_high, realized, k=0.10, m_frac=0.25, h_frac=0.50):
    """Test de justesse d'une journée plate. Retourne
    (signal_sideways, branch, counter, roi, in_band, degenerate_pi).
    roi est TOUJOURS None (pas de position directionnelle). realized peut être None (live)."""
    W = pi_high - pi_low
    degenerate_pi = int(W <= 0)

    # --- Signal à D : P(D) dans la bande ET mouvement prédit négligeable ---
    eps = k * W
    signal = (pi_low <= ref <= pi_high) and (abs(predicted - ref) <= eps)
    if not signal:
        return False, None, 0, None, None, degenerate_pi

    if realized is None:
        return True, None, None, None, None, degenerate_pi   # jour plat, non résolu

    # --- Résolution à D+1 : counter symétrique, pas de ROI ---
    in_band = int(pi_low <= realized <= pi_high)
    m, h = m_frac * W, h_frac * W
    if in_band and abs(realized - ref) <= m:      # 1 : quasi immobile
        branch, counter = 1, +2
    elif in_band:                                  # 2 : resté dans la bande
        branch, counter = 2, +1
    else:
        dist = (pi_low - realized) if realized < pi_low else (realized - pi_high)
        if dist <= h:                              # 3 : petit breakout
            branch, counter = 3, -1
        else:                                      # 4 : gros breakout
            branch, counter = 4, -2

    return True, branch, counter, None, in_band, degenerate_pi
```

---

## 8. KPIs (par `asset × model × regime`, et agrégé — OOS / live séparés)

Pas de ROI → les KPIs portent sur la **justesse de la prédiction de stabilité** :

1. **N Sideways / N total / `taux_signal`** — quelle part des jours le modèle classe « plats ». Garde-fou de calibration de `k`.
2. **Taux de justesse** — `mean(counter ≥ +1)` = % de jours plats prédits qui **restent dans la bande**.
3. **Taux « immobile »** — `mean(counter == +2)` = prédiction de stabilité au sens fort.
4. **Taux de breakout** — `mean(counter < 0)`, **décomposé haussier vs baissier** (pour détecter un biais directionnel du bucket « plat »).
5. **Counter TC** — `sum(counter)` et `mean(counter)` ; distribution des 4 branches.
6. **Couverture PI (`in_band`)** — `mean(in_band)` sur les jours Sideways ; doit rester cohérent avec `pi_coverage_95` global (garde-fou).
7. **Sensibilité à `k`** — recalculer 1-6 pour `k ∈ {0.05, 0.10, 0.15, 0.20}` ; reporter la courbe `taux_signal` vs `k` et la stabilité du taux de justesse. Sert à figer `k`.

---

## 9. Plan de tests unitaires (bloquants avant merge)

| Test | Ce qu'il verrouille |
|---|---|
| `test_sideways_signal_flat_vs_directional` | `|predicted − ref| ≤ k·W` et `ref` dans la bande ⇒ signal ; sinon pas de signal |
| `test_sideways_signal_requires_ref_in_band` | `ref` hors `[pi_low, pi_high]` ⇒ jamais Sideways, même si `|predicted−ref|` petit |
| `test_sideways_branches_exhaustives` | les 4 zones couvrent tout l'axe, une seule vraie par cas |
| `test_sideways_frontieres` | `realized` sur `pi_low`/`pi_high` ⇒ in-band ; sur `m` et `h` ⇒ bonne zone |
| `test_sideways_counter_values` | mapping branche→counter (+2/+1/−1/−2) |
| `test_sideways_symmetry` | breakout haussier et baissier de même ampleur ⇒ même counter |
| `test_sideways_roi_is_none` | `roi is None` sur toutes les branches (test de justesse pur) |
| `test_sideways_direction_ok_is_none` | `direction_ok` non renseigné |
| `test_sideways_no_lookahead` | `reference_price == actual[t-1]` (réutilise le verrou Bull-Calm) |
| `test_sideways_live_open` | `realized is None` ⇒ `status="open"`, hors KPIs |
| `test_sideways_degenerate_pi` | `W ≤ 0` ⇒ flag + exclusion |
| `test_sideways_k_sensitivity` | faire varier `k` change le nombre de signaux de façon monotone |

---

## 10. Livrables d'implémentation

- `validation/sim_trades.py` : fonction `sideways_d1(...)` + branchement dans le dispatch `rule_version → fonction` ;
  gestion `roi = NULL` et colonne `in_band` dans `generate_sim_trades` / `sync_live_trades`.
- Migration légère `sim_trades` : colonne `in_band` (lazy, comme le reste).
- `kpi_report` : variante « justesse » pour `rule_version = "sideways_d1"` (KPIs §8, pas de ROI) + option de balayage `k`.
- `validation/test_sim_trades.py` : la suite du §9, verte, écrite **avant** l'implémentation (rouge d'abord).
- Reporting : réutiliser le style existant, source OOS et live séparées.

---

## 11. Points ouverts à trancher

1. **`k` par défaut** : 0.10 proposé ; à figer après le balayage de sensibilité (§8.7) sur données réelles.
   *Reco* : viser un `taux_signal` Sideways de l'ordre de 15-30 %.
2. **`m` (cœur « immobile ») et `h` (marge de breakout)** : 0.25·W et 0.50·W proposés ; ajustables selon la
   distribution observée des counters (éviter que +2 ou −2 soit vide ou dominant).
3. **Étanchéité complète** : quand les 5 cases coexisteront, appliquer la zone morte `ε` à Bull-Calm et Bear-Calm
   (cf. §0) — petit refactor à planifier, hors périmètre de ce brief.
4. **ROI « coût d'opportunité »** : écarté en v1 (choix « justesse pure »). Réévaluable plus tard si le tuteur veut
   une métrique quasi-P&L, mais ce serait un regret, pas un vrai rendement.
