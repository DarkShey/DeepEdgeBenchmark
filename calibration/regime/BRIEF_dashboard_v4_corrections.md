# Corrections v4 — Dashboard Multi-Actifs DEITA (retour du 02/07/2026, suite)

Deux ajouts indépendants de `BRIEF_dashboard_v3_corrections.md` (peuvent être appliqués sans
attendre que v3 soit fait) :

1. Test de significativité statistique sur le tableau de corrélation stress/calme (onglet
   Comparaison), pour savoir quelles corrélations sont réellement exploitables et pas juste du
   bruit.
2. Réintroduction des checkboxes par catégorie d'événement (crypto/macro/monétaire/géopolitique).
   **Ceci annule le point correspondant de `BRIEF_dashboard_v2_corrections.md` §2** (qui les avait
   supprimées suite à une demande de Maéva de garder les événements toujours visibles — décision
   inversée ici, sur nouvelle demande explicite).

Ne pas retoucher `regime_state.py`, `regime_hmm.py`, `regime_bocpd.py`, `test_regime_agent.py`.

---

## 1. Significativité statistique des corrélations stress/calme

**Problème :** le tableau "Corrélation moyenne inter-actifs — stress vs calme" affiche des valeurs
brutes (ex. BTC-TLT stress = -0.025) sans indiquer si l'écart par rapport à 0 est statistiquement
significatif ou juste du bruit d'échantillonnage. Un chercheur rigoureux doit pouvoir distinguer les
deux avant de tirer une conclusion (ex. "SPX-TLT montre un vrai effet de contagion" vs "BTC-TLT
n'est pas exploitable, trop proche de 0").

**Méthode retenue** : test de significativité d'une corrélation de Pearson via la transformation de
Fisher (standard, ne nécessite aucune nouvelle dépendance — juste `math`) :

```
z_crit = 1.959964                          # quantile normal bilatéral à 95 %
r_crit(n) = tanh( z_crit / sqrt(n - 3) )   # seuil critique, fonction de la taille d'échantillon n
significatif  <=>  |r| > r_crit(n)
```

Le seuil `r_crit` dépend de `n` (nombre de jours dans le sous-échantillon stress ou calme) — il
n'est **pas** le même des deux côtés puisque les deux buckets n'ont pas la même taille (le bucket
"calme" = les 4 actifs simultanément calmes est probablement plus restrictif/petit que "stress" =
au moins 1 actif sur 4, ou l'inverse selon les données réelles — ne pas supposer, calculer les deux
`n` et les deux seuils séparément).

### `regime_analytics.py`

Ajouter :

```python
import math

def fisher_r_critical(n: int, z_crit: float = 1.959964) -> float | None:
    """
    Seuil critique |r| au-delà duquel une corrélation de Pearson calculée sur n observations
    est significativement différente de 0 (test bilatéral, transformation de Fisher).
    Retourne None si n <= 3 (transformation non définie, échantillon trop petit pour tester).
    """
    if n <= 3:
        return None
    return math.tanh(z_crit / math.sqrt(n - 3))


def correlation_significance(r: float, n: int, z_crit: float = 1.959964) -> dict:
    """
    Teste si r (calculé sur n observations) est significativement différent de 0.
    Retourne {"r_crit": float | None, "significant": bool, "n": int}.
    """
    r_crit = fisher_r_critical(n, z_crit)
    if r_crit is None:
        return {"r_crit": None, "significant": False, "n": n}
    return {"r_crit": r_crit, "significant": bool(abs(r) > r_crit), "n": n}
```

Modifier `stress_conditioned_correlation` pour exposer les tailles d'échantillon utilisées (déjà
calculées en interne via `union_stress`/`all_calm`, il ne manque que de les retourner) :

```python
return {
    "stress": stress_corr, "calm": calm_corr,
    "stress_mask": union_stress, "calm_mask": all_calm,
    "n_stress": int(union_stress.sum()), "n_calm": int(all_calm.sum()),
}
```

### `dashboard_builder.py` — `_comparison_payload`

```python
r_crit_stress = ra.fisher_r_critical(stress_cond["n_stress"])
r_crit_calm = ra.fisher_r_critical(stress_cond["n_calm"])

pairs_table = []
for a, b in itertools.combinations(shorts, 2):
    s_val = stress_cond["stress"].loc[a, b] if a in stress_cond["stress"].index else float("nan")
    c_val = stress_cond["calm"].loc[a, b] if a in stress_cond["calm"].index else float("nan")
    s_ok = not (isinstance(s_val, float) and np.isnan(s_val))
    c_ok = not (isinstance(c_val, float) and np.isnan(c_val))
    pairs_table.append({
        "pair": f"{a}-{b}",
        "stress": _num(s_val) if s_ok else None,
        "calm": _num(c_val) if c_ok else None,
        "stress_sig": bool(r_crit_stress is not None and s_ok and abs(s_val) > r_crit_stress),
        "calm_sig": bool(r_crit_calm is not None and c_ok and abs(c_val) > r_crit_calm),
    })
```

Ajouter au dict retourné par `_comparison_payload` :

```python
"corr_significance": {
    "n_stress": stress_cond["n_stress"],
    "n_calm": stress_cond["n_calm"],
    "r_crit_stress": _num(r_crit_stress) if r_crit_stress is not None else None,
    "r_crit_calm": _num(r_crit_calm) if r_crit_calm is not None else None,
},
```

### HTML/JS — affichage

Dans le tableau stress/calme (`initComparisonTab`), marquer les valeurs significatives d'un
astérisque et ajouter une note explicative sous le tableau (remplace le rendu actuel des lignes) :

```js
const pbody = document.getElementById('pairs-body');
const fmt = (v, sig) => v === null ? '&#8212;'
  : `${v.toFixed(3)}${sig ? ' <span class="sig-star" title="Significatif à 95%">*</span>' : ''}`;
COMPARISON.pairs_table.forEach(p => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${p.pair}</td><td>${fmt(p.stress, p.stress_sig)}</td><td>${fmt(p.calm, p.calm_sig)}</td>`;
  pbody.appendChild(tr);
});

const sig = COMPARISON.corr_significance;
document.getElementById('corr-sig-note').innerHTML =
  `* = corrélation significativement différente de 0 au seuil de confiance 95% ` +
  `(test de Fisher). Stress : n=${sig.n_stress} jours, seuil |r| > ${sig.r_crit_stress?.toFixed(3) ?? '—'}. ` +
  `Calme : n=${sig.n_calm} jours, seuil |r| > ${sig.r_crit_calm?.toFixed(3) ?? '—'}.`;
```

Ajouter dans le HTML, juste après le `<table>` du tableau stress/calme :
`<p class="chart-note" id="corr-sig-note"></p>`

Ajouter au CSS : `.sig-star{{color:#f39c12;font-weight:700}}`

### Tests — `test_regime_analytics.py`

```python
def test_fisher_r_critical_decreases_with_n():
    # Seuil plus bas (plus facile d'être significatif) quand n augmente
    assert fisher_r_critical(10) > fisher_r_critical(1000)

def test_fisher_r_critical_none_for_small_n():
    assert fisher_r_critical(3) is None
    assert fisher_r_critical(2) is None

def test_correlation_significance_basic():
    # r=0.9 sur 5 observations : pas assez de données pour être sûr, mais tester le mécanisme
    result_small_n = correlation_significance(0.9, n=5)
    result_large_n = correlation_significance(0.05, n=10000)
    assert result_large_n["significant"] is True   # petit r mais énorme échantillon -> significatif
    assert correlation_significance(0.01, n=10)["significant"] is False  # r quasi nul, petit n -> pas significatif
```

Importer `fisher_r_critical`, `correlation_significance` dans les imports de test en tête de
fichier, et vérifier que `stress_conditioned_correlation` retourne bien `n_stress`/`n_calm` dans le
test existant `test_stress_conditioned_correlation_strict_calm` (ajouter les assertions
correspondantes).

---

## 2. Réintroduction des checkboxes par catégorie d'événement

**Changement de décision** (Maéva) : remettre la possibilité de cocher/décocher l'affichage des
événements par catégorie (Crypto / Macro / Monétaire / Géopolitique) sur chaque onglet actif. Les
3 checkboxes régime (Calme/Tendanciel/Stress) restent inchangées. Le comportement de décongestion
automatique des libellés au zoom (§2 de v2) reste inchangé aussi — cette réintroduction ne concerne
que le filtre par catégorie, pas les libellés.

### `dashboard_builder.py` — `_asset_panel_html`

Réajouter dans le bloc `.legend`, après le séparateur qui suit les 3 checkboxes régime :

```html
<div class="sep"></div>
<div class="li"><input type="checkbox" class="cat-cb-{ticker}" value="crypto" checked><div class="dot" style="background:{_EVENT_COLORS['crypto']}"></div>Crypto</div>
<div class="li"><input type="checkbox" class="cat-cb-{ticker}" value="macro" checked><div class="dot" style="background:{_EVENT_COLORS['macro']}"></div>Macro</div>
<div class="li"><input type="checkbox" class="cat-cb-{ticker}" value="monetaire" checked><div class="dot" style="background:{_EVENT_COLORS['monetaire']}"></div>Mon&#233;taire</div>
<div class="li"><input type="checkbox" class="cat-cb-{ticker}" value="geopolitique" checked><div class="dot" style="background:{_EVENT_COLORS['geopolitique']}"></div>G&#233;opolitique</div>
```

(Les 4 `<div class="li">` actuels, sans checkbox, qui ne font qu'afficher les puces de couleur,
sont remplacés par ceux-ci.)

### JS

Dans l'état de l'onglet (`TABS[a.ticker].state`), remettre :

```js
state: {
  regimes: {calm: true, trending: true, stress: true},
  cats: {crypto: true, macro: true, monetaire: true, geopolitique: true},
  scale: 'annee',   // ou la valeur en place après application de v3
},
```

Dans `buildShapes(tabId)`, remettre le filtre catégorie sur les lignes d'événements :

```js
d.event_lines.forEach(e => {
  if (!st.cats[e.cat]) return;
  shapes.push({type:'line',xref:'x',yref:'paper',x0:e.x,x1:e.x,y0:0,y1:1,
    line:{color:e.color,width:1.2,dash:'dot'}});
});
```

Dans `buildAnnotations(tabId)`, filtrer aussi par catégorie en plus de la plage visible :

```js
const visible = d.event_annotations.filter(a => st.cats[a.cat] && inRange(a.x, TABS[tabId].currentXRange));
```

Dans `initAssetTab(tabId)`, réajouter le listener (à côté de celui des checkboxes régime) :

```js
document.querySelectorAll(`.cat-cb-${tabId}`).forEach(cb => {
  cb.addEventListener('change', () => { TABS[tabId].state.cats[cb.value] = cb.checked; refreshTab(tabId); });
});
```

---

## 3. Vérification finale

- `pytest calibration/regime/ -v` → tout vert, y compris les 4 nouveaux tests de significativité.
- Onglet Comparaison : le tableau stress/calme affiche des `*` sur les corrélations
  significatives ; la note sous le tableau indique bien `n` et le seuil pour chaque bucket.
  Vérifier à l'œil que SPX-TLT (la paire la plus prometteuse d'après les valeurs observées) obtient
  bien un ou deux astérisques, et que BTC-TLT / ETH-TLT (valeurs proches de 0) n'en obtiennent pas.
- Chaque onglet actif (BTC/ETH/SPX/TLT) : les 4 checkboxes catégorie sont de retour dans la
  légende, cochées par défaut, et décocher l'une d'elles fait bien disparaître les traits verticaux
  (et labels) de cette catégorie sur les 3 graphiques temporels de l'onglet.
