# Corrections v9 — Dashboard Multi-Actifs DEITA (audit de l'onglet Comparaison, 02/07/2026)

**Contexte :** audit rigoureux de l'onglet Comparaison sur les données réellement produites (pas
seulement le code). Trois problèmes de fond identifiés, avec les chiffres à l'appui. Ce patch les
corrige. Portée : `regime_hmm.py` + `regime_agent.py` pour le point 1 (encore une fois une
exception au principe "fichiers protégés", comme en v6, car c'est un vrai problème de modèle) ;
`regime_analytics.py` + `dashboard_builder.py` pour les points 2 et 3.

---

## 1. Bruit dans le split bull/bear (régimes de 1 jour)

**Constat chiffré** : sur les données actuelles, la durée médiane des segments `bear` est de
**1 jour pour SPX** (12 épisodes) et **1 jour pour TLT** (bull ET bear, 6 et 9 épisodes). Un
régime qui dure 1 jour n'est pas un régime, c'est du bruit de classification — le signe de
`DI+ - DI-` (qui tranche bull/bear à l'intérieur de la zone ADX > 25) oscille sans se stabiliser
sur ces actifs, ce qui pollue directement l'analyse de largeur des régimes (objectif n°2 de la
séance).

**Correction : lisser `di_diff` avant d'en prendre le signe**, exactement comme l'ADX lui-même est
déjà un indicateur lissé (Wilder, 14 jours) — cohérent avec la logique existante, pas une
rustine ad hoc.

### `regime_hmm.py` — `_compute_features`

```python
adx_df = ta.adx(prices["High"], prices["Low"], prices["Close"], length=self.ADX_PERIOD)
adx = adx_df[f"ADX_{self.ADX_PERIOD}"]
di_diff = adx_df[f"DMP_{self.ADX_PERIOD}"] - adx_df[f"DMN_{self.ADX_PERIOD}"]
di_diff_smooth = di_diff.rolling(5).mean()   # <- NOUVEAU : lissage 5 jours, réduit le bruit jour à jour
...
features = pd.DataFrame({
    "sigma_t": sigma_t,
    "adx": adx,
    "di_diff": di_diff,
    "di_diff_smooth": di_diff_smooth,
    "volume_norm": volume_norm,
}, index=prices.index)
return features.dropna()
```

Dans `predict()`, remplacer `di_diff_last = features["di_diff"].iloc[-1]` par
`di_diff_last = features["di_diff_smooth"].iloc[-1]` (le reste de la logique bull/bear ne
change pas).

### `regime_agent.py` — `_predict_history`

Remplacer `di_diff = features["di_diff"].values` par `di_diff = features["di_diff_smooth"].values`
(seule cette ligne change, le reste du calcul vectorisé bull/bear/calm/stress reste identique).

### Vérification

Après cette correction, régénérer et revérifier les durées médianes bull/bear par actif (même
requête que pour l'audit). Objectif : plus aucun régime avec une médiane à 1 jour. Si SPX/TLT
restent bruités avec une fenêtre de 5 jours, passer à 10 jours (`di_diff.rolling(10).mean()`) et
revérifier — ne pas se contenter du premier essai sans recontrôler les chiffres.

`test_regime_agent.py` TC4 (bull run BTC 2020-2021) doit continuer à passer sans modification :
la tendance y est assez forte et longue pour ne pas être affectée par un lissage à 5-10 jours.
Le confirmer en relançant les tests, pas en le supposant.

---

## 2. Corrélation stress/calme : définition par paire, pas globale sur 5 actifs

**Constat chiffré** : avec la définition actuelle (union du stress sur les 5 actifs / intersection
du calme sur les 5 actifs), le bucket "stress" couvre **1881 jours sur 2134 (88 %)** et le bucket
"calme" seulement **59 jours (2,8 %)**. Comparer "presque tout l'historique" à "un tout petit
échantillon" n'est pas exploitable, et c'est directement causé par le fait d'exiger une condition
simultanée sur 5 actifs sans rapport les uns aux autres pour tester la contagion entre 2 d'entre
eux en particulier.

**Correction : conditionner chaque paire sur elle-même**, pas sur les 3 autres actifs.

### `regime_analytics.py`

Remplacer `stress_conditioned_correlation` par :

```python
def pairwise_stress_calm_correlation(returns_by_asset: dict, stress_masks: dict, calm_masks: dict) -> pd.DataFrame:
    """
    Pour chaque paire (a, b), corrélation de Pearson des rendements conditionnée sur deux
    sous-échantillons SPÉCIFIQUES À LA PAIRE (pas un mask global sur tous les actifs du jeu de
    données) :
      - stress_pair = stress_masks[a] OR stress_masks[b]   (au moins l'un des deux stressé)
      - calm_pair   = calm_masks[a] AND calm_masks[b]       (les deux simultanément calmes)

    Justification : exiger qu'un actif sans rapport avec la paire testée (ex. TLT) soit lui aussi
    calme pour évaluer la corrélation BTC-SPX n'a pas de sens et réduit artificiellement
    l'échantillon (constaté : 59 jours sur 2134 avec la définition globale à 5 actifs). Chaque
    paire a maintenant son propre n_stress/n_calm et donc son propre seuil de significativité
    (fisher_r_critical) — les tailles d'échantillon diffèrent légitimement d'une paire à l'autre.

    Retourne un DataFrame, une ligne par paire :
    [pair, corr_stress, n_stress, r_crit_stress, stress_sig,
           corr_calm, n_calm, r_crit_calm, calm_sig].
    Si n_stress ou n_calm <= 3, la corrélation correspondante est NaN (échantillon insuffisant,
    cf. fisher_r_critical qui retourne déjà None dans ce cas).
    """
    import itertools
    keys = list(returns_by_asset.keys())
    rows = []
    for a, b in itertools.combinations(keys, 2):
        ra_, rb_ = returns_by_asset[a], returns_by_asset[b]
        common = ra_.index.intersection(rb_.index)
        ra_, rb_ = ra_.loc[common], rb_.loc[common]

        sa = stress_masks[a].reindex(common).fillna(False)
        sb = stress_masks[b].reindex(common).fillna(False)
        ca = calm_masks[a].reindex(common).fillna(False)
        cb = calm_masks[b].reindex(common).fillna(False)

        stress_pair = sa | sb
        calm_pair = ca & cb

        n_stress = int(stress_pair.sum())
        n_calm = int(calm_pair.sum())

        corr_stress = float(ra_.loc[stress_pair].corr(rb_.loc[stress_pair])) if n_stress > 3 else float("nan")
        corr_calm = float(ra_.loc[calm_pair].corr(rb_.loc[calm_pair])) if n_calm > 3 else float("nan")

        r_crit_s = fisher_r_critical(n_stress)
        r_crit_c = fisher_r_critical(n_calm)

        rows.append({
            "pair": f"{a}-{b}",
            "corr_stress": corr_stress, "n_stress": n_stress, "r_crit_stress": r_crit_s,
            "stress_sig": bool(r_crit_s is not None and corr_stress == corr_stress and abs(corr_stress) > r_crit_s),
            "corr_calm": corr_calm, "n_calm": n_calm, "r_crit_calm": r_crit_c,
            "calm_sig": bool(r_crit_c is not None and corr_calm == corr_calm and abs(corr_calm) > r_crit_c),
        })
    return pd.DataFrame(rows)
```

(`corr_stress == corr_stress` est un idiome pour "n'est pas NaN", évite un import `math.isnan`
supplémentaire ici — garder `fisher_r_critical`/`correlation_significance` existants tels quels,
déjà corrects.)

Supprimer `stress_conditioned_correlation` et `market_mask_union`/`market_mask_intersection` si
plus rien d'autre ne les utilise après ce changement (vérifier les autres appelants avant de
supprimer — `segment_boolean_mask` reste utile pour le point 3 ci-dessous, ne pas y toucher).

### `dashboard_builder.py`

Dans `compute_all_analytics`, remplacer l'appel à `stress_conditioned_correlation` par
`ra.pairwise_stress_calm_correlation(returns_by_asset, stress_masks, calm_masks)`.

Dans `_comparison_payload`, le `pairs_table` se construit directement depuis ce DataFrame (plus
besoin de boucler sur `itertools.combinations` séparément, la fonction le fait déjà) :

```python
pairwise = analytics["comparison"]["pairwise_stress_calm"]  # le nouveau DataFrame
pairs_table = [
    {
        "pair": row["pair"],
        "stress": _num(row["corr_stress"]) if row["corr_stress"] == row["corr_stress"] else None,
        "n_stress": int(row["n_stress"]),
        "r_crit_stress": _num(row["r_crit_stress"]) if row["r_crit_stress"] is not None else None,
        "stress_sig": bool(row["stress_sig"]),
        "calm": _num(row["corr_calm"]) if row["corr_calm"] == row["corr_calm"] else None,
        "n_calm": int(row["n_calm"]),
        "r_crit_calm": _num(row["r_crit_calm"]) if row["r_crit_calm"] is not None else None,
        "calm_sig": bool(row["calm_sig"]),
    }
    for _, row in pairwise.iterrows()
]
```

Supprimer le payload global `corr_significance` (n'a plus de sens, chaque paire a son propre n
et son propre seuil désormais — c'est plus rigoureux mais aussi plus honnête sur le fait que
l'échantillon diffère par paire).

### HTML/JS — table enrichie avec n et seuil par ligne

Remplacer les colonnes du tableau stress/calme pour inclure la taille d'échantillon à côté de
chaque valeur (plus de note globale unique en bas, l'info est maintenant par ligne) :

```html
<table><thead><tr><th>Paire</th><th>Corr. (stress)</th><th>n</th><th>Corr. (calme)</th><th>n</th></tr></thead>
<tbody id="pairs-body"></tbody></table>
<p class="chart-note">* = significatif à 95% (test de Fisher, seuil propre à chaque colonne car n diffère par paire). Survolez une valeur pour voir le seuil exact.</p>
```

```js
COMPARISON.pairs_table.forEach(p => {
  const tr = document.createElement('tr');
  const fmt = (v, sig, rcrit) => v === null ? '&#8212;'
    : `<span title="seuil |r| > ${rcrit !== null ? rcrit.toFixed(3) : '&#8212;'}">${v.toFixed(3)}${sig ? '<span class="sig-star">*</span>' : ''}</span>`;
  tr.innerHTML = `<td>${p.pair}</td>` +
    `<td>${fmt(p.stress, p.stress_sig, p.r_crit_stress)}</td><td>${p.n_stress}</td>` +
    `<td>${fmt(p.calm, p.calm_sig, p.r_crit_calm)}</td><td>${p.n_calm}</td>`;
  pbody.appendChild(tr);
});
```

Supprimer le bloc JS qui remplissait `#corr-sig-note` depuis `COMPARISON.corr_significance`
(n'existe plus) et retirer l'élément `<p id="corr-sig-note">` correspondant du HTML (remplacé par
la note générique ci-dessus).

---

## 3. Nettoyage du graphique de corrélation glissante (10 paires, trop chargé)

### a) Masquer par défaut les paires intra-classe (peu informatives)

BTC-ETH et ZN-TLT sont deux proxys du même sous-jacent chacun — toujours fortement corrélés
(~0,84 et ~0,80 sur les deux buckets), ils apportent peu et surchargent visuellement le graphique
au détriment des paires inter-classes (crypto vs indice, crypto vs obligation, indice vs
obligation), qui sont le vrai sujet. Les garder disponibles mais masquées par défaut (un clic sur
la légende les réaffiche — comportement natif Plotly, aucun code supplémentaire nécessaire pour
l'interaction elle-même) :

```js
function buildCcTraces(signal) {
  const cc = COMPARISON.cross_correlation[signal];
  const INTRA_CLASS = new Set(['BTC-ETH', 'ZN-TLT']);
  return Object.keys(cc.series).map((col,i) => ({
    type:'scatter', mode:'lines', x:cc.dates, y:cc.series[col], name:col,
    line:{width:1.5,color:CC_PALETTE[i % CC_PALETTE.length]},
    visible: INTRA_CLASS.has(col) ? 'legendonly' : true,
  }));
}
```

### b) Fond de stress : indicateur de majorité, pas d'union à 5 actifs

Le fond rouge actuel (union sur 5 actifs = 88 % des jours, cf. §2) ne discrimine plus rien
visuellement. Remplacer par un critère de majorité (au moins 3 des 5 actifs en stress le même
jour) — un indicateur de "stress de marché large", distinct et complémentaire du test rigoureux
par paire du tableau (§2), à ne pas confondre : préciser explicitement lequel est utilisé où.

`regime_analytics.py` : réutiliser `segment_boolean_mask` (inchangé) sur un nouveau mask calculé
côté `dashboard_builder.py` :

```python
# Dans compute_all_analytics ou _comparison_payload :
stress_count = pd.concat(stress_masks, axis=1).sum(axis=1)   # nombre d'actifs en stress, par jour
market_stress_majority = stress_count >= 3                    # majorité sur 5 actifs
stress_bands = [
    {"x0": str(s["start"].date()), "x1": str(s["end"].date())}
    for s in ra.segment_boolean_mask(market_stress_majority)
]
```

Si ce seuil de 3 s'avère encore trop large ou trop restrictif une fois régénéré (vérifier le
nombre de jours couverts, viser quelque chose de nettement inférieur à 88 % et nettement
supérieur à 3 %), essayer 4 avant de conclure.

Mettre à jour le `chart-note` du graphique de corrélation glissante pour expliciter cette
distinction :

```
Fond rouge = jours où au moins 3 des 5 actifs sont en régime stress (indicateur de marché large,
à but visuel). Le test statistique rigoureux de contagion, propre à chaque paire, est dans le
tableau ci-dessous — les deux ne se recouvrent pas nécessairement.
```

### c) Démoter volume/swing au profit de rendements/volatilité

Garder les 4 boutons mais distinguer visuellement les 2 signaux principaux (rendements,
volatilité — directement liés à la demande initiale) des 2 signaux exploratoires
(volume, swing — ajoutés sur demande complémentaire) :

```html
<div class="scale-sel">
  <button class="cc-btn cc-btn-active" data-signal="returns">Rendements</button>
  <button class="cc-btn" data-signal="volatility">Volatilité</button>
  <span class="sep"></span>
  <button class="cc-btn cc-btn-secondary" data-signal="volume">Volume</button>
  <button class="cc-btn cc-btn-secondary" data-signal="swing">Swing</button>
</div>
```

CSS : `.cc-btn-secondary{ opacity:.65; font-size:.68rem; }` (redevient opacity:1 au survol/actif,
même comportement de clic qu'avant — c'est purement une hiérarchie visuelle, pas un retrait de
fonctionnalité).

---

## 4. Tests à mettre à jour

`test_regime_analytics.py` :
- Supprimer les tests de `stress_conditioned_correlation` (fonction supprimée) et son import.
- Ajouter `test_pairwise_stress_calm_correlation_independent_of_third_asset` : construire 3 actifs
  synthétiques où le 3ᵉ a un régime totalement différent des deux premiers, vérifier que la
  corrélation stress/calme calculée pour la paire (1,2) est **identique** qu'on inclue ou non le
  3ᵉ actif dans le dict passé en entrée (preuve que le calcul ne dépend plus des actifs hors
  paire).
- Ajouter un test vérifiant que `n_stress`/`n_calm` sont bien spécifiques à chaque paire (pas une
  seule valeur globale partagée).

`test_regime_agent.py` : aucune nouvelle assertion requise pour le lissage `di_diff_smooth`
(comportement interne), mais **relancer TC4 et TC5 et confirmer qu'ils passent toujours** — ne pas
supposer que le lissage ne change rien sans vérifier.

---

## 5. Vérification finale

- `pytest calibration/regime/ -v` → tout vert.
- Régénérer `output/regime_dashboard.html` et revérifier les durées médianes bull/bear par actif
  (script d'audit similaire à celui utilisé pour ce diagnostic) : plus de médiane à 1 jour.
- Vérifier les nouvelles tailles d'échantillon du tableau stress/calme par paire : elles doivent
  être notablement plus équilibrées que 88 %/2,8 % (des valeurs différentes par paire sont
  normales et attendues, ce n'est plus un chiffre global unique).
- Sur le graphique de corrélation glissante : BTC-ETH et ZN-TLT masquées par défaut (visibles au
  clic sur la légende), fond rouge nettement moins étendu qu'avant, boutons Volume/Swing visuellement
  secondaires mais toujours cliquables.
