# Corrections v5 — Dashboard Multi-Actifs DEITA (retour du 02/07/2026, suite)

**Remarque du tuteur (1) :** TLT est un ETF (un fonds qui détient un panier d'obligations du Trésor
américain, avec une allocation qui peut varier sans qu'on sache précisément comment), pas une
obligation elle-même — donc pas un vrai "US Bond" au sens strict.

**Remarque du tuteur (2), qui tranche le choix ci-dessous :** *"The most traded U.S. bond is the
U.S. 10-year Treasury note. It serves as the primary global benchmark for interest rates and
drives the highest daily trading volume. The 2-year, 5-year, and 30-year Treasuries are also
heavily traded, particularly on the CME Group Treasury Futures market."*

## Décision retenue

**Remplacer `TLT` par `ZN=F`** (contrat futures sur le Treasury Note 10 ans, CME Group), pas par le
30 ans (`ZB=F`) ni par la dette française. Justification :

- Le 10 ans est **le benchmark mondial des taux d'intérêt** et le titre du Trésor US le plus
  échangé, avec le plus gros volume quotidien (indication directe du tuteur) — c'est l'instrument
  obligataire le plus liquide et le plus représentatif du marché, donc le choix le plus rigoureux
  du point de vue "quel actif représente le mieux le marché obligataire US".
- C'est un **contrat futures**, pas un ETF : un engagement à terme directement sur le sous-jacent
  obligataire, utilisé par les acteurs du marché obligataire lui-même (traders taux, desks de
  couverture) — pas "l'obligation physique" (ça reste un dérivé), mais nettement plus proche d'un
  "vrai bond" qu'un fonds indiciel packagé pour particuliers comme TLT, et sans le problème
  d'allocation variable et opaque relevé par le tuteur sur TLT.
- Bonne compatibilité avec le pipeline existant : contrairement aux indices de taux (`^TNX`, qui
  cotent un pourcentage et non un prix, et n'ont pas de vrai volume), les futures CME ont un
  historique quotidien Open/High/Low/Close/**Volume** exploitable tel quel par `RegimeHMM`
  (`volume_norm` a besoin d'un vrai volume).
- **Compromis assumé** : le 10 ans a une duration plus courte que l'ancien TLT (Treasury 20+ ans).
  Une obligation plus courte réagit moins fortement aux mouvements de taux qu'une obligation
  longue (duration ∝ sensibilité au taux) — on doit donc s'attendre à des régimes de stress moins
  extrêmes/moins fréquents sur `ZN=F` que ce qu'on observait sur `TLT`. C'est un changement
  légitime (le tuteur priorise "l'instrument le plus représentatif du marché" sur "faire coller la
  duration à l'ancien choix"), mais à mentionner explicitement si le tuteur compare les résultats
  avant/après changement de ticker.
- La piste "dette française" reste écartée pour la même raison qu'en v5 initiale : pas d'instrument
  souverain français avec un historique quotidien OHLCV+Volume fiable et gratuit comparable aux
  futures US (OAT françaises négociées de gré à gré, sans volume public).

**Limite à documenter honnêtement** (rigueur) : les séries de futures "continues" (`ZN=F`)
recollent bout à bout les contrats trimestriels successifs. Au moment du "roll" (bascule vers le
contrat suivant), un saut de prix artificiel peut apparaître si la série n'est pas parfaitement
ajustée — ce n'est pas un problème pour un ETF comme TLT (pas de mécanique de roll). À surveiller
visuellement sur le graphique généré : des sauts de prix suspects et répétés à intervalles
réguliers, sans événement macro correspondant, signalent un artefact de roll et non un vrai régime
de stress — à signaler si observé, plutôt que laisser passer inaperçu. Le 10 ans étant le contrat
Treasury le plus liquide au monde, ce risque y est cependant plus faible que sur un contrat moins
échangé.

---

## Changements

### `assets.py`

Remplacer l'entrée TLT dans `ASSETS` :

```python
{"ticker": "ZN=F", "label": "US Treasury 10Y Note Futures", "short": "ZN", "asset_class": "bond", "color": "#3498db"},
```

Renommer la clé du dict `ASSET_EVENTS` de `"TLT"` vers `"ZN=F"` (le contenu des événements reste le
même, seule la clé change) :

```python
ASSET_EVENTS = {
    ...
    "ZN=F": {
        "2022-03-16": ("Début hausses de taux Fed",   "monetaire"),
        "2023-10-19": ("US 10Y touche ~5%",           "monetaire"),
    },
}
```

### `dashboard_builder.py`

Trois occurrences textuelles à mettre à jour (pas de logique à changer, `ASSETS`/`events_for_ticker`
sont déjà génériques et suivront automatiquement le nouveau ticker) :

- Ligne ~2 (docstring du module) : `BTC/ETH/SPY/TLT` → `BTC/ETH/SPY/ZN=F`.
- Ligne ~350 (`<p class="sub">`) : `US Treasury 20+Y (TLT)` → `US Treasury 10Y Note Futures (ZN=F)`.
- Ligne ~376 (`chart-note` du graphique de corrélation glissante) : `TLT = US Treasury 20+ ans
  (TLT)` → `ZN = US Treasury 10Y Note Futures (ZN=F), le benchmark mondial des taux`.

### `regime_analytics.py`

Ligne ~33 (commentaire de `segment_regimes`) : généraliser `"SPY/TLT cotent ~5j/7"` en
`"SPY/ZN=F cotent ~5j/7 (marchés fermés le week-end)"` — le point de rigueur calendaire/trading
days reste valable à l'identique (les futures ne tradent pas non plus le week-end).

### Vérification des données avant de lancer tout le pipeline

Le 10 ans étant le contrat Treasury le plus liquide au monde, la disponibilité des données ne
devrait pas poser de problème — vérifier quand même rapidement avant de relancer le pipeline complet
(téléchargement + fit HMM sur 4 actifs, pas instantané) :

```python
import yfinance as yf
df = yf.download("ZN=F", start="2018-01-01", auto_adjust=True, progress=False)
print(len(df), df["Volume"].isna().mean(), df["Volume"].eq(0).mean())
```

- Si `len(df)` < 252 (MIN_TRAIN_DAYS de RegimeHMM) ou si `Volume` est majoritairement nul/NaN :
  basculer sur `ZF=F` (5-Year T-Note futures) ou `ZB=F` (30-Year T-Bond futures) à la place, et
  l'indiquer clairement en commentaire dans `assets.py` (pourquoi `ZN=F` a été écarté).
- Ne pas se rabattre silencieusement sur `TLT` en cas de souci — revenir vers moi pour retrancher.

---

## Vérification finale

- `pytest calibration/regime/ -v` → toujours vert (aucun test ne dépend du ticker `TLT` en dur à
  part la donnée synthétique de `test_rolling_cross_correlation_pairs_count`, qui utilise juste la
  chaîne `"TLT"` comme clé arbitraire dans un dict de test — pas besoin de la renommer, elle ne
  teste pas le vrai ticker).
- Régénérer `output/regime_dashboard.html` : l'onglet obligataire s'appelle maintenant `ZN` dans
  la barre d'onglets, toutes les mentions "TLT" dans les titres/sous-titres/notes ont disparu du
  rendu.
- Vérifier visuellement le graphique prix de l'onglet `ZN` : pas de saut de prix artificiel
  suspect et répété (signe d'un problème de roll de contrat, cf. limite documentée plus haut). Si
  ça apparaît, le signaler avant de livrer.
- Vérifier que les régimes détectés sur `ZN=F` sont bien moins extrêmes/moins fréquents en stress
  que ne l'était l'ancien `TLT` (cohérent avec la duration plus courte, cf. compromis assumé
  ci-dessus) — ce n'est pas un bug si c'est le cas.
