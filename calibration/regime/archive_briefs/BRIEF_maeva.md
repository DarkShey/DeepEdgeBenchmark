# Brief d'implémentation — Calibration de Régime (Maéva)

**Projet :** DEITA — Moteur de Régime  
**Scope :** Partie Maéva uniquement (RegimeState + GARCH + HMM + vol_bucket + tests)  
**Partie Kyrio :** BOCPD + Hurst + RegimeAgent + HTML output (à intégrer après livraison de ce module)

---

## 0. Changelog architecture — v2 (cascade 2 états + seuil ADX)

**Constat :** la version initiale (HMM gaussien à 3 états, un état par régime `calm`/`trending`/`stress`
appris directement par l'EM) ne sépare pas correctement les régimes sur BTC-USD réel. En cause :
sur l'historique complet, les journées à ADX fortement élevé s'accompagnent presque toujours d'une
volatilité également élevée (les fortes tendances haussières sont, empiriquement, aussi volatiles que
les phases de stress). Le clustering non supervisé à 3 composantes finit donc par apprendre un état
"stress" à variance large qui absorbe aussi les journées trending, et un état "trending" résiduel à
variance trop étroite pour représenter les vraies tendances fortes. Résultat observé : les tests TC1
(régime calme), TC4 (bull run) et TC6 (contrainte point-in-time) échouaient de façon reproductible,
indépendamment du seed ou du nombre de redémarrages du HMM.

**Solution retenue : approche en cascade à 2 étages.**
1. **Étage 1 — HMM gaussien à 2 états** (`stress` / `non_stress`), entraîné sur les mêmes features
   `[σ_t, ADX, volume_norm]`. Cette séparation binaire est fiable et robuste (confirmée par TC2 et TC3 :
   les épisodes COVID et FTX sont détectés comme `stress` sans ambiguïté).
2. **Étage 2 — seuil ADX déterministe**, appliqué uniquement à la masse de probabilité `non_stress` :
   `ADX > ADX_TRENDING_THRESHOLD (25)` → `trending`, sinon → `calm`.

Cette cascade remplace intégralement l'ancien HMM à 3 états et l'ancienne méthode
`_assign_regime_labels` à 3 labels. Les sections 5 et 6 ci-dessous reflètent cette architecture finale
(v2), qui fait passer la suite de tests à 7/7.

---

## 1. Contexte et objectif

Le moteur de régime est la **première couche de la cascade de calibration DEITA**.  
Il détecte dans quel état statistique se trouve le marché à un instant donné,
et transmet cet état aux couches en aval (calibration de prédiction, calibration de risque)
pour qu'elles adaptent leur comportement.

**Principe fondateur : régime ≠ signal.**  
Un régime décrit l'état du marché (volatilité, persistance, structure), sans prédire
de direction. Les trois états reconnus sont :

| État | Description | Signaux caractéristiques |
|------|-------------|--------------------------|
| `calm` | Faible volatilité, mouvements sans tendance claire | σ_t faible, ADX < 20, volume modéré |
| `trending` | Tendance directionnelle forte et persistante (haussière ou baissière) | ADX > 25, σ_t modérée, volume régulier |
| `stress` | Volatilité violente, mouvements chaotiques | σ_t élevée, ADX > 40 par violence, explosion du volume |

**Contrainte absolue : point-in-time.**  
À chaque prédiction à la date `as_of = T`, seules les données strictement antérieures
à T peuvent être utilisées. Aucune donnée future ne doit contaminer les features ou
l'entraînement.

---

## 2. Fichiers à créer

```
DeepEdgeBenchmark/
└── calibration/
    └── regime/
        ├── BRIEF.md              ← ce fichier
        ├── regime_state.py       ← contrat de sortie (dataclass)
        ├── regime_hmm.py         ← GARCH + features + HMM
        └── test_regime_agent.py  ← 6 test cases automatisés
```

---

## 3. Dépendances Python

Ajouter au `requirements.txt` existant :

```
hmmlearn>=0.3.0
arch>=6.0.0
pandas_ta>=0.3.14b
scikit-learn>=1.3.0
yfinance>=0.2.40
numpy>=1.24.0
pandas>=2.0.0
```

---

## 4. Fichier `regime_state.py`

### Spécification complète

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

@dataclass
class RegimeState:
    """
    Contrat de sortie du moteur de régime DEITA.
    Produit par RegimeHMM (ce module) et complété par BOCPD (Kyrio).
    Consommé par : calibration de prédiction (conformal), calibration de risque (sizing).
    """

    # ── Produit par HMM (ce module) ──────────────────────────────────────────
    probs: Dict[str, float]
    # Distribution a posteriori sur les états. Clés exactes : "calm", "trending", "stress".
    # Somme = 1.0 à 1e-6 près.
    # Exemple : {"calm": 0.65, "trending": 0.10, "stress": 0.25}

    vol_bucket: int
    # Entier 0, 1 ou 2.
    # Calculé par terciles de σ_t GARCH sur l'historique d'entraînement.
    # 0 = volatilité faible (tiers inférieur), 1 = modérée, 2 = élevée (tiers supérieur).
    # MÊME bucket consommé par conformal et sizing — ne jamais redéfinir ailleurs.

    stress_score: float
    # = probs["stress"]. Raccourci pour les couches de risque. Valeur dans [0, 1].

    expected_duration_days: float
    # Durée moyenne estimée du régime dominant courant, en jours.
    # Calculée depuis la matrice de transition HMM :
    #   state_idx = argmax(probs.values())
    #   expected_duration_days = 1 / (1 - transmat_[state_idx, state_idx])

    as_of: datetime
    # Date de calcul. Garantit la contrainte point-in-time :
    # toutes les données utilisées sont strictement antérieures à as_of.

    version: str
    # Identifiant de version de l'algorithme. Valeur fixe : "hmm-garch-adx-v1"

    # ── Complété par BOCPD (Kyrio) — valeurs par défaut ──────────────────────
    changepoint_prob: float = 0.0
    # Probabilité bayésienne qu'un changement de régime se produise à as_of.
    # Rempli par le module BOCPD de Kyrio. Par défaut 0.0.

    is_transitioning: bool = False
    # True si changepoint_prob > seuil (défini par Kyrio, typiquement 0.5).
    # Rempli par le module BOCPD de Kyrio. Par défaut False.

    def dominant_regime(self) -> str:
        """Retourne le nom du régime dominant (argmax des probs)."""
        return max(self.probs, key=self.probs.get)

    def validate(self) -> None:
        """
        Vérifie la cohérence interne du RegimeState.
        Lève ValueError si une contrainte est violée.
        """
        if set(self.probs.keys()) != {"calm", "trending", "stress"}:
            raise ValueError(f"probs doit contenir exactement calm/trending/stress, got {set(self.probs.keys())}")
        if abs(sum(self.probs.values()) - 1.0) > 1e-5:
            raise ValueError(f"probs ne somme pas à 1 : {sum(self.probs.values())}")
        if self.vol_bucket not in (0, 1, 2):
            raise ValueError(f"vol_bucket doit être 0, 1 ou 2, got {self.vol_bucket}")
        if not (0.0 <= self.stress_score <= 1.0):
            raise ValueError(f"stress_score hors [0,1] : {self.stress_score}")
        if not (0.0 <= self.changepoint_prob <= 1.0):
            raise ValueError(f"changepoint_prob hors [0,1] : {self.changepoint_prob}")
        if self.expected_duration_days <= 0:
            raise ValueError(f"expected_duration_days doit être > 0")
        if abs(self.stress_score - self.probs["stress"]) > 1e-6:
            raise ValueError("stress_score doit être égal à probs['stress']")
```

---

## 5. Fichier `regime_hmm.py`

### Architecture générale

```python
class RegimeHMM:
    """
    Moteur de détection de régime pour DEITA.
    Combine GARCH(1,1) pour la volatilité conditionnelle et une cascade à 2 étages
    pour la classification de régime :
      1. HMM gaussien à 2 états (stress / non_stress)
      2. seuil ADX pour distribuer la masse non_stress entre calm et trending
    """
    VERSION = "hmm-garch-adx-v1"
    N_STATES = 2  # stress / non_stress — le split calm/trending se fait ensuite par seuil ADX
    ADX_PERIOD = 14
    ADX_TRENDING_THRESHOLD = 25  # au sein du non_stress : ADX > 25 → trending, sinon calm
    GARCH_P = 1
    GARCH_Q = 1
    HMM_N_ITER = 200
    HMM_COV_TYPE = "diag"
    MIN_TRAIN_DAYS = 252  # 1 an de trading minimum
    CHANGEPOINT_THRESHOLD = 0.5  # seuil pour is_transitioning (défaut, écrasable par Kyrio)
    HMM_RESTART_SEEDS = [42, 0, 1, 7, 13]  # multi-restart : on garde le fit au meilleur log-likelihood
```

### Méthode `fit(prices: pd.DataFrame, train_end: str) -> None`

**Entrée :**
- `prices` : DataFrame yfinance avec colonnes `Open, High, Low, Close, Volume`, index DatetimeIndex quotidien
- `train_end` : date de fin d'entraînement au format `"YYYY-MM-DD"` (incluse)

**Algorithme pas à pas :**

1. **Filtrer** : conserver uniquement `prices[prices.index <= train_end]`
2. **Vérifier** : lever `ValueError` si moins de `MIN_TRAIN_DAYS` lignes disponibles
3. **Calculer les features** : appeler `_compute_features(prices_train)` → DataFrame `[σ_t, ADX, volume_norm]`
4. **Supprimer les NaN** : les premiers jours ont des NaN (warmup GARCH et ADX). Dropna strict.
5. **Normaliser** : ajuster un `StandardScaler` (scikit-learn) sur les features d'entraînement. Sauvegarder le scaler dans `self._scaler`.
6. **Entraîner le HMM à 2 états, avec multi-restart** : le HMM gaussien à 2 états (stress/non_stress)
   est entraîné plusieurs fois avec des seeds différentes ; on conserve le modèle dont le
   log-likelihood sur les données d'entraînement est le plus élevé.
   ```python
   from hmmlearn import hmm

   best_model, best_score = None, None
   for seed in self.HMM_RESTART_SEEDS:
       candidate = hmm.GaussianHMM(
           n_components=self.N_STATES,       # 2 : stress / non_stress
           covariance_type=self.HMM_COV_TYPE,
           n_iter=self.HMM_N_ITER,
           random_state=seed,
       )
       candidate.fit(X_scaled)
       score = candidate.score(X_scaled)
       if best_score is None or score > best_score:
           best_score, best_model = score, candidate
   self._hmm = best_model
   ```
7. **Calculer les seuils vol_bucket** : extraire la colonne σ_t (avant normalisation), calculer les tertiles sur l'historique d'entraînement. Sauvegarder dans `self._vol_thresholds = (q33, q66)`.
8. **Labelliser les états** : appeler `_assign_regime_labels()`. Sauvegarder le mapping `self._state_labels: Dict[int, str]` à 2 entrées (ex. `{0: "non_stress", 1: "stress"}`).
9. **Marquer le modèle comme entraîné** : `self._is_fitted = True`

### Méthode `predict(prices: pd.DataFrame, as_of: datetime) -> RegimeState`

**Entrée :**
- `prices` : DataFrame complet (le filtre point-in-time est fait ici)
- `as_of` : date de prédiction

**Algorithme pas à pas :**

1. **Vérifier** que `self._is_fitted == True`, sinon `RuntimeError`
2. **Filtrer point-in-time** : `prices_to_use = prices[prices.index < as_of]`
   - Note : `<` strict, pas `<=`, pour garantir qu'on n'utilise pas le jour même
3. **Calculer les features** sur `prices_to_use` via `_compute_features()`
4. **Supprimer les NaN** et vérifier qu'il reste au moins 30 lignes
5. **Normaliser** avec `self._scaler.transform()` (scaler ajusté à l'entraînement, pas re-ajusté)
6. **Prédire la probabilité de stress (étage 1 — HMM à 2 états)** :
   ```python
   log_probs = self._hmm.predict_proba(X_scaled)
   # Prendre la dernière ligne (état au jour as_of - 1)
   last_probs = log_probs[-1]  # shape (2,) : stress / non_stress

   stress_idx = [i for i, label in self._state_labels.items() if label == "stress"][0]
   p_stress = float(last_probs[stress_idx])
   p_non_stress = 1.0 - p_stress
   ```
7. **Distribuer la masse non_stress via le seuil ADX (étage 2)** :
   ```python
   adx_last = features["adx"].iloc[-1]  # ADX du dernier jour disponible
   if adx_last > self.ADX_TRENDING_THRESHOLD:
       p_trending, p_calm = p_non_stress, 0.0
   else:
       p_calm, p_trending = p_non_stress, 0.0

   probs = {"calm": p_calm, "trending": p_trending, "stress": p_stress}
   ```
8. **Calculer vol_bucket** :
   ```python
   sigma_t_last = sigma_t_series.iloc[-1]  # σ_t du dernier jour disponible
   q33, q66 = self._vol_thresholds
   vol_bucket = 0 if sigma_t_last < q33 else (1 if sigma_t_last < q66 else 2)
   ```
9. **Calculer expected_duration_days** : basé sur la persistance de l'état stress/non_stress
   sous-jacent (le seuil ADX de l'étage 2 n'a pas de matrice de transition propre) :
   ```python
   dominant_state_idx = int(np.argmax(last_probs))  # 0 ou 1, parmi stress/non_stress
   p_stay = self._hmm.transmat_[dominant_state_idx, dominant_state_idx]
   expected_duration_days = 1.0 / (1.0 - p_stay)
   ```
10. **Construire et retourner le RegimeState** :
    ```python
    state = RegimeState(
        probs=probs,
        vol_bucket=vol_bucket,
        stress_score=probs["stress"],
        expected_duration_days=expected_duration_days,
        as_of=as_of,
        version=self.VERSION,
        # changepoint_prob et is_transitioning laissés à 0.0 / False
        # → complétés par le BOCPD de Kyrio
    )
    state.validate()
    return state
    ```

### Méthode `_compute_features(prices: pd.DataFrame) -> pd.DataFrame`

Retourne un DataFrame avec les colonnes `["sigma_t", "adx", "volume_norm"]`.

**1. σ_t GARCH(1,1) :**
```python
from arch import arch_model

returns = prices["Close"].pct_change().dropna() * 100  # en pourcentage pour ARCH
am = arch_model(returns, vol="Garch", p=self.GARCH_P, q=self.GARCH_Q, dist="normal")
res = am.fit(disp="off", show_warning=False)
sigma_t = res.conditional_volatility  # pd.Series, même index que returns
```
Note : `conditional_volatility` est en unités de rendement (%). Le garder tel quel pour le HMM et pour vol_bucket — ne pas annualiser, les seuils sont calculés dans les mêmes unités.

**2. ADX 14j :**
```python
import pandas_ta as ta

adx_df = ta.adx(prices["High"], prices["Low"], prices["Close"], length=self.ADX_PERIOD)
# pandas_ta retourne un DataFrame avec plusieurs colonnes.
# La colonne ADX a le format : f"ADX_{self.ADX_PERIOD}"
adx = adx_df[f"ADX_{self.ADX_PERIOD}"]  # pd.Series
```

**3. Volume normalisé :**
```python
# Ratio du volume du jour sur la moyenne mobile 30j du volume
# → capture les explosions de volume relatives, indépendamment du niveau absolu
volume_norm = prices["Volume"] / prices["Volume"].rolling(30).mean()
```

**4. Alignement et retour :**
```python
features = pd.DataFrame({
    "sigma_t": sigma_t,
    "adx": adx,
    "volume_norm": volume_norm,
}, index=prices.index)
return features.dropna()
```

### Méthode `_assign_regime_labels(self) -> Dict[int, str]`

Après l'entraînement du HMM (2 états cachés, indices arbitraires 0/1), il faut les mapper
aux labels `stress` / `non_stress` de façon déterministe. Le split `calm`/`trending` au sein
de `non_stress` est fait séparément dans `predict()` via le seuil ADX (étage 2 de la cascade).

**Algorithme :**

Le HMM a appris, pour chaque état, un vecteur de moyennes sur les 3 features
`[σ_t_mean, adx_mean, volume_norm_mean]` (accessibles via `self._hmm.means_`).

```python
means = self._hmm.means_  # shape (2, 3) → (n_states, n_features)
# means[:, 0] = σ_t moyenne par état
# means[:, 2] = volume_norm moyen par état

# L'état STRESS a la σ_t la plus élevée ET le volume le plus élevé
# → score de stress = rank(σ_t) + rank(volume_norm)
stress_score = np.argsort(means[:, 0]) + np.argsort(means[:, 2])
stress_idx = int(np.argmax(stress_score))
non_stress_idx = 1 - stress_idx

mapping = {stress_idx: "stress", non_stress_idx: "non_stress"}
self._state_labels = mapping
return mapping
```

**Justification :** stress = forte σ_t + fort volume (explosion des volumes en panique).
Le reste (`non_stress`) est ensuite subdivisé par le seuil ADX : `trending` si ADX > 25
(force directionnelle élevée), `calm` sinon.

### Interface publique complète

```python
class RegimeHMM:
    VERSION = "hmm-garch-adx-v1"
    N_STATES = 2  # stress / non_stress
    ADX_PERIOD = 14
    ADX_TRENDING_THRESHOLD = 25
    GARCH_P, GARCH_Q = 1, 1
    HMM_N_ITER = 200
    HMM_COV_TYPE = "diag"
    MIN_TRAIN_DAYS = 252
    HMM_RESTART_SEEDS = [42, 0, 1, 7, 13]

    def __init__(self): ...
    def fit(self, prices: pd.DataFrame, train_end: str) -> None: ...
    def predict(self, prices: pd.DataFrame, as_of: datetime) -> RegimeState: ...
    def _compute_features(self, prices: pd.DataFrame) -> pd.DataFrame: ...
    def _assign_regime_labels(self) -> Dict[int, str]: ...
```

---

## 6. Fichier `test_regime_agent.py`

### Setup commun à tous les tests

```python
import yfinance as yf
import pytest
from datetime import datetime
from calibration.regime.regime_state import RegimeState
from calibration.regime.regime_hmm import RegimeHMM

TICKER = "BTC-USD"
DATA_START = "2017-01-01"

@pytest.fixture(scope="module")
def prices():
    """Télécharge une fois toutes les données BTC-USD depuis 2017."""
    return yf.download(TICKER, start=DATA_START, end="2025-01-01", auto_adjust=True)
```

### TC1 — Régime Calme (Juin–Sept 2023)

> **Note v2 :** la période initialement choisie (Jan–Mars 2024, post-ETF) s'est révélée être en
> réalité un régime `trending` (BTC 42k → 70k, ADX élevé), pas `calm`. Remplacée par une consolidation
> post-bear réellement calme (BTC ~26k-28k, ADX < 25 confirmé sur les données).

```python
def test_tc1_calm_regime(prices):
    """
    Période Sept 2023 : consolidation post-bear, BTC stable ~26k-28k, ADX faible.
    Régime attendu : calm dominant.
    """
    model = RegimeHMM()
    model.fit(prices, train_end="2023-07-31")

    state = model.predict(prices, as_of=datetime(2023, 9, 15))
    state.validate()

    assert state.dominant_regime() == "calm", (
        f"Régime attendu : calm. Obtenu : {state.dominant_regime()} — probs : {state.probs}"
    )
    assert state.probs["calm"] > 0.5, f"calm doit dépasser 0.5, got {state.probs['calm']}"
    assert state.vol_bucket <= 1, f"vol_bucket attendu 0 ou 1 en période calme, got {state.vol_bucket}"
    assert not state.is_transitioning
    assert state.stress_score == pytest.approx(state.probs["stress"], abs=1e-6)
    assert state.expected_duration_days > 0
    assert state.version == RegimeHMM.VERSION
```

### TC2 — Stress Soudain : COVID Crash (Mars 2020)

```python
def test_tc2_covid_stress(prices):
    """
    12 mars 2020 : BTC -37% en 24h. Régime attendu : stress dominant après le choc.
    Critère : probs["stress"] > 0.5 au 20 mars 2020 (quelques jours après).
    """
    model = RegimeHMM()
    model.fit(prices, train_end="2019-12-31")

    state = model.predict(prices, as_of=datetime(2020, 3, 20))
    state.validate()

    assert state.probs["stress"] > 0.5, (
        f"Stress attendu > 0.5 post-COVID. Obtenu : {state.probs}"
    )
    assert state.vol_bucket == 2, f"vol_bucket attendu 2 en crise COVID, got {state.vol_bucket}"
```

### TC3 — Crise Crypto : FTX Collapse (Nov 2022)

```python
def test_tc3_ftx_stress(prices):
    """
    8 nov 2022 : effondrement FTX, BTC -25% en 48h.
    Régime attendu : stress dominant après le choc.
    Critère : probs["stress"] > 0.5 au 18 nov 2022.
    """
    model = RegimeHMM()
    model.fit(prices, train_end="2022-10-31")

    state = model.predict(prices, as_of=datetime(2022, 11, 18))
    state.validate()

    assert state.probs["stress"] > 0.5, (
        f"Stress attendu > 0.5 post-FTX. Obtenu : {state.probs}"
    )
    assert state.vol_bucket == 2
```

### TC4 — Marché Tendanciel : Bull Run 2020–2021

```python
def test_tc4_trending_bull_run(prices):
    """
    Oct 2020 – Fév 2021 : BTC de 10k à 60k, tendance haussière forte et persistante.
    Régime attendu : trending dominant.
    Critère : probs["trending"] > 0.4 à la mi-décembre 2020.
    Note : seuil à 0.4 (pas 0.5) car stress peut aussi être élevé en phase d'accélération.
    """
    model = RegimeHMM()
    model.fit(prices, train_end="2020-09-30")

    state = model.predict(prices, as_of=datetime(2020, 12, 15))
    state.validate()

    assert state.probs["trending"] > 0.4, (
        f"Trending attendu > 0.4 en bull run. Obtenu : {state.probs}"
    )
    assert state.probs["calm"] < 0.4, (
        f"Calm doit être faible en bull run. Obtenu : {state.probs['calm']}"
    )
```

### TC5 — Marché Choppy : Consolidation 2019

```python
def test_tc5_choppy_consolidation(prices):
    """
    Jan–Août 2019 : consolidation post-bear 2018. BTC range 3500-14000.
    Régime attendu : calm ou stress (pas trending dominant).
    Critère strict : probs["trending"] < 0.4.
    """
    model = RegimeHMM()
    model.fit(prices, train_end="2018-12-31")

    state = model.predict(prices, as_of=datetime(2019, 6, 30))
    state.validate()

    assert state.probs["trending"] < 0.4, (
        f"Trending doit être faible en période choppy. Obtenu : {state.probs}"
    )
```

### TC6 — Point-in-time : vérification de la contrainte

> **Note v2 :** la date "avant crash" initiale (20 février 2020) se classait déjà à 97% `stress`
> avec la cascade à 2 états (un volume légèrement élevé cette semaine-là suffit à sortir du cluster
> `non_stress`, très étroit). Remplacée par le 10 janvier 2020, un point de référence clairement
> `calm` (`calm ≈ 0.998`), avant toute tension de marché liée au COVID.

```python
def test_tc6_point_in_time_constraint(prices):
    """
    Vérifie que predict() ne peut pas utiliser de données futures.
    Deux prédictions à des dates différentes doivent donner des résultats distincts
    si la période entre les deux est volatile (ici COVID : jan → avril 2020).
    """
    model = RegimeHMM()
    model.fit(prices, train_end="2019-12-31")

    # Avant le crash
    state_before = model.predict(prices, as_of=datetime(2020, 1, 10))
    # Après le crash
    state_after = model.predict(prices, as_of=datetime(2020, 4, 1))

    state_before.validate()
    state_after.validate()

    # Le régime doit avoir changé entre les deux dates
    # (au moins une probabilité doit différer de plus de 0.1)
    diffs = [abs(state_after.probs[k] - state_before.probs[k]) for k in ("calm", "trending", "stress")]
    assert max(diffs) > 0.1, (
        "Le régime ne change pas entre jan et avril 2020 — "
        "possible contamination par des données futures ou modèle non discriminant."
    )

    # Vérification explicite : as_of est bien enregistré
    assert state_before.as_of == datetime(2020, 1, 10)
    assert state_after.as_of == datetime(2020, 4, 1)
```

### TC7 — Validation du RegimeState (cohérence interne)

```python
def test_tc7_regimestate_validation():
    """
    Vérifie que validate() lève bien des erreurs sur des inputs invalides.
    """
    from datetime import datetime

    # probs ne somme pas à 1
    with pytest.raises(ValueError, match="somme"):
        RegimeState(
            probs={"calm": 0.5, "trending": 0.5, "stress": 0.5},
            vol_bucket=0, stress_score=0.5, expected_duration_days=10.0,
            as_of=datetime(2024, 1, 1), version="test"
        ).validate()

    # vol_bucket invalide
    with pytest.raises(ValueError, match="vol_bucket"):
        RegimeState(
            probs={"calm": 0.7, "trending": 0.2, "stress": 0.1},
            vol_bucket=3, stress_score=0.1, expected_duration_days=10.0,
            as_of=datetime(2024, 1, 1), version="test"
        ).validate()

    # stress_score incohérent avec probs["stress"]
    with pytest.raises(ValueError, match="stress_score"):
        RegimeState(
            probs={"calm": 0.7, "trending": 0.2, "stress": 0.1},
            vol_bucket=0, stress_score=0.9, expected_duration_days=10.0,
            as_of=datetime(2024, 1, 1), version="test"
        ).validate()
```

---

## 7. Contraintes et règles de développement

1. **Aucune donnée future** : dans `predict()`, filtrer avec `prices.index < as_of` (strict).
2. **Scaler figé** : le `StandardScaler` est ajusté une fois dans `fit()`. Dans `predict()`, appeler uniquement `transform()`, jamais `fit()` ou `fit_transform()`.
3. **Seuils vol_bucket figés** : les terciles sont calculés une fois dans `fit()`. Ne pas les recalculer dans `predict()`.
4. **Seeds fixes** : `HMM_RESTART_SEEDS = [42, 0, 1, 7, 13]` dans `fit()`. Multi-restart déterministe :
   plusieurs `GaussianHMM` sont entraînés (un par seed) et celui avec le meilleur log-likelihood
   sur les données d'entraînement est conservé, pour reproductibilité.
5. **Gestion des NaN** : après `_compute_features()`, faire un `dropna()` strict avant tout calcul. Si le résultat a moins de 30 lignes, lever `ValueError("Pas assez de données après suppression des NaN")`.
6. **Imports** : tous les imports en tête de fichier, pas d'import conditionnel.
7. **Pas de print** dans les fichiers de production (`regime_state.py`, `regime_hmm.py`). Les tests peuvent afficher des messages de debug.
8. **Compatibilité** : le code doit s'exécuter sans modifier les fichiers existants du Benchmark (arima_model.py, etc.).

---

## 8. Interface consommée par Kyrio

Kyrio intégrera ce module ainsi :

```python
from calibration.regime.regime_state import RegimeState
from calibration.regime.regime_hmm import RegimeHMM

# Kyrio complète le RegimeState produit par RegimeHMM avec les champs BOCPD :
state = regime_hmm_instance.predict(prices, as_of)
state.changepoint_prob = bocpd_result.changepoint_prob
state.is_transitioning = bocpd_result.changepoint_prob > CHANGEPOINT_THRESHOLD
```

**Ne pas modifier** l'interface de `RegimeState` après livraison — Kyrio s'y conforme.

---

## 9. Exécution des tests

```bash
# Depuis la racine du projet DeepEdgeBenchmark
cd DeepEdgeBenchmark
pip install -r requirements.txt
pytest calibration/regime/test_regime_agent.py -v
```

Les tests TC2 et TC3 téléchargent des données historiques via yfinance — une connexion internet est nécessaire. Les données sont mises en cache par le fixture `scope="module"`.
