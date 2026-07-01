import yfinance as yf
import pandas as pd
import pytest
from datetime import datetime
from calibration.regime.regime_state import RegimeState
from calibration.regime.regime_hmm import RegimeHMM

TICKER = "BTC-USD"
DATA_START = "2017-01-01"


@pytest.fixture(scope="module")
def prices():
    """Télécharge une fois toutes les données BTC-USD depuis 2017."""
    data = yf.download(TICKER, start=DATA_START, end="2025-01-01", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance récent renvoie des colonnes multi-index (Price, Ticker) même
        # pour un seul ticker. On aplatit pour obtenir Open/High/Low/Close/Volume.
        data.columns = data.columns.get_level_values(0)
    return data


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
