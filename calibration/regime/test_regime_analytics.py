import numpy as np
import pandas as pd

from calibration.regime.regime_analytics import (
    market_mask_intersection,
    market_mask_union,
    regime_width_stats,
    rolling_cross_correlation,
    segment_boolean_mask,
    segment_regimes,
    stress_conditioned_correlation,
    vol_spike_hit_rate,
)


def _make_history(regimes, freq="D", start="2020-01-01", seed=0):
    """Construit un petit DataFrame history synthétique (regime, p_*, sigma_t)."""
    rng = np.random.RandomState(seed)
    n = len(regimes)
    idx = pd.date_range(start, periods=n, freq=freq)

    p_calm, p_trending, p_stress = [], [], []
    for r in regimes:
        base = {"calm": (0.8, 0.15, 0.05), "trending": (0.1, 0.8, 0.1), "stress": (0.05, 0.1, 0.85)}[r]
        p_calm.append(base[0])
        p_trending.append(base[1])
        p_stress.append(base[2])

    sigma_t = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({
        "regime": regimes,
        "p_calm": p_calm,
        "p_trending": p_trending,
        "p_stress": p_stress,
        "sigma_t": sigma_t,
    }, index=idx)


def test_segment_regimes_basic():
    regimes = ["calm"] * 5 + ["stress"] * 3 + ["calm"] * 2
    df = _make_history(regimes)

    segments = segment_regimes(df)

    assert len(segments) == 3
    assert segments["regime"].tolist() == ["calm", "stress", "calm"]
    assert segments["n_days_trading"].tolist() == [5, 3, 2]


def test_segment_regimes_calendar_vs_trading_days():
    # Index avec des trous type marché actions (vendredi -> lundi, saute le week-end)
    dates = pd.to_datetime([
        "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09",  # semaine 1 (calm)
        "2020-01-10",  # vendredi -> stress commence
        "2020-01-13",  # lundi (saute le week-end)
    ])
    regimes = ["calm"] * 5 + ["stress"] * 2
    df = pd.DataFrame({
        "regime": regimes,
        "p_calm": [0.8] * 5 + [0.05] * 2,
        "p_trending": [0.15] * 5 + [0.1] * 2,
        "p_stress": [0.05] * 5 + [0.85] * 2,
        "sigma_t": np.linspace(1, 2, 7),
    }, index=dates)

    segments = segment_regimes(df)
    stress_seg = segments[segments["regime"] == "stress"].iloc[0]

    # 2020-01-10 (vendredi) -> 2020-01-13 (lundi) : 2 lignes de trading, mais 4 jours calendaires
    assert stress_seg["n_days_trading"] == 2
    assert stress_seg["n_days_calendar"] == 4
    assert stress_seg["n_days_calendar"] > stress_seg["n_days_trading"]


def test_vol_spike_hit_rate_bounds():
    regimes = (["calm"] * 20 + ["stress"] * 10) * 5
    df = _make_history(regimes)

    rate = vol_spike_hit_rate(df, lookback=3, quantile=0.75)

    assert 0.0 <= rate <= 1.0


def test_segment_boolean_mask_basic():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    mask = pd.Series([False, True, True, False, True], index=idx)

    segments = segment_boolean_mask(mask)

    assert len(segments) == 2
    assert segments[0]["start"] == idx[1]
    assert segments[0]["end"] == idx[2]
    assert segments[1]["start"] == idx[4]
    assert segments[1]["end"] == idx[4]


def test_market_mask_union_and_intersection():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    masks = {
        "A": pd.Series([True, False, False, False], index=idx),
        "B": pd.Series([False, True, False, False], index=idx),
        "C": pd.Series([False, False, False, False], index=idx),
    }

    union = market_mask_union(masks)
    intersection = market_mask_intersection(masks)

    assert union.tolist() == [True, True, False, False]
    assert intersection.tolist() == [False, False, False, False]

    all_true_first_day = {
        "A": pd.Series([True, True], index=idx[:2]),
        "B": pd.Series([True, False], index=idx[:2]),
    }
    assert market_mask_intersection(all_true_first_day).tolist() == [True, False]


def test_stress_conditioned_correlation_strict_calm():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    returns_by_asset = {
        "A": pd.Series([0.01, 0.02, -0.01, 0.03], index=idx),
        "B": pd.Series([0.015, -0.005, 0.02, -0.02], index=idx),
    }
    # jour 1 (index 1) : A est trending (ni calm ni stress), B est calm.
    # Ancienne définition ("pas de stress") aurait inclus ce jour dans "calme".
    # Nouvelle définition stricte (intersection) doit l'exclure.
    stress_masks = {
        "A": pd.Series([False, False, True, False], index=idx),
        "B": pd.Series([False, False, False, False], index=idx),
    }
    calm_masks = {
        "A": pd.Series([True, False, False, True], index=idx),
        "B": pd.Series([True, True, True, True], index=idx),
    }

    result = stress_conditioned_correlation(returns_by_asset, stress_masks, calm_masks)

    assert result["calm_mask"].tolist() == [True, False, False, True]
    assert result["stress_mask"].tolist() == [False, False, True, False]
    assert result["calm_mask"].iloc[1] == False


def test_rolling_cross_correlation_pairs_count():
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.RandomState(42)
    returns_by_asset = {
        "BTC": pd.Series(rng.normal(0, 0.02, n), index=idx),
        "ETH": pd.Series(rng.normal(0, 0.02, n), index=idx),
        "SPY": pd.Series(rng.normal(0, 0.01, n), index=idx),
        "TLT": pd.Series(rng.normal(0, 0.01, n), index=idx),
    }

    result = rolling_cross_correlation(returns_by_asset, window=20)

    assert result.shape[1] == 6  # C(4,2) = 6 paires
    valid = result.dropna()
    assert not valid.empty
    for col in result.columns:
        assert result[col].dropna().between(-1, 1).all()
