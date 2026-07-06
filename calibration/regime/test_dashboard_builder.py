import numpy as np
import pandas as pd
import pytest

from calibration.regime.dashboard_builder import _event_study_from_profile


def _make_profile(mean_values, std_value=0.2, n_events=15, window=10):
    rel_day = np.arange(-window, window + 1)
    assert len(mean_values) == len(rel_day)
    return pd.DataFrame({
        "rel_day": rel_day,
        "mean_sigma": mean_values,
        "std_sigma": std_value,
        "n_events": n_events,
    })


def test_event_study_from_profile_zscore_handles_signed_signal():
    # Signal oscillant autour de 0 (ex. MACD histogram) : baseline (-10..-5) proche de 0 et de
    # signe mêlé -> le mode "pct" (ratio à la baseline) serait instable/trompeur ; method="zscore"
    # doit rester numériquement stable et ne pas lever d'erreur.
    window = 10
    n = 2 * window + 1
    rel_day = np.arange(-window, window + 1)
    mean_values = np.where(rel_day < 0, np.sin(rel_day / 3.0) * 0.5, 2.0 + rel_day * 0.1)
    profile = _make_profile(mean_values)

    result = _event_study_from_profile(profile, "Test", "#123456", method="zscore")

    assert result["n_events"] == 15
    assert len(result["index_pct"]) == n
    assert any(v is not None for v in result["index_pct"])
    assert all(v is None or np.isfinite(v) for v in result["index_pct"])


def test_event_study_from_profile_zscore_guards_zero_baseline_std():
    # Si la période de référence (-10..-5) est parfaitement constante (std=0), le score z serait
    # une division par zéro -> la fonction doit détecter ce cas et retourner index_pct=None
    # plutôt que planter ou produire des inf/NaN silencieux.
    window = 10
    rel_day = np.arange(-window, window + 1)
    mean_values = np.where(rel_day < -5, 1.0, np.where(rel_day < 0, 1.0, 3.0 + rel_day * 0.1))
    profile = _make_profile(mean_values)

    result = _event_study_from_profile(profile, "Test", "#123456", method="zscore")

    assert result["first_reaction_day"] is None
    assert all(v is None for v in result["index_pct"])


def test_event_study_from_profile_pct_unaffected_by_default_method():
    # method="pct" (défaut) doit rester strictement identique au comportement pré-v15 pour un
    # signal toujours positif (sigma_t/volume_norm) — non-régression Q1/Q2/Partie 2.
    window = 10
    rel_day = np.arange(-window, window + 1)
    mean_values = np.where(rel_day < 0, 1.0, 1.5)
    profile = _make_profile(mean_values)

    result = _event_study_from_profile(profile, "Test", "#123456")

    assert result["index_pct"][window] == pytest.approx(50.0)
