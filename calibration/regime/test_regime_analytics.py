import numpy as np
import pandas as pd

from calibration.regime.regime_analytics import (
    correlation_significance,
    ensure_stationary,
    fisher_r_critical,
    fisher_r_critical_bonferroni,
    granger_causality_vol_to_stress,
    granger_causality_volume_to_stress,
    lead_lag_cross_correlation,
    market_mask_intersection,
    market_mask_union,
    pairwise_stress_calm_correlation,
    regime_transition_vol_profile,
    regime_width_stats,
    rolling_cross_correlation,
    segment_boolean_mask,
    segment_regimes,
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


def _make_sigma_history(regimes, sigma, start="2020-01-01", freq="D"):
    """DataFrame minimal (regime, sigma_t) pour tester regime_transition_vol_profile."""
    idx = pd.date_range(start, periods=len(regimes), freq=freq)
    return pd.DataFrame({"regime": regimes, "sigma_t": sigma}, index=idx)


def test_regime_transition_vol_profile_peak_near_event():
    window = 5
    # 3 segments bien espacés : calm(0-29), stress(30-59), calm(60-89) -> 2 transitions,
    # toutes deux avec une fenêtre complète des deux côtés.
    regimes = ["calm"] * 30 + ["stress"] * 30 + ["calm"] * 30
    n = len(regimes)
    sigma = np.ones(n)
    # Pic net juste avant la transition calm->stress (pos 30).
    for offset in range(-3, 1):
        sigma[30 + offset] = 5.0
    df = _make_sigma_history(regimes, sigma)

    # only_into="stress" : seule la transition calm->stress (pos 30) est retenue ; la
    # transition stress->calm (pos 60) va vers "calm", donc exclue.
    profile = regime_transition_vol_profile(df, window=window, alignment="start", only_into="stress")

    assert (profile["n_events"] == 1).all()
    peak_rel_day = profile.loc[profile["mean_sigma"].idxmax(), "rel_day"]
    assert peak_rel_day <= 0, (
        f"Le pic de vol construit avant la transition doit apparaître à rel_day <= 0, got {peak_rel_day}"
    )


def test_regime_transition_vol_profile_pools_all_transitions_by_default():
    window = 5
    regimes = ["calm"] * 30 + ["stress"] * 30 + ["calm"] * 30
    df = _make_sigma_history(regimes, np.ones(len(regimes)))

    # Sans only_into, les 2 transitions (calm->stress à pos 30, stress->calm à pos 60) sont
    # poolées ensemble ; seul le tout premier segment de l'historique est exclu (pas de "avant").
    profile = regime_transition_vol_profile(df, window=window, alignment="start")

    assert (profile["n_events"] == 2).all()


def test_regime_transition_vol_profile_excludes_incomplete_window():
    window = 5
    # calm(0-2), stress(3-7), calm(8-59) : la transition calm->stress (pos 3, 2e segment de
    # l'historique — donc pas le tout premier) a une fenêtre "avant" incomplète (3 - 5 < 0) et
    # doit être ignorée sans padding artificiel. La transition stress->calm (pos 8) a une
    # fenêtre complète et doit être retenue.
    regimes = ["calm"] * 3 + ["stress"] * 5 + ["calm"] * 52
    df = _make_sigma_history(regimes, np.ones(len(regimes)))

    profile = regime_transition_vol_profile(df, window=window, alignment="start")

    assert (profile["n_events"] == 1).all()


def test_regime_transition_vol_profile_end_alignment_excludes_last_segment():
    window = 5
    # 2 segments seulement : calm(0-29), stress(30-59). En alignment="end", le dernier segment
    # (stress, qui n'a pas de "après" dans les données) est exclu par construction ; seule la
    # fin du segment calm (pos 29) est un événement utilisable.
    regimes = ["calm"] * 30 + ["stress"] * 30
    df = _make_sigma_history(regimes, np.ones(len(regimes)))

    profile = regime_transition_vol_profile(df, window=window, alignment="end")

    assert (profile["n_events"] == 1).all()


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


def test_pairwise_stress_calm_correlation_independent_of_third_asset():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    rng = np.random.RandomState(0)
    returns_by_asset = {
        "A": pd.Series(rng.normal(0, 0.02, 20), index=idx),
        "B": pd.Series(rng.normal(0, 0.02, 20), index=idx),
        "C": pd.Series(rng.normal(0, 0.02, 20), index=idx),
    }
    # A et B partagent le même mask stress/calme ; C a un régime totalement indépendant (opposé
    # dans le temps), sans rapport avec la paire (A, B) testée.
    stress_masks = {
        "A": pd.Series([True] * 5 + [False] * 15, index=idx),
        "B": pd.Series([True] * 5 + [False] * 15, index=idx),
        "C": pd.Series([False] * 15 + [True] * 5, index=idx),
    }
    calm_masks = {
        "A": pd.Series([False] * 5 + [True] * 15, index=idx),
        "B": pd.Series([False] * 5 + [True] * 15, index=idx),
        "C": pd.Series([True] * 15 + [False] * 5, index=idx),
    }

    result_with_c = pairwise_stress_calm_correlation(returns_by_asset, stress_masks, calm_masks)
    row_with_c = result_with_c[result_with_c["pair"] == "A-B"].iloc[0]

    returns_without_c = {k: v for k, v in returns_by_asset.items() if k != "C"}
    stress_without_c = {k: v for k, v in stress_masks.items() if k != "C"}
    calm_without_c = {k: v for k, v in calm_masks.items() if k != "C"}
    result_without_c = pairwise_stress_calm_correlation(returns_without_c, stress_without_c, calm_without_c)
    row_without_c = result_without_c[result_without_c["pair"] == "A-B"].iloc[0]

    # La présence ou non de C dans le dict d'entrée ne doit strictement rien changer au résultat
    # de la paire (A, B) : preuve que le calcul ne dépend plus des actifs hors paire.
    assert row_with_c["n_stress"] == row_without_c["n_stress"]
    assert row_with_c["n_calm"] == row_without_c["n_calm"]
    assert row_with_c["corr_stress"] == row_without_c["corr_stress"]
    assert row_with_c["corr_calm"] == row_without_c["corr_calm"]


def test_pairwise_stress_calm_correlation_n_specific_to_pair():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    returns_by_asset = {
        "A": pd.Series(np.linspace(-0.01, 0.01, 10), index=idx),
        "B": pd.Series(np.linspace(0.01, -0.01, 10), index=idx),
        "C": pd.Series(np.linspace(-0.02, 0.02, 10), index=idx),
    }
    # A n'est jamais en stress ; B est en stress les 4 premiers jours ; C les 8 premiers jours.
    stress_masks = {
        "A": pd.Series([False] * 10, index=idx),
        "B": pd.Series([True] * 4 + [False] * 6, index=idx),
        "C": pd.Series([True] * 8 + [False] * 2, index=idx),
    }
    calm_masks = {
        "A": pd.Series([True] * 10, index=idx),
        "B": pd.Series([False] * 4 + [True] * 6, index=idx),
        "C": pd.Series([False] * 8 + [True] * 2, index=idx),
    }

    result = pairwise_stress_calm_correlation(returns_by_asset, stress_masks, calm_masks)
    n_ab = int(result.loc[result["pair"] == "A-B", "n_stress"].iloc[0])
    n_ac = int(result.loc[result["pair"] == "A-C", "n_stress"].iloc[0])

    # n_stress(A-B) = OR(stress_A, stress_B) = 4 jours ; n_stress(A-C) = OR(stress_A, stress_C) = 8 jours.
    # Des valeurs différentes par paire, pas une seule valeur globale partagée (constat qui a
    # motivé ce patch : l'ancienne définition à 5 actifs donnait un n_stress/n_calm unique).
    assert n_ab == 4
    assert n_ac == 8
    assert n_ab != n_ac


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
    assert result_small_n["r_crit"] is not None  # n=5 > 3, le mécanisme doit tourner
    assert result_large_n["significant"] is True   # petit r mais énorme échantillon -> significatif
    assert correlation_significance(0.01, n=10)["significant"] is False  # r quasi nul, petit n -> pas significatif


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


def _make_granger_synthetic_data(seed, column, lag=3, linked=True):
    """
    Construit un DataFrame synthétique {column, p_stress}.
    linked=True : p_stress dépend fortement de `column` décalé de `lag` jours (+ bruit faible)
    -> un vrai lien de Granger existe. linked=False : les deux séries sont indépendantes -> aucun
    lien ne doit être détecté. Réutilisé pour tester granger_causality_vol_to_stress (column=
    "sigma_t") et granger_causality_volume_to_stress (column="volume_norm") de façon identique.
    """
    rng = np.random.RandomState(seed)
    n = 500
    source = rng.uniform(1.0, 5.0, n)
    if linked:
        p_stress = np.empty(n)
        p_stress[lag:] = 0.8 * (source[:-lag] / 5.0) + rng.normal(0, 0.02, n - lag)
        p_stress[:lag] = rng.uniform(0.1, 0.3, lag)
    else:
        p_stress = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({column: source, "p_stress": p_stress})


def test_granger_causality_vol_to_stress_detects_real_lagged_relationship():
    df = _make_granger_synthetic_data(seed=0, column="sigma_t", lag=3, linked=True)

    result = granger_causality_vol_to_stress(df, maxlag=5)

    assert set(result.keys()) == {
        "adf_source_p", "source_differenced", "adf_pstress_p", "pstress_differenced",
        "p_values", "n_obs",
    }
    assert set(result["p_values"].keys()) == set(range(1, 6))
    assert min(result["p_values"].values()) < 0.01


def test_granger_causality_vol_to_stress_no_link_when_independent():
    # Seuil corrigé pour comparaisons multiples (Bonferroni, 0.05/5 = 0.01) — sinon le test
    # serait un faux positif à force de scanner 5 lags.
    df = _make_granger_synthetic_data(seed=1, column="sigma_t", linked=False)

    result = granger_causality_vol_to_stress(df, maxlag=5)

    assert min(result["p_values"].values()) > 0.01


def test_granger_causality_volume_to_stress_detects_real_lagged_relationship():
    # Même construction que pour la vol (Q1), appliquée à volume_norm (Q2) — même rigueur,
    # même méthode, prouve que le wrapper volume n'est pas câblé sur la mauvaise colonne.
    df = _make_granger_synthetic_data(seed=0, column="volume_norm", lag=3, linked=True)

    result = granger_causality_volume_to_stress(df, maxlag=5)

    assert set(result.keys()) == {
        "adf_source_p", "source_differenced", "adf_pstress_p", "pstress_differenced",
        "p_values", "n_obs",
    }
    assert min(result["p_values"].values()) < 0.01


def test_granger_causality_volume_to_stress_no_link_when_independent():
    df = _make_granger_synthetic_data(seed=1, column="volume_norm", linked=False)

    result = granger_causality_volume_to_stress(df, maxlag=5)

    assert min(result["p_values"].values()) > 0.01


def test_lead_lag_cross_correlation_detects_b_leading_a():
    # b(t) précède a(t) de 2 jours : a(t) = b(t-2) + bruit faible -> le pic de |corrélation|
    # doit être à lag=+2 (convention : lag>0 = b décalée dans le passé corrèle avec a aujourd'hui,
    # donc b passé explique a présent -> b mène a).
    rng = np.random.RandomState(0)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    b = pd.Series(rng.normal(0, 1, n), index=idx)
    a_vals = np.empty(n)
    a_vals[2:] = b.values[:-2] + rng.normal(0, 0.05, n - 2)
    a_vals[:2] = rng.normal(0, 1, 2)
    a = pd.Series(a_vals, index=idx)

    result = lead_lag_cross_correlation(a, b, max_lag=5)
    ccf = result["ccf"]

    peak_row = ccf.loc[ccf["corr"].abs().idxmax()]
    assert int(peak_row["lag"]) == 2
    assert peak_row["corr"] > 0.9


def test_lead_lag_cross_correlation_detects_a_leading_b():
    # a(t) précède b(t) de 3 jours : b(t) = a(t-3) + bruit faible -> le pic doit être à lag=-3
    # (convention symétrique : lag<0 = b décalée dans le futur corrèle avec a aujourd'hui,
    # donc a présent explique b futur -> a mène b).
    rng = np.random.RandomState(1)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    a = pd.Series(rng.normal(0, 1, n), index=idx)
    b_vals = np.empty(n)
    b_vals[3:] = a.values[:-3] + rng.normal(0, 0.05, n - 3)
    b_vals[:3] = rng.normal(0, 1, 3)
    b = pd.Series(b_vals, index=idx)

    result = lead_lag_cross_correlation(a, b, max_lag=5)
    ccf = result["ccf"]

    peak_row = ccf.loc[ccf["corr"].abs().idxmax()]
    assert int(peak_row["lag"]) == -3
    assert peak_row["corr"] > 0.9


def test_lead_lag_cross_correlation_n_decreases_at_larger_lags():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.RandomState(2)
    a = pd.Series(rng.normal(0, 1, 100), index=idx)
    b = pd.Series(rng.normal(0, 1, 100), index=idx)

    result = lead_lag_cross_correlation(a, b, max_lag=5)
    ccf = result["ccf"]

    n_at_lag0 = ccf.loc[ccf["lag"] == 0, "n"].iloc[0]
    n_at_lag5 = ccf.loc[ccf["lag"] == 5, "n"].iloc[0]
    assert n_at_lag0 == 100
    assert n_at_lag5 < n_at_lag0


def test_ensure_stationary_leaves_stationary_series_unchanged():
    # Bruit blanc : déjà stationnaire (ADF rejette la racine unitaire) -> pas de différenciation.
    rng = np.random.RandomState(0)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    series = pd.Series(rng.normal(0, 1, n), index=idx)

    result, adf_p, differenced = ensure_stationary(series)

    assert differenced is False
    assert adf_p < 0.05
    assert len(result) == n


def test_ensure_stationary_differences_a_random_walk():
    # Marche aléatoire (cumsum de bruit) : non stationnaire par construction (ADF ne rejette pas
    # la racine unitaire) -> doit être différenciée, ce qui la ramène à du bruit stationnaire.
    rng = np.random.RandomState(1)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    series = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx)

    result, adf_p, differenced = ensure_stationary(series)

    assert differenced is True
    assert adf_p > 0.05
    assert len(result) == n - 1  # diff().dropna() perd la première observation


def test_fisher_r_critical_bonferroni_more_conservative_than_uncorrected():
    # Corriger pour 11 tests (comme les 11 lags de lead_lag_cross_correlation) doit relever le
    # seuil |r| requis — sinon la correction ne sert à rien.
    n = 500
    assert fisher_r_critical_bonferroni(n, n_tests=11) > fisher_r_critical(n)


def test_fisher_r_critical_bonferroni_none_for_small_n():
    assert fisher_r_critical_bonferroni(3, n_tests=11) is None
