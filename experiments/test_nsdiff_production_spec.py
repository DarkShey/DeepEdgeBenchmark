"""Tests de `nsdiff_production_spec.py` -- la spec de la config candidate
production. Le point de ces tests : verrouiller le choix "concatener, pas
moyenner les bornes", qui est TOUTE la spec et qu'une refactorisation
distraite inverserait sans que rien ne casse ailleurs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nsdiff_production_spec as spec                                  # noqa: E402


def test_spec_constants_are_coherent():
    s = spec.PRODUCTION_SPEC
    assert s["n_samples_total"] == len(s["seeds"]) * s["n_samples_per_seed"] == 1000
    assert len(spec.SEEDS) == 5 and spec.N_SAMPLES_PER_SEED == 200


def test_aggregate_concatenates_not_averages():
    """Deux graines en desaccord franc : [0..99] et [1000..1099]. Le melange
    couvre les deux modes ; une moyenne de bornes serait coincee au milieu."""
    a = np.arange(100.0)
    b = np.arange(1000.0, 1100.0)
    cloud = spec.aggregate_cloud([a, b])
    assert cloud.size == 200
    assert cloud.min() == 0.0 and cloud.max() == 1099.0


def test_mixture_is_wider_than_mean_of_bounds_when_seeds_disagree():
    """L'invariant qui justifie la spec. Graines en desaccord -> la bande du
    melange doit ETRE PLUS LARGE que la moyenne des bandes individuelles."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(loc, 1.0, 500) for loc in (-5.0, 0.0, 5.0)]

    mix = spec.read_forecast(spec.aggregate_cloud(clouds))
    width_mix = mix["y_upper"] - mix["y_lower"]

    per_seed = [spec.read_forecast(c) for c in clouds]
    width_mean_of_bounds = np.mean([p["y_upper"] - p["y_lower"] for p in per_seed])

    assert width_mix > 2 * width_mean_of_bounds


def test_mixture_and_mean_of_bounds_agree_when_seeds_agree():
    """Symetriquement : quand les graines sont d'accord, la spec ne gonfle rien."""
    rng = np.random.default_rng(1)
    clouds = [rng.normal(0.0, 1.0, 4000) for _ in range(5)]
    mix = spec.read_forecast(spec.aggregate_cloud(clouds))
    per_seed = [spec.read_forecast(c) for c in clouds]
    width_mix = mix["y_upper"] - mix["y_lower"]
    width_mean = np.mean([p["y_upper"] - p["y_lower"] for p in per_seed])
    assert width_mix == pytest.approx(width_mean, rel=0.05)


def test_read_forecast_matches_the_repo_formula():
    """Meme lecture que `generate_nsdiff_asset` : moyenne + np.quantile 2.5/97.5."""
    rng = np.random.default_rng(2)
    c = rng.normal(100.0, 3.0, 1000)
    got = spec.read_forecast(c)
    lo, hi = np.quantile(c, [0.025, 0.975])
    assert got["y_pred"] == pytest.approx(float(c.mean()))
    assert got["y_lower"] == pytest.approx(float(lo))
    assert got["y_upper"] == pytest.approx(float(hi))
    assert got["n_samples"] == 1000


def test_production_forecast_end_to_end():
    clouds = np.arange(1000.0).reshape(5, 200)
    out = spec.production_forecast(clouds)
    assert out["n_seeds"] == 5 and out["n_samples"] == 1000
    assert out["y_pred"] == pytest.approx(499.5)
    assert out["y_lower"] < out["y_upper"]


def test_aggregate_accepts_a_single_cloud():
    c = np.arange(10.0)
    assert np.array_equal(spec.aggregate_cloud(c), c)


def test_read_forecast_rejects_empty():
    with pytest.raises(ValueError):
        spec.read_forecast(np.array([]))


def test_aggregate_rejects_3d():
    with pytest.raises(ValueError):
        spec.aggregate_cloud(np.zeros((2, 3, 4)))


def test_validate_clouds_accepts_the_spec_shape():
    spec.validate_clouds(np.zeros((5, 200)))


@pytest.mark.parametrize("shape", [(3, 200), (5, 50), (5, 1000), (6, 200)])
def test_validate_clouds_rejects_off_spec_material(shape):
    with pytest.raises(ValueError):
        spec.validate_clouds(np.zeros(shape))


def test_validate_clouds_rejects_1d():
    with pytest.raises(ValueError):
        spec.validate_clouds(np.zeros(1000))
