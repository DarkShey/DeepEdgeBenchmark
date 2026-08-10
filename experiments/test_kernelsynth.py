"""Tests de `kernelsynth.py` -- generateur de series synthetiques par GP.

Le risque reel avec un generateur : produire silencieusement des NaN, des
series constantes, ou une matrice de covariance non definie positive qui casse
la Cholesky. Ces tests visent cela d'abord, les proprietes mathematiques
ensuite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernelsynth as ks                                              # noqa: E402


# ── noyaux ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,params", list(ks.DEFAULT_BANK))
def test_every_bank_kernel_is_symmetric_psd(name, params):
    t = np.arange(40, dtype=float)
    K = ks.kernel_matrix(name, t, **params)
    assert K.shape == (40, 40)
    assert np.allclose(K, K.T), f"{name} non symetrique"
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-8, f"{name} non semi-defini positif (min={eig.min()})"


def test_rbf_hand_calc():
    t = np.array([0.0, 1.0])
    K = ks.kernel_matrix("rbf", t, lengthscale=1.0)
    assert K[0, 0] == pytest.approx(1.0)
    assert K[0, 1] == pytest.approx(np.exp(-0.5))


def test_periodic_repeats_at_the_period():
    t = np.array([0.0, 12.0, 6.0])
    K = ks.kernel_matrix("periodic", t, period=12.0, lengthscale=1.0)
    assert K[0, 1] == pytest.approx(1.0)          # distance = 1 periode -> correlation max
    assert K[0, 2] < 0.5                          # demi-periode -> correlation faible


def test_white_is_diagonal():
    K = ks.kernel_matrix("white", np.arange(5, dtype=float), level=2.0)
    assert np.allclose(K, 2.0 * np.eye(5))


def test_unknown_kernel_raises():
    with pytest.raises(KeyError):
        ks.kernel_matrix("matern", np.arange(3, dtype=float))


# ── composition ─────────────────────────────────────────────────────────────

def test_composed_kernel_stays_psd_and_symmetric():
    t = np.arange(60, dtype=float)
    for seed in range(25):
        K, desc = ks.compose_kernel(t, np.random.default_rng(seed))
        assert np.allclose(K, K.T), f"seed {seed} : {desc}"
        assert np.linalg.eigvalsh(K).min() > -1e-6, f"seed {seed} : {desc}"
        assert isinstance(desc, str) and desc


def test_composition_is_reproducible_for_a_given_seed():
    t = np.arange(30, dtype=float)
    a, da = ks.compose_kernel(t, np.random.default_rng(3))
    b, db = ks.compose_kernel(t, np.random.default_rng(3))
    assert np.allclose(a, b) and da == db


def test_normalisation_keeps_the_linear_kernel_from_dominating():
    """Sans normalisation, la variance du noyau lineaire croit en t^2 et ecrase
    tout produit. Apres normalisation la diagonale reste d'ordre 1."""
    t = np.arange(100, dtype=float)
    K = ks._normalise(ks.kernel_matrix("linear", t))
    assert np.mean(np.diag(K)) == pytest.approx(1.0)


# ── generation ──────────────────────────────────────────────────────────────

def test_generate_shape_and_finiteness():
    series, descs = ks.generate(n_series=12, length=64, seed=0)
    assert series.shape == (12, 64)
    assert len(descs) == 12
    assert np.all(np.isfinite(series))


def test_generated_series_are_standardised():
    series, _ = ks.generate(n_series=20, length=80, seed=1)
    assert np.allclose(series.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(series.std(axis=1), 1.0, atol=1e-9)


def test_generation_is_reproducible():
    a, da = ks.generate(6, 40, seed=7)
    b, db = ks.generate(6, 40, seed=7)
    assert np.allclose(a, b) and da == db


def test_different_seeds_give_different_series():
    a, _ = ks.generate(4, 40, seed=1)
    b, _ = ks.generate(4, 40, seed=2)
    assert not np.allclose(a, b)


def test_generated_series_are_diverse_not_all_noise():
    """Le point de KernelSynth : produire des FORMES variees. Si toutes les
    series etaient du bruit blanc, leur autocorrelation a l'ordre 1 serait
    concentree autour de 0 -- on verifie qu'elle s'etale."""
    series, _ = ks.generate(60, 120, seed=11)
    ac1 = np.array([np.corrcoef(s[:-1], s[1:])[0, 1] for s in series])
    assert np.nanstd(ac1) > 0.2, "les series generees manquent de diversite"
    assert np.nanmax(ac1) > 0.5, "aucune serie fortement autocorrelee generee"


def test_short_series_still_work():
    series, _ = ks.generate(3, length=8, seed=0)
    assert series.shape == (3, 8) and np.all(np.isfinite(series))


def test_single_kernel_composition_is_allowed():
    t = np.arange(20, dtype=float)
    K, desc = ks.compose_kernel(t, np.random.default_rng(0), max_kernels=1)
    assert "+" not in desc and "*" not in desc
    assert np.linalg.eigvalsh(K).min() > -1e-8
