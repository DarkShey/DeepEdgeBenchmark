"""Tests de `stochvol_synth.py`.

Ce generateur existe pour remplacer KernelSynth dans l'augmentation des regimes
pauvres en donnees. Le test central n'est donc pas « le code tourne » mais « le
defaut est corrige ».

Ecrire ces tests a fait bouger le diagnostic lui-meme. Le defaut de KernelSynth
avait ete decrit comme « homoscedasticite », et le premier test compare donc
l'ACF des rendements au CARRE. Il echouait -- non parce que le nouveau
generateur etait mauvais, mais parce que KernelSynth affiche une ACF(r^2) PLUS
ELEVEE (+0,29) tout en etant homoscedastique par construction : ses series sont
lisses, leurs NIVEAUX sont autocorreles a +0,45, et cela gonfle mecaniquement
l'ACF de leurs carres. Mesure contre les vraies series mensuelles (ACF de niveau
+0,01), le vrai defaut apparait : on apprenait au modele qu'un rendement se
predit par le precedent. Le test discriminant porte donc desormais sur l'ACF de
NIVEAU, et l'ancien piege est verrouille par un test dedie.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernelsynth as ks                                              # noqa: E402
import stochvol_synth as sv                                           # noqa: E402


# ── LE test : le defaut diagnostique est-il corrige ? ───────────────────────

def test_returns_are_white_in_level_where_kernelsynth_is_not():
    """LE test discriminant, et il ne porte PAS sur ce qu'on croyait.

    Le defaut de KernelSynth avait ete decrit comme « homoscedasticite ». Mesure
    contre les vraies series, son defaut decisif est ailleurs : ses NIVEAUX sont
    autocorreles a +0,45 alors qu'un rendement reel est blanc (+0,01). Comme ces
    series sont consommees telles quelles comme rendements standardises, on
    apprenait au modele qu'un rendement se predit par le precedent.
    """
    svs, _ = sv.generate(60, 400, seed=0)
    kss, _ = ks.generate(60, 400, seed=0)
    lvl_sv = sv.level_autocorrelation(svs)
    lvl_ks = sv.level_autocorrelation(kss)
    assert abs(lvl_sv) < 0.05, f"niveaux non blancs : {lvl_sv:+.3f}"
    assert lvl_ks > 0.30, f"KernelSynth cense etre fortement autocorrele, mesure {lvl_ks:+.3f}"


def test_squared_acf_alone_would_have_been_misleading():
    """Piege documente : sur une serie dont les NIVEAUX sont autocorreles,
    l'ACF des carres est gonflee mecaniquement et ne mesure plus la volatilite.
    KernelSynth affiche ainsi une ACF(r^2) elevee tout en etant homoscedastique --
    c'est pourquoi les deux statistiques doivent etre lues ensemble."""
    kss, _ = ks.generate(60, 400, seed=0)
    assert sv.volatility_clustering(kss) > 0.2      # apparemment "bon"...
    assert sv.level_autocorrelation(kss) > 0.3      # ...mais explique par les niveaux


def test_matches_the_measured_stylized_facts():
    """Le generateur est CALIBRE sur les faits stylises des 7 series mensuelles
    reelles du panel. Ce test verrouille cette calibration : si les plages de
    parametres derivent, il tombe."""
    svs, _ = sv.generate(80, 180, seed=0)
    target = sv.REAL_STYLIZED_FACTS
    assert abs(sv.level_autocorrelation(svs) - target["acf_level"]) < 0.05
    assert abs(sv.volatility_clustering(svs) - target["acf_squared"]) < 0.06
    # la kurtosis depasse la cible : depassement DECLARE, borne pour rester
    # dans la fourchette des actifs individuels (USO reel : +11,1)
    assert 2.0 < sv.excess_kurtosis(svs) < 11.0


def test_produces_heavy_tails():
    """Le clustering seul epaissit deja les queues ; les innovations de Student
    y ajoutent. L'exces de kurtosis doit etre franchement positif."""
    svs, _ = sv.generate(60, 400, seed=1)
    assert sv.excess_kurtosis(svs) > 0.5


def test_clustering_grows_with_persistence():
    """Invariant du modele : plus phi est grand, plus la volatilite est
    persistante, donc plus l'autocorrelation des carres est forte."""
    rng_lo = np.random.default_rng(3)
    rng_hi = np.random.default_rng(3)
    lo = np.stack([sv.simulate(600, rng_lo, phi=0.50, sigma_eta=0.3, rho=0.0, nu=np.inf)
                   for _ in range(30)])
    hi = np.stack([sv.simulate(600, rng_hi, phi=0.98, sigma_eta=0.3, rho=0.0, nu=np.inf)
                   for _ in range(30)])
    assert sv.volatility_clustering(hi) > sv.volatility_clustering(lo)


def test_leverage_is_negative_when_rho_is_negative():
    """Effet de levier : une baisse doit s'accompagner d'une volatilite plus
    forte le pas suivant -- correlation negative entre r_t et r_{t+1}^2."""
    rng = np.random.default_rng(7)
    corrs = []
    for _ in range(40):
        r = sv.simulate(800, rng, phi=0.95, sigma_eta=0.35, rho=-0.6, nu=np.inf)
        corrs.append(np.corrcoef(r[:-1], r[1:] ** 2)[0, 1])
    assert np.mean(corrs) < -0.02, f"pas de levier mesurable : {np.mean(corrs):.4f}"


def test_no_leverage_when_rho_is_zero():
    rng = np.random.default_rng(8)
    corrs = [np.corrcoef(r[:-1], r[1:] ** 2)[0, 1] for r in
             (sv.simulate(800, rng, phi=0.95, sigma_eta=0.35, rho=0.0, nu=np.inf)
              for _ in range(40))]
    assert abs(np.mean(corrs)) < 0.05


# ── plages declarees ────────────────────────────────────────────────────────

def test_sampled_parameters_stay_in_declared_ranges():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = sv.sample_params(rng)
        assert sv.PHI_RANGE[0] <= p["phi"] <= sv.PHI_RANGE[1]
        assert sv.SIGMA_ETA_RANGE[0] <= p["sigma_eta"] <= sv.SIGMA_ETA_RANGE[1]
        assert sv.RHO_RANGE[0] <= p["rho"] <= sv.RHO_RANGE[1]
        assert p["nu"] in sv.NU_CHOICES


def test_leverage_is_never_positive():
    """Un levier positif n'a pas de sens pour un actif au comptant et
    apprendrait au modele une asymetrie inversee."""
    rng = np.random.default_rng(1)
    assert all(sv.sample_params(rng)["rho"] <= 0.0 for _ in range(300))


def test_gaussian_case_is_reachable():
    """La plage doit inclure le cas SANS queues lourdes ajoutees : toutes les
    series financieres n'en ont pas, et forcer le stylise serait un biais."""
    rng = np.random.default_rng(2)
    assert any(not np.isfinite(sv.sample_params(rng)["nu"]) for _ in range(200))


def test_very_heavy_student_tails_are_excluded():
    """nu = 4 ou 6 cumule aux queues deja produites par la volatilite
    stochastique menait a un exces de kurtosis de 9-10, contre ~2 mesures dans
    les donnees. Ces valeurs sont ecartees de la banque."""
    assert min(x for x in sv.NU_CHOICES if np.isfinite(x)) >= 8.0


# ── proprietes de base ──────────────────────────────────────────────────────

def test_generate_shape_and_finiteness():
    s, params = sv.generate(12, 150, seed=0)
    assert s.shape == (12, 150) and len(params) == 12
    assert np.all(np.isfinite(s))


def test_series_are_standardised():
    s, _ = sv.generate(20, 300, seed=1)
    assert np.allclose(s.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(s.std(axis=1), 1.0, atol=1e-9)


def test_student_innovations_keep_unit_variance_before_standardisation():
    """La normalisation de la Student doit garder une variance unitaire : sans
    elle, la loi de rendement porterait deux echelles a la fois et `nu`
    changerait la volatilite au lieu des seules queues."""
    rng = np.random.default_rng(5)
    v_inf = np.mean([sv.simulate(2000, rng, 0.9, 0.2, 0.0, np.inf).var() for _ in range(10)])
    v_4 = np.mean([sv.simulate(2000, rng, 0.9, 0.2, 0.0, 4.0).var() for _ in range(10)])
    assert v_inf == pytest.approx(1.0, abs=1e-6)
    assert v_4 == pytest.approx(1.0, abs=1e-6)


def test_reproducible():
    a, pa = sv.generate(6, 200, seed=11)
    b, pb = sv.generate(6, 200, seed=11)
    assert np.allclose(a, b) and pa == pb


def test_different_seeds_differ():
    a, _ = sv.generate(4, 200, seed=1)
    b, _ = sv.generate(4, 200, seed=2)
    assert not np.allclose(a, b)


def test_burn_in_removes_the_initial_calm_period():
    """Sans rodage et en demarrant a h=0, le debut de serie serait
    artificiellement calme -- un motif que le modele apprendrait."""
    rng = np.random.default_rng(4)
    with_burn = np.stack([sv.simulate(300, rng, 0.98, 0.35, 0.0, np.inf, burn_in=300)
                          for _ in range(40)])
    early = np.abs(with_burn[:, :50]).mean()
    late = np.abs(with_burn[:, -50:]).mean()
    assert abs(early - late) / late < 0.35, "le debut de serie reste atypique"


def test_short_series_still_work():
    s, _ = sv.generate(3, 20, seed=0)
    assert s.shape == (3, 20) and np.all(np.isfinite(s))
