"""Tests de `econ_backtest.py` -- le moteur de backtest economique.

Trois familles de tests, dans cet ordre d'importance :
  1. CAUSALITE (aucune fonction ne regarde le futur) -- c'est le seul defaut qui
     rendrait tous les chiffres du chantier B faux sans qu'on s'en apercoive ;
  2. cas calcules a la main pour chaque formule ;
  3. entrees degenerees et invariants.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import econ_backtest as eb                                            # noqa: E402


# ── 1. causalite : le test qui compte ───────────────────────────────────────

def test_expanding_median_never_sees_the_present_or_the_future():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = eb.expanding_median(x, warmup=2)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(np.median(x[:2]))      # 1.5, PAS 2.0
    assert out[3] == pytest.approx(np.median(x[:3]))      # 2.0
    assert out[5] == pytest.approx(np.median(x[:5]))      # 3.0


def test_positions_are_unchanged_when_the_future_is_altered():
    """Test de causalite de bout en bout : on modifie tout ce qui suit
    l'instant t et on verifie que la position en t ne bouge pas. Si une
    strategie fuitait (mediane de la fenetre entiere, par exemple), ce test
    tomberait immediatement."""
    n = 20
    rng = np.random.default_rng(0)
    base = {
        "last_close": np.full(n, 100.0),
        "y_pred": 100.0 + rng.normal(0, 1, n),
        "y_lower": 100.0 - rng.uniform(2, 8, n),
        "y_upper": 100.0 + rng.uniform(2, 8, n),
    }
    cut = 12
    tampered = {k: v.copy() for k, v in base.items()}
    for k in ("y_pred", "y_lower", "y_upper"):
        tampered[k][cut:] += rng.normal(0, 50, n - cut)     # futur saccage

    for name, spec in eb.STRATEGIES.items():
        a = np.asarray(spec["fn"](base))[:cut]
        b = np.asarray(spec["fn"](tampered))[:cut]
        assert np.allclose(a, b), f"{name} regarde le futur"


def test_warmup_origins_hold_no_position():
    n = 12
    data = {"last_close": np.full(n, 100.0), "y_pred": np.full(n, 101.0),
            "y_lower": np.full(n, 95.0), "y_upper": np.full(n, 105.0)}
    w = eb.positions_inverse_width(**data)
    assert np.all(w[:eb.WARMUP_ORIGINS] == 0.0)
    assert np.all(w[eb.WARMUP_ORIGINS:] != 0.0)


# ── 2. cas calcules a la main ───────────────────────────────────────────────

def test_inverse_width_sizing_hand_calc():
    """8 origines de chauffe a largeur relative 0.10, puis une origine a 0.05.
    A t=8 : mediane_{<8} = 0.10, largeur = 0.05 -> taille 2.0, plafonnee a 1.0.
    Signe : y_pred 101 > 100 -> long."""
    n = 9
    lc = np.full(n, 100.0)
    lo = np.concatenate([np.full(8, 95.0), [97.5]])
    hi = np.concatenate([np.full(8, 105.0), [102.5]])
    w = eb.positions_inverse_width(lc, np.full(n, 101.0), lo, hi)
    assert w[8] == pytest.approx(1.0)


def test_inverse_width_sizes_down_when_less_confident():
    """Symetrique : une largeur DEUX FOIS l'echelle -> demi-position."""
    n = 9
    lc = np.full(n, 100.0)
    lo = np.concatenate([np.full(8, 95.0), [90.0]])
    hi = np.concatenate([np.full(8, 105.0), [110.0]])
    w = eb.positions_inverse_width(lc, np.full(n, 101.0), lo, hi)
    assert w[8] == pytest.approx(0.5)


def test_inverse_width_sign_follows_the_point_forecast():
    n = 10
    lc = np.full(n, 100.0)
    lo, hi = np.full(n, 95.0), np.full(n, 105.0)
    long_ = eb.positions_inverse_width(lc, np.full(n, 101.0), lo, hi)
    short = eb.positions_inverse_width(lc, np.full(n, 99.0), lo, hi)
    assert long_[-1] > 0 and short[-1] < 0
    assert long_[-1] == pytest.approx(-short[-1])


def test_var_from_lower_hand_calc():
    var = eb.var_from_lower([100.0, 200.0], [95.0, 190.0])
    assert var == pytest.approx([-0.05, -0.05])


def test_var_limit_sizing_hand_calc():
    """budget 3 %. VaR -6 % -> w = 0.5 ; VaR -3 % -> w = 1.0 ; VaR -1 % -> 3.0
    plafonne a 1.0."""
    w = eb.positions_var_limit([100.0, 100.0, 100.0], [94.0, 97.0, 99.0], budget=0.03)
    assert w == pytest.approx([0.5, 1.0, 1.0])


def test_var_limit_is_long_only():
    w = eb.positions_var_limit(np.full(5, 100.0), np.full(5, 90.0))
    assert np.all(w >= 0)


def test_filtered_direction_takes_position_only_when_pi_excludes_zero():
    lc = np.array([100.0, 100.0, 100.0, 100.0])
    lo = np.array([101.0, 95.0, 90.0, 100.0])       # au-dessus / a cheval / en dessous / touche
    hi = np.array([105.0, 105.0, 99.0, 100.0])
    w = eb.positions_filtered_direction(lc, lo, hi)
    assert w == pytest.approx([1.0, 0.0, -1.0, 0.0])


def test_filtered_direction_never_trades_with_a_very_wide_pi():
    n = 30
    lc = np.full(n, 100.0)
    w = eb.positions_filtered_direction(lc, np.full(n, 1.0), np.full(n, 1000.0))
    assert np.all(w == 0.0)


def test_sleeve_pnl_hand_calc():
    """w=0.5, r=+10 %, cout 5 bps aller simple -> 0.5*0.10 - 2*0.0005*0.5
    = 0.05 - 0.0005 = 0.0495."""
    pnl = eb.sleeve_pnl([0.5], [0.10], cost_bps=5.0)
    assert pnl[0] == pytest.approx(0.0495)


def test_sleeve_pnl_costs_a_short_the_same_as_a_long():
    long_ = eb.sleeve_pnl([1.0], [0.0], cost_bps=10.0)[0]
    short = eb.sleeve_pnl([-1.0], [0.0], cost_bps=10.0)[0]
    assert long_ == pytest.approx(short) == pytest.approx(-0.002)


def test_zero_position_costs_nothing():
    assert eb.sleeve_pnl([0.0], [0.5], cost_bps=100.0)[0] == 0.0


def test_max_drawdown_hand_calc():
    """PnL [+1, +1, -3, +1] -> cumul [1, 2, -1, 0], pic courant [1, 2, 2, 2],
    ecart min = -1 - 2 = -3."""
    assert eb.max_drawdown([1.0, 1.0, -3.0, 1.0]) == pytest.approx(-3.0)


def test_max_drawdown_is_zero_for_a_monotone_curve():
    assert eb.max_drawdown([0.1, 0.2, 0.3]) == pytest.approx(0.0)


def test_max_drawdown_counts_a_loss_from_the_very_start():
    assert eb.max_drawdown([-0.5, -0.2]) == pytest.approx(-0.7)


def test_sharpe_annualisation_depends_on_the_horizon():
    pnl = [0.01, 0.02, 0.00, 0.03, -0.01]
    s1 = eb.sharpe(pnl, horizon_weeks=1)
    s3 = eb.sharpe(pnl, horizon_weeks=3)
    assert s1 / s3 == pytest.approx(np.sqrt(3.0))


def test_sharpe_hand_calc():
    pnl = np.array([0.01, 0.03])
    expected = pnl.mean() / pnl.std(ddof=1) * np.sqrt(52.0)
    assert eb.sharpe(pnl, 1) == pytest.approx(expected)


def test_sharpe_is_nan_when_degenerate():
    assert np.isnan(eb.sharpe([0.01], 1))
    assert np.isnan(eb.sharpe([0.01, 0.01], 1))


def test_var_diagnostics_hand_calc():
    """VaR = -5 % partout. Rendements : -3 % (ok), -8 % (violation, exces 3 pts),
    -5 % (pile, pas une violation : strictement inferieur requis), +2 % (ok)."""
    d = eb.var_diagnostics([-0.03, -0.08, -0.05, 0.02], [-0.05] * 4)
    assert d["n_breaches"] == 1
    assert d["breach_rate"] == pytest.approx(0.25)
    assert d["total_excess_loss"] == pytest.approx(0.03)
    assert d["mean_excess_loss_given_breach"] == pytest.approx(0.03)
    assert d["worst_excess_loss"] == pytest.approx(0.03)
    assert d["breach_flags"] == [0, 1, 0, 0]


def test_var_diagnostics_with_no_breach():
    d = eb.var_diagnostics([0.01, 0.02], [-0.05, -0.05])
    assert d["n_breaches"] == 0
    assert d["total_excess_loss"] == 0.0
    assert d["mean_excess_loss_given_breach"] == 0.0


def test_gross_returns_hand_calc():
    assert eb.gross_returns([100.0, 50.0], [110.0, 45.0]) == pytest.approx([0.10, -0.10])


# ── 3. invariants et entrees degenerees ─────────────────────────────────────

def test_no_leverage_anywhere():
    n = 40
    rng = np.random.default_rng(7)
    data = {
        "last_close": np.full(n, 100.0),
        "y_pred": 100.0 + rng.normal(0, 3, n),
        "y_lower": 100.0 - rng.uniform(0.1, 20, n),
        "y_upper": 100.0 + rng.uniform(0.1, 20, n),
    }
    for name, spec in eb.STRATEGIES.items():
        w = np.asarray(spec["fn"](data))
        assert np.all(np.abs(w) <= eb.W_MAX + 1e-12), f"{name} depasse w_max"


def test_a_degenerate_zero_width_pi_does_not_produce_infinite_size():
    n = 12
    lc = np.full(n, 100.0)
    lo = np.concatenate([np.full(10, 95.0), [100.0, 100.0]])   # largeur nulle a la fin
    hi = np.concatenate([np.full(10, 105.0), [100.0, 100.0]])
    w = eb.positions_inverse_width(lc, np.full(n, 101.0), lo, hi)
    assert np.all(np.isfinite(w)) and np.all(np.abs(w) <= 1.0)


def test_var_limit_handles_zero_var():
    w = eb.positions_var_limit([100.0], [100.0])
    assert np.isfinite(w[0]) and w[0] == pytest.approx(eb.W_MAX)


def test_sleeve_pnl_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        eb.sleeve_pnl([1.0, 1.0], [0.1], cost_bps=5.0)


def test_run_strategy_rejects_unknown_strategy():
    with pytest.raises(KeyError):
        eb.run_strategy("martingale", {}, 5.0, 1)


def test_run_strategy_end_to_end():
    n = 15
    data = {
        "last_close": np.full(n, 100.0),
        "y_pred": np.full(n, 101.0),
        "y_lower": np.full(n, 95.0),
        "y_upper": np.full(n, 105.0),
        "y_true": np.full(n, 102.0),
    }
    out = eb.run_strategy("inverse_width", data, cost_bps=5.0, horizon_weeks=1)
    assert out["n"] == n
    assert len(out["positions"]) == n and len(out["pnl_series"]) == n
    assert out["n_active_origins"] == n - eb.WARMUP_ORIGINS
    # positions actives = 1.0 (largeur constante = son echelle), r = +2 %,
    # cout aller-retour 10 bps -> 0.02 - 0.001 = 0.019 par origine active
    assert out["pnl_total"] == pytest.approx(0.019 * (n - eb.WARMUP_ORIGINS))
    assert out["hit_rate_when_active"] == pytest.approx(1.0)


def test_cost_levels_are_ordered_and_cover_every_asset_class():
    classes = {"index", "bond", "crypto"}
    for level, table in eb.COST_LEVELS.items():
        assert set(table) == classes, f"{level} ne couvre pas toutes les classes"
    for cls in classes:
        vals = [eb.COST_LEVELS[lvl][cls] for lvl in ("faible", "central", "eleve")]
        assert vals == sorted(vals), f"niveaux de cout non ordonnes pour {cls}"
    for lvl in eb.COST_LEVELS:
        assert eb.COST_LEVELS[lvl]["crypto"] > eb.COST_LEVELS[lvl]["index"]


def test_buy_and_hold_is_always_fully_invested():
    assert np.all(eb.positions_buy_and_hold(np.full(5, 100.0)) == eb.W_MAX)
