"""Targeted tests for the Prophet D+7 adapter (benchmarks/multi_horizon.py) --
log-space fit + optional sigma_scale correction, adopted 2026-07-31 following
the same fix already validated for D+1 (models/prophet_model.py, cf.
HANDOFF_sigma_calibration_suivi.md §4). No existing test file covered this
module before; kept minimal (Prophet fits are the slow part here) rather than
exercising all 5 models -- see models/test_*.py for per-model coverage."""

import numpy as np
import pandas as pd
import pytest

from benchmarks import multi_horizon as mh


def _daily_series(n=250, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.Series(prices, index=idx)


def test_fit_prophet_uses_log_price():
    prophet = pytest.importorskip("prophet_model")
    train = _daily_series()
    model = mh.fit_prophet(train)
    # history["y"] is what Prophet was actually fit on -- must be log(price), not
    # the raw price, or the whole point of this fix (stationary-ish trend on a
    # non-stationary growth series) silently regresses.
    fitted_y = model.history["y"].to_numpy()
    assert np.allclose(fitted_y, np.log(train.values), atol=1e-6)


def test_forecast_from_fitted_prophet_bounds_sane():
    prophet = pytest.importorskip("prophet_model")
    train = _daily_series()
    model = mh.fit_prophet(train)
    out = mh.forecast_from_fitted_prophet(model, train.index[-1], [1, 3, 7])
    assert set(out) == {1, 3, 7}
    for h, (point, lo, hi) in out.items():
        assert lo <= point <= hi
        assert np.isfinite([point, lo, hi]).all()
        assert point > 0   # log-space round-trip: must come back as a real price


def test_forecast_from_fitted_prophet_sigma_scale_widens_interval():
    prophet = pytest.importorskip("prophet_model")
    train = _daily_series()
    model = mh.fit_prophet(train)
    horizons = [1, 5]
    base = mh.forecast_from_fitted_prophet(model, train.index[-1], horizons)
    wide = mh.forecast_from_fitted_prophet(model, train.index[-1], horizons,
                                           sigma_scale=2.0)
    narrow = mh.forecast_from_fitted_prophet(model, train.index[-1], horizons,
                                             sigma_scale=0.5)
    for h in horizons:
        base_w = base[h][2] - base[h][1]
        wide_w = wide[h][2] - wide[h][1]
        narrow_w = narrow[h][2] - narrow[h][1]
        assert wide_w > base_w > narrow_w
        # point forecast itself must be untouched by the width-only correction
        assert base[h][0] == pytest.approx(wide[h][0], rel=1e-9)
        assert base[h][0] == pytest.approx(narrow[h][0], rel=1e-9)


def test_forecast_horizons_prophet_default_matches_no_sigma_scale():
    prophet = pytest.importorskip("prophet_model")
    train = _daily_series()
    # sigma_scale defaults to None at every level (fit_prophet -> forecast_horizons_prophet
    # -> forecast_from_fitted_prophet) -- existing callers (model_artifacts/pipeline.py,
    # weekly_multimodel.py's REGIME_B_FORECAST) call with 2 positional args only and must
    # see unchanged behaviour. Reuses ONE fit for both paths (rather than fitting twice)
    # so the comparison isn't sensitive to any nondeterminism in Prophet's own MAP fit.
    #
    # NOTE: Model.predict() itself re-samples trend uncertainty (np.random.poisson/
    # laplace inside Prophet's own sample_predictive_trend, no seed) on EVERY call,
    # even against the same fitted model -- yhat (point) is deterministic, yhat_lower/
    # yhat_upper are not, by a few percent between two calls. Confirmed here, not a
    # bug introduced by this change -- exact equality on the bounds is the wrong
    # assertion; a generous relative tolerance is what's actually being verified
    # (that sigma_scale=None truly is a no-op, not that Prophet is deterministic).
    model = mh.fit_prophet(train)
    out_via_default_kwarg = mh.forecast_from_fitted_prophet(model, train.index[-1], [1, 2])
    out_explicit_none = mh.forecast_from_fitted_prophet(model, train.index[-1], [1, 2],
                                                         sigma_scale=None)
    for h in (1, 2):
        p1, lo1, hi1 = out_via_default_kwarg[h]
        p2, lo2, hi2 = out_explicit_none[h]
        assert p1 == pytest.approx(p2, rel=1e-9)   # point forecast IS deterministic
        assert lo1 == pytest.approx(lo2, rel=0.15)
        assert hi1 == pytest.approx(hi2, rel=0.15)
