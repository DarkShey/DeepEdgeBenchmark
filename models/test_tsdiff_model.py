"""
Unit tests for models/tsdiff_model.py — the standalone TSDiff diffusion port.

Kept deliberately tiny (hidden=8, depth=1, epochs=1, few samples) so the
pipeline's per-model test gate stays fast: these check the *contract*, not
forecast quality — output shapes, metric keys, and prediction-interval sanity.
"""

import numpy as np
import pandas as pd
import pytest

import tsdiff_model as td


def _series(n=80, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx)


def _tiny(**kw):
    base = dict(hidden=8, depth=1, epochs=1, n_samples=6, k_denoise=3)
    base.update(kw)
    return base


def test_run_tsdiff_contract():
    td.set_seed(0)
    s = _series()
    train, test = s.iloc[:60], s.iloc[60:]
    res = td.run_tsdiff(train, test, **_tiny())

    for key in ("RMSE", "MAE", "Dir. Acc (%)", "PI Cov 95% (%)",
                "predictions", "lower", "upper", "index", "actual"):
        assert key in res, f"missing key {key}"

    assert len(res["predictions"]) == len(test)
    assert len(res["lower"]) == len(test) == len(res["upper"])
    assert np.all(np.isfinite(res["predictions"]))
    # prediction interval is well-ordered and brackets the point estimate
    assert np.all(res["lower"] <= res["predictions"])
    assert np.all(res["predictions"] <= res["upper"])


def test_next_step_tsdiff_ordered():
    td.set_seed(1)
    s = _series()
    pred, lo, hi = td.next_step_tsdiff(s, **_tiny())
    assert lo <= pred <= hi
    assert np.isfinite(pred)


def test_forecast_horizons_tsdiff_shape():
    # exercised by the pipeline's multi-horizon live forecast path
    from benchmarks import multi_horizon as mh
    td.set_seed(2)
    s = _series()
    out = mh.forecast_horizons_tsdiff(s, [1, 7], seed=2, epochs=1)
    assert set(out) == {1, 7}
    for h, (point, lo, hi) in out.items():
        assert lo <= point <= hi
        assert np.isfinite(point)


def test_too_short_series_raises():
    td.set_seed(3)
    short = _series(n=20)                      # < SEQ_LEN + HORIZON
    with pytest.raises(ValueError):
        td.run_tsdiff(short.iloc[:15], short.iloc[15:], **_tiny())
