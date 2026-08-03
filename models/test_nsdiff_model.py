"""
Unit tests for models/nsdiff_model.py -- the standalone NsDiff diffusion port.

Mirrors models/test_tsdiff_model.py's approach: kept deliberately tiny
(hidden_mean=hidden_sigma=hidden_denoise=8, epochs=1, n_samples=6, k_denoise=3)
so the pipeline's per-model test gate stays fast -- these check the *contract*
(output shapes, metric keys, prediction-interval sanity, non-regression of the
fit/forecast split) plus one calibration sanity test specific to NsDiff's own
value proposition (the conditional-variance network g_psi / UANS schedule).
"""

import numpy as np
import pandas as pd
import pytest

import nsdiff_model as nd


def _series(n=80, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx)


def _tiny(**kw):
    base = dict(hidden_mean=8, hidden_sigma=8, hidden_denoise=8, epochs=1,
                n_samples=6, k_denoise=3)
    base.update(kw)
    return base


def test_run_nsdiff_contract():
    nd.set_seed(0)
    s = _series()
    train, test = s.iloc[:60], s.iloc[60:]
    res = nd.run_nsdiff(train, test, **_tiny())

    for key in ("RMSE", "MAE", "Dir. Acc (%)", "PI Cov 95% (%)",
                "predictions", "lower", "upper", "index", "actual"):
        assert key in res, f"missing key {key}"

    assert len(res["predictions"]) == len(test)
    assert len(res["lower"]) == len(test) == len(res["upper"])
    assert np.all(np.isfinite(res["predictions"]))
    assert np.all(np.isfinite(res["lower"]))
    assert np.all(np.isfinite(res["upper"]))
    # prediction interval is well-ordered and brackets the point estimate
    assert np.all(res["lower"] <= res["predictions"])
    assert np.all(res["predictions"] <= res["upper"])
    # keep_samples=False (défaut, cf. _tiny) : pas de nuage conservé -- non-régression
    # pour les appelants existants (CLI standalone, next_step_nsdiff, experiments/).
    assert "ensemble" not in res


def test_run_nsdiff_keep_samples_populates_step_clouds():
    """keep_samples=True conserve le nuage n_samples déjà tiré à chaque pas --
    consommé par model_artifacts/crps_kpis.py pour le CRPS empirique."""
    nd.set_seed(0)
    s = _series()
    train, test = s.iloc[:60], s.iloc[60:]
    res = nd.run_nsdiff(train, test, keep_samples=True, **_tiny())

    assert "ensemble" in res
    assert len(res["ensemble"]) == len(test)
    for cloud in res["ensemble"]:
        cloud = np.asarray(cloud, dtype=float)
        assert cloud.shape == (6,)   # n_samples de _tiny()
        assert np.all(np.isfinite(cloud))


def test_next_step_nsdiff_ordered():
    nd.set_seed(1)
    s = _series()
    pred, lo, hi = nd.next_step_nsdiff(s, **_tiny())
    assert lo <= pred <= hi
    assert np.isfinite(pred)


def test_forecast_horizons_nsdiff_shape():
    # exercised by the pipeline's multi-horizon live forecast path
    from benchmarks import multi_horizon as mh
    nd.set_seed(2)
    s = _series()
    out = mh.forecast_horizons_nsdiff(s, [1, 7], seed=2, epochs=1)
    assert set(out) == {1, 7}
    for h, (point, lo, hi) in out.items():
        assert lo <= point <= hi
        assert np.isfinite(point)


def test_too_short_series_raises():
    nd.set_seed(3)
    short = _series(n=20)                      # < SEQ_LEN + HORIZON
    with pytest.raises(ValueError, match="seq_len"):
        nd.run_nsdiff(short.iloc[:15], short.iloc[15:], **_tiny())


# ── forecast_from_fitted: non-regression vs. an independent reference walk-forward ──

def _reference_walk_forward(train, test, seq_len, horizon, hidden_mean, hidden_sigma,
                            hidden_denoise, k_denoise, epochs, n_samples,
                            sigma_kernel=nd.SIGMA_KERNEL, batch_size=nd.BATCH_SIZE):
    """Frozen, independent reference implementation of the walk-forward loop
    (inlined, no forecast_from_fitted / fit_nsdiff reuse). Kept only to catch
    behavioural drift in run_nsdiff -- do not let it fall out of sync with the
    extraction, that's the point of this test (mirrors
    test_tsdiff_model.py's own _reference_walk_forward)."""
    train_p = train.values.astype(float)
    r = nd._log_returns(train_p)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = nd._make_windows(z, seq_len, horizon)
    model = nd.NsDiff(seq_len, horizon, hidden_mean, hidden_sigma, hidden_denoise,
                      sigma_kernel, T=k_denoise)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)

    buffer = list(z)
    last_price = float(train_p[-1])
    test_p = test.values.astype(float)

    preds, lower, upper = [], [], []
    for i in range(len(test_p)):
        window = np.asarray(buffer[-seq_len:], dtype=np.float32)
        z_samples = model.sample_next(window, n_samples=n_samples)
        r_samples = z_samples * sd + mu
        price_samples = last_price * np.exp(r_samples)
        preds.append(float(np.mean(price_samples)))
        lower.append(float(np.quantile(price_samples, 0.025)))
        upper.append(float(np.quantile(price_samples, 0.975)))
        realised_r = np.log(test_p[i] / last_price)
        buffer.append((realised_r - mu) / sd)
        last_price = float(test_p[i])
    return np.asarray(preds), np.asarray(lower), np.asarray(upper)


def test_forecast_from_fitted_matches_reference_walk_forward():
    """run_nsdiff (built on fit_nsdiff + forecast_from_fitted) must produce
    bit-identical output to the frozen independent reference, same seed/data."""
    kw = _tiny(epochs=2, n_samples=6, k_denoise=3)
    s = _series(n=80, seed=5)
    train, test = s.iloc[:60], s.iloc[60:]

    nd.set_seed(11)
    ref_preds, ref_lower, ref_upper = _reference_walk_forward(
        train, test, seq_len=nd.SEQ_LEN, horizon=nd.HORIZON, **kw)

    nd.set_seed(11)
    res = nd.run_nsdiff(train, test, **kw)

    assert np.array_equal(res["predictions"], ref_preds)
    assert np.array_equal(res["lower"], ref_lower)
    assert np.array_equal(res["upper"], ref_upper)


def test_forecast_from_fitted_direct_call_matches_run_nsdiff():
    """Calling fit_nsdiff + forecast_from_fitted by hand, one origin at a time,
    must reproduce run_nsdiff's walk-forward output exactly -- this is the
    actual usage pattern of the train-once-forward head-to-head protocol."""
    kw = _tiny(epochs=2, n_samples=6, k_denoise=3)
    fit_kw = {k: v for k, v in kw.items() if k not in ("n_samples",)}
    s = _series(n=80, seed=9)
    train, test = s.iloc[:60], s.iloc[60:]

    nd.set_seed(21)
    res = nd.run_nsdiff(train, test, **kw)

    nd.set_seed(21)
    model, mu, sd = nd.fit_nsdiff(train, seq_len=nd.SEQ_LEN, horizon=nd.HORIZON, **fit_kw)
    train_p = train.values.astype(float)
    buffer = list((nd._log_returns(train_p) - mu) / sd)
    last_price = float(train_p[-1])
    test_p = test.values.astype(float)

    preds, lower, upper = [], [], []
    for i in range(len(test_p)):
        price_samples = nd.forecast_from_fitted(
            model, buffer, mu, sd, last_price, horizons=[1], n_samples=kw["n_samples"])[1]
        preds.append(float(np.mean(price_samples)))
        lower.append(float(np.quantile(price_samples, 0.025)))
        upper.append(float(np.quantile(price_samples, 0.975)))
        realised_r = np.log(test_p[i] / last_price)
        buffer.append((realised_r - mu) / sd)
        last_price = float(test_p[i])

    assert np.array_equal(res["predictions"], np.asarray(preds))
    assert np.array_equal(res["lower"], np.asarray(lower))
    assert np.array_equal(res["upper"], np.asarray(upper))


def test_forecast_from_fitted_multi_horizon_shape():
    """horizons=[1,2,3] on a fitted model returns one price-sample array per
    horizon, each of length n_samples, without ever calling train() again."""
    kw = _tiny(epochs=2, n_samples=6, k_denoise=3)
    fit_kw = {k: v for k, v in kw.items() if k not in ("n_samples",)}
    s = _series(n=80, seed=13)
    nd.set_seed(31)
    model, mu, sd = nd.fit_nsdiff(s.iloc[:60], seq_len=nd.SEQ_LEN, horizon=nd.HORIZON, **fit_kw)
    z = (nd._log_returns(s.iloc[:60].values.astype(float)) - mu) / sd
    out = nd.forecast_from_fitted(model, z, mu, sd, float(s.iloc[59]),
                                  horizons=[1, 2, 3], n_samples=kw["n_samples"])
    assert set(out) == {1, 2, 3}
    for h, samples in out.items():
        assert samples.shape == (kw["n_samples"],)
        assert np.all(np.isfinite(samples))
    means = [float(np.mean(out[h])) for h in (1, 2, 3)]
    assert len(set(means)) > 1


# ── Anti-lookahead (equivalent of test_models_common.test_point_in_time_no_lookahead,
# which does not parametrize over TSDiff/NsDiff -- each diffusion port carries its
# own dedicated test file, cf. brief §5) ──

def test_point_in_time_no_lookahead():
    """Truncate the test window and re-predict: the shared prefix of predictions
    must match exactly -- run_nsdiff only ever looks at buffer entries up to the
    step being predicted (mu/sd frozen at T0, realised values fed back one at a
    time), so it cannot be peeking at test rows beyond that."""
    kw = _tiny(epochs=1, n_samples=6, k_denoise=3)
    s = _series(n=80, seed=7)
    train, test = s.iloc[:60], s.iloc[60:]

    nd.set_seed(0)
    full_result = nd.run_nsdiff(train, test, **kw)

    nd.set_seed(0)
    truncated_result = nd.run_nsdiff(train, test.iloc[:3], **kw)

    full_prefix = np.asarray(full_result["predictions"], dtype=float)[:3]
    truncated = np.asarray(truncated_result["predictions"], dtype=float)
    assert np.array_equal(truncated, full_prefix)


# ── Calibration sanity (NsDiff-specific value proposition: g_psi/UANS should
# widen the predictive interval when the underlying series is heteroscedastic,
# unlike a toy constant-variance model) ──

def _heteroscedastic_series(n=140, seed=42, low_sigma=0.003, high_sigma=0.035):
    """Synthetic series whose return volatility switches regime half-way
    through (low-vol first half, high-vol second half) -- a minimal,
    deliberately crude proxy for the "variance depends on history" property
    that g_psi/UANS are supposed to track (GARCH-like calibration, cf. module
    docstring)."""
    rng = np.random.default_rng(seed)
    sigmas = np.where(np.arange(n) < n // 2, low_sigma, high_sigma)
    returns = rng.normal(0, sigmas)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx)


def test_g_psi_predicts_higher_variance_for_high_vol_window():
    """NsDiff-specific value proposition (brief §5.8): g_psi should learn to
    output a higher conditional variance for a look-back window drawn from a
    high-volatility regime than for one drawn from a low-volatility regime of
    the SAME fitted model -- direct evidence the conditional-variance network
    tracks history-dependent volatility (the "GARCH-like" property that is
    this model's entire reason for existing, cf. module docstring), decoupled
    from the extra sampling noise a full walk-forward coverage stat would add.
    Single fixed seed/config (deliberately not looped over seeds): this probes
    a trained network's structural behaviour, not a stochastic outcome, so it
    should hold reliably at this one setting rather than needing a wide
    tolerance band."""
    nd.set_seed(0)
    s = _heteroscedastic_series()
    train = s.iloc[:100]
    model, mu, sd = nd.fit_nsdiff(train, hidden_mean=16, hidden_sigma=16,
                                  hidden_denoise=16, epochs=30, k_denoise=5)

    train_p = train.values.astype(float)
    z = (nd._log_returns(train_p) - mu) / sd
    low_vol_window  = z[10:10 + model.seq_len]   # entirely inside the low-vol first half
    high_vol_window = z[-model.seq_len:]         # entirely inside the high-vol second half

    import torch
    with torch.no_grad():
        x_low  = torch.tensor(low_vol_window,  dtype=torch.float32).view(1, -1, 1)
        x_high = torch.tensor(high_vol_window, dtype=torch.float32).view(1, -1, 1)
        g_low  = float(model.sigma_net(x_low).mean())
        g_high = float(model.sigma_net(x_high).mean())

    assert np.isfinite(g_low) and np.isfinite(g_high)
    assert g_low > 0.0 and g_high > 0.0
    assert g_high > g_low * 1.05


def test_heteroscedastic_run_nsdiff_produces_finite_valid_pi_coverage():
    """End-to-end sanity (loose, cf. brief §5.8 "garder ce test tolérant") on
    the same heteroscedastic series: the full walk-forward run_nsdiff still
    produces a well-defined, finite 95% PI coverage percentage -- this exercises
    the whole pipeline on non-stationary data without asserting a precise
    calibration number (which the dedicated g_psi test above already covers
    more directly and stably)."""
    nd.set_seed(0)
    s = _heteroscedastic_series()
    train, test = s.iloc[:100], s.iloc[100:]
    res = nd.run_nsdiff(train, test, **_tiny(epochs=3, n_samples=30, k_denoise=5))

    pi_cov = res["PI Cov 95% (%)"]
    assert not isinstance(pi_cov, str)  # not "N/A"
    assert 0.0 <= pi_cov <= 100.0
