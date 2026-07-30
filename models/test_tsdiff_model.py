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
    # keep_samples=False (défaut, cf. _tiny) : pas de nuage conservé -- non-régression
    # pour les appelants existants (CLI standalone, next_step_tsdiff, experiments/).
    assert "ensemble" not in res


def test_run_tsdiff_keep_samples_populates_step_clouds():
    """keep_samples=True conserve le nuage n_samples déjà tiré à chaque pas (au lieu
    de le réduire à mean/quantiles puis le jeter) -- consommé par
    model_artifacts/crps_kpis.py pour le CRPS empirique."""
    td.set_seed(0)
    s = _series()
    train, test = s.iloc[:60], s.iloc[60:]
    res = td.run_tsdiff(train, test, keep_samples=True, **_tiny())

    assert "ensemble" in res
    assert len(res["ensemble"]) == len(test)
    for cloud in res["ensemble"]:
        cloud = np.asarray(cloud, dtype=float)
        assert cloud.shape == (6,)   # n_samples de _tiny()
        assert np.all(np.isfinite(cloud))


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


# ── forecast_from_fitted: non-regression vs. the pre-refactor run_tsdiff logic ──

def _reference_walk_forward(train, test, seq_len, horizon, hidden, depth, cond_dim,
                            T, epochs, k_denoise, n_samples, batch_size=td.BATCH_SIZE,
                            ddim_eta=td.DDIM_ETA):
    """Frozen copy of run_tsdiff's pre-refactor body (train + walk-forward loop
    inlined, no forecast_from_fitted / fit_tsdiff). Kept only as an independent
    reference to catch behavioural drift in the refactored run_tsdiff — do not
    let it fall out of sync with the extraction, that's the point of this test."""
    train_p = train.values.astype(float)
    r = td._log_returns(train_p)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = td._make_windows(z, seq_len, horizon)
    model = td.TSDiff(seq_len, horizon, hidden, depth, cond_dim, T)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)

    buffer = list(z)
    last_price = float(train_p[-1])
    test_p = test.values.astype(float)

    preds, lower, upper = [], [], []
    for i in range(len(test_p)):
        window = np.asarray(buffer[-seq_len:], dtype=np.float32)
        z_samples = model.sample_next(window, n_samples=n_samples,
                                      k_denoise=k_denoise, ddim_eta=ddim_eta)
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
    """run_tsdiff (now built on fit_tsdiff + forecast_from_fitted) must produce
    bit-identical output to the frozen pre-refactor reference, same seed/data."""
    kw = _tiny(epochs=2, n_samples=6, k_denoise=3)
    s = _series(n=80, seed=5)
    train, test = s.iloc[:60], s.iloc[60:]

    td.set_seed(11)
    ref_preds, ref_lower, ref_upper = _reference_walk_forward(
        train, test, seq_len=td.SEQ_LEN, horizon=td.HORIZON, cond_dim=td.COND_DIM,
        T=td.T_DIFFUSION, **kw)

    td.set_seed(11)
    res = td.run_tsdiff(train, test, **kw)

    assert np.array_equal(res["predictions"], ref_preds)
    assert np.array_equal(res["lower"], ref_lower)
    assert np.array_equal(res["upper"], ref_upper)


def test_forecast_from_fitted_direct_call_matches_run_tsdiff():
    """Calling fit_tsdiff + forecast_from_fitted by hand, one origin at a time,
    must reproduce run_tsdiff's walk-forward output exactly — this is the actual
    usage pattern of the train-once-forward head-to-head protocol."""
    kw = _tiny(epochs=2, n_samples=6, k_denoise=3)
    fit_kw = {k: v for k, v in kw.items() if k not in ("n_samples", "k_denoise")}
    s = _series(n=80, seed=9)
    train, test = s.iloc[:60], s.iloc[60:]

    td.set_seed(21)
    res = td.run_tsdiff(train, test, **kw)

    td.set_seed(21)
    model, mu, sd = td.fit_tsdiff(train, seq_len=td.SEQ_LEN, horizon=td.HORIZON,
                                  cond_dim=td.COND_DIM, T=td.T_DIFFUSION, **fit_kw)
    train_p = train.values.astype(float)
    buffer = list((td._log_returns(train_p) - mu) / sd)
    last_price = float(train_p[-1])
    test_p = test.values.astype(float)

    preds, lower, upper = [], [], []
    for i in range(len(test_p)):
        price_samples = td.forecast_from_fitted(
            model, buffer, mu, sd, last_price, horizons=[1],
            n_samples=kw["n_samples"], k_denoise=kw["k_denoise"])[1]
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
    fit_kw = {k: v for k, v in kw.items() if k not in ("n_samples", "k_denoise")}
    s = _series(n=80, seed=13)
    td.set_seed(31)
    model, mu, sd = td.fit_tsdiff(s.iloc[:60], seq_len=td.SEQ_LEN, horizon=td.HORIZON,
                                  cond_dim=td.COND_DIM, T=td.T_DIFFUSION, **fit_kw)
    z = (td._log_returns(s.iloc[:60].values.astype(float)) - mu) / sd
    out = td.forecast_from_fitted(model, z, mu, sd, float(s.iloc[59]),
                                  horizons=[1, 2, 3], n_samples=kw["n_samples"],
                                  k_denoise=kw["k_denoise"])
    assert set(out) == {1, 2, 3}
    for h, samples in out.items():
        assert samples.shape == (kw["n_samples"],)
        assert np.all(np.isfinite(samples))
    # cumulative horizons diverge from a common last_price: means should differ
    means = [float(np.mean(out[h])) for h in (1, 2, 3)]
    assert len(set(means)) > 1
