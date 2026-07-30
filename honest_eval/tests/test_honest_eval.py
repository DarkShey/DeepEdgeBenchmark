"""
Offline unit tests for honest_eval (no network, no heavy models).
Run:  python -m honest_eval.tests.test_honest_eval
Validates the statistical math on synthetic series so every claim in the brief
is checkable without downloading data or training a net.
"""

import sys
import numpy as np
import pandas as pd

from honest_eval import metrics, naive, validation, multistep, targets


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))


def rng():
    return np.random.default_rng(7)


def synth_prices(n=800, sigma=1.0, drift=0.0, s0=100.0):
    r = rng().normal(drift, sigma, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(s0 + np.cumsum(r), index=idx)


# ── Point 0: naive baseline ──────────────────────────────────────────────────

def test_naive():
    print("\n[Point 0] correct naive baseline")
    p = synth_prices()
    tr, te = p.iloc[:680], p.iloc[680:]
    ref = naive.naive_random_walk(tr, te)
    # prediction == previous close exactly
    expected_prev = np.concatenate([[tr.iloc[-1]], te.values[:-1]])
    check("naive == previous close", np.allclose(ref["predictions"], expected_prev))

    # verifier passes on the correct baseline, fails on a noise-injected one
    rep_ok = naive.verify_naive(tr, te, dashboard_predictions=ref["predictions"])
    check("verify_naive passes on clean baseline", rep_ok["passed"], str(rep_ok))

    noisy = ref["predictions"] + rng().normal(0, 0.03 * te.values.mean(), len(te))
    rep_bad = naive.verify_naive(tr, te, dashboard_predictions=noisy)
    check("verify_naive flags noise injection", not rep_bad["passed"])

    rep_rmse = naive.verify_naive(tr, te, dashboard_rmse=ref["sigma"] * 10)
    check("verify_naive flags inflated RMSE", not rep_rmse["passed"])


# ── Point 1: honest metrics ──────────────────────────────────────────────────

def test_metrics():
    print("\n[Point 1] variation-based metrics")
    p = synth_prices(n=500)
    actual = p.values[1:]
    prev = p.values[:-1]

    # naive predicts prev → Theil U == 1, MASE == 1 exactly
    check("Theil U(naive)==1", abs(metrics.theil_u(actual, prev, prev) - 1.0) < 1e-9)
    check("MASE(naive)==1", abs(metrics.mase(actual, prev, prev) - 1.0) < 1e-9)

    # a perfect forecast → U == 0
    check("Theil U(perfect)==0", metrics.theil_u(actual, actual, prev) < 1e-9)

    # change correlation of naive is nan (zero-variance Δpred)
    check("change_corr(naive) is nan", np.isnan(metrics.change_correlation(prev, prev, actual)))

    # a forecast correlated with the true change → positive change_corr
    true_change = actual - prev
    good_pred = prev + 0.5 * true_change
    check("change_corr(good)>0.4", metrics.change_correlation(good_pred, prev, actual) > 0.4)

    # directional accuracy: perfect-direction pred ~ 100%, coin ~ 50% within CI
    da_good = metrics.directional_accuracy(good_pred, prev, actual)
    check("DirAcc(good) high & sig", da_good["acc"] > 0.9 and da_good["p_vs_coin"] < 0.05,
          str(da_good))
    coin_pred = prev + rng().normal(0, 1, len(prev))
    da_coin = metrics.directional_accuracy(coin_pred, prev, actual)
    lo, hi = da_coin["ci95"]
    check("DirAcc(coin) CI contains 0.5", lo <= 0.5 <= hi, str(da_coin))

    # DM: identical errors → not significant; clearly better model → significant, DM<0
    ea = rng().normal(0, 1, 300)
    dm, dp, lag = metrics.diebold_mariano(ea, ea.copy(), h=1)
    check("DM(identical) not significant", dp > 0.99)
    better = ea * 0.3
    dm2, dp2, _ = metrics.diebold_mariano(better, ea, h=1)
    check("DM(better) significant & negative", dp2 < 0.05 and dm2 < 0, f"dm={dm2:.2f} p={dp2:.3f}")

    # DM Newey-West lag grows with horizon
    _, _, lag7 = metrics.diebold_mariano(better, ea, h=7)
    check("DM lag >= h-1 for h=7", lag7 >= 6, f"lag={lag7}")

    # dm_hac_test on a pre-computed differential (BRIEF_analyse_poolee.md) must
    # agree exactly with diebold_mariano()'s own internal loss differential --
    # it's the same computation, just fed the diff directly instead of two raw
    # error series (pooled/scaled differentials have no "raw error" to derive from).
    diff_sq = better**2 - ea**2
    dm3, dp3, lag3 = metrics.diebold_mariano(better, ea, h=1)
    out3 = metrics.dm_hac_test(diff_sq, h=1)
    check("dm_hac_test matches diebold_mariano on same squared-loss diff",
         abs(out3["dm_stat"] - dm3) < 1e-9 and abs(out3["p_value"] - dp3) < 1e-9
         and out3["lag"] == lag3, str(out3))
    check("dm_hac_test(identical) not significant",
         metrics.dm_hac_test(np.zeros(300), h=1)["p_value"] > 0.99)
    check("dm_hac_test(clear negative diff) significant & negative",
         metrics.dm_hac_test(better**2 - ea**2, h=1)["p_value"] < 0.05
         and metrics.dm_hac_test(better**2 - ea**2, h=1)["dm_stat"] < 0)
    check("dm_hac_test short series (T<8) returns p=1.0, no crash",
         metrics.dm_hac_test(np.array([1.0, -1.0, 2.0]), h=1)["p_value"] == 1.0)

    # skill verdict wording
    check("verdict no-skill", metrics.skill_verdict(1.0, 0.5) == "no better than naive")
    check("verdict beats", metrics.skill_verdict(0.8, 0.01) == "beats naive")


# ── probabilistic / vol / direction metrics ──────────────────────────────────

def test_prob_metrics():
    print("\n[Point 4 metrics] CRPS / PIT / QLIKE / Brier / AUC")
    r = rng()
    y = r.normal(0, 2, 5000)
    # PIT of a well-specified Gaussian is ~uniform
    pit = metrics.pit_values(np.zeros_like(y), np.full_like(y, 2.0), y)
    cal = metrics.pit_uniformity(pit)
    check("PIT uniform when calibrated (p>0.05)", cal["p"] > 0.05, str(cal))
    # mis-specified sigma → rejected
    pit_bad = metrics.pit_values(np.zeros_like(y), np.full_like(y, 0.5), y)
    check("PIT rejects miscalibration", metrics.pit_uniformity(pit_bad)["p"] < 0.05)

    # QLIKE minimised at the true variance
    rv = y ** 2
    ql_true = metrics.qlike(rv, np.full_like(rv, 4.0))
    ql_wrong = metrics.qlike(rv, np.full_like(rv, 16.0))
    check("QLIKE lower at truer variance", ql_true < ql_wrong)

    # Brier: perfect prob=1/0 → 0; coin 0.5 → 0.25
    yb = (y > 0).astype(int)
    check("Brier(perfect)==0", metrics.brier(yb.astype(float), yb) < 1e-12)
    check("Brier(coin)~0.25", abs(metrics.brier(np.full_like(yb, 0.5, float), yb) - 0.25) < 0.02)

    # AUC: informative score > 0.5, random ~0.5
    score = yb + r.normal(0, 0.3, len(yb))
    check("AUC informative > 0.7", metrics.roc_auc(score, yb) > 0.7)
    check("AUC(coin)~0.5", abs(metrics.roc_auc(r.normal(size=len(yb)), yb) - 0.5) < 0.05)


# ── Point 2: validation ──────────────────────────────────────────────────────

def test_validation():
    print("\n[Point 2] walk-forward + purged CV + subperiods")
    p = synth_prices(n=600)

    # expanding vs rolling both produce evaluations; naive fit_predict
    def fp_naive(train_prices, h):
        return np.repeat(train_prices[-1], h)   # random walk

    cmp = validation.compare_windows(p.values, fp_naive, test_start=400,
                                     windows=(None, 100, 250), step=1, horizon=1)
    check("compare_windows returns all settings", len(cmp) == 3)
    check("dense walk-forward n>=150", all(c.n >= 150 for c in cmp), str([c.n for c in cmp]))

    # purged K-fold: no train index falls within purge+embargo of the test block
    n, purge, emb = 200, 5, 3
    ok = True
    for tr, te in validation.purged_kfold_splits(n, n_splits=5, embargo=emb, purge=purge):
        lo, hi = te.min() - purge, te.max() + purge + emb
        if np.any((tr >= lo) & (tr <= hi)):
            ok = False
    check("purged folds exclude purge+embargo band", ok)
    # folds are contiguous (no shuffling)
    _, te0 = next(validation.purged_kfold_splits(100, n_splits=5))
    check("test folds contiguous", np.all(np.diff(te0) == 1))

    # subperiod report groups by quarter
    idx = p.index[1:]
    rep = validation.subperiod_report(idx, p.values[1:], p.values[:-1], by="quarter")
    check("subperiod report non-empty & has rmse", "rmse" in rep.columns and len(rep) > 1)

    regimes = validation.volatility_regimes(p.values, window=20, n_regimes=3)
    check("vol regimes labelled", set(np.unique(regimes)) <= {"low", "med", "high"})


# ── Point 3: multistep ───────────────────────────────────────────────────────

def test_multistep():
    print("\n[Point 3] dense rolling-origin multi-horizon")
    p = synth_prices(n=700, sigma=1.0)

    # a cheap analytic model: naive drift (should tie the naive benchmark)
    scores = multistep.evaluate_model_multih(
        p, multistep.naive_forecaster, horizons=(1, 7, 30),
        test_start=500, step=1)
    n7 = scores[7]["n"]
    check("D+7 dense n>=150", n7 >= 150, f"n={n7}")
    check("DM uses NW lag>=h-1 at h=7", scores[7]["dm_lag"] >= 6)
    # naive-vs-naive → Theil U ~ 1, not significant
    check("naive model ties naive (U~1)", abs(scores[7]["theil_u"] - 1.0) < 0.05,
          f"U={scores[7]['theil_u']}")

    # error grows with horizon for a random walk (RMSE_h ~ sqrt(h))
    curve = multistep.error_vs_horizon(p, multistep.naive_forecaster,
                                       horizons=(1, 5, 10, 20), test_start=500)
    rmse_1, rmse_20 = curve.loc[1, "rmse"], curve.loc[20, "rmse"]
    check("RMSE increases with horizon", rmse_20 > rmse_1 * 2, f"{rmse_1:.2f}->{rmse_20:.2f}")

    # a genuinely skillful multi-step model beats the naive at h=7
    true = p.values

    def oracle(history, h):
        origin = len(history)
        end = min(origin + h, len(true))
        mean = true[origin:end].astype(float)
        if len(mean) < h:                       # pad tail
            mean = np.concatenate([mean, np.repeat(mean[-1] if len(mean) else history.values[-1], h - len(mean))])
        return {"mean": mean, "lower": mean - 5, "upper": mean + 5}

    osc = multistep.evaluate_model_multih(p, oracle, horizons=(7,), test_start=500, step=1)
    check("oracle beats naive at h=7 (U<<1, DM sig)",
          osc[7]["theil_u"] < 0.2 and osc[7]["dm_p"] < 0.05,
          f"U={osc[7]['theil_u']:.3f} p={osc[7]['dm_p']:.3g}")


# ── Point 4: targets ─────────────────────────────────────────────────────────

def test_targets():
    print("\n[Point 4] volatility & direction targets")
    # volatility: GARCH-like clustered series so EWMA/GARCH should beat persistence
    r = rng()
    n = 900
    vol = np.empty(n); vol[0] = 1.0
    rets = np.empty(n)
    for t in range(1, n):
        vol[t] = np.sqrt(0.02 + 0.9 * vol[t-1]**2 * 0.0 + 0.1 * rets[t-1]**2 + 0.85 * vol[t-1]**2)
        rets[t] = r.normal(0, max(vol[t], 1e-6))
    prices = pd.Series(100 + np.cumsum(rets), index=pd.date_range("2020-01-01", periods=n, freq="D"))
    vol_res = targets.evaluate_volatility(prices, test_ratio=0.3, refit_every=25)
    check("volatility eval returns qlike for all methods",
          all("qlike" in vol_res[m] for m in ("persistence", "ewma", "garch")))
    check("ewma beats persistence on clustered vol",
          vol_res["ewma"]["qlike"] <= vol_res["persistence"]["qlike"] + 1e-9,
          f"ewma={vol_res['ewma']['qlike']} pers={vol_res['persistence']['qlike']}")

    # direction: an informative probability should beat the coin
    p2 = synth_prices(n=600)
    y_up = targets.direction_labels(p2)
    good_prob = np.clip(0.5 + 0.4 * (2 * y_up - 1) + r.normal(0, 0.1, len(y_up)), 0, 1)
    dres = targets.evaluate_direction(y_up, good_prob)
    check("direction: informative prob beats coin", dres["verdict"] == "beats coin", str(dres))
    coin_prob = np.full(len(y_up), 0.5)
    dres2 = targets.evaluate_direction(y_up, coin_prob)
    check("direction: coin does not beat coin", dres2["verdict"] == "no better than coin")

    # returns+features runs and reports a Theil U (unpredictable RW ⇒ U ~ 1+)
    fres = targets.evaluate_returns_with_features(p2.values, test_ratio=0.3)
    check("returns+features reports Theil U", np.isfinite(fres["theil_u_returns"]), str(fres))


def main():
    print("=" * 70)
    print("  honest_eval — offline test suite")
    print("=" * 70)
    for t in (test_naive, test_metrics, test_prob_metrics, test_validation,
              test_multistep, test_targets):
        t()
    print("\n" + "=" * 70)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILURES: " + ", ".join(FAIL))
    print("=" * 70)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
