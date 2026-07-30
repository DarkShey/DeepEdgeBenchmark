"""
lstm_sigma_variants.py — time-varying sigma for the LSTM WITHOUT an MDN.

Background (HANDOFF_dist_options_comparison.md): the production LSTM's 95% PI
uses ONE constant sigma (std of train residuals) for the whole test window --
the width never reacts to volatility regimes. The MDN prototype (option 3)
tried to fix this inside the network and failed: WORSE calibration (MACE 9.23
vs 6.77 prod), +23% cost, unstable across seeds. This script tests the simpler
family of fixes first: keep the production network + point forecast EXACTLY as
they are, and only replace the constant sigma with a sigma_t path estimated
from OBSERVED walk-forward residuals (no look-ahead: sigma_t only uses
residuals up to t-1, warm-started on train residuals).

Variants (sigma path):
  frozen  -- constant std(train residuals): the production baseline, reproduced
             here so every other row is measured against the same run.
  ewma94  -- RiskMetrics EWMA: sigma2_t = 0.94*sigma2_{t-1} + 0.06*eps2_{t-1}.
  roll20  -- rolling std of the last 20 observed residuals.
  garch   -- GARCH(1,1) (zero-mean) refit on all residuals-so-far every 20
             steps, exact variance-filter recursion between refits (same refit
             cadence as models/arima_model.py's GARCH_REFIT_FREQ).

Each sigma path is then pushed through the intern's option-1/2 machinery
(normal / student_t / GED / CQR on the calib/eval split), so results are
directly comparable with all_models_dist_options_results.json (2020-2024,
5 assets, seed 42) and with the MDN row (lstm_mdn_results.json).

NOTE on the baseline: all_models_dist_options.py did NOT seed TF before
lm.run_lstm, so its LSTM row is not bit-reproducible; this script seeds 42
before each asset's training. Small differences vs the intern's "frozen"
numbers are expected and harmless -- all variants here share the SAME run.

Resumable under a hard 45 s cap: per asset, unit 1 trains and saves weights,
unit 2 loads weights and materializes predictions/residuals, and the final
evaluation is pure numpy. Re-invoke until DONE.

Usage:
    python experiments/lstm_sigma_variants.py --budget-s 30
Output: experiments/lstm_sigma_variants_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import dist_options_common as doc  # noqa: E402
from all_models_dist_options import (  # noqa: E402
    option12_variants, ASSETS, START, END, TEST_RATIO, CALIB_FRAC,
)
from offline_prices import fetch_data_offline  # noqa: E402

SEED = 42
GARCH_REFIT_FREQ = 20
EWMA_LAMBDA = 0.94
ROLL_WINDOW = 20

# Window support (robustness follow-up): W1 keeps the legacy file names so the
# original single-window results stay untouched.
from robustness_windows import WINDOWS  # noqa: E402

_WINDOW = "W1"  # set by main() before any unit runs


def _paths():
    suffix = "" if _WINDOW == "W1" else f"_{_WINDOW}"
    return (ROOT / "experiments" / f"lstm_sigma_state{suffix}.json",
            ROOT / "experiments" / f"lstm_sigma_variants_results{suffix}.json",
            ROOT / "experiments" / f"lstm_sigma_ckpt{suffix}")


STATE_PATH, OUT_PATH, CKPT_DIR = _paths()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _window_dates():
    return WINDOWS.get(_WINDOW, (START, END))


def _prices_split(asset_short):
    start, end = _window_dates()
    prices = fetch_data_offline(ASSETS[asset_short], start, end)
    split = int(len(prices) * (1 - TEST_RATIO))
    return prices, prices.iloc[:split], prices.iloc[split:], split


# ── Unit 1: incremental training (resumable epoch by epoch) ──────────────────
# Mirrors lm.run_lstm's EarlyStopping(patience=5, restore_best_weights=True)
# semantics, but drives the epoch loop manually so training can be paused and
# resumed across hard 45 s execution caps (current + best weights checkpointed
# after every epoch). validation_split=0.1 re-selects the same trailing slice
# on every fit() call, so incremental epochs see the identical split.
def unit_train_incremental(asset_short, a, deadline):
    """Returns True when training is finished (best weights on disk)."""
    import lstm_model as lm
    prices, train, test, split = _prices_split(asset_short)
    t0 = time.time()
    if "scaler_min" not in a:
        lm.set_seed(SEED)
        a["scaler_min"] = float(train.values.min())
        a["scaler_max"] = float(train.values.max())
        a["epoch"] = 0
        a["best_val"] = None
        a["patience_left"] = 5
        a["train_time_s"] = 0.0
    train_scaled = (train.values - a["scaler_min"]) / (a["scaler_max"] - a["scaler_min"])
    X_train, y_train = lm.make_sequences(train_scaled, lm.SEQ_LEN)
    X_train = X_train.reshape(-1, lm.SEQ_LEN, 1)

    CKPT_DIR.mkdir(exist_ok=True)
    cur = CKPT_DIR / f"{asset_short}.cur.weights.h5"
    best = CKPT_DIR / f"{asset_short}.weights.h5"
    model = lm.build_lstm(lm.SEQ_LEN)
    if a["epoch"] > 0 and cur.exists():
        model.load_weights(str(cur))

    epoch_cost = a.get("epoch_cost_s", 4.0)
    while a["epoch"] < lm.EPOCHS and a["patience_left"] > 0:
        if time.time() + epoch_cost * 1.5 > deadline:
            a["train_time_s"] += time.time() - t0
            return False
        te = time.time()
        h = model.fit(X_train, y_train, epochs=1, batch_size=lm.BATCH_SIZE,
                      validation_split=0.1, verbose=0)
        epoch_cost = time.time() - te
        a["epoch_cost_s"] = round(epoch_cost, 2)
        a["epoch"] += 1
        val = float(h.history["val_loss"][0])
        if a["best_val"] is None or val < a["best_val"]:
            a["best_val"] = val
            a["patience_left"] = 5
            model.save_weights(str(best))
        else:
            a["patience_left"] -= 1
        model.save_weights(str(cur))
    a["train_time_s"] = round(a["train_time_s"] + time.time() - t0, 1)
    a["trained"] = True
    return True


# ── Unit 2: load weights -> walk-forward predictions + residuals ─────────────
def unit_predict(asset_short, meta):
    import lstm_model as lm
    prices, train, test, split = _prices_split(asset_short)
    smin, smax = meta["scaler_min"], meta["scaler_max"]
    scale = lambda v: (v - smin) / (smax - smin)          # noqa: E731
    unscale = lambda v: v * (smax - smin) + smin          # noqa: E731

    model = lm.build_lstm(lm.SEQ_LEN)
    model.load_weights(str(CKPT_DIR / f"{asset_short}.weights.h5"))

    train_scaled = scale(train.values)
    X_train, _ = lm.make_sequences(train_scaled, lm.SEQ_LEN)
    train_preds = unscale(model.predict(
        X_train.reshape(-1, lm.SEQ_LEN, 1), verbose=0).flatten())
    train_resid = train.values[lm.SEQ_LEN:] - train_preds

    # Walk-forward inputs only ever contain REALISED values, so the rolling
    # loop in lm.run_lstm is exactly a batched predict on sequences drawn from
    # the concatenated realised series (verified equivalent, dropout off).
    full_scaled = scale(np.concatenate([train.values[-lm.SEQ_LEN:], test.values]))
    X_test, _ = lm.make_sequences(full_scaled, lm.SEQ_LEN)
    preds = unscale(model.predict(
        X_test.reshape(-1, lm.SEQ_LEN, 1), verbose=0).flatten())

    return {"preds": list(map(float, preds)),
            "actual": list(map(float, test.values)),
            "train_resid": list(map(float, train_resid))}


# ── Sigma paths (strictly causal) ────────────────────────────────────────────
def sigma_frozen(train_resid, test_resid):
    return np.full(len(test_resid), np.std(train_resid))


def sigma_ewma(train_resid, test_resid, lam=EWMA_LAMBDA):
    s2 = float(np.var(train_resid))
    for e in train_resid[-250:]:
        s2 = lam * s2 + (1 - lam) * e * e
    out = np.empty(len(test_resid))
    for t, e in enumerate(test_resid):
        out[t] = np.sqrt(s2)          # forecast for step t uses e_{<t} only
        s2 = lam * s2 + (1 - lam) * e * e
    return out


def sigma_roll(train_resid, test_resid, w=ROLL_WINDOW):
    hist = list(train_resid[-w:])
    out = np.empty(len(test_resid))
    for t, e in enumerate(test_resid):
        out[t] = np.std(hist[-w:])
        hist.append(e)
    return out


def sigma_garch(train_resid, test_resid, refit_freq=GARCH_REFIT_FREQ):
    from arch import arch_model
    hist = list(train_resid)
    out = np.empty(len(test_resid))
    omega = alpha = beta = s2 = None
    for t, e in enumerate(test_resid):
        if t % refit_freq == 0:
            r = arch_model(np.asarray(hist), mean="Zero", vol="Garch",
                           p=1, q=1, rescale=True).fit(disp="off")
            sf = r.scale
            omega = float(r.params["omega"]) / sf**2
            alpha = float(r.params["alpha[1]"])
            beta = float(r.params["beta[1]"])
            s2 = float(r.conditional_volatility[-1] / sf) ** 2
            s2 = omega + alpha * (hist[-1] ** 2) + beta * s2   # 1-step-ahead
        out[t] = np.sqrt(s2)
        hist.append(e)
        s2 = omega + alpha * e * e + beta * s2
    return out


SIGMA_PATHS = {"frozen": sigma_frozen, "ewma94": sigma_ewma,
               "roll20": sigma_roll, "garch": sigma_garch}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget-s", type=float, default=30.0)
    p.add_argument("--window", default="W1", choices=list(WINDOWS))
    args = p.parse_args()
    deadline = time.time() + args.budget_s

    global _WINDOW, STATE_PATH, OUT_PATH, CKPT_DIR
    _WINDOW = args.window
    STATE_PATH, OUT_PATH, CKPT_DIR = _paths()

    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}

    def save():
        STATE_PATH.write_text(json.dumps(state, indent=2))

    for asset_short in ASSETS:
        a = state.setdefault(asset_short, {})
        if "preds" in a:
            continue
        if not a.get("trained"):
            log(f"{asset_short}: training (epoch {a.get('epoch', 0)}, seed {SEED}) ...")
            try:
                finished = unit_train_incremental(asset_short, a, deadline)
            finally:
                save()
            if not finished:
                log(f"PROGRESS {asset_short} epoch {a['epoch']} -- re-invoke"); return
            log(f"{asset_short}: trained ({a['epoch']} epochs, "
                f"best_val={a['best_val']:.6f})")
        if time.time() + 12 > deadline:
            save(); log("PROGRESS -- re-invoke"); return
        log(f"{asset_short}: predicting ...")
        a.update(unit_predict(asset_short, a))
        save()

    # ── Evaluation (cheap) ───────────────────────────────────────────────────
    w_start, w_end = _window_dates()
    out = {"config": {"window": _WINDOW,
                      "start": w_start, "end": w_end, "test_ratio": TEST_RATIO,
                      "calib_frac": CALIB_FRAC, "seed": SEED,
                      "ewma_lambda": EWMA_LAMBDA, "roll_window": ROLL_WINDOW,
                      "garch_refit_freq": GARCH_REFIT_FREQ},
           "assets": {}}
    mace_acc = {}
    for asset_short in ASSETS:
        a = state[asset_short]
        preds = np.asarray(a["preds"], float)
        actual = np.asarray(a["actual"], float)
        train_resid = np.asarray(a["train_resid"], float)
        test_resid = actual - preds
        n_calib = doc.calibration_eval_split(len(actual), CALIB_FRAC)
        entry = {"n_test": len(actual), "n_calib": n_calib,
                 "train_time_s": a["train_time_s"], "sigma_paths": {}}
        for name, fn in SIGMA_PATHS.items():
            t0 = time.time()
            sig = fn(train_resid, test_resid)
            path_cost = time.time() - t0
            variants, fitted, overhead = option12_variants(
                preds, sig, actual, n_calib)
            entry["sigma_paths"][name] = {
                "sigma_path_cost_s": round(path_cost, 3),
                "fitted_shapes": fitted, "variants": variants,
            }
            for vname, kpi in variants.items():
                mace = np.mean([abs(kpi[f"cov_{p}"] - p)
                                for p in (50, 80, 95)])
                mace_acc.setdefault((name, vname), []).append(mace)
        out["assets"][asset_short] = entry
    out["summary_mace_mean_5assets"] = {
        f"{sp}/{v}": round(float(np.mean(vals)), 2)
        for (sp, v), vals in sorted(mace_acc.items())}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    log(f"DONE -> {OUT_PATH}")
    log(json.dumps(out["summary_mace_mean_5assets"], indent=2))


if __name__ == "__main__":
    main()
