"""
lstm_mdn_prototype.py — option 3 (alternatives_distributions_pi.pdf §1.3): a
mixture-density-network head for LSTM, as an A/B test against the production
LSTM's `preds ± 1.96*std(train residuals)` band -- NOT wired into
models/lstm_model.py. Standalone prototype only, to measure whether the extra
training cost buys a real calibration improvement before deciding it's worth
productionising.

Architecture mirrors models/lstm_model.py's build_lstm exactly (same SEQ_LEN/
UNITS/DROPOUT_RATE/EPOCHS/BATCH_SIZE, same Dense-head-on-LSTM shape) so the
comparison isolates the head+loss change, not a capacity change -- only the
output layer (Dense(1) -> Dense(3*K): K mixture weights, K means, K stds) and
loss (MSE -> Gaussian-mixture negative log-likelihood) differ.

Kept a fully separate process/script on purpose: importing models/arima_model.py
(statsmodels/arch) in the same process as tensorflow, in either import order
after tensorflow, or importing tensorflow after statsmodels/yfinance, has a
confirmed deadlock on this machine (documented in
experiments/weekly_multimodel.py's TF-import-order guard) -- simplest fix is to
never import the statsmodels-based models here at all (own small fetch_data
instead of models.arima_model.fetch_data).

Distribution shape is native to the model here (no post-hoc fit like the manual
Student-t/GED path in dist_options_common.py) -- the whole point of option 3 is
that the network itself learns an arbitrary (possibly asymmetric, multimodal)
predictive shape at training time, at the cost of a heavier architecture/loss.

Usage (from repo root; set CURL_CA_BUNDLE first if yfinance SSL fails locally):
    python experiments/lstm_mdn_prototype.py
    python experiments/lstm_mdn_prototype.py --assets SPY BTC --n-components 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import dist_options_common as doc  # noqa: E402 (pure numpy, safe post-TF import)

ASSETS = {"SPY": "SPY", "BTC": "BTC-USD", "ETH": "ETH-USD", "ZN": "ZN=F", "TLT": "TLT"}
START, END = "2020-01-01", "2024-12-31"
TEST_RATIO = 0.15
CALIB_FRAC = 0.4   # kept identical to all_models_dist_options.py's split for a
                   # like-for-like eval window even though MDN needs no calibration fit
SEQ_LEN, UNITS, EPOCHS, BATCH_SIZE, DROPOUT_RATE = 30, 64, 60, 32, 0.2
# NLL-mixture training is visibly less stable than plain MSE regression on this
# data (confirmed empirically: unclipped Adam + patience=5, i.e. models/
# lstm_model.py's exact baseline settings, gave RMSE ~2.8x worse than the
# baseline LSTM on SPY -- gradient clipping + more patience brought it back to
# the same ballpark, ~1.2x worse). Kept as extra, explicit training cost on top
# of the architecture itself -- part of what "option 3" actually costs.
EARLYSTOP_PATIENCE = 10
ADAM_CLIPNORM = 1.0
N_COMPONENTS_DEFAULT = 3
N_SAMPLES = 300
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Minimal standalone fetch_data (see module docstring: no arima_model import) ─
def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker} between {start} and {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = pd.to_numeric(raw["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close.astype(float)


def make_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_mdn(seq_len, units, dropout_rate, n_components) -> Sequential:
    model = Sequential()
    model.add(LSTM(units, input_shape=(seq_len, 1), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3 * n_components))
    return model


def mdn_nll_loss(n_components):
    """Negative log-likelihood of a K-component Gaussian mixture. Output layer
    columns: [0:K)=mixture logits, [K:2K)=means, [2K:3K)=raw stds (softplus'd)."""
    def loss(y_true, y_pred):
        K = n_components
        logits = y_pred[:, :K]
        mus = y_pred[:, K:2 * K]
        sigmas = tf.nn.softplus(y_pred[:, 2 * K:3 * K]) + 1e-4
        log_pis = tf.nn.log_softmax(logits, axis=-1)
        y = y_true[:, :1]
        log_norm = (-0.5 * np.log(2 * np.pi) - tf.math.log(sigmas)
                   - 0.5 * tf.square((y - mus) / sigmas))
        return -tf.reduce_mean(tf.reduce_logsumexp(log_pis + log_norm, axis=-1))
    return loss


def mixture_params(model, x, n_components):
    """Raw (pi, mu, sigma) arrays, shape (n_points, K), from a batch of inputs
    (still in SCALED space -- caller inverse-transforms samples, not params,
    since MinMaxScaler is affine and sampling then transforming is exact)."""
    raw = model.predict(x, verbose=0)
    K = n_components
    pis = tf.nn.softmax(raw[:, :K], axis=-1).numpy()
    mus = raw[:, K:2 * K]
    sigmas = tf.nn.softplus(raw[:, 2 * K:3 * K]).numpy() + 1e-4
    return pis, mus, sigmas


def sample_mixture(pis, mus, sigmas, n_samples, rng):
    """(len(pis), n_samples) draws: one component index per draw per point,
    then a Normal(mu_k, sigma_k) draw -- exact Monte Carlo, no ppf needed since a
    Gaussian mixture has no closed-form quantile function."""
    n_points, K = pis.shape
    comp_idx = np.array([rng.choice(K, size=n_samples, p=pis[i]) for i in range(n_points)])
    out = np.empty((n_points, n_samples))
    for i in range(n_points):
        out[i] = rng.normal(mus[i][comp_idx[i]], sigmas[i][comp_idx[i]])
    return out


def run_lstm_mdn(train: pd.Series, test: pd.Series, n_components: int,
                 seed: int = SEED) -> dict:
    tf.random.set_seed(seed)
    t0 = time.time()
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    X_train, y_train = make_sequences(train_scaled.flatten(), SEQ_LEN)
    X_train = X_train.reshape(-1, SEQ_LEN, 1)
    y_train = y_train.reshape(-1, 1)

    model = build_lstm_mdn(SEQ_LEN, UNITS, DROPOUT_RATE, n_components)
    opt = tf.keras.optimizers.Adam(clipnorm=ADAM_CLIPNORM)
    model.compile(optimizer=opt, loss=mdn_nll_loss(n_components))
    es = EarlyStopping(patience=EARLYSTOP_PATIENCE, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
             validation_split=0.1, callbacks=[es], verbose=0)

    rng = np.random.default_rng(seed)
    buffer = list(train_scaled.flatten()[-SEQ_LEN:])
    preds, samples_all = [], []
    for i in range(len(test)):
        x = np.array(buffer[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        pis, mus, sigmas = mixture_params(model, x, n_components)
        preds.append(float(np.sum(pis[0] * mus[0])))   # mixture mean, scaled space
        samples_all.append(sample_mixture(pis, mus, sigmas, N_SAMPLES, rng)[0])
        buffer.append(test_scaled[i, 0])

    train_time = time.time() - t0

    preds_scaled = np.array(preds)
    preds_price = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    samples_price = scaler.inverse_transform(
        np.array(samples_all).reshape(-1, 1)).reshape(len(test), N_SAMPLES)

    return {
        "predictions": preds_price, "samples": samples_price,
        "actual": test.values, "train_time_s": round(train_time, 2),
    }


def evaluate_mdn(result: dict, n_calib: int) -> dict:
    samples_eval = result["samples"][n_calib:]
    actual_eval = np.asarray(result["actual"][n_calib:], float)

    bounds = {}
    for level in doc.LEVELS:
        alpha = 1.0 - level
        lo = np.quantile(samples_eval, alpha / 2.0, axis=1)
        hi = np.quantile(samples_eval, 1.0 - alpha / 2.0, axis=1)
        bounds[level] = (lo, hi)

    out = {}
    for level in doc.LEVELS:
        lo, hi = bounds[level]
        cov, width = doc.coverage_width(actual_eval, lo, hi)
        pct = int(round(level * 100))
        out[f"cov_{pct}"] = round(cov, 2)
        out[f"width_{pct}"] = round(width, 6)
    out["pinball"] = round(doc.avg_pinball(actual_eval, bounds), 6)
    scores = [doc.crps_empirical(samples_eval[i], actual_eval[i]) for i in range(len(actual_eval))]
    out["crps"] = round(float(np.mean(scores)), 6)
    rmse = float(np.sqrt(np.mean((actual_eval - result["predictions"][n_calib:]) ** 2)))
    out["rmse_eval"] = round(rmse, 4)
    return out


def aggregate_seeds(per_seed_kpis: list) -> dict:
    """mean +/- std across seeds for every KPI -- run-to-run spread is itself a
    cost/risk signal for the "is this worth it" question (NLL-mixture training
    is not exactly reproducible even with a fixed seed and single-threaded TF,
    confirmed empirically: same code/seed/data gave cov_50 in [14, 66] on SPY
    across two otherwise-identical runs -- see script docstring / final report)."""
    keys = per_seed_kpis[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([kpi[k] for kpi in per_seed_kpis], dtype=float)
        agg[k] = {"mean": round(float(vals.mean()), 6), "std": round(float(vals.std()), 6),
                  "min": round(float(vals.min()), 6), "max": round(float(vals.max()), 6)}
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--n-components", type=int, default=N_COMPONENTS_DEFAULT)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="multiple seeds per asset -- NLL-mixture training is not "
                        "exactly reproducible even with a fixed seed (see docstring); "
                        "reporting mean+spread rather than a single run is the honest "
                        "way to score this option")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "lstm_mdn_results.json"))
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip assets already present in --out (default on); "
                        "pass --no-resume to force a clean rerun")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    existing = {}
    if args.resume and Path(args.out).exists():
        try:
            existing = json.loads(Path(args.out).read_text())
        except Exception as exc:
            log(f"  --resume: couldn't read existing {args.out} ({exc}), starting fresh")

    payload = {"config": {"assets": args.assets, "n_components": args.n_components,
                          "seeds": args.seeds, "start": START, "end": END,
                          "test_ratio": TEST_RATIO, "calib_frac": CALIB_FRAC},
              "assets": existing.get("assets", {})}
    done = set(payload["assets"])
    if done:
        log(f"--resume: {sorted(done)} already in {args.out}, skipping")

    t_start = time.time()
    for asset_short in args.assets:
        if asset_short in done:
            continue
        ticker = ASSETS[asset_short]
        prices = fetch_data(ticker, START, END)
        split = int(len(prices) * (1 - TEST_RATIO))
        train, test = prices.iloc[:split], prices.iloc[split:]
        n_calib = doc.calibration_eval_split(len(test), CALIB_FRAC)
        log(f"  train={len(train)} test={len(test)} (calib={n_calib}, eval={len(test)-n_calib})")

        per_seed_kpis, per_seed_times = [], []
        for seed in args.seeds:
            log(f"=== {asset_short} ({ticker}) — LSTM-MDN(K={args.n_components}, seed={seed}) ===")
            result = run_lstm_mdn(train, test, args.n_components, seed=seed)
            kpi = evaluate_mdn(result, n_calib)
            per_seed_kpis.append(kpi)
            per_seed_times.append(result["train_time_s"])
            log(f"  seed={seed} done ({result['train_time_s']:.1f}s train)  {kpi}")

        agg = aggregate_seeds(per_seed_kpis)
        payload["assets"][asset_short] = {
            "n_test": len(test), "n_calib": n_calib,
            "train_time_s_per_seed": per_seed_times,
            "per_seed_kpi": per_seed_kpis,
            "kpi_agg": agg,
        }
        log(f"  {asset_short} aggregate over {len(args.seeds)} seeds: "
           f"cov_50={agg['cov_50']['mean']}+/-{agg['cov_50']['std']}  "
           f"crps={agg['crps']['mean']}+/-{agg['crps']['std']}")
        Path(args.out).write_text(json.dumps(payload, indent=2))

    elapsed = time.time() - t_start
    payload["config"]["elapsed_s"] = round(elapsed, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    log(f"DONE in {elapsed/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
