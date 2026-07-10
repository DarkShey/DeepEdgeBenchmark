"""
TSDiff Diffusion Forecasting Model
==================================
Standalone port of the **TSDiff** diffusion model from the DEITA time-series
benchmark (Kollovieh et al. 2023, "Predict, Refine, Synthesize").

What it does
------------
A conditional **1-D U-Net denoising diffusion** model of one-step log-returns.
The denoiser (UNet1D with FiLM conditioning + sinusoidal timestep embedding +
a bottleneck self-attention + a learned trend/seasonal decomposition) is trained
to predict the Gaussian noise added to a short horizon of standardized returns.
Forecasting is **walk-forward** (rolling 1-step-ahead): at each step the model is
conditioned on a learned embedding of the last `SEQ_LEN` returns, draws
`N_SAMPLES` denoised return paths via **DDIM** sampling, and the next-step return
is read off step 0 of the horizon. The realised value is fed back into the
look-back buffer, and so on. Prediction intervals come directly from the
diffusion model's own **sample distribution** (2.5 / 97.5 empirical quantiles) —
a genuine predictive distribution rather than a Gaussian residual band.

Adaptation vs the DEITA original
--------------------------------
The DEITA TSDiff is a *16-asset joint* model conditioned on an R^41 vector
(regime one-hot + macro context + history embedding). Those regime/macro signals
do not exist in DeepEdgeBenchmark, so this port is:
  * **univariate** (n_assets = 1), one price series like every other model here;
  * conditioned on a **learned embedding of the look-back window only** (the
    regime/macro parts of the R^41 vector are dropped);
otherwise the diffusion core (linear beta schedule, eps-prediction MSE loss,
DDIM eta=1 sampling, EMA of weights, learned decomposition) is preserved.

This file is fully self-contained — no dependency on any other DEITA module.

Quick start
-----------
    pip install numpy pandas yfinance torch scikit-learn statsmodels matplotlib

    python tsdiff_model.py                              # BTC-USD backtest
    python tsdiff_model.py --ticker SPY --plot out.png  # + save forecast plot
    python tsdiff_model.py --ticker GC=F --next-step    # single next-step forecast

Note: training a diffusion net + rolling DDIM sampling is CPU intensive — the
backtest takes a while. Defaults are sized for tractable CPU runs; quality scales
with --hidden / --depth / --epochs / --n-samples.
"""

import argparse
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config (defaults; override via CLI) ──────────────────────────────────────
SEQ_LEN      = 30      # look-back window fed to the history embedding
HORIZON      = 7       # diffusion generation length; step 0 is the 1-step forecast
HIDDEN       = 64      # U-Net hidden channels
DEPTH        = 2       # down / up residual blocks
COND_DIM     = 32      # history-embedding (conditioning) dimension
T_DIFFUSION  = 1000    # training diffusion steps
K_DENOISE    = 20      # DDIM inference steps
N_SAMPLES    = 50      # samples per forecast (drives point estimate + PI)
EPOCHS       = 40
BATCH_SIZE   = 32
LR           = 2e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY    = 0.999
DDIM_ETA     = 1.0     # 1.0 = DDPM-like stochastic sampling (needed for real PI)
DEFAULT_SEED = 42      # torch training isn't bit-exact across machines, but a fixed
                       # seed makes a given run reproducible on the same machine.


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python / numpy / torch (incl. CUDA) for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily Close prices and return a clean, tz-naive Series."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker} between {start} and {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = pd.to_numeric(raw["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close.astype(float)


# ── Metrics (identical contract to the other DeepEdgeBenchmark models) ────────
def compute_metrics(actual, predicted, pi_lower=None, pi_upper=None,
                    train_time=0.0) -> dict:
    actual    = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    mae   = mean_absolute_error(actual, predicted)
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mape  = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    smape = np.mean(2 * np.abs(actual - predicted) /
                    (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
    dir_acc = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
    try:
        lb_p = acorr_ljungbox(actual - predicted, lags=[10],
                              return_df=True)["lb_pvalue"].values[0]
    except Exception:
        lb_p = np.nan
    pi_cov = np.nan
    if pi_lower is not None and pi_upper is not None:
        pi_cov = np.mean((actual >= pi_lower) & (actual <= pi_upper)) * 100
    return {
        "RMSE":           round(rmse,  4),
        "MAE":            round(mae,   4),
        "MAPE (%)":       round(mape,  2),
        "SMAPE (%)":      round(smape, 2),
        "Dir. Acc (%)":   round(dir_acc, 2),
        "PI Cov 95% (%)": round(pi_cov, 2) if not np.isnan(pi_cov) else "N/A",
        "Ljung-Box p":    round(lb_p,  4) if not np.isnan(lb_p) else "N/A",
        "Train Time (s)": round(train_time, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Diffusion architecture (inlined from DEITA diffusion_pool/models/tsdiff.py)
# ══════════════════════════════════════════════════════════════════════════════

def linear_beta_schedule(T: int, beta_min: float = 1e-4, beta_max: float = 0.02):
    """VP-SDE linear beta schedule. Returns dict of numpy arrays [T]."""
    betas      = np.linspace(beta_min, beta_max, T, dtype=np.float64)
    alphas     = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    return {
        "betas":      betas,
        "alphas_bar": alphas_bar,
        "sqrt_ab":    np.sqrt(alphas_bar),
        "sqrt_1mab":  np.sqrt(1.0 - alphas_bar),
    }


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift from condition c."""
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x, c):
        gamma, beta = self.proj(c).chunk(2, dim=-1)     # each [B, hidden_dim]
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ResBlock1D(nn.Module):
    """Residual block with FiLM conditioning. Preserves the sequence length."""
    def __init__(self, dim: int, cond_dim: int, t_emb_dim: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1  = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.conv2  = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.norm1  = nn.GroupNorm(min(8, dim), dim)
        self.norm2  = nn.GroupNorm(min(8, dim), dim)
        self.film   = FiLM(cond_dim, dim)
        self.t_proj = nn.Linear(t_emb_dim, dim)

    def forward(self, x, c, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h.permute(0, 2, 1)                          # [B, L, dim]
        h = self.film(h, c)
        h = h + self.t_proj(t_emb).unsqueeze(1)         # broadcast over L
        h = h.permute(0, 2, 1)                          # [B, dim, L]
        h = F.silu(self.norm2(self.conv2(h)))
        return x + h


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ChannelAttention1D(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = x.permute(0, 2, 1)          # [B, L, dim]
        h, _ = self.attn(h, h, h)
        return (self.norm(h) + h).permute(0, 2, 1)


class LearnedDecomposition(nn.Module):
    """Learnable element-wise trend + seasonal masks; score operates on residual."""
    def __init__(self, seq_len: int, n_assets: int):
        super().__init__()
        self.trend_w  = nn.Parameter(torch.randn(seq_len, n_assets) * 0.01)
        self.season_w = nn.Parameter(torch.randn(seq_len, n_assets) * 0.01)

    def decompose(self, x):
        trend    = x * self.trend_w.unsqueeze(0)
        seasonal = x * self.season_w.unsqueeze(0)
        residual = x - trend - seasonal
        return residual, trend, seasonal


class UNet1D(nn.Module):
    """1-D U-Net (spatial size preserved throughout, so any horizon works)."""
    def __init__(self, in_channels=1, hidden_dim=64, depth=2, cond_dim=32, t_emb_dim=64):
        super().__init__()
        self.in_proj  = nn.Conv1d(in_channels, hidden_dim, 1)
        self.out_proj = nn.Conv1d(hidden_dim, in_channels, 1)

        self.t_emb  = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2), nn.SiLU(),
            nn.Linear(t_emb_dim * 2, t_emb_dim),
        )

        self.down_blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, cond_dim, t_emb_dim) for _ in range(depth)])
        self.mid_block1 = ResBlock1D(hidden_dim, cond_dim, t_emb_dim)
        self.mid_attn   = ChannelAttention1D(hidden_dim)
        self.mid_block2 = ResBlock1D(hidden_dim, cond_dim, t_emb_dim)
        self.up_projs  = nn.ModuleList(
            [nn.Conv1d(hidden_dim * 2, hidden_dim, 1) for _ in range(depth)])
        self.up_blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, cond_dim, t_emb_dim) for _ in range(depth)])

    def forward(self, x, t, c):
        # x: [B, H, A] → [B, A, H] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.in_proj(x)
        t_emb = self.t_proj(self.t_emb(t))

        skips = []
        for block in self.down_blocks:
            x = block(x, c, t_emb)
            skips.append(x)

        x = self.mid_block1(x, c, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c, t_emb)

        for i, (proj, block) in enumerate(zip(self.up_projs, self.up_blocks)):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = proj(x)
            x = block(x, c, t_emb)

        x = self.out_proj(x)
        return x.permute(0, 2, 1)       # [B, H, A]


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
            else:
                self.shadow[k] = v.clone()

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


class TSDiff:
    """Self-contained univariate TSDiff: history-conditioned return diffusion."""

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON, hidden=HIDDEN, depth=DEPTH,
                 cond_dim=COND_DIM, T=T_DIFFUSION, lr=LR, wd=WEIGHT_DECAY,
                 ema_decay=EMA_DECAY, device="cpu"):
        self.seq_len = seq_len
        self.horizon = horizon
        self.T       = T
        self.device  = torch.device(device)

        sched = linear_beta_schedule(T)
        self.alphas_bar = sched["alphas_bar"]
        self._sqrt_ab   = torch.tensor(sched["sqrt_ab"],   device=self.device, dtype=torch.float32)
        self._sqrt_1mab = torch.tensor(sched["sqrt_1mab"], device=self.device, dtype=torch.float32)

        # History embedding replaces DEITA's regime/macro conditioning.
        self.hist_embed = nn.Linear(seq_len, cond_dim).to(self.device)
        self.decomp     = LearnedDecomposition(horizon, n_assets=1).to(self.device)
        self.net        = UNet1D(1, hidden, depth, cond_dim).to(self.device)
        self.ema        = EMA(self.net, ema_decay)

        self.opt = torch.optim.AdamW(
            list(self.net.parameters()) + list(self.decomp.parameters())
            + list(self.hist_embed.parameters()),
            lr=lr, weight_decay=wd,
        )

    def train(self, hist_batch, target_batch, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=False):
        """hist_batch: [N, seq_len], target_batch: [N, horizon] (standardized returns)."""
        hist = torch.tensor(hist_batch,   dtype=torch.float32, device=self.device)
        tgt  = torch.tensor(target_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)  # [N,H,1]
        n = hist.shape[0]
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                h_b, x0 = hist[idx], tgt[idx]
                c = self.hist_embed(h_b)                         # [B, cond_dim]
                residual, _, _ = self.decomp.decompose(x0)
                t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
                noise = torch.randn_like(residual)
                s_ab = self._sqrt_ab[t].unsqueeze(-1).unsqueeze(-1)
                s_1m = self._sqrt_1mab[t].unsqueeze(-1).unsqueeze(-1)
                x_t  = s_ab * residual + s_1m * noise
                pred_noise = self.net(x_t, t, c)
                loss = F.mse_loss(pred_noise, noise)
                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + list(self.decomp.parameters())
                    + list(self.hist_embed.parameters()), 1.0)
                self.opt.step()
                self.ema.update(self.net)
        # Use the EMA-averaged weights for inference.
        self.ema.apply(self.net)

    @torch.no_grad()
    def sample_paths(self, hist_window, n_samples=N_SAMPLES, k_denoise=K_DENOISE,
                     ddim_eta=DDIM_ETA):
        """DDIM-sample n_samples full return paths conditioned on hist_window
        [seq_len]; return standardized step-returns as np [n_samples, horizon]."""
        self.net.eval()
        h = torch.tensor(hist_window, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,seq_len]
        c = self.hist_embed(h).expand(n_samples, -1)             # [n_samples, cond_dim]

        x = torch.randn(n_samples, self.horizon, 1, device=self.device)
        steps = torch.linspace(self.T - 1, 0, k_denoise + 1).long()
        for i in range(k_denoise):
            t_now, t_prev = steps[i].item(), steps[i + 1].item()
            t_ten = torch.full((n_samples,), t_now, device=self.device, dtype=torch.long)
            pred_noise = self.net(x, t_ten, c)
            ab_prev = float(self.alphas_bar[t_prev]) if t_prev >= 0 else 1.0
            s_ab    = float(self._sqrt_ab[t_now])
            s_1mab  = float(self._sqrt_1mab[t_now])
            pred_x0 = ((x - s_1mab * pred_noise) / max(s_ab, 1e-6)).clamp(-10.0, 10.0)
            ab_t    = s_ab * s_ab
            sigma   = ddim_eta * np.sqrt(max((1 - ab_prev) / max(1 - ab_t, 1e-6), 0.0)) \
                              * np.sqrt(max(1 - ab_t / max(ab_prev, 1e-6), 0.0))
            dir_coef = np.sqrt(max(1 - ab_prev - sigma * sigma, 0.0))
            noise    = torch.randn_like(x) if sigma > 0 else 0.0
            x = np.sqrt(ab_prev) * pred_x0 + dir_coef * pred_noise + sigma * noise
        x = x.clamp(-10.0, 10.0)
        return x[:, :, 0].detach().cpu().numpy()                 # [n_samples, horizon]

    def sample_next(self, hist_window, n_samples=N_SAMPLES, k_denoise=K_DENOISE,
                    ddim_eta=DDIM_ETA):
        """Step-0 (next-step) standardized-return samples as np [n_samples]."""
        return self.sample_paths(hist_window, n_samples, k_denoise, ddim_eta)[:, 0]


# ══════════════════════════════════════════════════════════════════════════════
#  Walk-forward backtest (DeepEdgeBenchmark run_<model> contract)
# ══════════════════════════════════════════════════════════════════════════════

def _log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


def _make_windows(z: np.ndarray, seq_len: int, horizon: int):
    """Sliding (history[seq_len] -> target[horizon]) pairs over standardized returns."""
    H, T = [], []
    for i in range(seq_len, len(z) - horizon + 1):
        H.append(z[i - seq_len:i])
        T.append(z[i:i + horizon])
    return np.asarray(H, dtype=np.float32), np.asarray(T, dtype=np.float32)


def run_tsdiff(train: pd.Series, test: pd.Series,
               seq_len=SEQ_LEN, horizon=HORIZON, hidden=HIDDEN, depth=DEPTH,
               cond_dim=COND_DIM, T=T_DIFFUSION, epochs=EPOCHS, batch_size=BATCH_SIZE,
               k_denoise=K_DENOISE, n_samples=N_SAMPLES, ddim_eta=DDIM_ETA) -> dict:
    """Train on the train window's returns, roll 1-step-ahead over the test window.

    Point forecast = mean of the DDIM sample cloud; 95% PI = 2.5/97.5 sample
    quantiles (the diffusion model's own predictive distribution).
    """
    if len(train) <= seq_len + horizon:
        raise ValueError(
            f"train series has {len(train)} points, but seq_len={seq_len} + "
            f"horizon={horizon} requires more than {seq_len + horizon} points.")
    t0 = time.time()

    train_p = train.values.astype(float)
    r = _log_returns(train_p)                                    # [len(train)-1]
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd                                           # standardized returns

    H_win, T_win = _make_windows(z, seq_len, horizon)
    if len(H_win) == 0:
        raise ValueError("not enough return history to build training windows.")

    model = TSDiff(seq_len, horizon, hidden, depth, cond_dim, T)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)

    # Walk-forward over the test window. `buffer` holds standardized returns up to
    # the point being predicted; `last_price` is the realised price at t-1.
    buffer     = list(z)
    last_price = float(train_p[-1])
    test_p     = test.values.astype(float)

    preds, lower, upper = [], [], []
    for i in range(len(test_p)):
        window = np.asarray(buffer[-seq_len:], dtype=np.float32)
        z_samples = model.sample_next(window, n_samples=n_samples,
                                      k_denoise=k_denoise, ddim_eta=ddim_eta)
        r_samples = z_samples * sd + mu                         # de-standardize returns
        price_samples = last_price * np.exp(r_samples)          # returns → price
        preds.append(float(np.mean(price_samples)))
        lower.append(float(np.quantile(price_samples, 0.025)))
        upper.append(float(np.quantile(price_samples, 0.975)))
        # walk forward with the realised value
        realised_r = np.log(test_p[i] / last_price)
        buffer.append((realised_r - mu) / sd)
        last_price = float(test_p[i])

    preds = np.asarray(preds); lower = np.asarray(lower); upper = np.asarray(upper)
    train_time = time.time() - t0
    metrics = compute_metrics(test_p, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    return {**metrics, "predictions": preds, "lower": lower, "upper": upper,
            "index": test.index, "actual": test_p}


def next_step_tsdiff(series: pd.Series, seq_len=SEQ_LEN, horizon=HORIZON,
                     hidden=HIDDEN, depth=DEPTH, cond_dim=COND_DIM, T=T_DIFFUSION,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, k_denoise=K_DENOISE,
                     n_samples=N_SAMPLES, ddim_eta=DDIM_ETA):
    """Single 1-step forecast beyond the last observation. Returns (pred, lo, hi)."""
    if len(series) <= seq_len + horizon:
        raise ValueError(
            f"series has {len(series)} points, but seq_len={seq_len} + "
            f"horizon={horizon} requires more than {seq_len + horizon} points.")
    prices = series.values.astype(float)
    r = _log_returns(prices)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = _make_windows(z, seq_len, horizon)
    model = TSDiff(seq_len, horizon, hidden, depth, cond_dim, T)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)

    window = z[-seq_len:].astype(np.float32)
    z_samples = model.sample_next(window, n_samples=n_samples,
                                  k_denoise=k_denoise, ddim_eta=ddim_eta)
    price_samples = float(prices[-1]) * np.exp(z_samples * sd + mu)
    return (float(np.mean(price_samples)),
            float(np.quantile(price_samples, 0.025)),
            float(np.quantile(price_samples, 0.975)))


# ── Plot (optional) ──────────────────────────────────────────────────────────
def save_plot(result: dict, ticker: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = result["index"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, result["actual"], label="Actual", color="black", lw=1.3)
    ax.plot(idx, result["predictions"], label="TSDiff forecast", color="tab:green", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:green",
                    alpha=0.20, label="95% PI (sample quantiles)")
    ax.set_title(f"TSDiff diffusion — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="TSDiff diffusion forecasting (DEITA port)")
    p.add_argument("--ticker", default="BTC-USD", help="yfinance ticker (BTC-USD, SPY, GC=F)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for reproducible training (numpy/torch)")
    p.add_argument("--next-step", action="store_true", help="only forecast the next step")
    p.add_argument("--plot", metavar="PATH", default=None, help="save a forecast plot")
    args = p.parse_args()

    set_seed(args.seed)

    print(f"Downloading {args.ticker} [{args.start} -> {args.end}] ...")
    prices = fetch_data(args.ticker, args.start, args.end)
    print(f"  {len(prices)} daily observations.\n")

    if args.next_step:
        pred, lo, hi = next_step_tsdiff(prices, hidden=args.hidden, depth=args.depth,
                                        epochs=args.epochs, n_samples=args.n_samples)
        print(f"Last close      : {prices.iloc[-1]:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  "
          f"TSDiff(hidden={args.hidden}, depth={args.depth}) epochs={args.epochs}\n")
    print("Note: training a diffusion net + rolling DDIM sampling can take a while.\n")

    result = run_tsdiff(train, test, hidden=args.hidden, depth=args.depth,
                        epochs=args.epochs, n_samples=args.n_samples)
    print(f"=== TSDiff diffusion — {args.ticker} ===")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
