"""
NsDiff Diffusion Forecasting Model
==================================
Standalone port of **NsDiff** (*Non-stationary Diffusion for Probabilistic
Time Series Forecasting*, Wang et al., ICML 2025, arXiv:2505.04278). Official
code: https://github.com/wwy155/NsDiff (src/models/NsDiff.py,
src/layer/{g_backbone,denoise}.py, src/layer/nsdiff_utils.py,
src/utils/sigma.py) -- the equations below are transcribed from there, not
reinvented.

Why this model, in one line
----------------------------
Every other diffuser in this benchmark (incl. tsdiff_model.TSDiff) adds
constant-variance noise: `Y = f(X) + eps`. NsDiff instead poses a
**location-scale noise model (LSNM)**:

    Y = f_phi(X) + sqrt(g_psi(X)) * eps

where `f_phi` is a conditional-mean network and `g_psi` is a **conditional
variance network** trained to regress the target's own realised rolling
variance. The forward diffusion process is then driven by an
**uncertainty-aware noise schedule (UANS)** whose per-step variance blends
`g_psi(X)` with the (training-time-only) realised variance via the recursive
schedule coefficients `alpha_tilde`/`alpha_hat` (ported verbatim below, see
`_compute_ns_schedule`). This is the generative analogue of GARCH: the
diffusion's own noise level is conditioned on the *history*, so the model's
predictive distribution can widen/narrow with realised volatility instead of
being a fixed-width band. That is NsDiff's entire reason for existing in this
benchmark (cf. BRIEF_integration_nsdiff.md) -- everything else here is
plumbing to make that idea run on daily/weekly return series.

Architecture decisions (DECLARED, not a silent deviation from the paper)
-------------------------------------------------------------------------
The official NsDiff is a *multivariate* forecaster built on a full
Non-stationary Transformer (`src/layer/mu_backbone.py`, d_model=512, 8 heads,
2+1 encoder/decoder layers, learned de-stationarisation) for `f_phi`. That is
enormously over-parameterised for this benchmark's univariate, `seq_len<=30`
windows (a handful of standardized daily/weekly returns) and would not train
usefully from so little data. Per the brief, the parts that are NsDiff's
actual scientific contribution are preserved religiously; the oversized mean
backbone is not:

  * `f_phi` (mean backbone) -- **replaced** by a small 2-layer MLP over the
    look-back window (`_MeanBackbone`). This mirrors exactly how
    `tsdiff_model.py` already drops DEITA's regime/macro conditioning and
    keeps only a look-back embedding -- same simplification philosophy,
    applied to NsDiff's mean path. NsDiff's contribution is the variance
    side, not this network.
  * `g_psi` (conditional variance network) -- **preserved**: the official
    `SigmaEstimation` (`src/layer/g_backbone.py`) computes a trailing
    rolling-variance feature of the look-back window and regresses it,
    through an MLP + softplus, onto the target horizon's own realised
    rolling variance. Ported as `_SigmaBackbone`, only the hidden width is
    reduced (512 -> a few dozen units) for these short univariate windows.
    This is the model's core; it is not touched beyond resizing.
  * Denoiser (`eps_theta`, `sigma_theta`) -- **preserved**: the official
    `ConditionalGuidedModel` (`src/layer/denoise.py`), a small
    diffusion-timestep-gated MLP conditioned on
    `[y_t, f_phi(x), g_psi(x)]`. Ported as `_Denoiser`, hidden width reduced
    (128 -> a few dozen) but the timestep-conditioning (`_ConditionalLinear`)
    and the joint eps/sigma output head are unchanged in structure.
  * UANS forward/reverse process (`_compute_ns_schedule`, `_forward_noise_var`,
    `_sigma_tilde`, `_calc_gammas`, `_q_sample`, `_p_sample_step`) --
    **transcribed verbatim** from `src/models/NsDiff.py` /
    `src/layer/nsdiff_utils.py`, generalised from the paper's multivariate
    `(B, O, N)` tensors down to `N=1`. Nothing here is approximated.
  * Diffusion step count -- the official config already trains with only
    `timesteps=20` (`NsDiffParameters.diffusion_steps`) and *always* samples
    all of them ancestrally (`p_sample_loop` iterates every `t` from `T-1`
    down to `0`; the repo defines no DDIM/step-skipping formula for this
    non-stationary schedule). Inventing a DDIM shortcut for a schedule the
    source doesn't define would be exactly the kind of approximation the
    brief forbids. So `K_DENOISE` here plays the role of *both* "number of
    training diffusion steps" and "number of ancestral sampling steps" --
    already CPU-tractable at the official default of 20, same order as
    `tsdiff_model.K_DENOISE`.
  * No calendar/date conditioning (`x_mark` in the official code) -- dropped,
    same univariate simplification as `tsdiff_model.py`'s own docstring
    ("Adaptation vs the DEITA original").

Everything else (walk-forward contract, `fit_*`/`forecast_from_fitted`
split, metrics, CLI shape, reproducibility/numerical-safety conventions) is
identical in spirit to `tsdiff_model.py` so the two diffusion models are
directly comparable in this benchmark.

This file is fully self-contained -- no dependency on any other module in
this repo (mirrors tsdiff_model.py's own self-containment note); `fetch_data`
and `compute_metrics` below are copied verbatim from tsdiff_model.py (same
contract for every model in this benchmark, nothing to reinvent).

Quick start
-----------
    python nsdiff_model.py                              # BTC-USD backtest
    python nsdiff_model.py --ticker SPY --plot out.png   # + save forecast plot
    python nsdiff_model.py --ticker GC=F --next-step     # single next-step forecast

Note: like tsdiff_model.py, this trains a diffusion net + rolls a reverse
diffusion sample at every test step -- CPU intensive but tractable with the
defaults below (measured budget: see NOTE_nsdiff_vs_tsdiff.md).
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
SEQ_LEN         = 30      # look-back window (standardized returns)
HORIZON         = 7       # diffusion generation length; step 0 is the 1-step forecast
HIDDEN_MEAN     = 32      # f_phi (mean backbone, dégraissé -- see module docstring)
HIDDEN_SIGMA    = 32      # g_psi (variance backbone, preserved from the official)
HIDDEN_DENOISE  = 32      # eps_theta/sigma_theta denoiser (preserved from the official)
SIGMA_KERNEL    = 8       # trailing rolling-variance window feeding g_psi (< seq_len)
K_DENOISE       = 20      # diffusion steps == ancestral sampling steps (see docstring)
N_SAMPLES       = 50      # samples per forecast (drives point estimate + PI)
EPOCHS          = 40
BATCH_SIZE      = 32
LR              = 2e-4
WEIGHT_DECAY    = 1e-4
BETA_START      = 1e-4    # official NsDiffParameters.beta_start
BETA_END        = 1e-2    # official NsDiffParameters.beta_end
DEFAULT_SEED    = 42      # torch training isn't bit-exact across machines, but a fixed
                          # seed makes a given run reproducible on the same machine.

EPS   = 1e-8   # numerical floor for every variance/denominator below
CLAMP = 10.0   # standardized-return-scale clamp for samples/x0 (mirrors tsdiff_model.py)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python / numpy / torch (incl. CUDA) for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data (identical contract to every other DeepEdgeBenchmark model) ─────────
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
#  LSNM backbones (inlined/adapted from wwy155/NsDiff)
# ══════════════════════════════════════════════════════════════════════════════

def _wv_sigma_trailing(x: torch.Tensor, window: int, discard_rep: bool = False) -> torch.Tensor:
    """Port of official src/utils/sigma.py:wv_sigma_trailing. For each time step
    t, the variance of x[t-window+1 : t+1] (trailing window). `x`: [B, T, N].
    `discard_rep=False` left-pads by `window` steps (replicate) so the output
    covers every original t (used for the training-time target `y_sigma`,
    computed over history+horizon); `discard_rep=True` skips the pad (used
    inside `_SigmaBackbone`, exactly as the official `SigmaEstimation.forward`
    does on its own look-back window)."""
    if not discard_rep:
        x = F.pad(x, (0, 0, window, 0), mode="replicate")
    windows = x.unfold(1, window, 1)           # [B, T', N, window]
    return windows.var(dim=3, unbiased=False)  # [B, T', N]


class _MeanBackbone(nn.Module):
    """f_phi: conditional mean of the return horizon. ARCHITECTURE DECISION
    (see module docstring): dégraissé from the official Non-stationary
    Transformer down to a small 2-layer MLP -- NsDiff's contribution lives in
    g_psi/UANS, not here."""

    def __init__(self, seq_len: int, horizon: int, hidden: int = HIDDEN_MEAN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, 1] -> y0_hat: [B, horizon, 1]."""
        return self.net(x[:, :, 0]).unsqueeze(-1)


class _SigmaBackbone(nn.Module):
    """g_psi: conditional variance predictor. PRESERVED from the official
    `SigmaEstimation` (src/layer/g_backbone.py) -- trailing rolling-variance
    features of the look-back window through an MLP, softplus output. This is
    NsDiff's actual contribution; only the hidden width is reduced."""

    def __init__(self, seq_len: int, horizon: int, hidden: int = HIDDEN_SIGMA,
                kernel: int = SIGMA_KERNEL):
        super().__init__()
        if kernel >= seq_len:
            raise ValueError(f"sigma_kernel={kernel} must be < seq_len={seq_len}.")
        self.kernel = kernel
        self.mlp = nn.Sequential(
            nn.Linear(seq_len - kernel, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, 1] -> g_x: [B, horizon, 1] (softplus, unclamped)."""
        t_len = x.shape[1]
        sigma = _wv_sigma_trailing(x, self.kernel, discard_rep=True)  # [B, T-k+1, 1]
        sigma = sigma[:, -(t_len - self.kernel):, :] + EPS            # [B, T-k, 1]
        out = self.mlp(sigma[:, :, 0])                                # [B, horizon]
        return F.softplus(out).unsqueeze(-1)


class _ConditionalLinear(nn.Module):
    """Diffusion-timestep-gated linear layer (official `ConditionalLinear`)."""

    def __init__(self, n_in: int, n_out: int, n_steps: int):
        super().__init__()
        self.n_out = n_out
        self.lin = nn.Linear(n_in, n_out)
        self.embed = nn.Embedding(n_steps, n_out)
        self.embed.weight.data.uniform_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        gamma = self.embed(t).view(t.shape[0], -1, self.n_out)
        return gamma * out


class _Denoiser(nn.Module):
    """eps_theta (noise prediction) + sigma_theta (posterior-variance
    prediction), jointly. PRESERVED from the official `ConditionalGuidedModel`
    (src/layer/denoise.py): conditions on `[y_t, f_phi(x), g_psi(x)]`
    concatenated on the feature axis, gated per diffusion step `t`."""

    def __init__(self, n_steps: int, hidden: int = HIDDEN_DENOISE):
        super().__init__()
        self.lin1 = _ConditionalLinear(3, hidden, n_steps)
        self.lin2 = _ConditionalLinear(hidden, hidden, n_steps)
        self.lin3 = _ConditionalLinear(hidden, hidden, n_steps)
        self.lin4 = nn.Linear(hidden, 1)
        self.sigma_lin = nn.Linear(hidden, 1)

    def forward(self, y_t, y0_hat, gx, t):
        h = torch.cat([y_t, y0_hat, gx], dim=-1)
        h = F.softplus(self.lin1(h, t))
        h = F.softplus(self.lin2(h, t))
        h = F.softplus(self.lin3(h, t))
        eps = self.lin4(h)
        sigma = F.softplus(self.sigma_lin(F.softplus(h)))
        return eps, sigma


# ══════════════════════════════════════════════════════════════════════════════
#  Uncertainty-Aware Noise Schedule (UANS) -- ported verbatim from
#  src/models/NsDiff.py + src/layer/nsdiff_utils.py (wwy155/NsDiff, ICML 2025)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ns_schedule(alphas: np.ndarray):
    """Recursive `alpha_tilde`/`alpha_hat` schedule coefficients (official
    `compute_tilde_alpha`/`compute_hat_alpha`, src/models/NsDiff.py). Closed-form
    recursion, algebraically identical to (and explicitly commented as such in)
    the official double-loop implementation:
        alpha_tilde[t] = alpha[t] * (1 + alpha_tilde[t-1]),  alpha_tilde[-1] := 0
        alpha_hat[t]   = alpha[t]**2 + alpha[t] * alpha_hat[t-1],  alpha_hat[-1] := 0
    """
    T = len(alphas)
    alphas_tilde = np.zeros(T, dtype=np.float64)
    alphas_hat   = np.zeros(T, dtype=np.float64)
    prev_tilde = 0.0
    prev_hat   = 0.0
    for t in range(T):
        a = alphas[t]
        alphas_tilde[t] = a * (1.0 + prev_tilde)
        alphas_hat[t]   = a * a + a * prev_hat
        prev_tilde, prev_hat = alphas_tilde[t], alphas_hat[t]
    return alphas_tilde, alphas_hat


class _UANSSchedule:
    """Forward/reverse-process coefficients for the uncertainty-aware noise
    schedule. `betas_tilde = alpha_tilde - alpha_hat` is the coefficient that
    lets the forward-process variance blend `g_psi(X)` with the endpoint
    variance as `t` grows (brief: `sigma_t^2 = beta_t^2 * g_psi(X) + alpha_t *
    beta_t * sigma_Y0`, schematically) -- this is what gives NsDiff its
    GARCH-like heteroscedastic calibration."""

    def __init__(self, T: int, beta_start: float = BETA_START, beta_end: float = BETA_END,
                device: str = "cpu"):
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_tilde, alphas_hat = _compute_ns_schedule(alphas)
        betas_bar = 1.0 - alphas_cumprod
        betas_tilde = np.clip(alphas_tilde - alphas_hat, 0.0, None)

        def _shift_right(a: np.ndarray) -> np.ndarray:
            return np.concatenate([[1.0], a[:-1]])

        self.T = T
        self.device = torch.device(device)
        _t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        self.alphas                    = _t(alphas)
        self.alphas_cumprod            = _t(alphas_cumprod)
        self.alphas_bar_sqrt           = _t(np.sqrt(alphas_cumprod))
        self.one_minus_alphas_bar_sqrt = _t(np.sqrt(1.0 - alphas_cumprod))
        self.betas_bar                 = _t(betas_bar)
        self.betas_tilde               = _t(betas_tilde)
        self.alphas_cumprod_prev       = _t(_shift_right(alphas_cumprod))
        self.betas_tilde_m_1           = _t(_shift_right(betas_tilde))
        self.betas_bar_m_1             = _t(_shift_right(betas_bar))

    def extract(self, coeffs_1d: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Gather per-batch-element schedule coefficients at diffusion step `t`
        and reshape to broadcast against `like` ([B, horizon, 1])."""
        out = coeffs_1d[t]
        return out.view(t.shape[0], *([1] * (like.dim() - 1)))


def _forward_noise_var(sched: _UANSSchedule, gx: torch.Tensor, y_sigma: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
    """Forward-process injected-noise variance at step t (official
    `cal_forward_noise`): blends g_psi(x) and the realised rolling variance."""
    b_bar   = sched.extract(sched.betas_bar, t, gx)
    b_tilde = sched.extract(sched.betas_tilde, t, gx)
    return (b_bar - b_tilde) * gx + b_tilde * y_sigma


def _sigma_tilde(sched: _UANSSchedule, gx: torch.Tensor, y_sigma: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    """Target posterior variance for the sigma_theta KL term (official
    `cal_sigma_tilde` / `cal_sigma12`)."""
    at         = sched.extract(sched.alphas, t, gx)
    b_tilde_m1 = sched.extract(sched.betas_tilde_m_1, t, gx)
    b_bar_m1   = sched.extract(sched.betas_bar_m_1, t, gx)
    Sigma_1 = (1 - at) ** 2 * gx + at * (1 - at) * y_sigma
    Sigma_2 = (b_bar_m1 - b_tilde_m1) * gx + b_tilde_m1 * y_sigma
    return (Sigma_1 * Sigma_2) / (at * Sigma_2 + Sigma_1 + EPS)


def _calc_gammas(sched: _UANSSchedule, gx: torch.Tensor, y_sigma: torch.Tensor, t: torch.Tensor):
    """Posterior-mean mixing coefficients (official `calc_gammas`): the reverse
    step's mean is `gamma_0 * y0_reparam + gamma_1 * y_t + gamma_2 * y_T_mean`."""
    at         = sched.extract(sched.alphas, t, gx)
    b_tilde_m1 = sched.extract(sched.betas_tilde_m_1, t, gx)
    b_bar_m1   = sched.extract(sched.betas_bar_m_1, t, gx)
    ab_t_m1    = sched.extract(sched.alphas_cumprod_prev, t, gx)
    Sigma_1 = (1 - at) ** 2 * gx + at * (1 - at) * y_sigma
    Sigma_2 = (b_bar_m1 - b_tilde_m1) * gx + b_tilde_m1 * y_sigma
    sqrt_at      = at.clamp(min=0).sqrt()
    sqrt_ab_t_m1 = ab_t_m1.clamp(min=0).sqrt()
    denom = at * Sigma_2 + Sigma_1 + EPS
    gamma_0 = sqrt_ab_t_m1 * Sigma_1 / denom
    gamma_1 = sqrt_at * Sigma_2 / denom
    gamma_2 = ((sqrt_at * (at - 1)) * Sigma_2 + (1 - sqrt_ab_t_m1) * Sigma_1) / denom
    return gamma_0, gamma_1, gamma_2


def _q_sample(y0: torch.Tensor, y_T_mean: torch.Tensor, sched: _UANSSchedule,
             t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Forward diffusion sample (official `q_sample`): `y_t = sqrt(ab_t)*y0 +
    (1 - sqrt(ab_t))*y_T_mean + noise` (noise already scaled by the caller)."""
    sqrt_ab_t = sched.extract(sched.alphas_bar_sqrt, t, y0)
    return sqrt_ab_t * y0 + (1 - sqrt_ab_t) * y_T_mean + noise


def _p_sample_step(denoiser: _Denoiser, y_t: torch.Tensor, y0_hat: torch.Tensor,
                   gx: torch.Tensor, y_T_mean: torch.Tensor, sched: _UANSSchedule,
                   t_scalar: int) -> torch.Tensor:
    """One reverse-diffusion step, y_t -> y_{t-1} (official `p_sample` for
    `t_scalar > 0`, `p_sample_t_1to0` for `t_scalar == 0` -- unified here since
    both solve the same quadratic for the implied Sigma_Y0 and differ only in
    whether posterior noise is added)."""
    b = y_t.shape[0]
    t = torch.full((b,), t_scalar, dtype=torch.long, device=y_t.device)
    eps_theta, sigma_theta = denoiser(y_t, y0_hat, gx, t)
    sigma_theta = sigma_theta.clamp(min=EPS)

    at         = sched.extract(sched.alphas, t, y_t)
    s1mab_t    = sched.extract(sched.one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_ab_t  = (1 - s1mab_t.square()).clamp(min=0).sqrt().clamp(min=EPS)
    b_tilde_m1 = sched.extract(sched.betas_tilde_m_1, t, y_t)
    b_bar_m1   = sched.extract(sched.betas_bar_m_1, t, y_t)
    b_tilde    = sched.extract(sched.betas_tilde, t, y_t)
    b_bar      = sched.extract(sched.betas_bar, t, y_t)

    # Estimate the implied Sigma_Y0 (real y_sigma is unobservable at inference --
    # the official code solves the quadratic lambda_0*x^2 + lambda_1*x + lambda_2 = 0).
    lambda_0 = at * (1 - at) * b_tilde_m1
    lambda_1 = ((1 - at) ** 2 * b_tilde_m1 + at * (1 - at) * (b_bar_m1 - b_tilde_m1)) * gx \
               - sigma_theta * (at * b_tilde_m1 + at * (1 - at))
    lambda_2 = gx ** 2 * (1 - at) ** 2 * (b_bar_m1 - b_tilde_m1) \
               - sigma_theta * gx * (at * b_bar_m1 - at * b_tilde_m1 + (1 - at) ** 2)
    disc = (lambda_1 ** 2 - 4 * lambda_0 * lambda_2).clamp(min=0.0)
    sigma_y0_hat = ((-lambda_1 + disc.sqrt()) / (2 * lambda_0 + EPS)).clamp(min=EPS)

    noise_var = ((b_bar - b_tilde) * gx + b_tilde * sigma_y0_hat).clamp(min=EPS)
    y0_reparam = (y_t - (1 - sqrt_ab_t) * y_T_mean - eps_theta * noise_var.sqrt()) / sqrt_ab_t
    y0_reparam = y0_reparam.clamp(-CLAMP, CLAMP)

    gamma_0, gamma_1, gamma_2 = _calc_gammas(sched, gx, sigma_y0_hat, t)
    mean = gamma_0 * y0_reparam + gamma_1 * y_t + gamma_2 * y_T_mean

    if t_scalar == 0:
        return mean.clamp(-CLAMP, CLAMP)   # deterministic final step (official p_sample_t_1to0)
    z = torch.randn_like(y_t)
    return (mean + sigma_theta.sqrt() * z).clamp(-CLAMP, CLAMP)


# ══════════════════════════════════════════════════════════════════════════════
#  NsDiff model wrapper (mirrors tsdiff_model.TSDiff's role/shape)
# ══════════════════════════════════════════════════════════════════════════════

class NsDiff:
    """Self-contained univariate NsDiff: LSNM (f_phi/g_psi) + UANS diffusion."""

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON, hidden_mean=HIDDEN_MEAN,
                hidden_sigma=HIDDEN_SIGMA, hidden_denoise=HIDDEN_DENOISE,
                sigma_kernel=SIGMA_KERNEL, T=K_DENOISE, beta_start=BETA_START,
                beta_end=BETA_END, lr=LR, wd=WEIGHT_DECAY, device="cpu"):
        self.seq_len = seq_len
        self.horizon = horizon
        self.T       = T
        self.device  = torch.device(device)

        self.sched     = _UANSSchedule(T, beta_start, beta_end, device)
        self.mean_net  = _MeanBackbone(seq_len, horizon, hidden_mean).to(self.device)
        self.sigma_net = _SigmaBackbone(seq_len, horizon, hidden_sigma, sigma_kernel).to(self.device)
        self.denoiser  = _Denoiser(T + 1, hidden_denoise).to(self.device)
        self.sigma_kernel = sigma_kernel

        self.opt = torch.optim.AdamW(
            list(self.mean_net.parameters()) + list(self.sigma_net.parameters())
            + list(self.denoiser.parameters()),
            lr=lr, weight_decay=wd,
        )

    def _nets(self):
        return self.mean_net, self.sigma_net, self.denoiser

    def train(self, hist_batch, target_batch, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=False):
        """hist_batch: [N, seq_len], target_batch: [N, horizon] (standardized
        returns). Joint training of f_phi, g_psi and the denoiser (official
        loss: `kl_loss + loss1 + loss2`, see module docstring)."""
        hist = torch.tensor(hist_batch,   dtype=torch.float32, device=self.device).unsqueeze(-1)
        tgt  = torch.tensor(target_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
        n = hist.shape[0]
        for net in self._nets():
            net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                x, y = hist[idx], tgt[idx]
                b = x.shape[0]

                y0_hat = self.mean_net(x)
                gx     = self.sigma_net(x).clamp(min=EPS)
                y_sigma = _wv_sigma_trailing(torch.cat([x, y], dim=1), self.sigma_kernel)
                y_sigma = y_sigma[:, -y.shape[1]:, :].clamp(min=EPS)

                t = torch.randint(0, self.T, (b // 2 + 1,), device=self.device)
                t = torch.cat([t, self.T - 1 - t], dim=0)[:b]

                y_T_mean = y0_hat
                noise_var = _forward_noise_var(self.sched, gx, y_sigma, t).clamp(min=EPS)
                e = torch.randn_like(y)
                noise = e * noise_var.sqrt()
                y_t = _q_sample(y, y_T_mean, self.sched, t, noise)

                eps_pred, sigma_theta = self.denoiser(y_t, y0_hat, gx, t)
                sigma_theta = sigma_theta.clamp(min=EPS)
                sig_tilde = _sigma_tilde(self.sched, gx, y_sigma, t).clamp(min=EPS)

                loss_mean  = F.mse_loss(y0_hat, y)
                loss_sigma = F.mse_loss(gx.sqrt(), y_sigma.sqrt())
                ratio = sig_tilde / sigma_theta
                kl_loss = F.mse_loss(eps_pred, e) + ratio.mean() - torch.log(ratio.clamp(min=EPS)).mean()
                loss = kl_loss + loss_mean + loss_sigma

                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.mean_net.parameters()) + list(self.sigma_net.parameters())
                    + list(self.denoiser.parameters()), 1.0)
                self.opt.step()
        for net in self._nets():
            net.eval()

    @torch.no_grad()
    def sample_paths(self, hist_window, n_samples=N_SAMPLES, **_ignored):
        """Ancestrally sample `n_samples` full return paths conditioned on
        `hist_window` [seq_len]; return standardized step-returns as np
        [n_samples, horizon]. `**_ignored` absorbs a stray `k_denoise`/
        `ddim_eta` kwarg so call sites shared with tsdiff_model's contract
        (which does take those) don't need a special case."""
        for net in self._nets():
            net.eval()
        x1 = torch.tensor(hist_window, dtype=torch.float32, device=self.device).view(1, -1, 1)
        y0_hat = self.mean_net(x1).expand(n_samples, -1, -1)
        gx     = self.sigma_net(x1).clamp(min=EPS).expand(n_samples, -1, -1)
        y_T_mean = y0_hat

        z = torch.randn_like(y_T_mean)
        y = (gx.clamp(min=0).sqrt() * z + y_T_mean).clamp(-CLAMP, CLAMP)
        for t_scalar in reversed(range(self.T)):
            y = _p_sample_step(self.denoiser, y, y0_hat, gx, y_T_mean, self.sched, t_scalar)
        return y[:, :, 0].detach().cpu().numpy()

    def sample_next(self, hist_window, n_samples=N_SAMPLES, **_ignored):
        """Step-0 (next-step) standardized-return samples as np [n_samples]."""
        return self.sample_paths(hist_window, n_samples, **_ignored)[:, 0]


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


def fit_nsdiff(train: pd.Series, seq_len=SEQ_LEN, horizon=HORIZON, hidden_mean=HIDDEN_MEAN,
              hidden_sigma=HIDDEN_SIGMA, hidden_denoise=HIDDEN_DENOISE,
              sigma_kernel=SIGMA_KERNEL, k_denoise=K_DENOISE, epochs=EPOCHS,
              batch_size=BATCH_SIZE):
    """Fit an NsDiff model once on `train`'s log-returns. Returns (model, mu, sd).

    `mu`/`sd` are the standardization stats of `train` -- the caller is
    responsible for reusing them (never recomputing) on any later data to
    avoid lookahead in a walk-forward / train-once-forward protocol."""
    if len(train) <= seq_len + horizon:
        raise ValueError(
            f"train series has {len(train)} points, but seq_len={seq_len} + "
            f"horizon={horizon} requires more than {seq_len + horizon} points.")

    train_p = train.values.astype(float)
    r = _log_returns(train_p)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd

    H_win, T_win = _make_windows(z, seq_len, horizon)
    if len(H_win) == 0:
        raise ValueError("not enough return history to build training windows.")

    model = NsDiff(seq_len, horizon, hidden_mean, hidden_sigma, hidden_denoise,
                   sigma_kernel, T=k_denoise)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)
    return model, mu, sd


def forecast_from_fitted(model: NsDiff, hist_window, mu: float, sd: float,
                         last_price: float, horizons=None, n_samples=N_SAMPLES,
                         **_ignored) -> dict:
    """Sample forecasts from an already-fitted model -- no `model.train()` call.

    `hist_window` is a standardized-return history (>= model.seq_len values);
    only the last `model.seq_len` are used as conditioning. `mu`/`sd`/`last_price`
    de-standardize the sampled returns back into a price. `horizons` is a list of
    steps-ahead (1-indexed, capped at `model.horizon`); defaults to `[1]`. Multi-step
    horizons read off the cumulative sum of the sampled return path, exactly like
    `benchmarks/multi_horizon.forecast_horizons_nsdiff`.

    Returns `{h: price_samples}` -- one `[n_samples]` array of price samples per
    requested horizon."""
    if horizons is None:
        horizons = [1]
    window = np.asarray(hist_window[-model.seq_len:], dtype=np.float32)
    paths = model.sample_paths(window, n_samples=n_samples)   # [n_samples, model.horizon]
    out = {}
    for h in horizons:
        hh = min(int(h), model.horizon)
        cum_r = paths[:, :hh].sum(axis=1) * sd + hh * mu
        out[h] = last_price * np.exp(cum_r)
    return out


def run_nsdiff(train: pd.Series, test: pd.Series,
              seq_len=SEQ_LEN, horizon=HORIZON, hidden_mean=HIDDEN_MEAN,
              hidden_sigma=HIDDEN_SIGMA, hidden_denoise=HIDDEN_DENOISE,
              sigma_kernel=SIGMA_KERNEL, k_denoise=K_DENOISE, epochs=EPOCHS,
              batch_size=BATCH_SIZE, n_samples=N_SAMPLES,
              keep_samples: bool = False) -> dict:
    """Train on the train window's returns, roll 1-step-ahead over the test window.

    Point forecast = mean of the sample cloud; 95% PI = 2.5/97.5 sample
    quantiles (NsDiff's own predictive distribution, LSNM+UANS-driven).

    `keep_samples` (False = off, default -- no extra memory for existing callers):
    the `n_samples`-wide price cloud already drawn at each step is instead kept in
    `result["ensemble"]` (list of length len(test), one [n_samples] price array
    per step) for empirical CRPS (cf. model_artifacts/crps_kpis.py)."""
    t0 = time.time()
    model, mu, sd = fit_nsdiff(train, seq_len, horizon, hidden_mean, hidden_sigma,
                              hidden_denoise, sigma_kernel, k_denoise, epochs, batch_size)

    train_p    = train.values.astype(float)
    buffer     = list((_log_returns(train_p) - mu) / sd)
    last_price = float(train_p[-1])
    test_p     = test.values.astype(float)

    preds, lower, upper = [], [], []
    ensembles = [] if keep_samples else None
    for i in range(len(test_p)):
        price_samples = forecast_from_fitted(
            model, buffer, mu, sd, last_price, horizons=[1], n_samples=n_samples)[1]
        preds.append(float(np.mean(price_samples)))
        lower.append(float(np.quantile(price_samples, 0.025)))
        upper.append(float(np.quantile(price_samples, 0.975)))
        if keep_samples:
            ensembles.append(price_samples)
        realised_r = np.log(test_p[i] / last_price)
        buffer.append((realised_r - mu) / sd)
        last_price = float(test_p[i])

    preds = np.asarray(preds); lower = np.asarray(lower); upper = np.asarray(upper)
    train_time = time.time() - t0
    metrics = compute_metrics(test_p, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    result = {**metrics, "predictions": preds, "lower": lower, "upper": upper,
              "index": test.index, "actual": test_p}
    if ensembles is not None:
        result["ensemble"] = ensembles
    return result


def next_step_nsdiff(series: pd.Series, seq_len=SEQ_LEN, horizon=HORIZON,
                     hidden_mean=HIDDEN_MEAN, hidden_sigma=HIDDEN_SIGMA,
                     hidden_denoise=HIDDEN_DENOISE, sigma_kernel=SIGMA_KERNEL,
                     k_denoise=K_DENOISE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     n_samples=N_SAMPLES):
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
    model = NsDiff(seq_len, horizon, hidden_mean, hidden_sigma, hidden_denoise,
                   sigma_kernel, T=k_denoise)
    model.train(H_win, T_win, epochs=epochs, batch_size=batch_size)

    window = z[-seq_len:].astype(np.float32)
    z_samples = model.sample_next(window, n_samples=n_samples)
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
    ax.plot(idx, result["predictions"], label="NsDiff forecast", color="tab:purple", lw=1.3)
    ax.fill_between(idx, result["lower"], result["upper"], color="tab:purple",
                    alpha=0.20, label="95% PI (sample quantiles)")
    ax.set_title(f"NsDiff diffusion — {ticker} (walk-forward 1-step)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="NsDiff diffusion forecasting (wwy155/NsDiff port)")
    p.add_argument("--ticker", default="BTC-USD", help="yfinance ticker (BTC-USD, SPY, GC=F)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--hidden-mean", type=int, default=HIDDEN_MEAN)
    p.add_argument("--hidden-sigma", type=int, default=HIDDEN_SIGMA)
    p.add_argument("--hidden-denoise", type=int, default=HIDDEN_DENOISE)
    p.add_argument("--sigma-kernel", type=int, default=SIGMA_KERNEL)
    p.add_argument("--k-denoise", type=int, default=K_DENOISE,
                   help="diffusion steps == ancestral sampling steps (see module docstring)")
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
        pred, lo, hi = next_step_nsdiff(prices, hidden_mean=args.hidden_mean,
                                        hidden_sigma=args.hidden_sigma,
                                        hidden_denoise=args.hidden_denoise,
                                        sigma_kernel=args.sigma_kernel,
                                        k_denoise=args.k_denoise, epochs=args.epochs,
                                        n_samples=args.n_samples)
        print(f"Last close      : {prices.iloc[-1]:,.4f}")
        print(f"Next-step point : {pred:,.4f}")
        print(f"95% interval    : [{lo:,.4f}, {hi:,.4f}]")
        return

    split = int(len(prices) * (1 - args.test_ratio))
    train, test = prices.iloc[:split], prices.iloc[split:]
    print(f"Train: {len(train)}  Test: {len(test)}  "
          f"NsDiff(hidden_mean={args.hidden_mean}, hidden_sigma={args.hidden_sigma}) "
          f"epochs={args.epochs}\n")
    print("Note: training a diffusion net + rolling ancestral sampling can take a while.\n")

    t0 = time.time()
    result = run_nsdiff(train, test, hidden_mean=args.hidden_mean, hidden_sigma=args.hidden_sigma,
                        hidden_denoise=args.hidden_denoise, sigma_kernel=args.sigma_kernel,
                        k_denoise=args.k_denoise, epochs=args.epochs, n_samples=args.n_samples)
    elapsed = time.time() - t0
    print(f"=== NsDiff diffusion — {args.ticker} ===  ({elapsed:.1f}s total)")
    for k, v in result.items():
        if k in ("predictions", "lower", "upper", "index", "actual"):
            continue
        print(f"  {k:<18}: {v}")

    if args.plot:
        save_plot(result, args.ticker, args.plot)


if __name__ == "__main__":
    main()
