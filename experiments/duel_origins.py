"""
duel_origins.py — common origin generator for the diffusion-vs-classiques duel
(BRIEF_unification_protocole_duel.md §2.1, resolves ecart E2 de l'audit).

Produces ONE set of origins + target dates, shared by all 6 models (TSDiff,
ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive) — the asymmetry that made every
existing comparison (production pipelines A & B) statistically incomparable
(TSDiff frozen at T0, the other 5 refit every origin, cf. pooled_analysis.py's
own refusal to rank models for that reason) is not tolerated here: every
adapter in duel_sampling_adapters.py is handed the exact same
(train_end_pos, val_origins_pos, test_origins_pos) produced by
build_common_origins() below, so "mêmes origines, mêmes dates-cibles" is
enforced by construction, not by convention.

Re-estimation rule (declared, not hidden — brief §2.1 makes this mandatory):
FROZEN AT T0 for ALL SIX models, not "refit every origin" for the 5 classics.
This mirrors weekly_headtohead(_v2).py's existing TSDiff protocol exactly:
each model is fit/trained EXACTLY ONCE on data <= T0, then every later origin
only updates the model's *conditioning state* with realised data (SARIMAX
`.append(refit=False)`, GARCH's own fixed variance recursion fed realised
residuals, an LSTM look-back buffer, TSDiff's returns buffer, Prophet's
deterministic future extrapolation from its original fit) — never its
estimated parameters/weights. This is a deliberate, declared choice (not the
"everyone refits every origin" alternative), for two reasons: (1) symmetry is
the brief's actual hard requirement, and freezing is what TSDiff and the whole
weekly_headtohead* line already does, so extending it to the 5 classics costs
one fit per (asset, model) instead of one fit per (asset, model, origin) — of
order n_origins times cheaper, tractable on CPU; (2) it makes "same rule for
everyone" auditable structurally (one fit call per model, full stop) rather
than by convention. The residual asymmetry this trades away — a model refit at
every origin could in principle adapt faster to a regime change than one whose
parameters are frozen at T0 — is DECLARED here, not measured: quantifying it
would require also running the "refit every origin" protocol side by side,
which is out of scope of this brief (cf. brief's own "briefs suivants" list,
#6 "entraînement global").

Horizons: 3 weekly horizons (W1/W2/W3, brief §2.4's "3 horizons"), matching
weekly_headtohead(_v2).py's own grid — deliberately, so this new common-origin
duel targets the SAME (asset, horizon) grid the tuteur's audit already reads.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epoch_sweep import three_way_split, week_targets  # noqa: E402 (reuse, don't reimplement)

HORIZONS_WEEKS = (1, 2, 3)
HORIZON_LABELS = ("W1", "W2", "W3")
HORIZON_MAX = max(HORIZONS_WEEKS)


def build_common_origins(weekly: pd.Series, n_val: int, n_test: int,
                         horizon_max: int = HORIZON_MAX, embargo: int = None):
    """One shared (train_end_pos, val_origins_pos, test_origins_pos), handed
    identically to every one of the 6 sampling adapters.

    Purge (structural, always on, nothing to do here): three_way_split's three
    blocks are disjoint and chronological, and every window/label builder used
    by the 6 adapters (TSDiff's `_make_windows`, ARIMA/SARIMA/Prophet/LSTM's
    training calls) constructs its multi-step training targets strictly WITHIN
    the training block (never reading past train_end_pos for a training
    label) — no training example's label window can overlap the val/test
    block already, for any of the 6 models.

    Embargo (this function's actual job — brief §2.1 "purge + embargo pour les
    horizons > 1 jour", López de Prado): `embargo` extra origins (default
    horizon_max - 1, the longest reach-back of a max-horizon W3 target) are
    dropped from the START of the validation block and from the START of the
    test block. Without this gap, the LAST train-block origin's own W3 target
    could land inside (or immediately adjacent to) the FIRST validation
    origin's own W1-W3 target window — adjacent origins' target windows
    overlap by construction (documented already in paired_test.py's block-
    bootstrap docstring) — so the embargo removes that origin-level adjacency
    at both the train/validation and validation/test boundaries.

    Hyperparameter selection guardrail (brief §2.1, verrou E1): `val_pos` is
    the ONLY block any model-selection step (e.g. TSDiff epoch count) may
    read; `test_pos` must never be touched until the final scoring pass —
    exactly epoch_sweep.py's existing three_way_split contract, preserved
    here unchanged.
    """
    if embargo is None:
        embargo = horizon_max - 1
    if embargo < 0:
        raise ValueError(f"embargo={embargo} must be >= 0")
    train_end_pos, val_pos_full, test_pos_full = three_way_split(
        weekly, n_val + embargo, n_test + embargo)
    val_pos = val_pos_full[embargo:]
    test_pos = test_pos_full[embargo:]
    return train_end_pos, val_pos, test_pos


def describe_origins(weekly_dates: pd.Series, daily: pd.Series, positions: list) -> list:
    """[{origin_date, daily_pos, target_dates[3], daily_horizons[3]}, ...] for
    `positions` — a thin wrapper around epoch_sweep.week_targets (reused, not
    reimplemented) so callers get one record per origin instead of a raw
    tuple."""
    out = []
    for m in positions:
        origin_date, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
        out.append({
            "pos": int(m), "origin_date": origin_date, "daily_pos": daily_pos,
            "target_dates": target_dates, "daily_horizons": daily_horizons,
        })
    return out
