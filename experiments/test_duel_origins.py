"""Correctness tests for duel_origins.build_common_origins/describe_origins —
the embargo guardrail is the whole point of this module, so it gets the most
scrutiny (a silent off-by-one here would leak a training-adjacent origin into
validation/test for every one of the 6 models downstream)."""

import numpy as np
import pandas as pd
import pytest

import duel_origins as do
import epoch_sweep as es


def _synthetic_weekly(n=120, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-08", periods=n, freq="W-FRI")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.Series(prices, index=idx, name="Close")


def test_default_embargo_is_horizon_max_minus_one():
    assert do.HORIZON_MAX == 3
    weekly = _synthetic_weekly()
    train_end, val_pos, test_pos = do.build_common_origins(weekly, n_val=10, n_test=10)
    train_end_e, val_pos_e, test_pos_e = do.build_common_origins(
        weekly, n_val=10, n_test=10, embargo=do.HORIZON_MAX - 1)
    assert (train_end, val_pos, test_pos) == (train_end_e, val_pos_e, test_pos_e)


def test_embargo_gap_between_train_and_validation():
    weekly = _synthetic_weekly()
    embargo = 2
    train_end, val_pos, _ = do.build_common_origins(weekly, n_val=10, n_test=10, embargo=embargo)
    # `embargo` positions strictly between train_end and the first surviving
    # validation origin are skipped (gap of embargo+1 in position terms).
    assert val_pos[0] - train_end == embargo + 1


def test_embargo_gap_between_validation_and_test():
    weekly = _synthetic_weekly()
    embargo = 2
    _, val_pos, test_pos = do.build_common_origins(weekly, n_val=10, n_test=10, embargo=embargo)
    assert test_pos[0] - val_pos[-1] == embargo + 1


def test_zero_embargo_matches_three_way_split_exactly():
    weekly = _synthetic_weekly()
    train_end, val_pos, test_pos = do.build_common_origins(weekly, n_val=10, n_test=10, embargo=0)
    train_end_ref, val_pos_ref, test_pos_ref = es.three_way_split(weekly, n_val=10, n_test=10)
    assert (train_end, val_pos, test_pos) == (train_end_ref, val_pos_ref, test_pos_ref)


def test_val_and_test_counts_unaffected_by_embargo():
    weekly = _synthetic_weekly()
    for embargo in (0, 1, 2, 5):
        _, val_pos, test_pos = do.build_common_origins(weekly, n_val=8, n_test=12, embargo=embargo)
        assert len(val_pos) == 8
        assert len(test_pos) == 12


def test_blocks_stay_disjoint_and_chronological():
    weekly = _synthetic_weekly()
    train_end, val_pos, test_pos = do.build_common_origins(weekly, n_val=10, n_test=10, embargo=2)
    assert train_end < val_pos[0]
    assert val_pos == sorted(val_pos)
    assert val_pos[-1] < test_pos[0]
    assert test_pos == sorted(test_pos)
    assert len(set(val_pos) & set(test_pos)) == 0


def test_negative_embargo_raises():
    weekly = _synthetic_weekly()
    with pytest.raises(ValueError):
        do.build_common_origins(weekly, n_val=10, n_test=10, embargo=-1)


def test_raises_when_not_enough_history_for_embargo():
    weekly = _synthetic_weekly(n=30)
    with pytest.raises(ValueError):
        do.build_common_origins(weekly, n_val=10, n_test=10, embargo=5)


# ── describe_origins ─────────────────────────────────────────────────────────

def _synthetic_daily(n_days=900, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.Series(prices, index=idx, name="Close")


def test_describe_origins_matches_week_targets():
    daily = _synthetic_daily()
    weekly, weekly_dates = es.build_weekly(daily)
    train_end, val_pos, test_pos = do.build_common_origins(weekly, n_val=5, n_test=8, embargo=2)

    records = do.describe_origins(weekly_dates, daily, val_pos)
    assert len(records) == len(val_pos)
    for rec, m in zip(records, val_pos):
        origin_date, daily_pos, target_dates, daily_horizons = es.week_targets(weekly_dates, daily, m)
        assert rec["pos"] == m
        assert rec["origin_date"] == origin_date
        assert rec["daily_pos"] == daily_pos
        assert rec["target_dates"] == target_dates
        assert rec["daily_horizons"] == daily_horizons


def test_describe_origins_three_targets_per_origin():
    daily = _synthetic_daily()
    weekly, weekly_dates = es.build_weekly(daily)
    _, val_pos, _ = do.build_common_origins(weekly, n_val=5, n_test=8, embargo=2)
    records = do.describe_origins(weekly_dates, daily, val_pos)
    for rec in records:
        assert len(rec["target_dates"]) == 3
        assert len(rec["daily_horizons"]) == 3
        assert rec["daily_horizons"] == sorted(rec["daily_horizons"])
