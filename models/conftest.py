"""
Shared pytest fixtures for models/ — synthetic data + deterministic seeds.

No network access anywhere in this suite: fetch_data() is never exercised with
real arguments, only monkeypatched or called on data already in memory.
"""

import importlib
import os
import random

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

MODEL_MODULE_NAMES = ["arima_model", "sarima_model", "prophet_model", "lstm_model"]

# run_<model> / next_step_<model> are named per-model, not uniformly — map them here
# rather than assuming a naming convention (cf. actual code, not the brief's guess).
RUN_FN_NAME = {
    "arima_model": "run_arima_garch",
    "sarima_model": "run_sarima",
    "prophet_model": "run_prophet",
    "lstm_model": "run_lstm",
}
NEXT_STEP_FN_NAME = {
    "arima_model": "next_step_arima_garch",
    "sarima_model": "next_step_sarima",
    "prophet_model": "next_step_prophet",
    "lstm_model": "next_step_lstm",
}
# LSTM is the only one slow enough to need a reduced epoch count for test speed.
EXTRA_RUN_KWARGS = {"lstm_model": {"epochs": 2}}


def synthetic_series(n=120, seed=0):
    """Deterministic synthetic price series: random walk + slight drift, always positive."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.Series(prices, index=idx, name="Close")


def make_train_test(n_train=90, n_test=10, seed=0):
    """Train/test split sized to exceed the LSTM look-back (30) while staying fast."""
    s = synthetic_series(n=n_train + n_test, seed=seed)
    return s.iloc[:n_train], s.iloc[n_train:]


def reset_seeds(seed=0):
    """Reset all RNGs — call before each run_<model>/next_step_<model> invocation that
    needs to be reproducible (esp. LSTM/TensorFlow, which is not bit-exact across
    machines but is reproducible run-to-run within one process given a fixed seed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _fixed_seeds():
    reset_seeds(0)
    yield


@pytest.fixture
def synthetic_series_fixture():
    return synthetic_series()


@pytest.fixture(params=MODEL_MODULE_NAMES)
def model_module(request):
    """Parametrizes a test over all 4 forecaster modules."""
    return importlib.import_module(request.param)


@pytest.fixture
def run_fn(model_module):
    return getattr(model_module, RUN_FN_NAME[model_module.__name__])


@pytest.fixture
def next_step_fn(model_module):
    return getattr(model_module, NEXT_STEP_FN_NAME[model_module.__name__])


@pytest.fixture
def run_kwargs(model_module):
    return dict(EXTRA_RUN_KWARGS.get(model_module.__name__, {}))
