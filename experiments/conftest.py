"""Shared pytest fixtures for experiments/.

Same TensorFlow import-order guardrail as models/conftest.py (that conftest
only protects tests collected under models/, not experiments/ — this file
plugs the same hole here). duel_sampling_adapters.py imports arima_model /
sarima_model (which import yfinance/statsmodels) at module load time and only
imports lstm_model / tensorflow lazily inside fit_lstm_state(); if the first
TensorFlow op of the pytest session happens after that import ordering
without this pre-configuration, model.fit() deadlocks indefinitely on this
machine (confirmed: TFE_Execute -> absl::Mutex::Block -> a Notification never
signalled, 0% CPU afterwards — not a slow run, a hang). Importing + forcing
TensorFlow single-threaded here, before test collection imports anything
else, eliminates the deadlock class (cf. models/conftest.py and
model_artifacts/pipeline.py for the original investigation)."""

import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
    _tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
