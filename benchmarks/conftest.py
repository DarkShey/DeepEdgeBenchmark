"""
benchmarks/conftest.py — désactive le GPU/Metal pour TensorFlow avant toute collecte de
test, même garde-fou que models/conftest.py / model_artifacts/conftest.py / experiments/conftest.py.

multi_horizon.py importe lstm_model paresseusement (fit_lstm/forecast_from_fitted_lstm) --
si le premier `import tensorflow` de la session pytest survient après que
yfinance/statsmodels aient déjà tourné (ordre interne de lstm_model.py), le premier
model.fit()/model.predict() se bloque indéfiniment sur cette machine (deadlock confirmé,
cf. model_artifacts/lstm_worker.py pour l'investigation complète). Poser la config ici,
avant toute collecte de test, élimine la classe de deadlock.
"""

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
