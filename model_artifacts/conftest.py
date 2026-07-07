"""
model_artifacts/conftest.py — désactive le GPU/Metal pour TensorFlow avant toute
collecte de test.

pytest importe les conftest.py du répertoire cible avant d'importer les modules de
test eux-mêmes — c'est l'endroit le plus fiable pour poser ceci avant le premier
`import tensorflow` déclenché en interne par lstm_model.py (importé paresseusement
par model_artifacts.pipeline). Sur cette machine, laisser TensorFlow initialiser le
backend Metal/GPU bloque le process indéfiniment (confirmé par un `sample` du PID
pendant le blocage : pile Python figée dans les frameworks GPUCompiler/MPS, 0% CPU
ensuite, aucun accès réseau en cause). Un LSTM de cette taille (seq_len=30, 64
unités) n'a de toute façon rien à gagner du GPU.

Note : `CUDA_VISIBLE_DEVICES` est une variable NVIDIA — sans effet sur Metal
(Apple Silicon). Le vrai réglage qui désactive le GPU ici est
`tf.config.set_visible_devices([], 'GPU')`, appelé juste après l'import.

Deadlock distinct constaté ensuite lors du run réel : le thread principal bloqué
dans TFE_Execute -> absl::Mutex::Block -> attente d'une Notification jamais
signalée — un deadlock du pool de threads interne de TensorFlow (connu sur
certaines configs Apple Silicon en cas de sur-souscription de threads). Forcer
le mono-thread élimine cette classe de deadlock, sans coût mesurable pour un
LSTM aussi petit.
"""

import os
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    # best-effort : si TF a déjà exécuté une op ailleurs (import concurrent, config déjà
    # verrouillée), on ne bloque pas la collecte des tests pour un réglage best-effort.
    pass
