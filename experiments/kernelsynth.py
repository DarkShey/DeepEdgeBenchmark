"""
kernelsynth.py -- generateur de series synthetiques par composition de noyaux
gaussiens, d'apres la recette KernelSynth de Chronos (Ansari et al., 2024, §4.2).

POURQUOI ICI. Chantier C : le regime MENSUEL ne dispose que d'une centaine
d'observations par actif. A `seq_len=30` et `horizon=3`, cela laisse quelques
dizaines de fenetres d'entrainement -- deux ordres de grandeur sous ce dont
dispose le regime quotidien. Le brief prevoit explicitement l'augmentation par
donnees synthetiques comme troisieme voie, "si (i) sous-entraine".

LA RECETTE, telle qu'implementee ici :
  1. une BANQUE DE NOYAUX declaree : lineaire, RBF a plusieurs longueurs de
     correlation, periodiques a plusieurs periodes, et bruit blanc ;
  2. on tire J noyaux dans la banque (avec remise), J ~ U{1..J_max} ;
  3. on les combine deux a deux par une operation tiree au hasard, + ou x
     (l'addition superpose des comportements, la multiplication les module --
     c'est ce qui produit des motifs qu'aucun noyau seul ne donne) ;
  4. on tire une realisation du processus gaussien de moyenne nulle et de
     covariance le noyau composite, sur une grille reguliere de longueur L.

CE QUE CE MODULE NE FAIT PAS, volontairement : il ne connait NI les prix, NI
les actifs, NI NsDiff. Il produit des series standardisees, point. La decision
de les melanger a des donnees reelles, et dans quelle proportion, appartient a
l'appelant et doit y etre declaree -- pas cachee ici.

LIMITE A CITER AVEC TOUT RESULTAT QUI EN DEPEND : une serie KernelSynth n'est
pas un actif financier. Elle n'a ni queues lourdes, ni clustering de volatilite,
ni asymetrie -- un processus gaussien a covariance fixe est, par construction,
homoscedastique conditionnellement. L'augmentation apporte de la DIVERSITE DE
FORMES (tendances, cycles, ruptures de correlation), pas du realisme
stylise. C'est utile contre le sur-apprentissage d'un echantillon minuscule ;
ce n'est pas un substitut a des donnees.
"""

import numpy as np

JITTER = 1e-8
DEFAULT_MAX_KERNELS = 5

# Banque declaree. Les periodes sont exprimees en PAS de la grille : sur une
# serie mensuelle, 12 = cycle annuel, 6 = semestriel, 4 = trimestriel, 3 = ...
# Le choix couvre les periodicites plausibles a ce pas de temps sans en
# privilegier une.
DEFAULT_BANK = (
    ("linear", {}),
    ("rbf", {"lengthscale": 1.0}),
    ("rbf", {"lengthscale": 3.0}),
    ("rbf", {"lengthscale": 10.0}),
    ("rbf", {"lengthscale": 30.0}),
    ("periodic", {"period": 3.0, "lengthscale": 3.0}),
    ("periodic", {"period": 4.0, "lengthscale": 4.0}),
    ("periodic", {"period": 6.0, "lengthscale": 6.0}),
    ("periodic", {"period": 12.0, "lengthscale": 12.0}),
    ("white", {"level": 1.0}),
)


def _linear(t: np.ndarray, c: float = 1.0) -> np.ndarray:
    x = t[:, None]
    return x @ x.T + c


def _rbf(t: np.ndarray, lengthscale: float = 1.0) -> np.ndarray:
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / lengthscale) ** 2)


def _periodic(t: np.ndarray, period: float = 12.0, lengthscale: float = 1.0) -> np.ndarray:
    d = np.abs(t[:, None] - t[None, :])
    return np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 / lengthscale ** 2)


def _white(t: np.ndarray, level: float = 1.0) -> np.ndarray:
    return level * np.eye(t.size)


KERNELS = {"linear": _linear, "rbf": _rbf, "periodic": _periodic, "white": _white}


def kernel_matrix(name: str, t: np.ndarray, **params) -> np.ndarray:
    if name not in KERNELS:
        raise KeyError(f"noyau inconnu : {name!r} (banque : {sorted(KERNELS)})")
    return KERNELS[name](t, **params)


def _normalise(K: np.ndarray) -> np.ndarray:
    """Ramene la diagonale a l'ordre de grandeur 1. Sans cela, le noyau lineaire
    (dont la variance croit en t^2) ecraserait tous les autres des qu'il entre
    dans un produit."""
    scale = float(np.mean(np.diag(K)))
    return K / scale if scale > 0 else K


def compose_kernel(t: np.ndarray, rng: np.random.Generator, bank=DEFAULT_BANK,
                   max_kernels: int = DEFAULT_MAX_KERNELS) -> tuple:
    """Tire J noyaux et les combine de gauche a droite par + ou x. Renvoie
    (matrice de covariance, description lisible de la composition)."""
    n = int(rng.integers(1, max_kernels + 1))
    picks = [bank[i] for i in rng.integers(0, len(bank), size=n)]
    K = _normalise(kernel_matrix(picks[0][0], t, **picks[0][1]))
    desc = [f"{picks[0][0]}{picks[0][1] or ''}"]
    for name, params in picks[1:]:
        op = "+" if rng.random() < 0.5 else "*"
        Kj = _normalise(kernel_matrix(name, t, **params))
        K = K + Kj if op == "+" else K * Kj
        desc.append(f" {op} {name}{params or ''}")
    return _normalise(K), "".join(desc)


def sample_series(length: int, rng: np.random.Generator, bank=DEFAULT_BANK,
                  max_kernels: int = DEFAULT_MAX_KERNELS) -> tuple:
    """Une realisation du processus gaussien de covariance composite, centree
    reduite. Renvoie (serie [length], description du noyau)."""
    t = np.arange(length, dtype=float)
    K, desc = compose_kernel(t, rng, bank, max_kernels)
    L = np.linalg.cholesky(K + JITTER * np.eye(length) * max(1.0, float(np.max(np.diag(K)))))
    y = L @ rng.standard_normal(length)
    sd = y.std()
    return (y - y.mean()) / (sd if sd > 1e-12 else 1.0), desc


def generate(n_series: int, length: int, seed: int = 0, bank=DEFAULT_BANK,
             max_kernels: int = DEFAULT_MAX_KERNELS) -> tuple:
    """`n_series` series synthetiques standardisees de longueur `length`.
    Renvoie (tableau [n_series, length], liste des compositions tirees)."""
    rng = np.random.default_rng(seed)
    series, descs = [], []
    for _ in range(n_series):
        y, desc = sample_series(length, rng, bank, max_kernels)
        series.append(y)
        descs.append(desc)
    return np.stack(series), descs
