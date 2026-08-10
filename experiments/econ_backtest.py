"""
econ_backtest.py -- chantier B : transformer des intervalles de prevision en
DECISIONS, et mesurer le resultat en argent.

Pourquoi ce module existe : la calibration est necessaire mais ne dit rien de la
valeur en trading. Un modele peut couvrir 95 % a la perfection et ne rien
rapporter ; un autre peut sous-couvrir et fournir un meilleur signal de taille
de position. Le programme a mesure la calibration jusqu'a epuisement -- il n'a
jamais mesure l'argent.

TOUT EST DECLARE A PRIORI. Aucune strategie, marge, seuil ou niveau de cout de
ce fichier n'a ete choisi apres lecture d'un resultat. Les trois familles
viennent du brief, les niveaux de cout de la structure des actifs (un chiffre
unique de SPY a ETH serait faux partout), les bornes de position du bon sens
(pas de levier).

LES MEMES REGLES POUR LES DEUX MODELES. Chaque fonction de position ne prend en
entree que (last_close, y_pred, y_lower, y_upper) -- ce que les deux modeles
publient. Rien dans ce fichier ne sait quel modele l'appelle : une strategie qui
avantagerait un bras par construction serait visible ici.

CAUSALITE. Aucune fonction ne regarde au-dela de l'origine courante. La
strategie 1 a besoin d'une echelle de largeur pour normaliser : elle utilise la
MEDIANE EXPANSIVE des largeurs deja observees (strictement avant t), jamais la
mediane de la fenetre de test -- qui serait une fuite, et exactement le genre de
fuite qui rend un backtest joli.

RECOUVREMENT DES HORIZONS, declare. A l'horizon h, chaque origine ouvre une
position tenue h semaines : les positions se chevauchent pour h > 1, donc les
PnL par origine sont autocorreles. C'est la situation que le bootstrap PAR BLOCS
du repo traite deja (`paired_test.paired_block_bootstrap_test`, block_length=3),
et c'est ce test qui est utilise -- jamais un t-test i.i.d. Le Sharpe est
annualise par sqrt(52/h), pas sqrt(52).

Ce module ne fait AUCUNE entree-sortie et ne connait ni la base ni les artefacts :
il est entierement testable a la main (`test_econ_backtest.py`).
"""

import numpy as np

# ── constantes declarees ────────────────────────────────────────────────────

W_MAX = 1.0                 # pas de levier : |position| <= 1x le capital
WARMUP_ORIGINS = 8          # strategie 1 : origines sans position, le temps que
                            # la mediane expansive des largeurs ait un sens
VAR_LEVEL = 0.025           # le quantile bas d'un PI 95 % EST la VaR a 2.5 %
VAR_BUDGET = 0.03           # strategie 2 : perte a 2.5 % toleree par sleeve, 3 %
                            # du capital -- ordre de grandeur d'une limite de risque
                            # hebdomadaire courante, declare, jamais optimise
WEEKS_PER_YEAR = 52.0

# Couts unidirectionnels en points de base, par classe d'actif et par niveau.
# Trois niveaux parce que les actifs vont de SPY (spread ~1 bp) a ETH (spread +
# frais d'echange, un ordre de grandeur au-dessus) : un chiffre unique serait
# faux partout. Le cout applique a chaque sleeve est un ALLER-RETOUR = 2 x bps.
COST_LEVELS = {
    "faible":  {"index": 1.0, "bond": 1.0, "crypto": 10.0},
    "central": {"index": 5.0, "bond": 5.0, "crypto": 30.0},
    "eleve":   {"index": 10.0, "bond": 10.0, "crypto": 60.0},
}


# ── 1. familles de strategies ───────────────────────────────────────────────

def expanding_median(x, warmup: int = WARMUP_ORIGINS) -> np.ndarray:
    """Mediane des valeurs STRICTEMENT anterieures a chaque instant. Les
    `warmup` premiers points valent NaN : il n'y a pas encore d'echelle."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.size, np.nan)
    for t in range(warmup, x.size):
        out[t] = np.median(x[:t])
    return out


def positions_inverse_width(last_close, y_pred, y_lower, y_upper,
                            warmup: int = WARMUP_ORIGINS, w_max: float = W_MAX) -> np.ndarray:
    """FAMILLE 1 -- taille par inverse de la largeur du PI.

    La largeur du PI est un proxy de confiance : plus le modele est sur, plus la
    position est grande. Elle est normalisee par la mediane EXPANSIVE des
    largeurs relatives du modele lui-meme, de sorte que la strategie mesure la
    confiance RELATIVE du modele a cet instant -- pas son niveau absolu de
    largeur, qui differe d'un modele et d'un actif a l'autre et rendrait la
    comparaison sans objet.

        w_t = signe(y_pred_t - last_close_t) x min(w_max, mediane_{<t} / largeur_t)

    Le signe vient du point predictif (la mediane du nuage cote diffusion, la
    prevision centrale cote GARCH) compare au prix courant.
    """
    last_close = np.asarray(last_close, dtype=float)
    width_rel = (np.asarray(y_upper, dtype=float) - np.asarray(y_lower, dtype=float)) / last_close
    scale = expanding_median(width_rel, warmup)
    sign = np.sign(np.asarray(y_pred, dtype=float) - last_close)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.where(width_rel > 0, scale / width_rel, 0.0)
    size = np.clip(np.nan_to_num(size, nan=0.0, posinf=0.0), 0.0, w_max)
    return sign * size


def var_from_lower(last_close, y_lower) -> np.ndarray:
    """VaR a 2.5 % en RENDEMENT (nombre negatif quand la borne basse est sous le
    prix courant) : le quantile bas d'un PI 95 % EST la VaR a 2.5 %, il n'y a
    rien a estimer de plus."""
    last_close = np.asarray(last_close, dtype=float)
    return np.asarray(y_lower, dtype=float) / last_close - 1.0


def positions_var_limit(last_close, y_lower, budget: float = VAR_BUDGET,
                        w_max: float = W_MAX) -> np.ndarray:
    """FAMILLE 2 -- limite de risque.

    Sleeve LONG-ONLY (declare) dimensionne pour que sa VaR a 2.5 % vaille
    exactement le budget de risque, plafonne a `w_max` :

        w_t = min(w_max, budget / |VaR_t|)

    Long-only volontairement : y ajouter le signe melangerait la question
    "le modele dimensionne-t-il mieux le risque ?" avec celle de la famille 1.
    Ce qui est teste ici, c'est uniquement la qualite du quantile bas.
    """
    var = np.abs(var_from_lower(last_close, y_lower))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(var > 0, budget / var, w_max)
    return np.clip(np.nan_to_num(w, nan=0.0, posinf=w_max), 0.0, w_max)


def positions_filtered_direction(last_close, y_lower, y_upper, w_max: float = W_MAX) -> np.ndarray:
    """FAMILLE 3 -- signal directionnel filtre par l'intervalle.

    On ne prend position que si le PI EXCLUT le rendement nul, c'est-a-dire si
    l'intervalle tout entier est au-dessus (long) ou au-dessous (short) du prix
    courant. Mesure directement si des intervalles "honnetes" filtrent mieux les
    faux signaux : un modele trop etroit prendra beaucoup de positions et se
    trompera, un modele trop large n'en prendra jamais.
    """
    last_close = np.asarray(last_close, dtype=float)
    lo = np.asarray(y_lower, dtype=float)
    hi = np.asarray(y_upper, dtype=float)
    return np.where(lo > last_close, w_max, np.where(hi < last_close, -w_max, 0.0))


def positions_normalised_direction(last_close, y_pred, y_lower, y_upper, k: float,
                                   w_max: float = W_MAX) -> np.ndarray:
    """FAMILLE 3', REMPLACANTE -- signal directionnel NORMALISE.

    La famille 3 originelle (`positions_filtered_direction`) n'emet jamais : elle
    exige que l'intervalle tout entier exclue le prix courant, or la largeur
    mediane depasse toujours le drift median a 1-3 semaines. Le mecanisme n'est
    pas ameliorable en abaissant le niveau du PI -- ce serait un balayage
    post-hoc, et cela ne changerait pas le fait que la comparaison est binaire.

    Ce que la famille voulait mesurer est un RATIO drift / incertitude. On le
    mesure donc directement :

        position si |y_pred_t - last_close_t| > k x (y_upper_t - y_lower_t)
        signe donne par y_pred_t - last_close_t

    `k` est le seul parametre, et ses deux valeurs sont declarees par le brief
    (0,25 et 0,5), avant tout calcul. Aucune autre valeur n'est evaluee.
    L'echelle de largeur est celle du PI publie par le bras -- la meme des deux
    cotes, sans quoi la comparaison mesurerait la convention et pas le signal.
    """
    last_close = np.asarray(last_close, dtype=float)
    drift = np.asarray(y_pred, dtype=float) - last_close
    width = np.asarray(y_upper, dtype=float) - np.asarray(y_lower, dtype=float)
    return np.where(np.abs(drift) > k * width, np.sign(drift) * w_max, 0.0)


# Les deux valeurs de k du brief, et elles seules -- enregistrees comme deux
# strategies distinctes pour que chacune porte sa propre entree de famille Holm.
NORMALISED_DIRECTION_K = (0.25, 0.5)

STRATEGIES = {
    "inverse_width": {
        "label": "taille par inverse de la largeur du PI",
        "needs": ("last_close", "y_pred", "y_lower", "y_upper"),
        "fn": lambda d: positions_inverse_width(d["last_close"], d["y_pred"], d["y_lower"], d["y_upper"]),
    },
    "var_limit": {
        "label": "limite de risque : sleeve long dimensionne a VaR 2.5 % = budget",
        "needs": ("last_close", "y_lower"),
        "fn": lambda d: positions_var_limit(d["last_close"], d["y_lower"]),
    },
    "filtered_direction": {
        "label": "signal directionnel pris seulement si le PI exclut le rendement nul",
        "needs": ("last_close", "y_lower", "y_upper"),
        "fn": lambda d: positions_filtered_direction(d["last_close"], d["y_lower"], d["y_upper"]),
    },
    **{
        f"normalised_direction_k{str(k).replace('.', '')}": {
            "label": f"signal directionnel normalise : position si |drift| > {k} x largeur du PI",
            "needs": ("last_close", "y_pred", "y_lower", "y_upper"),
            "fn": (lambda k_: lambda d: positions_normalised_direction(
                d["last_close"], d["y_pred"], d["y_lower"], d["y_upper"], k_))(k),
        }
        for k in NORMALISED_DIRECTION_K
    },
}

# Reference de contexte, pas une famille du brief : acheter et garder. Sert a
# repondre a "cette strategie vaut-elle mieux que ne rien faire ?", question
# qu'aucun test modele-contre-modele ne pose.
def positions_buy_and_hold(last_close) -> np.ndarray:
    return np.full(np.asarray(last_close, dtype=float).size, W_MAX)


# ── 2. PnL et metriques ─────────────────────────────────────────────────────

def gross_returns(last_close, y_true) -> np.ndarray:
    return np.asarray(y_true, dtype=float) / np.asarray(last_close, dtype=float) - 1.0


def sleeve_pnl(positions, returns, cost_bps: float) -> np.ndarray:
    """PnL par origine, net de couts. Chaque origine ouvre un sleeve de taille
    |w_t| et le referme a l'echeance -> un ALLER-RETOUR, donc 2 x le cout
    unidirectionnel, applique au notionnel echange."""
    w = np.asarray(positions, dtype=float)
    r = np.asarray(returns, dtype=float)
    if w.shape != r.shape:
        raise ValueError(f"positions/returns non apparies : {w.shape} vs {r.shape}")
    return w * r - 2.0 * (cost_bps * 1e-4) * np.abs(w)


def max_drawdown(pnl) -> float:
    """Drawdown maximal de la courbe cumulee des PnL par origine. PROXY declare :
    a l'horizon h > 1 les sleeves se chevauchent, donc cette courbe n'est pas
    l'equity d'un compte reel -- c'est la sequence des resultats, lue dans
    l'ordre. Comparable entre modeles (meme convention des deux cotes), pas
    interpretable comme un drawdown de compte."""
    equity = np.cumsum(np.asarray(pnl, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    return float(np.min(equity - peak))


def sharpe(pnl, horizon_weeks: int) -> float:
    """Sharpe annualise. Le facteur est sqrt(52/h) et non sqrt(52) : a l'horizon
    h, chaque observation couvre h semaines."""
    x = np.asarray(pnl, dtype=float)
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(WEEKS_PER_YEAR / float(horizon_weeks)))


def var_diagnostics(returns, var, level: float = VAR_LEVEL) -> dict:
    """Violations de VaR realisees et cout des depassements. Le test formel est
    Kupiec (`calibration_tests.kupiec_lr_uc`, importe par l'appelant) ; ici on
    fournit les compteurs et le cout economique, que Kupiec ne donne pas."""
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    breach = r < v
    excess = np.where(breach, v - r, 0.0)          # >= 0, perte au-dela de la VaR
    n = r.size
    return {
        "n": int(n),
        "n_breaches": int(breach.sum()),
        "breach_rate": float(breach.mean()) if n else float("nan"),
        "expected_rate": float(level),
        "mean_var": float(v.mean()) if n else float("nan"),
        "total_excess_loss": float(excess.sum()),
        "mean_excess_loss_given_breach": float(excess[breach].mean()) if breach.any() else 0.0,
        "worst_excess_loss": float(excess.max()) if n else float("nan"),
        "breach_flags": breach.astype(int).tolist(),
    }


def evaluate(positions, returns, cost_bps: float, horizon_weeks: int) -> dict:
    """Toutes les metriques economiques d'un bras, sur une cellule."""
    pnl = sleeve_pnl(positions, returns, cost_bps)
    w = np.asarray(positions, dtype=float)
    return {
        "n": int(pnl.size),
        "pnl_total": float(pnl.sum()),
        "pnl_mean_per_origin": float(pnl.mean()) if pnl.size else float("nan"),
        "sharpe_annualised": sharpe(pnl, horizon_weeks),
        "max_drawdown": max_drawdown(pnl),
        "turnover_mean_abs_position": float(np.abs(w).mean()) if w.size else float("nan"),
        "n_active_origins": int((np.abs(w) > 0).sum()),
        "hit_rate_when_active": (float((pnl[np.abs(w) > 0] > 0).mean())
                                 if (np.abs(w) > 0).any() else float("nan")),
        "pnl_series": pnl.tolist(),
    }


def run_strategy(strategy: str, data: dict, cost_bps: float, horizon_weeks: int) -> dict:
    """Applique une famille declaree a une cellule et l'evalue. `data` doit
    porter last_close / y_pred / y_lower / y_upper / y_true, apparies et en
    ordre chronologique d'origine."""
    if strategy not in STRATEGIES:
        raise KeyError(f"strategie inconnue : {strategy!r} (connues : {sorted(STRATEGIES)})")
    positions = STRATEGIES[strategy]["fn"](data)
    r = gross_returns(data["last_close"], data["y_true"])
    out = evaluate(positions, r, cost_bps, horizon_weeks)
    out["positions"] = np.asarray(positions, dtype=float).tolist()
    return out
