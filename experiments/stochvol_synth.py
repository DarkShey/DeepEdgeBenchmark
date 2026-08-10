"""
stochvol_synth.py -- chantier D3 du BRIEF extension/puissance : le generateur de
series synthetiques a VOLATILITE STOCHASTIQUE, remplacant de KernelSynth pour
l'augmentation des regimes pauvres en donnees.

POURQUOI IL EXISTE -- un defaut nomme, pas une intuition. L'augmentation par
KernelSynth (`kernelsynth.py`) a ete testee au chantier C precedent et a
DEGRADE le pilote mensuel : significativement pire que GARCH-monthly a M+2 et
M+3, sous-couverture partout, plus mauvais RMSE du lot. Le diagnostic etait
structurel et avait ete declare dans le module AVANT le run : une serie tiree
d'un processus gaussien a covariance FIXE est homoscedastique par construction.
Entrainer un modele d'INCERTITUDE sur des fenetres homoscedastiques lui apprend
exactement la mauvaise chose -- que la volatilite ne change pas.

Ce generateur-ci corrige ce point -- et, en le mesurant, en a revele un autre,
plus grave encore, que la section suivante detaille.

LE MODELE : volatilite stochastique log-normale (Taylor 1986), la formulation la
plus simple qui produise du clustering de volatilite sans rien postuler d'autre.

    h_t = mu_h + phi * (h_{t-1} - mu_h) + sigma_eta * eta_t      (log-variance, AR(1))
    r_t = exp(h_t / 2) * eps_t                                    (rendement)

avec (eta_t, eps_t) gaussiens correles par `rho` -- l'effet de levier, negatif
pour un actif financier : une baisse s'accompagne d'une hausse de volatilite.
`eps_t` peut suivre une loi de Student, ce qui epaissit les queues AU-DELA de ce
que le clustering produit deja.

CE QUE LE DIAGNOSTIC A REVELE EN CHEMIN, et qui corrige la note precedente.
Le defaut de KernelSynth avait ete decrit comme « homoscedasticite ». En le
mesurant contre les VRAIES series mensuelles du panel (7 actifs), on trouve pire
et plus net -- son defaut decisif est ailleurs :

                          ACF(r) niveau   ACF(r^2)   exces de kurtosis
    reel mensuel, moyenne      +0.008       +0.181         +1.9
    KernelSynth                +0.451       +0.289         -0.31
    ce module (calibre)        -0.007       +0.178         +6.1

KernelSynth produit des series dont les NIVEAUX sont autocorreles a +0,45, la ou
un rendement reel est blanc (+0,01). Comme ces series sont consommees TELLES
QUELLES comme rendements standardises, on apprenait au modele qu'un rendement se
predit par le precedent -- ce qui est faux, et bien plus dommageable que
l'homoscedasticite. Son ACF(r^2) apparemment correct (+0,289) n'est d'ailleurs
PAS du clustering de volatilite : c'est un artefact mecanique de la regularite
des niveaux, un processus gaussien a covariance fixe ayant par construction une
variance conditionnelle deterministe.

PARAMETRES TIRES PAR SERIE, dans des plages DECLAREES et CALIBREES sur les faits
stylises mesures ci-dessus (calibration faite AVANT tout run de D3, sur les
series reelles, jamais sur un resultat de test) :

    phi        U(0.94, 0.995)   persistance de la volatilite. Cale pour
                                reproduire l'ACF(r^2) reel de 0,18 ; les series
                                financieres se situent effectivement vers
                                0.95-0.98.
    sigma_eta  U(0.25, 0.55)    volatilite de la volatilite, calee de meme.
    rho        U(-0.60, 0.00)   levier. Borne a 0 : un levier POSITIF n'a pas de
                                sens pour un actif au comptant, et l'autoriser
                                apprendrait au modele une asymetrie inversee.
    nu         {inf, 12, 8}     degres de liberte des innovations de rendement.
                                `inf` = gaussien : la plage inclut le cas sans
                                queues lourdes ajoutees, pour ne pas forcer un
                                stylise que toutes les series n'ont pas. Les
                                valeurs tres basses (4, 6) ont ete ecartees : la
                                volatilite stochastique produit DEJA des queues
                                epaisses, les cumuler menait a un exces de
                                kurtosis de 9-10 contre ~2 dans les donnees.

RESERVE SUR LA KURTOSIS : a +6,1 le module reste au-dessus de la moyenne reelle
(+1,9), dans la fourchette des actifs individuels (USO : +11,1). C'est un
depassement DECLARE et volontairement non corrige davantage : pour un modele
d'INCERTITUDE qui sur-couvrait, s'entrainer sur des queues legerement trop
epaisses pousse du cote conservateur, pas du cote dangereux.

CE QUE CE MODULE NE FAIT PAS, comme `kernelsynth` : il ne connait ni les prix, ni
les actifs, ni NsDiff. Il produit des series standardisees. La decision de les
melanger a des donnees reelles, et dans quelle proportion, appartient a
l'appelant et doit y etre declaree.

LIMITE A CITER, symetrique de celle de KernelSynth : une serie a volatilite
stochastique log-normale a du clustering, des queues et du levier -- elle n'a ni
sauts, ni changements de regime discrets, ni saisonnalite. Elle est plus proche
d'un actif financier que ne l'etait KernelSynth ; elle n'en est pas un.
"""

import numpy as np

# Plages declarees, tirees uniformement par serie.
PHI_RANGE = (0.94, 0.995)
SIGMA_ETA_RANGE = (0.25, 0.55)
RHO_RANGE = (-0.60, 0.0)
NU_CHOICES = (np.inf, 12.0, 8.0)
BURN_IN = 200               # pas jetes, le temps que l'AR(1) atteigne sa loi stationnaire


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "phi": float(rng.uniform(*PHI_RANGE)),
        "sigma_eta": float(rng.uniform(*SIGMA_ETA_RANGE)),
        "rho": float(rng.uniform(*RHO_RANGE)),
        "nu": float(rng.choice(NU_CHOICES)),
    }


def simulate(length: int, rng: np.random.Generator, phi: float, sigma_eta: float,
             rho: float, nu: float, burn_in: int = BURN_IN) -> np.ndarray:
    """Une trajectoire de rendements a volatilite stochastique, standardisee.

    `h` est initialise a SA LOI STATIONNAIRE (variance sigma_eta^2/(1-phi^2))
    puis brule `burn_in` pas : demarrer a h=0 produirait une periode initiale de
    volatilite artificiellement calme, que le modele apprendrait comme un motif.
    """
    n = length + burn_in
    h = np.empty(n)
    h[0] = rng.normal(0.0, sigma_eta / np.sqrt(max(1.0 - phi ** 2, 1e-6)))

    # levier : eps correle a eta. On tire eta d'abord, puis eps conditionnellement.
    eta = rng.standard_normal(n)
    z = rng.standard_normal(n)
    eps = rho * eta + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z
    if np.isfinite(nu):
        # Student-t standardisee (variance 1) : on divise par sqrt(chi2_nu / nu),
        # puis on renormalise par sqrt(nu/(nu-2)) pour garder une variance unitaire
        # -- sans quoi la loi de rendement porterait deux echelles a la fois.
        chi = rng.chisquare(nu, size=n) / nu
        eps = eps / np.sqrt(chi) / np.sqrt(nu / (nu - 2.0))

    for t in range(1, n):
        h[t] = phi * h[t - 1] + sigma_eta * eta[t]

    r = np.exp(h / 2.0) * eps
    r = r[burn_in:]
    sd = r.std()
    return (r - r.mean()) / (sd if sd > 1e-12 else 1.0)


def sample_series(length: int, rng: np.random.Generator) -> tuple:
    p = sample_params(rng)
    return simulate(length, rng, **p), p


def generate(n_series: int, length: int, seed: int = 0) -> tuple:
    """`n_series` series standardisees de longueur `length`, et leurs parametres."""
    rng = np.random.default_rng(seed)
    out, params = [], []
    for _ in range(n_series):
        s, p = sample_series(length, rng)
        out.append(s)
        params.append(p)
    return np.stack(out), params


# ── diagnostic : le generateur produit-il bien ce pour quoi il existe ? ──────

# Faits stylises MESURES sur les 7 series mensuelles reelles du panel etendu
# (`prices_v3/`), qui servent de cible de calibration.
REAL_STYLIZED_FACTS = {"acf_level": 0.008, "acf_squared": 0.181, "excess_kurtosis": 1.9,
                       "source": "7 actifs, rendements log mensuels standardises, prices_v3"}


def level_autocorrelation(series: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation des rendements EN NIVEAU. C'est LE discriminant : un
    rendement reel est quasi blanc (+0,01), KernelSynth est a +0,45. Une serie
    synthetique qui echoue ici apprend au modele qu'un rendement se predit par le
    precedent -- defaut plus grave que l'homoscedasticite."""
    arr = np.atleast_2d(np.asarray(series, dtype=float))
    out = [float(np.corrcoef(s[:-lag], s[lag:])[0, 1]) for s in arr if s.std() > 1e-12]
    return float(np.mean(out)) if out else float("nan")


def volatility_clustering(series: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation des rendements AU CARRE. A NE LIRE QU'AVEC
    `level_autocorrelation` : sur une serie dont les niveaux sont autocorreles,
    cette statistique est gonflee mecaniquement et ne mesure plus la volatilite
    (KernelSynth : ACF(r^2) = +0,29 alors qu'il est homoscedastique)."""
    arr = np.atleast_2d(np.asarray(series, dtype=float))
    out = []
    for s in arr:
        x = s ** 2
        if x.std() < 1e-12:
            continue
        out.append(float(np.corrcoef(x[:-lag], x[lag:])[0, 1]))
    return float(np.mean(out)) if out else float("nan")


def excess_kurtosis(series: np.ndarray) -> float:
    arr = np.atleast_2d(np.asarray(series, dtype=float))
    k = [float(((s - s.mean()) ** 4).mean() / max((s.var()) ** 2, 1e-12) - 3.0) for s in arr]
    return float(np.mean(k))
