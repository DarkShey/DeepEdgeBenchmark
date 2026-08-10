"""
calibration_tests.py -- tests FORMELS de couverture et d'equivalence, absents
du repo jusqu'ici (BRIEF "NsDiff : consolider le verdict daily vs weekly",
taches 1 et 3).

Trois briques, et rien d'autre (aucune n'existait ; tout ce qui existait deja
-- bootstrap par blocs, Winkler, skill-score, conformal -- est importe, jamais
recopie) :

  1. `kupiec_lr_uc`          -- couverture inconditionnelle (Kupiec 1995 POF).
  2. `christoffersen_lr_ind` -- independance des violations (Christoffersen
     1998), et `christoffersen_lr_cc` = uc + ind, le test conjoint.
  3. `tost_relative_rmse`    -- TOST (two one-sided tests) sur le RMSE
     relatif, seule facon de conclure POSITIVEMENT a l'equivalence : un test
     non significatif ne prouve pas l'absence de difference, surtout a
     `effective_n` ~ 30 (cf. la puissance modeste declaree partout dans ce
     chantier).

CAVEAT commun aux tests 1 et 2, a citer partout ou leurs p-values sont
rapportees : la theorie asymptotique chi2 de Kupiec/Christoffersen suppose des
violations i.i.d. dans le temps. Ici les origines sont hebdomadaires et les
cibles W+2/W+3 se chevauchent d'une origine a l'autre -- exactement la raison
pour laquelle le reste du repo utilise un bootstrap PAR BLOCS. Les p-values
chi2 sont donc OPTIMISTES (variance sous-estimee) et doivent etre lues comme
un complement de manuel au test de reference du chantier -- le bootstrap par
blocs sur l'ecart de couverture (`paired_test.paired_block_bootstrap_test`,
importe ci-dessous, PAS reimplemente) -- et non comme le verdict principal.
`christoffersen_lr_ind` est en outre structurellement biaise vers le rejet a
W+2/W+3 : le chevauchement des cibles cree une dependance MECANIQUE entre
violations consecutives, qui n'est pas un defaut du modele.
"""

import numpy as np
from scipy import stats

from paired_test import paired_block_bootstrap_test

BLOCK_LENGTH = 3          # meme convention que matrice_paired_tests / dashboard_d7_w1


def _xlogx_terms(count: int, prob: float) -> float:
    """count * log(prob), avec la convention 0 * log(0) = 0 (le terme d'une
    categorie vide ne contribue pas a la log-vraisemblance)."""
    if count == 0:
        return 0.0
    if prob <= 0.0:
        return -np.inf
    return float(count) * float(np.log(prob))


def kupiec_lr_uc(hits, alpha_target: float = 0.05) -> dict:
    """Test POF de Kupiec (1995) : la frequence observee de violations est-elle
    compatible avec le taux nominal `alpha_target` ?

    `hits` : sequence 0/1 ou 1 = VIOLATION (y_true hors de l'intervalle).
    Attention au sens : ailleurs dans ce repo `in_interval` vaut 1 quand la
    valeur est DEDANS -- passer `1 - in_interval`.

    LR_uc = -2 [ ln L(alpha_target) - ln L(pi_hat) ] ~ chi2(1).
    """
    hits = np.asarray(hits, dtype=float)
    n = hits.size
    if n == 0:
        return {"status": "insufficient_data", "n": 0}
    n1 = int(hits.sum())
    n0 = n - n1
    pi_hat = n1 / n

    ll_null = _xlogx_terms(n0, 1.0 - alpha_target) + _xlogx_terms(n1, alpha_target)
    ll_alt = _xlogx_terms(n0, 1.0 - pi_hat) + _xlogx_terms(n1, pi_hat)
    lr = -2.0 * (ll_null - ll_alt)
    return {
        "status": "tested", "n": int(n), "n_violations": n1,
        "violation_rate": float(pi_hat), "alpha_target": float(alpha_target),
        "lr_uc": float(lr), "p_value": float(stats.chi2.sf(lr, df=1)),
        "significant_at_05": bool(stats.chi2.sf(lr, df=1) < 0.05),
    }


def christoffersen_lr_ind(hits) -> dict:
    """Test d'independance de Christoffersen (1998) : la probabilite d'une
    violation depend-elle de ce qui s'est passe a l'origine precedente
    (clustering) ? LR_ind ~ chi2(1), sur la chaine de Markov d'ordre 1 des
    transitions 0->0 / 0->1 / 1->0 / 1->1.

    `hits` DOIT etre en ordre chronologique. Non calculable si aucune
    violation n'est suivie d'une observation (pi_11 indefini) -- renvoie alors
    status='not_identified' plutot qu'un chiffre invente.
    """
    hits = np.asarray(hits, dtype=int)
    n = hits.size
    if n < 2:
        return {"status": "insufficient_data", "n": int(n)}

    prev, cur = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"status": "not_identified", "n": int(n),
                "counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
                "reason": "un des deux etats precedents n'est jamais observe -- "
                          "pi_01 ou pi_11 non identifie"}

    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = _xlogx_terms(n00 + n10, 1.0 - pi) + _xlogx_terms(n01 + n11, pi)
    ll_alt = (_xlogx_terms(n00, 1.0 - pi_01) + _xlogx_terms(n01, pi_01)
              + _xlogx_terms(n10, 1.0 - pi_11) + _xlogx_terms(n11, pi_11))
    lr = -2.0 * (ll_null - ll_alt)
    p = float(stats.chi2.sf(lr, df=1))
    return {
        "status": "tested", "n": int(n),
        "counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "pi_01": float(pi_01), "pi_11": float(pi_11),
        "lr_ind": float(lr), "p_value": p, "significant_at_05": bool(p < 0.05),
    }


def christoffersen_lr_cc(hits, alpha_target: float = 0.05) -> dict:
    """Test conjoint de couverture conditionnelle : LR_cc = LR_uc + LR_ind
    ~ chi2(2). Renvoie aussi ses deux composantes, pour pouvoir dire OU le
    rejet se produit (niveau de couverture vs clustering)."""
    uc = kupiec_lr_uc(hits, alpha_target)
    ind = christoffersen_lr_ind(hits)
    if uc["status"] != "tested" or ind["status"] != "tested":
        return {"status": ind.get("status", uc["status"]), "uc": uc, "ind": ind}
    lr = uc["lr_uc"] + ind["lr_ind"]
    p = float(stats.chi2.sf(lr, df=2))
    return {
        "status": "tested", "n": uc["n"], "lr_cc": float(lr), "p_value": p,
        "significant_at_05": bool(p < 0.05), "uc": uc, "ind": ind,
    }


def coverage_gap_block_test(in_interval, target: float = 0.95,
                            block_length: int = BLOCK_LENGTH, seed: int = 0) -> dict:
    """Test de REFERENCE de ce chantier pour la couverture : ecart
    (indicateur - cible) bootstrappe par blocs -- meme convention que
    `matrice_paired_tests.comparison_2_calibration` et
    `tsdiff_recalibrate.coverage_gap_test`. Contrairement a Kupiec/
    Christoffersen il ne suppose PAS l'independance des violations.

    `in_interval` : 1 = DEDANS (convention de la DB), ordre chronologique.
    Accepte des valeurs fractionnaires (ex. taux de couverture moyenne sur les
    graines a une origine donnee) -- le bootstrap ne teste que la moyenne.
    """
    x = np.asarray(in_interval, dtype=float)
    if x.size < 2:
        return {"status": "insufficient_data", "n": int(x.size)}
    test = paired_block_bootstrap_test(x - target, block_length=min(block_length, x.size), seed=seed)
    return {"status": "tested", "coverage": float(x.mean()), "target": float(target),
            "coverage_gap": test["mean_diff"], **{k: v for k, v in test.items() if k != "mean_diff"}}


def _one_sided_p(boot_means: np.ndarray, side: str) -> float:
    """p unilaterale par percentile bootstrap. side='less' -> H1: moyenne < 0
    (p = fraction des replicats >= 0) ; side='greater' -> H1: moyenne > 0."""
    if side == "less":
        return float(np.mean(boot_means >= 0.0))
    if side == "greater":
        return float(np.mean(boot_means <= 0.0))
    raise ValueError(f"side must be 'less' or 'greater', got {side!r}")


def tost_relative_rmse(sq_error_a, sq_error_b, margin_rel: float = 0.05,
                       block_length: int = BLOCK_LENGTH, seed: int = 0,
                       n_boot: int = 10000) -> dict:
    """TOST : les deux RMSE sont-ils EQUIVALENTS a +/- `margin_rel` pres ?

    Hypotheses (RMSE_a / RMSE_b, donc en MSE : delta_hi = (1+m)^2,
    delta_lo = (1-m)^2) :
        H0_hi : MSE_a >= delta_hi * MSE_b      (a est pire de plus de m)
        H0_lo : MSE_a <= delta_lo * MSE_b      (a est meilleur de plus de m)
    Equivalence conclue ssi les DEUX sont rejetees a 5% ; p_tost = max des
    deux p unilaterales (convention TOST standard).

    Chaque cote est teste sur la serie APPARIEE par origine
    `sq_a_i - delta * sq_b_i`, avec le meme bootstrap par blocs que partout
    ailleurs (`paired_test.paired_block_bootstrap_test`, appele avec
    `return_boot_means=True`, aucune reimplementation du reechantillonnage).
    Les deux vecteurs doivent etre apparies ET en ordre chronologique.

    Un resultat 'inconclusive' est un resultat : ni difference etablie, ni
    equivalence etablie -- c'est le cas attendu quand la puissance manque, et
    c'est precisement ce que le brief demande de ne plus confondre avec
    "indistinguable donc interchangeable".
    """
    a = np.asarray(sq_error_a, dtype=float)
    b = np.asarray(sq_error_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"sq_error_a/b must be paired and same length: {a.shape} vs {b.shape}")
    n = a.size
    if n < 2:
        return {"status": "insufficient_data", "n": int(n)}

    delta_hi = (1.0 + margin_rel) ** 2
    delta_lo = (1.0 - margin_rel) ** 2
    bl = min(block_length, n)

    # H0_hi rejetee si mean(a - delta_hi * b) < 0 significativement
    test_hi = paired_block_bootstrap_test(a - delta_hi * b, block_length=bl, seed=seed,
                                          n_boot=n_boot, return_boot_means=True)
    p_hi = _one_sided_p(test_hi["boot_means"], "less")
    # H0_lo rejetee si mean(a - delta_lo * b) > 0 significativement
    test_lo = paired_block_bootstrap_test(a - delta_lo * b, block_length=bl, seed=seed,
                                          n_boot=n_boot, return_boot_means=True)
    p_lo = _one_sided_p(test_lo["boot_means"], "greater")

    p_tost = max(p_hi, p_lo)
    rmse_a, rmse_b = float(np.sqrt(a.mean())), float(np.sqrt(b.mean()))
    return {
        "status": "tested", "n": int(n), "block_length": int(bl),
        "effective_n": int(n // bl), "margin_rel": float(margin_rel),
        "rmse_a": rmse_a, "rmse_b": rmse_b,
        "rmse_ratio": float(rmse_a / rmse_b) if rmse_b else float("nan"),
        "p_upper": p_hi, "p_lower": p_lo, "p_tost": float(p_tost),
        "equivalent_at_05": bool(p_tost < 0.05),
        "verdict": "equivalent" if p_tost < 0.05 else "inconclusive",
    }
