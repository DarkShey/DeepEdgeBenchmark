"""
nsdiff_scenarios.py -- chantier B3 : le cas d'usage DIFFERENCIANT.

La question. ARIMA-GARCH publie trois intervalles marginaux (W+1, W+2, W+3).
NsDiff publie un NUAGE DE TRAJECTOIRES. Si la diffusion a une valeur propre,
elle doit apparaitre sur un usage qui consomme la trajectoire entiere -- pas
seulement deux quantiles par horizon. Si elle ne se differencie pas la non plus,
la conclusion du programme est complete.

CE QUI EST TESTE, trois quantites qu'un couple de quantiles marginaux ne donne
pas :

  1. LE MINIMUM DE PARCOURS sur 3 semaines -- min_h (P_{t+h}/P_t - 1). C'est le
     drawdown intra-fenetre, la quantite que regarde un gerant qui doit tenir une
     limite de perte entre deux revisions. Mesuree par CRPS empirique
     (`crps_metrics.crps_empirical`, importe) et par la couverture de son
     quantile a 5 %.
  2. LE PRIX D'UN PUT ATM a 3 semaines -- E[max(K - S_3, 0)]/K avec K = P_t.
     Consomme toute la queue gauche, pas un seul quantile. Evalue par le PnL de
     la vente systematique au prix du modele : prix encaisse moins payoff realise.
     Taux sans risque = 0, declare (les 3 semaines d'actualisation sont sous la
     resolution de l'exercice et introduire une courbe ajouterait un parametre
     non teste).
  3. UN DIGITAL A BARRIERE BASSE -- probabilite que le parcours touche
     0.95 x P_t a un moment quelconque des 3 semaines. PUREMENT dependant du
     chemin : aucune combinaison de quantiles marginaux ne la donne. Evaluee par
     score de Brier.

COMMENT LE BRAS GARCH EST CONSTRUIT -- le point delicat, a ne pas transformer en
homme de paille. On ne prete a GARCH aucune faiblesse qu'il n'a pas : on
reconstruit le generateur de scenarios que ses PROPRES sorties definissent.
Verifie empiriquement avant d'ecrire une ligne :
  * `log(y_lower) + log(y_upper) = 2 log(y_pred)` a la precision machine sur
    100 % des lignes -> l'intervalle est exactement symetrique en log autour du
    point, donc (mu_h, sigma_h) se recuperent SANS approximation :
        mu_h = log(y_pred_h / P_t),  sigma_h = (log y_upper_h - log y_lower_h) / (2 z)
  * sigma_h croit strictement avec h sur 100 % des lignes -> les variances
    d'increments v_h = sigma_h^2 - sigma_{h-1}^2 sont toutes positives, la
    reconstruction est toujours definie.
Les trajectoires GARCH sont alors des marches aleatoires gaussiennes a
increments independants calees sur (mu_h, sigma_h) : c'est litteralement ce
qu'un praticien construirait a partir de ce que GARCH publie, et cela reproduit
EXACTEMENT ses trois intervalles marginaux. La difference entre les deux bras
est donc, et seulement, la STRUCTURE de la loi jointe.

Ce que cette construction concede a GARCH et qu'il faut citer : ses increments
sont gaussiens et independants par construction. S'il avait une opinion sur
l'asymetrie ou le clustering intra-fenetre, elle n'apparait pas dans les trois
intervalles publies -- l'ecart mesure porte donc sur "ce que le format de sortie
de GARCH permet", pas sur "ce que le modele GARCH sait". C'est la comparaison
pertinente pour une decision de production, ou l'on consomme des sorties.

BUDGET D'ECHANTILLONNAGE STRICTEMENT EGAL : 1000 trajectoires des deux cotes
(NsDiff = 5 graines x 200 concatenees, la spec production ; GARCH = 1000 tirages
de sa loi reconstruite, graine fixee).

Sortie : experiments/nsdiff_scenarios.json
Usage   : python nsdiff_scenarios.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matrice_paired_tests as mpt                                    # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from crps_metrics import crps_empirical                               # noqa: E402
from nsdiff_vs_garch_w23 import load_challenger                       # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_scenarios.json"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
HORIZONS = ["W+1", "W+2", "W+3"]
Z975 = 1.959963984540054
N_PATHS = 1000
PATH_SEED = 42
POOL_SEED = 42
BLOCK_LENGTH = 3
BARRIER = 0.95            # digital : le parcours touche-t-il -5 % ? declare a priori
STRESS_Q = 0.05           # quantile de stress du minimum de parcours
BOND_ASSETS = {"ZN=F", "TLT"}


# ── 1. trajectoires ─────────────────────────────────────────────────────────

def nsdiff_paths(rows: pd.DataFrame, samples: np.ndarray, asset: str, regime: str) -> dict:
    """{cutoff_date: (P_t, chemin [n_paths, 3])}. Les 3 horizons d'une meme
    (graine, origine) partagent l'index de tirage -- ils sortent d'UN SEUL appel
    a `sample_paths`, dont `forecast_from_fitted` lit les sommes cumulees.
    Verifie : la correlation entre tirage i a W+1 et a W+2 vaut ~0.76, ce qu'un
    tirage independant ne produirait pas. Les graines sont CONCATENEES sur l'axe
    des trajectoires (spec production) -- jamais melangees entre elles a
    l'interieur d'un chemin, ce qui casserait la structure temporelle."""
    sub = rows[(rows["asset"] == asset) & (rows["frequence"] == regime)]
    out = {}
    for cutoff, g in sub.groupby("cutoff_date"):
        per_seed = []
        for _, gs in g.groupby("seed"):
            gs = gs.set_index("horizon_unit")
            idx = [gs.loc[h, "_row"] for h in HORIZONS]
            per_seed.append(samples[idx].T)                     # [n_samples, 3]
        paths = np.concatenate(per_seed, axis=0)
        out[cutoff] = (float(g["last_close"].iloc[0]), paths)
    return out


def garch_paths(rows: pd.DataFrame, asset: str, regime: str, n_paths: int = N_PATHS,
                seed: int = PATH_SEED) -> dict:
    """Marches aleatoires gaussiennes calees sur les trois intervalles publies.
    Reproduit exactement les quantiles marginaux de GARCH -- seule la structure
    jointe est ajoutee, parce qu'il en faut une pour parler de trajectoire."""
    sub = rows[(rows["asset"] == asset) & (rows["frequence"] == regime)]
    rng = np.random.default_rng(seed)
    out = {}
    for cutoff, g in sub.groupby("cutoff_date"):
        g = g.set_index("horizon_unit")
        p0 = float(g["last_close"].iloc[0])
        mu = np.array([np.log(float(g.loc[h, "y_pred"]) / p0) for h in HORIZONS])
        sig = np.array([(np.log(float(g.loc[h, "y_upper"])) - np.log(float(g.loc[h, "y_lower"])))
                        / (2.0 * Z975) for h in HORIZONS])
        var_incr = np.diff(np.concatenate([[0.0], sig ** 2]))
        if np.any(var_incr <= 0):
            raise SystemExit(f"[{asset}/{regime}/{cutoff}] sigma non croissant -- "
                             "reconstruction de trajectoire impossible")
        # rendement log cumule a h = mu_h + bruit, dont la variance cumulee vaut
        # sigma_h^2 par construction -> les trois marginales sont reproduites exactement
        incr = rng.normal(0.0, np.sqrt(var_incr), size=(n_paths, 3))
        out[cutoff] = (p0, p0 * np.exp(np.cumsum(incr, axis=1) + mu))
    return out


# ── 2. fonctionnelles de trajectoire ────────────────────────────────────────

def path_min_return(paths: np.ndarray, p0: float) -> np.ndarray:
    """min_h (P_h / P_0 - 1) : le drawdown intra-fenetre de chaque trajectoire."""
    return np.min(paths, axis=1) / p0 - 1.0


def put_price(paths: np.ndarray, p0: float) -> float:
    """Put ATM a l'echeance (3 semaines), en fraction du spot. Taux nul declare."""
    return float(np.mean(np.maximum(p0 - paths[:, -1], 0.0)) / p0)


def barrier_prob(paths: np.ndarray, p0: float, barrier: float = BARRIER) -> float:
    """P(le parcours touche `barrier` x P_0 a un moment des 3 semaines).
    Purement dependant du chemin."""
    return float(np.mean(np.min(paths, axis=1) <= barrier * p0))


def realised(actuals: np.ndarray, p0: float) -> dict:
    return {
        "path_min": float(np.min(actuals) / p0 - 1.0),
        "put_payoff": float(max(p0 - actuals[-1], 0.0) / p0),
        "barrier_hit": float(np.min(actuals) <= BARRIER * p0),
    }


# ── 3. evaluation ───────────────────────────────────────────────────────────

def evaluate_cell(model_paths: dict, actual_by_cutoff: dict) -> dict:
    """Une cellule = (actif, regime). Renvoie les series par origine des pertes
    a MINIMISER, plus les diagnostics."""
    cutoffs = sorted(set(model_paths) & set(actual_by_cutoff))
    crps, breach, put_pnl, brier, prices, probs = [], [], [], [], [], []
    for c in cutoffs:
        p0, paths = model_paths[c]
        act = actual_by_cutoff[c]
        real = realised(act, p0)

        pm = path_min_return(paths, p0)
        crps.append(crps_empirical(pm, real["path_min"]))
        breach.append(float(real["path_min"] < np.quantile(pm, STRESS_Q)))

        price = put_price(paths, p0)
        prices.append(price)
        put_pnl.append(price - real["put_payoff"])          # PnL du VENDEUR du put

        prob = barrier_prob(paths, p0)
        probs.append(prob)
        brier.append((prob - real["barrier_hit"]) ** 2)

    return {
        "n": len(cutoffs), "cutoffs": cutoffs,
        "path_min_crps": np.array(crps),
        "stress_breach": np.array(breach),
        "put_seller_pnl": np.array(put_pnl),
        "put_price_mean": float(np.mean(prices)),
        "barrier_brier": np.array(brier),
        "barrier_prob_mean": float(np.mean(probs)),
    }


def paired(diffs, label_a: str, label_b: str, lower_is_better: bool = True) -> dict:
    d = np.asarray(diffs, dtype=float)
    if d.size < 5:
        return {"status": "insufficient_data", "n": int(d.size)}
    t = paired_block_bootstrap_test(d, block_length=min(BLOCK_LENGTH, d.size), seed=POOL_SEED)
    if not t["significant_at_05"]:
        verdict = "indistinguishable"
    else:
        a_wins = (t["mean_diff"] < 0) if lower_is_better else (t["mean_diff"] > 0)
        verdict = f"{label_a}_significantly_better" if a_wins else f"{label_b}_significantly_better"
    return {"status": "tested", "verdict": verdict, **t}


def pool_series(by_asset: dict, key: str) -> np.ndarray:
    """Moyenne cross-sectionnelle par origine, avec dedoublonnage des deux actifs
    de taux (meme convention que partout dans le programme)."""
    lengths = {len(v[key]) for v in by_asset.values()}
    if len(lengths) != 1:
        return None
    bonds = [by_asset[a][key] for a in by_asset if a in BOND_ASSETS]
    parts = [by_asset[a][key] for a in by_asset if a not in BOND_ASSETS]
    if bonds:
        parts.append(np.mean(bonds, axis=0))
    return np.mean(parts, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir), model="NsDiff")
    rows = rows.reset_index(drop=True)
    rows["_row"] = rows.index
    assets = v2.assets(rows)
    garch = load_challenger(assets, HORIZONS)
    n_paths_nsdiff = rows["seed"].nunique() * samples.shape[1]
    print(f"trajectoires : NsDiff {n_paths_nsdiff} (5 graines x 200, concatenees) | "
          f"ARIMA-GARCH {N_PATHS} (loi reconstruite sur ses propres intervalles)")
    if n_paths_nsdiff != N_PATHS:
        raise SystemExit(f"budget d'echantillonnage inegal : {n_paths_nsdiff} vs {N_PATHS}")

    results, per_asset = {}, {}
    for regime in ("weekly", "daily"):
        per_asset[regime] = {}
        for asset in assets:
            sub = rows[(rows["asset"] == asset) & (rows["frequence"] == regime)]
            actual = {c: np.array([float(g[g["horizon_unit"] == h]["y_true"].iloc[0]) for h in HORIZONS])
                      for c, g in sub[sub["seed"] == sub["seed"].min()].groupby("cutoff_date")}
            ns = evaluate_cell(nsdiff_paths(rows, samples, asset, regime), actual)
            gc = evaluate_cell(garch_paths(garch, asset, regime), actual)
            if ns["cutoffs"] != gc["cutoffs"]:
                raise SystemExit(f"[{asset}/{regime}] origines desalignees entre bras")
            per_asset[regime][asset] = {"nsdiff": ns, "garch": gc}

            results[f"{asset}|{regime}"] = {
                "n": ns["n"],
                "path_min_crps": {"nsdiff": float(ns["path_min_crps"].mean()),
                                  "garch": float(gc["path_min_crps"].mean()),
                                  "test": paired(ns["path_min_crps"] - gc["path_min_crps"],
                                                 "nsdiff", "garch")},
                "stress_q05_breach_rate": {"nsdiff": float(ns["stress_breach"].mean()),
                                           "garch": float(gc["stress_breach"].mean()),
                                           "target": STRESS_Q},
                "put_seller_pnl": {"nsdiff_total": float(ns["put_seller_pnl"].sum()),
                                   "garch_total": float(gc["put_seller_pnl"].sum()),
                                   "nsdiff_price_mean": ns["put_price_mean"],
                                   "garch_price_mean": gc["put_price_mean"],
                                   "test": paired(ns["put_seller_pnl"] - gc["put_seller_pnl"],
                                                  "nsdiff", "garch", lower_is_better=False)},
                "barrier_brier": {"nsdiff": float(ns["barrier_brier"].mean()),
                                  "garch": float(gc["barrier_brier"].mean()),
                                  "nsdiff_prob_mean": ns["barrier_prob_mean"],
                                  "garch_prob_mean": gc["barrier_prob_mean"],
                                  "test": paired(ns["barrier_brier"] - gc["barrier_brier"],
                                                 "nsdiff", "garch")},
            }

    pooled = {}
    for regime, by_asset in per_asset.items():
        ns_all = {a: v["nsdiff"] for a, v in by_asset.items()}
        gc_all = {a: v["garch"] for a, v in by_asset.items()}
        entry = {}
        for key, lower_better in (("path_min_crps", True), ("put_seller_pnl", False),
                                  ("barrier_brier", True)):
            a, b = pool_series(ns_all, key), pool_series(gc_all, key)
            if a is None or b is None:
                entry[key] = {"status": "not_poolable"}
                continue
            entry[key] = {"nsdiff_mean": float(a.mean()), "garch_mean": float(b.mean()),
                          "test": paired(a - b, "nsdiff", "garch", lower_is_better=lower_better)}
        pooled[regime] = entry

    fam = {f"{regime}|{key}": entry[key]["test"] for regime, entry in pooled.items()
           for key in entry if entry[key].get("status") != "not_poolable"}
    corrected = mt.correct_family(fam)
    holm = {"family": corrected["family"], "summary": mt.family_summary(corrected)}

    payload = {
        "question": "la diffusion se differencie-t-elle sur un usage qui consomme la "
                    "TRAJECTOIRE entiere, que les quantiles marginaux de GARCH ne donnent pas ?",
        "config": {
            "n_paths_each_side": N_PATHS,
            "nsdiff_paths": "5 graines x 200 tirages concatenees (spec production) ; les 3 "
                            "horizons d'une (graine, origine) partagent l'index de tirage -- "
                            "un seul appel a sample_paths, correlation ~0.76 entre W+1 et W+2",
            "garch_paths": "marche aleatoire gaussienne a increments independants calee sur "
                           "(mu_h, sigma_h) recuperes EXACTEMENT de ses trois intervalles publies "
                           "(log-symetrie verifiee a la precision machine sur 100 % des lignes, "
                           "sigma strictement croissant sur 100 %)",
            "garch_concession": "ses increments sont gaussiens et independants par construction : "
                                "l'ecart mesure porte sur ce que le FORMAT DE SORTIE de GARCH "
                                "permet, pas sur ce que le modele GARCH sait. C'est la "
                                "comparaison pertinente pour une decision de production.",
            "functionals": {
                "path_min": "min_h (P_h/P_0 - 1) sur 3 semaines -- drawdown intra-fenetre, "
                            f"evalue par CRPS empirique + couverture du quantile a {STRESS_Q}",
                "put_atm_3w": "E[max(K - S_3, 0)]/K, K = P_t, taux nul declare -- evalue par le "
                              "PnL du vendeur systematique (prix encaisse - payoff realise)",
                "barrier_digital": f"P(le parcours touche {BARRIER} x P_0 en 3 semaines) -- "
                                   "purement dependant du chemin, evalue par score de Brier",
            },
            "multiple_testing": "Holm sur la famille des 6 tests pooles (2 regimes x 3 "
                                "fonctionnelles)",
        },
        "per_cell": results, "pooled": pooled, "holm": holm,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n{'cellule':<20}{'CRPS min-parcours ns/garch':>32}{'verdict':>34}"
          f"{'Brier barriere ns/garch':>28}{'verdict':>34}")
    for key, r in results.items():
        c, b = r["path_min_crps"], r["barrier_brier"]
        crps_txt = "{:.4g} / {:.4g}".format(c["nsdiff"], c["garch"])
        brier_txt = "{:.4f} / {:.4f}".format(b["nsdiff"], b["garch"])
        print(f"{key:<20}{crps_txt:>32}{c['test']['verdict']:>34}"
              f"{brier_txt:>28}{b['test']['verdict']:>34}")

    print("\n=== pooles tous actifs ===")
    for regime, entry in pooled.items():
        for key, e in entry.items():
            if e.get("status") == "not_poolable":
                continue
            print(f"  {regime:<8}{key:<18} ns={e['nsdiff_mean']:+.5g}  garch={e['garch_mean']:+.5g}  "
                  f"{e['test']['verdict']} (p={e['test']['p_value']:.4f})")
    s = holm["summary"]
    print(f"\n[Holm] m={s['m']}, {s['n_significant_raw']} rejets bruts -> {s['n_significant_holm']} "
          f"apres correction" + (f" | survivants : {', '.join(s['survivors'])}" if s["survivors"] else ""))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
