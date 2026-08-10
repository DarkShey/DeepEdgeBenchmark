"""
nsdiff_ensemble_vs_garch.py -- chantier A2 : la question de production reelle.

Tous les verdicts du programme comparent GARCH a "un run NsDiff a graine tiree
au hasard". Ce n'est pas ce qu'on deploierait : la configuration candidate
production est l'ENSEMBLE des 5 graines a 200 tirages (`nsdiff_production_spec`,
nuage de 1000), qui est mesuree meilleure que la graine unique attendue sur 7
cellules daily / 15 au Winkler, pour un cout de calcul nul. La question
"la config qu'on deploierait bat-elle GARCH ?" n'avait jamais ete posee.

Ce script la pose. Aucun refit : les nuages sont sur disque
(`diffusion_multiseed_v2/NsDiff/`), la meme machinerie de match que la tache 7
est reutilisee telle quelle (`diffusion_headtohead.run_match`), et le bras
GARCH est relu en lecture seule dans `tracking.db`.

CE QUI CHANGE PAR RAPPORT A LA TACHE 7, et rien d'autre : le bras A. La tache 7
opposait a GARCH les lignes NsDiff graine par graine + leur moyenne par origine
(la "graine tiree au hasard"). Ici le bras A est UNE seule configuration --
l'ensemble -- donc il n'y a pas de dimension graine a parcourir : `seeds=[]`.
Les p-values ne sont pas comparables une a une entre les deux scripts (bras A
different), le verdict l'est.

TESTS MULTIPLES, applique et non plus renvoye en note de bas de page (chantier
A3) : `multiple_testing.holm_bonferroni`. Familles declarees A PRIORI :

  * FAMILLE DE DECISION, une par metrique (RMSE, Winkler) : les 6 tests pooles
    GLOBAUX (3 horizons x 2 regimes). C'est sur eux que porte la conclusion du
    programme, et c'est la famille annoncee par `NOTE_duel_nsdiff_vs_tsdiff_
    budget_egal.md` §0.1 ;
  * famille etendue, rapportee a cote : les 24 tests pooles (4 groupes x 3
    horizons x 2 regimes) par metrique, pour montrer ce que la correction
    coute quand on compte aussi les classes d'actifs ;
  * les tests PAR CELLULE ne sont pas corriges -- exploratoires par
    construction, declares comme tels, aucune conclusion ne s'y appuie.

ASYMETRIE DE PROTOCOLE, a citer avec tout chiffre : ARIMA-GARCH est refit a
chaque origine et deterministe ; NsDiff est train-once-forward. L'ecart mesure
melange modele et protocole. C'est la reserve que porte tout classement
inter-modeles du repo -- et c'est precisement ce que `nsdiff_refit_cadence.py`
chiffre.

Sortie : experiments/nsdiff_ensemble_vs_garch.json
Usage   : python nsdiff_ensemble_vs_garch.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import diffusion_headtohead as h2h                                    # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import nsdiff_production_spec as spec                                 # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from nsdiff_seed_ensemble import build_ensemble_rows                  # noqa: E402
from nsdiff_vs_garch_w23 import load_challenger                       # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_ensemble_vs_garch.json"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
LABEL_A, LABEL_B = "NsDiff-ensemble", "ARIMA-GARCH"
GROUPS = ("global", "crypto", "index", "bond")
METRICS = ("skill_sqerror", "skill_winkler")


def collect_pooled(horizons: dict, groups=GROUPS) -> dict:
    """{metrique: {"regime|horizon|groupe": resultat_de_test}} -- extraction
    plate des tests pooles, seule forme sur laquelle Holm se raisonne."""
    out = {m: {} for m in METRICS}
    for hu, res in horizons.items():
        for regime, block in res["per_regime"].items():
            for group in groups:
                pooled = block["pooled_across_assets"].get(group, {})
                if pooled.get("status") != "tested":
                    continue
                for metric in METRICS:
                    out[metric][f"{regime}|{hu}|{group}"] = pooled[metric]
    return out


def apply_holm(pooled_by_metric: dict) -> dict:
    """Deux familles par metrique : decision (global seul) et etendue (tous
    groupes). Les deux sont rapportees -- montrer ce que la correction coute
    fait partie de la correction."""
    out = {}
    for metric, tests in pooled_by_metric.items():
        decision = {k: v for k, v in tests.items() if k.endswith("|global")}
        out[metric] = {
            "decision_family": {
                "definition": "les 6 tests pooles GLOBAUX (3 horizons x 2 regimes)",
                **mt.correct_family(decision),
                "summary": mt.family_summary(mt.correct_family(decision)),
            },
            "extended_family": {
                "definition": "les 24 tests pooles (4 groupes x 3 horizons x 2 regimes)",
                **mt.correct_family(tests),
                "summary": mt.family_summary(mt.correct_family(tests)),
            },
        }
    return out


def _print_holm(holm: dict) -> None:
    for metric, fams in holm.items():
        for fam_name, fam in fams.items():
            s = fam["summary"]
            print(f"\n[Holm] {metric} / {fam_name} -- m={s['m']}, seuil le plus strict "
                  f"{s['smallest_threshold']:.4f}")
            print(f"       {s['n_significant_raw']} rejets bruts -> {s['n_significant_holm']} apres correction")
            if s["lost_to_correction"]:
                print(f"       perdus : {', '.join(s['lost_to_correction'])}")
            if s["survivors"]:
                for name in s["survivors"]:
                    t = fam["family"][name]
                    print(f"       SURVIT : {name} -- {t['verdict']} "
                          f"(p={t['p_value']:.4f}, p_ajustee={t['holm_p_adjusted']:.4f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir), model="NsDiff")
    n_seeds, n_per_seed = rows["seed"].nunique(), samples.shape[1]
    print(f"artefact : {len(rows)} lignes, {n_seeds} graines x {n_per_seed} tirages "
          f"({args.v2_dir})")
    if (n_seeds, n_per_seed) != (len(spec.SEEDS), spec.N_SAMPLES_PER_SEED):
        raise SystemExit(f"artefact hors spec production : {n_seeds}x{n_per_seed} au lieu de "
                         f"{len(spec.SEEDS)}x{spec.N_SAMPLES_PER_SEED} -- refus de produire un "
                         "chiffre etiquete 'production'")

    ens = build_ensemble_rows(rows, samples)
    print(f"ensemble : {len(ens)} lignes (1 par actif/regime/horizon/origine), "
          f"{int(ens['n_samples_total'].iloc[0])} tirages chacune")
    arm_a = h2h.with_winkler(ens)

    assets = sorted(rows["asset"].unique())
    arm_b = h2h.broadcast_seeds(load_challenger(assets, args.horizons), [])
    print(f"{LABEL_B} : {len(arm_b)} lignes oos (lecture seule)")

    cache = v2.price_cache(rows)
    horizons = {}
    for hu in args.horizons:
        print(f"\n=== {hu} ===")
        res = h2h.run_match(arm_a, arm_b, cache, hu, [], assets, LABEL_A, LABEL_B)
        h2h.print_match(hu, res, [], LABEL_A, LABEL_B)
        horizons[hu] = res

    pooled = collect_pooled(horizons)
    holm = apply_holm(pooled)
    _print_holm(holm)

    payload = {
        "question": "la configuration qu'on DEPLOIERAIT (ensemble 5 graines x 200 tirages) "
                    "bat-elle ARIMA-GARCH ?",
        "config": {
            "arm_a": {"label": LABEL_A, **spec.PRODUCTION_SPEC, "artifact": str(args.v2_dir)},
            "arm_b": {"label": LABEL_B, "source": "tracking.db source='oos' (lecture seule)",
                      "intervals": "ANALYTIQUES -- aucun tirage, aucun biais de quantile"},
            "pairing": "par (actif, horizon, target_date), a l'interieur d'un meme regime",
            "difference_vs_task7": "le bras A n'est plus 'une graine tiree au hasard' mais UNE "
                                   "configuration unique (l'ensemble) -- pas de dimension graine, "
                                   "seeds=[]. Les p-values ne sont pas comparables une a une "
                                   "entre les deux scripts.",
            "protocol_asymmetry": f"{LABEL_B} refit a chaque origine et deterministe ; {LABEL_A} "
                                  "train-once-forward. L'ecart mesure melange modele et protocole "
                                  "-- chiffre par nsdiff_refit_cadence.py.",
            "multiple_testing": "Holm-Bonferroni (multiple_testing.py). Familles declarees a "
                                "priori : decision = 6 tests pooles globaux par metrique ; "
                                "etendue = 24 tests pooles par metrique. Tests par cellule non "
                                "corriges, declares exploratoires.",
        },
        "horizons": horizons,
        "pooled_flat": pooled,
        "holm": holm,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
