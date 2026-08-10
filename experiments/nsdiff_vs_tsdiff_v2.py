"""
nsdiff_vs_tsdiff_v2.py -- match diffusion-vs-diffusion A BUDGET
D'ECHANTILLONNAGE EGAL : NsDiff vs TSDiff, n_samples=200 des DEUX cotes, sur
les 90 origines `oos`, aux 3 horizons (W+1/2/3), dans les 2 regimes
(daily-B / weekly-natif-C), 5 graines.

CE QUI EXISTAIT DEJA, ET POURQUOI CE MATCH N'EST PAS UN DOUBLON.
`NOTE_compare_weekly_tsdiff_nsdiff.md` compare deja les deux modeles, et le
fait bien : m=500 des deux cotes (donc deja a budget egal), 5 graines, CRPS +
Cov50/80/95 + sharpness + Winkler + PIT. Mais elle est limitee au **regime
weekly-natif** et aux **30 origines du duel**. Ce script ajoute les deux
choses qui manquaient au chantier daily-vs-weekly :
  1. le **regime daily (B)**, jamais compare entre les deux diffusions ;
  2. la **grille des 90 origines `oos`**, celle sur laquelle tout le reste de
     la consolidation est mesure -- donc des chiffres directement comparables
     a `NOTE_nsdiff_consolidation_daily_vs_weekly.md`, ce que les 30 origines
     du duel ne permettaient pas.

POURQUOI IL FALLAIT REGENERER TSDIFF. Les lignes `oos` de TSDiff dans
`tracking.db` ont ete produites a **n_samples=50**. Or a 50 tirages, le
quantile empirique a 97.5% n'estime en fait que le niveau ~0.9564 : un
intervalle etiquete 95% n'en couvre reellement que 91.3% (contre 94.1% a
n=200). Comparer NsDiff@200 a TSDiff@50 offrirait donc ~2.8 points de
couverture gratuits a NsDiff -- exactement le biais que la tache 4 du brief a
mis au jour. Les deux bras sont ici regeneres par
`diffusion_multiseed_v2.py`, dans le meme run, sur la MEME serie de prix
gelee, par la MEME boucle de generation (kwarg `engine` de
`generate_nsdiff_asset`).

ASYMETRIE RESTANTE, DECLAREE : les budgets d'epoques ne sont pas egaux et ne
peuvent pas l'etre honnetement -- TSDiff a 40/60/80 selon (actif, graine),
selectionnes sur validation et relus verbatim du repo ; NsDiff a 40, plat.
Egaliser les epoques reviendrait a jeter la selection de TSDiff, donc a
l'handicaper. L'asymetrie va dans le sens de TSDiff : tout verdict favorable
a NsDiff en est d'autant plus conservateur.

Tests : `diffusion_headtohead` (module partage avec le match vs ARIMA-GARCH,
meme code, memes conventions de signe) -- bootstrap par blocs apparie par
origine, ecart de couverture, pooling entre actifs via skill-scores vs marche
aleatoire, verdicts par graine ET graines poolees.

Sortie : experiments/nsdiff_vs_tsdiff_v2.json
Usage   : python nsdiff_vs_tsdiff_v2.py
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
import nsdiff_v2_data as v2                                           # noqa: E402

DUEL_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2"
OUT_PATH = Path(__file__).resolve().parent / "nsdiff_vs_tsdiff_v2.json"
LABEL_A, LABEL_B = "NsDiff", "TSDiff"


def load_arm(model: str) -> pd.DataFrame:
    rows = v2.load_rows(data_dir=DUEL_DIR / model, model=model)
    rows = h2h.with_winkler(rows)
    return pd.concat([rows, h2h.seed_pooled_rows(rows)], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    arm_a, arm_b = load_arm(LABEL_A), load_arm(LABEL_B)
    seeds = [s for s in sorted(arm_a["seed"].unique()) if s >= 0]
    assets = sorted(arm_a["asset"].unique())
    cfg = json.loads((DUEL_DIR / "config.json").read_text())
    print(f"{LABEL_A}: {len(arm_a)} lignes | {LABEL_B}: {len(arm_b)} lignes | "
          f"graines {seeds} | n_samples={cfg['n_samples']}")

    cache = v2.price_cache(arm_a[arm_a["seed"] >= 0])
    horizons = {}
    for hu in args.horizons:
        print(f"\n=== {hu} ===")
        res = h2h.run_match(arm_a, arm_b, cache, hu, seeds, assets, LABEL_A, LABEL_B)
        h2h.print_match(hu, res, seeds, LABEL_A, LABEL_B)
        horizons[hu] = res

    payload = {
        "config": {
            **cfg,
            "match": f"{LABEL_A} vs {LABEL_B}, n_samples identiques des deux cotes",
            "origins": "90 origines oos (la grille de tout le chantier de consolidation), "
                       "contre 30 pour NOTE_compare_weekly_tsdiff_nsdiff.md",
            "adds_vs_existing_note": "le regime daily (B), jamais compare entre les deux diffusions, "
                                     "et la grille 90 origines directement comparable au reste du chantier",
            "why_regenerated": "les lignes oos de TSDiff dans tracking.db sont a n_samples=50 : "
                               "un PI 95% n'y couvre reellement que ~91.3% (quantile empirique "
                               "biaise vers l'interieur), soit ~2.8 points offerts a NsDiff",
            "epoch_asymmetry": "TSDiff 40/60/80 par (actif, graine), selectionnes sur validation ; "
                               "NsDiff 40 plat. Avantage TSDiff, herite du repo -- tout verdict "
                               "favorable a NsDiff en est d'autant plus conservateur.",
        },
        "horizons": horizons,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
