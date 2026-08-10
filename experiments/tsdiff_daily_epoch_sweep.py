"""
tsdiff_daily_epoch_sweep.py -- selection du budget d'epoques de TSDiff-D
(regime daily) par VALIDATION, sur une fenetre strictement anterieure a la
grille de test `oos`.

POURQUOI CE SWEEP EXISTE. Le duel diffusion-vs-diffusion a d'abord ete genere
en reutilisant cote daily le budget hebdomadaire declare de chaque modele --
convention heritee de `oos_nsdiff_daily_weekly.py`, sure pour NsDiff (40
plat), FAUSSE pour TSDiff. La serie quotidienne compte ~2465 observations
contre ~511 en hebdomadaire : a nombre d'epoques egal, ~5x plus de pas de
gradient. Mesure sur SPY, largeur mediane du PI 95% (en % du prix) :

    epochs   10  ->  12.39 %
    epochs   20  ->   3.41 %      <- ordre de grandeur des lignes oos en base (3.65 %)
    epochs   40  ->   0.46 %
    epochs   80  ->   0.12 %      <- ce que le premier run avait utilise

Au-dela d'une vingtaine d'epoques l'echantillonneur de TSDiff perd sa
variance et la couverture observee tombe a quelques pour cent -- ce n'est plus
un modele sous-calibre, c'est un modele effondre. Le repo connaissait deja la
fragilite (`weekly_headtohead_v2.run_pair_v2` desactive TSDiff-D par defaut en
le qualifiant de "structurally under-calibrated" ; `weekly_headtohead_results.
json`, entraine a 300 epoques, enregistre `coverage_95: 0.0`). Le budget
daily doit donc etre choisi, pas herite.

CE QUI EST RESOLU EN MEME TEMPS : une FUITE. Les epoques hebdomadaires de
TSDiff (40/60/80) viennent d'une selection faite dans le protocole du duel,
dont le bloc de validation (SPY : 2025-09-19 -> 2025-12-05) tombe A
L'INTERIEUR de la grille de test `oos` (2024-10-18 -> 2026-07-02). Ici la
fenetre de validation est ancree AVANT la premiere origine de test, donc
strictement hors grille -- aucune des 90 origines evaluees n'a servi a choisir
quoi que ce soit.

DECOUPAGE (SPY, illustratif) :
    entrainement du sweep   <  premiere origine de validation
    validation              12 origines hebdo, juste AVANT 2024-10-18
    test (jamais touche)    90 origines, 2024-10-18 -> 2026-07-02
Le modele final, lui, est reajuste sur TOUT ce qui precede 2024-10-18 (donc
validation comprise) avec le budget selectionne -- convention standard.

REUTILISE TEL QUEL, aucune fonction recopiee : `epoch_sweep.fit_checkpoints`
(entrainement INCREMENTAL -- les 3 candidats coutent 40 epoques au total, pas
70), `epoch_sweep._sweep_one_model` (CRPS/Cov95/rel_std sur les origines de
validation) et `epoch_sweep.select_epochs` (argmin CRPS_val, le critere maison
-- CRPS etant propre, il penalise de lui-meme les lois effondrees).
Prix relus depuis `diffusion_multiseed_v2/prices/` (les memes series gelees
que le duel, aucun appel reseau).

Sortie : experiments/tsdiff_daily_epochs.json  ({actif|seedN: {epochs, ...}})
Usage   : python tsdiff_daily_epoch_sweep.py
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

import epoch_sweep as es                                               # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly, HORIZON_DAILY  # noqa: E402
from backtest_rolling_tsdiffw import load_baseline_triplets, _weekly_position       # noqa: E402
from oos_nsdiff_daily_weekly import DEFAULT_K_DENOISE                  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ASSETS = list(ASSET_TICKERS.values())
CANDIDATES = (10, 20, 40)     # bracket encadrant la zone d'effondrement mesuree
N_VAL = 12                    # convention maison (n_val du protocole du duel)
N_SAMPLES = 200               # meme budget d'echantillonnage que le duel
PRICE_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "prices"
OUT_PATH = Path(__file__).resolve().parent / "tsdiff_daily_epochs.json"


def validation_block(weekly: pd.Series, weekly_dates: pd.Series, first_test_cutoff: str,
                     n_val: int) -> tuple:
    """(positions des origines de validation, position de fin d'entrainement),
    ancrees juste AVANT la premiere origine de test -- contrairement a
    `epoch_sweep.three_way_split`, qui ancre sur la FIN de la serie et ferait
    donc tomber la validation a l'interieur de notre grille de test."""
    test_pos = _weekly_position(weekly_dates, first_test_cutoff)
    if test_pos is None:
        raise SystemExit(f"premiere origine de test {first_test_cutoff} absente de la grille W-FRI")
    val_pos = list(range(test_pos - n_val, test_pos))
    if val_pos[0] <= 0:
        raise SystemExit(f"historique insuffisant avant le bloc de validation "
                         f"(1re origine de test en position {test_pos}, n_val={n_val})")
    return val_pos, val_pos[0]        # entrainement STRICTEMENT avant la 1re origine de validation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=ASSETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--candidates", type=int, nargs="+", default=list(CANDIDATES))
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    origins = load_baseline_triplets(args.assets)
    records = []
    for asset in args.assets:
        first_cutoff = sorted(origins[origins["asset"] == asset]["cutoff_date"].unique())[0]
        daily = pd.read_parquet(PRICE_DIR / f"{asset.replace('=', '_')}.parquet")["close"]
        weekly, weekly_dates = build_weekly(daily)
        val_pos, train_end = validation_block(weekly, weekly_dates, first_cutoff, args.n_val)
        train_daily = daily.iloc[:int(daily.index.get_loc(weekly_dates.iloc[train_end]))]
        print(f"\n=== {asset} === test des {len(val_pos)} origines "
              f"{weekly_dates.iloc[val_pos[0]].date()} -> {weekly_dates.iloc[val_pos[-1]].date()} "
              f"(1re origine de TEST : {first_cutoff}, jamais touchee) | "
              f"entrainement {len(train_daily)} obs quotidiennes")

        for seed in args.seeds:
            recs = es._sweep_one_model(
                f"{asset}|seed{seed}", "TSDiff-D", train_daily, HORIZON_DAILY,
                daily, seed, args.candidates, val_pos, weekly, weekly_dates, daily,
                args.n_samples, DEFAULT_K_DENOISE)
            records.extend(recs)

    selected = es.select_epochs(records)
    payload = {
        "config": {
            "candidates": args.candidates, "n_val_origins": args.n_val,
            "n_samples": args.n_samples, "k_denoise": DEFAULT_K_DENOISE,
            "criterion": "argmin CRPS_val (epoch_sweep.select_epochs, reutilise tel quel)",
            "validation_anchor": "les n_val origines hebdo STRICTEMENT avant la 1re origine oos "
                                 "-- aucune des 90 origines de test n'entre dans la selection",
            "why": "le budget hebdomadaire de TSDiff, reutilise cote daily, sur-entraine le modele "
                   "(~5x plus de fenetres) et effondre la variance de son echantillonneur : "
                   "largeur du PI 12.4% -> 0.12% du prix entre 10 et 80 epoques sur SPY",
            "prices": "series gelees de diffusion_multiseed_v2/prices/ (aucun appel reseau)",
        },
        "records": records,
        "selected_epochs": selected,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n=== budgets retenus (argmin CRPS_val) ===")
    for key in sorted(selected):
        s = selected[key]
        print(f"  {key:<28} epochs={s['epochs']:<4} CRPS_val={s['crps_val']:.4f} "
              f"Cov95_val={s['cov95_val']:.2f} rel_std%={s['rel_std_pct_val']:.3f}")
    print(f"\n-> {args.out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
