"""
grid2020_refs.py -- chantier R2 du BRIEF « regeneration oos et famille 3 » :
les QUATRE MODELES DE REFERENCE CLASSIQUES (SARIMA, Prophet, Naive, LSTM) sur la
grille regeneree -- 340 origines a partir de 2020-01, 7 actifs, prix geles
`prices_v3/`.

Le chantier precedent avait produit les deux bras qui tranchaient l'hypothese
primaire (NsDiff et ARIMA-GARCH) et laisse ces quatre-la « chiffres et non
executes ». Le present brief demande la grille COMPLETE : « tous modeles », parce
que la comparabilite mutuelle du benchmark l'exige -- un dashboard qui compare six
modeles dont deux seulement ont ete regeneres ne compare plus rien.

CE QUI EST REUTILISE, ET POURQUOI CA COMPTE. Le protocole hebdo de ces quatre
modeles n'est pas « appeler leur fonction de prevision ». Il porte trois choix
declares ailleurs, qu'il aurait ete facile de perdre en reecrivant la boucle :

  * SARIMA et Prophet ont des variantes HEBDO-NATIVES (saisonnalite quotidienne
    desactivee, dates futures sur W-FRI) -- `weekly_multimodel.REGIME_C_FORECAST` ;
  * SARIMA, Prophet, Naive et LSTM portent la CALIBRATION SIGMA EWMA causale
    adoptee le 2026-07-31, avec son lag de resolution (le z d'une origine k
    n'entre dans l'etat de W+j qu'a l'origine k+j) ;
  * LSTM regime C a un SEQ_LEN choisi PAR ACTIF sur un bloc de validation.

La boucle de `weekly_multimodel.run_model_asset` est donc appelee telle quelle,
via trois parametres opt-in ajoutes pour ce chantier (`daily`, `test_pos`,
`forecast_fn`) : serie gelee au lieu du reseau, grille d'origines imposee au lieu
de `three_way_split`. Le chemin historique reste inchange bit-for-bit.

LE BLOC DE VALIDATION DU LSTM, ET UNE FUITE QU'ON NE REPRODUIT PAS. Le SEQ_LEN*
publie a ete choisi sur les 12 origines qui precedent immediatement les 30
origines de test de l'ancienne grille -- lesquelles tombent EN PLEIN dans les 340
origines de la nouvelle. Le reutiliser tel quel importerait une fuite (meme
mecanisme que celle deja declaree pour les epoques hebdo de TSDiff). Le sweep est
donc rejoue ici sur un bloc de validation STRICTEMENT ANTERIEUR a 2020-01, cible
W+3 comprise -- le meme principe que le train-once-forward de NsDiff, qui ne voit
rien apres 2020 non plus. Cout : ~3 candidats x 12 origines x 7 actifs.

Sortie : experiments/grid2020_refs/
    lstm_seq_len.json          SEQ_LEN* par actif, selectionne avant 2020
    <MODELE>/<regime>_<ACTIF>.parquet   checkpoint reprenable par cellule
    <MODELE>/bands_<regime>.parquet     concatenation
    config_<regime>.json
Usage :
    python grid2020_refs.py --smoke                 # plomberie : SPY, 5 origines
    python grid2020_refs.py                         # phase W (regime hebdo natif)
    python grid2020_refs.py --regime daily          # phase D, conditionnee
"""

# L'ordre d'import est contraint : `weekly_multimodel` doit venir EN PREMIER, il
# importe tensorflow avant yfinance/statsmodels (deadlock confirme sinon, cf. sa
# docstring). Ne pas reordonner.
import weekly_multimodel as wmm                                        # noqa: E402,I001

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

import lstm_weekly_sweep as lws                                        # noqa: E402
from backtest_rolling_tsdiffw import HORIZON_UNITS                     # noqa: E402
from grid2020 import load_asset                                        # noqa: E402
from prices_v3 import ORIGIN_START, PANEL                              # noqa: E402
from prices_v3 import slug                                             # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "grid2020_refs"
MODELS = ("SARIMA", "Prophet", "Naive", "LSTM")
REGIME_CODE = {"weekly": "C", "daily": "B"}      # nomenclature base <- nomenclature weekly_multimodel
N_VAL = 12                                        # taille du bloc de validation du sweep LSTM
WEEK_MARGIN = 3                                   # W+3 : la validation doit finir 3 semaines avant
HORIZON_BY_UNIT = {v: k for k, v in HORIZON_UNITS.items()}
BAND_COLS = ["asset", "frequence", "horizon", "horizon_unit", "cutoff_date", "target_date",
             "last_close", "y_pred", "y_lower", "y_upper", "y_true", "sigma_scale"]


def validation_origins(test_pos: list, n_val: int) -> list:
    """Les `n_val` origines qui precedent la grille de test, marge W+3 comprise :
    la cible la plus lointaine du bloc de validation tombe encore avant la
    premiere origine de test. Zero recouvrement, par construction."""
    last = test_pos[0] - 1 - WEEK_MARGIN
    first = last - n_val + 1
    if first < 0:
        raise SystemExit(f"historique insuffisant pour {n_val} origines de validation avant "
                         f"{ORIGIN_START}")
    return list(range(first, last + 1))


def lstm_seq_len(assets, out_dir: Path, n_val: int, candidates=None) -> dict:
    """SEQ_LEN* par actif, selectionne sur validation pre-2020 (cf. docstring).
    Resultat cache : le sweep ne se rejoue pas d'un run a l'autre."""
    path = out_dir / "lstm_seq_len.json"
    cache = json.loads(path.read_text()) if path.exists() else {"selected_seq_len": {}, "sweep": {}}
    missing = [a for a in assets if a not in cache["selected_seq_len"]]
    if not missing:
        return cache["selected_seq_len"]

    for asset in missing:
        daily, _, weekly_dates, test_pos = load_asset(asset)
        val_pos = validation_origins(test_pos, n_val)
        print(f"[sweep LSTM][{asset}] validation {weekly_dates.iloc[val_pos[0]].date()} -> "
              f"{weekly_dates.iloc[val_pos[-1]].date()} ({len(val_pos)} origines, "
              f"strictement avant {ORIGIN_START})")
        res = lws.sweep_asset(asset, asset, n_val, 0, None, None,
                              candidates=candidates or lws.SEQ_LEN_CANDIDATES,
                              daily=daily, val_pos=val_pos, with_regime_b=False)
        cache["selected_seq_len"][asset] = res["selected"]["seq_len"]
        cache["sweep"][asset] = res
        cache["protocol"] = (f"SEQ_LEN* regime hebdo natif, regle 1-SE sur CRPS de validation, "
                             f"bloc de {n_val} origines STRICTEMENT anterieur a {ORIGIN_START} "
                             f"(cible W+3 comprise) -- la grille de test n'est jamais touchee")
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, default=str))
    return cache["selected_seq_len"]


def normalise_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """`weekly_multimodel` etiquette ses horizons « W1/W2/W3 »
    (`weekly_headtohead.HORIZON_LABELS`), la grille et `tracking.db` « W+1/W+2/W+3 »
    (`backtest_rolling_tsdiffw.HORIZON_UNITS`). Les deux conventions coexistent
    dans le depot depuis longtemps ; le pont entre elles se fait ICI, en un seul
    endroit, et la colonne `horizon` entiere en est deduite -- pas recopiee.

    Idempotent : rejouer la conversion sur un artefact deja converti ne le change
    pas, ce qui permet de reparer les checkpoints ecrits avant ce correctif sans
    recalculer une seule origine."""
    out = df.copy()
    hu = out["horizon_unit"].astype(str).str.replace(r"^W(\d+)$", r"W+\1", regex=True)
    out["horizon_unit"] = hu
    out["horizon"] = hu.map(HORIZON_BY_UNIT)
    if out["horizon"].isna().any():
        bad = sorted(hu[out["horizon"].isna()].unique())
        raise SystemExit(f"horizons non reconnus : {bad}")
    out["horizon"] = out["horizon"].astype(int)
    return out


def to_bands(records: list, regime: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = pd.DataFrame({
        "asset": df["asset"], "frequence": regime,
        "horizon": 0, "horizon_unit": df["horizon"],
        "cutoff_date": df["origin_date"], "target_date": df["target_date"],
        "last_close": df["last_close"], "y_pred": df["point"],
        "y_lower": df["lower"], "y_upper": df["upper"], "y_true": df["actual"],
        "sigma_scale": df["sigma_scale"],
    })
    return normalise_horizon(out)[BAND_COLS]


def run_cell(model: str, asset: str, regime: str, seq_len: dict, origins_limit=None) -> dict:
    daily, _, _, test_pos = load_asset(asset)
    if origins_limit:
        test_pos = test_pos[:origins_limit]
    return wmm.run_model_asset(model, asset, asset, REGIME_CODE[regime],
                               n_val=0, n_test=0, start=None, end=None,
                               lstm_weekly_seq_len=seq_len, calibrate_sigma="on",
                               daily=daily, test_pos=test_pos)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--regime", default="weekly", choices=list(REGIME_CODE))
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    assets, out_dir, limit = args.assets, Path(args.out_dir), None
    if args.smoke:
        assets, limit = ["SPY"], 5
        out_dir = out_dir.parent / "grid2020_refs_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seq_len = (lstm_seq_len(assets, out_dir, args.n_val)
               if "LSTM" in args.models and args.regime == "weekly" else {})

    meta = {}
    for model in args.models:
        frames = []
        for asset in assets:
            ckpt = out_dir / model / f"{args.regime}_{slug(asset)}.parquet"
            if ckpt.exists():
                # normalise a la relecture : repare les checkpoints ecrits avant le
                # correctif d'etiquetage d'horizon, sans recalculer une origine
                df = normalise_horizon(pd.read_parquet(ckpt))[BAND_COLS]
                df.to_parquet(ckpt, index=False)
                frames.append(df)
                print(f"[{model}][{asset}] checkpoint relu ({len(df)} lignes)")
                continue
            t_cell = time.time()
            res = run_cell(model, asset, args.regime, seq_len, limit)
            df = to_bands(res["records"], args.regime)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ckpt, index=False)
            frames.append(df)
            meta[f"{model}|{asset}"] = {"n_rows": len(df), "n_failed": res["n_failed"],
                                        "T0": res["T0"], "elapsed_s": round(time.time() - t_cell, 1)}
            print(f"[{model}][{asset}] {len(df)} lignes, {res['n_failed']} origines en echec, "
                  f"{(time.time() - t_cell) / 60:.1f} min")
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(out_dir / model / f"bands_{args.regime}.parquet", index=False)
        print(f"[{model}] {len(out)} lignes -> {out_dir / model}")

    (out_dir / f"config_{args.regime}.json").write_text(json.dumps({
        "scope": f"chantier R2 phase {'W' if args.regime == 'weekly' else 'D'} -- modeles de "
                 f"reference classiques sur la grille regeneree",
        "origin_start": ORIGIN_START, "assets": assets, "models": args.models,
        "regime": args.regime, "regime_code_weekly_multimodel": REGIME_CODE[args.regime],
        "protocol": "refit a CHAQUE origine (protocole naturel de ces modeles), variantes "
                    "hebdo-natives pour SARIMA/Prophet, calibration sigma EWMA causale pour "
                    "SARIMA/Prophet/Naive/LSTM (ARIMA-GARCH en est exclu partout, son sigma "
                    "GARCH est deja dynamique)",
        "lstm_seq_len": seq_len,
        "lstm_seq_len_selection": f"1-SE sur CRPS de validation, bloc de {args.n_val} origines "
                                  f"strictement anterieur a {ORIGIN_START} -- le SEQ_LEN* publie "
                                  f"avait ete choisi sur un bloc qui tombe dans cette grille de "
                                  f"test, il n'est pas reutilise",
        "prices": "prices_v3/ -- series gelees, aucun appel reseau",
        "cells": meta,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }, indent=2, default=str, ensure_ascii=False))
    print(f"\nTotal : {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
