"""
grid2020.py -- chantier B du BRIEF extension/puissance : REGENERER la grille
d'origines a depart 2020-01 (340 origines, `effective_n` ~ 113) sur le panel
etendu `prices_v3/`, pour les deux bras dont depend tout test du chantier :
NsDiff (config production) et ARIMA-GARCH.

POURQUOI CE PERIMETRE ET PAS TOUS LES MODELES. Le brief demande la grille « pour
tous les modeles de reference restants, la comparabilite mutuelle l'exige », et
demande AUSSI de « chiffrer le cout complet avant de lancer, et de decouper si
necessaire ». Le chiffrage (`cost_grid_2020.json`) donne :

    LSTM          13,72 h     <- 84 % du total a lui seul
    Prophet        1,01 h
    ARIMA-GARCH    0,87 h
    SARIMA         0,40 h
    NsDiff         0,30 h
    Naive          0,00 h
    TOTAL         16,31 h     -> hors du budget declare (2 h)

Le decoupage suit la regle posee par le brief : l'hypothese primaire pre-declaree
est `var_limit` sur SPY, W+2/W+3, regime weekly, NsDiff-ensemble vs ARIMA-GARCH.
Or AUCUN test du chantier B ne fait intervenir SARIMA, Prophet, LSTM ou Naive :
ils servent la comparabilite du DASHBOARD, pas le test confirmatoire. Ce script
produit donc les deux bras qui tranchent (1,17 h), et les quatre modeles
classiques restent chiffres et non executes -- ce qui est dit tel quel dans la
note, pas passe sous silence.

PROTOCOLES, inchanges et declares :
  * NsDiff -- TRAIN-ONCE-FORWARD sur les donnees STRICTEMENT anterieures a
    2020-01, 5 graines (42-46), 200 tirages par graine, budget plat 40 epoques.
    La config production est l'ensemble des 5, lue par `nsdiff_production_spec`.
  * ARIMA-GARCH -- REFIT A CHAQUE ORIGINE, son protocole naturel, avec
    `dist="normal"` : la loi d'innovation du bras GARCH du benchmark, etablie par
    mesure au chantier precedent (elle reproduit les lignes `oos` a 1,45e-06,
    contre 2,3e-02 pour le skew-t declare aujourd'hui dans `arima_model`).
    Les bandes a 95 % ET a 80 % sont produites dans le meme appel -- ici on
    genere, donc rien n'a besoin d'etre derive.
  * L'asymetrie de protocole entre les deux est declaree comme partout, et a ete
    chiffree au chantier A3-ii : une cadence de refit jusqu'a 24x plus rapide ne
    deplace aucun verdict.

PRIX : `prices_v3/` uniquement, series gelees, aucun appel reseau. Les deux bras
lisent la MEME serie, donc voient la meme cible et le meme prix courant a chaque
origine -- verifie par construction (les deux tirent `y_true` et `last_close` du
meme objet, il n'y a pas deux sources a reconcilier).

Sortie : experiments/grid2020/
    NsDiff/seed{S}_{ACTIF}.npz    checkpoint reprenable
    NsDiff/rows.parquet + samples.npy
    ARIMA-GARCH/bands.parquet
    config.json
Usage :
    python grid2020.py --smoke              # plomberie : 1 graine, SPY, 5 origines
    python grid2020.py                      # complet (~1,2 h)
    python grid2020.py --models NsDiff      # un seul bras
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

import nsdiff_model as nm                                             # noqa: E402
from benchmarks.multi_horizon import forecast_horizons_arima          # noqa: E402
from backtest_rolling_tsdiffw import HORIZON_UNITS                    # noqa: E402
from epoch_sweep import week_targets                                  # noqa: E402
from prices_v3 import ORIGIN_START, OUT_DIR as PRICES_V3, PANEL, slug  # noqa: E402
from weekly_headtohead import build_weekly, HORIZON_WEEKLY, HORIZON_DAILY  # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W                  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "grid2020"
SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES = 200
K_DENOISE = nm.K_DENOISE
GARCH_DIST = "normal"
LEVELS = (0.95, 0.80)
ROW_COLS = ["seed", "asset", "frequence", "horizon", "horizon_unit", "cutoff_date",
            "target_date", "last_close", "y_pred", "y_lower", "y_upper", "y_true"]


def load_asset(asset: str):
    daily = pd.read_parquet(PRICES_V3 / f"{slug(asset)}.parquet")["close"]
    weekly, weekly_dates = build_weekly(daily)
    origin = pd.Timestamp(ORIGIN_START)
    test_pos = [i for i, d in enumerate(weekly_dates) if d >= origin]
    test_pos = test_pos[:-3]            # W+3 exige trois semaines de marge
    return daily, weekly, weekly_dates, test_pos


# ── bras NsDiff ─────────────────────────────────────────────────────────────

def generate_nsdiff(asset: str, seed: int, n_samples: int, epochs: int,
                    origins_limit: int = None) -> tuple:
    daily, weekly, weekly_dates, test_pos = load_asset(asset)
    if origins_limit:
        test_pos = test_pos[:origins_limit]
    origin = pd.Timestamp(ORIGIN_START)
    train_w = weekly[weekly.index < origin]
    train_d = daily[daily.index < origin]

    t0 = time.time()
    nm.set_seed(seed)
    model_w, mu_w, sd_w = nm.fit_nsdiff(train_w, horizon=HORIZON_WEEKLY, epochs=epochs)
    nm.set_seed(seed)
    model_d, mu_d, sd_d = nm.fit_nsdiff(train_d, horizon=HORIZON_DAILY, epochs=epochs)
    print(f"[NsDiff][seed={seed}][{asset}] fits : hebdo {len(train_w)} obs, quotidien "
          f"{len(train_d)} obs, {time.time() - t0:.0f}s")

    z_w = (nm._log_returns(weekly.values.astype(float)) - mu_w) / sd_w
    z_d = (nm._log_returns(daily.values.astype(float)) - mu_d) / sd_d

    rows, clouds = [], []
    for k, m in enumerate(test_pos):
        _, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
        last_price = float(weekly.iloc[m])
        # meme convention de graine que partout : set_seed(seed + k) par origine
        nm.set_seed(seed + k)
        s_w = nm.forecast_from_fitted(model_w, z_w[:m], mu_w, sd_w, last_price,
                                      horizons=[1, 2, 3], n_samples=n_samples)
        nm.set_seed(seed + k)
        s_d = nm.forecast_from_fitted(model_d, z_d[:daily_pos], mu_d, sd_d, last_price,
                                      horizons=daily_horizons, n_samples=n_samples)
        for h in (1, 2, 3):
            for regime, s in (("weekly", s_w[h]), ("daily", s_d[daily_horizons[h - 1]])):
                lo, hi = (float(q) for q in np.quantile(s, [0.025, 0.975]))
                rows.append({
                    "seed": seed, "asset": asset, "frequence": regime, "horizon": h,
                    "horizon_unit": HORIZON_UNITS[h],
                    "cutoff_date": str(weekly_dates.iloc[m].date()),
                    "target_date": str(target_dates[h - 1].date()),
                    "last_close": last_price, "y_pred": float(np.mean(s)),
                    "y_lower": lo, "y_upper": hi, "y_true": float(weekly.iloc[m + h]),
                })
                clouds.append(np.asarray(s, dtype=np.float32))
        if (k + 1) % 100 == 0 or k == len(test_pos) - 1:
            print(f"[NsDiff][seed={seed}][{asset}] origine {k + 1}/{len(test_pos)}")
    return pd.DataFrame(rows)[ROW_COLS], np.stack(clouds)


def run_nsdiff(assets, seeds, out_dir: Path, n_samples: int, epochs: int,
               origins_limit=None) -> None:
    metas, blocks = [], []
    for seed in seeds:
        for asset in assets:
            ckpt = out_dir / "NsDiff" / f"seed{seed}_{slug(asset)}.npz"
            if ckpt.exists():
                blob = np.load(ckpt, allow_pickle=False)
                meta = pd.DataFrame(json.loads(str(blob["meta"])))[ROW_COLS]
                print(f"[NsDiff][seed={seed}][{asset}] checkpoint relu ({len(meta)} lignes)")
                metas.append(meta)
                blocks.append(blob["samples"])
                continue
            meta, samples = generate_nsdiff(asset, seed, n_samples, epochs, origins_limit)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ckpt, meta=json.dumps(meta.to_dict(orient="list")), samples=samples)
            metas.append(meta)
            blocks.append(samples)
    rows = pd.concat(metas, ignore_index=True)
    samples = np.concatenate(blocks, axis=0)
    assert len(rows) == len(samples), "rows/samples désalignés"
    (out_dir / "NsDiff").mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_dir / "NsDiff" / "rows.parquet", index=False)
    np.save(out_dir / "NsDiff" / "samples.npy", samples)
    print(f"[NsDiff] {len(rows)} lignes x {samples.shape[1]} tirages -> {out_dir / 'NsDiff'}")


# ── bras ARIMA-GARCH ────────────────────────────────────────────────────────

def run_garch(assets, out_dir: Path, origins_limit=None,
              dist: str = GARCH_DIST, arm: str = "ARIMA-GARCH") -> None:
    """`dist` et `arm` sont opt-in (defauts = bras gaussien du benchmark, chemin
    historique inchange). H1 du BRIEF_nsdiff_regeneration_oos_et_famille3.md les
    utilise pour produire le bras skew-t SUR LE MEME CODE : les deux lois ne
    different alors que par ce seul argument, ce qui est la condition pour que la
    comparaison gaussien/skew-t mesure la loi et rien d'autre."""
    frames = []
    for asset in assets:
        ckpt = out_dir / arm / f"{slug(asset)}.parquet"
        if ckpt.exists():
            frames.append(pd.read_parquet(ckpt))
            print(f"[{arm}][{asset}] checkpoint relu ({len(frames[-1])} lignes)")
            continue
        daily, weekly, weekly_dates, test_pos = load_asset(asset)
        if origins_limit:
            test_pos = test_pos[:origins_limit]
        rows, t0 = [], time.time()
        for k, m in enumerate(test_pos):
            _, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
            last_price = float(weekly.iloc[m])
            for regime, train, horizons in (("weekly", weekly.iloc[:m + 1], [1, 2, 3]),
                                            ("daily", daily.iloc[:daily_pos + 1], daily_horizons)):
                try:
                    res = {lvl: forecast_horizons_arima(train, horizons, dist=dist, level=lvl)
                           for lvl in LEVELS}
                except Exception as exc:
                    print(f"[{arm}][{asset}][{regime}] origine {m} ECHEC : "
                          f"{type(exc).__name__}: {exc}")
                    continue
                for wi, h in enumerate(horizons):
                    point, lo95, hi95 = res[0.95][h]
                    _, lo80, hi80 = res[0.80][h]
                    rows.append({
                        "asset": asset, "frequence": regime, "horizon": wi + 1,
                        "horizon_unit": HORIZON_UNITS[wi + 1],
                        "cutoff_date": str(weekly_dates.iloc[m].date()),
                        "target_date": str(target_dates[wi].date()),
                        "last_close": last_price, "y_pred": float(point),
                        "y_lower": float(lo95), "y_upper": float(hi95),
                        "y_lower80": float(lo80), "y_upper80": float(hi80),
                        "y_true": float(weekly.iloc[m + wi + 1]),
                    })
            if (k + 1) % 50 == 0 or k == len(test_pos) - 1:
                print(f"[{arm}][{asset}] origine {k + 1}/{len(test_pos)} "
                      f"({time.time() - t0:.0f}s)")
        df = pd.DataFrame(rows)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ckpt, index=False)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(out_dir / arm / "bands.parquet", index=False)
    print(f"[{arm}] {len(out)} lignes -> {out_dir / arm}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["NsDiff", "ARIMA-GARCH"])
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS_W)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--garch-dist", default=GARCH_DIST,
                   help="loi d'innovation du bras GARCH ; 'skewt' produit le bras de H1")
    p.add_argument("--garch-arm", default=None,
                   help="nom du sous-dossier du bras GARCH (defaut : ARIMA-GARCH pour la loi "
                        "gaussienne, ARIMA-GARCH[<loi>] sinon)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    assets, seeds, out_dir = args.assets, args.seeds, Path(args.out_dir)
    n_samples, epochs, limit = args.n_samples, args.epochs, None
    if args.smoke:
        assets, seeds, n_samples, epochs, limit = ["SPY"], [42], 16, 2, 5
        out_dir = out_dir.parent / "grid2020_smoke"

    t0 = time.time()
    _, _, _, test_pos = load_asset(assets[0])
    n_origins = limit or len(test_pos)
    print(f"grille : {n_origins} origines a partir de {ORIGIN_START}, {len(assets)} actifs, "
          f"2 regimes\n")

    if "NsDiff" in args.models:
        run_nsdiff(assets, seeds, out_dir, n_samples, epochs, limit)
    garch_arm = args.garch_arm or ("ARIMA-GARCH" if args.garch_dist == GARCH_DIST
                                   else f"ARIMA-GARCH[{args.garch_dist}]")
    if "ARIMA-GARCH" in args.models:
        run_garch(assets, out_dir, limit, dist=args.garch_dist, arm=garch_arm)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Un run sur une autre loi d'innovation ecrit son propre bandeau : ecraser celui
    # du bras gaussien reviendrait a perdre la trace de la config qui a produit les
    # artefacts deja sur le disque.
    cfg_name = "config.json" if garch_arm == "ARIMA-GARCH" else f"config_{garch_arm}.json"
    (out_dir / cfg_name).write_text(json.dumps({
        "scope": "chantier B -- grille d'origines a depart 2020-01, panel etendu",
        "origin_start": ORIGIN_START, "n_origins": n_origins,
        "effective_n": n_origins // 3,
        "assets": assets, "seeds": seeds, "n_samples": n_samples, "epochs": epochs,
        "k_denoise": K_DENOISE,
        "prices": f"{PRICES_V3} -- series gelees, aucun appel reseau ; les deux bras lisent "
                  "la MEME serie, donc voient la meme cible et le meme prix courant",
        "nsdiff_protocol": "train-once-forward sur donnees < 2020-01, config production = "
                           "ensemble des 5 graines (1000 tirages)",
        "garch_arm": garch_arm,
        "garch_protocol": f"refit a chaque origine, dist={args.garch_dist!r} (loi d'innovation du bras "
                          "GARCH du benchmark, etablie par mesure au chantier precedent) ; "
                          "bandes 95 % et 80 % produites dans le meme appel",
        "models_not_regenerated": {
            "SARIMA": "0,40 h", "Prophet": "1,01 h", "LSTM": "13,72 h", "Naive": "0,00 h",
            "raison": "aucun test du chantier B ne les fait intervenir -- ils servent la "
                      "comparabilite du dashboard, pas le test confirmatoire. Chiffres dans "
                      "cost_grid_2020.json, non executes, declares tels quels.",
        },
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }, indent=2, ensure_ascii=False))
    print(f"\nTotal : {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
