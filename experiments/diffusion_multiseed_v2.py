"""
diffusion_multiseed_v2.py -- generation multi-graines des DEUX modeles de
diffusion (NsDiff et TSDiff) a budget d'echantillonnage eleve (n_samples=200),
sur les 90 origines `oos`, aux 3 horizons W+1/2/3 et dans les 2 regimes
(daily-B / weekly-natif-C), nuages predictifs complets conserves.

Anciennement `nsdiff_multiseed_v2.py` (tache 4 du brief de consolidation) ;
etendu a TSDiff pour le match diffusion-vs-diffusion a budget egal.

POURQUOI 200 TIRAGES (rappel, le coeur de la tache 4). Les bornes ne sont pas
produites par le modele : elles sont LUES sur le nuage, en quantiles
empiriques 2.5%/97.5%. Or `np.quantile` interpole vers le rang (n-1)*0.975+1,
dont l'esperance vaut ~F^-1(k/(n+1)) : a n=50 cela estime le niveau 0.9564,
soit un intervalle nominalement "95%" qui n'en couvre en verite que 91.3% ;
a n=200, 0.9703 -> 94.1% ; a n=500, 94.6%. Le biais est purement mecanique
(il ne depend meme pas de la loi) et 200 en recupere l'essentiel -- passer a
500 ne rendrait que 0.6 point de plus.

POURQUOI LES DEUX MODELES DANS UN SEUL RUN. yfinance resert les memes cotes
avec un bruit de derniere decimale d'un appel a l'autre (~2e-7 en relatif,
phenomene deja documente dans `dashboard_d7_w1.load_price_history_cache`), et
un fit de 40-80 epoques amplifie cela a ~6e-6 sur les echantillons. Deux bras
d'un duel telecharges separement ne verraient donc PAS exactement les memes
prix. Ici les prix sont recuperes UNE fois, geles sur disque
(`prices/<actif>.parquet`) et partages par les deux modeles -- condition
elementaire d'un match a budget egal.

BUDGETS D'EPOQUES -- declares, jamais inventes ici :
  * NsDiff : 40, plat (`weekly_nsdiff_production.NSDIFF_EPOCHS_W`).
  * TSDiff : 40/60/80 selon (actif, graine), relus VERBATIM depuis
    `compare_weekly_diffusion.json` (`meta_by_asset_seed.epochs_tsdiff_w`),
    ou ils avaient ete selectionnes sur validation. Aucun re-sweep.
  * Regime daily, NsDiff : 40, le meme budget plat des deux cotes.
  * Regime daily, TSDiff : SELECTIONNE PAR VALIDATION
    (`tsdiff_daily_epoch_sweep.py` -> `tsdiff_daily_epochs.json`), sur une
    fenetre strictement anterieure a la grille de test.
    Le premier run de ce script reutilisait cote daily le budget hebdomadaire
    de chaque modele (convention heritee de `oos_nsdiff_daily_weekly.py`).
    C'est sur pour NsDiff, FAUX pour TSDiff : la serie quotidienne compte ~5x
    plus de fenetres d'entrainement, donc a epoques egales ~5x plus de pas de
    gradient, et la variance de l'echantillonneur de TSDiff s'effondre --
    largeur du PI 95% sur SPY : 12.39% du prix a 10 epoques, 3.41% a 20,
    0.46% a 40, 0.12% a 80, pour une couverture observee tombant a 1-6%. Le
    bras daily du premier run est donc INVALIDE et a ete regenere.
  ASYMETRIE A DECLARER : cote hebdomadaire, TSDiff beneficie d'un budget regle
  par validation actif par actif, NsDiff d'un budget plat. C'est un avantage
  pour TSDiff, herite des selections deja faites dans le repo et non introduit
  ici -- tout verdict favorable a NsDiff en est d'autant plus conservateur.
  Cote daily, les deux budgets sont desormais choisis de la meme facon (plat
  declare pour NsDiff, validation hors grille pour TSDiff).

Cadrage inchange : artefact ISOLE, `tracking.db` n'est JAMAIS ecrit (la piste
`oos`/dashboard reste single-seed 42 et n_samples=50, comparable aux 6 autres
modeles).

Reutilise tel quel : `oos_nsdiff_daily_weekly.generate_nsdiff_asset` (via son
kwarg opt-in `engine`, qui fait passer les DEUX modeles par la meme boucle,
les memes origines et la meme lecture de quantiles), `.fetch_verified`,
`.load_baseline_triplets_daily`, `oos_nsdiff_tlt.fetch_tlt_patched`,
`backtest_rolling_tsdiffw.load_baseline_triplets`, `weekly_headtohead.
build_weekly`.

Sorties (`experiments/diffusion_multiseed_v2/`) :
    prices/<actif>.parquet         serie quotidienne gelee, partagee
    <Modele>/seed{S}_{ACTIF}.npz   checkpoint reprenable par (graine, actif)
    <Modele>/rows.parquet          1 ligne = (graine, actif, horizon, regime, origine)
    <Modele>/samples.npy           (n_rows, n_samples) float32, aligne sur rows
    config.json                    budgets, duree, empreinte des origines

Usage :
    python diffusion_multiseed_v2.py --smoke                 # plomberie : 1 graine, SPY, epochs=2
    python diffusion_multiseed_v2.py                         # NsDiff + TSDiff, 5 graines x 5 actifs
    python diffusion_multiseed_v2.py --models NsDiff         # un seul bras
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

import tsdiff_model as td                                                    # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly          # noqa: E402
from backtest_rolling_tsdiffw import load_baseline_triplets, FETCH_START, FETCH_END  # noqa: E402
from oos_nsdiff_daily_weekly import (                                        # noqa: E402
    DiffusionEngine, nsdiff_engine, load_baseline_triplets_daily, fetch_verified,
    generate_nsdiff_asset, DEFAULT_K_DENOISE,
)
from oos_nsdiff_tlt import fetch_tlt_patched                                 # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W                         # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ASSETS = list(ASSET_TICKERS.values())          # SPY, BTC-USD, ETH-USD, ZN=F, TLT
N_SAMPLES_V2 = 200
OUT_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2"
TSDIFF_EPOCH_SOURCE = Path(__file__).resolve().parent / "compare_weekly_diffusion.json"
TSDIFF_DAILY_EPOCH_SOURCE = Path(__file__).resolve().parent / "tsdiff_daily_epochs.json"

ROW_COLS = ["seed", "asset", "frequence", "horizon", "horizon_unit",
            "cutoff_date", "target_date", "last_close", "y_pred", "y_lower",
            "y_upper", "y_true"]


def tsdiff_engine() -> DiffusionEngine:
    """Meme boucle que NsDiff ; seules differences reelles : le nom du fit, et
    `k_denoise` qui est un parametre d'INFERENCE pour TSDiff (nombre de pas
    DDIM) alors qu'il est fit-time pour NsDiff -- il est donc passe au
    forecast ici, et pas au fit."""
    return DiffusionEngine(
        "TSDiff", td,
        lambda train, horizon, epochs, k: td.fit_tsdiff(train, horizon=horizon, epochs=epochs),
        lambda model, hist, mu, sd, last_price, horizons, n_samples, k: td.forecast_from_fitted(
            model, hist, mu, sd, last_price, horizons=horizons, n_samples=n_samples, k_denoise=k),
    )


ENGINES = {"NsDiff": nsdiff_engine, "TSDiff": tsdiff_engine}


def load_tsdiff_epochs() -> dict:
    """{(ticker, graine): epochs} relus verbatim de compare_weekly_diffusion.json."""
    meta = json.loads(TSDIFF_EPOCH_SOURCE.read_text())["meta_by_asset_seed"]
    out = {}
    for key, v in meta.items():
        short, seed_key = key.split("|")
        out[(ASSET_TICKERS[short], int(seed_key.replace("seed", "")))] = int(v["epochs_tsdiff_w"])
    return out


def load_tsdiff_daily_epochs() -> dict:
    """{(ticker, graine): epochs} du regime DAILY, selectionnes par validation
    sur une fenetre strictement anterieure a la grille de test
    (`tsdiff_daily_epoch_sweep.py`). Necessaire parce que reutiliser le budget
    hebdomadaire cote daily sur-entraine TSDiff : ~5x plus de fenetres, et sa
    variance d'echantillonnage s'effondre (PI de 12.4% a 0.12% du prix entre
    10 et 80 epoques). NsDiff ne montre pas cette fragilite mais passe par le
    meme mecanisme, pour que les deux bras soient traites identiquement."""
    if not TSDIFF_DAILY_EPOCH_SOURCE.exists():
        raise SystemExit(f"{TSDIFF_DAILY_EPOCH_SOURCE.name} absent -- lancer d'abord "
                         "`python tsdiff_daily_epoch_sweep.py` (refus de reutiliser le budget "
                         "hebdomadaire cote daily : il effondre l'echantillonneur de TSDiff).")
    selected = json.loads(TSDIFF_DAILY_EPOCH_SOURCE.read_text())["selected_epochs"]
    out = {}
    for key, v in selected.items():                     # "SPY|seed42|TSDiff-D"
        asset, seed_key, _ = key.split("|")
        out[(asset, int(seed_key.replace("seed", "")))] = int(v["epochs"])
    return out


def epochs_for(model: str, asset: str, seed: int, tsdiff_epochs: dict, override=None) -> int:
    if override is not None:
        return override
    if model == "NsDiff":
        return NSDIFF_EPOCHS_W
    try:
        return tsdiff_epochs[(asset, seed)]
    except KeyError:
        raise SystemExit(f"budget d'epoques TSDiff absent pour ({asset}, seed={seed}) dans "
                         f"{TSDIFF_EPOCH_SOURCE.name} -- refus d'en inventer un.")


def epochs_daily_for(model: str, asset: str, seed: int, daily_epochs: dict, override=None) -> int:
    """NsDiff garde son budget plat (40) des deux cotes -- sa calibration ne
    depend pas du budget de la meme facon, et c'est la convention declaree du
    repo. TSDiff prend le budget daily selectionne par validation."""
    if override is not None:
        return override
    if model == "NsDiff":
        return NSDIFF_EPOCHS_W
    try:
        return daily_epochs[(asset, seed)]
    except KeyError:
        raise SystemExit(f"budget d'epoques DAILY absent pour TSDiff ({asset}, seed={seed}) -- "
                         "lancer `python tsdiff_daily_epoch_sweep.py`.")


# ── prix geles, partages par les deux modeles ───────────────────────────────

def load_price_data(assets: list, out_dir: Path) -> dict:
    """Origines + series de prix verifiees, par actif. Telechargees une seule
    fois puis GELEES sur disque : un second run (ou le second bras du duel)
    relit le parquet au lieu de re-solliciter yfinance, dont le bruit de
    derniere decimale suffirait a desaligner les deux bras."""
    price_dir = out_dir / "prices"
    price_dir.mkdir(parents=True, exist_ok=True)
    origins_c_all = load_baseline_triplets(assets)
    origins_b_all = load_baseline_triplets_daily(assets)

    prices = {}
    for asset in assets:
        origins_c = origins_c_all[origins_c_all["asset"] == asset].reset_index(drop=True)
        origins_b = origins_b_all[origins_b_all["asset"] == asset].reset_index(drop=True)
        frozen = price_dir / f"{asset.replace('=', '_')}.parquet"
        if frozen.exists():
            daily = pd.read_parquet(frozen)["close"]
            print(f"[{asset}] prix geles relus ({frozen.name}, {len(daily)} obs) -- aucun appel reseau.")
        else:
            if asset == "TLT":
                daily = fetch_tlt_patched(asset, FETCH_START, FETCH_END)
            else:
                fetched = fetch_verified(asset, origins_c, origins_b)
                if fetched is None:
                    raise SystemExit(f"[{asset}] verification des prix echouee -- run impossible.")
                daily = fetched[0]
            daily.to_frame("close").to_parquet(frozen)
            print(f"[{asset}] prix telecharges puis geles -> {frozen.name}")
        weekly, weekly_dates = build_weekly(daily)
        prices[asset] = (daily, weekly, weekly_dates, origins_c, origins_b)
    return prices


# ── generation ──────────────────────────────────────────────────────────────

def _split_rows(rows: list, seed: int) -> tuple:
    samples = np.stack([np.asarray(r.pop("samples"), dtype=np.float32) for r in rows])
    meta = pd.DataFrame(rows)
    meta["seed"] = seed
    return meta[ROW_COLS], samples


def run_cell(model: str, seed: int, asset: str, prices: dict, epochs: int,
             epochs_daily: int, n_samples: int, k_denoise: int, out_dir: Path) -> tuple:
    path = out_dir / model / f"seed{seed}_{asset.replace('=', '_')}.npz"
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        meta = pd.DataFrame(json.loads(str(blob["meta"])))[ROW_COLS]
        print(f"[{model}][seed={seed}][{asset}] checkpoint relu ({len(meta)} lignes)")
        return meta, blob["samples"]

    daily, weekly, weekly_dates, origins_c, origins_b = prices[asset]
    t0 = time.time()
    rows_c, rows_b = generate_nsdiff_asset(
        asset, daily, weekly, weekly_dates, origins_c, origins_b,
        epochs, seed, n_samples, k_denoise, collect_samples=True,
        engine=ENGINES[model](), epochs_daily=epochs_daily,
    )
    meta, samples = _split_rows(rows_c + rows_b, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta.to_dict(orient="list")), samples=samples)
    print(f"[{model}][seed={seed}][{asset}] {len(meta)} lignes x {samples.shape[1]} tirages "
          f"(epochs W={epochs} / D={epochs_daily}) en {time.time() - t0:.0f}s -> checkpoint")
    return meta, samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["NsDiff", "TSDiff"], choices=list(ENGINES))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--assets", nargs="+", default=ASSETS)
    p.add_argument("--epochs", type=int, default=None,
                   help="force le meme budget pour tout le monde (defaut : budget declare de chaque modele)")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_V2)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--smoke", action="store_true",
                   help="plomberie : graine 42, SPY, epochs=2, 8 tirages, repertoire temporaire")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    seeds, assets = args.seeds, args.assets
    n_samples, epochs_override = args.n_samples, args.epochs
    if args.smoke:
        seeds, assets, n_samples, epochs_override = [42], ["SPY"], 8, 2
        out_dir = out_dir.parent / "diffusion_multiseed_v2_smoke"

    t_start = time.time()
    print("=== prix (telecharges une fois, geles, partages par les deux modeles) ===")
    prices = load_price_data(assets, out_dir)
    need_tsdiff = "TSDiff" in args.models and epochs_override is None
    tsdiff_epochs = load_tsdiff_epochs() if need_tsdiff else {}
    tsdiff_daily_epochs = load_tsdiff_daily_epochs() if need_tsdiff else {}

    per_model = {}
    for model in args.models:
        print(f"\n########## {model} ##########")
        metas, blocks, budgets = [], [], {}
        for seed in seeds:
            for asset in assets:
                ep = epochs_for(model, asset, seed, tsdiff_epochs, epochs_override)
                ep_d = epochs_daily_for(model, asset, seed, tsdiff_daily_epochs, epochs_override)
                budgets[f"{asset}|seed{seed}"] = {"weekly": ep, "daily": ep_d}
                meta, samples = run_cell(model, seed, asset, prices, ep, ep_d, n_samples,
                                         args.k_denoise, out_dir)
                metas.append(meta)
                blocks.append(samples)

        rows = pd.concat(metas, ignore_index=True)
        samples = np.concatenate(blocks, axis=0)
        assert len(rows) == len(samples), "rows/samples desalignes"
        (out_dir / model).mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out_dir / model / "rows.parquet", index=False)
        np.save(out_dir / model / "samples.npy", samples)
        per_model[model] = {"n_rows": int(len(rows)), "epochs_by_asset_seed": budgets}
        print(f"[{model}] {len(rows)} lignes x {samples.shape[1]} tirages -> {out_dir / model}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "models": args.models, "seeds": seeds, "assets": assets,
        "n_samples": n_samples, "k_denoise": args.k_denoise,
        "epochs_source": {
            "NsDiff": f"plat, weekly_nsdiff_production.NSDIFF_EPOCHS_W={NSDIFF_EPOCHS_W}",
            "TSDiff": f"par (actif, graine), relu verbatim de {TSDIFF_EPOCH_SOURCE.name} "
                      "(selection sur validation, aucun re-sweep ici)",
            "regime_daily": "NsDiff : budget plat (40) des deux cotes. TSDiff : budget DAILY "
                            "selectionne par validation sur une fenetre strictement anterieure a la "
                            "grille de test (tsdiff_daily_epochs.json) -- reutiliser son budget "
                            "hebdomadaire cote daily le sur-entraine (~5x plus de fenetres) et "
                            "effondre son echantillonneur : PI de 12.4%% a 0.12%% du prix entre 10 "
                            "et 80 epoques sur SPY.",
            "asymetrie_declaree": "TSDiff a un budget regle par validation actif par actif, NsDiff un "
                                  "budget plat -- avantage TSDiff, herite du repo, non introduit ici.",
        },
        "per_model": per_model,
        "prices": "telecharges une fois puis geles dans prices/*.parquet, partages par les deux "
                  "modeles (yfinance resert avec ~2e-7 de bruit relatif d'un appel a l'autre)",
        "note": "artefact ISOLE : jamais ecrit dans tracking.db.",
        "elapsed_s": round(time.time() - t_start, 1),
    }, indent=2))
    print(f"\nTotal : {(time.time() - t_start) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
