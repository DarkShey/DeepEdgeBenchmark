"""
monthly_feasibility.py -- chantier C : ETUDE DE FAISABILITE du regime mensuel,
avant tout benchmark.

Le monthly n'a jamais ete teste et aucune conclusion daily/weekly n'y est
transposable : ce n'est pas le meme horizon, ce n'est surtout pas le meme volume
de donnees. Ce script repond, dans l'ordre : combien de donnees y a-t-il
vraiment, que deviennent `seq_len` et le budget d'epoques a ce volume, quelle
voie de generation tient le mieux sur un actif pilote, et le resultat passe-t-il
des criteres go/no-go DECLARES A PRIORI.

DEFINITION DU MOIS, choix declare et structurant : le mois est la DERNIERE
OBSERVATION HEBDOMADAIRE (grille W-FRI) de chaque mois calendaire. Consequence
voulue : la grille mensuelle est un SOUS-ENSEMBLE de la grille hebdomadaire,
donc les trois voies partagent exactement les memes origines et les memes
cibles. Definir le mois sur la derniere seance du mois (qui n'est pas un
vendredi) aurait rendu la voie (ii) incomparable aux deux autres -- elle aurait
vise des dates que le nuage hebdomadaire ne produit pas.

LES TROIS VOIES, sur le pilote (SPY, le mieux couvert du panier) :
  (i)   `monthly_native`      -- NsDiff entraine sur la serie MENSUELLE ;
  (ii)  `weekly_propagated`   -- NsDiff entraine sur la serie HEBDOMADAIRE puis
        propage jusqu'a la date-cible mensuelle (le nombre exact de pas
        hebdomadaires, 4 ou 5 selon le mois, jamais une moyenne) ;
  (iii) `synthetic_augmented` -- comme (i), mais les fenetres d'entrainement
        reelles sont completees par des fenetres tirees de series KernelSynth
        (`kernelsynth.py`). Declenchee par la condition du brief : "si (i)
        sous-entraine".

BASELINE, construite dans le meme mouvement comme l'exige le brief :
`garch_monthly` -- ARIMA-GARCH refit A CHAQUE ORIGINE sur la serie mensuelle
(son protocole naturel, via `benchmarks.multi_horizon.forecast_horizons_arima`,
importe tel quel). Un regime sans baseline classique n'est pas interpretable.

CRITERES GO / NO-GO, fixes AVANT d'avoir vu le moindre chiffre :
  1. couverture a 95 % du pilote DANS [0.90, 0.98] aux trois horizons ;
  2. Winkler NON significativement pire que `garch_monthly` (bootstrap par
     blocs apparie par origine, alpha=5 %).
Une voie qui echoue a l'un des deux est no-go. Si AUCUNE voie ne passe, le
mensuel sort du perimetre NsDiff et c'est documente -- ce qui est un resultat,
pas un echec de l'etude.

SELECTION D'HYPERPARAMETRES, hors grille de test : `seq_len` et le budget
d'epoques sont choisis par CRPS sur un bloc de validation strictement ANTERIEUR
a la premiere origine de test. Aucune des origines evaluees n'entre dans la
selection. C'est le point que le brief souligne ("le levier identifie des
l'etape 1 bis etait le budget d'entrainement par actif") et c'est aussi la
correction de la fuite qui avait entache la selection hebdomadaire de TSDiff.

PUISSANCE, declaree d'emblee : 36 origines de test, blocs de 3 -> `effective_n`
= 12. C'est trois fois moins que le corps du programme. Aucun "indistinguable"
de ce fichier ne doit etre lu comme une absence d'effet.

Sortie : experiments/monthly_feasibility.json
Usage :
    python monthly_feasibility.py --inventory-only     # inventaire seul, aucun fit
    python monthly_feasibility.py                       # pilote complet
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

import kernelsynth as ks                                              # noqa: E402
import stochvol_synth as sv                                           # noqa: E402
import nsdiff_model as nm                                             # noqa: E402
from benchmarks.multi_horizon import forecast_horizons_arima          # noqa: E402
from crps_metrics import crps_empirical                               # noqa: E402
from dashboard_d7_w1 import winkler_score                             # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly   # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "monthly_feasibility.json"
# Panel ETENDU (chantier A) par defaut, ancien repertoire en repli : le chantier D2
# du brief demande explicitement de rejouer le pilote « sur les donnees etendues »,
# c'est-a-dire sur ~183 mois au lieu de ~105-139 pour les actifs longs.
PRICE_DIR = Path(__file__).resolve().parent / "prices_v3"
PRICE_DIR_LEGACY = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "prices"

PILOT = "SPY"
SEEDS = [42, 43, 44, 45, 46]
N_TEST = 36                    # 3 ans d'origines mensuelles
N_VAL = 12                     # 1 an de validation, STRICTEMENT avant le test
HORIZONS = [1, 2, 3]           # M+1, M+2, M+3
N_SAMPLES = 200                # la reference actee
K_DENOISE = 20
BLOCK_LENGTH = 3
POOL_SEED = 42
TARGET_COVERAGE = 0.95

SEQ_LEN_GRID = (12, 30)        # 30 = defaut du repo ; 12 = variante adaptee au volume
# La grille d'epoques monte jusqu'a 640. Historique de son elargissement, qui suit
# a chaque fois la MEME regle declaree -- « argmin au bord => elargir une fois » :
#   * (20, 40, 80)                 : argmin a 80, au bord -> elargi ;
#   * (20, 40, 80, 160, 320)       : argmin a 160 sur `monthly_native`, a l'interieur
#                                    -> la grille etait suffisante A CE VOLUME DE DONNEES ;
#   * (20, 40, 80, 160, 320, 640)  : chantier D1 du brief extension/puissance. Le volume
#                                    d'entrainement DOUBLE (l'historique mensuel passe de
#                                    ~105-139 a ~183 observations), donc l'optimum se
#                                    deplace vers le haut : une grille calee sur l'ancien
#                                    volume redeviendrait tronquante. Elargie AVANT de
#                                    regarder le moindre resultat de test, comme les fois
#                                    precedentes.
# C'est le levier que le programme identifie depuis l'etape 1 bis : le budget
# d'entrainement par actif, particulierement critique quand les fenetres se comptent
# en dizaines.
EPOCH_GRID = (20, 40, 80, 160, 320, 640)
SEQ_LEN_WEEKLY = 30            # la serie hebdo est abondante : pas de contrainte de volume

# Voie (iii). SYNTH_RATIO est le rapport DECLARE fenetres synthetiques / fenetres
# reelles. A 5, le materiel synthetique domine sans noyer le signal reel ; laisser
# les 200 series produire toutes leurs fenetres donnerait un rapport de ~350:1,
# ce qui ne serait plus une augmentation mais un pre-entrainement synthetique --
# un autre chantier, avec d'autres questions.
N_SYNTH_POOL = 200             # taille du vivier de formes tirees
SYNTH_LENGTH = 160             # > au nombre de rendements mensuels reels disponibles
SYNTH_RATIO = 5

# Criteres go/no-go, DECLARES A PRIORI
GO_COVERAGE_MIN, GO_COVERAGE_MAX = 0.90, 0.98


# ── 1. grilles et inventaire ────────────────────────────────────────────────

def monthly_from_weekly(weekly: pd.Series) -> pd.Series:
    """Derniere observation hebdomadaire de chaque mois calendaire. La grille
    mensuelle est donc incluse dans la grille hebdomadaire (cf. docstring)."""
    idx = pd.to_datetime(weekly.index)
    keys = idx.to_period("M")
    last_pos = pd.Series(np.arange(len(weekly)), index=keys).groupby(level=0).max()
    return weekly.iloc[last_pos.values]


def inventory(assets: list) -> dict:
    """Combien de donnees, et combien de FENETRES D'ENTRAINEMENT il en reste --
    la seule quantite qui compte pour un modele a fenetre glissante."""
    out = {}
    for asset in assets:
        frozen = PRICE_DIR / f"{asset.replace('=', '_')}.parquet"
        if not frozen.exists():
            frozen = PRICE_DIR_LEGACY / f"{asset.replace('=', '_')}.parquet"
        if not frozen.exists():
            out[asset] = {"status": "prix geles absents"}
            continue
        daily = pd.read_parquet(frozen)["close"]
        weekly, _ = build_weekly(daily)
        monthly = monthly_from_weekly(weekly)
        n_train_months = len(monthly) - N_TEST
        entry = {
            "n_daily": int(len(daily)), "n_weekly": int(len(weekly)),
            "n_monthly": int(len(monthly)),
            "monthly_span": f"{monthly.index[0].date()} -> {monthly.index[-1].date()}",
            "n_train_months_before_test": int(n_train_months),
            "training_windows_by_seq_len": {},
        }
        for seq_len in SEQ_LEN_GRID:
            n_returns = n_train_months - 1
            n_win = max(0, n_returns - seq_len - max(HORIZONS) + 1)
            entry["training_windows_by_seq_len"][seq_len] = {
                "monthly": int(n_win),
                "weekly_equivalent": int(max(0, (len(weekly) - N_TEST * 4) - 1 - seq_len - 15 + 1)),
                "ratio_weekly_over_monthly": (round((len(weekly) - 1 - seq_len - 15 + 1) / n_win, 1)
                                              if n_win > 0 else None),
            }
        out[asset] = entry
    return out


def build_grid(asset: str) -> dict:
    frozen = PRICE_DIR / f"{asset.replace('=', '_')}.parquet"
    if not frozen.exists():
        frozen = PRICE_DIR_LEGACY / f"{asset.replace('=', '_')}.parquet"
    daily = pd.read_parquet(frozen)["close"]
    weekly, _ = build_weekly(daily)
    monthly = monthly_from_weekly(weekly)
    # position de chaque date mensuelle dans la grille hebdomadaire : indispensable
    # a la voie (ii), qui doit propager le NOMBRE EXACT de pas hebdomadaires
    weekly_pos = {d: i for i, d in enumerate(weekly.index)}
    month_weekly_pos = np.array([weekly_pos[d] for d in monthly.index])
    n = len(monthly)
    test_origins = list(range(n - max(HORIZONS) - N_TEST, n - max(HORIZONS)))
    val_origins = list(range(test_origins[0] - N_VAL, test_origins[0]))
    if val_origins[0] <= max(SEQ_LEN_GRID) + max(HORIZONS):
        raise SystemExit(f"[{asset}] historique mensuel insuffisant pour un bloc de validation "
                         "hors grille de test")
    return {"weekly": weekly, "monthly": monthly, "month_weekly_pos": month_weekly_pos,
            "test_origins": test_origins, "val_origins": val_origins}


# ── 2. les trois voies ──────────────────────────────────────────────────────

def _standardise(series: pd.Series) -> tuple:
    r = nm._log_returns(series.values.astype(float))
    mu, sd = float(r.mean()), float(r.std())
    return (r - mu) / (sd if sd > 1e-8 else 1.0), mu, (sd if sd > 1e-8 else 1.0)


def fit_windows(H_win, T_win, seq_len: int, horizon: int, epochs: int, seed: int):
    """Fit NsDiff sur des fenetres DEJA construites -- necessaire pour la voie
    (iii), ou les fenetres reelles et synthetiques sont concatenees. Le reste du
    chemin (architecture, optimiseur, boucle d'entrainement) est celui de
    `nsdiff_model.fit_nsdiff`, appele au meme grain : seule la provenance des
    fenetres change."""
    nm.set_seed(seed)
    model = nm.NsDiff(seq_len, horizon, nm.HIDDEN_MEAN, nm.HIDDEN_SIGMA,
                      nm.HIDDEN_DENOISE, nm.SIGMA_KERNEL, T=K_DENOISE)
    model.train(H_win, T_win, epochs=epochs, batch_size=nm.BATCH_SIZE)
    return model


def real_windows(z: np.ndarray, seq_len: int, horizon: int):
    return nm._make_windows(z, seq_len, horizon)


def stochvol_windows(seq_len: int, horizon: int, n_target: int, seed: int = 0,
                     n_series: int = N_SYNTH_POOL, length: int = SYNTH_LENGTH):
    """Voie (iv), chantier D3 : memes fenetres, meme ratio, meme protocole que la
    voie (iii) -- SEUL LE GENERATEUR change. C'est la condition pour que la
    comparaison entre les deux dise quelque chose sur les generateurs et non sur
    le protocole d'augmentation.

    `stochvol_synth` remplace `kernelsynth` parce que ce dernier produit des
    series dont les NIVEAUX sont autocorreles a +0,45 la ou un rendement reel est
    blanc (+0,01) : on apprenait au modele qu'un rendement se predit par le
    precedent. Le nouveau generateur est calibre sur les faits stylises mesures
    des 7 series mensuelles du panel."""
    series, params = sv.generate(n_series, length, seed=seed)
    H, T = [], []
    for s_ in series:
        h, t = nm._make_windows(s_, seq_len, horizon)
        if len(h):
            H.append(h)
            T.append(t)
    H, T = np.concatenate(H), np.concatenate(T)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(H), size=min(n_target, len(H)), replace=False)
    return H[idx], T[idx], params


def synthetic_windows(seq_len: int, horizon: int, n_target: int, seed: int = 0,
                      n_series: int = N_SYNTH_POOL, length: int = SYNTH_LENGTH):
    """`n_target` fenetres tirees uniformement dans un vivier de `n_series`
    series KernelSynth. Les series sont deja centrees reduites, donc directement
    dans l'espace des rendements standardises du modele -- aucune remise a
    l'echelle a partir de l'actif reel, qui ferait fuiter sa volatilite dans le
    materiel synthetique.

    Le vivier est large (diversite de formes) mais l'echantillon prelevé est
    calibre sur le volume reel (`SYNTH_RATIO`) : c'est une augmentation, pas un
    pre-entrainement deguise."""
    series, descs = ks.generate(n_series, length, seed=seed)
    H, T = [], []
    for s in series:
        h, t = nm._make_windows(s, seq_len, horizon)
        if len(h):
            H.append(h)
            T.append(t)
    H, T = np.concatenate(H), np.concatenate(T)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(H), size=min(n_target, len(H)), replace=False)
    return H[idx], T[idx], descs


def forecast_monthly(model, z_full: np.ndarray, mu: float, sd: float, grid: dict,
                     origins: list, seed: int, native: bool) -> pd.DataFrame:
    """Previsions aux origines demandees. `native=True` : le modele vit sur la
    grille mensuelle, h = 1/2/3 mois. `native=False` (voie ii) : il vit sur la
    grille hebdomadaire et h = nombre EXACT de pas hebdomadaires jusqu'a la date
    cible mensuelle (4 ou 5 selon le mois)."""
    monthly, wpos = grid["monthly"], grid["month_weekly_pos"]
    rows = []
    for k, m in enumerate(origins):
        last_price = float(monthly.iloc[m])
        if native:
            hist_end, horizons, keymap = m, HORIZONS, {h: h for h in HORIZONS}
        else:
            hist_end = int(wpos[m])
            steps = [int(wpos[m + h] - wpos[m]) for h in HORIZONS]
            horizons, keymap = steps, dict(zip(HORIZONS, steps))
        nm.set_seed(seed + k)
        samples = nm.forecast_from_fitted(model, z_full[:hist_end], mu, sd, last_price,
                                          horizons=horizons, n_samples=N_SAMPLES)
        for h in HORIZONS:
            s = samples[keymap[h]]
            lo, hi = (float(q) for q in np.quantile(s, [0.025, 0.975]))
            rows.append({
                "origin": int(m), "horizon": h, "horizon_unit": f"M+{h}",
                "cutoff_date": str(monthly.index[m].date()),
                "target_date": str(monthly.index[m + h].date()),
                "last_close": last_price, "y_pred": float(np.mean(s)),
                "y_lower": lo, "y_upper": hi, "y_true": float(monthly.iloc[m + h]),
                "crps": crps_empirical(s, float(monthly.iloc[m + h])),
            })
    return pd.DataFrame(rows)


def run_voie(voie: str, grid: dict, origins: list, seq_len: int, epochs: int,
             seed: int, train_end: int) -> pd.DataFrame:
    """`train_end` : indice mensuel avant lequel (STRICTEMENT) l'entrainement a
    lieu. Vaut la premiere origine de validation pour le sweep, la premiere
    origine de test pour le run final."""
    monthly, weekly, wpos = grid["monthly"], grid["weekly"], grid["month_weekly_pos"]
    if voie == "weekly_propagated":
        w_end = int(wpos[train_end])
        z_full, mu, sd = _standardise(weekly)
        z_train = z_full[:w_end - 1]
        horizon = int(max(wpos[o + max(HORIZONS)] - wpos[o] for o in origins))
        H, T = real_windows(z_train, SEQ_LEN_WEEKLY, horizon)
        model = fit_windows(H, T, SEQ_LEN_WEEKLY, horizon, epochs, seed)
        return forecast_monthly(model, z_full, mu, sd, grid, origins, seed, native=False)

    z_full, mu, sd = _standardise(monthly)
    z_train = z_full[:train_end - 1]
    horizon = max(HORIZONS)
    H, T = real_windows(z_train, seq_len, horizon)
    if voie == "synthetic_augmented":
        Hs, Ts, _ = synthetic_windows(seq_len, horizon, n_target=SYNTH_RATIO * len(H), seed=seed)
        H, T = np.concatenate([H, Hs]), np.concatenate([T, Ts])
    elif voie == "stochvol_augmented":
        Hs, Ts, _ = stochvol_windows(seq_len, horizon, n_target=SYNTH_RATIO * len(H), seed=seed)
        H, T = np.concatenate([H, Hs]), np.concatenate([T, Ts])
    model = fit_windows(H, T, seq_len, horizon, epochs, seed)
    return forecast_monthly(model, z_full, mu, sd, grid, origins, seed, native=True)


def run_garch_monthly(grid: dict, origins: list) -> pd.DataFrame:
    """Refit A CHAQUE ORIGINE sur la serie mensuelle -- le protocole naturel du
    modele classique, celui qu'il a partout ailleurs dans le repo."""
    monthly = grid["monthly"]
    rows = []
    for m in origins:
        train = monthly.iloc[:m + 1]
        fc = forecast_horizons_arima(train, HORIZONS)
        for h in HORIZONS:
            point, lo, hi = fc[h]
            rows.append({
                "origin": int(m), "horizon": h, "horizon_unit": f"M+{h}",
                "cutoff_date": str(monthly.index[m].date()),
                "target_date": str(monthly.index[m + h].date()),
                "last_close": float(monthly.iloc[m]), "y_pred": point,
                "y_lower": lo, "y_upper": hi, "y_true": float(monthly.iloc[m + h]),
                "crps": float("nan"),      # bornes analytiques : pas de nuage a scorer
            })
    return pd.DataFrame(rows)


# ── 3. metriques, selection, go/no-go ───────────────────────────────────────

def metrics(df: pd.DataFrame) -> dict:
    inside = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    w = winkler_score(df["y_true"], df["y_lower"], df["y_upper"])
    return {
        "n": int(len(df)),
        "cov95": float(inside.mean()),
        "pi_width_pct_of_price": float(((df["y_upper"] - df["y_lower"]) / df["last_close"]).mean() * 100),
        "winkler_mean": float(np.mean(w)),
        "rmse": float(np.sqrt(((df["y_pred"] - df["y_true"]) ** 2).mean())),
        "crps_mean": (float(df["crps"].mean()) if df["crps"].notna().any() else None),
    }


def seed_pool(frames: list) -> pd.DataFrame:
    """Metriques moyennees sur les graines, origine par origine -- la convention
    de pooling du programme."""
    keys = ["origin", "horizon", "horizon_unit", "cutoff_date", "target_date"]
    cat = pd.concat(frames, ignore_index=True)
    agg = cat.groupby(keys, as_index=False).agg(
        last_close=("last_close", "first"), y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"), y_lower=("y_lower", "mean"),
        y_upper=("y_upper", "mean"), crps=("crps", "mean"))
    return agg.sort_values(["horizon", "origin"]).reset_index(drop=True)


def sweep(voie: str, grid: dict) -> dict:
    """Selection (seq_len, epochs) par CRPS sur le bloc de validation, STRICTEMENT
    anterieur a la grille de test. `weekly_propagated` ne balaye que les epoques :
    sa serie d'entrainement est abondante, `seq_len` n'y est pas contraint."""
    val, train_end = grid["val_origins"], grid["val_origins"][0]
    seq_grid = (SEQ_LEN_WEEKLY,) if voie == "weekly_propagated" else SEQ_LEN_GRID
    records = []
    for seq_len in seq_grid:
        for epochs in EPOCH_GRID:
            frames = [run_voie(voie, grid, val, seq_len, epochs, s, train_end) for s in SEEDS]
            m = metrics(seed_pool(frames))
            records.append({"seq_len": seq_len, "epochs": epochs, **m})
            print(f"  [{voie}] seq_len={seq_len:<3} epochs={epochs:<3} "
                  f"CRPS_val={m['crps_mean']:.4f}  Cov95_val={m['cov95']:.3f}  "
                  f"largeur={m['pi_width_pct_of_price']:.2f}%")
    best = min(records, key=lambda r: r["crps_mean"])
    print(f"  [{voie}] RETENU : seq_len={best['seq_len']}, epochs={best['epochs']} "
          f"(argmin CRPS_val)")
    return {"records": records, "selected": {"seq_len": best["seq_len"], "epochs": best["epochs"]},
            "criterion": "argmin CRPS sur le bloc de validation (12 origines strictement "
                         "anterieures a la premiere origine de test)"}


def go_no_go(voie_df: pd.DataFrame, garch_df: pd.DataFrame) -> dict:
    """Les deux criteres declares a priori, evalues horizon par horizon."""
    per_h, coverage_ok, winkler_ok = {}, True, True
    for h in HORIZONS:
        a = voie_df[voie_df["horizon"] == h].sort_values("origin").reset_index(drop=True)
        b = garch_df[garch_df["horizon"] == h].sort_values("origin").reset_index(drop=True)
        if not (a["target_date"].values == b["target_date"].values).all():
            raise SystemExit(f"M+{h} : origines desalignees entre la voie et la baseline")
        wa = winkler_score(a["y_true"], a["y_lower"], a["y_upper"])
        wb = winkler_score(b["y_true"], b["y_lower"], b["y_upper"])
        t = paired_block_bootstrap_test(np.asarray(wa) - np.asarray(wb),
                                        block_length=BLOCK_LENGTH, seed=POOL_SEED)
        cov = float(((a["y_true"] >= a["y_lower"]) & (a["y_true"] <= a["y_upper"])).mean())
        cov_ok = GO_COVERAGE_MIN <= cov <= GO_COVERAGE_MAX
        wk_ok = not (t["significant_at_05"] and t["mean_diff"] > 0)   # pire = Winkler plus grand
        coverage_ok &= cov_ok
        winkler_ok &= wk_ok
        per_h[f"M+{h}"] = {
            "cov95": cov, "coverage_in_band": cov_ok,
            "winkler_voie": float(np.mean(wa)), "winkler_garch": float(np.mean(wb)),
            "winkler_test_vs_garch": {**t, "voie_significantly_worse": bool(
                t["significant_at_05"] and t["mean_diff"] > 0)},
            "winkler_not_worse": wk_ok,
        }
    return {"per_horizon": per_h, "criterion_coverage": coverage_ok,
            "criterion_winkler": winkler_ok, "verdict": "GO" if (coverage_ok and winkler_ok) else "NO-GO"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", default=PILOT)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--voies", nargs="+",
                   default=["monthly_native", "weekly_propagated", "synthetic_augmented"],
                   help="voies à évaluer. `stochvol_augmented` = chantier D3 (générateur à "
                        "volatilité stochastique, remplaçant de KernelSynth).")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    inv = inventory(args.assets)
    print("=== inventaire donnees ===")
    for asset, e in inv.items():
        if "n_monthly" not in e:
            print(f"  {asset:<9} {e.get('status')}")
            continue
        w30 = e["training_windows_by_seq_len"][30]
        print(f"  {asset:<9} {e['n_daily']:>5} quotid. | {e['n_weekly']:>4} hebdo | "
              f"{e['n_monthly']:>3} mensuelles ({e['monthly_span']}) | "
              f"fenetres d'entrainement a seq_len=30 : {w30['monthly']} "
              f"(x{w30['ratio_weekly_over_monthly']} de moins qu'en hebdo)")

    payload = {
        "scope": "chantier C -- faisabilite du regime mensuel, avant tout benchmark",
        "month_definition": "derniere observation hebdomadaire (grille W-FRI) de chaque mois "
                            "calendaire -- la grille mensuelle est incluse dans l'hebdomadaire, "
                            "donc les trois voies partagent origines et cibles",
        "inventory": inv,
        "protocol": {
            "pilot": args.pilot, "n_test_origins": N_TEST, "n_val_origins": N_VAL,
            "horizons": [f"M+{h}" for h in HORIZONS], "seeds": args.seeds,
            "n_samples": N_SAMPLES, "block_length": BLOCK_LENGTH,
            "effective_n": N_TEST // BLOCK_LENGTH,
            "hyperparameter_selection": "seq_len et epoques par argmin CRPS sur un bloc de "
                                        "validation STRICTEMENT anterieur a la grille de test",
            "synthetic_augmentation": {
                "generator": "kernelsynth.py (recette Chronos : composition aleatoire de noyaux "
                             "GP, tirage dans le prior)",
                "pool": N_SYNTH_POOL, "series_length": SYNTH_LENGTH,
                "ratio_synthetic_over_real_windows": SYNTH_RATIO,
                "caveat": "une serie KernelSynth n'a ni queues lourdes ni clustering de "
                          "volatilite : elle apporte de la diversite de FORMES, pas du realisme "
                          "stylise",
            },
            "go_no_go": {
                "coverage_band": [GO_COVERAGE_MIN, GO_COVERAGE_MAX],
                "winkler": "non significativement pire que garch_monthly (bootstrap par blocs, 5 %)",
                "declared": "avant tout resultat",
            },
        },
    }

    if args.inventory_only:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n--inventory-only -> {args.out}")
        return

    grid = build_grid(args.pilot)
    test, val = grid["test_origins"], grid["val_origins"]
    monthly = grid["monthly"]
    print(f"\n=== pilote {args.pilot} ===")
    print(f"  validation : {len(val)} origines {monthly.index[val[0]].date()} -> "
          f"{monthly.index[val[-1]].date()}")
    print(f"  test       : {len(test)} origines {monthly.index[test[0]].date()} -> "
          f"{monthly.index[test[-1]].date()} (jamais touchees par la selection)")

    print(f"\n=== baseline garch_monthly (refit a chaque origine) ===")
    garch = run_garch_monthly(grid, test)
    print(f"  {metrics(garch)}")

    sweeps, results, verdicts = {}, {"garch_monthly": metrics(garch)}, {}
    per_horizon = {"garch_monthly": {f"M+{h}": metrics(garch[garch['horizon'] == h])
                                     for h in HORIZONS}}
    for voie in args.voies:
        print(f"\n=== {voie} : sweep sur validation (hors grille de test) ===")
        sw = sweep(voie, grid)
        sweeps[voie] = sw
        seq_len, epochs = sw["selected"]["seq_len"], sw["selected"]["epochs"]
        frames = [run_voie(voie, grid, test, seq_len, epochs, s, test[0]) for s in args.seeds]
        pooled = seed_pool(frames)
        results[voie] = metrics(pooled)
        per_horizon[voie] = {f"M+{h}": metrics(pooled[pooled["horizon"] == h]) for h in HORIZONS}
        verdicts[voie] = go_no_go(pooled, garch)
        print(f"  test : {results[voie]}")
        print(f"  go/no-go : {verdicts[voie]['verdict']} "
              f"(couverture {verdicts[voie]['criterion_coverage']}, "
              f"Winkler {verdicts[voie]['criterion_winkler']})")

    payload.update({"sweeps": sweeps, "results": results, "per_horizon": per_horizon,
                    "go_no_go": verdicts,
                    "overall": ("GO" if any(v["verdict"] == "GO" for v in verdicts.values())
                                else "NO-GO")})
    payload["elapsed_min"] = round((time.time() - t0) / 60, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== synthese pilote ===")
    print(f"{'voie':<24}{'Cov95':>8}{'largeur %':>12}{'Winkler':>12}{'RMSE':>10}{'CRPS':>10}{'verdict':>10}")
    for name, m in results.items():
        v = verdicts.get(name, {}).get("verdict", "baseline")
        crps = f"{m['crps_mean']:.4f}" if m["crps_mean"] is not None else "n/a"
        print(f"{name:<24}{m['cov95']:>8.3f}{m['pi_width_pct_of_price']:>12.2f}"
              f"{m['winkler_mean']:>12.4g}{m['rmse']:>10.4g}{crps:>10}{v:>10}")
    print(f"\nVERDICT GLOBAL DU CADRAGE MONTHLY : {payload['overall']}")
    print(f"-> {args.out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
