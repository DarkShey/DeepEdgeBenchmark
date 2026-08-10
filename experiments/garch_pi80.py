"""
garch_pi80.py -- chantier 1.3 : la bande de prevision d'ARIMA-GARCH au niveau
80 %, sur les memes 90 origines `oos` et dans les deux regimes.

POURQUOI ELLE MANQUE. La famille 3 du chantier B (« prendre position seulement si
l'intervalle exclut le rendement nul ») n'a jamais ouvert une position a 95 % :
sur 2 700 origines et pour les deux modeles, l'intervalle contient toujours le
prix courant. Le brief rouvre la famille a un niveau declare a priori -- 80 %, et
lui seul. Cote NsDiff c'est immediat (quantiles 10/90 du nuage deja produit).
Cote GARCH, `tracking.db` ne stocke que la bande a 95 %.

LA LOI D'INNOVATION, etablie par MESURE et non par lecture du code. `arima_model`
declare aujourd'hui `GARCH_DIST = "skewt"`, mais les lignes `oos` datent du run
`20260717-weekly-multimodel`. Reproduire ces lignes avec chaque loi candidate
departage sans ambiguite (SPY, 8 origines, ecart relatif max sur les bornes) :

    dist="normal"  ->  1.45e-06      <- reproduit la base
    dist="t"       ->  2.71e-03
    dist="ged"     ->  2.80e-03
    dist="skewt"   ->  2.28e-02

Le bras GARCH du benchmark est donc la variante a innovation GAUSSIENNE.
(Cela valide au passage, a posteriori, la reconstruction gaussienne des
trajectoires GARCH du chantier B3 : ce n'etait pas une approximation commode
mais la loi effectivement ajustee.)

DEUX CHEMINS, ET POURQUOI ON PREND CELUI-LA.

  * REGENERER integralement (refit par origine, `forecast_horizons_arima` avec
    son kwarg `level`) reproduit la base a 3e-7 en mediane... mais pas partout :
    9 lignes SPY sur 540 devient jusqu'a 1,6e-3, toutes groupees sur trois
    origines d'avril 2025. Sur ces lignes le POINT bouge aussi -- ce n'est donc
    pas un desaccord de bande mais un optimum local different atteint par
    l'optimiseur d'ARIMA/GARCH sur une fenetre d'entrainement traversant un
    episode de volatilite extreme. Rien ne dit que la version regeneree est
    « moins bonne » ; simplement, elle n'est plus LA ligne publiee, et un
    chantier qui compare des modeles ne peut pas se permettre de changer
    silencieusement l'un des deux bras.

  * DERIVER la bande a 80 % de la bande a 95 % DEJA EN BASE. La loi etant
    gaussienne, la bande est exactement log-symetrique autour du point (verifie a
    la precision machine sur 100 % des lignes) et le passage d'un niveau a
    l'autre est un simple facteur sur la demi-largeur en log :

        lo80 = y_pred * (lo95 / y_pred) ** r,   r = z(0.90) / Z_95 = 0.653853
        hi80 = y_pred * (hi95 / y_pred) ** r

C'est le second chemin qui est retenu : il s'ancre sur les lignes publiees, donc
le bras GARCH reste bit-pour-bit celui du benchmark.

LES GARDE-FOUS, deux, et leur ordre compte.

  (A) TEST DU RATIO -- le principal. La derivation repose sur une seule
      affirmation : le passage de 95 % a 80 % est le facteur constant
      z(0.90)/Z_95 sur la demi-largeur en log. C'est une propriete de la
      FONCTION QUANTILE, pas du fit. On la mesure donc directement sur les
      sorties regenerees : si l'optimiseur a derive, `point`, `hi80` et `hi95`
      bougent ENSEMBLE et le ratio ne bouge pas. Ce test couvre donc 100 % des
      lignes. Mesure : ecart maximal 1.25e-14 -- la precision machine.
  (B) TEST ABSOLU -- comparaison directe des bandes a 80 %, derivee contre
      regeneree, mais seulement sur les lignes ou la regeneration reproduit
      d'abord la bande a 95 %. Plus intuitif, et AVEUGLE la ou l'optimiseur
      derive systematiquement : au premier run, il ne couvrait aucune ligne de
      TLT/weekly. C'est ce trou qui a motive le test (A), qui le comble.

Le test (A) a par ailleurs attrape une vraie erreur : le denominateur du ratio
doit etre `arima_model.Z_95` (1.96, le multiplicateur avec lequel la bande a
95 % en base a ete construite) et non `norm.ppf(0.975)` (1.959964). Cf.
`level_ratio`.

Sortie : experiments/garch_pi80/bands.parquet (+ config.json)
Usage :
    python garch_pi80.py                    # derive + valide (regenere), tous actifs
    python garch_pi80.py --no-validate      # derive seulement (instantane)
    python garch_pi80.py --validate-assets SPY   # valide sur un sous-ensemble
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import arima_model                                                   # noqa: E402
from benchmarks.multi_horizon import forecast_horizons_arima          # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH, HORIZON_UNITS, _weekly_position  # noqa: E402
from epoch_sweep import week_targets                                  # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly   # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "garch_pi80"
PRICE_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "prices"
MODEL = "ARIMA-GARCH"
DIST = "normal"                 # etabli par mesure, cf. docstring
TARGET_LEVEL = 0.80
BASE_LEVEL = 0.95
# ecart relatif au prix en deca duquel une ligne regeneree est jugee FIDELE a la
# base -- au-dela, l'optimiseur a change de chemin et la ligne est ecartee de la
# validation (elle n'est pas ecartee du jeu de donnees : celui-ci vient de la base).
FIDELITY_TOL = 5e-5
VALIDATION_TOL = 2e-4           # ecart tolere entre derivation et regeneration (test B)
RATIO_TOL = 1e-9                # ecart tolere sur le ratio de niveau (test A) -- exact en theorie


def level_ratio(target: float = TARGET_LEVEL) -> float:
    """r = z((1+target)/2) / Z_95 -- facteur de passage du niveau 95 % au niveau
    cible, sur la demi-largeur EN LOG, pour une innovation gaussienne.

    LE DENOMINATEUR EST `arima_model.Z_95` (= 1.96), PAS `norm.ppf(0.975)`
    (= 1.959964). Ce n'est pas un detail cosmetique : c'est le multiplicateur
    avec lequel la bande a 95 % EN BASE a reellement ete construite. Le repo
    documente cet arrondi (`forecast_from_fitted_arima` : "aucun risque de derive
    numerique 1.96 vs norm.ppf(0.975)") et le chemin `dist='normal', level=0.95`
    l'emprunte, alors que tout autre niveau passe par la fonction quantile exacte.

    Prendre `norm.ppf(0.975)` au denominateur produisait un ratio faux de
    1,2e-05 -- ecart minuscule, sans effet sur aucune conclusion (il deplace la
    bande de ~1e-06 du prix), mais que le test du ratio a detecte immediatement.
    Il est corrige plutot que tolere : une derivation dite exacte qui ne l'est
    qu'a 1e-05 pres n'est pas exacte, et la tolerance du test doit rester serree
    pour continuer a detecter un vrai changement de loi.
    """
    return float(stats.norm.ppf(0.5 + target / 2.0) / arima_model.Z_95)


def derive_band(y_pred, y_lower, y_upper, ratio: float):
    """Bande au niveau cible, derivee de la bande a 95 % publiee. Exacte sous
    innovation gaussienne ; le garde-fou ci-dessous verifie qu'elle l'est."""
    p = np.asarray(y_pred, dtype=float)
    lo = p * (np.asarray(y_lower, dtype=float) / p) ** ratio
    hi = p * (np.asarray(y_upper, dtype=float) / p) ** ratio
    return lo, hi


def load_oos(assets: list, db_path: str = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT asset, frequence, horizon_type, horizon_unit, cutoff_date, target_date,
                   last_close, y_pred, y_lower, y_upper, y_true
            FROM predictions
            WHERE source='oos' AND model=? AND horizon_type='weekly'
                  AND asset IN ({}) AND y_true IS NOT NULL
            """.format(",".join("?" * len(assets))), con, params=[MODEL, *assets])
    finally:
        con.close()
    return df.sort_values(["asset", "frequence", "horizon_unit", "cutoff_date"]).reset_index(drop=True)


def check_log_symmetry(df: pd.DataFrame) -> dict:
    """La derivation suppose une bande log-symetrique autour du point. Verifie,
    pas suppose."""
    mid = (np.log(df["y_lower"]) + np.log(df["y_upper"])) / 2.0
    dev = np.abs(mid - np.log(df["y_pred"]))
    return {"max_abs_log_deviation": float(dev.max()), "median": float(dev.median()),
            "is_log_symmetric": bool(dev.max() < 1e-9)}


def regenerate_asset(asset: str, oos: pd.DataFrame) -> pd.DataFrame:
    """Regeneration independante (refit par origine) des bandes 95 % ET 80 %.
    Sert UNIQUEMENT de garde-fou -- jamais de source de donnees."""
    frozen = PRICE_DIR / f"{asset.replace('=', '_')}.parquet"
    if not frozen.exists():
        raise SystemExit(f"prix geles absents pour {asset} -- refus de retelecharger")
    daily = pd.read_parquet(frozen)["close"]
    weekly, weekly_dates = build_weekly(daily)
    sub = oos[oos["asset"] == asset]
    rows = []
    cutoffs = sorted(sub["cutoff_date"].unique())
    for k, cutoff in enumerate(cutoffs):
        m = _weekly_position(weekly_dates, cutoff)
        _, daily_pos, _, daily_horizons = week_targets(weekly_dates, daily, m)
        for regime, train, horizons in (("weekly", weekly.iloc[:m + 1], [1, 2, 3]),
                                        ("daily", daily.iloc[:daily_pos + 1], daily_horizons)):
            res = {lvl: forecast_horizons_arima(train, horizons, dist=DIST, level=lvl)
                   for lvl in (BASE_LEVEL, TARGET_LEVEL)}
            for wi, h in enumerate(horizons):
                point, lo95, hi95 = res[BASE_LEVEL][h]
                _, lo80, hi80 = res[TARGET_LEVEL][h]
                rows.append({"asset": asset, "frequence": regime,
                             "horizon_unit": HORIZON_UNITS[wi + 1], "cutoff_date": cutoff,
                             "regen_pred": float(point), "regen_lower": float(lo95),
                             "regen_upper": float(hi95), "regen_lower80": float(lo80),
                             "regen_upper80": float(hi80)})
        if (k + 1) % 30 == 0 or k == len(cutoffs) - 1:
            print(f"  [{asset}] validation, origine {k + 1}/{len(cutoffs)}")
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, regen: pd.DataFrame, ratio: float) -> dict:
    """Deux garde-fous, dans cet ordre d'importance.

    (A) TEST DU RATIO -- le test principal, et le seul qui couvre 100 % des
    lignes. La derivation repose sur une unique affirmation : pour ce modele, le
    passage du niveau 95 % au niveau 80 % est le facteur constant
    z(0.90)/z(0.975) sur la demi-largeur en log. C'est une propriete de la
    FONCTION QUANTILE de la loi d'innovation, pas du fit. On la mesure donc
    DIRECTEMENT sur les sorties regenerees :

        r_mesure = log(hi80 / point) / log(hi95 / point)

    Si l'optimiseur a derive, `point`, `hi80` et `hi95` bougent ENSEMBLE et le
    ratio ne bouge pas. Ce test est donc immune a la derive -- et c'est pour cela
    qu'il est le principal : le test (B), lui, doit ecarter les lignes derivees,
    ce qui laissait TLT/weekly sans aucune ligne de controle (constate au premier
    run, et c'est ce qui a motive ce test-ci).

    (B) TEST ABSOLU -- comparaison directe des bandes a 80 %, derivee contre
    regeneree, sur les seules lignes ou la regeneration reproduit d'abord la
    bande a 95 %. Plus intuitif, mais aveugle la ou l'optimiseur derive
    systematiquement. Conserve comme confirmation, pas comme condition.
    """
    keys = ["asset", "frequence", "horizon_unit", "cutoff_date"]
    m = df.merge(regen, on=keys, how="inner")
    if m.empty:
        return {"status": "no_overlap"}

    # ── (A) test du ratio, sur toutes les lignes ────────────────────────────
    r_hi = np.log(m["regen_upper80"] / m["regen_pred"]) / np.log(m["regen_upper"] / m["regen_pred"])
    r_lo = np.log(m["regen_lower80"] / m["regen_pred"]) / np.log(m["regen_lower"] / m["regen_pred"])
    dev = np.abs(np.r_[r_hi, r_lo] - ratio)
    ratio_by_cell = {}
    for (a, rg), g in m.groupby(["asset", "frequence"]):
        rh = np.log(g["regen_upper80"] / g["regen_pred"]) / np.log(g["regen_upper"] / g["regen_pred"])
        rl = np.log(g["regen_lower80"] / g["regen_pred"]) / np.log(g["regen_lower"] / g["regen_pred"])
        ratio_by_cell[f"{a}|{rg}"] = {"n": int(len(g)),
                                      "max_abs_dev_from_expected_ratio":
                                          float(np.abs(np.r_[rh, rl] - ratio).max())}
    ratio_ok = bool(dev.max() <= RATIO_TOL)

    # ── (B) test absolu, sur les lignes fideles ─────────────────────────────
    rel = lambda a, b: np.abs(m[a] - m[b]) / m["last_close"]
    faithful = ((rel("y_pred", "regen_pred") < FIDELITY_TOL)
                & (rel("y_lower", "regen_lower") < FIDELITY_TOL)
                & (rel("y_upper", "regen_upper") < FIDELITY_TOL))
    d_lo, d_hi = rel("y_lower80", "regen_lower80"), rel("y_upper80", "regen_upper80")
    worst = float(max(d_lo[faithful].max(), d_hi[faithful].max())) if faithful.any() else float("nan")
    m = m.assign(_faithful=faithful)
    cover = (m.groupby(["asset", "frequence"])["_faithful"]
             .agg(n_total="size", n_faithful="sum").reset_index())
    cover["pct_faithful"] = (cover["n_faithful"] / cover["n_total"] * 100).round(1)

    out = {
        "status": "validated",
        "primary_ratio_test": {
            "expected_ratio": ratio,
            "max_abs_deviation": float(dev.max()),
            "median_abs_deviation": float(np.median(dev)),
            "tolerance": RATIO_TOL, "passes": ratio_ok,
            "n_rows_covered": int(len(m)), "coverage": "100 % des lignes -- immune a la derive",
            "by_asset_regime": ratio_by_cell,
        },
        "secondary_absolute_test": {
            "n_compared": int(len(m)), "n_faithful_at_95": int(faithful.sum()),
            "n_optimizer_drift": int((~faithful).sum()),
            "coverage_by_asset_regime": cover.to_dict(orient="records"),
            "cells_without_any_faithful_row": [
                f"{r.asset}|{r.frequence}" for r in cover.itertuples() if r.n_faithful == 0],
            "max_rel_diff": worst,
            "median_rel_diff": (float(np.median(np.r_[d_lo[faithful], d_hi[faithful]]))
                                if faithful.any() else float("nan")),
            "tolerance": VALIDATION_TOL,
            "passes": bool(np.isnan(worst) or worst <= VALIDATION_TOL),
            "note": "aveugle la ou l'optimiseur derive systematiquement -- c'est le test du "
                    "ratio qui fait foi",
        },
    }
    if not ratio_ok:
        raise SystemExit(f"le ratio de niveau mesure devie de {dev.max():.2e} (tolerance "
                         f"{RATIO_TOL:.0e}) -- l'innovation n'est pas celle supposee, "
                         "la derivation analytique est invalidee.")
    if not out["secondary_absolute_test"]["passes"]:
        raise SystemExit(f"bande 80 % derivee vs regeneree : ecart {worst:.2e} > {VALIDATION_TOL:.0e}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()))
    p.add_argument("--validate-assets", nargs="+", default=None,
                   help="actifs sur lesquels tourner la régénération de contrôle (défaut : tous)")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    t0 = time.time()
    ratio = level_ratio()
    oos = load_oos(args.assets)
    sym = check_log_symmetry(oos)
    print(f"{len(oos)} lignes oos {MODEL} (lecture seule)")
    print(f"log-symétrie de la bande 95 % : écart max {sym['max_abs_log_deviation']:.2e} "
          f"-> {'OK' if sym['is_log_symmetric'] else 'ÉCHEC'}")
    if not sym["is_log_symmetric"]:
        raise SystemExit("bande non log-symétrique : la dérivation analytique ne s'applique pas.")

    df = oos.copy()
    df["y_lower80"], df["y_upper80"] = derive_band(df["y_pred"], df["y_lower"], df["y_upper"], ratio)
    w95 = (df["y_upper"] - df["y_lower"]) / df["last_close"] * 100
    w80 = (df["y_upper80"] - df["y_lower80"]) / df["last_close"] * 100
    print(f"ratio de niveau r = z(0.90)/Z_95 = {ratio:.6f}  (denominateur = arima_model.Z_95 = {arima_model.Z_95})")
    print(f"largeur médiane : 95 % -> {w95.median():.2f} % du prix | 80 % -> {w80.median():.2f} %")

    validation = {"status": "skipped"}
    if not args.no_validate:
        va = args.validate_assets or args.assets
        print(f"\n=== garde-fou : régénération indépendante sur {va} ===")
        regen = pd.concat([regenerate_asset(a, oos) for a in va], ignore_index=True)
        validation = validate(df, regen, ratio)
        a, b = validation["primary_ratio_test"], validation["secondary_absolute_test"]
        print(f"  (A) TEST DU RATIO -- ratio attendu {a['expected_ratio']:.6f}, "
              f"écart max mesuré {a['max_abs_deviation']:.2e} sur {a['n_rows_covered']} lignes "
              f"(100 %, immune à la dérive) -> {'OK' if a['passes'] else 'ÉCHEC'}")
        for cell, r in a["by_asset_regime"].items():
            print(f"        {cell:<18}n={r['n']:<5} écart max {r['max_abs_dev_from_expected_ratio']:.2e}")
        print(f"  (B) test absolu -- {b['n_faithful_at_95']}/{b['n_compared']} lignes fidèles à 95 % ; "
              f"dérivée vs régénérée : médiane {b['median_rel_diff']:.2e}, max {b['max_rel_diff']:.2e}")
        if b["cells_without_any_faithful_row"]:
            print(f"        angle mort du test (B) : {b['cells_without_any_faithful_row']} "
                  f"-- couvert par le test (A)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "bands.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps({
        "model": MODEL, "dist": DIST,
        "dist_evidence": "dist='normal' reproduit les lignes oos a 1.45e-06 ; skewt 2.28e-02, "
                         "t 2.71e-03, ged 2.80e-03 -- innovation gaussienne etablie par mesure",
        "method": "bande 80 % DERIVEE de la bande 95 % publiee (log-symetrique, innovation "
                  "gaussienne) : lo80 = y_pred*(lo95/y_pred)**r, r = z(0.90)/z(0.975)",
        "level_ratio": ratio, "base_level": BASE_LEVEL, "target_level": TARGET_LEVEL,
        "level_ratio_denominator": "arima_model.Z_95 = 1.96, le multiplicateur reellement "
                                   "utilise pour batir la bande 95 % en base -- pas "
                                   "norm.ppf(0.975). Ecart de 1.2e-05 detecte par le test "
                                   "du ratio, puis corrige.",
        "why_not_regenerate": "la regeneration integrale reproduit la base a 3e-7 en mediane mais "
                              "derive jusqu'a 1.6e-3 sur 9 lignes SPY /540 (origines d'avril 2025, "
                              "le POINT bougeant aussi) : optimum local different de l'optimiseur "
                              "ARIMA/GARCH sur une fenetre a volatilite extreme. Deriver depuis la "
                              "base garde le bras GARCH bit-pour-bit celui du benchmark.",
        "log_symmetry_check": sym,
        "validation": validation,
        "median_width_pct_of_price": {"95": float(w95.median()), "80": float(w80.median())},
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {out_dir / 'bands.parquet'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
