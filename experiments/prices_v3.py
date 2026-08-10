"""
prices_v3.py -- chantier A du BRIEF "NsDiff : extension de donnees, puissance,
dashboard, re-jugement mensuel" : constituer le PANEL ETENDU de series de prix,
gele, sur lequel les chantiers B et D tourneront.

CE QUI CHANGE PAR RAPPORT A `diffusion_multiseed_v2/prices/` :

  1. DEPART COMMUN a 2011-05 la ou l'historique le permet (contre 2015-01
     aujourd'hui), soit ~800 semaines a date -- l'extension chiffree au chantier 2
     de la note precedente, qui montrait qu'une grille demarrant en 2020 rendait
     detectables les meilleures cellules SPY tout en laissant 261 semaines
     d'entrainement.
  2. DEUX ACTIFS DE PLUS : GLD (or) et USO (petrole). Ce sont les deux seuls
     candidats de la note precedente a apporter une exposition REELLEMENT
     nouvelle ; QQQ, EFA et IEF sont ecartes par le critere declare (correlation
     0,8-0,9 avec un actif deja present -- ils ajoutent des lignes, pas de
     l'information independante).
  3. PANEL DESEQUILIBRE ASSUME (decision actee en entree du brief) : chaque actif
     part a son historique maximal propre. BTC ne commence qu'en 2014-09 et ETH
     en 2017-11 : on ne tronque PAS les autres a la plus courte des series. Un
     panel equilibre couterait 400 semaines sur SPY/ZN/TLT/GLD/USO pour faire
     plaisir a une contrainte de symetrie dont aucun test n'a besoin -- le
     bootstrap par blocs travaille cellule par cellule.

CIBLE D'ENTRAINEMENT DECLAREE : >= 2 000 fenetres daily par actif la ou
l'historique le permet (niveau du train actuel, valide ; ratio NsDiff
<= 4 parametres par fenetre). BTC et surtout ETH n'y arrivent pas : ils sont
DECLARES cellules faibles du panel, ni tronques ni completes par une source
alternative -- l'homogeneite de source prime, un actif dont la moitie de
l'historique viendrait d'ailleurs ne serait plus comparable aux autres.

LES ANCIENNES SERIES SONT CONSERVEES INTACTES. `prices_v3/` est un repertoire
NOUVEAU. Aucun chiffre deja publie n'est recalcule sur ces series : tout resultat
qui les utilise appartient aux chantiers B et D, et le declare.

VERIFICATION BLOQUANTE (A2 du brief) : sur la periode COMMUNE aux deux jeux, les
nouvelles series doivent reproduire les anciennes. La question est : SUR QUELLE
QUANTITE ?

Un premier jet comparait les PRIX, a 2e-7. Il a bloque sur SPY (7,1e-7) et sur
TLT (4,0e-3). Le diagnostic separe les deux :

  * SPY -- ratio nouveau/ancien median exactement 1,00000000, dispersion
    1,4e-06, aucune date au-dela de 1e-5. C'est le bruit de derniere decimale
    que yfinance reserve d'un appel a l'autre, deja documente dans
    `dashboard_d7_w1.load_price_history_cache`. La tolerance de 2e-7 etait
    simplement trop serree : la convention de travail du repo est 1e-6
    (`oos_nsdiff_daily_weekly._verify_origin_prices`) a 4e-6 (source offline).
  * TLT -- ratio median 0,99598782, dispersion 2,5e-06. Un facteur
    MULTIPLICATIF CONSTANT, pas une revision : l'ancienne serie vient de
    `oos_nsdiff_tlt.fetch_tlt_patched` (instantane offline), la nouvelle de
    yfinance, et les deux n'appliquent pas le meme ajustement de dividendes.

Or tous les modeles de ce repo travaillent sur les LOG-RENDEMENTS, que multiplier
tous les prix par une constante laisse strictement inchanges. La verification
porte donc sur ce que les modeles voient reellement :

  1. LOG-RENDEMENTS -- bloquant, tolerance 1e-5. Mesure : SPY 1,1e-06,
     TLT 1,9e-06, ZN=F / BTC / ETH exactement 0.
  2. DISPERSION DU RATIO DE PRIX -- bloquant, tolerance 1e-4. Un ratio constant
     est un changement de base d'ajustement, sans effet sur les rendements ; un
     ratio QUI DERIVE serait une revision d'historique, et celle-la invaliderait
     tout. Cette verification-la, le test sur les prix bruts ne la faisait pas.
  3. NIVEAU DU RATIO -- rapporte, jamais bloquant. Il est declare dans le JSON
     d'artefact pour TLT (0,996), parce qu'il rend les PRIX de `prices_v3` non
     directement comparables aux anciens, meme si les rendements le sont.

Sortie : experiments/prices_v3/<actif>.parquet + config.json
Usage :
    python prices_v3.py                 # telecharge, verifie, gele
    python prices_v3.py --check-only    # re-verifie un jeu deja gele, sans reseau
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

from arima_model import fetch_data                                    # noqa: E402
from backtest_rolling_tsdiffw import FETCH_END                        # noqa: E402
from weekly_headtohead import build_weekly, HORIZON_DAILY             # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "prices_v3"
OLD_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "prices"

COMMON_START = "2011-05-01"        # ~800 semaines a FETCH_END
# Tolerances du recouvrement, cf. docstring. Elles portent sur les LOG-RENDEMENTS
# (ce que les modeles voient) et sur la STABILITE du ratio de prix (ce qui
# distingue un changement de base d'ajustement d'une revision d'historique) --
# jamais sur le niveau des prix, qui depend d'une convention de dividendes.
LOGRET_TOL = 1e-5                  # convention du repo : 1e-6 (origines) a 4e-6 (source offline)
RATIO_DISPERSION_TOL = 1e-4        # un ratio qui derive = revision d'historique = bloquant
ORIGIN_START = "2020-01-01"        # grille du chantier B
SEQ_LEN = 30
TRAIN_WINDOW_TARGET = 2000         # cible declaree, fenetres daily

# `start` = None -> depart commun ; sinon depart de source (crypto).
PANEL = {
    "SPY":     {"start": None,          "classe": "index", "note": "exposition actions US"},
    "ZN=F":    {"start": None,          "classe": "bond",  "note": "future 10-Year T-Note"},
    "TLT":     {"start": None,          "classe": "bond",  "note": "ETF obligations longues"},
    "BTC-USD": {"start": "2014-09-01",  "classe": "crypto", "note": "debut de la source yfinance"},
    "ETH-USD": {"start": "2017-11-01",  "classe": "crypto", "note": "debut de la source yfinance"},
    "GLD":     {"start": None,          "classe": "commodity",
                "note": "NOUVEAU -- or ; exposition non correlee au panier existant"},
    "USO":     {"start": None,          "classe": "commodity",
                "note": "NOUVEAU -- petrole ; exposition non correlee au panier existant"},
}

# Regimes de marche traverses, par tranche -- le gain de l'extension est de
# validite externe autant que de puissance (un backtest 2024-2026 ne raconte pas
# la meme histoire qu'un backtest qui traverse 2020 et 2022).
REGIMES_TRAVERSED = {
    "2011-2015": "QE, taux zero, volatilite basse",
    "2015-2020": "normalisation des taux, deux corrections (2015, 2018)",
    "2020": "choc COVID -- la plus forte dislocation de l'echantillon",
    "2022": "marche baissier de taux, inflation, correlation actions/obligations positive",
    "2024-2026": "la fenetre du benchmark actuel",
}


def slug(asset: str) -> str:
    return asset.replace("=", "_")


def fetch_one(asset: str, start: str) -> pd.Series:
    print(f"[{asset}] téléchargement {start} -> {FETCH_END} ...")
    s = fetch_data(asset, start, FETCH_END)
    return s.sort_index()


def overlap_check(asset: str, new: pd.Series, old_dir: Path = None) -> dict:
    """Bloquant, sur les LOG-RENDEMENTS et sur la STABILITE du ratio de prix --
    pas sur le niveau des prix. Cf. docstring du module pour le raisonnement.

    `old_dir` est opt-in (defaut : les series du chantier precedent). `prices_v4`
    s'en sert pour se comparer a `prices_v3` : un gel qui succede a un autre doit
    prouver qu'il le reproduit sur les actifs communs, sinon les grilles bati es
    sur l'un ne se comparent plus a celles baties sur l'autre."""
    old_path = Path(old_dir or OLD_DIR) / f"{slug(asset)}.parquet"
    if not old_path.exists():
        return {"status": "no_previous_series", "asset": asset}
    old = pd.read_parquet(old_path)["close"]
    common = new.index.intersection(old.index)
    if len(common) == 0:
        raise SystemExit(f"[{asset}] aucune date commune avec l'ancienne serie -- verification impossible")

    n, o = new.loc[common].values, old.loc[common].values
    ratio = n / o
    d_logret = np.abs(np.diff(np.log(n)) - np.diff(np.log(o)))
    dispersion = float(ratio.max() - ratio.min())

    out = {
        "status": "checked", "n_common": int(len(common)),
        "span_common": f"{common[0].date()} -> {common[-1].date()}",
        "logret_max_abs_diff": float(d_logret.max()),
        "logret_median_abs_diff": float(np.median(d_logret)),
        "logret_tolerance": LOGRET_TOL,
        "price_ratio_median": float(np.median(ratio)),
        "price_ratio_dispersion": dispersion,
        "price_ratio_dispersion_tolerance": RATIO_DISPERSION_TOL,
        "n_dates_only_in_old": int(len(old.index.difference(new.index))),
        "n_dates_only_in_new": int(len(new.index.difference(old.index))),
    }
    out["logret_ok"] = bool(out["logret_max_abs_diff"] <= LOGRET_TOL)
    out["ratio_stable"] = bool(dispersion <= RATIO_DISPERSION_TOL)
    out["passes"] = bool(out["logret_ok"] and out["ratio_stable"])
    # Un niveau de ratio different de 1 n'est pas un echec, mais il doit etre VU :
    # les PRIX de prices_v3 ne sont alors pas directement comparables aux anciens.
    if abs(out["price_ratio_median"] - 1.0) > 1e-4:
        out["price_level_shift"] = (
            f"ratio constant {out['price_ratio_median']:.8f} -- base d'ajustement de dividendes "
            f"differente (ancienne serie : source offline `fetch_tlt_patched` ; nouvelle : "
            f"yfinance). Sans effet sur les log-rendements, donc sans effet sur les modeles ; "
            f"mais les NIVEAUX de prix ne sont pas comparables d'un jeu a l'autre.")
    return out


def counts(asset: str, daily: pd.Series) -> dict:
    """Comptages du tableau A1 du brief, calcules et non repris a la main."""
    weekly, weekly_dates = build_weekly(daily)
    origin = pd.Timestamp(ORIGIN_START)
    n_train_weeks = int((weekly_dates < origin).sum())
    n_test_origins = max(0, int((weekly_dates >= origin).sum()) - 3)
    train_daily = daily[daily.index < origin]
    # fenetres = rendements - seq_len - horizon + 1
    n_train_windows_daily = max(0, len(train_daily) - 1 - SEQ_LEN - HORIZON_DAILY + 1)
    n_train_windows_weekly = max(0, n_train_weeks - 1 - SEQ_LEN - 3 + 1)
    monthly = weekly.iloc[pd.Series(np.arange(len(weekly)),
                                    index=pd.to_datetime(weekly.index).to_period("M")
                                    ).groupby(level=0).max().values]
    return {
        "n_daily": int(len(daily)), "n_weekly": int(len(weekly)), "n_monthly": int(len(monthly)),
        "span": f"{daily.index[0].date()} -> {daily.index[-1].date()}",
        "n_train_weeks_before_origin": n_train_weeks,
        "n_test_origins_from_2020": n_test_origins,
        "effective_n_from_2020": n_test_origins // 3,
        "n_train_windows_daily": n_train_windows_daily,
        "n_train_windows_weekly": n_train_windows_weekly,
        "meets_daily_target": bool(n_train_windows_daily >= TRAIN_WINDOW_TARGET),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--check-only", action="store_true",
                   help="re-vérifie un jeu déjà gelé, sans aucun appel réseau")
    args = p.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series, per_asset, failures = {}, {}, []
    for asset in args.assets:
        frozen = out_dir / f"{slug(asset)}.parquet"
        start = PANEL[asset]["start"] or COMMON_START
        if frozen.exists():
            daily = pd.read_parquet(frozen)["close"]
            print(f"[{asset}] série gelée relue ({len(daily)} obs) -- aucun appel réseau")
        elif args.check_only:
            failures.append({"asset": asset, "reason": "absente et --check-only"})
            continue
        else:
            try:
                daily = fetch_one(asset, start)
            except Exception as exc:
                print(f"[{asset}] ÉCHEC : {type(exc).__name__}: {exc}")
                failures.append({"asset": asset, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if daily.empty:
                failures.append({"asset": asset, "reason": "série vide"})
                continue
            daily.to_frame("close").to_parquet(frozen)
        series[asset] = daily
        per_asset[asset] = {**PANEL[asset], "requested_start": start,
                            "counts": counts(asset, daily),
                            "overlap_vs_v2": overlap_check(asset, daily)}

    # ── A2 : vérifications bloquantes ───────────────────────────────────────
    print("\n=== A2 : recouvrement avec les séries de `diffusion_multiseed_v2/prices/` ===")
    blocking = []
    for asset, info in per_asset.items():
        o = info["overlap_vs_v2"]
        if o["status"] == "no_previous_series":
            print(f"  {asset:<9} pas de série antérieure (nouvel actif) -- rien à vérifier")
            continue
        verdict = "OK" if o["passes"] else "ÉCHEC"
        print(f"  {asset:<9}{o['n_common']:>5} dates communes ({o['span_common']}) | "
              f"log-rendements max {o['logret_max_abs_diff']:.2e} | ratio de prix "
              f"médian {o['price_ratio_median']:.8f}, dispersion {o['price_ratio_dispersion']:.2e}"
              f" -> {verdict}")
        if "price_level_shift" in o:
            print(f"  {'':<9}  décalage de niveau déclaré : {o['price_level_shift'][:110]}...")
        if not o["passes"]:
            blocking.append(asset)
    if blocking:
        raise SystemExit(f"recouvrement hors tolérance sur {blocking} -- la source a révisé son "
                         "historique ; aucune comparaison avec les résultats publiés ne tient. "
                         "STOP, à documenter.")

    print("\n=== comptages (grille d'origines à partir de 2020-01) ===")
    print(f"{'actif':<10}{'daily':>7}{'weekly':>8}{'mensuel':>9}{'span':>26}"
          f"{'train hebdo':>13}{'fen. daily':>12}{'cible 2000':>12}{'origines':>10}{'eff_n':>7}")
    for asset, info in per_asset.items():
        c = info["counts"]
        print(f"{asset:<10}{c['n_daily']:>7}{c['n_weekly']:>8}{c['n_monthly']:>9}{c['span']:>26}"
              f"{c['n_train_weeks_before_origin']:>13}{c['n_train_windows_daily']:>12}"
              f"{('oui' if c['meets_daily_target'] else 'NON'):>12}"
              f"{c['n_test_origins_from_2020']:>10}{c['effective_n_from_2020']:>7}")

    weak = [a for a, i in per_asset.items() if not i["counts"]["meets_daily_target"]]
    print(f"\ncellules faibles du panel (< {TRAIN_WINDOW_TARGET} fenêtres daily) : {weak or 'aucune'}"
          f"\n  -> déclarées telles quelles : ni tronquées, ni complétées par une autre source")

    payload = {
        "scope": "chantier A -- panel étendu, prix gelés, panel déséquilibré assumé",
        "config": {
            "common_start": COMMON_START, "fetch_end": FETCH_END,
            "origin_start_for_counts": ORIGIN_START, "seq_len": SEQ_LEN,
            "horizon_daily": HORIZON_DAILY,
            "daily_train_window_target": TRAIN_WINDOW_TARGET,
            "overlap_checks": {
                "on": "LOG-RENDEMENTS (ce que les modeles voient) et STABILITE du ratio de prix "
                      "(ce qui distingue un changement de base d'ajustement d'une revision) -- "
                      "jamais le niveau des prix, qui depend d'une convention de dividendes",
                "logret_tolerance": LOGRET_TOL,
                "price_ratio_dispersion_tolerance": RATIO_DISPERSION_TOL,
            },
            "unbalanced_panel": "assumé (décision actée) : chaque actif à son historique maximal "
                                "propre, aucune troncature au plus court",
            "new_assets": ["GLD", "USO"],
            "excluded_candidates": {"QQQ": "corrélé ~0,9 à SPY", "EFA": "corrélé ~0,8 à SPY",
                                    "IEF": "corrélé ~0,9 à TLT/ZN=F"},
            "previous_series_untouched": str(OLD_DIR),
        },
        "regimes_traversed": REGIMES_TRAVERSED,
        "per_asset": per_asset,
        "weak_cells": weak,
        "failures": failures,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {out_dir}  ({time.time() - t0:.0f}s)")
    if failures:
        print(f"ÉCHECS : {failures}")


if __name__ == "__main__":
    main()
