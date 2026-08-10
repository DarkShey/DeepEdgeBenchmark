"""
cta_gate0bis.py -- chantier 0-bis-B du BRIEF « porte 0-bis : le CTA corrige, juge
dans son habitat » : LA PORTE, avec le signal corrige, l'instrument corrige et
l'univers elargi.

CE QUI A CHANGE DEPUIS LA PORTE 0, et rien d'autre. Les trois faits sont acquis
pour des raisons mecaniques, aucun n'a ete ajuste sur un resultat :
  1. le SIGNAL est repare -- conviction-singleton (bug 1) et calendrier (bug 2) ;
     la conviction hierarchique redevient donc le signal retenu, elle n'est plus
     un acheter-et-garder deguise ;
  2. l'INSTRUMENT est repare -- la branche 2 porte sur l'EXCES vs B&H, elle n'est
     plus franchissable par un signal constant ;
  3. l'UNIVERS s'elargit a 18 actifs, sur une limite DECLAREE AVANT le verdict de
     la porte 0 (« un panel de 7 actifs est petit pour un CTA -- la diversification
     fait l'essentiel du Sharpe »).

CE QUI NE CHANGE PAS : la fenetre. 2020-01 -> 2026-07, memes origines hebdo que la
grille `oos2020`. Elargir l'univers sur une justification pre-existante est
legitime ; changer de fenetre apres avoir vu les resultats ne l'est pas.

CRITERE DE PORTE, DECLARE AVANT LE RUN (deux branches, l'une OU l'autre) :
  1. au moins un actif a PnL net positif ET p < 0,05 brut vs B&H ;
  2. Sharpe poole de l'EXCES (PnL signal - PnL B&H par origine, portefeuille
     equipondere) > 0 ET exces positif sur >= 3 classes sur 4.
Les quatre classes sont figees dans `prices_v4.PANEL` : actions / taux /
matieres-or / crypto. VXX en est exclu et rapporte a part -- ETN de volatilite a
decroissance de roulement, il n'appartient a aucune (declare avant tout calcul).

LECTURE COMPLEMENTAIRE, declaree non decisionnelle : le portefeuille diversifie
est la vraie unite d'un CTA. Sharpe de l'exces du portefeuille equipondere avec
son IC bootstrap PAR BLOCS (meme longueur de bloc que partout : 3), et
comportement par tranche -- le crisis alpha de mars 2020, que le correctif de
calendrier restaure, doit se voir la ou nulle part.

SI LA PORTE ECHOUE : cloture definitive du dossier trading, toute la ligne. Signal
corrige, instrument corrige, habitat naturel, 6,5 ans traversant trois regimes --
il n'y aura pas de porte 0-ter.

Sortie : experiments/cta_gate0bis.json
Usage   : python cta_gate0bis.py
Code de sortie : 0 si la porte est franchie, 1 sinon.
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

import econ_backtest as eb                                            # noqa: E402
import real_fees as rf                                                # noqa: E402
from cta_gate0 import (BLOCK_LENGTH, BPS, POOL_SEED, attach_signal,   # noqa: E402
                       branch2_verdict, by_period, cell_series, evaluate)
from epoch_sweep import week_targets                                  # noqa: E402
from prices_v3 import ORIGIN_START                                    # noqa: E402
from prices_v4 import CLASSES, OUT_DIR as PRICES_V4, PANEL, slug      # noqa: E402
from weekly_headtohead import build_weekly                            # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "cta_gate0bis.json"
SIGNAL_DIR = Path(__file__).resolve().parent / "deita_cta_signal_v4"
HORIZON_WEEKS = {"W+1": 1}          # horizon de detention declare par le brief
REGIME = "weekly"
N_BOOT = 2000


def build_market(assets, prices_dir: Path) -> pd.DataFrame:
    """Grille hebdo (origine, cible) construite DIRECTEMENT sur les prix geles.

    La porte 0 lisait ces colonnes sur les bandes GARCH de `grid2020`, qui
    n'existent que pour les 7 actifs du panel. Ici il n'y a aucun modele a lire :
    une porte de CTA n'a besoin que du prix courant et du prix a echeance."""
    rows = []
    for a in assets:
        daily = pd.read_parquet(Path(prices_dir) / f"{slug(a)}.parquet")["close"]
        weekly, weekly_dates = build_weekly(daily)
        origin = pd.Timestamp(ORIGIN_START)
        test_pos = [i for i, d in enumerate(weekly_dates) if d >= origin][:-3]
        for m in test_pos:
            _, _, target_dates, _ = week_targets(weekly_dates, daily, m)
            for hu, h in HORIZON_WEEKS.items():
                rows.append({"asset": a, "frequence": REGIME, "horizon_unit": hu,
                             "cutoff_date": str(weekly_dates.iloc[m].date()),
                             "target_date": str(target_dates[h - 1].date()),
                             "last_close": float(weekly.iloc[m]),
                             "y_true": float(weekly.iloc[m + h])})
    return pd.DataFrame(rows)


def load_signal(assets, signal_dir: Path) -> pd.DataFrame:
    return pd.DataFrame({a: pd.read_parquet(Path(signal_dir) / f"{slug(a)}.parquet")["signal"]
                         for a in assets}).sort_index()


def block_bootstrap_sharpe_ci(x: np.ndarray, hw: int, block: int = BLOCK_LENGTH,
                              n_boot: int = N_BOOT, seed: int = POOL_SEED,
                              alpha: float = 0.05) -> dict:
    """IC du Sharpe par bootstrap PAR BLOCS -- pas i.i.d. A l'horizon W+1 les
    sleeves ne se chevauchent pas, mais le rendement des actifs, lui, est
    autocorrele par paquets ; on garde la convention du depot (blocs de 3) plutot
    que d'en inventer une pour cette seule mesure."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < block * 2:
        return {"point": float("nan"), "ci": [None, None], "n_boot": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.integers(0, starts_max + 1, n_blocks)])[:n]
        boots.append(eb.sharpe(x[idx], hw))
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    return {"point": eb.sharpe(x, hw), "n_boot": int(boots.size),
            "ci": [float(np.quantile(boots, alpha / 2)),
                   float(np.quantile(boots, 1 - alpha / 2))],
            "share_positive": float((boots > 0).mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prices-dir", default=str(PRICES_V4))
    p.add_argument("--signal-dir", default=str(SIGNAL_DIR))
    p.add_argument("--level", default=rf.DECISION_LEVEL, choices=list(rf.LEVELS))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    assets = list(PANEL)
    market = build_market(assets, args.prices_dir)
    df = attach_signal(market, load_signal(assets, args.signal_dir))
    n_origins = df["cutoff_date"].nunique()
    in_class = [a for a in assets if PANEL[a]["classe"]]
    print(f"porte 0-bis : {n_origins} origines, {len(assets)} actifs "
          f"({len(in_class)} dans les 4 classes, {len(assets) - len(in_class)} hors), "
          f"regime {REGIME}, frais niveau {args.level} (roulement inclus)")

    cells, excess, pnl_by_inst = {}, {}, {}
    for inst, spec in rf.INSTRUMENTS.items():
        asset = spec["asset"]
        if asset not in assets:
            continue
        for hu, hw in HORIZON_WEEKS.items():
            d = cell_series(df, asset, hu)
            if d["y_true"].size == 0:
                continue
            r = evaluate(d, inst, hw, args.level)
            r.update({"instrument": inst, "asset": asset, "horizon_unit": hu,
                      "classe": PANEL[asset]["classe"],
                      "periods": by_period(r["_pnl"], r["_cutoff"], hw)})
            cells[f"{inst}|{hu}"] = r
            excess[inst] = r["_pnl"] - r["_pnl_bh"]
            pnl_by_inst[inst] = r["_pnl"]

    # Un instrument par actif pour le portefeuille et le test de classes : deux
    # vehicules du meme actif portent le MEME signal, les compter deux fois
    # pondererait cet actif double.
    one_per_asset = {}
    for inst in sorted(excess):
        one_per_asset.setdefault(rf.INSTRUMENTS[inst]["asset"], inst)
    keys_all = sorted(one_per_asset.values())
    keys = [k for k in keys_all if PANEL[rf.INSTRUMENTS[k]["asset"]]["classe"]]
    class_of = {k: PANEL[rf.INSTRUMENTS[k]["asset"]]["classe"] for k in keys}

    b2 = branch2_verdict({k: excess[k] for k in keys}, class_of, HORIZON_WEEKS["W+1"])
    branch1 = sorted(k for k, c in cells.items()
                     if c["pnl_net_cta_bps"] > 0 and c["significant_raw"])
    branch1_alt = sorted(k for k, c in cells.items()
                         if c["pnl_net_cta_bps"] > 0 and c["significant_raw_vs_zero"])
    gate = bool(branch1) or b2["passes"]

    # ── lecture complementaire, declaree non decisionnelle ──────────────────
    n = min(len(excess[k]) for k in keys)
    port_excess = np.mean([excess[k][:n] for k in keys], axis=0)
    port_pnl = np.mean([pnl_by_inst[k][:n] for k in keys], axis=0)
    ref_cut = cells[f"{keys[0]}|W+1"]["_cutoff"][:n]
    complementary = {
        "portefeuille_diversifie": {
            "n_instruments": len(keys), "n_origins": int(n),
            "excess_mean_bps": float(port_excess.mean() * BPS),
            "sharpe_excess": block_bootstrap_sharpe_ci(port_excess, 1),
            "pnl_mean_bps": float(port_pnl.mean() * BPS),
            "sharpe_pnl": block_bootstrap_sharpe_ci(port_pnl, 1),
            "max_drawdown_excess": eb.max_drawdown(port_excess),
        },
        "par_tranche_exces": by_period(port_excess, ref_cut, 1),
        "hors_classes": {k: {"instrument": k,
                             "pnl_net_bps": cells[f"{k}|W+1"]["pnl_net_cta_bps"],
                             "edge_vs_bh_bps": cells[f"{k}|W+1"]["edge_vs_bh_bps"]}
                         for k in keys_all if k not in keys},
        "status": "descriptif -- ne porte aucune decision",
    }

    payload = {
        "scope": "chantier 0-bis-B -- la porte, signal corrige, instrument corrige, univers elargi",
        "declared_criterion": {
            "branch_1": "au moins un actif a PnL net positif ET p < 0,05 brut vs B&H",
            "branch_2": b2["formulation"],
            "classes": list(CLASSES),
            "hors_classes": [a for a in assets if not PANEL[a]["classe"]],
            "on_failure": "cloture definitive du dossier trading, toute la ligne -- il n'y aura "
                          "pas de porte 0-ter",
        },
        "setup": {"prices": args.prices_dir, "signal": args.signal_dir,
                  "n_origins": n_origins, "assets": assets, "regime": REGIME,
                  "horizon": "W+1", "fees_level": args.level, "roll_included": True,
                  "window": f"{ORIGIN_START} -> 2026-07 (inchangee depuis la porte 0)"},
        "per_cell": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in cells.items()},
        "gate": {"branch_1_hits": branch1, "branch_1_passes": bool(branch1),
                 "branch_1_alt_reading_vs_zero": {"hits": branch1_alt,
                                                  "passes": bool(branch1_alt)},
                 "branch_2": b2, "passes": gate},
        "complementary_reading": complementary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"\n=== CTA corrige seul, taille fixe, W+1 ===")
    print(f"  {'instrument':<12}{'classe':<10}{'long':>6}{'PnL net':>10}{'vs B&H':>10}"
          f"{'Sharpe':>8}{'p':>8}")
    for k in sorted(cells, key=lambda k: -cells[k]["edge_vs_bh_bps"]):
        c = cells[k]
        print(f"  {c['instrument']:<12}{str(c['classe'] or '(hors)'):<10}{c['share_long']:>5.0%}"
              f"{c['pnl_net_cta_bps']:>10.2f}{c['edge_vs_bh_bps']:>10.2f}"
              f"{c['sharpe_annualised']:>8.2f}{c['p_value_vs_bh']:>8.3f}")

    pf = complementary["portefeuille_diversifie"]
    print(f"\n=== portefeuille diversifie ({pf['n_instruments']} instruments) — descriptif ===")
    print(f"  exces moyen {pf['excess_mean_bps']:+.2f} bps/origine | "
          f"Sharpe de l'exces {pf['sharpe_excess']['point']:.2f} "
          f"IC95 [{pf['sharpe_excess']['ci'][0]:.2f} ; {pf['sharpe_excess']['ci'][1]:.2f}] "
          f"({pf['sharpe_excess']['share_positive']:.0%} des tirages > 0)")
    print(f"  PnL moyen   {pf['pnl_mean_bps']:+.2f} bps/origine | "
          f"Sharpe brut {pf['sharpe_pnl']['point']:.2f}")
    for name, v in complementary["par_tranche_exces"].items():
        print(f"    {name:<20}n={v['n']:>3}  exces {v['pnl_mean_bps']:+8.2f} bps  "
              f"Sharpe {v['sharpe']:>6.2f}")

    print(f"\n  branche 1 : {'PASS -- ' + ', '.join(branch1) if branch1 else 'echec'}")
    print(f"    lecture concurrente (p vs zero) : "
          f"{'aurait passe -- ' + ', '.join(branch1_alt) if branch1_alt else 'echec aussi'}")
    print(f"  branche 2 : {'PASS' if b2['passes'] else 'echec'} "
          f"(Sharpe exces {b2['sharpe_excess_annualised']:.2f}, "
          f"{b2['n_classes_positive']}/{len(CLASSES)} classes, "
          f"exces moyen {b2['excess_mean_bps']:+.2f} bps)")
    for cls, v in b2["per_class"].items():
        print(f"    {cls:<10}{v['excess_mean_bps']:+8.2f} bps  "
              f"{'positif' if v['positive'] else 'negatif'}")
    print(f"  >>> PORTE 0-bis : {'FRANCHIE' if gate else 'ECHEC -- cloture definitive'}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
