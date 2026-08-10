"""
grid2020_tests.py -- chantier B du BRIEF extension/puissance : les TESTS sur la
grille regeneree a depart 2020-01 (340 origines, `effective_n` ~ 113, panel
etendu `prices_v3/`).

TROIS QUESTIONS, ET UNE SEULE EST CONFIRMATOIRE.

  1. HYPOTHESE PRIMAIRE, PRE-DECLAREE DANS LE BRIEF -- et c'est la reponse a la
     limite « l'edge n'est pas corrige pour la selection d'instrument » que la
     note precedente s'imposait. Elle est fixee AVANT d'avoir vu la moindre
     donnee de cette grille :

         `var_limit`, SPY (vehicules ES et ETF), W+2 et W+3, regime WEEKLY,
         NsDiff-ensemble contre ARIMA-GARCH.

     Motif du choix, lui aussi anterieur : l'analyse de puissance donnait pour
     ces cellules un n requis SOUS HOLM de 231 a 270 origines, contre 340
     disponibles ici -- ce sont les seules du programme a etre a la fois
     favorables et atteignables. Famille de Holm : ces 4 tests, et eux seuls.

  2. REPLICATION des 4 survivants « acheter-et-garder bat la strategie » (SPY,
     W+1 weekly, ~-12 bps par origine). Un resultat qui ne se replique pas sur
     une grille quatre fois plus longue et trois regimes de marche de plus
     n'etait pas un resultat.

  3. CALIBRATION : le match Winkler / couverture NsDiff-ensemble vs GARCH au
     nouveau depart. C'est la contrepartie du volet economique -- si la parite de
     calibration etablie au chantier A2 ne tient pas sur 340 origines, elle
     n'etait qu'un artefact de puissance.

TOUT LE RESTE EST EXPLORATOIRE et etiquete tel quel : autres instruments, GLD et
USO, regime daily, autres horizons. Ils sont calcules et rapportes, jamais
utilises pour conclure.

CRITERE D'ARRET GLOBAL, declare dans le brief : si l'hypothese primaire ne
survit pas a Holm sur 340 origines, le volet economique du programme se clot
definitivement -- il n'y aura pas de troisieme univers de test.

Briques reutilisees telles quelles : `econ_backtest` (moteur, inchange),
`real_fees` (grille etendue a GLD/USO), `nsdiff_production_spec`,
`paired_test`, `multiple_testing`, `mcs.spa_test`, `calibration_tests`,
`dashboard_d7_w1.winkler_score`.

Sortie : experiments/grid2020_tests.json
Usage   : python grid2020_tests.py
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

import calibration_tests as ct                                        # noqa: E402
import dashboard_d7_w1 as dash                                        # noqa: E402
import econ_backtest as eb                                            # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import nsdiff_production_spec as spec                                 # noqa: E402
import real_fees as rf                                                # noqa: E402
from mcs import spa_test                                              # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "grid2020_tests.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
POOL_SEED = 42
BLOCK_LENGTH = 3
BPS = 1e4
HORIZON_WEEKS = {"W+1": 1, "W+2": 2, "W+3": 3}
LEVEL_BY_STRATEGY = {"inverse_width": 0.95, "var_limit": 0.95, "filtered_direction": 0.80}
CELL_KEYS = ["asset", "frequence", "horizon_unit", "cutoff_date", "target_date"]

# ── L'HYPOTHESE PRIMAIRE, en dur, avant tout calcul ─────────────────────────
PRIMARY = {
    "strategy": "var_limit",
    "instruments": ["SPY-ES", "SPY-ETF"],
    "horizons": ["W+2", "W+3"],
    "regime": "weekly",
    "comparison": "test_vs_garch",
    "declared_in": "BRIEF_nsdiff_extension_puissance_mensuel.md, chantier B",
    "rationale": "n requis sous Holm de 231 a 270 origines (power_analysis.json) contre 340 "
                 "disponibles -- les seules cellules a la fois favorables et atteignables",
}
REPLICATION = {"strategies": ["inverse_width", "var_limit"], "instruments": ["SPY-ES", "SPY-ETF"],
               "horizons": ["W+1"], "regime": "weekly", "comparison": "test_vs_bh",
               "expected": "buy_and_hold_significantly_better (~-12 bps/origine sur 90 origines)"}


def load_arms(grid_dir: Path, garch_arm: str = "ARIMA-GARCH"):
    """`garch_arm` est opt-in, defaut = bras gaussien (chemin historique inchange).
    H1 du BRIEF_nsdiff_regeneration_oos_et_famille3.md peut designer une autre loi
    d'innovation comme config championne ; les tests confirmatoires doivent alors
    se rejouer CONTRE ELLE, sinon le mur GARCH est teste dans une version que le
    benchmark ne retient plus."""
    rows = pd.read_parquet(grid_dir / "NsDiff" / "rows.parquet")
    samples = np.load(grid_dir / "NsDiff" / "samples.npy")
    garch = pd.read_parquet(grid_dir / garch_arm / "bands.parquet")
    return rows.reset_index(drop=True), samples, garch


def nsdiff_bands(rows: pd.DataFrame, samples: np.ndarray, level: float,
                 point: str = "mean") -> pd.DataFrame:
    """`point` est opt-in, defaut "mean" -- chemin historique inchange. La famille
    3' du BRIEF_nsdiff_regeneration_oos_et_famille3.md est declaree sur la MEDIANE
    predictive, et le bras GARCH qui lui fait face publie sa mediane : lire une
    moyenne d'un cote et une mediane de l'autre mesurerait l'asymetrie du nuage,
    pas le signal."""
    alpha = (1.0 - level) / 2.0
    out = []
    for keys, g in rows.groupby(CELL_KEYS, sort=False):
        cloud = spec.aggregate_cloud(samples[g.index.to_numpy()])
        lo, hi = (float(q) for q in np.quantile(cloud, [alpha, 1.0 - alpha]))
        y_pred = float(np.median(cloud)) if point == "median" else float(cloud.mean())
        out.append({**dict(zip(CELL_KEYS, keys)), "y_pred": y_pred,
                    "y_lower": lo, "y_upper": hi, "y_true": float(g["y_true"].iloc[0]),
                    "last_close": float(g["last_close"].iloc[0])})
    return pd.DataFrame(out)


def nsdiff_bands_seed(rows: pd.DataFrame, samples: np.ndarray, level: float, seed: int) -> pd.DataFrame:
    alpha = (1.0 - level) / 2.0
    mask = (rows["seed"] == seed).to_numpy()
    sub, cloud = rows[mask].reset_index(drop=True), samples[mask]
    out = sub[CELL_KEYS + ["y_true", "last_close"]].copy()
    out["y_pred"] = cloud.mean(axis=1)
    out["y_lower"] = np.quantile(cloud, alpha, axis=1)
    out["y_upper"] = np.quantile(cloud, 1.0 - alpha, axis=1)
    return out


def garch_bands(g: pd.DataFrame, level: float) -> pd.DataFrame:
    lo, hi = ("y_lower", "y_upper") if level == 0.95 else ("y_lower80", "y_upper80")
    out = g[CELL_KEYS + ["y_true", "last_close", "y_pred"]].copy()
    out["y_lower"], out["y_upper"] = g[lo].values, g[hi].values
    return out


def align(frames: dict) -> dict:
    key = lambda d: list(zip(d["asset"], d["frequence"], d["horizon_unit"], d["cutoff_date"]))
    common = None
    for df in frames.values():
        k = set(key(df))
        common = k if common is None else (common & k)
    out = {n: df[[k in common for k in key(df)]].sort_values(
        ["asset", "frequence", "horizon_unit", "cutoff_date"]).reset_index(drop=True)
        for n, df in frames.items()}
    ref = next(iter(out.values()))
    for n, df in out.items():
        if not np.allclose(df["y_true"].values, ref["y_true"].values):
            raise SystemExit(f"bras '{n}' ne voit pas la meme cible")
    return out


def cell(df: pd.DataFrame, asset: str, regime: str, hu: str) -> dict:
    s = df[(df["asset"] == asset) & (df["frequence"] == regime)
           & (df["horizon_unit"] == hu)].sort_values("cutoff_date")
    return {c: s[c].to_numpy(dtype=float)
            for c in ("last_close", "y_pred", "y_lower", "y_upper", "y_true")}


def decompose(strategy: str, data: dict, cost_one_way: float) -> dict:
    w = (eb.positions_buy_and_hold(data["last_close"]) if strategy == "buy_and_hold"
         else eb.STRATEGIES[strategy]["fn"](data))
    w = np.asarray(w, dtype=float)
    r = eb.gross_returns(data["last_close"], data["y_true"])
    gross = w * r
    fees = 2.0 * (cost_one_way * 1e-4) * np.abs(w)
    return {"positions": w, "gross": gross, "fees": fees, "net": gross - fees}


def paired(a, b, la: str, lb: str) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 5:
        return {"status": "insufficient_data", "n": int(d.size)}
    t = paired_block_bootstrap_test(d, block_length=min(BLOCK_LENGTH, d.size), seed=POOL_SEED)
    v = ("indistinguishable" if not t["significant_at_05"]
         else (f"{la}_significantly_better" if t["mean_diff"] > 0 else f"{lb}_significantly_better"))
    return {"status": "tested", "verdict": v, **t}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--garch-arm", default="ARIMA-GARCH",
                   help="bras GARCH oppose a NsDiff : la config championne actee par H1")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples, garch = load_arms(Path(args.grid_dir), args.garch_arm)
    seeds = sorted(int(s) for s in rows["seed"].unique())
    assets = sorted(rows["asset"].unique())
    n_origins = rows["cutoff_date"].nunique()
    print(f"grille : {n_origins} origines, {len(assets)} actifs, {len(seeds)} graines, "
          f"{samples.shape[1]} tirages | effective_n ~ {n_origins // BLOCK_LENGTH}")

    arms_by_level = {}
    for level in sorted(set(LEVEL_BY_STRATEGY.values())):
        frames = {"nsdiff_ensemble": nsdiff_bands(rows, samples, level),
                  "garch": garch_bands(garch, level)}
        for s in seeds:
            frames[f"nsdiff_seed{s}"] = nsdiff_bands_seed(rows, samples, level, s)
        arms_by_level[level] = align(frames)
    seed_arms = [f"nsdiff_seed{s}" for s in seeds]

    # ── volet economique, toutes cellules ───────────────────────────────────
    lvl = rf.DECISION_LEVEL
    cells = {}
    for strategy, pi_level in LEVEL_BY_STRATEGY.items():
        arms = arms_by_level[pi_level]
        for inst, sp in rf.INSTRUMENTS.items():
            if sp["asset"] not in assets:
                continue
            for hu in HORIZON_WEEKS:
                for regime in ("weekly", "daily"):
                    data = {n: cell(df, sp["asset"], regime, hu) for n, df in arms.items()}
                    if data["garch"]["y_true"].size == 0:
                        continue
                    cost = rf.one_way_bps(inst, lvl)
                    dec = {n: decompose(strategy, d, cost) for n, d in data.items()}
                    dec["buy_and_hold"] = decompose("buy_and_hold", data["garch"], cost)
                    dec["nsdiff_pooled"] = {k: np.mean([dec[a][k] for a in seed_arms], axis=0)
                                            for k in ("gross", "fees", "net", "positions")}
                    ns, gc, bh = dec["nsdiff_ensemble"], dec["garch"], dec["buy_and_hold"]
                    edge_brut = float(np.mean(ns["gross"] - gc["gross"]) * BPS)
                    surcout = float(np.mean(ns["fees"] - gc["fees"]) * BPS)
                    cells[f"{inst}|{strategy}|{hu}|{regime}"] = {
                        "instrument": inst, "asset": sp["asset"], "pi_level": pi_level,
                        "n": int(ns["net"].size), "round_trip_bps": rf.round_trip_bps(inst, lvl),
                        "edge_brut_bps": edge_brut, "surcout_bps": surcout,
                        "edge_net_bps": edge_brut - surcout,
                        "pnl_net_nsdiff_bps": float(np.mean(ns["net"]) * BPS),
                        "pnl_net_garch_bps": float(np.mean(gc["net"]) * BPS),
                        "pnl_net_bh_bps": float(np.mean(bh["net"]) * BPS),
                        "exposition_moyenne": float(np.mean(np.abs(ns["positions"]))),
                        "n_origines_actives": int((np.abs(ns["positions"]) > 0).sum()),
                        "test_vs_garch": paired(ns["net"], gc["net"], "nsdiff", "garch"),
                        "test_vs_bh": paired(ns["net"], bh["net"], "nsdiff", "buy_and_hold"),
                        "test_pooled_vs_garch": paired(dec["nsdiff_pooled"]["net"], gc["net"],
                                                       "nsdiff_pooled", "garch"),
                        "spa_vs_garch": spa_test(-gc["net"], {"nsdiff_ensemble": -ns["net"]},
                                                 block_length=BLOCK_LENGTH, seed=POOL_SEED),
                    }

    # ── 1. hypothese primaire : famille de Holm = ses 4 tests, et eux seuls ──
    fam = {f"{i}|{h}": cells[f"{i}|{PRIMARY['strategy']}|{h}|{PRIMARY['regime']}"][PRIMARY["comparison"]]
           for i in PRIMARY["instruments"] for h in PRIMARY["horizons"]
           if f"{i}|{PRIMARY['strategy']}|{h}|{PRIMARY['regime']}" in cells}
    corrected = mt.correct_family(fam)
    primary = {**PRIMARY, "family": corrected["family"],
               "summary": mt.family_summary(corrected)}
    primary["survives"] = bool(primary["summary"]["n_significant_holm"] > 0
                               and all(fam[n]["verdict"].startswith("nsdiff")
                                       for n in primary["summary"]["survivors"]))

    # ── 2. replication des survivants « B&H bat la strategie » ──────────────
    rep = {}
    for s in REPLICATION["strategies"]:
        f = {f"{i}|{h}": cells[f"{i}|{s}|{h}|{REPLICATION['regime']}"][REPLICATION["comparison"]]
             for i in REPLICATION["instruments"] for h in REPLICATION["horizons"]
             if f"{i}|{s}|{h}|{REPLICATION['regime']}" in cells}
        c = mt.correct_family(f)
        rep[s] = {"family": c["family"], "summary": mt.family_summary(c),
                  "replicates": bool(any(f[n]["verdict"] == "buy_and_hold_significantly_better"
                                         for n in c["family"] if c["family"][n]["holm_reject"]))}

    # ── 3. calibration NsDiff-ensemble vs GARCH ─────────────────────────────
    calib = {}
    arms95 = arms_by_level[0.95]
    for asset in assets:
        for regime in ("weekly", "daily"):
            for hu in HORIZON_WEEKS:
                a = cell(arms95["nsdiff_ensemble"], asset, regime, hu)
                b = cell(arms95["garch"], asset, regime, hu)
                if a["y_true"].size == 0:
                    continue
                wa = dash.winkler_score(a["y_true"], a["y_lower"], a["y_upper"])
                wb = dash.winkler_score(b["y_true"], b["y_lower"], b["y_upper"])
                ia = ((a["y_true"] >= a["y_lower"]) & (a["y_true"] <= a["y_upper"])).astype(float)
                ib = ((b["y_true"] >= b["y_lower"]) & (b["y_true"] <= b["y_upper"])).astype(float)
                calib[f"{asset}|{regime}|{hu}"] = {
                    "n": int(len(wa)),
                    "cov95_nsdiff": float(ia.mean()), "cov95_garch": float(ib.mean()),
                    "winkler_nsdiff": float(wa.mean()), "winkler_garch": float(wb.mean()),
                    "winkler_test": paired(-wa, -wb, "nsdiff", "garch"),
                    "coverage_gap_nsdiff": ct.coverage_gap_block_test(ia, target=0.95, seed=POOL_SEED),
                    "coverage_gap_garch": ct.coverage_gap_block_test(ib, target=0.95, seed=POOL_SEED),
                }
    calib_fam = {k: v["winkler_test"] for k, v in calib.items() if "|weekly|" in k}
    calib_corr = mt.correct_family(calib_fam)

    payload = {
        "grid": {"n_origins": n_origins, "effective_n": n_origins // BLOCK_LENGTH,
                 "assets": assets, "seeds": seeds, "n_samples_per_seed": int(samples.shape[1]),
                 "source": str(args.grid_dir), "garch_arm": args.garch_arm},
        "primary_hypothesis": primary,
        "replication_bh": rep,
        "calibration": {"per_cell": calib,
                        "holm_weekly": {"family": calib_corr["family"],
                                        "summary": mt.family_summary(calib_corr)}},
        "exploratory_cells": cells,
        "note": "Seule l'hypothese primaire est confirmatoire. Tout le reste (autres instruments, "
                "GLD/USO, regime daily, autres horizons) est EXPLORATOIRE et ne fonde aucune "
                "conclusion.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n=== 1. HYPOTHESE PRIMAIRE (confirmatoire) : {PRIMARY['strategy']}, SPY, "
          f"{'/'.join(PRIMARY['horizons'])}, {PRIMARY['regime']}, vs GARCH ===")
    for name, t in sorted(primary["family"].items()):
        c = cells[f"{name.split('|')[0]}|{PRIMARY['strategy']}|{name.split('|')[1]}|{PRIMARY['regime']}"]
        print(f"  {name:<16} edge net {c['edge_net_bps']:+7.2f} bps  p={t['p_value']:.4f}  "
              f"p_aj={t['holm_p_adjusted']:.4f}  {t['verdict']}  -> Holm : {t['holm_verdict']}")
    s = primary["summary"]
    print(f"  famille m={s['m']}, seuil le plus strict {s['smallest_threshold']:.4f} | "
          f"{s['n_significant_raw']} bruts -> {s['n_significant_holm']} Holm")
    print(f"  >>> HYPOTHESE PRIMAIRE : {'SURVIT' if primary['survives'] else 'NE SURVIT PAS'}")

    print(f"\n=== 2. REPLICATION « acheter-et-garder bat la strategie » (SPY W+1 weekly) ===")
    for s_, r in rep.items():
        for name, t in sorted(r["family"].items()):
            print(f"  {s_:<15}{name:<14} p={t['p_value']:.4f}  {t['verdict']}  "
                  f"-> Holm : {t['holm_verdict']}")
        print(f"  {s_:<15}se replique : {r['replicates']}")

    print(f"\n=== 3. CALIBRATION NsDiff-ensemble vs GARCH (regime weekly, Winkler) ===")
    cs = mt.family_summary(calib_corr)
    print(f"  famille m={cs['m']} | {cs['n_significant_raw']} bruts -> {cs['n_significant_holm']} Holm")
    for k in sorted(calib_fam):
        v = calib[k]
        print(f"  {k:<22}Cov95 {v['cov95_nsdiff']:.3f}/{v['cov95_garch']:.3f}  "
              f"Winkler {v['winkler_nsdiff']:.4g}/{v['winkler_garch']:.4g}  "
              f"{calib_corr['family'][k]['verdict']} -> Holm : {calib_corr['family'][k]['holm_verdict']}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
