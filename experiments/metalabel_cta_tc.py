"""
metalabel_cta_tc.py -- BRIEF_metalabeling_cta_filtre_tc_nsdiff.md : le CTA corrigé
filtré par la taxonomie TC de NsDiff (méta-labeling deux étages, Lopez de Prado 2018).

LA QUESTION : un filtre dérivé de la géométrie prévision/PI de NsDiff améliore-t-il la
politique CTA en supprimant les faux signaux ? Le méta-labeler ne prédit pas le marché,
il prédit quand le primaire est fiable (prédiction sélective, El-Yaniv-Wiener).

ZÉRO APPRENTISSAGE, ZÉRO GÉNÉRATION. Les états TC se dérivent des lignes weekly déjà en
base ; le signal CTA est gelé et hashé ; `tracking.db` est ouvert en LECTURE SEULE. Ce
module est une jointure et un backtest, pas un modèle.

## Trois choix d'exécution qu'il faut lire avant les chiffres

**1. Quel « CTA corrigé, direction Hull » ?** Le brief demande « CTA corrigé (direction
Hull, v2-2026-08-08) ». Aucun artefact ne porte exactement ce nom, parce que la version
corrigée du moteur (`deita_cta_signal_v4/`) retient la CONVICTION, pas la direction Hull.
Le bon artefact est `deita_cta_signal_own/`, et voici pourquoi c'est bien « corrigé » :
  - bug 1 (conviction-carré) vit dans `_conviction_level` ; la direction Hull est
    UNIVARIÉE (`trend_direction`, cf. deita_cta_signal.py) et ne le traverse jamais ;
  - bug 2 (calendrier ffill) est précisément ce que `calendar="own"` évite -- le code le
    dit en toutes lettres : « le moteur corrigé n'a plus besoin du ffill : le bug 2 est
    corrigé en amont, calendar_policy="own" est son défaut » ;
  - bug 3 était un look-ahead du harnais de validation DEITA, pas du signal.
`deita_cta_signal_own/` = direction Hull, sans bug 1 (hors chemin), sans bug 2, sur les
7 actifs de la grille oos2020. C'est le primaire.

**2. Précédence des états TC -- lacune du brief, comblée et déclarée AVANT les runs.**
Le brief liste les états comme exclusifs, mais les règles de `sim_trades.py` ne le sont
pas toutes : Stress et Calm sont étanches par construction (garde-fous explicites dans
bull_calm_d1/bear_calm_d1), **mais Sideways recouvre Calm** -- une journée à dérive
faible émet les deux. Précédence retenue : **Stress > Sideways > Calm**. Motif : Sideways
signifie « dérive négligeable devant la largeur » (|predicted − ref| <= k·W), donc
« NsDiff n'exprime PAS de direction ». Le lire comme Bull/Bear reviendrait à prendre du
bruit pour une direction. L'ordre inverse (Calm > Sideways) viderait la ligne Sideways de
la matrice, rendrait F1 et F2 presque identiques, et le brief sans objet.

**3. Le placebo est à couverture EXACTEMENT égale** à celle de F2, tirage sans remise sur
les mêmes cellules, graine fixée, 100 tirages -- la discipline risque-couverture.

Sortie : experiments/metalabel_cta_tc.json
Usage   : python experiments/metalabel_cta_tc.py
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from validation import sim_trades as st                                # noqa: E402
import econ_backtest as eb                                             # noqa: E402
import real_fees as rf                                                 # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402
from multiple_testing import correct_family                            # noqa: E402

DB_PATH = str(ROOT / "validation" / "tracking.db")
SIGNAL_DIR = ROOT / "experiments" / "deita_cta_signal_own"
OUT_PATH = Path(__file__).resolve().parent / "metalabel_cta_tc.json"

BPS = 1e4
BLOCK_LENGTH = 3               # même longueur de bloc que partout dans le programme
POOL_SEED = 42
PLACEBO_SEED = 20260810
N_PLACEBO = 100
SOURCE = "oos2020"
HORIZON_UNIT = "W+1"
HORIZON_WEEKS = 1
COST_LEVEL = rf.DECISION_LEVEL
PRIMARY_MODEL = "NsDiff"
CONTROL_MODEL = "ARIMA-GARCH"

# Régimes de marché, figés avant les runs (mêmes tranches que cta_gate0).
PERIODS = {"2020 (COVID)": ("2020-01-01", "2020-12-31"),
           "2022 (inflation/taux)": ("2022-01-01", "2022-12-31"),
           "2024-2026 (récent)": ("2024-01-01", "2026-12-31")}

BULL_STATES = ("Bull-Calm", "Bull-Stress")
BEAR_STATES = ("Bear-Calm", "Bear-Stress")


def slug(asset: str) -> str:
    return asset.replace("=", "_")


# ── 1. états TC : les règles de sim_trades.py, importées telles quelles ─────

def tc_state(ref: float, predicted: float, pi_low: float, pi_high: float) -> str:
    """État TC d'une cellule, par appel des RULES de `sim_trades` SANS modification.

    `realized=None` : on ne demande que la partie SIGNAL des règles (émet / n'émet pas),
    qui ne dépend que de (ref, predicted, pi_low, pi_high) -- aucune information de
    résolution n'entre ici, donc aucun look-ahead possible par construction.

    Précédence Stress > Sideways > Calm : cf. §2 du docstring du module."""
    if st.RULES["pi95_conf"](ref, predicted, pi_low, pi_high, None)[0]:
        return "Bull-Stress"
    if st.RULES["bear_stress_d1"](ref, predicted, pi_low, pi_high, None)[0]:
        return "Bear-Stress"
    if st.RULES["sideways_d1"](ref, predicted, pi_low, pi_high, None)[0]:
        return "Sideways"
    if st.RULES["bull_calm_d1"](ref, predicted, pi_low, pi_high, None)[0]:
        return "Bull-Calm"
    if st.RULES["bear_calm_d1"](ref, predicted, pi_low, pi_high, None)[0]:
        return "Bear-Calm"
    return "aucun"


# ── 2. la matrice de décision, fixée a priori ──────────────────────────────

def decide(state: str, cta_sign: float, filter_name: str) -> float:
    """Poids du trade après méta-filtre. |w| = 1 sur les trades pris, 0 sinon -- le
    sizing n'est pas la question de ce brief.

    F1 (faible)  : veto sur CONTRADICTION seule (Bull x short, Bear x long).
    F2 (strict)  : agir uniquement sur CONCORDANCE (Bull x long, Bear x short) ;
                   sideways, contradiction et « aucun état » sont vetoés.
    « Aucun état » : le filtre ne sait pas, donc il ne bloque pas -> F1 prend.
    """
    if cta_sign == 0:
        return 0.0
    bull, bear = state in BULL_STATES, state in BEAR_STATES
    concordant = (bull and cta_sign > 0) or (bear and cta_sign < 0)
    contradictory = (bull and cta_sign < 0) or (bear and cta_sign > 0)
    if filter_name == "C0":
        return cta_sign
    if filter_name == "F1":
        return 0.0 if contradictory else cta_sign
    if filter_name == "F2":
        return cta_sign if concordant else 0.0
    raise ValueError(f"filtre inconnu : {filter_name!r}")


def placebo_weights(cta: np.ndarray, n_veto: int, rng: np.random.Generator) -> np.ndarray:
    """Veto ALÉATOIRE portant sur exactement `n_veto` cellules parmi celles où le CTA
    a un signal -- couverture identique à celle du filtre comparé, jamais approchée.
    C'est le contrôle canonique du méta-labeling : retirer des trades au hasard améliore
    parfois le PnL par chance ; F2 ne « marche » que s'il bat ça à couverture égale."""
    w = cta.copy().astype(float)
    eligible = np.flatnonzero(cta != 0)
    if n_veto > eligible.size:
        raise ValueError(f"veto demandé ({n_veto}) > cellules éligibles ({eligible.size})")
    w[rng.choice(eligible, size=n_veto, replace=False)] = 0.0
    return w


# ── 3. données (lecture seule) ─────────────────────────────────────────────

def load_geometry(db_path: str = DB_PATH) -> pd.DataFrame:
    """Géométrie weekly W+1 des deux modèles sur la grille oos2020. LECTURE SEULE."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            "SELECT model, asset, cutoff_date, target_date, last_close, y_pred, "
            "y_lower, y_upper, y_true FROM predictions "
            "WHERE source = ? AND horizon_unit = ? AND model IN (?, ?) "
            "ORDER BY model, asset, cutoff_date",
            conn, params=(SOURCE, HORIZON_UNIT, PRIMARY_MODEL, CONTROL_MODEL))
    finally:
        conn.close()


def load_signal(assets, signal_dir: Path = SIGNAL_DIR) -> dict:
    """Signal CTA gelé, lu par date EXACTE à l'origine -- aucun ffill qui pourrait
    glisser une valeur postérieure (même convention qu'attach_signal de cta_gate0)."""
    return {a: pd.read_parquet(Path(signal_dir) / f"{slug(a)}.parquet")["signal"]
            for a in assets}


def build_cells(geo: pd.DataFrame, signal: dict, model: str) -> dict:
    """{asset: DataFrame(cutoff_date, last_close, y_true, cta, state)} pour un modèle."""
    out = {}
    for asset, g in geo[geo["model"] == model].groupby("asset"):
        g = g.sort_values("cutoff_date").reset_index(drop=True)
        cta = np.array([signal[asset].get(pd.Timestamp(d), np.nan) for d in g["cutoff_date"]])
        keep = ~np.isnan(cta)
        g, cta = g[keep].reset_index(drop=True), np.sign(cta[keep])
        g["cta"] = cta
        g["state"] = [tc_state(r.last_close, r.y_pred, r.y_lower, r.y_upper)
                      for r in g.itertuples()]
        out[asset] = g
    return out


def instruments_for(assets) -> dict:
    """Un instrument par actif (le premier déclaré dans real_fees), comme cta_gate0."""
    one = {}
    for inst, spec in rf.INSTRUMENTS.items():
        if spec["asset"] in set(assets):
            one.setdefault(spec["asset"], inst)
    return one


# ── 4. bras : PnL net par origine, portefeuille équipondéré ─────────────────

def arm_pnl(cells: dict, instruments: dict, weight_fn) -> dict:
    """`weight_fn(df) -> np.ndarray` donne les poids d'un actif. Retourne le PnL net par
    origine du portefeuille équipondéré + les séries par actif."""
    per_asset, active = {}, {}
    for asset, inst in sorted(instruments.items()):
        df = cells[asset]
        w = np.asarray(weight_fn(df), dtype=float)
        r = eb.gross_returns(df["last_close"].to_numpy(float), df["y_true"].to_numpy(float))
        cost = rf.one_way_total_bps(inst, HORIZON_WEEKS, COST_LEVEL)
        per_asset[asset] = eb.sleeve_pnl(w, r, cost)
        active[asset] = (w != 0)
    n = min(len(v) for v in per_asset.values())
    port = np.mean([per_asset[a][:n] for a in sorted(per_asset)], axis=0)
    n_active = int(sum(active[a][:n].sum() for a in active))
    n_cells = int(sum(active[a][:n].size for a in active))
    cutoffs = cells[sorted(instruments)[0]]["cutoff_date"].to_numpy()[:n]
    return {"portfolio": port, "per_asset": {a: v[:n] for a, v in per_asset.items()},
            "n_active": n_active, "n_cells": n_cells,
            "coverage": n_active / n_cells if n_cells else 0.0, "cutoffs": cutoffs}


def bh_pnl(cells: dict, instruments: dict) -> dict:
    return arm_pnl(cells, instruments, lambda df: np.ones(len(df)))


def filter_weights(df: pd.DataFrame, filter_name: str) -> np.ndarray:
    return np.array([decide(s, c, filter_name) for s, c in zip(df["state"], df["cta"])])


# ── 5. lectures ────────────────────────────────────────────────────────────

def paired_vs(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    if d.size < 5 or np.allclose(d, 0):
        return {"mean_diff_bps": float(d.mean() * BPS), "p_value": float("nan"),
                "significant_at_05": False,
                "note": "différence identiquement nulle" if np.allclose(d, 0) else "n < 5"}
    t = paired_block_bootstrap_test(d, block_length=min(BLOCK_LENGTH, d.size), seed=POOL_SEED)
    return {"mean_diff_bps": float(d.mean() * BPS), "p_value": float(t["p_value"]),
            "significant_at_05": bool(t["significant_at_05"]),
            "ci95_bps": [float(t["ci95_lo"] * BPS), float(t["ci95_hi"] * BPS)],
            "effective_n": float(t["effective_n"]),
            "n_origins_differing": int((np.abs(d) > 1e-15).sum())}


def selective_reading(cells: dict, instruments: dict, filter_name: str) -> dict:
    """Lecture sélective (descriptive) : qualité des trades PRIS vs VETOÉS. Toujours
    rapportée en couple (couverture, qualité) -- jamais l'une sans l'autre."""
    taken, vetoed, by_state = [], [], {}
    for asset, inst in sorted(instruments.items()):
        df = cells[asset]
        w = filter_weights(df, filter_name)
        cta = df["cta"].to_numpy(float)
        r = eb.gross_returns(df["last_close"].to_numpy(float), df["y_true"].to_numpy(float))
        cost = rf.one_way_total_bps(inst, HORIZON_WEEKS, COST_LEVEL)
        pnl_if_taken = eb.sleeve_pnl(cta, r, cost)     # PnL du trade CTA, pris ou non
        for i, state in enumerate(df["state"]):
            if cta[i] == 0:
                continue
            (taken if w[i] != 0 else vetoed).append(pnl_if_taken[i])
            by_state.setdefault(state, {"taken": [], "vetoed": []})
            by_state[state]["taken" if w[i] != 0 else "vetoed"].append(pnl_if_taken[i])

    def summarize(x):
        x = np.asarray(x, dtype=float)
        return {"n": int(x.size),
                "pnl_moyen_bps": float(x.mean() * BPS) if x.size else None,
                "hit_rate": float((x > 0).mean()) if x.size else None}

    return {"pris": summarize(taken), "vetoes": summarize(vetoed),
            "couverture_prise": len(taken) / max(1, len(taken) + len(vetoed)),
            "par_etat": {s: {"pris": summarize(v["taken"]), "vetoes": summarize(v["vetoed"])}
                         for s, v in sorted(by_state.items())}}


def by_period(pnl: np.ndarray, cutoffs: np.ndarray) -> dict:
    dates = pd.to_datetime(pd.Series(cutoffs))
    out = {}
    for label, (start, end) in PERIODS.items():
        m = (dates >= start).to_numpy() & (dates <= end).to_numpy()
        out[label] = {"n": int(m.sum()),
                      "pnl_moyen_bps": float(pnl[m].mean() * BPS) if m.any() else None}
    return out


def placebo_study(cells: dict, instruments: dict, target_coverage_veto: int,
                  c0_port: np.ndarray, n_draws: int = N_PLACEBO) -> dict:
    """R : `n_draws` vetos aléatoires à couverture EXACTEMENT égale à celle de F2,
    graine fixée. Rapporte la distribution du PnL et le rang de F2 dedans."""
    rng = np.random.default_rng(PLACEBO_SEED)
    per_asset_veto = {}
    for asset in sorted(instruments):
        df = cells[asset]
        w2 = filter_weights(df, "F2")
        per_asset_veto[asset] = int(((df["cta"].to_numpy() != 0) & (w2 == 0)).sum())
    means = []
    for _ in range(n_draws):
        per_asset_pnl = {}
        for asset, inst in sorted(instruments.items()):
            df = cells[asset]
            w = placebo_weights(df["cta"].to_numpy(float), per_asset_veto[asset], rng)
            r = eb.gross_returns(df["last_close"].to_numpy(float), df["y_true"].to_numpy(float))
            cost = rf.one_way_total_bps(inst, HORIZON_WEEKS, COST_LEVEL)
            per_asset_pnl[asset] = eb.sleeve_pnl(w, r, cost)
        n = min(len(v) for v in per_asset_pnl.values())
        means.append(float(np.mean([per_asset_pnl[a][:n] for a in sorted(per_asset_pnl)],
                                   axis=0).mean() * BPS))
    return {"n_draws": n_draws, "seed": PLACEBO_SEED,
            "veto_par_actif": per_asset_veto,
            "pnl_moyen_bps": {"mediane": float(np.median(means)),
                              "p05": float(np.percentile(means, 5)),
                              "p95": float(np.percentile(means, 95)),
                              "min": float(np.min(means)), "max": float(np.max(means))},
            "_means": means}


def power_analysis(c0: np.ndarray, arms: dict, effective_n: int = 113) -> dict:
    """Analyse de puissance, calculée AVANT de lire les verdicts (§4 du brief) :
    l'écart détectable à effective_n, en tenant compte du fait qu'un filtre ne diffère
    de C0 que sur les origines qu'il modifie."""
    out = {"effective_n": effective_n, "n_origins": int(c0.size)}
    for name, port in arms.items():
        d = port - c0
        n_diff = int((np.abs(d) > 1e-15).sum())
        sd = float(np.std(d, ddof=1))
        se = sd / np.sqrt(effective_n)
        out[name] = {
            "origines_modifiees": n_diff,
            "part_origines_modifiees": n_diff / max(1, d.size),
            "ecart_observe_bps": float(d.mean() * BPS),
            "se_bps": float(se * BPS),
            # 2.8 ~ z(0.975) + z(0.80) : écart minimal détectable à 80 % de puissance
            "mde_80_bps": float(2.8 * se * BPS),
            "detectable": bool(abs(d.mean()) > 2.8 * se),
        }
    return out


# ── 6. main ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--signal-dir", default=str(SIGNAL_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    geo = load_geometry(args.db_path)
    assets = sorted(geo["asset"].unique())
    signal = load_signal(assets, Path(args.signal_dir))
    instruments = instruments_for(assets)

    cells = {m: build_cells(geo, signal, m) for m in (PRIMARY_MODEL, CONTROL_MODEL)}
    n_origins = cells[PRIMARY_MODEL][assets[0]]["cutoff_date"].nunique()
    print(f"grille {SOURCE} {HORIZON_UNIT} : {n_origins} origines, {len(assets)} actifs, "
          f"frais niveau {COST_LEVEL} (roulement inclus)")

    etats = {m: pd.concat([c["state"] for c in cells[m].values()]).value_counts().to_dict()
             for m in cells}
    for m, v in etats.items():
        print(f"  états TC {m:12s} : {v}")

    arms = {"C0": arm_pnl(cells[PRIMARY_MODEL], instruments,
                          lambda df: filter_weights(df, "C0"))}
    for tag, model in (("F", PRIMARY_MODEL), ("G", CONTROL_MODEL)):
        for f in ("F1", "F2"):
            arms[f"{tag}{f[-1]}"] = arm_pnl(cells[model], instruments,
                                            lambda df, _f=f: filter_weights(df, _f))
    bh = bh_pnl(cells[PRIMARY_MODEL], instruments)

    c0 = arms["C0"]["portfolio"]
    power = power_analysis(c0, {k: v["portfolio"] for k, v in arms.items() if k != "C0"})
    print("\n--- analyse de puissance (avant lecture des verdicts) ---")
    for k, v in power.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['origines_modifiees']}/{power['n_origins']} origines modifiées, "
                  f"écart {v['ecart_observe_bps']:+.2f} bps, MDE(80%) {v['mde_80_bps']:.1f} bps, "
                  f"détectable={v['detectable']}")

    n_veto_f2 = arms["C0"]["n_active"] - arms["F2"]["n_active"]
    placebo = placebo_study(cells[PRIMARY_MODEL], instruments, n_veto_f2, c0)
    f2_mean = float(arms["F2"]["portfolio"].mean() * BPS)
    rank = float(np.mean([m >= f2_mean for m in placebo["_means"]]))
    placebo["f2_pnl_bps"] = f2_mean
    placebo["part_placebos_meilleurs_que_F2"] = rank
    placebo["F2_bat_le_placebo"] = bool(rank < 0.05)
    del placebo["_means"]

    tests = {}
    for name in ("F1", "F2"):
        tests[f"{name} vs C0"] = paired_vs(arms[name]["portfolio"], c0)
    famille = correct_family({k: {"p_value": v["p_value"]} for k, v in tests.items()})

    explor = {f"{k} vs C0": paired_vs(arms[k]["portfolio"], c0) for k in ("G1", "G2")}
    explor["C0 vs B&H"] = paired_vs(c0, bh["portfolio"])
    explor["F2 vs B&H"] = paired_vs(arms["F2"]["portfolio"], bh["portfolio"])

    payload = {
        "scope": "méta-labeling artisanal : CTA corrigé filtré par la taxonomie TC",
        "grid": {"source": SOURCE, "horizon_unit": HORIZON_UNIT, "n_origins": int(n_origins),
                 "assets": assets, "instruments": instruments,
                 "effective_n": 113, "cost_level": COST_LEVEL},
        "signal": {"dir": str(args.signal_dir),
                   "retenu": "direction Hull, calendrier own (bug 2 évité ; bug 1 hors "
                             "chemin car trend_direction est univariée)"},
        "precedence_etats": "Stress > Sideways > Calm (déclarée avant les runs)",
        "etats_tc": etats,
        "bras": {k: {"pnl_moyen_bps": float(v["portfolio"].mean() * BPS),
                     "trades_actifs": v["n_active"], "cellules": v["n_cells"],
                     "couverture_prise": v["coverage"],
                     "sharpe_annualise": eb.sharpe(v["portfolio"], HORIZON_WEEKS),
                     "max_drawdown": eb.max_drawdown(v["portfolio"]),
                     "par_regime": by_period(v["portfolio"], v["cutoffs"])}
                 for k, v in arms.items()},
        "buy_and_hold": {"pnl_moyen_bps": float(bh["portfolio"].mean() * BPS)},
        "analyse_de_puissance": power,
        "hypothese_primaire": {
            "famille_holm_m2": famille,
            "placebo": placebo,
            "conjonctive": "F2 bat C0 (Holm) ET F2 bat le placebo à couverture égale",
        },
        "exploratoire_non_decisionnel": explor,
        "lecture_selective": {name: selective_reading(cells[PRIMARY_MODEL], instruments, name)
                              for name in ("F1", "F2")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
