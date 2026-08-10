"""
cta_gate0.py -- chantier 0 du BRIEF « couplage CTA (DEITA) x sizing NsDiff » :
LA PORTE D'ENTREE. Le signal de tendance vit-il encore sur cette fenetre ?

POURQUOI CETTE PORTE EXISTE. La decroissance du momentum apres 2010 est un fait
documente de la litterature ; sizer finement un signal mort serait un exercice
vide. Le brief en fait donc une condition BLOQUANTE : pas de couplage avant
d'avoir montre que le signal a un edge. Cette porte se juge AVANT d'avoir regarde
le moindre resultat de sizing, et son critere est declare ci-dessous.

CE QUI EST TESTE, et rien d'autre : le CTA SEUL, taille fixe |w| = 1, signe donne
par le signal gele du chantier 0a (direction Hull, cf. sa docstring pour la
decision d'interface). Grille `oos2020` -- 340 origines, 7 actifs, regime weekly,
prix geles `prices_v3`. PnL net de frais `real_fees`, cout de roulement H2 inclus
pour les futures. Horizon de detention W+1 ; W+2 et W+3 en descriptif.

CRITERE DE PORTE, DECLARE AVANT LE RUN (repris du brief mot pour mot) :
  au moins un actif avec PnL net positif ET p < 0,05 BRUT (pas de Holm ici : c'est
  une porte, pas une conclusion), OU un Sharpe poole > 0 avec direction coherente
  sur les classes d'actifs.
« Direction coherente » est operationnalise ici, avant lecture : au moins 3 des 4
classes (Equity, Bond, Crypto, Commodity) positives.
Si la porte echoue : le momentum est mort sur cette fenetre pour ce panel, le
programme s'arrete avant tout couplage, et c'est la conclusion.

BRANCHE 2 REPAREE (PATCH_gate0_branche2_et_holm_m2.md, P2), posterieurement au
verdict et sans le modifier : elle porte sur l'EXCES vs acheter-et-garder, pas sur
le PnL brut -- cf. `branch2_verdict`. La formulation d'origine reste calculee et
rapportee dans le JSON, parce que c'est elle que la NOTE cite et dont le defaut est
documente. Les trois variantes de signal ont ete rejouees sous la branche reparee
et restent toutes en echec (`patch_gate0_branche2_holm.json`).

CONTRE QUOI. Acheter-et-garder, sur les memes origines, aux memes frais. C'est la
comparaison qui compte pour un CTA : un signal toujours long ne serait pas un
signal, ce serait un B&H deguise -- et c'est exactement ce que le chantier 0a a
ecarte en rejetant la conviction hierarchique.

Sortie : experiments/cta_gate0.json
Usage   : python cta_gate0.py
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
from deita_cta_signal import ASSET_MAP, OUT_DIR as SIGNAL_DIR         # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402
from prices_v3 import slug                                            # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "cta_gate0.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
BPS = 1e4
BLOCK_LENGTH = 3
POOL_SEED = 42
HORIZON_WEEKS = {"W+1": 1, "W+2": 2, "W+3": 3}
PRIMARY_HORIZON = "W+1"
REGIME = "weekly"
MIN_CLASSES_POSITIVE = 3
# Tranches de marche, declarees : le trend doit montrer son « crisis alpha » en
# 2020 et 2022 ou nulle part.
PERIODS = {"2020 (COVID)": ("2020-01-01", "2020-12-31"),
           "2021": ("2021-01-01", "2021-12-31"),
           "2022 (bear taux)": ("2022-01-01", "2022-12-31"),
           "2023": ("2023-01-01", "2023-12-31"),
           "2024-2026": ("2024-01-01", "2026-12-31")}


def load_market(grid_dir: Path) -> pd.DataFrame:
    """Prix courants et cibles par cellule, lus sur la grille regeneree. Tous les
    bras du benchmark partagent ces colonnes -- il n'y a pas deux sources a
    reconcilier."""
    g = pd.read_parquet(grid_dir / "ARIMA-GARCH" / "bands.parquet")
    return g[["asset", "frequence", "horizon_unit", "cutoff_date", "target_date",
              "last_close", "y_true"]].copy()


def load_signal(assets, which: str = "trend", signal_dir: Path = None) -> pd.DataFrame:
    """Signal gele, en tableau large indexe par date.

    `which="conviction"` relit la conviction hierarchique archivee au chantier 0a
    -- celle qui degenere en « toujours long » sur 4 actifs sur 7. Elle n'est la
    que pour un CONTROLE DE ROBUSTESSE du verdict de porte : si la porte echoue
    avec les deux signaux, l'arret ne depend pas de la decision d'interface."""
    d = Path(signal_dir or SIGNAL_DIR)
    if which == "conviction":
        conv = pd.read_parquet(d / "conviction.parquet")
        return conv[[a for a in assets if a in conv.columns]].sort_index()
    return pd.DataFrame({a: pd.read_parquet(d / f"{slug(a)}.parquet")["signal"]
                         for a in assets}).sort_index()


def attach_signal(market: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Le signal a l'origine t, et rien d'autre : lecture par date exacte, aucun
    remplissage vers l'avant qui pourrait glisser une valeur posterieure."""
    idx = pd.to_datetime(market["cutoff_date"])
    out = market.copy()
    out["signal"] = [signal[a].get(d, np.nan) for a, d in zip(market["asset"], idx)]
    return out.dropna(subset=["signal"])


def cell_series(df: pd.DataFrame, asset: str, hu: str) -> dict:
    s = df[(df["asset"] == asset) & (df["frequence"] == REGIME)
           & (df["horizon_unit"] == hu)].sort_values("cutoff_date")
    return {"cutoff_date": s["cutoff_date"].to_numpy(),
            **{c: s[c].to_numpy(dtype=float) for c in ("last_close", "y_true", "signal")}}


def evaluate(d: dict, inst: str, hw: int, level: str) -> dict:
    """CTA a taille fixe contre acheter-et-garder, memes origines, memes frais."""
    cost = rf.one_way_total_bps(inst, hw, level)
    r = eb.gross_returns(d["last_close"], d["y_true"])
    w_cta = np.sign(d["signal"])
    pnl_cta = eb.sleeve_pnl(w_cta, r, cost)
    pnl_bh = eb.sleeve_pnl(np.ones_like(r), r, cost)
    diff = pnl_cta - pnl_bh
    t = (paired_block_bootstrap_test(diff, block_length=min(BLOCK_LENGTH, diff.size),
                                     seed=POOL_SEED) if diff.size >= 5 else {})
    # Seconde lecture du critere de porte, calculee ET declaree : le brief ecrit
    # « PnL net positif ET p < 0,05 brut » juste apres avoir decrit un test VS B&H.
    # « p » peut donc se lire de deux facons -- contre B&H, ou contre zero (le
    # signal a-t-il un edge en absolu ?). Les deux sont rapportees ; le verdict de
    # porte suit la lecture du brief (vs B&H), l'autre eclaire la note.
    t0_ = (paired_block_bootstrap_test(pnl_cta, block_length=min(BLOCK_LENGTH, pnl_cta.size),
                                       seed=POOL_SEED) if pnl_cta.size >= 5 else {})
    return {
        "p_value_vs_zero": float(t0_.get("p_value", np.nan)),
        "significant_raw_vs_zero": bool(t0_.get("significant_at_05", False)),
        "n": int(r.size), "round_trip_bps": rf.total_round_trip_bps(inst, hw, level),
        "pnl_net_cta_bps": float(pnl_cta.mean() * BPS),
        "pnl_net_bh_bps": float(pnl_bh.mean() * BPS),
        "edge_vs_bh_bps": float(diff.mean() * BPS),
        "share_long": float((w_cta > 0).mean()), "share_short": float((w_cta < 0).mean()),
        "hit_rate": float((pnl_cta > 0).mean()),
        "sharpe_annualised": eb.sharpe(pnl_cta, hw),
        "sharpe_bh_annualised": eb.sharpe(pnl_bh, hw),
        "max_drawdown": eb.max_drawdown(pnl_cta),
        "p_value_vs_bh": float(t.get("p_value", np.nan)),
        "significant_raw": bool(t.get("significant_at_05", False)),
        "verdict_vs_bh": ("indistinguishable" if not t.get("significant_at_05")
                          else "cta_better" if t.get("mean_diff", 0) > 0 else "bh_better"),
        "_pnl": pnl_cta, "_pnl_bh": pnl_bh, "_cutoff": d["cutoff_date"],
    }


def branch2_verdict(excess_by_instrument: dict, class_of: dict, hw: int = 1,
                    min_classes: int = MIN_CLASSES_POSITIVE) -> dict:
    """BRANCHE 2, VERSION REPAREE (PATCH_gate0_branche2_et_holm_m2.md, P2).

    La formulation d'origine -- « Sharpe poole > 0 et >= 3 classes sur 4 a PnL
    positif » -- est franchissable par un signal CONSTANT : sur un panel haussier,
    le Sharpe poole du PnL brut mesure la pente du marche, pas l'apport du signal.
    C'est prouve, pas suppose : la conviction degeneree de DEITA (toujours longue,
    donc acheter-et-garder deguise, ecart de PnL exactement 0,00 bps et p = 1,000)
    franchissait cette branche avec un Sharpe de 0,77 et 3 classes sur 4.

    La branche porte desormais sur l'EXCES par origine -- PnL du signal moins PnL
    d'acheter-et-garder, apparie origine par origine. Un signal constant a un exces
    identiquement nul : son Sharpe d'exces n'est pas defini (ecart-type nul) et
    aucune classe n'est positive. L'echec devient mecanique, quelle que soit la
    pente du marche. C'est la seule modification : le seuil (3 classes sur 4) et
    l'esprit du critere sont inchanges.
    """
    keys = sorted(excess_by_instrument)
    n = min(len(excess_by_instrument[k]) for k in keys)
    port = np.mean([excess_by_instrument[k][:n] for k in keys], axis=0)
    per_class = {}
    for cls in sorted({class_of[k] for k in keys}):
        members = [k for k in keys if class_of[k] == cls]
        v = float(np.mean([excess_by_instrument[k][:n].mean() for k in members]) * BPS)
        per_class[cls] = {"instruments": members, "excess_mean_bps": v,
                          "positive": bool(v > 0)}
    sharpe = eb.sharpe(port, hw)
    n_pos = sum(1 for v in per_class.values() if v["positive"])
    return {
        "n_origins": int(n), "instruments": keys,
        "excess_mean_bps": float(port.mean() * BPS),
        "sharpe_excess_annualised": sharpe,
        "per_class": per_class, "n_classes_positive": n_pos,
        "min_classes_required": min_classes,
        "passes": bool(np.isfinite(sharpe) and sharpe > 0 and n_pos >= min_classes),
        "formulation": "Sharpe poole de l'EXCES (PnL signal - PnL B&H, par origine) > 0 "
                       f"ET >= {min_classes} classes sur 4 a exces positif",
    }


def by_period(pnl: np.ndarray, cutoffs: np.ndarray, hw: int) -> dict:
    dates = pd.to_datetime(pd.Series(cutoffs))
    out = {}
    for name, (a, b) in PERIODS.items():
        m = ((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))).to_numpy()
        if m.sum() < 3:
            continue
        out[name] = {"n": int(m.sum()), "pnl_mean_bps": float(pnl[m].mean() * BPS),
                     "sharpe": eb.sharpe(pnl[m], hw)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--level", default=rf.DECISION_LEVEL, choices=list(rf.LEVELS))
    p.add_argument("--signal-dir", default=None,
                   help="dossier du signal gele (defaut : convention DEITA)")
    p.add_argument("--signal", default="trend", choices=["trend", "conviction"],
                   help="'trend' = signal retenu ; 'conviction' = controle de robustesse")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    market = load_market(Path(args.grid_dir))
    assets = sorted(market["asset"].unique())
    df = attach_signal(market, load_signal(assets, args.signal, args.signal_dir))
    n_origins = df["cutoff_date"].nunique()
    print(f"grille : {n_origins} origines, {len(assets)} actifs, regime {REGIME}, "
          f"frais niveau {args.level} (roulement H2 inclus)")

    cells, pooled_pnl, pooled_excess = {}, {}, {}
    for inst, spec in rf.INSTRUMENTS.items():
        asset = spec["asset"]
        if asset not in assets:
            continue
        for hu, hw in HORIZON_WEEKS.items():
            d = cell_series(df, asset, hu)
            if d["y_true"].size == 0:
                continue
            r = evaluate(d, inst, hw, args.level)
            r["periods"] = by_period(r["_pnl"], r["_cutoff"], hw)
            r["instrument"], r["asset"], r["horizon_unit"] = inst, asset, hu
            r["classe"] = ASSET_MAP[asset]["sector"]
            if hu == PRIMARY_HORIZON:
                pooled_pnl[inst] = r["_pnl"]
                pooled_excess[inst] = r["_pnl"] - r["_pnl_bh"]
            cells[f"{inst}|{hu}"] = r

    # ── portefeuille equipondere, horizon primaire : un instrument par actif ──
    one_per_asset = {}
    for inst in pooled_pnl:
        a = rf.INSTRUMENTS[inst]["asset"]
        one_per_asset.setdefault(a, inst)
    keys = sorted(one_per_asset.values())
    n = min(len(pooled_pnl[k]) for k in keys)
    port = np.mean([pooled_pnl[k][:n] for k in keys], axis=0)
    pooled = {
        "instruments": keys, "n_origins": int(n),
        "pnl_mean_bps": float(port.mean() * BPS),
        "sharpe_annualised": eb.sharpe(port, HORIZON_WEEKS[PRIMARY_HORIZON]),
        "max_drawdown": eb.max_drawdown(port),
    }
    by_class = {}
    for cls in sorted({ASSET_MAP[rf.INSTRUMENTS[k]["asset"]]["sector"] for k in keys}):
        members = [k for k in keys if ASSET_MAP[rf.INSTRUMENTS[k]["asset"]]["sector"] == cls]
        v = float(np.mean([pooled_pnl[k][:n].mean() for k in members]) * BPS)
        by_class[cls] = {"instruments": members, "pnl_mean_bps": v, "positive": bool(v > 0)}

    # ── le critere de porte, applique tel qu'il est declare ──────────────────
    branch1 = sorted(k for k, c in cells.items()
                     if c["horizon_unit"] == PRIMARY_HORIZON
                     and c["pnl_net_cta_bps"] > 0 and c["significant_raw"])
    branch1_alt = sorted(k for k, c in cells.items()
                         if c["horizon_unit"] == PRIMARY_HORIZON
                         and c["pnl_net_cta_bps"] > 0 and c["significant_raw_vs_zero"])
    n_classes_pos = sum(1 for v in by_class.values() if v["positive"])
    class_of = {k: ASSET_MAP[rf.INSTRUMENTS[k]["asset"]]["sector"] for k in keys}
    b2 = branch2_verdict({k: pooled_excess[k] for k in keys}, class_of,
                         HORIZON_WEEKS[PRIMARY_HORIZON])
    branch2 = b2["passes"]
    # Formulation d'origine, conservee pour la tracabilite : c'est elle que la NOTE
    # cite, et c'est elle dont le defaut est documente. Rapportee, jamais decisive.
    branch2_original = bool(pooled["sharpe_annualised"] > 0
                            and n_classes_pos >= MIN_CLASSES_POSITIVE)
    gate = bool(branch1) or branch2

    payload = {
        "scope": "chantier 0 -- porte d'entree : le signal CTA a-t-il un edge sur la fenetre",
        "declared_criterion": {
            "branch_1": "au moins un actif a PnL net positif ET p < 0,05 brut (pas de Holm : "
                        "c'est une porte, pas une conclusion)",
            "branch_2": "REPAREE : Sharpe poole de l'EXCES (PnL signal - PnL B&H, par "
                        "origine) > 0 ET au moins 3 des 4 classes a exces positif. La "
                        "formulation d'origine portait sur le PnL brut et etait franchissable "
                        "par un signal constant (cf. PATCH_gate0_branche2_et_holm_m2.md).",
            "on_failure": "le momentum est mort sur cette fenetre pour ce panel ; le programme "
                          "s'arrete avant tout couplage, et c'est la conclusion",
        },
        "setup": {"grid": str(args.grid_dir), "n_origins": n_origins, "assets": assets,
                  "regime": REGIME, "position": "taille fixe |w| = 1, signe = signal gele",
                  "fees_level": args.level, "roll_included": True,
                  "signal": str(args.signal_dir or SIGNAL_DIR), "signal_variant": args.signal,
                  "primary_horizon": PRIMARY_HORIZON},
        "per_cell": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in cells.items()},
        "pooled_portfolio": pooled, "by_class": by_class,
        "gate": {"branch_1_hits": branch1, "branch_1_passes": bool(branch1),
                 "branch_2": b2,
                 "branch_2_original_formulation": {
                     "passes": branch2_original,
                     "sharpe_pnl_brut": pooled["sharpe_annualised"],
                     "n_classes_positive": n_classes_pos,
                     "why_replaced": "franchissable par un signal constant -- cf. "
                                     "PATCH_gate0_branche2_et_holm_m2.md, P2. Rapportee pour la "
                                     "tracabilite, jamais decisive."},
                 "branch_1_alt_reading_vs_zero": {
                     "hits": branch1_alt, "passes": bool(branch1_alt),
                     "note": "lecture concurrente de « p < 0,05 » : contre zero au lieu de contre "
                             "B&H. Rapportee, non decisive -- le brief decrit le test vs B&H juste "
                             "avant d'enoncer le critere."},
                 "n_classes_positive": n_classes_pos, "branch_2_passes": branch2,
                 "passes": gate},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"\n=== CTA seul, taille fixe, horizon {PRIMARY_HORIZON} ===")
    print(f"  {'instrument':<12}{'long':>7}{'PnL net':>10}{'vs B&H':>10}{'hit':>7}"
          f"{'Sharpe':>8}{'p':>8}  verdict")
    for k in sorted(cells):
        c = cells[k]
        if c["horizon_unit"] != PRIMARY_HORIZON:
            continue
        print(f"  {c['instrument']:<12}{c['share_long']:>6.0%}{c['pnl_net_cta_bps']:>10.2f}"
              f"{c['edge_vs_bh_bps']:>10.2f}{c['hit_rate']:>7.1%}"
              f"{c['sharpe_annualised']:>8.2f}{c['p_value_vs_bh']:>8.3f}  {c['verdict_vs_bh']}")
    print(f"\n  portefeuille equipondere ({len(keys)} instruments) : "
          f"PnL {pooled['pnl_mean_bps']:+.2f} bps/origine, Sharpe {pooled['sharpe_annualised']:.2f}")
    for cls, v in by_class.items():
        print(f"    {cls:<11}{v['pnl_mean_bps']:+8.2f} bps  {'positif' if v['positive'] else 'negatif'}")

    print(f"\n=== comportement par tranche (portefeuille : SPY-ETF, reference) ===")
    ref = cells.get(f"SPY-ETF|{PRIMARY_HORIZON}")
    if ref:
        for name, v in ref["periods"].items():
            print(f"  {name:<20}n={v['n']:>3}  PnL {v['pnl_mean_bps']:+8.2f} bps  "
                  f"Sharpe {v['sharpe']:>6.2f}")

    print(f"\n  branche 1 (un actif a PnL positif et p < 0,05 vs B&H) : "
          f"{'PASS -- ' + ', '.join(branch1) if branch1 else 'echec'}")
    print(f"    lecture concurrente (p < 0,05 vs zero) : "
          f"{'aurait passe -- ' + ', '.join(branch1_alt) if branch1_alt else 'echec aussi'}")
    print(f"  branche 2 (Sharpe de l'EXCES > 0 et >= {MIN_CLASSES_POSITIVE} classes positives) : "
          f"{'PASS' if branch2 else 'echec'} "
          f"(Sharpe exces {b2['sharpe_excess_annualised']:.2f}, "
          f"{b2['n_classes_positive']}/4 classes, exces moyen "
          f"{b2['excess_mean_bps']:+.2f} bps)")
    print(f"    formulation d'origine (PnL brut), pour memoire : "
          f"{'aurait passe' if branch2_original else 'echec aussi'} "
          f"(Sharpe {pooled['sharpe_annualised']:.2f}, {n_classes_pos}/4)")
    print(f"  >>> PORTE 0 : {'FRANCHIE' if gate else 'ECHEC -- le programme s arrete ici'}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
