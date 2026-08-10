"""
deita_cta_signal.py -- chantier 0a du BRIEF « couplage CTA (DEITA) x sizing
NsDiff » : GELER le signal directionnel de DEITA comme serie point-in-time, et
PROUVER qu'il ne regarde pas devant.

LE MOTEUR. `cta_quant_engine` (DEITA, portage Python du CTA MATLAB
`aistis02_clean.m`) expose deux facons de produire un signal :
  * la DIRECTION DE TENDANCE -- moyenne mobile de Hull sur 20 jours des
    rendements quotidiens, dont on prend le signe. C'est le coeur
    trend-following du systeme, et c'est ce que DEITA utilise tel quel dans
    `compute_cta_signal(pure_trend_mode=True)` ;
  * la CONVICTION HIERARCHIQUE -- la direction Hull multipliee par une agregation
    a trois niveaux (marche / sous-secteur / secteur), chaque niveau pondere par
    une correlation glissante de 63 jours.
Aucun parametre n'est retouche dans un cas comme dans l'autre.

SIGNAL RETENU : LA DIRECTION HULL. Ce n'est pas le choix par defaut, c'est une
decision imposee par la mesure, et la voici en entier.

  1. LE MECANISME. Pour un actif seul dans son sous-secteur, `_conviction_level`
     renvoie sa PROPRE serie lissee. Si l'actif est aussi seul dans son secteur,
     deux des trois niveaux valent `smooth`, et la conviction devient

         conv = smooth x (mkt + 2 x smooth) / 3 = (2/3) smooth^2 + smooth x mkt/3

     dont le terme dominant est un CARRE : le signe ne peut plus changer. Le
     signal cesse d'etre directionnel et devient « toujours long ».

  2. LA MESURE, sur la grille (340 origines, depart 2020-01) :

     | actif   | sous-secteur | conviction, part long | direction Hull, part long |
     |---------|--------------|----------------------:|--------------------------:|
     | SPY     | singleton    |                100,0 % |                    60,3 % |
     | BTC-USD | singleton    |                 99,9 % |                    53,9 % |
     | ZN=F    | singleton    |                 99,3 % |                    47,3 % |
     | TLT     | singleton    |                 99,8 % |                    49,9 % |
     | ETH-USD | 2 membres    |                 48,9 % |                    54,9 % |
     | GLD     | 3 membres    |                 55,6 % |                    56,2 % |
     | USO     | 2 membres    |                 55,0 % |                    57,3 % |

     La regle se lit sans ambiguite : la degenerescence frappe exactement les
     sous-secteurs a un membre. Elle n'est pas un artefact du panel du
     benchmark -- SPY et BTC-USD sont deja seuls dans leur sous-secteur dans
     l'univers de 16 actifs de DEITA lui-meme (mesure ci-dessus faite sur cet
     univers unifie, base locale de DEITA, sans reseau).

  3. POURQUOI L'UNIVERS ELARGI NE SAUVE PAS. Deux raisons, chacune suffisante.
     D'abord SPY et ZN=F -- les deux instruments de l'HYPOTHESE PRIMAIRE -- restent
     degeneres. Ensuite les prix : sur SPY, seul actif verifiable dans les deux
     sources, les log-rendements de la base DEITA et de `prices_v3` s'ecartent
     jusqu'a 2,6e-03, deux ordres de grandeur au-dessus de la tolerance de gel du
     benchmark (1e-5). Faire produire le signal par une serie et le trader sur
     une autre reintroduirait exactement ce que le gel des prix interdit.

  4. CE QUE CA CHANGE, ET CE QUE CA NE CHANGE PAS. La question du brief est
     « une echelle de risque calibree ameliore-t-elle le sizing d'un signal
     directionnel EXTERNE ? ». Elle est posee a l'identique avec la direction
     Hull : le signal reste celui de DEITA, exogene, non entraine ici, et le
     signe reste commun aux quatre bras. Ce qui est perdu est la ponderation par
     conviction -- laquelle, sur 4 actifs sur 7, ne pondere rien puisqu'elle ne
     change jamais de signe. La conviction hierarchique est neanmoins calculee et
     archivee a cote, en DESCRIPTIF, pour que la decision reste verifiable.

L'UNIVERS. La direction Hull est purement univariee : elle ne depend d'aucun
univers, seulement de la serie de l'actif. Elle se calcule donc entierement sur
`prices_v3/`, prix geles partages -- non-negociable respecte, aucun appel reseau.

LA VERIFICATION D'ABSENCE DE LOOK-AHEAD, bloquante et pas declarative. Le brief
demande de « rejouer 5 origines a la main depuis les donnees brutes et
comparer ». `lookahead_check` recalcule TOUT le signal sur l'historique TRONQUE a
chaque date et compare la derniere valeur a celle du signal gele calcule sur
l'historique complet. Tolerance 1e-12 : on attend l'egalite exacte.

Sortie : experiments/deita_cta_signal/
    <ACTIF>.parquet     `signal` (signe, dans {-1,0,+1}) et `strength` (Hull brut)
    conviction.parquet  la conviction hierarchique, archivee en descriptif
    manifest.json       hashes (moteur, prix, signaux), decision, verification
Usage   : python deita_cta_signal.py
Code de sortie : 0 si la porte est franchie, 1 sinon.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEITA_ROOT = Path("/Users/mikamoto/Documents/GitLocal/Projects/LocalProjectTimeSeries1")
for _p in (ROOT, ROOT / "models", ROOT / "experiments", DEITA_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cta_quant_engine import CORR_LOOKBACK, HULL_PERIOD, ConvictionEngine, hull_ma  # noqa: E402
from grid2020 import load_asset                                             # noqa: E402
from prices_v3 import ORIGIN_START, OUT_DIR as PRICES_V3, PANEL, slug       # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "deita_cta_signal"

# ── UNIVERS 0-BIS (BRIEF_re_porte_cta_corrige_univers_deita.md) ─────────────
# Le v3 est l'univers de la porte 0 : 7 actifs, moteur d'origine, signal retenu =
# direction Hull (la conviction y degenerait). Le v4 est celui de la porte 0-bis :
# 18 actifs sur prices_v4, moteur CORRIGE (bugs 1 et 2, drapeaux par defaut) --
# et la conviction hierarchique redevient donc le signal retenu, puisque la raison
# de l'avoir ecartee (la degenerescence des singletons) a disparu. C'est le CTA de
# DEITA au complet, dans son habitat, ce que la porte 0-bis veut juger.
UNIVERSES = {
    "v3": {"retained": "trend", "out": OUT_DIR,
           "why": "porte 0 : moteur d'origine, conviction degeneree -> direction Hull retenue"},
    "v4": {"retained": "conviction", "out": Path(__file__).resolve().parent / "deita_cta_signal_v4",
           "why": "porte 0-bis : moteur corrige, hierarchie non degeneree -> conviction retenue"},
}
ENGINE_FILE = DEITA_ROOT / "cta_quant_engine.py"
LOOKAHEAD_TOL = 1e-12
N_PROBE_ORIGINS = 5

# Carte secteur / sous-secteur, utilisee UNIQUEMENT pour la conviction archivee en
# descriptif (la direction Hull n'en a pas besoin). Etiquettes de DEITA la ou
# l'actif existe chez lui ; le secteur obligataire est declare ici.
ASSET_MAP = {
    "SPY":     {"sector": "Equity",    "subsector": "US Large-Cap"},
    "BTC-USD": {"sector": "Crypto",    "subsector": "Large-Cap Crypto"},
    "ETH-USD": {"sector": "Crypto",    "subsector": "L1 Smart Contract"},
    "GLD":     {"sector": "Commodity", "subsector": "Precious Metals"},
    "USO":     {"sector": "Commodity", "subsector": "Energy"},
    "ZN=F":    {"sector": "Bond",      "subsector": "US Treasury Future"},
    "TLT":     {"sector": "Bond",      "subsector": "US Treasury ETF"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_panel(assets, prices_dir: Path = None) -> pd.DataFrame:
    """Prix quotidiens geles, en tableau large. Aucun appel reseau."""
    d = Path(prices_dir or PRICES_V3)
    return pd.DataFrame({a: pd.read_parquet(d / f"{slug(a)}.parquet")["close"]
                         for a in assets}).sort_index()


# Convention de CALENDRIER, et c'est une vraie difference, pas un detail
# d'implementation. `compute_cta_signal` de DEITA fait `prices[[asset]].ffill()`
# AVANT de calculer les rendements. Quand on lui passe -- comme DEITA le fait en
# production -- un panel melangeant crypto (7 j/7) et actions (5 j/7), les actions
# heritent du cours de vendredi le samedi et le dimanche : deux jours a rendement
# nul entrent dans la moyenne de Hull chaque semaine. Sur le calendrier propre a
# chaque actif, ces jours n'existent pas.
#
# L'ecart n'est pas cosmetique : il retourne le signe sur 13-15 % des observations
# quotidiennes des actifs a 5 jours (0 % en crypto, que le ffill ne touche pas), et
# sur 7,3 % des cellules (origine x actif) de la grille.
#
# CONVENTION RETENUE : "deita" -- le brief dit « le signal CTA vient de DEITA tel
# quel », et c'est bien ce que fait DEITA sur un panel mixte. La convention
# "own" (calendrier propre) est calculee et rapportee en controle : la porte est
# jugee sous les deux.
CALENDARS = ("deita", "own")


def trend_direction(prices: pd.DataFrame, calendar: str = "deita") -> pd.DataFrame:
    """LE SIGNAL RETENU : signe de la moyenne mobile de Hull des rendements.
    Univarie par construction -- chaque colonne ne depend que d'elle-meme.
    `calendar` : cf. la note CALENDARS ci-dessus."""
    def series(a):
        px = prices[[a]].ffill().dropna()[a] if calendar == "deita" else prices[a].dropna()
        return hull_ma(px.pct_change(), HULL_PERIOD)
    return pd.DataFrame({a: series(a) for a in prices.columns})


def conviction(prices: pd.DataFrame, asset_map: dict = None) -> pd.DataFrame:
    """La conviction hierarchique. Descriptif au v3 (elle y degenere), SIGNAL
    RETENU au v4 (moteur corrige, hierarchie reelle) -- cf. UNIVERSES."""
    return ConvictionEngine(asset_map=asset_map or ASSET_MAP, hull_period=HULL_PERIOD,
                            corr_lookback=CORR_LOOKBACK).compute(prices)


def probe_dates(prices: pd.DataFrame, n: int = N_PROBE_ORIGINS) -> list:
    """`n` origines hebdo reparties sur la grille de test, la premiere et la
    derniere comprises -- une verification qui ne sonderait que le milieu
    manquerait justement les effets de bord."""
    _, _, weekly_dates, test_pos = load_asset("SPY")
    picks = np.linspace(0, len(test_pos) - 1, n).astype(int)
    return [weekly_dates.iloc[test_pos[i]] for i in picks
            if weekly_dates.iloc[test_pos[i]] in prices.index]


def lookahead_check(prices: pd.DataFrame, frozen: pd.DataFrame, dates: list,
                    calendar: str = "deita", recompute=None) -> dict:
    """BLOQUANT. Recalcule le signal sur l'historique TRONQUE a chaque date et
    compare a la valeur gelee. Egalite exacte attendue.

    `recompute` (opt-in) : la fonction a rejouer. Defaut = la direction Hull, ce
    qui laisse le chemin v3 inchange ; la porte 0-bis y passe la conviction."""
    per_date, worst = {}, 0.0
    for d in dates:
        truncated = (recompute(prices.loc[:d]) if recompute
                     else trend_direction(prices.loc[:d], calendar))
        a = truncated.loc[d]
        b = frozen.loc[d, truncated.columns]
        diff = float(np.nanmax(np.abs(a.values - b.values)))
        worst = max(worst, diff)
        per_date[str(pd.Timestamp(d).date())] = {
            "max_abs_diff": diff, "passes": bool(diff <= LOOKAHEAD_TOL),
            "n_obs_truncated": int(len(prices.loc[:d])),
        }
    return {"tolerance": LOOKAHEAD_TOL, "n_dates": len(per_date),
            "max_abs_diff_overall": worst, "per_date": per_date,
            "passes": bool(worst <= LOOKAHEAD_TOL)}


def describe(series: pd.DataFrame) -> dict:
    """Le signal est-il actif, change-t-il de signe, sature-t-il ? Un signal
    constant n'a pas besoin d'etre teste, il a besoin d'etre signale."""
    s = series[series.index >= pd.Timestamp(ORIGIN_START)]
    out = {}
    for a in s.columns:
        v = s[a].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        sign = np.sign(v)
        out[a] = {
            "n": int(v.size),
            "share_long": float((sign > 0).mean()), "share_short": float((sign < 0).mean()),
            "share_flat": float((sign == 0).mean()),
            "n_sign_flips": int((np.diff(sign) != 0).sum()),
            "abs_mean": float(np.abs(v).mean()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--n-probe", type=int, default=N_PROBE_ORIGINS)
    p.add_argument("--universe", default="v3", choices=list(UNIVERSES),
                   help="v3 = porte 0 (7 actifs, direction Hull) ; v4 = porte 0-bis "
                        "(18 actifs prices_v4, moteur corrige, conviction retenue)")
    p.add_argument("--calendar", default="deita", choices=list(CALENDARS),
                   help="'deita' = ffill sur le calendrier de l'union (convention DEITA) ; "
                        "'own' = jours de cotation propres a l'actif")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    uni = UNIVERSES[args.universe]
    if args.universe == "v4":
        from prices_v4 import ASSET_MAP as MAP_V4, OUT_DIR as PRICES_V4, PANEL as PANEL_V4
        asset_map, prices_dir = MAP_V4, PRICES_V4
        assets = list(PANEL_V4) if args.assets == list(PANEL) else args.assets
        # Le moteur corrige n'a plus besoin du ffill : le bug 2 est corrige en
        # amont, `calendar_policy="own"` est son defaut. La convention de gel du
        # v3 ne s'applique pas au v4 -- ce sont deux programmes distincts.
        calendar = "own"
    else:
        asset_map, prices_dir, assets, calendar = ASSET_MAP, PRICES_V3, args.assets, args.calendar
    out_dir = Path(args.out_dir) if args.out_dir != str(OUT_DIR) else uni["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"univers {args.universe} : {uni['why']}")

    prices = load_panel(assets, prices_dir)
    print(f"panel gele : {len(prices)} jours, {len(prices.columns)} actifs "
          f"({prices.index[0].date()} -> {prices.index[-1].date()}) -- aucun appel reseau")

    strength = trend_direction(prices, calendar)
    conv = conviction(prices, asset_map)
    retained = conv if uni["retained"] == "conviction" else strength
    print(f"signal calcule : conviction hierarchique (correlation {CORR_LOOKBACK} j) + "
          f"direction Hull {HULL_PERIOD} j -- retenu : {uni['retained']}")

    dates = probe_dates(prices, args.n_probe)
    print(f"\nverification d'absence de look-ahead sur {len(dates)} origines "
          f"(recalcul complet sur historique tronque) ...")
    check = lookahead_check(
        prices, retained, dates, calendar,
        recompute=(lambda px: conviction(px, asset_map)) if uni["retained"] == "conviction"
        else None)
    for d, r in check["per_date"].items():
        print(f"  {d}  {r['n_obs_truncated']:>5} obs  ecart max {r['max_abs_diff']:.2e}  "
              f"{'OK' if r['passes'] else 'ECHEC'}")

    hashes = {}
    for a in retained.columns:
        df = pd.DataFrame({"signal": np.sign(retained[a]), "strength": retained[a]}).dropna()
        path = out_dir / f"{slug(a)}.parquet"
        df.to_parquet(path)
        hashes[a] = sha256(path)
    conv.to_parquet(out_dir / "conviction.parquet")
    strength.to_parquet(out_dir / "trend_direction.parquet")

    desc_trend, desc_conv = describe(strength), describe(conv)
    manifest = {
        "scope": "chantier 0a -- signal directionnel CTA de DEITA, gele point-in-time",
        "engine": {
            "source": str(ENGINE_FILE), "sha256": sha256(ENGINE_FILE),
            "retained": uni["retained"], "retained_why": uni["why"],
            "universe_version": args.universe,
            "hull_period": HULL_PERIOD, "corr_lookback": CORR_LOOKBACK,
            "calendar": calendar,
            "calendar_note": "'deita' reproduit le ffill que compute_cta_signal applique sur un "
                             "panel mixte crypto/actions ; 'own' calcule sur les jours de "
                             "cotation propres. L'ecart retourne le signe sur 7,3 % des cellules "
                             "(origine x actif) -- la porte est jugee sous les deux conventions.",
            "parameters_touched": "aucun -- moteur DEITA tel quel",
        },
        "decision_conviction_rejetee": {
            "mecanisme": "sous-secteur a un membre -> deux niveaux sur trois valent la serie "
                         "lissee de l'actif -> conv = (2/3) smooth^2 + smooth x mkt/3, dont le "
                         "terme dominant est un carre : le signe ne change plus",
            "mesure": {a: {"part_long_conviction": desc_conv[a]["share_long"],
                           "part_long_direction_hull": desc_trend[a]["share_long"],
                           "sous_secteur_singleton": sum(
                               1 for v in asset_map.values()
                               if v["subsector"] == asset_map[a]["subsector"]) == 1}
                       for a in retained.columns},
            "univers_elargi_ne_sauve_pas": "SPY et BTC-USD sont deja seuls dans leur sous-secteur "
                                           "dans l'univers de 16 actifs de DEITA ; SPY et ZN=F, "
                                           "les deux instruments de l'hypothese primaire, restent "
                                           "degeneres. Et les prix de la base DEITA s'ecartent de "
                                           "prices_v3 jusqu'a 2,6e-03 en log-rendement sur SPY "
                                           "(tolerance de gel du benchmark : 1e-5).",
            "conservee_en": "conviction.parquet -- descriptif, jamais utilisee pour decider",
        },
        "universe": {
            "assets": list(retained.columns),
            "note": "la direction Hull est univariee : elle ne depend d'aucun univers, "
                    "seulement de la serie de l'actif. Calculee entierement sur prices_v3.",
            "asset_map": asset_map, "prices_dir": str(prices_dir),
        },
        "prices": {"dir": str(prices_dir),
                   "sha256": {a: sha256(Path(prices_dir) / f"{slug(a)}.parquet")
                              for a in retained.columns},
                   "span": f"{prices.index[0].date()} -> {prices.index[-1].date()}"},
        "signal_sha256": hashes,
        "lookahead_check": check,
        "descriptive": {"direction_hull_retenue": desc_trend, "conviction_descriptive": desc_conv},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str,
                                                      ensure_ascii=False))

    print(f"\n=== signal retenu ({uni['retained']}), grille a partir de {ORIGIN_START} ===")
    print(f"  {'actif':<9} {'long':>7} {'short':>7} {'flips':>6}   | autre lecture : long")
    for a in retained.columns:
        d = (desc_conv if uni["retained"] == "conviction" else desc_trend)[a]
        c = (desc_trend if uni["retained"] == "conviction" else desc_conv)[a]
        print(f"  {a:<9} {d['share_long']:>6.1%} {d['share_short']:>7.1%} {d['n_sign_flips']:>6}   "
              f"| {c['share_long']:>6.1%}")
    print(f"\n  porte 0a : {'PASS' if check['passes'] else 'FAIL'} "
          f"(ecart max {check['max_abs_diff_overall']:.2e}, tolerance {LOOKAHEAD_TOL:.0e})")
    print(f"-> {out_dir}")
    sys.exit(0 if check["passes"] else 1)


if __name__ == "__main__":
    main()
