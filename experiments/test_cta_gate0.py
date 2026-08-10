"""
Tests unitaires du chantier 0 (BRIEF « couplage CTA (DEITA) x sizing NsDiff ») :
le gel du signal et la porte d'entree.

Ce que ces tests protegent en priorite, c'est la CAUSALITE. Un signal de trading
qui regarde devant produit des resultats magnifiques et faux ; la verification a
5 origines du script est une preuve empirique, ces tests-ci en sont la preuve
structurelle -- ils reconstruisent le cas ou une fuite existerait et verifient
que le mecanisme la rendrait visible.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cta_gate0 as gate                                              # noqa: E402
import deita_cta_signal as sig                                        # noqa: E402


def _prices(n: int = 400, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    out = {}
    for i, a in enumerate(("SPY", "TLT")):
        r = rng.normal(drift, 0.01, n)
        out[a] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=idx)


# ── le signal ne regarde pas devant ─────────────────────────────────────────

def test_direction_hull_est_causale_tronquer_le_futur_ne_change_rien():
    """Le coeur du chantier 0a : la valeur a t ne doit dependre que de x[:t]."""
    px = _prices()
    full = sig.trend_direction(px)
    cut = px.index[300]
    truncated = sig.trend_direction(px.loc[:cut])
    assert np.allclose(truncated.loc[cut].values, full.loc[cut].values, atol=1e-12)


def test_direction_hull_bouge_si_le_passe_change():
    """Contre-epreuve : le test precedent ne doit pas passer trivialement. Si on
    modifie une donnee ANTERIEURE, le signal doit bouger -- sinon il ne regarde
    rien du tout et la verification de causalite ne prouverait rien."""
    px = _prices()
    cut = px.index[300]
    altered = px.copy()
    altered.iloc[290:295] *= 1.10
    assert not np.allclose(sig.trend_direction(altered).loc[cut].values,
                           sig.trend_direction(px).loc[cut].values, atol=1e-12)


def test_lookahead_check_detecte_une_fuite_injectee(monkeypatch):
    """Si le calcul du signal regardait devant, la porte doit echouer. On injecte
    une fuite (le signal devient la moyenne de TOUTE la serie) et on verifie que
    le mecanisme de verification la voit."""
    px = _prices()
    frozen = sig.trend_direction(px)
    dates = [px.index[250], px.index[300]]
    assert sig.lookahead_check(px, frozen, dates)["passes"]

    def leaky(prices, calendar="deita"):
        return pd.DataFrame({c: pd.Series(prices[c].mean(), index=prices.index)
                             for c in prices.columns})

    monkeypatch.setattr(sig, "trend_direction", leaky)
    assert not sig.lookahead_check(px, leaky(px), dates)["passes"]


def test_describe_compte_les_changements_de_signe():
    s = pd.DataFrame({"X": [1.0, 1.0, -1.0, -1.0, 1.0]},
                     index=pd.bdate_range("2021-01-01", periods=5))
    d = sig.describe(s)["X"]
    assert d["n_sign_flips"] == 2
    assert d["share_long"] == pytest.approx(0.6)
    assert d["share_short"] == pytest.approx(0.4)


def test_describe_repere_un_signal_toujours_long():
    """La degenerescence de la conviction hierarchique doit se lire dans le
    descriptif, pas se decouvrir dans les resultats."""
    s = pd.DataFrame({"X": [0.2, 0.5, 0.1, 0.9]},
                     index=pd.bdate_range("2021-01-01", periods=4))
    d = sig.describe(s)["X"]
    assert d["share_long"] == 1.0 and d["n_sign_flips"] == 0


# ── la porte lit le signal a la bonne date ──────────────────────────────────

def test_attach_signal_lit_la_date_exacte_sans_remplissage():
    """Aucun ffill : une origine sans valeur de signal est ECARTEE, jamais
    servie par la valeur d'une date voisine -- laquelle pourrait etre posterieure."""
    market = pd.DataFrame({
        "asset": ["SPY", "SPY"], "frequence": ["weekly"] * 2, "horizon_unit": ["W+1"] * 2,
        "cutoff_date": ["2021-01-04", "2021-01-11"], "target_date": ["2021-01-11", "2021-01-18"],
        "last_close": [100.0, 101.0], "y_true": [101.0, 102.0]})
    signal = pd.DataFrame({"SPY": [1.0]}, index=pd.to_datetime(["2021-01-04"]))
    out = gate.attach_signal(market, signal)
    assert len(out) == 1 and out["cutoff_date"].iloc[0] == "2021-01-04"


def test_attach_signal_ne_decale_pas_le_signal():
    market = pd.DataFrame({
        "asset": ["SPY", "SPY"], "frequence": ["weekly"] * 2, "horizon_unit": ["W+1"] * 2,
        "cutoff_date": ["2021-01-04", "2021-01-11"], "target_date": ["2021-01-11", "2021-01-18"],
        "last_close": [100.0, 101.0], "y_true": [101.0, 102.0]})
    signal = pd.DataFrame({"SPY": [1.0, -1.0]},
                          index=pd.to_datetime(["2021-01-04", "2021-01-11"]))
    out = gate.attach_signal(market, signal)
    assert out["signal"].tolist() == [1.0, -1.0]


# ── le moteur d'evaluation ──────────────────────────────────────────────────

def _cell(signal, last_close, y_true):
    return {"cutoff_date": np.array([f"2021-01-{i + 1:02d}" for i in range(len(signal))]),
            "last_close": np.asarray(last_close, dtype=float),
            "y_true": np.asarray(y_true, dtype=float),
            "signal": np.asarray(signal, dtype=float)}


def test_un_signal_toujours_long_est_exactement_acheter_et_garder():
    """Le controle qui a servi : la conviction hierarchique degeneree n'est pas un
    signal, c'est un B&H deguise -- et l'edge vs B&H doit alors valoir zero
    EXACTEMENT, pas approximativement."""
    d = _cell([1, 1, 1, 1, 1, 1], [100] * 6, [101, 99, 102, 98, 103, 100])
    r = gate.evaluate(d, "SPY-ETF", 1, "central")
    assert r["edge_vs_bh_bps"] == pytest.approx(0.0, abs=1e-12)
    assert r["pnl_net_cta_bps"] == pytest.approx(r["pnl_net_bh_bps"])


def test_un_signal_short_inverse_le_pnl_brut():
    long_ = gate.evaluate(_cell([1] * 6, [100] * 6, [101] * 6), "SPY-ETF", 1, "central")
    short = gate.evaluate(_cell([-1] * 6, [100] * 6, [101] * 6), "SPY-ETF", 1, "central")
    # meme cout des deux cotes (|w| identique), donc les PnL sont symetriques
    # autour de -2 x cout
    cost = 2 * (gate.rf.one_way_total_bps("SPY-ETF", 1, "central") * 1e-4) * gate.BPS
    assert long_["pnl_net_cta_bps"] + short["pnl_net_cta_bps"] == pytest.approx(-2 * cost)


def test_les_frais_incluent_le_roulement_pour_un_future():
    """SPY-ES roule, SPY-ETF non : a signal identique, le future doit payer plus
    que son seul aller-retour outright."""
    d = _cell([1] * 6, [100] * 6, [101] * 6)
    assert gate.evaluate(d, "SPY-ES", 3, "central")["round_trip_bps"] > \
           gate.rf.round_trip_bps("SPY-ES", "central")
    assert gate.evaluate(d, "SPY-ETF", 3, "central")["round_trip_bps"] == \
           pytest.approx(gate.rf.round_trip_bps("SPY-ETF", "central"))


def test_hit_rate_et_sharpe_sont_calcules_sur_le_pnl_net():
    d = _cell([1, 1, 1, 1, 1, 1], [100] * 6, [90, 90, 90, 90, 90, 90])
    r = gate.evaluate(d, "SPY-ETF", 1, "central")
    assert r["hit_rate"] == 0.0
    assert r["pnl_net_cta_bps"] < 0


def test_by_period_decoupe_sur_la_date_d_origine():
    pnl = np.array([0.01, 0.02, -0.01, 0.03])
    cutoffs = np.array(["2020-03-06", "2020-06-05", "2022-03-04", "2022-06-03"])
    out = gate.by_period(pnl, cutoffs, 1)
    assert out == {} or all(v["n"] >= 3 for v in out.values())
    out2 = gate.by_period(np.tile(pnl, 2),
                          np.array(["2020-01-03", "2020-02-07", "2020-03-06", "2020-04-03",
                                    "2022-01-07", "2022-02-04", "2022-03-04", "2022-04-01"]), 1)
    assert out2["2020 (COVID)"]["n"] == 4 and out2["2022 (bear taux)"]["n"] == 4


# ── convention de calendrier : la difference doit etre reelle et maitrisee ───

def _mixed_calendar_panel(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Un actif a 5 jours (actions) et un a 7 jours (crypto), comme le panel
    reel -- c'est cette configuration qui fait diverger les deux conventions."""
    rng = np.random.default_rng(seed)
    idx7 = pd.date_range("2019-01-01", periods=n, freq="D")
    crypto = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx7)
    idx5 = idx7[idx7.dayofweek < 5]
    equity = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx5)))), index=idx5)
    return pd.DataFrame({"CRYPTO": crypto, "EQUITY": equity}).sort_index()


def test_les_deux_calendriers_different_sur_un_actif_a_cinq_jours():
    px = _mixed_calendar_panel()
    a = sig.trend_direction(px, "deita")["EQUITY"].dropna()
    b = sig.trend_direction(px, "own")["EQUITY"].dropna()
    common = a.index.intersection(b.index)
    assert not np.allclose(a.loc[common].values, b.loc[common].values)


def test_les_deux_calendriers_coincident_sur_un_actif_a_sept_jours():
    """Le ffill ne peut rien changer la ou il n'y a pas de jour manquant."""
    px = _mixed_calendar_panel()
    a = sig.trend_direction(px, "deita")["CRYPTO"].dropna()
    b = sig.trend_direction(px, "own")["CRYPTO"].dropna()
    common = a.index.intersection(b.index)
    assert np.allclose(a.loc[common].values, b.loc[common].values, atol=1e-12)


def test_la_convention_deita_reproduit_compute_cta_signal():
    """Le signal gele doit valoir EXACTEMENT le `hull_slope` que DEITA publie via
    son propre point d'entree -- sinon ce n'est plus « le signal de DEITA »."""
    from cta_quant_engine import compute_cta_signal
    px = _mixed_calendar_panel()
    frozen = np.sign(sig.trend_direction(px, "deita"))
    for d in (px.index[300], px.index[-1]):
        for a in ("CRYPTO", "EQUITY"):
            expected = compute_cta_signal(a, [a], px.loc[:d], pure_trend_mode=True).hull_slope
            got = float(frozen[a].reindex([d]).iloc[0])
            if not np.isnan(got):
                assert got == expected, (a, d)


# ── branche 2 reparee (PATCH_gate0_branche2_et_holm_m2.md, P2) ──────────────

CLASSES = {"A1": "Equity", "A2": "Bond", "A3": "Crypto", "A4": "Commodity"}


def test_branche2_immune_au_signal_constant():
    """Contre-epreuve du defaut documente : un signal constant a un exces
    identiquement nul, donc la branche doit echouer MECANIQUEMENT -- quelle que
    soit la pente du marche, qui n'entre plus dans le calcul."""
    zero = {k: np.zeros(60) for k in CLASSES}
    r = gate.branch2_verdict(zero, CLASSES)
    assert r["passes"] is False
    assert r["n_classes_positive"] == 0
    assert r["excess_mean_bps"] == pytest.approx(0.0)


def test_branche2_immune_meme_sur_un_marche_tres_haussier():
    """Le cas exact qui faisait passer la formulation d'origine : marche
    fortement haussier, signal toujours long. L'exces reste nul."""
    rng = np.random.default_rng(0)
    bull = rng.normal(0.02, 0.01, 60)          # marche en forte hausse
    excess = {k: bull - bull for k in CLASSES}  # signal constant = B&H
    assert gate.branch2_verdict(excess, CLASSES)["passes"] is False


def test_branche2_detecte_un_vrai_excess():
    """La branche repare, elle ne condamne pas par construction : un exces
    reellement positif sur 3 classes sur 4 doit passer."""
    rng = np.random.default_rng(1)
    good = {k: rng.normal(0.0015, 0.002, 120) for k in ("A1", "A2", "A3")}
    good["A4"] = rng.normal(-0.0005, 0.002, 120)
    r = gate.branch2_verdict(good, CLASSES)
    assert r["passes"] is True
    assert r["n_classes_positive"] == 3
    assert r["sharpe_excess_annualised"] > 0


def test_branche2_exige_les_deux_conditions():
    """Sharpe positif mais 2 classes seulement : echec. C'est la condition qui a
    fait echouer la porte reelle."""
    rng = np.random.default_rng(2)
    mixed = {"A1": rng.normal(0.004, 0.002, 120), "A2": rng.normal(0.004, 0.002, 120),
             "A3": rng.normal(-0.001, 0.002, 120), "A4": rng.normal(-0.001, 0.002, 120)}
    r = gate.branch2_verdict(mixed, CLASSES)
    assert r["n_classes_positive"] == 2 and r["passes"] is False


# ── non-regression : le verdict reste ECHEC sous les trois variantes ────────

@pytest.mark.parametrize("artefact", ["cta_gate0.json", "cta_gate0_own_calendar.json",
                                      "cta_gate0_conviction.json"])
def test_la_porte_reste_en_echec_sous_les_trois_variantes(artefact):
    """Le patch repare l'instrument sans toucher au verdict. Si un de ces trois
    artefacts passait, c'est que la reparation aurait change une conclusion --
    ce qu'elle n'a pas le droit de faire sans re-declaration."""
    import json
    path = Path(__file__).resolve().parent / artefact
    if not path.exists():
        pytest.skip(f"{artefact} absent -- rejouer cta_gate0.py")
    g = json.loads(path.read_text())["gate"]
    assert g["passes"] is False
    assert g["branch_2"]["formulation"].startswith("Sharpe poole de l'EXCES")


def test_la_conviction_degeneree_ne_franchit_plus_la_branche2():
    """Le cas historique, sur l'artefact reel : la formulation d'origine passait,
    la reparee echoue."""
    import json
    path = Path(__file__).resolve().parent / "cta_gate0_conviction.json"
    if not path.exists():
        pytest.skip("artefact absent")
    g = json.loads(path.read_text())["gate"]
    assert g["branch_2_original_formulation"]["passes"] is True
    assert g["branch_2"]["passes"] is False


# ── P3 : verrou sur la convention de calendrier du signal gele ──────────────

def test_le_signal_gele_reste_sur_la_convention_qui_l_a_gele():
    """Verrou anti-derive sur l'ARTEFACT, pas sur une serie synthetique.

    Il pointe sur la convention de GEL (`union_ffill`), pas sur le comportement
    courant de DEITA -- et la distinction est devenue concrete : le correctif du
    bug 2 (TICKET_DEITA_cta_quant_engine.md) a fait passer DEITA au calendrier
    propre. Le signal archive, lui, ne bouge pas : c'est celui qui a produit le
    verdict de porte archive, et un verdict clos ne se reecrit pas parce qu'un
    moteur amont a evolue. Re-juger la porte avec le CTA repare est une
    REOUVERTURE, qui se decide et demande son propre brief.

    Les versions de `cta_quant_engine` anterieures au correctif n'ont pas le
    parametre : on retombe alors sur l'appel simple, qui avait ce comportement."""
    from cta_quant_engine import compute_cta_signal
    d = Path(__file__).resolve().parent / "deita_cta_signal"
    if not (d / "SPY.parquet").exists():
        pytest.skip("signal gele absent -- rejouer deita_cta_signal.py")
    px = sig.load_panel(list(sig.ASSET_MAP))
    for a in ("SPY", "TLT", "BTC-USD"):
        frozen = pd.read_parquet(d / f"{a.replace('=', '_')}.parquet")["signal"]
        for date in (pd.Timestamp("2022-06-03"), pd.Timestamp("2024-11-15")):
            try:
                expected = compute_cta_signal(a, [a], px.loc[:date], pure_trend_mode=True,
                                              calendar_policy="union_ffill").hull_slope
            except TypeError:                      # moteur anterieur au correctif
                expected = compute_cta_signal(a, [a], px.loc[:date],
                                              pure_trend_mode=True).hull_slope
            assert float(frozen.loc[date]) == expected, (a, date)


def test_le_moteur_amont_a_change_est_signale_et_ne_reecrit_rien():
    """Le manifeste porte le hash du moteur qui a gele le signal. S'il a change,
    l'artefact reste valide -- mais un re-run produirait autre chose, et cela doit
    etre VISIBLE plutot que decouvert dans un resultat."""
    import json
    d = Path(__file__).resolve().parent / "deita_cta_signal"
    if not (d / "manifest.json").exists():
        pytest.skip("manifeste absent")
    man = json.loads((d / "manifest.json").read_text())
    engine = Path(man["engine"]["source"])
    if not engine.exists():
        pytest.skip("moteur DEITA introuvable depuis ce depot")
    assert man["engine"]["sha256"], "le manifeste doit porter le hash du moteur"
    # Le hash courant PEUT differer (correctifs DEITA) : ce n'est pas une erreur,
    # c'est une information. Ce qui serait une erreur, c'est de ne pas la porter.
    assert "calendar" in man["engine"], "la convention de calendrier doit etre consignee"
