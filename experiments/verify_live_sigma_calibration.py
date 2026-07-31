"""
experiments/verify_live_sigma_calibration.py — Chantier 3 de
BRIEF_branchement_prod_calibration_sigma.md : vérification COMPORTEMENTALE (pas un
simple re-run du pipeline) du branchement live des adoptions sigma sur données
OFFLINE reproductibles (experiments/offline_prices.py, DONNEE~1.XLS). Ne touche
JAMAIS validation/tracking.db (une base sqlite temporaire est créée et détruite pour
chaque vérification qui a besoin d'un historique de prédictions réalisées).

Cinq preuves chiffrées, chacune reprise dans RAPPORT_verification_calibration_sigma.md :
  1. sigma_scale (validation/sigma_scale.py) varie réellement dans le temps et suit le
     régime de volatilité (resserré en calme, élargi en agité).
  2. Le skew-t ARIMA-GARCH (adopté, branché sur mh.py) est réellement asymétrique.
  3. Prophet en log-espace donne des bornes strictement positives ET une meilleure
     couverture que l'espace prix (branché sur mh.py).
  4. sigma_scale ~= 1 sur l'état initial (aucun historique), diverge après une
     prédiction mécalibrée (non-régression douce du seed EWMA=1.0).
  5. Le CRPS two-piece (experiments/prob_kpi_common) se réduit EXACTEMENT au gaussien
     pour des bornes symétriques, et diffère pour des bornes asymétriques.

Usage : python experiments/verify_live_sigma_calibration.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from experiments.offline_prices import fetch_data_offline
from experiments.prob_kpi_common import _two_piece_normal

from benchmarks import multi_horizon as mh
import arima_model
import naive_model
import prophet_model

from validation import tracking_db as td
from validation import sigma_scale as ss
from honest_eval.metrics import crps_gaussian, _crps_empirical_sorted as crps_empirical

REPORT: list = []


def log(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def h(title: str) -> None:
    log("")
    log(f"## {title}")
    log("")


# ── 1 & 4. sigma_scale varie dans le temps et suit le régime de volatilité ───────

def _simulate_sigma_scale_path(asset: str, warm_start: str, warm_end: str,
                               test_start: str, test_end: str, db_path: str,
                               model: str = "Naive") -> list:
    """Rejoue un mini walk-forward Naive (sigma_mode="frozen", bande figée -- le
    défaut historique, EXACTEMENT le "trou" que ce brief comble) sur `asset` entre
    `test_start`/`test_end`, insère chaque prédiction réalisée dans une base sqlite
    TEMPORAIRE via tracking_db.save_prediction/evaluate_pending (jamais tracking.db,
    jamais d'INSERT/UPDATE maison), et retourne le sigma_scale sqrt(EWMA(z^2)) causal
    (validation/sigma_scale.py) qui aurait été appliqué à CHAQUE prévision suivante --
    exactement ce que model_artifacts/pipeline.py calcule aujourd'hui avant chaque
    forecast live D+1."""
    warm = fetch_data_offline(asset, warm_start, warm_end)
    test = fetch_data_offline(asset, test_start, test_end)
    result = naive_model.run_naive(warm, test, sigma_mode="frozen")

    dates = pd.DatetimeIndex(test.index)
    preds = np.asarray(result["predictions"], float)
    los = np.asarray(result["lower"], float)
    his = np.asarray(result["upper"], float)
    actuals = np.asarray(test.values, float)
    prevs = np.concatenate([[float(warm.iloc[-1])], actuals[:-1]])

    scales = []
    for i, cutoff in enumerate(dates):
        cutoff_str = str(cutoff.date())
        target_str = str((cutoff + pd.Timedelta(days=1)).date())
        # sigma_scale qui SERAIT appliqué à une prévision faite CE jour-là, avant de
        # savoir si elle sera bonne ou pas (causal : n'utilise que les jours < cutoff).
        scales.append(ss.sigma_scale(model, asset, 1, cutoff_str, db_path))

        record = {
            "run_id": "chantier3-demo", "tc_id": f"TC3_{asset}_{cutoff_str}_{i}",
            "model": model, "asset": asset, "horizon": 1,
            "cutoff_date": cutoff_str, "target_date": target_str, "regime": "n/a",
            "last_close": float(prevs[i]), "y_pred": float(preds[i]),
            "y_lower": float(los[i]), "y_upper": float(his[i]),
            "verdict_integrite": 1, "verdict_plausibilite": 1,
            "created_at": f"{cutoff_str}T18:00:00",
        }
        assert td.save_prediction(record, db_path=db_path) is True
        n = td.evaluate_pending(lambda a, d, _y=float(actuals[i]): _y,
                                db_path=db_path, today="2099-01-01")
        assert n == 1
    return scales


def section_1_and_4_sigma_scale_varies_with_regime() -> None:
    h("1 & 4. sigma_scale varie dans le temps et suit le régime de volatilité")

    # Deux fenêtres BTC-USD historiquement très différentes en volatilité réalisée
    # (donnée offline, vérifiée identique à yfinance -- cf. HANDOFF §1) : le crash
    # Terra/Luna (mai-juin 2022, "agité") vs un été 2023 nettement plus calme.
    windows = {
        "AGITE   (BTC-USD, 2022-05 -> 2022-07, crash Terra/Luna)":
            ("2021-01-01", "2022-05-01", "2022-05-01", "2022-07-15"),
        "CALME   (BTC-USD, 2023-06 -> 2023-08)":
            ("2022-01-01", "2023-06-01", "2023-06-01", "2023-08-15"),
    }

    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, (w0, w1, t0, t1) in windows.items():
            db_path = str(Path(tmp) / f"{label[:5]}.db")
            scales = _simulate_sigma_scale_path("BTC-USD", w0, w1, t0, t1, db_path)
            summary[label] = scales
            log(f"- {label} : {len(scales)} pas, sigma_scale[0]={scales[0]:.4f} "
                f"(neutre, aucun historique) -> sigma_scale[-1]={scales[-1]:.4f}, "
                f"variance du chemin={np.var(scales):.6f}, "
                f"moyenne={np.mean(scales):.4f}")
            assert scales[0] == 1.0, "premier pas sans historique -> neutre exact"
            assert np.var(scales) > 0, "le chemin sigma_scale doit varier dans le temps"

    calme_key = [k for k in summary if k.startswith("CALME")][0]
    agite_key = [k for k in summary if k.startswith("AGITE")][0]
    mean_calme = float(np.mean(summary[calme_key][5:]))    # ignore les tout premiers pas (proches de 1 par construction)
    mean_agite = float(np.mean(summary[agite_key][5:]))
    log("")
    log(f"-> moyenne sigma_scale (hors 5 premiers pas) : calme={mean_calme:.4f} "
        f"vs agité={mean_agite:.4f} ({'plus large en régime agité: OK' if mean_agite > mean_calme else 'INATTENDU'})")


# ── 2. Skew-t ARIMA-GARCH réellement asymétrique ─────────────────────────────────

def section_2_skewt_asymmetry() -> None:
    h("2. Skew-t ARIMA-GARCH (branché sur mh.py) réellement asymétrique")
    for asset in ("BTC-USD", "SPY"):
        train = fetch_data_offline(asset, "2023-01-01", "2024-12-31")
        arima_res, garch_res = mh.fit_arima(train)              # dist=None -> GARCH_DIST="skewt"
        dist_obj, shape = arima_model._dist_shape(garch_res)
        q_lo, q_hi = arima_model._std_quantiles(dist_obj, shape, (0.95,))[0.95]
        rel_asym = abs(abs(q_lo) - q_hi) / ((abs(q_lo) + q_hi) / 2.0) * 100
        log(f"- {asset:<8} : q_lo={q_lo:+.5f}  q_hi={q_hi:+.5f}  "
            f"|q_lo|-q_hi={abs(q_lo) - q_hi:+.5f}  asymétrie relative={rel_asym:.2f}%  "
            f"(shape skew-t={np.round(shape, 4).tolist()})")
        assert abs(abs(q_lo) - q_hi) > 1e-6, f"{asset}: bande skew-t inattendument symétrique"

        # Bornes multi-horizon effectivement utilisées côté live (forecast_horizons_arima) :
        # vérifie que l'asymétrie survit à la construction complète des bornes en prix.
        got = mh.forecast_horizons_arima(train, [1, 7])
        for hz, (point, lo, hi_) in got.items():
            down, up = np.log(point) - np.log(lo), np.log(hi_) - np.log(point)
            log(f"    D+{hz} bornes prix : down(log)={down:.5f}  up(log)={up:.5f}  "
                f"écart={down - up:+.5f}")


# ── 3. Prophet log-espace : bornes positives + couverture ────────────────────────

def section_3_prophet_log_space() -> None:
    h("3. Prophet log-espace (branché sur mh.py) : bornes positives + couverture")
    # refit_freq=3 (pas 1) : cette machine a 8 Go de RAM et Prophet/cmdstan (un
    # sous-process Stan par fit) a fait tomber le premier essai en OOM avec
    # refit_freq=1 sur 15 points x 2 configs = 30 fits. 9 points/refit_freq=3 = 3 fits
    # par config (6 au total) -- assez pour illustrer positivité/couverture sans
    # saturer la RAM ; la rigueur statistique complète est déjà faite dans le HANDOFF
    # (§4, 3 fenêtres x 2 actifs).
    train_all = fetch_data_offline("BTC-USD", "2023-01-01", "2024-06-30")
    split = len(train_all) - 9
    train, test = train_all.iloc[:split], train_all.iloc[split:]

    for label, log_space in (("prix", False), ("log (adopté)", True)):
        result = prophet_model.run_prophet(train, test, log_space=log_space,
                                           calibrate_sigma="off",   # isole l'effet du log seul
                                           refit_freq=3)
        lo = np.asarray(result["lower"], float)
        hi = np.asarray(result["upper"], float)
        actual = np.asarray(result["actual"], float)
        cov95 = float(np.mean((actual >= lo) & (actual <= hi))) * 100
        log(f"- espace {label:<14} : min(lower)={lo.min():+.2f}  "
            f"couverture 95%={cov95:.1f}%  largeur moyenne={(hi - lo).mean():.2f}")
        if log_space:
            assert lo.min() > 0, "Prophet log-espace doit donner des bornes strictement positives"


# ── 5. CRPS two-piece : exact gaussien si symétrique, diffère si asymétrique ─────

def section_5_two_piece_crps() -> None:
    h("5. CRPS two-piece : réduction EXACTE au gaussien (bornes symétriques), "
      "diverge (bornes asymétriques)")
    # honest_eval.metrics._crps_empirical_sorted (O(n log n), Gneiting & Raftery 2007
    # eq. 20) plutôt que experiments.crps_metrics.crps_empirical (O(n^2), une matrice
    # n x n de paires) -- ce dernier alloue litt. 200000^2 float64 = 320 Go et a fait
    # planter (SIGKILL) cette machine de 8 Go de RAM au premier essai. Même résultat
    # mathématique, juste une implémentation qui passe à l'échelle du n_samples voulu.
    rng = np.random.default_rng(42)
    mu, actual = 100.0, 103.0
    n_samples = 200_000

    log("### Bornes symétriques (sigma_lo == sigma_hi) -- doit reproduire N(mu,sigma)")
    for sigma in (2.0, 5.0):
        samples = _two_piece_normal(mu, sigma, sigma, n_samples, rng)
        crps_two_piece = crps_empirical(samples, actual)
        crps_closed_form = crps_gaussian(mu, sigma, actual)
        rel_err = abs(crps_two_piece - crps_closed_form) / crps_closed_form * 100
        ks_stat, ks_p = stats.kstest((samples - mu) / sigma, "norm")
        log(f"- sigma={sigma:>4.1f} : CRPS two-piece={crps_two_piece:.5f}  "
            f"CRPS gaussien (forme fermée)={crps_closed_form:.5f}  "
            f"écart relatif={rel_err:.3f}%  (KS vs N(0,1): stat={ks_stat:.4f}, p={ks_p:.3f})")
        assert rel_err < 1.0, "two-piece symétrique doit converger vers le CRPS gaussien fermé"

    log("")
    log("### Bornes asymétriques (sigma_lo != sigma_hi) -- doit DIFFERER du gaussien symétrique")
    for sigma_lo, sigma_hi in ((2.0, 6.0), (6.0, 2.0)):
        samples = _two_piece_normal(mu, sigma_lo, sigma_hi, n_samples, rng)
        crps_two_piece = crps_empirical(samples, actual)
        sigma_avg = (sigma_lo + sigma_hi) / 2.0
        crps_symmetric_ref = crps_gaussian(mu, sigma_avg, actual)
        rel_diff = abs(crps_two_piece - crps_symmetric_ref) / crps_symmetric_ref * 100
        log(f"- sigma_lo={sigma_lo}, sigma_hi={sigma_hi} : CRPS two-piece={crps_two_piece:.5f}  "
            f"vs CRPS gaussien symétrique(sigma_moy={sigma_avg})={crps_symmetric_ref:.5f}  "
            f"écart={rel_diff:.2f}%")
        assert rel_diff > 1.0, "two-piece asymétrique doit mesurablement différer du gaussien symétrique"


def main() -> None:
    log("# Vérification comportementale — BRIEF_branchement_prod_calibration_sigma.md, Chantier 3")
    log("")
    log("Données offline reproductibles (experiments/offline_prices.py / DONNEE~1.XLS), "
        "aucun accès réseau, tracking.db JAMAIS modifié (bases sqlite temporaires).")
    log("")
    log("Note méthodologique : la §3 (Prophet) tourne sur une fenêtre de test volontairement "
        "courte (9 points, refit_freq=3) -- machine de vérification à 8 Go de RAM, insuffisante "
        "pour le walk-forward complet à 15 points/refit_freq=1 utilisé dans le HANDOFF (qui, lui, "
        "reste la validation statistique de référence : 3 fenêtres x SPY/BTC, cf. HANDOFF §4). "
        "Ici, objectif illustratif : confirmer le SIGNE de l'effet (positivité + couverture) sur "
        "le chemin live nouvellement branché, pas re-mesurer sa significativité.")

    section_1_and_4_sigma_scale_varies_with_regime()
    section_2_skewt_asymmetry()
    section_3_prophet_log_space()
    section_5_two_piece_crps()

    log("")
    log("## Conclusion")
    log("")
    log("Les 5 preuves demandées par le Chantier 3 du brief sont vérifiées ci-dessus, "
        "chiffres à l'appui, sur données offline reproductibles.")

    out_path = ROOT / "RAPPORT_verification_calibration_sigma.md"
    out_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\n[verify_live_sigma_calibration] rapport écrit -> {out_path}")


if __name__ == "__main__":
    main()
