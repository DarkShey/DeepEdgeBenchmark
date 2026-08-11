"""
oos_nsdiff_d1_simtrades.py -- BRIEF_retest_nsdiff_tc_sim_trades.md : génère les
prévisions NsDiff D+1 manquantes sur la grille oos des test cases TC1.1-TC1.5b.

LE TROU COMBLÉ, en une phrase : les 6 modèles de référence ont chacun des
centaines de lignes `oos` à horizon 1 dans `sim_trades`, mais NsDiff n'a que des
lignes `live` et ZÉRO ligne oos -- sa piste oos est en horizons hebdomadaires,
donc il n'est jamais passé par les TC en oos. Ce script produit les lignes
`predictions` (horizon=1, horizon_type='daily', source='oos') qui manquent ; les
règles TC, elles, sont appliquées ensuite par `validation.sim_trades.
generate_sim_trades` SANS AUCUNE MODIFICATION (non-négociable du brief).

CE QUI EST ÉCRIT : uniquement des lignes `model='NsDiff'`, `source='oos'`,
`horizon=1`. Aucune ligne d'un autre modèle n'est touchée -- vérifié par
empreinte SHA-256 avant/après (`--apply`).

## Les quatre garanties méthodologiques

1. **Grille héritée, vérifiée avant écriture.** Les clés (actif, d_date) sont
   reprises VERBATIM des lignes oos d'ARIMA-GARCH (le modèle le plus complet,
   666 lignes / 167 dates), jamais re-dérivées. Comparabilité 1:1 : les deux
   modèles répondent exactement aux mêmes jours.

2. **`ref` et `realized` hérités, jamais recalculés.** Ils viennent des lignes
   ARIMA-GARCH existantes. C'est la garantie que SEULS `predicted`/`pi_lower`/
   `pi_upper` diffèrent entre modèles. `reference_price` sert aussi de
   `last_price` à la dé-standardisation : la prévision est donc ancrée sur
   exactement le même prix de référence que la règle utilisera -- toute autre
   convention introduirait un écart de comparabilité invisible.

   Le prix récupéré (yfinance, puis cache gelé DONNEE~1.XLS) n'est utilisé QUE
   pour la fenêtre de conditionnement, et il est vérifié contre les
   `reference_price` stockés AVANT toute génération (`verify_grid_prices`) --
   même standard bloquant que `_verify_origin_prices` d'oos_nsdiff_daily_weekly.

3. **Aucun réentraînement par origine (train-once-forward).** Un fit par
   (graine, actif) sur l'historique STRICTEMENT antérieur à la première d_date
   de la grille, puis forecast seul à chaque origine -- même mécanique que
   `nsdiff_multiseed_v2.py`. Le fit utilise `horizon=HORIZON_DAILY` (15), donc
   c'est LE MÊME modèle daily que le régime B d'oos_nsdiff_daily_weekly : on ne
   lit que le pas 1 de son chemin échantillonné, on ne le refit pas autrement.

4. **Anti-look-ahead par construction.** À l'origine `d_date` d'indice `pos`,
   la fenêtre passée à `forecast_from_fitted` est `daily_z[:pos]`. Comme
   `_log_returns` produit N-1 rendements où `r[i] = log(p[i+1]/p[i])`, la
   tranche `[:pos]` s'arrête au rendement `log(p[pos]/p[pos-1])` : elle contient
   l'information jusqu'à d_date INCLUS, et rien après. La distance train->test
   est donc : fits gelés sur l'avant-grille, historique récent vu uniquement par
   la fenêtre d'entrée de seq_len=30 jours. Vérifié par recalcul tronqué
   (`verify_no_lookahead`) et contre-épreuve par fuite injectée dans les tests.

## Config de référence (ensemble, alignée sur la piste oos hebdo)

5 graines (42-46) x 200 tirages = 1000, nuages CONCATÉNÉS (mélange des lois
prédictives), jamais moyenne de bornes déjà quantilées -- c'est la convention
actée par `repoint_oos_to_ensemble.py`. `predicted` = MÉDIANE du nuage concaténé
(le brief le spécifie ainsi ; la piste hebdo utilise la moyenne -- l'écart est
déclaré, pas subi), `pi_lower`/`pi_upper` = quantiles empiriques 2,5 / 97,5 %.

## Usage

    python experiments/oos_nsdiff_d1_simtrades.py                 # dry-run (défaut)
    python experiments/oos_nsdiff_d1_simtrades.py --apply         # sauvegarde + écriture + règles TC

Reprenable : un checkpoint JSON par (graine, actif) sous
`experiments/checkpoints_nsdiff_d1_simtrades/` -- relancer reprend où le run
s'est arrêté, aucun refit des couples déjà faits.
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import nsdiff_model as nm                                              # noqa: E402
from offline_prices import fetch_data_offline                          # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, HORIZON_DAILY   # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W as NSDIFF_EPOCHS  # noqa: E402

DB_PATH = str(ROOT / "validation" / "tracking.db")

# Modèle dont la grille oos fait référence : le plus complet des 6 (666 lignes,
# 167 dates), donc la borne supérieure de comparabilité.
GRID_MODEL = "ARIMA-GARCH"

SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES_PER_SEED = 200          # 5 x 200 = 1000, config de production (chantier A2)
K_DENOISE = nm.K_DENOISE          # fit-time pour NsDiff
RUN_ID = "20260810-NsDiff-oos-D1-simtrades"

FETCH_START = "2015-01-01"
FETCH_END = "2026-07-24"          # figé (pas "aujourd'hui") -- reproductibilité

# Tolérance relative de la vérification prix-vs-base. 1e-6 est le standard de
# _verify_origin_prices ; on le garde, un écart au-delà est un signal, pas du bruit.
PRICE_TOL = 1e-6

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_nsdiff_d1_simtrades"
OUT_PATH = Path(__file__).resolve().parent / "oos_nsdiff_d1_simtrades.json"

# Les 6 règles TC, dans l'ordre du brief. Appliquées telles quelles.
RULE_VERSIONS = ["bull_calm_d1", "pi95_conf", "bear_calm_d1",
                 "bear_stress_d1", "sideways_d1", "sideways_gated_d1"]
# fee_bps : LU, pas choisi. generate_sim_trades() est appelé sans fee_bps par
# sim_trades.main() (--ingest-oos), donc les lignes oos existantes ont été
# produites au défaut 0.0 -- revérifié sur une ligne réelle, où
# roi == (realized-ref)/ref exactement.
FEE_BPS = 0.0


class GridMismatchError(RuntimeError):
    """Levée dès qu'une clé ou un prix de la grille ne correspond pas -- le run
    s'arrête AVANT toute écriture (le standard qui a déjà sauvé 28 560 lignes)."""


# ── 1. grille héritée ───────────────────────────────────────────────────────

def load_grid(db_path: str = DB_PATH, grid_model: str = GRID_MODEL) -> pd.DataFrame:
    """Clés (asset, d_date) + ref/target/realized lues VERBATIM depuis les lignes
    oos du modèle de grille. Rien n'est recalculé ici, et rien ne doit l'être."""
    conn = sqlite3.connect(db_path)
    try:
        grid = pd.read_sql_query(
            "SELECT asset, d_date, target_date, reference_price, realized_price "
            "FROM all_predictions WHERE model = ? AND source = 'oos' AND horizon = 1 "
            "ORDER BY asset, d_date",
            conn, params=(grid_model,))
    finally:
        conn.close()
    if grid.empty:
        raise GridMismatchError(f"aucune ligne oos horizon=1 pour le modèle de grille {grid_model!r}")
    dupes = grid.duplicated(subset=["asset", "d_date"]).sum()
    if dupes:
        raise GridMismatchError(f"{dupes} clé(s) (asset, d_date) en double dans la grille {grid_model!r}")
    return grid


def existing_nsdiff_keys(db_path: str = DB_PATH) -> set:
    """Clés NsDiff oos D+1 DÉJÀ en base -- doit être vide avant le premier run
    (constat du brief : « NsDiff n'a que 218 lignes live et zéro ligne oos »)."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT asset, d_date FROM all_predictions "
            "WHERE model = 'NsDiff' AND source = 'oos' AND horizon = 1").fetchall()
    finally:
        conn.close()
    return {(a, d) for a, d in rows}


# ── 2. prix : récupérés pour le conditionnement, vérifiés contre la base ────

def audit_grid_prices(daily: pd.Series, grid_asset: pd.DataFrame, tol: float = PRICE_TOL) -> dict:
    """Confronte la série récupérée aux `reference_price` stockés. Ne lève rien --
    c'est `verify_grid_prices` qui décide. Retourne {n_missing, missing,
    n_mismatched, mismatched, max_abs_rel}."""
    index = {pd.Timestamp(ts).strftime("%Y-%m-%d"): i for i, ts in enumerate(daily.index)}
    missing, mismatched, max_rel = [], [], 0.0
    for _, row in grid_asset.iterrows():
        pos = index.get(row["d_date"])
        if pos is None:
            missing.append(row["d_date"])
            continue
        fetched, stored = float(daily.iloc[pos]), float(row["reference_price"])
        rel = abs(fetched - stored) / max(1e-12, abs(stored))
        max_rel = max(max_rel, rel)
        if abs(fetched - stored) > tol * max(1.0, abs(stored)):
            mismatched.append({"d_date": row["d_date"], "serie": fetched,
                               "base": stored, "ecart_rel_pct": rel * 100})
    return {"n_missing": len(missing), "missing": missing[:10],
            "n_mismatched": len(mismatched), "mismatched": mismatched[:10],
            "max_abs_rel_pct": max_rel * 100}


# Part maximale de d_date dont le NIVEAU de prix peut diverger de la base sans
# bloquer le run. Pourquoi une tolérance existe ici alors que les clés, elles,
# sont non-négociables : la série récupérée ne sert QU'À la fenêtre de
# conditionnement, qui est faite de LOG-RENDEMENTS. Un réajustement de dividendes
# multiplie toute la série par une constante et laisse donc les rendements
# rigoureusement inchangés -- la fenêtre est insensible au millésime. L'ancrage,
# lui, est le `reference_price` HÉRITÉ, jamais celui de la série. Une divergence
# de niveau isolée est donc sans effet sur `predicted`/`pi_*`, mais elle reste
# le symptôme d'un bug de données : elle est comptée, journalisée dans le JSON
# de sortie, et bloque au-delà de ce seuil (une divergence massive, elle,
# signifierait qu'on ne regarde pas la même série -- cas yfinance/TLT, 112/112).
MAX_MISMATCH_FRACTION = 0.05


def fetch_verified_daily(ticker: str, grid_asset: pd.DataFrame) -> tuple:
    """(série, label, audit) pour la source dont les prix collent LE MIEUX aux
    `reference_price` stockés -- yfinance puis cache gelé DONNEE~1.XLS, comme
    `fetch_verified` d'oos_nsdiff_daily_weekly, mais en choisissant la meilleure
    plutôt que la première : sur TLT, yfinance re-adjuste tout l'historique à
    chaque distribution (112/112 divergences) là où le cache gelé colle à 109/112.

    BLOQUANT si la meilleure source laisse une d_date absente (désalignement de
    clés, jamais toléré) ou dépasse MAX_MISMATCH_FRACTION de niveaux divergents."""
    errors, candidates = [], []
    for label, fetch_fn in (("live/yfinance", nm.fetch_data),
                            ("offline/DONNEE~1.XLS", fetch_data_offline)):
        try:
            daily = fetch_fn(ticker, FETCH_START, FETCH_END)
        except Exception as exc:                      # noqa: BLE001 -- on essaie la source suivante
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if daily is None or not len(daily):
            errors.append(f"{label}: série vide")
            continue
        candidates.append((label, daily, audit_grid_prices(daily, grid_asset)))

    if not candidates:
        raise GridMismatchError(f"[{ticker}] aucune source de prix exploitable -- " + " | ".join(errors))

    label, daily, audit = min(candidates, key=lambda c: (c[2]["n_missing"], c[2]["n_mismatched"]))
    if audit["n_missing"]:
        raise GridMismatchError(
            f"[{ticker}] {audit['n_missing']} d_date de la grille absente(s) de toutes les "
            f"sources de prix (ex. {audit['missing'][:3]}) -- désalignement de clés, run interrompu.")
    if audit["n_mismatched"] > MAX_MISMATCH_FRACTION * len(grid_asset):
        ex = [f"{m['d_date']}: série={m['serie']:.6f} vs base={m['base']:.6f}"
              for m in audit["mismatched"][:3]]
        raise GridMismatchError(
            f"[{ticker}] meilleure source ({label}) : {audit['n_mismatched']}/{len(grid_asset)} "
            f"niveaux de prix divergents, au-delà du seuil "
            f"{MAX_MISMATCH_FRACTION:.0%} (ex. {ex}) -- run interrompu.")
    return daily, label, audit


def verify_grid_prices(daily: pd.Series, grid_asset: pd.DataFrame, asset: str,
                       tol: float = PRICE_TOL) -> dict:
    """Version bloquante d'audit_grid_prices, conservée pour les tests et les
    appels directs : lève au moindre écart (clé absente OU niveau divergent)."""
    audit = audit_grid_prices(daily, grid_asset, tol=tol)
    if audit["n_missing"] or audit["n_mismatched"]:
        detail = []
        if audit["n_missing"]:
            detail.append(f"{audit['n_missing']} d_date absente(s) (ex. {audit['missing'][:3]})")
        if audit["n_mismatched"]:
            ex = [f"{m['d_date']}: série={m['serie']:.6f} vs base={m['base']:.6f}"
                  for m in audit["mismatched"][:3]]
            detail.append(f"{audit['n_mismatched']} prix divergent(s) (ex. {ex})")
        raise GridMismatchError(f"[{asset}] " + " ; ".join(detail))
    return audit


# ── 3. génération : un fit par (graine, actif), forecast seul par origine ───

def checkpoint_path(seed: int, asset: str) -> Path:
    return CHECKPOINT_DIR / f"seed{seed}_{asset.replace('=', '_')}.json"


def _standardized_returns(prices: pd.Series, mu: float, sd: float) -> np.ndarray:
    return (nm._log_returns(prices.values.astype(float)) - mu) / sd


def _date_index(daily: pd.Series) -> dict:
    return {pd.Timestamp(ts).strftime("%Y-%m-%d"): i for i, ts in enumerate(daily.index)}


def fit_asset(daily: pd.Series, grid_asset: pd.DataFrame, seed: int,
              epochs: int = NSDIFF_EPOCHS, k_denoise: int = K_DENOISE) -> tuple:
    """LE fit unique d'une (graine, actif) : sur l'historique STRICTEMENT
    antérieur à la première origine de la grille. Retourne (model, mu, sd).
    Séparé du forecast pour que le contrôle anti-look-ahead puisse rejouer les
    origines avec EXACTEMENT le même modèle -- sans quoi le test comparerait
    deux modèles différents et échouerait pour une mauvaise raison."""
    first_pos = min(_date_index(daily)[d] for d in grid_asset["d_date"])
    train = daily.iloc[:first_pos]        # STRICTEMENT avant la première origine
    nm.set_seed(seed)
    model, mu, sd = nm.fit_nsdiff(train, horizon=HORIZON_DAILY, epochs=epochs, k_denoise=k_denoise)
    return model, mu, sd


def forecast_grid(model, mu: float, sd: float, daily: pd.Series, grid_asset: pd.DataFrame,
                  seed: int, n_samples: int = N_SAMPLES_PER_SEED,
                  offsets: dict = None) -> dict:
    """{d_date: [n_samples prix]} par forecast SEUL (aucun refit). `offsets` mappe
    d_date -> indice de graine k pour reproduire `set_seed(seed + k)` à
    l'identique quand on ne rejoue qu'un sous-ensemble d'origines."""
    index = _date_index(daily)
    daily_z = _standardized_returns(daily, mu, sd)
    out = {}
    for k, (_, row) in enumerate(grid_asset.iterrows()):
        d_date = row["d_date"]
        pos = index[d_date]
        nm.set_seed(seed + (offsets[d_date] if offsets else k))
        # `daily_z[:pos]` s'arrête au rendement log(p[pos]/p[pos-1]) : information
        # jusqu'à d_date incluse, rien après. `last_price` = le reference_price
        # HÉRITÉ, pour ancrer la prévision sur le prix qu'utilisera la règle.
        samples = nm.forecast_from_fitted(
            model, daily_z[:pos], mu, sd, float(row["reference_price"]),
            horizons=[1], n_samples=n_samples)[1]
        out[d_date] = [float(x) for x in np.asarray(samples).ravel()]
    return out


def generate_seed_asset(asset: str, daily: pd.Series, grid_asset: pd.DataFrame, seed: int,
                        n_samples: int = N_SAMPLES_PER_SEED, epochs: int = NSDIFF_EPOCHS,
                        k_denoise: int = K_DENOISE) -> dict:
    """{d_date: [n_samples prix]} pour une (graine, actif) : un fit, puis forecast
    seul à chaque origine (train-once-forward)."""
    model, mu, sd = fit_asset(daily, grid_asset, seed, epochs=epochs, k_denoise=k_denoise)
    return forecast_grid(model, mu, sd, daily, grid_asset, seed, n_samples=n_samples)


def load_or_generate(asset: str, daily: pd.Series, grid_asset: pd.DataFrame, seed: int,
                     **kwargs) -> dict:
    path = checkpoint_path(seed, asset)
    if path.exists():
        print(f"[seed={seed}][{asset}] checkpoint réutilisé")
        return json.loads(path.read_text())
    t0 = time.time()
    clouds = generate_seed_asset(asset, daily, grid_asset, seed, **kwargs)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clouds))
    print(f"[seed={seed}][{asset}] {len(clouds)} origines en {time.time() - t0:.0f}s -> checkpoint")
    return clouds


def build_ensemble_rows(asset: str, grid_asset: pd.DataFrame, clouds_by_seed: dict) -> list:
    """Concatène les nuages des graines (JAMAIS une moyenne de bornes déjà
    quantilées) puis lit médiane et quantiles 2,5/97,5 % sur le nuage de 1000."""
    rows = []
    for _, row in grid_asset.iterrows():
        d_date = row["d_date"]
        cloud = np.concatenate([np.asarray(clouds_by_seed[s][d_date], dtype=float) for s in SEEDS])
        lo, hi = (float(q) for q in np.quantile(cloud, [0.025, 0.975]))
        rows.append({
            "run_id": RUN_ID, "model": "NsDiff", "asset": asset, "horizon": 1,
            "regime": "unknown", "cutoff_date": d_date, "target_date": row["target_date"],
            "last_close": float(row["reference_price"]),
            "y_pred": float(np.median(cloud)), "y_lower": lo, "y_upper": hi,
            "y_true": None if pd.isna(row["realized_price"]) else float(row["realized_price"]),
            "source": "oos", "frequence": "daily", "horizon_type": "daily",
            "horizon_unit": "D+1", "n_samples_total": int(cloud.size),
        })
    return rows


# ── 4. contrôle anti-look-ahead par recalcul tronqué ────────────────────────

def verify_no_lookahead(asset: str, daily: pd.Series, grid_asset: pd.DataFrame, seed: int,
                        n_dates: int = 5, n_samples: int = 8, epochs: int = 2) -> list:
    """Recalcule quelques origines sur une série TRONQUÉE juste après d_date : si
    la prévision dépendait d'un prix futur, elle changerait. Égalité exacte
    exigée. Budget volontairement minuscule (epochs=2, n_samples=8) : on teste le
    câblage de la fenêtre, pas la qualité du modèle.

    Retourne la liste des d_date contrôlées. Lève GridMismatchError au moindre écart."""
    index = _date_index(daily)
    dates = list(grid_asset["d_date"])
    if len(dates) > n_dates:                      # première et dernière TOUJOURS incluses
        step = (len(dates) - 1) / (n_dates - 1)
        picked = sorted({dates[min(len(dates) - 1, round(i * step))] for i in range(n_dates)})
    else:
        picked = dates
    offsets = {d: k for k, d in enumerate(dates)}   # l'indice de graine du run complet

    # UN SEUL modèle, réutilisé des deux côtés : seule la série change.
    model, mu, sd = fit_asset(daily, grid_asset, seed, epochs=epochs)
    sub = grid_asset[grid_asset["d_date"].isin(picked)].reset_index(drop=True)
    full = forecast_grid(model, mu, sd, daily, sub, seed, n_samples=n_samples, offsets=offsets)

    for d_date in picked:
        truncated = daily.iloc[:index[d_date] + 1]   # plus AUCUN prix après d_date
        one = sub[sub["d_date"] == d_date].reset_index(drop=True)
        trunc = forecast_grid(model, mu, sd, truncated, one, seed,
                              n_samples=n_samples, offsets=offsets)
        if not np.array_equal(np.asarray(full[d_date]), np.asarray(trunc[d_date])):
            raise GridMismatchError(
                f"[{asset}] LOOK-AHEAD à {d_date} : la prévision change quand on tronque "
                f"la série après cette date (série complète={full[d_date][:3]} vs "
                f"tronquée={trunc[d_date][:3]})")
    return picked


# ── 5. écriture ────────────────────────────────────────────────────────────

def fingerprint_untouched(db_path: str, run_id: str = RUN_ID) -> str:
    """Empreinte SHA-256 de TOUT `predictions` + `sim_trades` SAUF les lignes de
    ce run. Comparée avant/après : rien d'autre ne doit bouger d'un octet.

    Volontairement plus strict qu'un « tout sauf NsDiff » : NsDiff possède déjà
    des lignes oos HEBDOMADAIRES (l'ensemble 5x200 repointé) et des lignes live.
    Les exclure de l'empreinte reviendrait à ne pas surveiller le voisinage le
    plus exposé -- une insertion mal filtrée ou un upsert trop large les
    toucherait sans qu'on le voie."""
    conn = sqlite3.connect(db_path)
    try:
        h = hashlib.sha256()
        for table in ("predictions", "sim_trades"):
            cur = conn.execute(f"SELECT * FROM {table} WHERE run_id != ? ORDER BY id", (run_id,))
            for row in cur:
                h.update(repr(row).encode())
        return h.hexdigest()
    finally:
        conn.close()


def insert_prediction_rows(rows: list, db_path: str) -> int:
    """Insère les lignes NsDiff D+1 dans `predictions` (la vue all_predictions les
    exposera : horizon=1, horizon_type='daily', daily_duplicate=0). Idempotent via
    l'index unique partiel `idx_predictions_oos_unique`."""
    conn = sqlite3.connect(db_path)
    try:
        n = 0
        for r in rows:
            cur = conn.execute("""
                INSERT OR IGNORE INTO predictions (
                    run_id, model, asset, horizon, cutoff_date, target_date, regime,
                    last_close, y_pred, y_lower, y_upper, y_true, source,
                    frequence, horizon_type, horizon_unit, daily_duplicate, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)
            """, (r["run_id"], r["model"], r["asset"], r["horizon"], r["cutoff_date"],
                  r["target_date"], r["regime"], r["last_close"], r["y_pred"], r["y_lower"],
                  r["y_upper"], r["y_true"], r["source"], r["frequence"], r["horizon_type"],
                  r["horizon_unit"], pd.Timestamp.utcnow().isoformat()))
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--apply", action="store_true",
                   help="écrit réellement en base (défaut : dry-run, aucune écriture)")
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_SEED)
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS)
    p.add_argument("--skip-lookahead-check", action="store_true",
                   help="saute le recalcul tronqué (déconseillé -- il est rapide)")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    grid = load_grid(args.db_path)
    already = existing_nsdiff_keys(args.db_path)
    print(f"Grille {GRID_MODEL} : {len(grid)} clés, {grid['asset'].nunique()} actifs, "
          f"{grid['d_date'].nunique()} dates ({grid['d_date'].min()} -> {grid['d_date'].max()})")
    if already:
        print(f"  ATTENTION : {len(already)} ligne(s) NsDiff oos D+1 déjà en base "
              f"-- elles seront ignorées par l'insertion idempotente.")

    all_rows, per_asset, lookahead_checked = [], {}, {}
    for asset in args.assets:
        grid_asset = grid[grid["asset"] == asset].reset_index(drop=True)
        if grid_asset.empty:
            print(f"[{asset}] absent de la grille -- ignoré")
            continue
        daily, source_label, audit = fetch_verified_daily(asset, grid_asset)
        print(f"[{asset}] prix: {source_label} -- {len(grid_asset)} clés alignées, "
              f"{audit['n_mismatched']} niveau(x) divergent(s) "
              f"(écart max {audit['max_abs_rel_pct']:.4f} %)")
        if audit["n_mismatched"]:
            print(f"  divergences de NIVEAU (sans effet sur les log-rendements ni sur "
                  f"l'ancrage hérité, mais journalisées) : "
                  f"{[m['d_date'] for m in audit['mismatched']]}")

        if not args.skip_lookahead_check:
            lookahead_checked[asset] = verify_no_lookahead(asset, daily, grid_asset, args.seeds[0])
            print(f"[{asset}] anti-look-ahead OK sur {len(lookahead_checked[asset])} dates "
                  f"(dont la première et la dernière)")

        clouds_by_seed = {s: load_or_generate(asset, daily, grid_asset, s,
                                              n_samples=args.n_samples, epochs=args.epochs)
                          for s in args.seeds}
        rows = build_ensemble_rows(asset, grid_asset, clouds_by_seed)
        all_rows.extend(rows)
        per_asset[asset] = {"n_rows": len(rows), "price_source": source_label,
                            "d_date_min": grid_asset["d_date"].min(),
                            "d_date_max": grid_asset["d_date"].max(),
                            "price_audit": audit}

    payload = {
        "run_id": RUN_ID, "applied": bool(args.apply), "db_path": args.db_path,
        "grid_model": GRID_MODEL, "n_rows": len(all_rows),
        "config": {"seeds": args.seeds, "n_samples_per_seed": args.n_samples,
                   "n_samples_total": len(args.seeds) * args.n_samples,
                   "epochs": args.epochs, "k_denoise": K_DENOISE,
                   "seq_len": nm.SEQ_LEN, "fit_horizon": HORIZON_DAILY,
                   "point": "médiane du nuage concaténé",
                   "interval": "quantiles empiriques 0.025 / 0.975 du nuage concaténé",
                   "aggregation": "concaténation des nuages, PAS moyenne des bornes",
                   "refit": "aucun -- train-once-forward, fit sur l'avant-grille",
                   "fee_bps": FEE_BPS},
        "per_asset": per_asset, "lookahead_checked": lookahead_checked,
        "rule_versions": RULE_VERSIONS,
    }

    if not args.apply:
        payload["note"] = "dry-run : aucune écriture. Relancer avec --apply."
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n--dry-run : {len(all_rows)} ligne(s) prêtes, rien écrit. Plan -> {args.out}")
        return

    before = fingerprint_untouched(args.db_path)
    backup = Path(args.db_path).with_suffix(
        f".db.bak_nsdiff_d1_simtrades_{time.strftime('%Y-%m-%dT%H%M%S')}")
    shutil.copy2(args.db_path, backup)
    print(f"\nSauvegarde -> {backup.name}")

    n_pred = insert_prediction_rows(all_rows, args.db_path)
    print(f"predictions : {n_pred} ligne(s) NsDiff D+1 insérée(s)")

    sys.path.insert(0, str(ROOT / "validation"))
    import sim_trades as st                                            # noqa: E402
    n_trades = {}
    for rule in RULE_VERSIONS:
        n_trades[rule] = st.generate_sim_trades(db_path=args.db_path, rule_version=rule,
                                                fee_bps=FEE_BPS, source="oos")
        print(f"sim_trades[{rule}] : +{n_trades[rule]}")

    after = fingerprint_untouched(args.db_path)
    if before != after:
        raise GridMismatchError(
            "des lignes hors de ce run ont changé pendant l'écriture -- "
            f"empreinte avant={before[:16]} après={after[:16]}. Restaurer {backup.name}.")
    print("Empreinte de tout le reste de la base : INCHANGÉE")

    payload.update({"n_predictions_inserted": n_pred, "n_sim_trades_by_rule": n_trades,
                    "backup": str(backup), "untouched_fingerprint_stable": True})
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
