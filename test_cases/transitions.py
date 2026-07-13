"""
test_cases/transitions.py — Détection des transitions de régime + sélection des occurrences
=============================================================================================
Lit test_cases/data/<ticker>.csv (Close + Regime quotidien, déjà calculé par le moteur
calibration/regime — cf. test_cases/convert_source_data.py) et repère les transitions de
régime demandées par chaque test case (registry_test_cases.TEST_CASES).

Convention (cohérente avec model_artifacts/pipeline.py) : pour une transition détectée à
la position `i` (1er jour du nouveau régime), `cutoff = i-1` (dernier jour de l'ancien
régime, dernière donnée connue avant prévision) et la cible à l'horizon `h` (en jours de
bourse, la série ne contient que des jours de bourse) est la ligne `cutoff + h`.
"""

import pandas as pd

from test_cases.registry_assets import ASSET_BY_TICKER
from test_cases.registry_test_cases import MAX_OCCURRENCES_PER_ASSET


def load_series(ticker: str) -> pd.DataFrame:
    """Charge l'historique (Date, Close, Sigma_t_pct, ..., Regime) d'un actif, trié
    chronologiquement, index entier 0..n-1 (positions utilisées pour cutoff/horizon)."""
    csv_path = ASSET_BY_TICKER[ticker]["csv_path"]
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def find_raw_transitions(df: pd.DataFrame, regime_from: str, regime_to: str) -> list:
    """Positions (entiers) des lignes où Regime passe de regime_from (ligne i-1) à
    regime_to (ligne i) — i = 1er jour du nouveau régime."""
    regimes = df["Regime"].tolist()
    return [i for i in range(1, len(regimes)) if regimes[i - 1] == regime_from and regimes[i] == regime_to]


def tc12_stress_filter(df: pd.DataFrame, i: int) -> bool:
    """Filtre TC1.2 (bull->stress) : confirme que la transition est un "vrai" décrochage,
    pas un simple relabelling HMM en limite. Bande de référence générique construite avec
    les colonnes déjà présentes dans le fichier de données (Close/Sigma_t_pct du dernier
    jour "bull", i-1) — volontairement PAS un forecast d'un des 5 modèles, pour éviter une
    dépendance circulaire avant même d'avoir lancé les modèles :

        PI_low(D) = Close(D-1) * (1 - 1.96 * Sigma_t_pct(D-1) / 100)

    Condition retenue : Close(D) < PI_low(D) — le prix réel au jour D casse déjà la borne
    basse à 95% qu'une bande de volatilité GARCH aurait projetée la veille pour ce jour.
    Cf. test_cases/README.md pour la discussion de cette interprétation (isolée ici : la
    corriger ne change rien d'autre à l'architecture)."""
    if i < 1:
        return False
    close_cutoff = df["Close"].iloc[i - 1]
    sigma_cutoff = df["Sigma_t_pct"].iloc[i - 1]
    close_d = df["Close"].iloc[i]
    if pd.isna(close_cutoff) or pd.isna(sigma_cutoff) or pd.isna(close_d):
        return False
    pi_low = close_cutoff * (1 - 1.96 * sigma_cutoff / 100.0)
    return close_d < pi_low


EXTRA_FILTERS = {
    "tc12_stress_filter": tc12_stress_filter,
}


def select_occurrences(ticker: str, test_case: dict, max_n: int = None) -> list:
    """Retourne les `max_n` occurrences les plus récentes (triées croissant) d'un test
    case pour un actif, chacune sous la forme :
        {"transition_date", "cutoff_date", "cutoff_idx", "last_close",
         "target_dates": {h: date}, "target_actuals": {h: close réel}}
    Liste vide si aucune transition ne correspond (ex. TLT pour TC1.2). Les occurrences
    trop proches de la fin de la série pour calculer le plus grand horizon demandé sont
    exclues (pas de cible réelle disponible)."""
    max_n = max_n if max_n is not None else MAX_OCCURRENCES_PER_ASSET
    df = load_series(ticker)

    candidates = find_raw_transitions(df, test_case["regime_from"], test_case["regime_to"])

    filter_name = test_case.get("extra_filter")
    if filter_name:
        filter_fn = EXTRA_FILTERS[filter_name]
        candidates = [i for i in candidates if filter_fn(df, i)]

    horizons = test_case["horizons_days"]
    max_h = max(horizons)
    candidates = [i for i in candidates if (i - 1 + max_h) < len(df)]

    candidates = sorted(candidates)
    if max_n:
        candidates = candidates[-max_n:]

    occurrences = []
    for i in candidates:
        cutoff_idx = i - 1
        occ = {
            "transition_date": df["Date"].iloc[i],
            "cutoff_date": df["Date"].iloc[cutoff_idx],
            "cutoff_idx": int(cutoff_idx),
            "last_close": float(df["Close"].iloc[cutoff_idx]),
            "target_dates": {},
            "target_actuals": {},
        }
        for h in horizons:
            t_idx = cutoff_idx + h
            occ["target_dates"][h] = df["Date"].iloc[t_idx]
            occ["target_actuals"][h] = float(df["Close"].iloc[t_idx])
        occurrences.append(occ)
    return occurrences
