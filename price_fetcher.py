"""
price_fetcher.py — implémentation yfinance de la fonction price_fetcher(asset, target_date)
attendue par tracking_db.evaluate_pending() (cf. BRIEF_tracking_db.md §8).

Volontairement séparé de tracking_db.py : le cœur du module de suivi reste
bibliothèque standard uniquement et testable hors-ligne (on injecte un mock
price_fetcher dans les tests) ; ce fichier-ci est le seul endroit du projet
Partie B qui touche au réseau.
"""

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


def yfinance_price_fetcher(asset: str, target_date: str):
    """Retourne la clôture ajustée de `asset` à `target_date` (ISO 'YYYY-MM-DD'),
    ou None si indisponible (week-end, jour férié, donnée pas encore publiée)."""
    start = target_date
    end = (date.fromisoformat(target_date) + timedelta(days=1)).isoformat()

    data = yf.download(asset, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return float(data["Close"].iloc[0])
