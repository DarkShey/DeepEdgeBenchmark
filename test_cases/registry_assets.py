"""
test_cases/registry_assets.py — Tableau des actifs couverts par les test cases
===============================================================================
Réutilise calibration.regime.assets.ASSETS (source unique de vérité déjà utilisée
par le reste du repo — pas de duplication) et associe à chaque ticker le CSV
d'historique (Close + Regime quotidien) produit par test_cases/convert_source_data.py.

Pour ajouter un actif, voir test_cases/README.md.
"""

from pathlib import Path

from calibration.regime.assets import ASSETS as _BASE_ASSETS

DATA_DIR = Path(__file__).parent / "data"

# Le nom de feuille dans DONNEE~1.XLS est identique au "label" de calibration/regime/assets.py
# (vérifié : 'Bitcoin', 'Ethereum', 'S&P 500 (SPY)', 'US Treasury 10Y Note Futures',
# 'US Treasury 20+Y (ETF)'). On le réutilise tel quel comme nom de feuille source.
ASSETS = [
    {
        "ticker": a["ticker"],
        "label": a["label"],
        "short": a["short"],
        "asset_class": a["asset_class"],
        "color": a["color"],
        "sheet_name": a["label"],
        "csv_path": DATA_DIR / f"{a['ticker'].replace('=', '_')}.csv",
    }
    for a in _BASE_ASSETS
]

ASSET_BY_TICKER = {a["ticker"]: a for a in ASSETS}
