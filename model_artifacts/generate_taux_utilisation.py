"""
model_artifacts/generate_taux_utilisation.py — Matrice modèle × actif du taux
d'utilisation, restreinte aux vraies prédictions (real_flag='live').

Même définition de "taux d'utilisation" que le dashboard (cf. computeUsage en JS dans
generate_dashboard.py) : une ligne jour×modèle est "utilisable" si au moins un test
case TC1.1-TC1.5 s'y déclenche avec un compteur résolu positif ; le taux rapporte ce
compte au nombre total de lignes de la sélection (signaux encore ouverts comptés
non-utilisables). Ici recalculé côté Python (pas de JS/JSON à charger) pour produire un
fichier HTML autonome, statique, committable tel quel à la racine du dépôt.

Exécution (depuis DeepEdgeBenchmark/) :
    python -m model_artifacts.generate_taux_utilisation
    python -m model_artifacts.generate_taux_utilisation --out taux_utilisation.html
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT))
from validation import sim_trades as st
from model_artifacts.generate_dashboard import (
    SIM_TRADES_ASSETS, SIM_TRADES_DB_PATH, SIM_TRADES_MODELS,
)

DEFAULT_OUT = REPO_ROOT / "taux_utilisation.html"

# Seuils de couleur (bornes basses incluses), cf. demande explicite.
BANDS = [
    (70, "band-green"),
    (60, "band-blue"),
    (50, "band-yellow"),
    (0, "band-red"),
]


def _band(pct: float) -> str:
    for lower, cls in BANDS:
        if pct >= lower:
            return cls
    return "band-red"


def _usage(rows: list) -> tuple:
    """(usable, total) sur des lignes daily_detail déjà filtrées real_flag='live'."""
    total = len(rows)
    usable = 0
    for r in rows:
        row_usable = any(
            sig["counter"] is not None and sig["counter"] > 0 for sig in r["signals"]
        )
        if row_usable:
            usable += 1
    return usable, total


def compute_matrix(db_path: str = SIM_TRADES_DB_PATH, models: list = None,
                    assets: list = None) -> list:
    """Une entrée par modèle, dans l'ordre de `models` : {"model": str, "cells": [(usable,
    total), ...]} aligné sur `assets`."""
    models = models if models is not None else SIM_TRADES_MODELS
    assets = assets if assets is not None else SIM_TRADES_ASSETS

    by_asset = {
        asset: [r for r in st.daily_detail(db_path=db_path, asset=asset, models=models)
                if r["real_flag"] == "live"]
        for asset in assets
    }
    matrix = []
    for model in models:
        cells = [_usage([r for r in by_asset[asset] if r["model"] == model])
                 for asset in assets]
        matrix.append({"model": model, "cells": cells})
    return matrix


def render_html(matrix: list, assets: list, generated_at: str) -> str:
    def cell_html(usable: int, total: int) -> str:
        if total == 0:
            return '<td class="cell"><div class="cell-inner"><span class="pct">—</span></div></td>'
        pct = usable / total * 100
        cls = _band(pct)
        pct_text = f"{pct:.1f}".replace(".", ",") + " %"
        return (
            f'<td class="cell {cls}"><div class="cell-inner">'
            f'<span class="pct">{pct_text}</span>'
            f'<span class="frac">{usable} / {total}</span></div></td>'
        )

    header_cells = "".join(f"<th>{asset}</th>" for asset in assets)
    body_rows = "".join(
        f'<tr><th>{row["model"]}</th>'
        + "".join(cell_html(usable, total) for usable, total in row["cells"])
        + "</tr>"
        for row in matrix
    )

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taux d'utilisation — vraies prédictions</title>
<style>
  :root {{
    --bg: #f5f6f8;
    --surface: #ffffff;
    --ink: #1c2230;
    --ink-soft: #4b5468;
    --ink-mute: #8790a3;
    --line: #dde1e9;

    --band-red-bg:    #f6d3d3;
    --band-red-ink:   #7a1f1f;
    --band-yellow-bg: #f8e3ab;
    --band-yellow-ink:#6b4c06;
    --band-blue-bg:   #cfe0f4;
    --band-blue-ink:  #1c3f66;
    --band-green-bg:  #cde8d5;
    --band-green-ink: #185c34;
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14171e;
      --surface: #1b1f29;
      --ink: #e8eaf0;
      --ink-soft: #aab0c2;
      --ink-mute: #6d7488;
      --line: #2c3140;

      --band-red-bg:    #4a2020;
      --band-red-ink:   #f3b6b6;
      --band-yellow-bg: #4a3c10;
      --band-yellow-ink:#f3d98a;
      --band-blue-bg:   #1f3555;
      --band-blue-ink:  #a9c8ee;
      --band-green-bg:  #1c3f2b;
      --band-green-ink: #a4dcb8;
    }}
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, "Segoe UI", "Inter", system-ui, sans-serif;
    padding: 40px 24px 64px;
  }}

  main {{ max-width: 880px; margin: 0 auto; }}

  h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0 0 6px;
  }}

  .subtitle {{
    color: var(--ink-soft);
    font-size: 0.92rem;
    line-height: 1.5;
    max-width: 62ch;
    margin: 0;
  }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 22px 0 24px; }}

  .legend-chip {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 5px 11px 5px 9px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--line);
    font-size: 0.78rem;
    color: var(--ink-soft);
  }}

  .legend-swatch {{ width: 10px; height: 10px; border-radius: 3px; flex: none; }}

  .table-wrap {{
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: var(--surface);
  }}

  table {{ border-collapse: collapse; width: 100%; min-width: 640px; }}

  thead th {{
    text-align: center;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--ink-mute);
    padding: 14px 12px 12px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }}

  thead th:first-child {{ text-align: left; color: var(--ink-soft); }}

  tbody th {{
    text-align: left;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
    padding: 13px 16px;
    white-space: nowrap;
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
  }}

  tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: none; }}

  td.cell {{ text-align: center; border-bottom: 1px solid var(--line); }}

  .cell-inner {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
    padding: 12px 10px;
  }}

  .pct {{ font-variant-numeric: tabular-nums; font-size: 0.98rem; font-weight: 700; }}
  .frac {{ font-variant-numeric: tabular-nums; font-size: 0.68rem; opacity: 0.75; }}

  .band-red    {{ background: var(--band-red-bg);    color: var(--band-red-ink); }}
  .band-yellow {{ background: var(--band-yellow-bg); color: var(--band-yellow-ink); }}
  .band-blue   {{ background: var(--band-blue-bg);   color: var(--band-blue-ink); }}
  .band-green  {{ background: var(--band-green-bg);  color: var(--band-green-ink); }}

  footer {{
    margin-top: 22px;
    font-size: 0.76rem;
    line-height: 1.6;
    color: var(--ink-mute);
    max-width: 68ch;
  }}

  footer code {{
    font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
    font-size: 0.72rem;
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 1px 5px;
    border-radius: 4px;
  }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Taux d'utilisation — vraies prédictions</h1>
    <p class="subtitle">
      Part des jours où au moins un signal (TC1.1–TC1.5) s'est déclenché et a résolu
      positivement, rapportée au nombre total de jours de prédiction, par modèle et par
      actif. Restreint aux vraies prédictions (<code>real_flag = 'live'</code>), horizon
      D+1.
    </p>
  </header>

  <div class="legend">
    <span class="legend-chip"><span class="legend-swatch" style="background:var(--band-red-bg)"></span>&lt; 50 %</span>
    <span class="legend-chip"><span class="legend-swatch" style="background:var(--band-yellow-bg)"></span>50 % – 60 %</span>
    <span class="legend-chip"><span class="legend-swatch" style="background:var(--band-blue-bg)"></span>60 % – 70 %</span>
    <span class="legend-chip"><span class="legend-swatch" style="background:var(--band-green-bg)"></span>&ge; 70 %</span>
  </div>

  <div class="table-wrap">
    <table>
      <thead><tr><th>Modèle</th>{header_cells}</tr></thead>
      <tbody>{body_rows}</tbody>
    </table>
  </div>

  <footer>
    Généré le {generated_at} depuis <code>validation/tracking.db</code>
    (<code>validation/sim_trades.py::daily_detail</code>), régénéré automatiquement à
    chaque exécution de <code>validation/evaluate_daily.py</code> (workflow
    <code>evaluate-daily.yml</code>). Sous chaque taux : nombre de jours utilisables /
    nombre total de jours vraies-prédictions disponibles pour ce couple modèle-actif.
  </footer>
</main>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default=SIM_TRADES_DB_PATH)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    matrix = compute_matrix(db_path=args.db_path)
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z")
    html = render_html(matrix, SIM_TRADES_ASSETS, generated_at)

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"[generate_taux_utilisation] écrit -> {out_path}")


if __name__ == "__main__":
    main()
