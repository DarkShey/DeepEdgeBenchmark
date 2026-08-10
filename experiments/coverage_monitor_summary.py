"""
coverage_monitor_summary.py -- H3 en routine : le resume lisible des alertes de
couverture, ecrit dans le resume de job GitHub Actions (`$GITHUB_STEP_SUMMARY`).

POURQUOI UN SCRIPT SEPARE. `coverage_monitor.py` produit un JSON de 180 cellules ;
personne ne l'ouvre chaque matin. Un suivi en routine ne vaut que si l'anomalie
saute aux yeux la ou on regarde deja -- ici, le resume du job quotidien. Ce script
ne recalcule rien : il lit les artefacts que le pas precedent vient d'ecrire et en
tire ce qu'un operateur doit voir en dix secondes.

CE QU'IL MONTRE, dans cet ordre :
  1. le compte d'alertes par piste, avec le sens (sous- ou sur-couverture) ;
  2. les DIX pires cellules en sous-couverture -- c'est le defaut qui coute, la
     sur-couverture ne coute qu'en Winkler ;
  3. les episodes en cours les plus longs -- une derive installee depuis dix
     semaines n'est pas la meme chose que le bruit d'une semaine.

Aucune ecriture, aucune base ouverte : il ne lit que du JSON.

Usage : python coverage_monitor_summary.py [--top 10]
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACKS = {"oos (dashboard)": HERE / "coverage_monitor.json",
          "oos2020 (grille regeneree)": HERE / "coverage_monitor_grid2020.json"}


def current_episode_length(cell: dict) -> int:
    """Longueur de l'episode EN COURS : l'episode qui touche la derniere origine.
    Un episode clos appartient au passe, il n'appelle pas d'action aujourd'hui."""
    eps, n = cell.get("episodes") or [], cell.get("n", 0)
    last = eps[-1] if eps else None
    return last["length"] if last and last["end"] == n - 1 else 0


def summarise(path: Path, top: int) -> list:
    if not path.exists():
        return [f"_artefact absent : `{path.name}` — le pas de monitoring n'a pas tourné._", ""]
    d = json.loads(path.read_text())
    cells = d["per_cell"]
    under = {k: v for k, v in cells.items() if v["status"] == "sous_couverture"}
    over = {k: v for k, v in cells.items() if v["status"] == "sur_couverture"}
    band = d["declared"]["band"]

    out = [f"**{d['n_flagged']} cellules en alerte sur {d['n_cells']}** "
           f"— {len(under)} en sous-couverture, {len(over)} en sur-couverture "
           f"(fenêtre {d['declared']['window']} origines, bande "
           f"[{band[0]:.2f} ; {band[1]:.2f}])", ""]
    if not under:
        out += ["_Aucune sous-couverture._", ""]
        return out

    out += [f"| cellule | couverture courante | plein échantillon | épisode en cours |",
            "|---|---:|---:|---:|"]
    worst = sorted(under.items(), key=lambda kv: kv[1]["coverage_current_window"])[:top]
    for key, v in worst:
        ep = current_episode_length(v)
        out.append(f"| `{key}` | {v['coverage_current_window']:.3f} | "
                   f"{v['coverage_full_sample']:.3f} | "
                   f"{str(ep) + ' origines' if ep else '—'} |")
    out.append("")
    longest = max(under.items(), key=lambda kv: current_episode_length(kv[1]))
    ep = current_episode_length(longest[1])
    if ep:
        out += [f"Épisode en cours le plus long : `{longest[0]}`, **{ep} origines** "
                f"consécutives hors bande.", ""]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    lines = ["## Suivi de couverture (H3)", "",
             "Sortir de la bande déclenche une **investigation**, pas un verdict : "
             "le test formel reste Kupiec sur l'échantillon complet.", ""]
    for label, path in TRACKS.items():
        lines += [f"### Piste `{label}`", ""] + summarise(path, args.top)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
