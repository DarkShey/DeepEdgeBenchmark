"""
patch_gate0_branche2_holm.py -- PATCH_gate0_branche2_et_holm_m2.md : hygiene
documentaire du chantier 0 CTA.

CE QUE CE PATCH FAIT, ET CE QU'IL NE FAIT PAS. Il repare un INSTRUMENT et corrige
une DECLARATION. Il ne rejoue aucun verdict et n'en modifie aucun : la porte 0
echoue sous les trois variantes de signal, avant comme apres la reparation de la
branche 2 -- ce script le REJOUE et l'ecrit dans son artefact plutot que de
l'affirmer.

  P1  BRIEF -- « Famille de Holm primaire (m = 4) » est incoherent avec la famille
      qu'il decrit ({SPY-ES, ZN-FUT} x {W+1}, A vs B = 2 tests). Corrige en m = 2,
      avec la trace de la correction. Legitimite : les chantiers 1-2 n'ont jamais
      tourne (arret a la porte 0), la correction reste donc une declaration a
      priori ; et l'erreur allait dans le sens conservateur (m surdeclare = seuils
      plus stricts). La famille n'est PAS etendue a W+2/W+3 pour « justifier » le
      4 : W+1 est l'horizon de detention declare.

  P2  NOTE -- la branche 2, dont le defaut est documente au §2.4, est reparee dans
      `cta_gate0.branch2_verdict` (elle porte desormais sur l'EXCES vs
      acheter-et-garder). La piste de reouverture n° 2 du §4 est donc traitee :
      elle sort de la liste des pistes ouvertes, et le §2.4 dit que la correction
      est posterieure au verdict sans l'affecter.

  P3  Le verrou de calendrier vit dans les tests
      (`test_le_signal_gele_reste_sur_la_convention_de_deita`), pas ici : rien a
      patcher dans les documents, la decision y est deja consignee.

DISCIPLINE, celle de `patch_note_decision` / `repoint_oos_to_m200` : dry-run par
defaut, REPERAGE PAR CONTENU (jamais par numero de ligne -- un patch qui compte
les lignes casse au premier ajout de paragraphe), sauvegarde horodatee, `--apply`
explicite, IDEMPOTENT (il verifie le texte attendu, et si la correction est deja
en place il le dit au lieu de la refaire ou d'echouer).

Sortie : experiments/patch_gate0_branche2_holm.json
Usage :
    python patch_gate0_branche2_holm.py            # dry-run : montre les diffs
    python patch_gate0_branche2_holm.py --apply    # sauvegarde + patch
"""

import argparse
import difflib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = Path(__file__).resolve().parent / "patch_gate0_branche2_holm.json"
BRIEF = ROOT / "BRIEF_couplage_cta_deita_sizing_nsdiff.md"
NOTE = ROOT / "NOTE_couplage_cta_deita_sizing_nsdiff.md"
GATE_ARTEFACTS = ("cta_gate0.json", "cta_gate0_own_calendar.json", "cta_gate0_conviction.json")

# Chaque edition est (fichier, etiquette, texte_attendu, texte_nouveau). Le
# reperage se fait sur `expected`, qui doit apparaitre EXACTEMENT UNE FOIS.
EDITS = [
    (BRIEF, "P1 -- famille de Holm m = 4 -> m = 2",
     "- **Famille de Holm primaire (m = 4)** : A vs B sur {SPY-ES, ZN-FUT} × {W+1}, "
     "les deux instruments à frais bas.",
     "- **Famille de Holm primaire (m = 2)** : A vs B sur {SPY-ES, ZN-FUT} × {W+1}, "
     "les deux instruments à frais bas. *(m = 2, corrigé le 2026-08-08 avant tout "
     "run — le brief annonçait m = 4 pour une famille qui en compte 2 ; cf. NOTE §3 "
     "et PATCH_gate0_branche2_et_holm_m2.md. La famille n'est pas étendue à W+2/W+3 "
     "pour justifier le 4 : W+1 est l'horizon de détention déclaré.)*"),

    (NOTE, "P1 -- remarque du §3 : l'ecart est tranche",
     "**Une remarque sur la famille de Holm déclarée, pour le jour où la question\n"
     "rouvrirait** : le brief annonce m = 4 pour « A vs B sur {SPY-ES, ZN-FUT} × {W+1} »,\n"
     "ce qui fait 2 tests, pas 4. L'écart devra être tranché avant les runs — pas après.",
     "**La famille de Holm déclarée a été corrigée, avant tout run** : le brief\n"
     "annonçait m = 4 pour « A vs B sur {SPY-ES, ZN-FUT} × {W+1} », qui compte 2 tests.\n"
     "Le brief porte désormais **m = 2**, avec la trace de la correction. Elle reste\n"
     "une déclaration *a priori* — les chantiers 1-2 n'ont jamais tourné — et l'erreur\n"
     "allait dans le sens conservateur (m surdéclaré = seuils plus stricts). La famille\n"
     "n'a pas été étendue à W+2/W+3 pour justifier le 4 : W+1 est l'horizon de\n"
     "détention déclaré."),

    (NOTE, "P2 -- §2.4 : la branche 2 est reparee",
     "> Sur un panel haussier, un Sharpe poolé positif mesure le bêta du panel, pas\n"
     "> l'edge du signal. La branche 2 est franchissable par n'importe quel « signal »\n"
     "> toujours long. Seule la branche 1 — la comparaison appariée contre\n"
     "> acheter-et-garder — teste ce que la porte prétend tester.",
     "> Sur un panel haussier, un Sharpe poolé positif mesure le bêta du panel, pas\n"
     "> l'edge du signal. La branche 2 est franchissable par n'importe quel « signal »\n"
     "> toujours long. Seule la branche 1 — la comparaison appariée contre\n"
     "> acheter-et-garder — teste ce que la porte prétend tester.\n"
     "\n"
     "**Réparé depuis** (`PATCH_gate0_branche2_et_holm_m2.md`, P2) : la branche 2 porte\n"
     "désormais sur l'**excès** — PnL du signal moins PnL d'acheter-et-garder, apparié\n"
     "par origine — au lieu du PnL brut. Un signal constant a un excès identiquement\n"
     "nul, donc l'échec devient mécanique quelle que soit la pente du marché. La\n"
     "correction est **postérieure au verdict et ne l'affecte pas**, et ce n'est pas une\n"
     "affirmation : les trois variantes ont été rejouées sous la branche réparée et\n"
     "restent en échec. Le cas historique bascule comme attendu — la conviction\n"
     "dégénérée passait la branche 2 d'origine (Sharpe 0,77 ; 3/4 classes), elle y\n"
     "échoue désormais (1/4 classe à excès positif, excès moyen +0,12 bps). Les deux\n"
     "formulations restent rapportées dans `cta_gate0.json` pour que la comparaison\n"
     "soit vérifiable."),

    (NOTE, "P2 -- §4 : la piste 2 sort des pistes ouvertes",
     "2. **Corriger la branche 2 du critère de porte.** Telle qu'écrite, elle est\n"
     "   franchissable par un signal constant. Une version qui tienne debout comparerait\n"
     "   le Sharpe du signal à celui de l'acheter-et-garder, pas à zéro.\n"
     "3. **Changer de fenêtre ou de panel.**",
     "2. **Changer de fenêtre ou de panel.**"),

    (NOTE, "P2 -- §4 : chapeau, il ne reste que deux pistes",
     "Le critère d'arrêt ferme ce programme-ci. Trois pistes distinctes existent, et\n"
     "aucune n'est autorisée par le brief actuel — chacune demanderait une décision\n"
     "explicite :",
     "Le critère d'arrêt ferme ce programme-ci. Il restait trois pistes ; la deuxième —\n"
     "« corriger la branche 2 » — est **traitée** par\n"
     "`PATCH_gate0_branche2_et_holm_m2.md` et sort donc de cette liste (cf. §2.4). Les\n"
     "deux autres subsistent, et aucune n'est autorisée par le brief actuel — chacune\n"
     "demanderait une décision explicite :"),
]


def apply_edit(text: str, expected: str, new: str) -> tuple:
    """Retourne (texte, statut). Statuts : 'applied', 'already', 'missing',
    'ambiguous'. Le reperage est exact et doit etre UNIQUE -- un ancrage qui
    matche deux fois est un ancrage trop court, pas une raison de choisir.

    L'ORDRE DES TESTS COMPTE, et c'est un piege verifie sur ce patch : quand
    `new` CONTIENT `expected` (cas d'un ajout de paragraphe apres un bloc
    conserve), l'ancrage reste present apres application. Chercher `expected`
    d'abord ferait donc re-appliquer l'edition a chaque passage. On teste
    l'etat final EN PREMIER : si `new` est deja la, il n'y a rien a faire."""
    if new in text:
        return text, "already"
    n = text.count(expected)
    if n == 1:
        return text.replace(expected, new), "applied"
    return text, "missing" if n == 0 else "ambiguous"


def unified(before: str, after: str, label: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True),
                                        after.splitlines(keepends=True),
                                        fromfile=f"{label} (avant)",
                                        tofile=f"{label} (apres)", n=1))


def replay_gate() -> dict:
    """Non-regression, exigee par le patch : le verdict reste ECHEC sous les trois
    variantes, et la branche 2 rejouee est bien la version reparee."""
    out = {}
    for name in GATE_ARTEFACTS:
        path = Path(__file__).resolve().parent / name
        if not path.exists():
            out[name] = {"status": "absent"}
            continue
        g = json.loads(path.read_text())["gate"]
        out[name] = {
            "status": "rejoue",
            "gate_passes": g["passes"],
            "branch_1_passes": g["branch_1_passes"],
            "branch_2_repaired_passes": g["branch_2"]["passes"],
            "branch_2_repaired_sharpe_excess": g["branch_2"]["sharpe_excess_annualised"],
            "branch_2_repaired_classes_positive": g["branch_2"]["n_classes_positive"],
            "branch_2_original_passes": g["branch_2_original_formulation"]["passes"],
        }
    replayed = [v for v in out.values() if v["status"] == "rejoue"]
    return {"per_variant": out,
            "n_replayed": len(replayed),
            "all_still_fail": bool(replayed) and all(not v["gate_passes"] for v in replayed),
            "conviction_no_longer_passes_branch2": (
                out.get("cta_gate0_conviction.json", {}).get("branch_2_original_passes") is True
                and out.get("cta_gate0_conviction.json", {}).get("branch_2_repaired_passes")
                is False)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="ecrit reellement (defaut : dry-run)")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    by_file, results = {}, []
    for path, label, expected, new in EDITS:
        text = by_file.get(path, path.read_text())
        after, status = apply_edit(text, expected, new)
        by_file[path] = after
        results.append({"file": path.name, "label": label, "status": status,
                        "diff": unified(expected, new, label) if status == "applied" else ""})
        mark = {"applied": "A APPLIQUER", "already": "deja en place",
                "missing": "ANCRAGE INTROUVABLE", "ambiguous": "ANCRAGE AMBIGU"}[status]
        print(f"[{mark:<20}] {path.name} -- {label}")
        if status == "applied":
            for line in results[-1]["diff"].splitlines():
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                    print(f"      {line[:150]}")

    blocking = [r for r in results if r["status"] in ("missing", "ambiguous")]
    replay = replay_gate()
    print(f"\n=== non-regression : {replay['n_replayed']} variantes rejouees ===")
    for name, v in replay["per_variant"].items():
        if v["status"] != "rejoue":
            print(f"  {name:<34} absent")
            continue
        print(f"  {name:<34} porte {'ECHEC' if not v['gate_passes'] else 'PASSE'} | "
              f"branche 2 reparee {'echec' if not v['branch_2_repaired_passes'] else 'passe'} "
              f"(Sharpe exces {v['branch_2_repaired_sharpe_excess']:+.2f}, "
              f"{v['branch_2_repaired_classes_positive']}/4) | "
              f"formulation d'origine {'passait' if v['branch_2_original_passes'] else 'echouait'}")
    print(f"  toutes en echec : {replay['all_still_fail']} | la conviction ne franchit plus "
          f"la branche 2 : {replay['conviction_no_longer_passes_branch2']}")

    payload = {
        "scope": "PATCH_gate0_branche2_et_holm_m2.md -- P1 (Holm m=2), P2 (branche 2 reparee)",
        "applied": bool(args.apply), "edits": results,
        "n_to_apply": sum(1 for r in results if r["status"] == "applied"),
        "n_already": sum(1 for r in results if r["status"] == "already"),
        "blocking": [r["label"] for r in blocking],
        "non_regression": replay,
        "code_change": "cta_gate0.branch2_verdict -- la branche 2 porte sur l'exces vs B&H ; "
                       "la formulation d'origine reste rapportee dans le JSON pour tracabilite",
        "tests": "test_cta_gate0.py -- branche 2 immunisee au signal constant, detection d'un "
                 "vrai exces, non-regression des trois variantes, verrou de calendrier",
    }

    if blocking:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(f"\n{len(blocking)} ancrage(s) introuvable(s) ou ambigu(s) : "
                         f"{[r['label'] for r in blocking]} -- rien n'est ecrit")
    if not replay["all_still_fail"]:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit("\nla non-regression echoue : une variante passe desormais la porte. "
                         "Un patch d'instrument n'a pas le droit de changer un verdict -- "
                         "rien n'est ecrit.")

    if not args.apply:
        payload["note"] = "dry-run : aucune ecriture. Relancer avec --apply."
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n--dry-run : rien ecrit. Plan -> {args.out}")
        return

    stamp = time.strftime("%Y-%m-%dT%H%M%S")
    backups = {}
    for path, after in by_file.items():
        if after == path.read_text():
            continue
        bak = path.with_suffix(f".md.bak_patch_gate0_{stamp}")
        shutil.copy2(path, bak)
        path.write_text(after)
        backups[path.name] = bak.name
        print(f"\n{path.name} patche (sauvegarde {bak.name})")
    payload["backups"] = backups
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
