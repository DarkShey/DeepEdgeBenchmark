"""
multiple_testing.py -- politique de TESTS MULTIPLES du programme, mise dans la
machinerie partagee au lieu de rester une note de bas de page (chantier A3 du
BRIEF "NsDiff : valeur economique, re-cadrage 200 tirages, cadrage monthly").

Le probleme, deja declare mais jamais outille : les notes de consolidation
comptent 12 tests pooles (4 groupes x 3 horizons) et jusqu'a 30 cellules par
match. A alpha=0.05 sans correction, on attend ~0.6 faux positif par famille
de 12 et ~1.5 par famille de 30 -- soit exactement l'ordre de grandeur des
"cases isolees a p~0.04" que les deux notes precedentes demandent de lire avec
prudence (`NOTE_nsdiff_consolidation_daily_vs_weekly.md` §9,
`NOTE_duel_nsdiff_vs_tsdiff_budget_egal.md` §0.1).

CONVENTION RETENUE : **Holm-Bonferroni** (Holm 1979), pas Bonferroni simple ni
Benjamini-Hochberg.
  * vs Bonferroni : Holm le domine uniformement (il rejette tout ce que
    Bonferroni rejette, et parfois plus) au meme cout d'hypothese -- aucune
    raison de preferer Bonferroni ;
  * vs Benjamini-Hochberg : BH controle le FDR, pas le FWER. Ici la question
    posee est decisionnelle et binaire ("ce modele bat-il la baseline ?"),
    une seule case fausse suffit a fonder une mauvaise decision de production.
    On controle donc le taux d'erreur PAR FAMILLE. C'est le choix conservateur,
    et c'est celui que §0.1 du duel annoncait deja ("une correction de Holm sur
    les 12 tests pooles donnerait un seuil effectif de 0.05/12 ~ 0.004").
  * Holm ne suppose AUCUNE structure de dependance entre les p-values -- ce qui
    compte ici, ou les tests d'une meme famille partagent les memes origines et
    sont donc fortement correles.

DEFINITION D'UNE FAMILLE, declaree a priori et appliquee partout : une famille
= l'ensemble des tests pooles d'un meme MATCH et d'une meme METRIQUE. Les tests
par cellule ne sont PAS corriges (ils sont exploratoires par construction et
declares comme tels) ; leur nombre de rejets bruts est rapporte tel quel, mais
aucune conclusion du programme ne repose sur une cellule isolee.

Ce module ne fait QUE la correction : il ne calcule aucune p-value (celles-ci
viennent toutes du bootstrap par blocs de `paired_test`), il n'en invente
aucune, et il ne modifie aucun verdict brut -- il en AJOUTE un second, corrige,
a cote. Les deux sont rapportes ensemble : cacher le brut empecherait de voir
combien la correction coute.
"""

from typing import Sequence

ALPHA = 0.05


def holm_bonferroni(p_values: Sequence[float], alpha: float = ALPHA) -> dict:
    """Procedure descendante (step-down) de Holm sur une famille de m tests.

    Trie les p par ordre croissant, les compare a alpha/(m-i) pour i=0..m-1, et
    s'arrete au PREMIER echec : tout ce qui suit est non rejete, meme si sa p
    brute passe son propre seuil. C'est ce qui distingue Holm d'une simple
    comparaison p_i < alpha/(m-i) test par test.

    Renvoie aussi les `p_adjusted` (p corrigees, monotonisees), qui permettent
    de comparer directement a alpha sans refaire la procedure -- p_adj_(i) =
    max sur j<=i de min(1, (m-j) * p_(j)).

    `p_values` peut contenir des None (test non rendu : donnees insuffisantes).
    Ils sont EXCLUS de la famille -- corriger pour des tests qui n'ont pas eu
    lieu gonflerait m artificiellement -- et ressortent avec reject=None.
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    if m == 0:
        return {"m": 0, "alpha": float(alpha), "reject": [None] * len(p_values),
                "p_adjusted": [None] * len(p_values), "thresholds": [None] * len(p_values),
                "n_rejected": 0, "smallest_threshold": None}

    order = sorted(indexed, key=lambda t: t[1])
    reject = [None] * len(p_values)
    p_adj = [None] * len(p_values)
    thresholds = [None] * len(p_values)

    still_rejecting = True
    running_max = 0.0
    for rank, (idx, p) in enumerate(order):
        thr = alpha / (m - rank)
        thresholds[idx] = thr
        if still_rejecting and p >= thr:
            still_rejecting = False      # tout le reste de la sequence tombe avec lui
        reject[idx] = bool(still_rejecting)
        running_max = max(running_max, min(1.0, (m - rank) * p))
        p_adj[idx] = running_max

    return {
        "m": m, "alpha": float(alpha),
        "reject": reject, "p_adjusted": p_adj, "thresholds": thresholds,
        "n_rejected": sum(1 for r in reject if r),
        "smallest_threshold": alpha / m,
    }


def correct_family(tests: dict, p_key: str = "p_value", verdict_key: str = "verdict",
                   alpha: float = ALPHA, null_verdict: str = "indistinguishable") -> dict:
    """Applique Holm a une famille donnee sous forme de {nom: resultat_de_test}.

    Chaque valeur doit etre un dict portant au moins `p_key`. On y AJOUTE :
        holm_reject      -- rejete apres correction ? (None si non teste)
        holm_p_adjusted  -- p corrigee
        holm_threshold   -- seuil qui s'appliquait a ce rang
        holm_verdict     -- le verdict brut s'il survit, `null_verdict` sinon

    Le verdict brut n'est jamais ecrase : les deux coexistent, et toute lecture
    du programme doit citer lequel des deux elle utilise. Les tests sans
    `p_key` (status != 'tested') traversent sans etre comptes dans m.
    """
    names = list(tests)
    p_values = [tests[n].get(p_key) if isinstance(tests[n], dict) else None for n in names]
    holm = holm_bonferroni(p_values, alpha=alpha)

    out = {}
    for i, name in enumerate(names):
        res = tests[name]
        if not isinstance(res, dict):
            out[name] = res
            continue
        rejected = holm["reject"][i]
        out[name] = {
            **res,
            "holm_reject": rejected,
            "holm_p_adjusted": holm["p_adjusted"][i],
            "holm_threshold": holm["thresholds"][i],
            "holm_verdict": (res.get(verdict_key) if rejected else null_verdict) if rejected is not None else None,
        }
    return {"family": out, "holm": {k: v for k, v in holm.items()
                                    if k in ("m", "alpha", "n_rejected", "smallest_threshold")}}


def family_summary(corrected: dict, verdict_key: str = "verdict",
                   null_verdict: str = "indistinguishable") -> dict:
    """Ce que la correction a coute, en une ligne : combien de rejets bruts,
    combien survivent, et lesquels tombent. Rapporte systematiquement a cote du
    tableau des tests -- une correction dont on ne montre pas le cout invite a
    ne citer que le tableau qui arrange."""
    fam = corrected["family"]
    raw = [n for n, r in fam.items()
           if isinstance(r, dict) and r.get(verdict_key) not in (None, null_verdict)]
    kept = [n for n in raw if fam[n].get("holm_reject")]
    return {
        "m": corrected["holm"]["m"], "alpha": corrected["holm"]["alpha"],
        "smallest_threshold": corrected["holm"]["smallest_threshold"],
        "n_significant_raw": len(raw), "n_significant_holm": len(kept),
        "lost_to_correction": sorted(set(raw) - set(kept)),
        "survivors": sorted(kept),
    }
