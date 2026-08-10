"""
real_fees.py -- chantier 1.1 du BRIEF "NsDiff : rouvrir la question economique
par le rapport edge/frais" : remplacer les trois niveaux FORFAITAIRES du
chantier B par des frais realistes PAR INSTRUMENT.

POURQUOI. Le chantier B a mesure un edge de +2 a +5 points de base par origine et
l'a compare a une grille de couts par CLASSE d'actif (1/5/10 bps unidirectionnels
pour actions et obligations, 10/30/60 pour la crypto). Cette grille repond a la
question « l'edge survit-il a des frais moyens ? ». Elle ne repond pas a la vraie
question de production : « existe-t-il un VEHICULE D'EXECUTION ou les frais sont
sous l'edge ? » -- car on n'est pas oblige de prendre l'exposition actions par un
ETF au comptant si un future la donne cinq fois moins cher.

CONVENTION D'UNITE, le piege a eviter. Les chiffres ci-dessous sont des frais
ALLER-RETOUR, tout compris (commission + traversee du spread a l'entree et a la
sortie). Le moteur `econ_backtest.sleeve_pnl` attend, lui, un cout
UNIDIRECTIONNEL qu'il double lui-meme. `one_way_bps()` fait la conversion en un
seul endroit -- confondre les deux doublerait silencieusement toute la grille et
inverserait la conclusion du chantier.

STATUT DES CHIFFRES, a citer avec tout resultat qui en depend. Ce sont des
HYPOTHESES DECLAREES, aux ordres de grandeur fournis par le brief, pas un releve
de bordereaux de courtage. Elles sont fixees avant les runs et ne sont jamais
ajustees apres lecture des resultats. Trois niveaux par instrument (bas / central
/ haut) encadrent l'incertitude sur ces hypotheses ; le niveau CENTRAL porte la
decision, les deux autres disent si elle tient. Remplacer ces chiffres par une
grille tarifaire reelle est un travail de dix minutes qui ne touche que ce
fichier -- c'est pour cela qu'ils y sont tous, et nulle part ailleurs.

RAISONNEMENT DERRIERE CHAQUE ORDRE DE GRANDEUR :
  * FUTURES (ES pour l'exposition actions, ZN pour les taux) -- les contrats les
    plus liquides du monde : spread d'un tick sur un notionnel de plusieurs
    dizaines de milliers de dollars, commission de quelques dollars par contrat.
    L'aller-retour tout compris tombe dans 1-2 bps. C'est le vehicule le moins
    cher pour ces deux expositions, et de loin.
  * ETF AU COMPTANT (SPY, TLT) -- spread d'un cent sur des prix a trois chiffres,
    plus une commission qui depend du courtier. 2-5 bps l'aller-retour.
  * CRYPTO AU COMPTANT (BTC, ETH) -- commissions de plateforme d'un ordre de
    grandeur au-dessus des marches organises, spread plus large et variable.
    10-60 bps l'aller-retour ; la borne haute est conservee telle quelle, c'est
    elle qui condamnait la crypto au chantier B.

DEUX ROUTES POUR L'EXPOSITION ACTIONS. `SPY` est evalue sous DEUX vehicules :
l'ETF (SPY lui-meme) et le future (ES). Les previsions sont identiques -- c'est le
meme sous-jacent, les memes bornes, les memes signaux -- seuls les frais changent.
C'est exactement le levier que le brief demande de tester : le meme edge, execute
autrement. TLT/ZN=F/BTC/ETH n'ont qu'une route chacun (l'exposition taux passe
deja par un future avec ZN=F, et TLT est un ETF par nature).
"""

# Frais ALLER-RETOUR, en points de base du notionnel, tout compris.
# Cle : identifiant de VEHICULE. `asset` = l'actif dont les previsions sont
# utilisees ; deux vehicules peuvent partager le meme actif (SPY / SPY-ES).
INSTRUMENTS = {
    "SPY-ETF": {
        "asset": "SPY", "vehicle": "ETF au comptant", "class": "index",
        "round_trip_bps": {"bas": 2.0, "central": 3.5, "haut": 5.0},
        "rationale": "spread d'un cent sur un prix a trois chiffres + commission courtier",
    },
    "SPY-ES": {
        "asset": "SPY", "vehicle": "future E-mini S&P 500 (ES)", "class": "index",
        "round_trip_bps": {"bas": 1.0, "central": 1.5, "haut": 2.0},
        "rationale": "un des contrats les plus liquides au monde : un tick de spread sur "
                     "un notionnel de plusieurs dizaines de milliers de dollars",
        "caveat": "memes previsions que SPY-ETF (meme sous-jacent) ; seul le cout change. "
                  "Le roulement trimestriel EST modelise depuis H2 (`roll_cost_bps`) ; la "
                  "base ES/SPY ne l'est pas, et la raison est declaree au bloc H2 -- dans "
                  "une comparaison financee, le portage du future est compense par "
                  "l'interet sur le capital non immobilise.",
    },
    "TLT-ETF": {
        "asset": "TLT", "vehicle": "ETF au comptant", "class": "bond",
        "round_trip_bps": {"bas": 2.0, "central": 3.5, "haut": 5.0},
        "rationale": "meme structure de cout que SPY-ETF",
    },
    "ZN-FUT": {
        "asset": "ZN=F", "vehicle": "future 10-Year T-Note (ZN)", "class": "bond",
        "round_trip_bps": {"bas": 1.0, "central": 1.5, "haut": 2.0},
        "rationale": "future de taux le plus liquide ; l'actif du panier EST deja ce future",
    },
    "BTC-SPOT": {
        "asset": "BTC-USD", "vehicle": "crypto au comptant", "class": "crypto",
        "round_trip_bps": {"bas": 10.0, "central": 30.0, "haut": 60.0},
        "rationale": "commissions de plateforme un ordre de grandeur au-dessus des marches "
                     "organises, spread plus large et variable",
    },
    "ETH-SPOT": {
        "asset": "ETH-USD", "vehicle": "crypto au comptant", "class": "crypto",
        "round_trip_bps": {"bas": 10.0, "central": 30.0, "haut": 60.0},
        "rationale": "idem BTC",
    },
    # Ajoutes au chantier A du brief extension/puissance. Ce sont les deux seuls
    # candidats de `power_analysis.CANDIDATE_ASSETS` a apporter une exposition
    # REELLEMENT nouvelle : QQQ, EFA et IEF correlent 0,8-0,9 avec un actif deja
    # present et ajouteraient des lignes sans information independante.
    "GLD-ETF": {
        "asset": "GLD", "vehicle": "ETF au comptant (or)", "class": "commodity",
        "round_trip_bps": {"bas": 2.5, "central": 4.0, "haut": 6.0},
        "rationale": "ETF matiere premiere tres liquide, spread d'un cent sur un prix a trois "
                     "chiffres ; legerement plus cher que SPY, moins qu'un ETF sectoriel",
    },
    "USO-ETF": {
        "asset": "USO", "vehicle": "ETF au comptant (petrole)", "class": "commodity",
        "round_trip_bps": {"bas": 3.0, "central": 5.0, "haut": 8.0},
        "rationale": "ETF matiere premiere, spread plus large que GLD (sous-jacent a terme, "
                     "roulement mensuel repercuté dans la liquidite)",
        "caveat": "USO porte un cout de roulement structurel (contango) qui n'est PAS un frais "
                  "de transaction et n'est donc pas modelise ici -- il affecte le rendement du "
                  "sous-jacent, pas l'ecart entre deux modeles qui le previennent tous les deux.",
    },
    # ── Univers 0-bis (BRIEF_re_porte_cta_corrige_univers_deita.md) ─────────
    # Ordres de grandeur DECLARES par famille, comme l'exige le brief, et fixes
    # avant tout run. Aucun n'est ajuste sur un resultat.
    #   ETF actions       2-5 bps aller-retour, majores pour les moins liquides
    #   futures           1-2 bps + roulement (H2)
    #   crypto au comptant 10-60 bps
    "QQQ-ETF": {
        "asset": "QQQ", "vehicle": "ETF au comptant (Nasdaq-100)", "class": "index",
        "round_trip_bps": {"bas": 2.0, "central": 3.5, "haut": 5.0},
        "rationale": "liquidite comparable a SPY : meme structure de cout",
    },
    "IWM-ETF": {
        "asset": "IWM", "vehicle": "ETF au comptant (Russell 2000)", "class": "index",
        "round_trip_bps": {"bas": 2.5, "central": 4.0, "haut": 6.0},
        "rationale": "petites capitalisations : spread un cran plus large que SPY/QQQ",
    },
    "EFA-ETF": {
        "asset": "EFA", "vehicle": "ETF au comptant (actions developpees hors AmNord)",
        "class": "index",
        "round_trip_bps": {"bas": 3.0, "central": 5.0, "haut": 7.0},
        "rationale": "sous-jacents sur fuseaux horaires decales : spread plus large et "
                     "prime/decote de VNI plus frequente",
    },
    "EEM-ETF": {
        "asset": "EEM", "vehicle": "ETF au comptant (marches emergents)", "class": "index",
        "round_trip_bps": {"bas": 3.0, "central": 5.5, "haut": 8.0},
        "rationale": "idem EFA, avec des marches sous-jacents moins liquides",
    },
    "SI-FUT": {
        "asset": "SI=F", "vehicle": "future argent (SI)", "class": "commodity",
        "round_trip_bps": {"bas": 1.5, "central": 2.5, "haut": 4.0},
        "rationale": "future liquide mais nettement moins que ES/ZN : un tick vaut plus cher "
                     "en relatif",
    },
    "HG-FUT": {
        "asset": "HG=F", "vehicle": "future cuivre (HG)", "class": "commodity",
        "round_trip_bps": {"bas": 1.5, "central": 3.0, "haut": 5.0},
        "rationale": "metal industriel, carnet plus mince que les metaux precieux",
    },
    "ZC-FUT": {
        "asset": "ZC=F", "vehicle": "future mais (ZC)", "class": "commodity",
        "round_trip_bps": {"bas": 2.0, "central": 4.0, "haut": 6.5},
        "rationale": "agricole : spread plus large, liquidite concentree sur quelques echeances",
        "caveat": "le mais roule ~5 fois par an et non 4 ; le modele de roulement H2 le traite "
                  "au rythme trimestriel, ce qui SOUS-estime son cout de roulement. Declare.",
    },
    "SOL-SPOT": {
        "asset": "SOL-USD", "vehicle": "crypto au comptant", "class": "crypto",
        "round_trip_bps": {"bas": 15.0, "central": 40.0, "haut": 80.0},
        "rationale": "hors des deux plus grosses capitalisations : spread et frais de "
                     "plateforme au-dessus de BTC/ETH",
    },
    "BNB-SPOT": {
        "asset": "BNB-USD", "vehicle": "crypto au comptant", "class": "crypto",
        "round_trip_bps": {"bas": 15.0, "central": 40.0, "haut": 80.0},
        "rationale": "idem SOL ; liquidite concentree sur une plateforme",
    },
    "LINK-SPOT": {
        "asset": "LINK-USD", "vehicle": "crypto au comptant", "class": "crypto",
        "round_trip_bps": {"bas": 20.0, "central": 50.0, "haut": 100.0},
        "rationale": "capitalisation plus faible encore : la borne haute est volontairement "
                     "punitive",
    },
    "VXX-ETN": {
        "asset": "VXX", "vehicle": "ETN de volatilite court terme", "class": "index",
        "round_trip_bps": {"bas": 5.0, "central": 10.0, "haut": 20.0},
        "rationale": "ETN, pas ETF : spread plus large et risque emetteur",
        "caveat": "VXX porte une decroissance structurelle de roulement (contango de la courbe "
                  "VIX) qui n'est PAS un frais de transaction et n'est pas modelisee ici. "
                  "L'actif est declare hors des quatre classes de la porte 0-bis.",
    },
}

LEVELS = ("bas", "central", "haut")
DECISION_LEVEL = "central"

# ── H2 : realisme d'execution des futures ───────────────────────────────────
# Le chantier precedent declarait « la base ES/SPY et le roulement trimestriel ne
# sont PAS modelises -- hypothese favorable au future ». H2 leve la moitie de
# cette reserve : le ROULEMENT, qui est un vrai cout de transaction, est modelise
# ci-dessous. L'autre moitie, la BASE, ne l'est pas -- et ce n'est pas un oubli :
#
#   La base d'un future vaut, a l'equilibre, (taux sans risque - rendement du
#   sous-jacent) x duree. Un long future subit donc un portage negatif que l'ETF
#   ne subit pas. Mais le future n'immobilise que sa marge : le capital non
#   deploye rapporte le taux sans risque, ce qui compense ce portage au premier
#   ordre. Dans une comparaison FINANCEE -- la seule que ce backtest fasse, sans
#   levier, |w| <= 1 -- les deux s'annulent. Mettre un chiffre de portage sans
#   creediter l'interet sur le cash penaliserait le future d'un cout qu'il ne paie
#   pas. La reserve qui subsiste est donc etroite et declaree : la base a court
#   terme n'est pas exactement a l'equilibre, et cet ecart residuel n'est pas
#   mesure ici.
#
# COUT D'UN ROULEMENT. Fermer le contrat proche et ouvrir le suivant traverse un
# spread calendaire. Hypothese declaree, conservatrice : un roulement coute un
# aller-retour complet de l'instrument. Rien ne dit que le spread calendaire soit
# plus large que le spread outright ; l'inverse est meme frequent sur ES et ZN.
#
# FREQUENCE VUE PAR UNE ORIGINE. Les contrats roulent tous les trimestres, soit
# ~13 semaines. Un sleeve tenu h semaines traverse une echeance avec probabilite
# h/13 (h <= 3 ici, donc jamais plus d'une). Le cout ATTENDU par origine est donc
# (h/13) x cout_d_un_roulement -- une esperance, pas un tirage : le backtest ne
# connait pas le calendrier d'echeances reel des contrats.
ROLLED_INSTRUMENTS = {
    "SPY-ES": {"rolls_per_year": 4, "weeks_per_period": 13.0},
    "ZN-FUT": {"rolls_per_year": 4, "weeks_per_period": 13.0},
    # Ajoutes au 0-bis. Rythme trimestriel declare pour les trois ; pour ZC=F
    # (mais, ~5 roulements par an) c'est une SOUS-estimation, signalee dans son
    # entree ci-dessus plutot que corrigee par un chiffre invente.
    "SI-FUT": {"rolls_per_year": 4, "weeks_per_period": 13.0},
    "HG-FUT": {"rolls_per_year": 4, "weeks_per_period": 13.0},
    "ZC-FUT": {"rolls_per_year": 4, "weeks_per_period": 13.0},
}


def roll_cost_bps(instrument: str, horizon_weeks: float,
                  level: str = DECISION_LEVEL) -> float:
    """Cout de roulement ATTENDU sur la duree de detention, en bps aller-retour.
    Vaut 0 pour tout instrument qui ne roule pas (ETF, crypto au comptant)."""
    spec = ROLLED_INSTRUMENTS.get(instrument)
    if spec is None:
        return 0.0
    return round_trip_bps(instrument, level) * float(horizon_weeks) / spec["weeks_per_period"]


def total_round_trip_bps(instrument: str, horizon_weeks: float,
                         level: str = DECISION_LEVEL) -> float:
    """Aller-retour tout compris, roulement inclus. C'est ce chiffre qu'il faut
    comparer a l'edge quand l'instrument est un future."""
    return round_trip_bps(instrument, level) + roll_cost_bps(instrument, horizon_weeks, level)


def one_way_total_bps(instrument: str, horizon_weeks: float,
                      level: str = DECISION_LEVEL) -> float:
    """Version unidirectionnelle de `total_round_trip_bps`, pour `econ_backtest`
    qui double lui-meme. Meme point de conversion unique que `one_way_bps`."""
    return total_round_trip_bps(instrument, horizon_weeks, level) / 2.0

# Instruments dont les frais sont sous l'edge mesure au chantier B (+2 a +5 bps
# par origine, aller-retour). Calcule, pas ecrit a la main -- si la grille change,
# cette liste change avec elle.
EDGE_REFERENCE_BPS = 5.0     # borne HAUTE de l'edge mesure, la plus favorable


def round_trip_bps(instrument: str, level: str = DECISION_LEVEL) -> float:
    return INSTRUMENTS[instrument]["round_trip_bps"][level]


def one_way_bps(instrument: str, level: str = DECISION_LEVEL) -> float:
    """Cout UNIDIRECTIONNEL attendu par `econ_backtest.sleeve_pnl`, qui le double
    lui-meme pour former l'aller-retour. Point de conversion UNIQUE du chantier."""
    return round_trip_bps(instrument, level) / 2.0


def instruments_for_asset(asset: str) -> list:
    return [k for k, v in INSTRUMENTS.items() if v["asset"] == asset]


def viable_instruments(level: str = DECISION_LEVEL, edge_bps: float = EDGE_REFERENCE_BPS) -> list:
    """Instruments dont l'aller-retour est sous l'edge de reference. C'est le
    filtre AVANT tout backtest : si aucun instrument ne passe, le chantier 1 est
    clos avant d'avoir tourne."""
    return sorted(k for k in INSTRUMENTS if round_trip_bps(k, level) < edge_bps)


def summary_table() -> list:
    """Lignes pretes a imprimer / a poser dans la note."""
    out = []
    for key, spec in INSTRUMENTS.items():
        rt = spec["round_trip_bps"]
        out.append({
            "instrument": key, "actif": spec["asset"], "vehicule": spec["vehicle"],
            "classe": spec["class"],
            "aller_retour_bps": {lvl: rt[lvl] for lvl in LEVELS},
            "sous_edge_5bps_au_central": bool(rt[DECISION_LEVEL] < EDGE_REFERENCE_BPS),
            "justification": spec["rationale"],
            **({"reserve": spec["caveat"]} if "caveat" in spec else {}),
        })
    return out
