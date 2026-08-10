"""
nsdiff_production_spec.py -- la CONFIGURATION CANDIDATE PRODUCTION de NsDiff,
ecrite comme specification executable et non comme paragraphe de note
(chantier A2 du BRIEF "NsDiff : valeur economique, re-cadrage 200 tirages,
cadrage monthly").

POURQUOI CE FICHIER EXISTE. Tous les verdicts du programme (daily vs weekly,
vs GARCH, vs TSDiff) sont enonces pour "un run a graine tiree au hasard" --
c'est la convention de pooling declaree dans `nsdiff_v2_data`. Or ce n'est PAS
ce qu'on deploierait : la meilleure configuration mesuree est l'ensemble des 5
graines (tache 6 de la consolidation), qui ameliore significativement le
Winkler sur 7 cellules daily / 15 pour un cout de calcul nul. Tant que la spec
de cette configuration vit dans une note, chaque script la reimplemente et rien
ne garantit qu'ils la reimplementent pareil. Ici elle est appelable.

LA SPEC, en toutes lettres :

  1. 5 graines (42, 43, 44, 45, 46), chacune entrainee train-once-forward avec
     le budget plat NSDIFF_EPOCHS_W = 40 epoques, seq_len=30, k_denoise=20 ;
  2. 200 tirages par graine, soit 1000 tirages au total ;
  3. CONCATENATION des 5 nuages -- pas moyenne des bornes. Concatener revient a
     lire les quantiles du MELANGE des 5 lois predictives : la bande s'elargit
     quand les graines sont en desaccord, ce qui est exactement l'effet
     recherche cote calibration. Moyenner les bornes aurait au contraire lisse
     ce desaccord et rendu l'ensemble faussement confiant ;
  4. point = MOYENNE du nuage concatene ; bandes = quantiles empiriques
     2.5 % / 97.5 % du nuage concatene. Meme lecture que pour un run simple,
     aucune formule nouvelle ;
  5. aucun refit : les 5 nuages sont produits par le meme passage forward.

BIAIS DE QUANTILE RESIDUEL, declare : lus sur 1000 tirages, les quantiles
2.5/97.5 estiment en esperance le niveau 0.9740, soit une couverture bilaterale
reelle de 94.81 % pour une etiquette 95 % -- 0.19 point de deficit mecanique,
contre 0.94 point a 200 tirages et 3.73 a 50. C'est le meme regime que Prophet
(1000 tirages internes) et sous la resolution de tout ce que le programme
mesure. Voir `oos_reference_audit.expected_quantile_level`.

CE QUE LA SPEC NE DIT PAS, volontairement : la cadence de refit. Elle est
etudiee separement (`nsdiff_refit_cadence.py`, chantier A3) parce que c'est une
question de protocole d'exploitation, pas de lecture du nuage.
"""

import numpy as np

SEEDS = (42, 43, 44, 45, 46)
N_SAMPLES_PER_SEED = 200
EPOCHS = 40                  # weekly_nsdiff_production.NSDIFF_EPOCHS_W, budget plat declare
SEQ_LEN = 30
K_DENOISE = 20
Q_LOW, Q_HIGH = 0.025, 0.975

PRODUCTION_SPEC = {
    "name": "NsDiff-ensemble-5x200",
    "seeds": list(SEEDS),
    "n_samples_per_seed": N_SAMPLES_PER_SEED,
    "n_samples_total": len(SEEDS) * N_SAMPLES_PER_SEED,
    "epochs": EPOCHS, "seq_len": SEQ_LEN, "k_denoise": K_DENOISE,
    "aggregation": "concatenation des nuages (melange des lois predictives), "
                   "PAS moyenne des bornes",
    "point": "moyenne du nuage concatene",
    "interval": f"quantiles empiriques {Q_LOW} / {Q_HIGH} du nuage concatene",
    "refit": "aucun -- train-once-forward, les 5 nuages viennent du meme passage forward",
    "residual_quantile_bias": "couverture bilaterale reelle ~94.81 % pour une etiquette 95 % "
                              "(1000 tirages) -- 0.19 point, meme regime que Prophet",
}


def aggregate_cloud(clouds) -> np.ndarray:
    """Concatene les nuages des graines en un seul. `clouds` : iterable de
    tableaux 1-D (un par graine) ou tableau 2-D (n_seeds, n_samples).

    Concatener, pas empiler puis moyenner : le resultat EST le melange des lois
    predictives des graines. Toute la spec tient dans ce choix."""
    arr = np.asarray(clouds, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"clouds doit etre 1-D ou 2-D, recu ndim={arr.ndim}")
    return arr.ravel()


def read_forecast(cloud) -> dict:
    """Point et bandes lus sur un nuage, avec LA formule du repo (celle de
    `oos_nsdiff_daily_weekly.generate_nsdiff_asset`) -- reproduite ici a
    l'identique pour qu'un seul endroit la porte."""
    c = np.asarray(cloud, dtype=float)
    if c.size == 0:
        raise ValueError("read_forecast: nuage vide")
    lo, hi = (float(q) for q in np.quantile(c, [Q_LOW, Q_HIGH]))
    return {"y_pred": float(c.mean()), "y_lower": lo, "y_upper": hi, "n_samples": int(c.size)}


def production_forecast(clouds) -> dict:
    """La spec, de bout en bout : concatener puis lire. C'est cette fonction que
    tout consommateur de la configuration production doit appeler."""
    out = read_forecast(aggregate_cloud(clouds))
    out["n_seeds"] = 1 if np.asarray(clouds, dtype=float).ndim == 1 else len(clouds)
    return out


def validate_clouds(clouds, expect_seeds: int = len(SEEDS),
                    expect_samples: int = N_SAMPLES_PER_SEED) -> None:
    """Garde-fou explicite : refuse de produire une prevision "production" a
    partir d'un materiel qui n'est pas celui de la spec. Silencieusement
    accepter 3 graines ou 50 tirages produirait un chiffre etiquete production
    qui n'en est pas un."""
    arr = np.asarray(clouds, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"la spec attend {expect_seeds} nuages (2-D), recu ndim={arr.ndim}")
    n_seeds, n_samples = arr.shape
    if n_seeds != expect_seeds:
        raise ValueError(f"la spec attend {expect_seeds} graines, recu {n_seeds}")
    if n_samples != expect_samples:
        raise ValueError(f"la spec attend {expect_samples} tirages par graine, recu {n_samples}")
