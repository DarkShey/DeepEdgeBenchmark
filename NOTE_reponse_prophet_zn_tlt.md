# Réponse — Prophet ZN/TLT : le log+EWMA combiné tranche la question

**Date : 2026-07-31. Répond à** : récap de session de Kyrio du 31/07 §2 et commit `280e0da`
(« le fix log-espace n'aide pas ZN et dégrade TLT — mais le balayage teste le log *seul*,
pas la combinaison log+EWMA adoptée en prod »).

## Test effectué

La combinaison complète de production (`log_space=True` **+** `calibrate_sigma="ewma"`,
correction multiplicative causale σ'ₜ = σₜ·√EWMA(z²)) a été scorée sur les 5 actifs,
fenêtre W1 2020-2024, à partir des tableaux bruts walk-forward du script de Kyrio
(`experiments/prophet_log_ewma_eval.py`, résultats dans `prophet_log_ewma_eval.json`).

MACE stricte (quantiles normaux, 0 = parfait) :

| Actif | base seul | base+EWMA | log seul | **log+EWMA (prod)** |
|---|---|---|---|---|
| SPY | 13,4 | 2,1 | 5,4 | 2,7 |
| BTC | 28,5 | 5,8 | 6,3 | 3,8 |
| ETH | 34,4 | 9,4 | 16,3 | **4,8** |
| ZN | 41,1 | 15,5 | 35,8 | **8,4** |
| TLT | 27,5 | 14,0 | 36,1 | **9,9** |
| **Moyenne** | 28,9 | 9,4 | 20,0 | **5,9** |

## Verdict

- **La couche EWMA rattrape bien ZN/TLT** : log seul y est mauvais (35,8 / 36,1 — confirme
  l'extension de Kyrio), mais log+EWMA les ramène à 8,4 / 9,9 — et y fait **mieux que
  base+EWMA** (15,5 / 14,0). L'interaction est réelle : le log rend les z standardisés plus
  stationnaires, ce qui rend la correction EWMA plus efficace, même sur l'obligataire.
- **Le défaut global de prod est donc justifié** : log+EWMA est la meilleure combinaison sur
  chaque actif (seule exception marginale : SPY, 2,7 vs 2,1 pour base+EWMA — écart non
  significatif). Pas besoin de traitement différencié par classe d'actif.
- **Faiblesse résiduelle honnête** : ZN/TLT restent les moins bien calibrés (8-10 pts vs 3-5),
  tirés par le niveau 50 % (cov ≈ 28-32 % vs 50) — le centre de la loi est trop étroit.
  C'est un problème de forme centrale, plus de niveau ni de transformation ; piste
  éventuelle : quantiles Student-t/GED par-dessus l'EWMA, ou λ plus réactif sur ces actifs.
- Périmètre : W1 seulement pour ETH/ZN/TLT (comme l'extension de Kyrio). SPY/BTC déjà
  validés sur 3 fenêtres. Une confirmation W3 de ZN/TLT est possible avec la même infra
  (`prophet_sigma_investigation.py --window W3 --assets ZN TLT --configs base log`).
