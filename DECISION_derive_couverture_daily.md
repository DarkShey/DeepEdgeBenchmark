# DÉCISION À PRENDRE — la « dérive de couverture daily » n'est pas une dérive

2026-08-08. Prépare la décision laissée ouverte par `SYNTHESE_finale_programme_nsdiff.md`
(« la seule porte encore ouverte »). **Ce document ne décide pas** : il caractérise,
chiffre les options, recommande, et laisse le choix. Aucun run n'est lancé sur la
base de ce qui suit.

## 1. Ce que 51 cellules sur 90 recouvrent réellement

La synthèse retenait « 51 cellules en alerte sur 90, dont 26 en sous-couverture ».
Le suivi H3 permet maintenant de séparer deux choses que ce comptage confond, et
la séparation change la nature du problème. Le critère est simple et déclaré ici :
une cellule dont **le plein échantillon est lui aussi hors bande** n'a jamais
couvert — ce n'est pas une dérive, c'est un défaut permanent.

| Régime | cellules | en alerte | défaut permanent | dérive réelle |
|---|---:|---:|---:|---:|
| daily | 90 | 51 | **35** | 16 |
| weekly | 90 | 28 | 6 | 22 |

Et parmi les 16 dérives réelles en daily, **12 sont des sur-couvertures** — un
défaut de largeur, pas de risque. Il ne reste donc que **4 vraies dérives en
sous-couverture** :

| Cellule | plein échantillon | fenêtre courante | épisode en cours |
|---|---:|---:|---:|
| `Prophet\|ZN=F\|daily\|W+3` | 0,911 | 0,769 | 3 origines |
| `NsDiff\|BTC-USD\|daily\|W+3` | 0,900 | 0,808 | 8 origines |
| `Prophet\|ZN=F\|daily\|W+2` | 0,933 | 0,808 | 2 origines |
| `NsDiff\|BTC-USD\|daily\|W+2` | 0,922 | 0,846 | 6 origines |

À l'inverse, les défauts permanents sont massifs et anciens :

| Cellule | plein échantillon | fenêtre courante | épisode en cours |
|---|---:|---:|---:|
| `Prophet\|BTC-USD\|daily\|W+3` | **0,289** | 0,000 | 65 origines |
| `Prophet\|BTC-USD\|daily\|W+2` | 0,300 | 0,000 | 65 origines |
| `Prophet\|BTC-USD\|daily\|W+1` | 0,333 | 0,000 | 65 origines |
| `Prophet\|ETH-USD\|daily\|W+3` | 0,556 | 0,538 | 65 origines |

65 origines, c'est **toute la fenêtre disponible** (90 origines, moins les 26 du
démarrage de fenêtre). Ces cellules n'ont jamais couvert, pas une seule semaine.
Les 35 défauts permanents se répartissent sur Naive (10), Prophet (9), SARIMA (9),
LSTM (6), NsDiff (1) — et zéro pour ARIMA-GARCH.

**La reformulation qui s'impose** : il n'y a pas un problème de dérive daily, il y
en a deux, et ils n'appellent pas la même réponse.

- **Problème A — 35 cellules qui n'ont jamais fonctionné.** Un intervalle à 29 %
  de couverture pour une cible de 95 % n'est pas mal calibré, il est faux. C'est
  un défaut de construction, sur la piste daily et presque uniquement là.
- **Problème B — 4 cellules qui dérivent.** Épisodes de 2 à 8 origines, écart de 9
  à 14 points sous la bande. C'est exactement ce que H3 est fait pour attraper, et
  c'est traitable au cas par cas.

## 2. Pourquoi le programme ne l'avait pas vu

Parce qu'aucun verdict rétrospectif ne regarde une fenêtre glissante, et parce que
le duel qui occupait le programme était NsDiff contre ARIMA-GARCH — les deux seuls
modèles quasi exempts du problème (NsDiff : 1 défaut permanent ; ARIMA-GARCH : 0).
Les quatre modèles de référence classiques n'étaient regardés que par le dashboard,
et le dashboard affiche une couverture agrégée sur tout l'échantillon, où
0,289 sur BTC daily se noie dans la moyenne des sept actifs.

## 3. Les options

### Option 1 — Ne rien faire, documenter
Coût : nul. Le suivi H3 tourne désormais en routine (`evaluate-daily.yml`) et les
alertes remontent chaque jour dans le résumé du job. Le défaut reste visible.
**Contre** : la piste daily du dashboard continue d'afficher des intervalles faux
comme s'ils étaient comparables aux autres. Quelqu'un finira par les lire.

### Option 2 — Marquer les cellules non fiables dans le dashboard
Coût : petit (une colonne d'état lue depuis `coverage_monitor.json`, un bandeau).
Le dashboard cesse de présenter comme comparable ce qui ne l'est pas, sans rien
recalculer ni rien effacer.
**Contre** : traite le symptôme. Les intervalles restent faux, ils sont juste
étiquetés.

**Option 2 : implémentée le 2026-08-12, 41 cellules marquées sur la piste `oos`
(35 daily, 6 weekly — rapprochement exact avec les comptes du §1), dont 9
visibles sur le dashboard D7/W1 (horizon W+1).** Marquage dérivé de tracking.db
en lecture seule à chaque génération (`coverage_monitor.permanent_defect_cells`,
bande partagée avec le suivi H3), jamais une liste codée en dur ; pastilles
« ⚠ non fiable » / « ◇ sur-couvert », légende et compteur dans le bandeau de
configuration.

### Option 3 — Diagnostiquer les 35 défauts permanents
Coût : moyen — une journée d'analyse, pas de compute lourd. L'hypothèse la plus
probable est identifiable sans run : la piste daily du dashboard est le **régime B**
(modèle entraîné en quotidien, prévision multi-pas jusqu'à la cible hebdo), et la
calibration sigma EWMA adoptée le 2026-07-31 y est appliquée avec un lag de
résolution conçu pour le régime hebdo. Sur crypto, où la volatilité quotidienne est
un ordre de grandeur au-dessus, une bande construite pour 1 pas et étirée à ~5 pas
sous-couvre mécaniquement. C'est vérifiable en lisant les bandes brutes contre les
bandes corrigées, sans régénérer quoi que ce soit.
**Contre** : ouvre un chantier. Il faudra décider quoi en faire.

### Option 4 — Régénérer la piste daily (phase D)
Coût : ~9 h de compute chiffrées (`regen_plan_r0.json`), plus l'upsert.
**Contre** : le critère de phase D a déjà été évalué et **non rempli**
(`NOTE_nsdiff_regeneration_oos_et_famille3.md` §9). Régénérer ne corrigerait pas
un défaut de construction : les mêmes modèles, sur une grille plus longue,
produiraient les mêmes bandes fausses avec plus de précision. C'est l'option qui
coûte le plus et qui répond le moins.

## 4. Recommandation

**Option 2 puis 3, dans cet ordre, et pas l'option 4.**

L'option 2 arrête le risque de lecture immédiatement, pour un coût faible et sans
rien détruire. L'option 3 s'attaque à la cause, et son hypothèse principale est
testable en lecture seule — donc bon marché et concluante dans un sens comme dans
l'autre. L'option 4 est disqualifiée par son propre critère : la phase D n'est pas
justifiée, et elle ne traiterait pas le problème A de toute façon.

Un point de vocabulaire pour la suite, parce qu'il a induit en erreur : ne plus
appeler ça « la dérive de couverture daily ». C'est **35 cellules fausses et 4
dérives**. Les traiter comme un seul objet conduit à l'option 4, qui est le mauvais
outil pour les deux.

## 5. Ce qui est déjà fait, et n'attend aucune décision

- **H3 est branché en routine** : le job quotidien `evaluate-daily.yml` recalcule
  la couverture glissante sur les deux pistes (`oos` et `oos2020`), commite les
  artefacts et écrit un résumé lisible dans le résumé de job — les dix pires
  cellules en sous-couverture et la durée de l'épisode en cours. Le pas est en
  `continue-on-error` : une alerte de couverture est une information, pas une
  panne, et faire échouer la résolution quotidienne des prédictions pour ça
  garantirait qu'on désactive le pas.
- **Le dashboard est sur l'ensemble 5×200** — vérifié en base
  (`run_id = 20260808-oos-repoint-ensemble`, 2 700 lignes, Cov95 0,9452) et dans le
  bandeau de configuration. La synthèse le déclarait « toujours ouvert » ; c'est
  périmé, le chantier C a été appliqué.

## 6. Ce que la décision ne couvre pas

La piste `oos2020` (grille régénérée, weekly) porte elle aussi 50 cellules en
alerte sur 126, dont 34 en sous-couverture — mais son plein échantillon est plus
souvent dans la bande, donc la part de dérive réelle y est plus élevée. Elle n'est
pas branchée au dashboard et ne présente pas le même risque de lecture. Elle
mériterait sa propre lecture, plus tard, et pas dans la même décision.
