# BRIEF — Synthèse statistique rigoureuse de la matrice (toutes combinaisons)

## 0. Contexte

La matrice est complète (240/240 : 6 modèles × 5 actifs × {D+1, D+7, W+1, W+2, W+3}). Mais les
chiffres actuels ne sont **pas exploitables comme résultats** : aucun test statistique (que des
estimations ponctuelles), des écarts parfois minuscules probablement dans le bruit, et des
ambiguïtés de définition.

**Objectif** : produire une **synthèse statistique rigoureuse, honnête même si non concluante**,
comparant modèle × actif × fréquence × horizon — un livrable présentable au tuteur. On teste
avant de déclarer quoi que ce soit gagnant, et on couvre **tous les horizons**, pas seulement
D+7 vs W+1.

## 1. Corriger les définitions AVANT toute comparaison

### 1.1 Unifier D+7 = 7 jours calendaires (même jour de semaine) — sans ambiguïté
Il existe aujourd'hui **deux D7** dans le repo :
- **D7 backtest (matrice)** = 5 jours de bourse → tombe **7 jours calendaires** après (lundi →
  lundi suivant). ✅ C'est la définition voulue.
- **D7 business (live)** = 7 jours de bourse → tombe un mercredi. ❌ À supprimer.

**Décision assumée** : **D+7 = cutoff + 7 jours calendaires**, soit le même jour de semaine la
semaine suivante. Si ce jour est férié (marché fermé), on prend le jour de bourse le plus proche.
- Unifier cette définition **partout** (backtest ET live), supprimer la variante 7-jours-de-bourse.
- **Vérifier** que chaque ligne D+7 de la matrice a bien `target_date = cutoff + 7 j calendaires`
  (au jour férié près) ; recomputer `target_date`/`y_true` des lignes non conformes.

### 1.2 Table de correspondance horizon → date-cible
| Horizon | Définition | Cible |
|---|---|---|
| D+1 | cutoff + 1 jour de bourse | lendemain de bourse |
| **D+7** | **cutoff + 7 j calendaires** | **même jour, semaine suivante** |
| W+1 | 1 semaine (résample W-FRI) | vendredi suivant |
| W+2 | 2 semaines | ~10 j de bourse |
| W+3 | 3 semaines | ~15 j de bourse |

Note : D+7 (même jour de semaine) et W+1 (vendredi) **coïncident uniquement quand l'origine est
un vendredi** → la comparaison D+7↔W+1 (§4) se fait sur origines-vendredi.

## 2. Règles de comparabilité (quoi vs quoi)

1. **Jamais de RMSE absolu entre actifs.** Comparaisons **intra-actif**, ou métrique **sans
   échelle** (skill score vs Naive, ou MASE) pour agréger entre actifs.
2. **Inter-modèles sur une même cellule** : juste seulement si **protocole identique** (§3),
   sinon caveat.
3. **Daily vs weekly d'un même modèle** (régime B vs C, ou D+7 vs W+1) : **déjà juste** (protocole
   constant à l'intérieur d'un modèle).

## 3. Neutraliser les asymétries (là où elles gênent l'inter-modèles)

- **Protocole** : TSDiff figé à T0 vs les 5 autres ré-entraînés à chaque origine. Pour les
  classements inter-modèles, unifier sur UN protocole. *Recommandation* : refit-par-origine pour
  tous (plus réaliste, déjà le mode des 5 autres) — Claude Code estime d'abord le coût du refit
  TSDiff ; si trop cher, figer les 5 autres à T0. **Décision à confirmer.**
- **Réglage** : seul TSDiff a eu un balayage d'epochs. *Minimum* : re-sélectionner l'ordre
  ARIMA/SARIMA via AIC sur la série hebdo là où c'est évident ; sinon documenter la limite.
- **Échelle** : déjà gérée (intra-actif) — garder.

## 4. Portée de la synthèse : TOUS les horizons

La synthèse doit couvrir l'ensemble, avec test statistique partout :

- **Par horizon (D+1, D+7, W+1, W+2, W+3), par actif** : classement des 6 modèles avec
  significativité (qui domine, qui est indistinguable).
- **Calibration par horizon** : quels modèles couvrent ~0.95 à quel horizon (le point faible de
  TSDiff-D, la bonne tenue d'ARIMA-GARCH, etc.).
- **Daily vs weekly par modèle** : régime B (daily→weekly) vs régime C (weekly natif) pour chaque
  modèle — la question « faut-il entraîner en weekly ? » posée modèle par modèle.
- **Comparaison phare D+7 vs W+1** (sur origines-vendredi alignées) : « pour 1 semaine, daily ou
  weekly ? », par modèle × actif, testée.
- **Patterns par classe d'actif** : crypto / actions / obligations (ZN=F & TLT corrélés → ne pas
  double-compter).

## 5. Rigueur statistique (partout)

- Réutiliser `experiments/paired_test.py` : bootstrap apparié sur les différences par origine
  (RMSE et/ou CRPS).
- **Aucun verdict sur estimation ponctuelle** — « X > Y » seulement si significatif.
- **Puissance limitée** : 30 origines à cibles chevauchantes → puissance effective ~10-15.
  P-values optimistes → le dire explicitement dans la synthèse.
- Assumer et afficher les résultats **non concluants** : c'est un résultat honnête, pas un échec.

## 6. Plan d'implémentation

1. **Définitions** (§1) : unifier D+7 = 7 j calendaires partout, supprimer la variante
   7-jours-de-bourse, vérifier/recomputer les `target_date`/`y_true` non conformes, produire la
   table horizon→date-cible.
2. **Métriques homogènes** : RMSE, Cov95, DirAcc, CRPS (si échantillons), + skill score sans
   échelle pour l'agrégation inter-actifs.
3. **Décision protocole** (§3) : estimer le coût du refit TSDiff, trancher l'unification, re-produire
   les cellules concernées pour l'inter-modèles.
4. **Tests appariés** sur toutes les comparaisons retenues (par horizon × actif ; B vs C par
   modèle ; D+7 vs W+1).
5. **Rapport de synthèse** (le livrable tuteur) : tableaux par horizon et par classe d'actif, avec
   significativité ; ce qui est établi, ce qui est du bruit, ce qui est bloqué par la puissance.
6. **Vérification** : cohérence des définitions (aucun D+7 hors 7-j-calendaires), alignement des
   dates entre modèles d'une même cellule, pas de doublon, relecture des p-values sous l'angle
   puissance.

## 7. Livrables

- `documentation/matrice_dictionnaire.md` — définitions (dont D+7 unifié), table horizon→date-cible.
- `experiments/matrice_stats.json` — toutes les métriques + p-values par comparaison.
- **Synthèse pour le tuteur** (`documentation/synthese_matrice.md` ou PDF) — la lecture claire et
  honnête : combinaisons gagnantes significatives par actif/classe/horizon, résultats non
  concluants assumés, limites de puissance.

## 8. Ce que ce brief ne fait PAS

- Pas de nouveau modèle ni d'horizon hors des 240 cellules.
- Pas de « modèle gagnant » déclaré sans test apparié.
- Pas de migration DB / dashboard tant que la synthèse n'a pas livré des résultats fiables.
