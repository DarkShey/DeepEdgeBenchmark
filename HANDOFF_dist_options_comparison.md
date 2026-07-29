# Handoff — comparaison options 1/2/3 de calibration des PI (tous modèles)

Statut au 29 juillet 2026, ~16h45. Écrit en cours de route pour permettre une reprise
sur une autre machine / par quelqu'un d'autre sans perdre le travail déjà fait.
**Rien n'est committé sur git pour l'instant** — voir §5, c'est le point le plus
important si tu changes de machine.

## 1. Contexte (pourquoi ce travail existe)

Suite à `alternatives_distributions_pi.pdf` (option 1 testée sur ARIMA-GARCH seul,
voir `comparaison_lois_garch_resultats.pdf`) et à `documentation/
Loi_Gaussienne_vs_Empirique.pdf` §3.4 (tableau des points de contact gaussiens dans
les 5 modèles), la demande était : étendre l'option 1 (loi alternative) à **tous les
modèles**, ajouter l'option 2 (CQR) et l'option 3 (MDN, LSTM seulement), en test
comparatif seulement — **rien de tout ça n'est branché dans models/*.py en
production**, sauf `models/arima_model.py` qui avait déjà reçu `dist=`/`pi_levels`/
`n_crps_samples` lors d'une étape précédente (voir son diff git, déjà en place avant
cette session de handoff).

Objectif final : un rapport comparatif complet (tableaux/graphes) avec un verdict
coût/bénéfice par modèle — **pas encore construit**, c'est la dernière étape (§6).

## 2. Fichiers créés cette session

| Fichier | Rôle |
|---|---|
| `experiments/dist_options_common.py` | Machinerie partagée : fit Student-t/GED (MLE bornée, voir §4), conformalisation CQR, pinball loss, CRPS par échantillonnage. Docstring du module = la méthodologie complète. |
| `experiments/all_models_dist_options.py` | Orchestrateur principal : pour chaque actif, lance le backtest walk-forward **existant et non modifié** de chacun des 5 modèles une seule fois, puis dérive tout le reste (option 1 manuelle, option 2 CQR, + option 1 native pour ARIMA-GARCH) en post-traitement. `--resume` actif par défaut (voir §3). |
| `experiments/lstm_mdn_prototype.py` | Prototype MDN (option 3) pour LSTM seul — processus **volontairement séparé** (conflit d'import TF/statsmodels documenté dans `experiments/weekly_multimodel.py`). Multi-seed (3 par défaut) car l'entraînement NLL-mixture s'est révélé non reproductible même à seed fixe (voir §4). `--resume` actif par défaut. |
| `experiments/all_models_dist_options_results.json` | Résultats incrémentaux du script principal (sauvé après chaque actif). |
| `experiments/lstm_mdn_results.json` | Résultats incrémentaux du prototype MDN (sauvé après chaque actif). |

Aucun fichier de `models/` n'a été modifié cette session (seul `arima_model.py`
l'avait été **avant**, dans une étape précédente déjà actée).

## 3. État d'avancement des deux runs de fond

Les deux tournent en arrière-plan dans **cette session/sandbox** — si tu changes de
machine, ces processus ne te suivront pas, il faudra relancer les commandes
ci-dessous (elles reprendront automatiquement là où c'est resté, grâce à
`--resume`).

- `experiments/all_models_dist_options.py --assets SPY BTC ETH ZN TLT` :
  **SPY terminé** (les 4 autres actifs restent). ~4 min/actif estimées (SARIMA
  ~55s, Prophet ~150s, LSTM ~25s, Naive <1s, ARIMA-GARCH ~20s) → **~16 min
  restantes**.
- `experiments/lstm_mdn_prototype.py --assets SPY BTC ETH ZN TLT --seeds 42 43 44` :
  **SPY terminé** (3 seeds). ~2.5-4 min/actif (×3 seeds) → **~12-15 min
  restantes**.

### Pour vérifier l'avancement à tout moment
```bash
python -c "
import json
for f in ['experiments/all_models_dist_options_results.json','experiments/lstm_mdn_results.json']:
    d = json.load(open(f))
    print(f, '->', list(d['assets'].keys()))
"
```

### Pour reprendre là où c'est resté (même machine ou après un git pull ailleurs)
```bash
# nécessaire sur CETTE machine seulement (Avast intercepte le TLS, cf. §5) --
# à tester d'abord sans sur une autre machine, ne sera probablement pas nécessaire
export CURL_CA_BUNDLE="C:\ProgramData\Avast Software\Avast\wscert.pem"

python experiments/all_models_dist_options.py --assets SPY BTC ETH ZN TLT
python experiments/lstm_mdn_prototype.py --assets SPY BTC ETH ZN TLT --seeds 42 43 44
```
Les deux scripts chargent le JSON existant, sautent les actifs déjà présents,
calculent le reste, et sauvent après chaque actif — aucune perte si interrompu de
nouveau. Les deux peuvent tourner en parallèle sans conflit (fichiers de sortie
différents, aucun état partagé).

## 4. Ce qu'on a déjà appris (pour ne pas repartir de zéro conceptuellement)

- **Sur SPY**, sous loi normale, tous les modèles sur-couvrent au centre (50 %) —
  cohérent avec `Loi_Gaussienne_vs_Empirique.pdf` §3.3. Student-t/GED (fit manuel
  sur fenêtre de calibration) et CQR déplacent la couverture vers la cible pour la
  plupart des modèles, à coût de calcul quasi nul (millisecondes — voir
  `overhead_s` dans le JSON, à comparer à `base_train_time_s`).
- **Bug corrigé** : `scipy.stats.t.fit`/`gennorm.fit` en MLE non contrainte
  divergent vers des dof ~10¹² sur des résidus quasi gaussiens (numériquement
  instable, pas un vrai signal). Remplacé par une MLE 1-paramètre bornée
  ([2.05, 200] pour dof, [0.3, 20] pour beta GED) — voir le commentaire en tête de
  `fit_student_t`/`fit_ged` dans `dist_options_common.py`.
- **MDN (option 3) : instable d'un run à l'autre, même à seed fixe.** Sur SPY, un
  premier run a donné cov_50=66 %, un second (code strictement identique,
  seed=42 identique) a donné cov_50=14 %. Causes : (a) le scaler MinMax est
  entraîné sur le train set — SPY atteint de nouveaux plus-hauts pendant le test,
  donc le réseau doit extrapoler hors de sa plage d'entraînement ; (b) la loss NLL
  de mélange est nettement plus instable à entraîner que le MSE simple du LSTM de
  prod (confirmé : `clipnorm=1.0` + `patience=10` au lieu des réglages de prod
  ont réduit l'écart de RMSE mais pas éliminé la variance inter-seed). D'où le
  passage à 3 seeds/actif avec agrégation mean±std (`kpi_agg` dans le JSON) — lire
  le `std` avant de conclure quoi que ce soit sur le MDN, un seul run est
  trompeur. Coût d'entraînement mesuré : ~45-90s/run (×3 seeds), contre ~25-28s
  pour le LSTM de prod — et ça, c'est AVANT de compter le temps qu'il a fallu pour
  stabiliser l'entraînement (qui est lui-même un coût réel à documenter dans le
  rapport final).

## 5. ⚠️ Rien n'est committé — le point le plus important pour une autre machine

`git status --short` à la racine montre tous les fichiers du §2 en `??` (untracked).
Un `git pull` sur une autre machine ne les fera PAS apparaître tant qu'ils ne sont
pas committés (et poussés sur `origin` = `github.com/DarkShey/DeepEdgeBenchmark`).

Je n'ai pas committé de moi-même (règle du projet : jamais sans demande explicite).
Si tu veux que je committe/pousse avant de partir, dis-le-moi — je le ferai en
scopant strictement aux fichiers de cette session (§2 + ce fichier), sans toucher
aux autres fichiers déjà modifiés/untracked qui traînaient avant cette conversation
(`benchmarks/multi_horizon.py`, `experiments/weekly_multimodel.py`,
`calibration/pi_recalibration.py`, `experiments/ensemble_weights.py`,
`validation/tracking_export.csv`, etc. — visiblement du travail en cours d'une
session précédente, pas à moi d'en décider).

À défaut d'un commit, il faudrait copier manuellement les fichiers du §2 (+ ce
fichier) vers l'autre machine pour reprendre sans perte.

## 6. Ce qu'il reste à faire une fois les deux runs terminés

1. Vérifier les résultats complets (5 actifs × 5 modèles × options) pour cohérence,
   comme fait ici pour SPY.
2. Construire un rapport comparatif complet — tableaux + graphiques (coverage
   50/80/95 vs cible, largeur, CRPS/pinball, coût de calcul par option et par
   modèle) — avec un verdict "rentable ou non" par modèle/option, demandé
   explicitement par l'utilisateur. Pas encore commencé. Le skill `dataviz` de
   Claude Code a déjà été chargé dans la session d'origine pour ce travail si tu
   continues avec Claude Code.
3. Décider avec l'utilisateur du format de sortie (artifact HTML interactif
   probablement, vu le volume — cf. discussion précédente dans la conversation).
