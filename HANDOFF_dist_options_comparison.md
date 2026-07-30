# Handoff — comparaison options 1/2/3 de calibration des PI (tous modèles)

**Statut : terminé.** Mis à jour le 29 juillet 2026, ~18h00. Les deux runs de fond
(5 actifs × 5 modèles × options, + MDN multi-seed) sont complets, le résumé agrégé
et le rapport final sont générés. Gardé pour référence si quelqu'un doit
comprendre/régénérer/étendre ce travail plus tard.

## 1. Contexte (pourquoi ce travail existe)

Suite à `alternatives_distributions_pi.pdf` (option 1 testée sur ARIMA-GARCH seul,
voir `comparaison_lois_garch_resultats.pdf`) et à `documentation/
Loi_Gaussienne_vs_Empirique.pdf` §3.4 (tableau des points de contact gaussiens dans
les 5 modèles), la demande était : étendre l'option 1 (loi alternative) à **tous les
modèles**, ajouter l'option 2 (CQR) et l'option 3 (MDN, LSTM seulement), en test
comparatif seulement — **rien de tout ça n'est branché dans models/*.py en
production**, sauf `models/arima_model.py` qui avait déjà reçu `dist=`/`pi_levels`/
`n_crps_samples` lors d'une étape précédente.

## 2. Fichiers (tous committés sur `origin/main`, sauf le rapport HTML — voir §4)

| Fichier | Rôle |
|---|---|
| `experiments/dist_options_common.py` | Machinerie partagée : fit Student-t/GED (MLE bornée), conformalisation CQR, pinball loss, CRPS par échantillonnage. Docstring du module = la méthodologie complète. |
| `experiments/all_models_dist_options.py` | Orchestrateur : pour chaque actif, lance le backtest walk-forward **existant et non modifié** de chacun des 5 modèles une fois, dérive le reste en post-traitement (option 1 manuelle, CQR, + option 1 native ARIMA-GARCH). `--resume` actif par défaut. |
| `experiments/lstm_mdn_prototype.py` | Prototype MDN (option 3), processus séparé (conflit d'import TF/statsmodels). Multi-seed (3) car l'entraînement NLL-mixture n'est pas reproductible même à seed fixe — voir §3. |
| `experiments/all_models_dist_options_results.json` | Résultats bruts, 5 actifs × 5 modèles × variantes. |
| `experiments/lstm_mdn_results.json` | Résultats bruts MDN, 5 actifs × 3 seeds, agrégés (mean/std/min/max). |
| `experiments/summarize_dist_options.py` | Agrège les deux JSON ci-dessus en un résumé par (modèle, option) — couverture moyenne, erreur de calibration, largeur/pinball/CRPS **relatifs au cas gaussien** (ratio, pas valeur brute — nécessaire vu l'échelle très différente BTC/ZN=F), surcoût de calcul. |
| `experiments/dist_options_summary.json` | Sortie du script ci-dessus — la source de vérité pour le rapport. |
| `experiments/generate_dist_options_report.py` | Génère le rapport HTML final à partir de `dist_options_summary.json`. **Pas de sortie committée** — voir §4. |

## 3. Ce qu'on a appris

- **Option 1 (loi alternative) est quasi gratuite et efficace quand le problème est
  la forme de la queue** — SARIMA, Naive, ARIMA-GARCH : erreur de calibration
  moyenne réduite de 60 à 80 %, pour un surcoût de calcul de l'ordre de la
  milliseconde (contre 3 à 70 s de backtest de base selon le modèle).
- **Elle ne corrige rien quand le problème est le niveau du σ, pas sa forme.**
  Prophet sous-couvre massivement à tous les niveaux (erreur moyenne 28 points) —
  Student-t/GED n'y changent presque rien (28 → 28). CQR aide un peu (→ 20,6) sans
  résoudre le fond.
- **Elle peut activement nuire quand le σ est figé dans le temps.** LSTM (σ =
  écart-type unique des résidus d'entraînement, jamais recalculé) voit son erreur
  de calibration *empirer* de 6,77 à 11,25 sous Student-t. CQR y est neutre.
- **Le refit natif ARIMA-GARCH (vrai réajustement GARCH sous GED/skew-t) n'apporte
  rien de mesurable** par rapport au simple changement de quantile post-hoc — pour
  ~800× plus cher (2,4 s contre 0,003 s). Pas rentable.
- **Bug corrigé en cours de route** : `scipy.stats.t.fit`/`gennorm.fit` en MLE non
  contrainte divergent vers des dof ~10¹² sur des résidus quasi gaussiens
  (numériquement instable, pas un vrai signal) — remplacé par une MLE 1-paramètre
  bornée ([2.05, 200] pour dof, [0.3, 20] pour beta GED). Voir le commentaire en
  tête de `fit_student_t`/`fit_ged` dans `dist_options_common.py`.
- **MDN (option 3) : plus cher, moins bien calibré en moyenne, et instable d'un
  entraînement à l'autre même à seed fixe.** Sur SPY, deux runs strictement
  identiques (même code, même seed=42) ont donné cov_50=66 % puis 14 % avant
  stabilisation (`clipnorm=1.0` + `patience=10`). Même stabilisé, l'écart-type
  inter-seeds reste jusqu'à 4,2 points sur 3 seeds. Erreur de calibration moyenne
  finale : 9,23 (MDN) contre 6,77 (LSTM de prod) — **pire**, pour +23 % de temps
  d'entraînement. Verdict : pas rentable tel qu'implémenté.

## 4. ⚠️ Le rapport HTML n'est pas dans git (volontairement)

`experiments/generate_dist_options_report.py` embarque Cambria/Consolas (polices
sous licence Microsoft, extraites de `C:\Windows\Fonts` au moment de la génération)
en base64 pour un rendu autonome. Committer le HTML généré redistribuerait ces
polices — **jamais fait**, `experiments/dist_options_report.html` est dans
`.gitignore`. Pour régénérer :
```bash
python experiments/generate_dist_options_report.py
```
Sur une machine sans Cambria/Consolas (non-Windows), le script bascule
automatiquement sur des piles de polices système sans rien embarquer.

Le rapport a aussi été publié comme Artifact Claude (page privée par défaut) —
lien donné dans la conversation d'origine ; à republier si besoin depuis une
nouvelle session (un nouveau lien sera créé, cf. doc de l'outil Artifact).

## 5. Reproduire depuis zéro (si les JSON de résultats sont un jour perdus)

```bash
# Avast intercepte le TLS sur CETTE machine -- probablement pas nécessaire ailleurs
export CURL_CA_BUNDLE="C:\ProgramData\Avast Software\Avast\wscert.pem"

python experiments/all_models_dist_options.py --assets SPY BTC ETH ZN TLT   # ~23 min
python experiments/lstm_mdn_prototype.py --assets SPY BTC ETH ZN TLT --seeds 42 43 44  # ~9 min
python experiments/summarize_dist_options.py
python experiments/generate_dist_options_report.py
```
Les deux premiers scripts reprennent automatiquement (`--resume`, actif par
défaut) s'ils sont interrompus — aucune perte si relancés.
