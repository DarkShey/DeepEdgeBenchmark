# BRIEF — Mini-dashboard externalisé : comparaison poussée D+7 vs W+1

## 0. Contexte

La matrice est complète (240/240) et testée par paires (`experiments/matrice_paired_tests.json`,
`comparison_4_d7_vs_w1_friday_aligned`). La comparaison **D+7 vs W+1** — « pour prévoir 1 semaine,
vaut-il mieux un modèle daily projeté à 7 jours calendaires (régime A) ou un modèle weekly natif
(régime C) ? » — mérite une lecture dédiée. Mais elle n'a **pas sa place dans `Run/dashboard.html`**
(déjà lourd, et un empilement de synchronisations JS y a déjà provoqué un gel navigateur, cf.
`calibration/regime/archive_briefs/CORRECTIF_dashboard_v4_boucle_infinie.md`).

**Objectif** : une page HTML **autonome, séparée, ouvrable directement dans Firefox en `file://`**,
qui présente la comparaison D+7↔W+1 de façon rigoureuse — verdicts testés par cellule, agrégation
inter-actifs pour gagner en puissance, et affichage explicite de l'incertitude. Livrable présentable
au tuteur.

**Ce dashboard ne modifie NI le pipeline, NI la DB, NI `Run/dashboard.html`.** Il lit
`validation/tracking.db` et produit un fichier isolé.

---

## 1. La comparaison, et ses deux pièges (à porter dans l'UI, pas à masquer)

### 1.1 Définition de l'appariement (identique à `comparison_4_d7_vs_w1`)
- **D+7** = régime A, modèle daily, cible = cutoff + 7 jours calendaires (même jour de semaine).
- **W+1** = régime C, weekly natif (`frequence = "weekly"`), cible = vendredi suivant.
- Ils **coïncident uniquement quand l'origine est un vendredi** → l'appariement se fait sur les
  **origines-vendredi** : ligne D+7 dont `cutoff_date` est un vendredi, jointe à la ligne W+1
  partageant ce `cutoff_date`. Par `(model, asset)`.

### 1.2 Piège n°1 — puissance très faible (à AFFICHER)
L'appariement donne **9 à 14 paires par cellule** (30 cellules = 6 modèles × 5 actifs, 358 paires
au total), et le bootstrap par blocs ramène le **`effective_n` à ~3-4**. Toute p-value par cellule
est donc **optimiste et fragile**. Décision assumée : le dashboard **affiche `n` et `effective_n`
partout**, et ne titre jamais un verdict de cellule sans le rappel de puissance. Un « indistinguable »
est un résultat honnête, pas un trou à combler.

### 1.3 Piège n°2 — confusion horizon × régime d'entraînement (à DIRE)
D+7 vs W+1 n'est **pas** un test propre de « daily vs weekly à horizon égal » : on compare un
**régime A (daily→7 j)** à un **régime C (weekly natif)**. L'écart mélange l'effet de la définition
d'horizon ET l'effet du régime d'entraînement. C'est **intra-modèle** (même modèle des deux côtés) →
l'asymétrie de protocole (TSDiff figé à T0) ne s'applique pas, c'est donc une comparaison propre au
sens protocole — mais elle doit être **libellée « régime A vs régime C sur cible-vendredi »**, pas
« daily vs weekly » tout court. À écrire noir sur blanc dans l'en-tête.

---

## 2. Décisions de conception assumées (choix de quant)

### 2.1 Métriques retenues — et pourquoi
La DB ne stocke que `(y_pred, y_lower, y_upper, y_true)`, **pas les ensembles**. On ne calcule donc
**pas** de vrai CRPS (qui exige les échantillons) : on choisit des métriques honnêtes vu la donnée
disponible.

| Métrique | Rôle | Source (colonnes DB) |
|---|---|---|
| **RMSE** (par origine : erreur²) | précision du point ; continuité avec `comparison_4` | `y_pred, y_true` |
| **Winkler / Interval Score @95 %** | **note probabiliste principale** : pénalise à la fois une mauvaise couverture ET des intervalles trop larges — la *proper scoring rule* calculable à partir des bornes | `y_lower, y_upper, y_true` |
| **Cov95** (couverture réelle) | calibration : atteint-on ~0,95 ? | `in_interval` |
| **Largeur moyenne du PI 95 %** | finesse : départage à couverture égale | `y_lower, y_upper` |
| **Exactitude directionnelle** | diagnostic secondaire (utile trading) — **à baliser « très bruité à n≈10 »** | `direction_correct` |

> Le **Winkler score** est la bonne réponse quant à « donnée = bornes de PI, pas d'ensemble » : ne
> pas maquiller un CRPS approximé en CRPS. Si (et seulement si) `experiments/kpi_probabilistes.json`
> contient des échantillons couvrant ces origines-vendredi, on **peut** ajouter un vrai CRPS en
> phase 2, clairement étiqueté ; sinon on s'en tient au Winkler.

### 2.2 Agrégation inter-actifs pour gagner en puissance — sans tricher
Le faible `effective_n` par cellule se corrige en **poolant les origines entre actifs**. Mais le
RMSE/Winkler absolu **n'est jamais comparable entre actifs** (échelles différentes). Donc :

1. Métrique **sans échelle par origine** : `skill = 1 − score_modèle / score_RW`, où RW = baseline
   marche aléatoire (point = dernier close ; PI = quantiles des rendements cumulés à h, fenêtre ≤
   origine) aux **mêmes origines/cibles**. On pool la **différence de skill D+7 − W+1** par origine.
2. **Grouper par classe d'actif** (crypto : BTC, ETH / actions : SPY / obligations : ZN=F, TLT) —
   et **ne pas double-compter ZN=F & TLT** (corrélés) : les agréger en une contribution « taux »
   pondérée, pas deux voix indépendantes.
3. **Test poolé** par bootstrap par blocs sur les origines partagées (blocs = origines consécutives),
   avec `effective_n` poolé affiché. Verdict global + par classe.

> C'est la voie pour un verdict tranché, **à condition** d'afficher le caveat de corrélation
> inter-actifs : le pooling gonfle `n` mais la corrélation résiduelle entre séries peut rendre les
> IC encore un peu optimistes.

### 2.3 Rigueur statistique (partout)
- **Réutiliser** `experiments/paired_test.py` (bootstrap par blocs déjà écrit et testé) — ne pas
  réimplémenter le test.
- **Bootstrap par blocs** obligatoire (origines corrélées), pas de tirage indépendant.
- **Verdict seulement si significatif** (p < 0,05) : sinon « indistinguable ».
- **Seed fixe** → p-values reproductibles d'un run à l'autre.

---

## 3. Le livrable technique : générateur + page autonome

### 3.1 Générateur
- Nouveau script isolé : **`experiments/dashboard_d7_w1.py`** (n'importe pas et ne modifie pas
  `model_artifacts/generate_dashboard.py`).
- Entrée : `validation/tracking.db`. Sortie : **`experiments/dashboard_d7_w1.html`** (un seul
  fichier).
- Recompute l'appariement exactement comme `comparison_4_d7_vs_w1` (mêmes cellules, mêmes `n`) —
  idéalement en **important** la logique de `experiments/matrice_paired_tests.py` plutôt qu'en la
  recopiant.
- CLI : `python -m experiments.dashboard_d7_w1 --db-path validation/tracking.db --out experiments/dashboard_d7_w1.html --seed 42`.

### 3.2 Contraintes navigateur (non négociables)
- **Autonome et `file://`-compatible dans Firefox** : tout inline (JS de tracé **embarqué**, pas de
  `fetch` ni de CDN requis pour ouvrir la page hors-ligne). Suivre la philosophie du mode `--inline`
  de `generate_dashboard.py`.
- **Aucune boucle de synchronisation JS.** Leçon du CORRECTIF v4 : les graphiques Plotly qui
  s'écoutent mutuellement au zoom ont figé Firefox. Ici : **garder les graphiques indépendants**
  (pas de zoom synchronisé). Si une synchro est vraiment voulue, garde-fou de ré-entrance
  **asynchrone** (remise à `false` après le `plotly_relayout` différé, pas juste après l'appel) —
  mais par défaut, on s'en passe.
- Réutiliser les **tokens CSS** de `Run/dashboard.html` (`:root` clair/sombre) pour la cohérence
  visuelle, sans importer son JS.

### 3.3 Contenu de la page (panneaux)
1. **En-tête / méthodo repliée** : définition D+7 vs W+1, appariement origines-vendredi, les deux
   caveats (§1.2 puissance, §1.3 confusion régime), source DB + date de génération + seed.
2. **Verdict par cellule** (table triable, 30 lignes `model × asset`) : `RMSE_d7`, `RMSE_w1`,
   `Winkler_d7`, `Winkler_w1`, `Cov95_d7`, `Cov95_w1`, `mean_diff`, IC bootstrap, `p_value`, `n`,
   `effective_n`, **badge verdict coloré** (`daily_D+7_significantly_better` /
   `weekly_native_significantly_better` / `indistinguishable`). Recoupe `comparison_4`.
3. **Agrégat poolé** (§2.2) : skill-score D+7 vs W+1 **global et par classe d'actif** (crypto /
   actions / obligations-taux dédoublonnées), test poolé, verdict tranché + caveat corrélation.
4. **Trajectoires par origine** (pour une cellule sélectionnée) : barres signées de la différence
   d'erreur par origine → on **voit** si un verdict tient à une origine isolée ou à un vrai motif ;
   + largeur de PI par origine.
5. **Calibration** : Cov95 réelle D+7 vs W+1 par cellule face à la cible 0,95 (qui sous-couvre).
6. **Pied de page** : limites (puissance, confusion régime, corrélation inter-actifs), formule du
   Winkler, définition RW.

---

## 4. Plan d'implémentation

1. **Chargement + appariement** depuis la DB, identique à `comparison_4_d7_vs_w1` (Friday-only,
   join sur `cutoff_date`).
2. **Métriques par origine** : erreur², Winkler@95, in-interval, largeur PI, direction.
3. **Baseline RW** aux mêmes origines/cibles, pour les skill-scores.
4. **Tests par cellule** via `paired_test.py` (bootstrap par blocs, seed fixe).
5. **Agrégation poolée** par classe d'actif (dédoublonnage taux) + test poolé.
6. **Rendu HTML autonome** (panneaux §3.3), tokens CSS de `Run/dashboard.html`, graphiques Plotly
   **indépendants**, tout inline.
7. **Vérification** (§6).

---

## 5. Livrables

- `experiments/dashboard_d7_w1.py` — le générateur.
- `experiments/dashboard_d7_w1.html` — la page autonome (ouvrable en `file://` sous Firefox).
- `experiments/dashboard_d7_w1_data.json` (optionnel) — les métriques/verdicts calculés, pour
  traçabilité/tests.
- Point d'étape court : nb de cellules, verdicts significatifs (cellule + poolé), et ce qui reste
  indistinguable.

---

## 6. Vérification

- **Recoupement** : les cellules et `n` de la table §3.3-2 **matchent** `comparison_4` de
  `matrice_paired_tests.json` ; le verdict RMSE par cellule est identique (mêmes diffs, même seed).
- **Firefox `file://`** : la page ouvre sans serveur, les box/tables/sélecteurs sont cliquables, et
  **aucun gel** — rejouer un clic de zoom (playwright firefox) pour confirmer l'absence de la boucle
  v4.
- **Aucun RMSE/Winkler comparé entre actifs** en absolu (seul le skill sans échelle est poolé).
- **`effective_n` affiché partout** ; aucun verdict titré sans rappel de puissance.
- **Reproductibilité** : deux runs à seed égal → mêmes p-values.
- **Winkler** vérifié sur un cas jouet (borne connue) avant de peupler la page.

---

## 7. Ce que ce brief ne fait PAS

- Ne touche **pas** `Run/dashboard.html`, ni le pipeline, ni le schéma DB.
- **Ne recalcule pas** les prédictions : lit la DB existante telle quelle.
- N'ajoute **ni modèle ni horizon** ; strictement D+7 vs W+1 sur origines-vendredi.
- **Pas de vrai CRPS** tant que les ensembles ne sont pas disponibles pour ces origines (Winkler à
  la place ; CRPS = phase 2 optionnelle, étiquetée).
- **Aucun verdict** sur estimation ponctuelle ; aucun « gagnant » non significatif.
- Pas de synchronisation JS multi-graphiques (leçon CORRECTIF v4).
