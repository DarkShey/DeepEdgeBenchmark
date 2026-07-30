# Message à coller dans Claude Code

Lis `BRIEF_dashboard_clarte.md` à la racine et suis-le. Objectif unique : rendre
`experiments/dashboard_d7_w1.html` **limpide** pour un public non-expert (moi + mon tuteur).
Aujourd'hui c'est illisible : table à 18 colonnes, jargon partout (Winkler, skill, effective_n,
bootstrap, régime B/C, RW, Cov95…). On **garde la même question et les mêmes calculs** (comparer,
pour prévoir à 1 semaine, le modèle hebdomadaire vs quotidien) — on ne change **que la
présentation** (`dashboard_d7_w1_template.py`, libellés générés dans `dashboard_d7_w1.py`).

Non-négociables :

- **Divulgation progressive.** Par défaut, seul le **niveau 1** est visible : la **réponse en une
  phrase** en haut + une **grille de cartes** (une par actif/groupe) avec un verdict en langage
  courant et une **jauge de fiabilité 🔴🟠🟢**. Le reste (tables, chiffres, méthode) va dans des
  blocs **« Détails » repliés par défaut**.
- **Verdicts en clair** : « Le modèle hebdomadaire est meilleur » / « Le modèle quotidien est
  meilleur » / « **Match nul** — aucune différence fiable » / « Pas assez de recul ». On ne déclare
  un gagnant que si la différence est solide (honnêteté conservée : un match nul reste un match nul).
- **GARDER l'explication au clic** (§6bis) : cliquer un verdict déplie **pourquoi** on a ce verdict,
  en 3-4 phrases de français courant, sans jargon (ce que dit le verdict · pourquoi via les 2
  questions concrètes · à quel point on est sûr). Texte généré depuis les métriques, gabarit en
  langage simple. Un toggle « voir les chiffres exacts » peut révéler le détail technique dessous.
- **Fiabilité en visuel, pas en chiffres** : jauge 3 niveaux à partir de `effective_n` (faible <15,
  moyenne 15–40, forte >40). `p`, `effective_n`, IC… restent au niveau « Détails » seulement.
- **Deux KPI max en vue par défaut**, reformulés en questions : « le prix prévu est-il proche du
  réel ? » (précision) et « la fourchette est-elle fiable ? » (le vrai prix tombe dedans le plus
  souvent, sans être trop large). Les autres → « Détails ».
- **Zéro jargon visible au chargement.** Aucun de ces mots au niveau 1 : Winkler, skill, effective_n,
  bootstrap, régime, RW, Cov95, RMSE, IC95. S'ils apparaissent dans « Détails », chacun est **défini
  en une ligne** (voir §7 du brief).
- Contraintes inchangées : autonome `file://`, tout inline, **aucune synchro JS**, tokens CSS
  existants, **aucun changement des calculs/verdicts/chiffres**.

Vérification (§9) : test des 30 secondes (un non-initié répond « hebdo ou quotidien ? » sans ouvrir
un seul bloc Détails) ; recherche dans le rendu par défaut de « Winkler/skill/effective_n/bootstrap/
régime/RW/Cov95 » → doit tout être dans les blocs repliés ; Firefox `file://` sans gel. Si tu bloques
sur une ambiguïté réelle, signale-la — ne devine pas.
