# BRIEF — Refonte « clarté » du dashboard Daily vs Weekly (W+1 vs D+7)

## 0. Problème

Le dashboard `experiments/dashboard_d7_w1.html` est devenu **illisible** pour son public réel
(étudiante + tuteur, non-spécialistes de la stat) : table à ~18 colonnes, jargon partout (Winkler,
skill-score, effective_n, bootstrap par blocs, régime B/C, RW, Cov95, IC95…). **Personne ne
comprend la réponse.**

**On garde exactement la même question et les mêmes calculs** — comparer, pour une prévision à
1 semaine, le modèle **hebdomadaire** et le modèle **quotidien**. **Le seul objectif de ce brief
est la clarté.** Rien à recalculer côté données : on ne touche qu'à la **présentation**
(`dashboard_d7_w1_template.py`, et au besoin les libellés produits par `dashboard_d7_w1.py`).

**Mot d'ordre unique : limpide.** Un lecteur non-expert doit comprendre la réponse principale en
**moins de 30 secondes**, sans lire une note de méthode.

---

## 1. Principe directeur : divulgation progressive

Trois niveaux, du plus simple au plus détaillé. **Par défaut, seul le niveau 1 est visible.**

1. **La réponse** — une phrase, en français courant, par actif (et une synthèse globale).
2. **Le pourquoi, en clair** — 1 à 2 phrases par verdict, sans jargon.
3. **Les détails techniques** — chiffres, tests, définitions : **repliés** dans un bloc « Détails
   (pour les curieux) », fermé par défaut. On n'y met AUCUN terme technique sans sa définition en
   une ligne.

---

## 2. Niveau 1 — la réponse en haut de page

En tête, remplacer le titre technique par une **question simple + réponse** :

> **Pour prévoir un prix à 1 semaine, vaut-il mieux un modèle hebdomadaire ou quotidien ?**
> Réponse courte : *[générée depuis les résultats]* — ex. « Sur la plupart des actifs, les deux se
> valent. L'hebdomadaire donne une fourchette de prix plus fiable pour la crypto ; le quotidien est
> un peu plus précis sur les actions et les obligations. »

Puis une **grille de cartes**, une par actif (ou par groupe : crypto / actions / obligations), avec
pour chacune :
- un **verdict en gros, en langage courant** (voir §3),
- une **jauge de fiabilité** du verdict (voir §4),
- une phrase de « pourquoi » (niveau 2).

Pas de tableau en vue par défaut.

---

## 3. Verdicts en langage courant (traduction obligatoire)

Remplacer partout les libellés techniques par des phrases que tout le monde comprend :

| Ancien (technique) | Nouveau (limpide) |
|---|---|
| `daily_significantly_better` | **Le modèle quotidien est meilleur** |
| `weekly_native_significantly_better` | **Le modèle hebdomadaire est meilleur** |
| `indistinguishable` | **Match nul** — aucune différence fiable |
| `n insuffisant` | **Pas assez de recul** pour trancher |

Règle d'honnêteté (à garder telle quelle) : **on ne déclare un gagnant que si la différence est
solide**. Sinon c'est « match nul » — et on l'assume, ce n'est pas un trou à combler.

---

## 4. La fiabilité, en visuel — pas en chiffres

Le public n'a pas à lire « p=0,057 » ni « effective_n=10 ». Traduire la solidité statistique en une
**jauge simple à 3 niveaux** :

- 🔴 **Fiabilité faible** — peu de recul, à prendre avec prudence.
- 🟠 **Fiabilité moyenne**.
- 🟢 **Fiabilité forte** — on peut s'appuyer dessus.

Seuils (à partir de `effective_n`, déjà calculé) : faible < 15, moyenne 15–40, forte > 40. Le
chiffre exact reste disponible au niveau 3 seulement. Un « match nul » reste un « match nul » quelle
que soit la jauge (elle dit juste à quel point on est sûr qu'il n'y a pas de différence).

---

## 5. Les KPI : n'en montrer que DEUX, en mots simples

Aujourd'hui on affiche 5 indicateurs techniques. En vue par défaut, n'en garder que **deux**,
reformulés en question concrète :

1. **« Le prix prévu est-il proche de la réalité ? »** → la précision (ex-RMSE).
2. **« La fourchette de prix est-elle fiable ? »** → la vraie valeur tombe-t-elle dedans le plus
   souvent, sans que la fourchette soit trop large (ex-Winkler + ex-couverture 95 %, fusionnés en
   un seul message clair).

Les autres (largeur d'intervalle seule, exactitude directionnelle, skill-score vs marche aléatoire)
partent au **niveau 3** avec une définition en une ligne — ou sont retirés s'ils n'ajoutent rien
pour le public. Chaque carte dit, en clair : « prévision **aussi précise** des deux côtés, mais
fourchette **plus fiable** côté hebdo », etc.

---

## 6. Ce qu'on cache / déplace / supprime

- **Table à 18 colonnes** → retirée de la vue par défaut. Au niveau 3, une table **réduite**
  (Modèle · Actif · Verdict · Fiabilité) suffit ; le reste des colonnes derrière un bouton
  « tout afficher ».
- **Panneaux « Trajectoires par origine » et « Calibration »** → niveau 3 (repliés), avec un titre
  compréhensible (« Semaine par semaine, qui a gagné ? » / « Les fourchettes sont-elles bien
  calibrées ? ») et une phrase d'explication.
- **Mini-rapports actuels par verdict** → **on garde l'idée (clic sur un verdict = explication du
  pourquoi), mais réécrite en clair** (voir §6bis). On enlève le jargon, pas l'explication.
- **Pied de page méthodo** (Winkler, RW, bootstrap…) → conservé mais **entièrement dans le bloc
  replié**, chaque terme défini en une ligne en langage simple.

---

## 6bis. Cliquer un verdict = son explication, en clair (À GARDER)

L'idée du mini-rapport au clic est **conservée** — c'est même le cœur pédagogique. Ce qui change :
elle doit être **limpide**. Cliquer un verdict (carte ou ligne) déplie une explication de **3 à 4
phrases maximum**, en français courant, sans un seul terme technique :

1. **Ce que dit le verdict**, en une phrase. Ex. : *« Match nul : sur cet actif, le modèle
   hebdomadaire et le quotidien font aussi bien l'un que l'autre. »*
2. **Pourquoi**, via les deux questions concrètes (§5) : *« Les deux prévoient le prix avec une
   précision comparable. La fourchette de prix est un peu plus fiable côté hebdomadaire, mais
   l'écart est faible. »*
3. **À quel point on est sûr**, en clair : *« On n'a qu'une quinzaine de semaines de recul, donc ce
   verdict est à prendre avec prudence. »* (traduit la jauge 🔴🟠🟢, sans chiffre).
4. *(optionnel)* un lien/toggle **« voir les chiffres exacts »** qui, lui seul, révèle le détail
   technique (niveau 3) pour qui le veut.

Contraintes : texte **généré depuis les métriques** (pas écrit à la main, comme aujourd'hui), mais
avec un **gabarit de phrases en langage courant**. Un verdict « match nul » dit clairement qu'il n'y
a pas de gagnant, même si un indicateur penche légèrement. Interaction simple (clic = déplie/replie),
**aucune synchro JS**.

---

## 7. Vocabulaire — bannir ou définir

Aucun de ces mots ne doit apparaître en vue par défaut. S'ils apparaissent au niveau 3, ils sont
**définis en une ligne** juste à côté :

- « Winkler / Interval Score » → *fiabilité de la fourchette de prix*.
- « couverture 95 % / Cov95 » → *à quelle fréquence le vrai prix tombe dans la fourchette (on vise
  95 %)*.
- « effective_n / bootstrap par blocs » → *quantité de recul réellement exploitable*.
- « skill-score / marche aléatoire (RW) » → *fait-on mieux qu'une prévision naïve, et de combien*.
- « régime B / régime C » → *modèle quotidien / modèle hebdomadaire* (ne jamais écrire « régime »).
- « RMSE » → *précision du prix prévu*.

---

## 8. Contraintes techniques (inchangées)

- Page **autonome `file://`** sous Firefox : tout inline, aucun CDN/fetch, **aucune synchro JS
  multi-graphiques** (leçon `CORRECTIF_dashboard_v4_boucle_infinie.md`).
- Réutiliser les **tokens CSS** existants (`:root` clair/sombre).
- **Aucun changement des calculs** : mêmes cellules, mêmes verdicts, mêmes chiffres — seule la
  présentation change. Les libellés/phrases « clair » peuvent être **générés dans
  `dashboard_d7_w1.py`** (fonctions pures des métriques déjà calculées) pour rester data-driven.

---

## 9. Test de clarté (vérification finale)

- **Test des 30 secondes** : montrer la page à quelqu'un qui n'a jamais vu le projet ; il doit
  répondre à « alors, hebdo ou quotidien ? » sans ouvrir le moindre bloc « Détails ».
- **Zéro jargon visible** au chargement : rechercher dans le rendu par défaut « Winkler », « skill »,
  « effective_n », « bootstrap », « régime », « RW », « Cov95 » → doivent tous être **dans les blocs
  repliés uniquement**.
- **Honnêteté préservée** : les « match nul » restent des matchs nuls ; aucune fiabilité gonflée ;
  les chiffres exacts restent accessibles au niveau 3, identiques à avant.
- **Firefox `file://`** : ouverture sans serveur, blocs repliables cliquables, aucun gel.
