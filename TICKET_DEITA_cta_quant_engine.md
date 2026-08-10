# TICKET DEITA — cta_quant_engine.py : deux défauts du signal CTA affectant la production

2026-08-08. Émis depuis DeepEdgeBenchmark (chantier « couplage CTA × sizing NsDiff », NOTE_couplage_cta_deita_sizing_nsdiff.md). À copier dans le repo DEITA. Les deux défauts ont été découverts en *vérifiant* l'interface plutôt qu'en l'affirmant ; preuves et hashes dans `experiments/deita_cta_signal/manifest.json`, `cta_gate0.json`, `cta_gate0_conviction.json`.

**Priorité suggérée : haute si la conviction hiérarchique ou le trend Hull alimentent des décisions réelles.** Les deux corrections changent les signaux de production — elles doivent être validées côté DEITA, avec ses propres tests ; rien dans DeepEdgeBenchmark n'en dépend (le programme concerné y est clos).

## Bug 1 — `_conviction_level` : la conviction des actifs singleton est un acheter-et-garder déguisé

**Mécanisme.** Pour un actif seul dans son groupe, `_conviction_level` renvoie sa propre série lissée. Un actif seul dans son sous-secteur ET son secteur obtient donc :

```
conv = smooth × (mkt + 2·smooth)/3 = (2/3)·smooth² + smooth·mkt/3
```

Le terme dominant est un **carré** : le signe ne change plus jamais.

**Preuve mesurée.** Sur la grille 2020-2026 : SPY 100,0 % de positions longues, BTC 100,0 %, TLT 99,9 %, ZN=F 99,4 % — contre 47-60 % pour la direction Hull sous-jacente. Le défaut est présent dans l'univers de 16 actifs de DEITA lui-même (tout sous-secteur à un membre). Preuve numérique du caractère « B&H déguisé » : rejoué en backtest, l'écart de PnL vs acheter-et-garder vaut **exactement 0,00 bps, p = 1,000** (test unitaire figé : `test_un_signal_toujours_long_est_exactement_acheter_et_garder`).

**Impact production.** Pour tout actif singleton, la « conviction » ne module rien et ne protège de rien : l'exposition est structurellement longue permanente, quel que soit le trend. Le risque est invisible en marché haussier et se paie en baissier.

**Correction proposée.** Pour un groupe à un seul membre, faire hériter la conviction du **niveau supérieur** (sous-secteur singleton → conviction secteur ; secteur singleton → conviction marché) au lieu de renvoyer la série de l'actif — ce qui supprime le carré. Alternative plus conservatrice : neutraliser le niveau dégénéré (poids nul) dans l'agrégation.

**Critères d'acceptation.** (i) Sur les singletons, la part de positions longues redevient du même ordre que celle de la direction Hull (~45-60 %), plus jamais ~100 % ; (ii) test unitaire : conviction d'un singleton = fonction du niveau supérieur, jamais un carré de sa propre série ; (iii) non-régression sur les actifs à pairs (ETH/GLD/USO dans l'univers unifié : comportement inchangé à tolérance déclarée) ; (iv) diff des signaux de production avant/après, revu explicitement.

## Bug 2 — `ffill()` sur le calendrier d'union : le trend des actifs 5 j/7 est dilué, surtout dans les krachs

**Mécanisme.** `prices[[asset]].ffill()` est appliqué sur le calendrier de l'union du panel. Avec la crypto (7 j/7) dans l'univers, les actions/obligations héritent du prix du vendredi le samedi et le dimanche : **deux jours à rendement nul entrent chaque semaine** dans la moyenne de Hull 20 jours des actifs à 5 jours.

**Preuve mesurée.** Comparé à un calcul sur les jours de cotation propres : 3 divergences sur 28 comparaisons d'interface ; le signe du signal quotidien diffère sur **13-15 % des observations** des actifs 5 j/7 (7,3 % des cellules de la grille hebdo). Effet le plus parlant : le crisis alpha de mars 2020 sur SPY passe de **+84 bps/origine (Sharpe 1,41) à −2 bps (Sharpe −0,03)** selon le calendrier — les rendements de krach sont dilués dans des jours nuls, et le trend réagit avec retard précisément quand il doit protéger.

**Impact production.** Sur tous les actifs à 5 jours de l'univers, le signal trend est amorti et retardé ; l'effet est maximal dans les chocs — le moment où un CTA justifie son existence. (Observation issue du benchmark : sous la convention actuelle, le CTA sur SPY est significativement battu par acheter-et-garder, p ≈ 0,04, −42 bps/origine — descriptif, panel limité, mais cohérent avec le mécanisme.)

**Correction proposée.** Calculer les rendements et la moyenne de Hull sur le **calendrier de cotation propre à chaque actif**, et ne réindexer sur l'union qu'après (ffill du *signal*, pas des *prix*). Alternative : masquer les jours non cotés en NaN et les exclure de la moyenne glissante.

**Critères d'acceptation.** (i) Test d'équivalence : sur un actif 7 j/7, la correction ne change rien ; (ii) sur un actif 5 j/7, la moyenne de Hull ne contient plus de rendements nuls hérités de week-ends ; (iii) rejeu d'une fenêtre de choc (février-avril 2020) avec diff avant/après documenté ; (iv) diff des signaux de production, revu explicitement.

## Note d'interaction entre les deux bugs

Les corriger séparément et dans cet ordre (1 puis 2), avec un diff de production à chaque étape : le bug 1 masque partiellement le bug 2 (une conviction toujours longue rend le calendrier du trend sans conséquence sur le signe). Après correction du bug 1, l'effet du bug 2 devient pleinement visible sur les singletons.
