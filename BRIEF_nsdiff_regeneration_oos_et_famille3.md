# Brief Claude Code — Régénération complète de la grille oos, famille 3 reposée proprement, hygiène du benchmark

2026-08-08. Résout les deux « Non fait, volontairement » de NOTE_nsdiff_edge_vs_frais.md §6. Ce brief **détaille et remplace le chantier B** de BRIEF_nsdiff_extension_puissance_mensuel.md (même objectif, ici au niveau exécutable) ; les chantiers A (données), C (dashboard) et D (mensuel) de ce brief-là restent valables tels quels. TsDiff hors périmètre.

## Chantier R — Régénérer la grille oos complète (origines 2020-01, panel étendu, tous les modèles)

C'est « le chantier en soi ». Le traiter comme une migration de base, pas comme une expérience : phasé, chiffré avant exécution, réversible à chaque étape.

### R0. Inventaire et chiffrage (dry-run, aucune écriture)

- Recenser les modèles de référence actifs et leur mode de bornes : analytiques (ARIMA-GARCH, SARIMA, Naive, LSTM — rien à échantillonner), échantillonnés (Prophet 1000 tirages internes, NsDiff 5×200), retirés (TSDiff — aucune ligne régénérée).
- Chiffrer par modèle : temps de fit/refit × 340 origines × 5 actifs (+ GLD/USO) × 2 régimes, avant de lancer quoi que ce soit. Sortie : un JSON de plan avec coût estimé et ordre d'exécution. Si le total dépasse un seuil raisonnable (à déclarer, ex. 48 h de compute), découper et le dire.
- Réutiliser `oos_reference_audit.py` comme socle du dry-run.

### R1. Gel des données (dépend du chantier A du brief précédent)

`prices_v3/` : panel déséquilibré déclaré (SPY/ZN=F/TLT depuis 2011-05, BTC 2014-09, ETH 2017-11, + GLD/USO depuis 2011-05), une seule source, vérification de recouvrement avec `diffusion_multiseed_v2/prices/` sur la période commune (tolérance ~2e-7 relative, bloquant sinon).

### R2. Régénération par phases, weekly d'abord

L'hypothèse primaire pré-déclarée (var_limit / SPY / W+2-W+3 weekly vs GARCH ; n requis sous Holm 231-270 < 340) est weekly : cette phase suffit à trancher le programme économique.

1. **Phase W** : grille weekly 340 origines, tous modèles, 7 actifs. Protocoles naturels (GARCH refit/origine ; NsDiff train-once-forward sur <2020, 5 graines × 200, ensemble 1000). Tests confirmatoires de l'hypothèse primaire + réplication des 4 survivants « B&H bat la stratégie » (SPY W+1) + match calibration NsDiff vs GARCH.
2. **Phase D** : idem daily — seulement si la phase W justifie de continuer (critère déclaré : hypothèse primaire survivante OU signal de calibration nouveau à documenter).
3. **Écriture en base** : un seul script d'upsert par phase, standard `repoint_oos_to_m200` (dry-run par défaut, sauvegarde horodatée, `--apply`, vérification 1:1 des clés, backfill des colonnes dérivées, bandeau de config). tracking.db en lecture seule partout ailleurs.

### R3. Comparabilité et déclarations

- Aucun mélange ancien/nouveau : les verdicts issus de la grille 90 origines restent cités comme tels ; la note de synthèse porte un tableau « ancienne grille vs nouvelle grille » pour les conclusions-clés (couverture, Winkler, TOST, PnL), avec explication de tout renversement.
- Familles de Holm redéclarées avant les runs (structure identique au chantier précédent). Tout ce qui n'est pas l'hypothèse primaire est étiqueté exploratoire.
- GLD/USO : premiers entrants — validation d'intégrité des données (jours fériés, rolls pour USO via ETF) avant inclusion dans les tests poolés ; sinon rapportés à part.

## Chantier F3 — La famille 3, reposée correctement (pas « rejouée »)

Le constat est structurel : « position si le PI exclut le prix courant » n'émet jamais de signal — à 95 % comme à 80 %, 0 position sur 3 240 origines-instruments, car la largeur médiane (8 % du prix à 80 %) dépasse toujours le drift médian à 1-3 semaines. Balayer d'autres niveaux (70 %, 60 %...) après avoir vu ces résultats serait exactement le p-hacking que le programme s'interdit — et ne changerait rien au mécanisme. La famille est close **dans sa formulation actuelle** ; ce chantier la remplace par la bonne question, pré-déclarée.

- **Nouvelle famille déclarée a priori** : signal directionnel normalisé. Position si |médiane prédictive − prix courant| > k × largeur du nuage (échelle interquantile), signe donné par la médiane. k ∈ {0,25 ; 0,5} — deux valeurs, déclarées ici, pas de balayage. C'est le ratio drift/incertitude que la famille 3 mesurait maladroitement par exclusion binaire.
- **Statut** : exploratoire sur la grille actuelle (90 origines) pour vérifier que des signaux sont émis (comptage descriptif) ; confirmatoire uniquement sur la nouvelle grille 340 origines, famille Holm dédiée, PnL net de frais vs B&H et vs GARCH (GARCH évalué avec le même signal sur ses propres quantiles — symétrie complète).
- **Critère de clôture déclaré** : si aux deux valeurs de k la famille n'émet pas de signaux exploitables nets de frais sur la nouvelle grille, la question « les intervalles de NsDiff portent-ils un signal directionnel ? » est close définitivement, sans autre reformulation.

## Chantier H — Hygiène du benchmark (les faiblesses de l'arbitre, pas du candidat)

Trois points issus des notes elles-mêmes, par ordre d'importance :

### H1. Le champion GARCH n'est pas à sa meilleure config — à corriger en priorité

NOTE_nsdiff_edge_vs_frais §3.4 l'a établi par mesure : les lignes oos de GARCH sont la variante **gaussienne**, alors que `arima_model` déclare `GARCH_DIST="skewt"` aujourd'hui. Le benchmark fait donc jouer son champion avec des queues fines sur des actifs à queues épaisses. Régénérer le bras GARCH en skew-t sur la nouvelle grille (il est refit par origine — coût marginal nul dans R2), comparer gaussien vs skew-t, et acter la config championne. Enjeu : si skew-t couvre mieux en crypto, le « mur GARCH » se renforce et le verdict NsDiff en sort plus honnête ; si gaussien reste meilleur, c'est documenté au lieu d'être un accident.

### H2. Réalisme d'exécution des futures

La base ES/SPY et le roulement trimestriel ne sont pas modélisés (simplification déclarée favorable au future). Avant toute décision réelle fondée sur SPY-ES/ZN-FUT : modéliser le roll (coût par trimestre) et vérifier que l'edge net y survit. De même, remplacer la grille de frais hypothétique par des bordereaux réels dès qu'ils existent (`real_fees.py`, un seul fichier à toucher — dix minutes déclarées).

### H3. Monitoring de couverture en ligne (le benchmark s'arrête où la production commence)

Tout le programme mesure une couverture rétrospective sur grille figée. Pour un usage réel, ajouter une brique de suivi : couverture glissante (fenêtre ~26 semaines) par actif avec bande d'alerte déclarée (ex. sortie de [0,88 ; 0,99] → investigation), branchée sur la piste oos du dashboard. C'est la version opérationnelle de Kupiec — et le seul moyen de détecter une dérive de régime avant qu'elle coûte.

## Ordre et critère d'arrêt

R0 immédiat (aucune écriture) → R1 → R2 phase W (avec H1 intégré) → F3 confirmatoire et R2 phase D selon leurs critères → H2/H3 en parallèle de la phase D. Critère d'arrêt inchangé et rappelé : si l'hypothèse primaire ne survit pas à Holm sur 340 origines, le volet économique se clôt définitivement — F3 inclus si sa famille est aussi négative.

## Non-négociables (inchangés)

Multi-graines 42-46 ; conventions descriptif / graine fixe / poolé ; familles Holm et tous k, seuils, grilles déclarés avant les runs (ceux de ce brief font foi) ; briques réutilisées (econ_backtest, paired_test, mcs.spa_test, multiple_testing, real_fees, oos_reference_audit, standard repoint) ; tracking.db lecture seule hors scripts d'upsert dédiés ; prix gelés partagés ; budget d'échantillonnage égal entre modèles comparés ; pytest vert avant/après (573 passed, 1 skipped actuel).
