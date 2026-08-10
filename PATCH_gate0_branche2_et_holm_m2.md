# PATCH — Chantier 0 CTA : branche 2 réparée, famille de Holm corrigée (m = 2)

2026-08-08. Patch de protocole, côté DeepEdgeBenchmark. Aucun run relancé, aucun verdict modifié — la NOTE_couplage_cta_deita_sizing_nsdiff.md l'a montré : la porte échoue sous toutes les variantes de signal réel (calendrier propre ET convention DEITA), et le seul « passage » observé transitait par la branche 2 défectueuse. Ce patch répare l'instrument pour le jour où la question rouvrirait, et fige la correction documentaire du brief. Standard `patch_note_decision` : dry-run par défaut, repérage par contenu (pas par index), sauvegarde horodatée, `--apply` explicite, idempotent.

## P1 — Brief : famille de Holm m = 4 → m = 2

`BRIEF_couplage_cta_deita_sizing_nsdiff.md`, section « Hypothèse primaire et familles » : « Famille de Holm primaire (m = 4) » est incohérent avec la famille décrite — {SPY-ES, ZN-FUT} × {W+1}, A vs B = **2 tests**. Corriger en **m = 2**.

Légitimité de la correction : les chantiers 1-2 n'ont jamais tourné (arrêt à la porte 0), la correction reste donc une déclaration a priori ; et l'erreur allait dans le sens conservateur (m surdéclaré = seuils plus stricts). Ne PAS étendre la famille à W+2/W+3 pour « justifier » le 4 : W+1 est l'horizon de détention déclaré. Tracer la correction dans le brief lui-même (une ligne : « m = 2, corrigé le 2026-08-08 avant tout run, cf. NOTE §3 »).

## P2 — Branche 2 de la porte : immunisation contre le signal constant

Défaut prouvé (NOTE §2.4) : « Sharpe poolé > 0 avec ≥ 3/4 classes positives » est franchissable par un signal toujours long — sur un panel haussier, le Sharpe poolé mesure le bêta, pas le signal. La conviction dégénérée (= B&H déguisé, edge exactement 0,00, p = 1,000) la franchit avec Sharpe 0,77 et 3/4 classes.

**Nouvelle formulation, dans `cta_gate0.py`** : branche 2 = « Sharpe poolé de l'**excès** (PnL signal − PnL B&H, par origine) > 0, ET ≥ 3 classes sur 4 à excès positif ». Un signal constant a un excès exactement nul partout → échec mécanique, quelle que soit la pente du marché.

Tests unitaires exigés :

1. `test_branche2_immune_au_signal_constant` : le signal toujours long doit désormais échouer à la branche 2 (excès nul → ni Sharpe > 0 ni classes positives). C'est la contre-épreuve du défaut documenté.
2. `test_branche2_detecte_un_vrai_excess` : un signal synthétique à excès positif construit doit passer. La branche répare, elle ne condamne pas par construction.
3. Non-régression : rejouer l'évaluation de la porte sur les artefacts archivés (`cta_gate0.json`, `_conviction.json`, les deux calendriers) et vérifier que le verdict global reste ÉCHEC sous toutes les variantes — l'écrire dans le JSON du patch.

**Addendum à la NOTE** (§2.4 et §4, repérage par contenu) : la piste de réouverture n° 2 (« corriger la branche 2 ») est traitée par ce patch — la retirer de la liste des pistes ouvertes ; les pistes 1 (DEITA) et 3 (fenêtre/panel — p-hacking, fermée) restent en l'état. Mentionner que la correction est faite *après* le verdict mais ne l'affecte pas, preuve rejouée à l'appui.

## P3 — Consigner la décision de calendrier (déjà actée, à verrouiller)

La convention DEITA (`ffill` sur calendrier d'union) a été adoptée pour l'équivalence au moteur (« tel quel »), signal refigé, 0 écart sur 28, test d'équivalence en place — et le verdict de porte est inchangé (il se renforce : Sharpe 0,21 → 0,17, 1/4 classes, crisis alpha 2020 disparu). Ce patch ajoute seulement le verrou : un test qui échoue si la convention de calendrier du signal gelé diverge de celle de `compute_cta_signal(pure_trend_mode=True)`, pour qu'aucun refactor futur ne réintroduise silencieusement le calendrier propre. L'observation « le ffill dilue les rendements de krach » appartient au ticket DEITA, pas à ce patch — elle n'est pas utilisée pour changer de convention après coup.

## Critères de done

pytest vert (744 passed + les nouveaux) ; dry-run montrant les diffs exacts du brief et de la note avant `--apply` ; JSON d'artefact avec le rejeu de non-régression des trois variantes ; aucune écriture hors de ces deux fichiers markdown et de `cta_gate0.py`/tests.
