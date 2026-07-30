"""
dashboard_d7_w1_template.py — le HTML/CSS/JS statique de experiments/dashboard_d7_w1.py.

Séparé du générateur pour lisibilité seulement (aucune logique ici, aucun calcul).
100% autonome : pas de <script src>, pas de fetch(), rien qui requiert le réseau
pour ouvrir la page en file://. Volontairement SANS librairie de graphes
(pas de Plotly) : les graphiques (barres signées par origine, calibration) sont
de simples <div> dimensionnés en CSS, générés/mis à jour par un JS vanille
minimal, sans qu'aucun graphique n'écoute un autre -- élimine structurellement
la classe de bug du CORRECTIF_dashboard_v4_boucle_infinie.md.

Refonte clarté (BRIEF_dashboard_clarte.md) : divulgation progressive à 3 niveaux.
Niveau 1 (visible par défaut) = question + réponse en clair + grille de 3 cartes
(crypto/actions/obligations), verdict en langage courant, jauge de fiabilité
🔴🟠🟢. Niveau 2 = phrase de "pourquoi" affichée sur chaque carte + explication
3 phrases au clic (verdict = <summary> d'un <details>, natif, sans JS de plus).
Niveau 3 = tout le reste (table complète, trajectoires, calibration, chiffres
exacts, méthodo) dans des <details> repliés par défaut -- le natif <details>
garantit qu'aucun de ces blocs n'apparaît dans le rendu par défaut (innerText),
sans code JS supplémentaire pour gérer l'ouverture/fermeture. Les libellés/
verdicts "clairs" sont des champs déjà calculés par dashboard_d7_w1.py (fonctions
pures) -- ce fichier ne fait qu'assembler le HTML, aucun calcul ni texte figé
indépendant des chiffres.

__DATA_JSON__ est remplacé par le générateur avec le payload JSON complet
(cellules, trajectoires, agrégats, config, langage clair).
"""

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Hebdomadaire ou quotidien pour prévoir à 1 semaine ?</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --surface-1:      #fcfcfb;
  --page-plane:     #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --grid-line:      #e1e0d9;
  --baseline:       #c3c2b7;
  --border-ring:    rgba(11,11,11,0.10);
  --card-shadow:    0 1px 2px rgba(11,11,11,0.06);
  --pos-color:      #1baf7a;
  --neg-color:      #d64550;
  --d7-color:       #c98a2c;
  --w1-color:       #2f6fb0;
  --tie-color:      #898781;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --grid-line:      #2c2c2a;
    --baseline:       #383835;
    --border-ring:    rgba(255,255,255,0.10);
    --card-shadow:    0 1px 3px rgba(0,0,0,0.4);
    --pos-color:      #2ecc9a;
    --neg-color:      #e5606b;
    --d7-color:       #e0a83f;
    --w1-color:       #5b9bd9;
    --tie-color:      #9a988f;
  }
}
:root[data-theme="dark"] {
  --surface-1: #1a1a19; --page-plane: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --text-muted: #898781; --grid-line: #2c2c2a; --baseline: #383835; --border-ring: rgba(255,255,255,0.10);
  --card-shadow: 0 1px 3px rgba(0,0,0,0.4); --pos-color: #2ecc9a; --neg-color: #e5606b;
  --d7-color: #e0a83f; --w1-color: #5b9bd9; --tie-color: #9a988f;
}
:root[data-theme="light"] {
  --surface-1: #fcfcfb; --page-plane: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --text-muted: #898781; --grid-line: #e1e0d9; --baseline: #c3c2b7; --border-ring: rgba(11,11,11,0.10);
  --card-shadow: 0 1px 2px rgba(11,11,11,0.06); --pos-color: #1baf7a; --neg-color: #d64550;
  --d7-color: #c98a2c; --w1-color: #2f6fb0; --tie-color: #898781;
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 32px 64px; background:var(--page-plane); color:var(--text-primary);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size:22px; margin:0 0 10px; max-width: 760px; }
h2 { font-size:15px; margin:0 0 12px; }
h3 { font-size:13px; margin:0 0 8px; color:var(--text-secondary); }
.subtitle { color:var(--text-secondary); font-size:13px; margin:0 0 4px; }
.meta-line { color:var(--text-muted); font-size:12px; margin:2px 0 20px; }
.header-answer { font-size:16px; line-height:1.6; margin: 0 0 26px; max-width: 780px; color:var(--text-primary); }
.card { background:var(--surface-1); border:1px solid var(--border-ring); border-radius:10px;
        box-shadow:var(--card-shadow); padding:18px 20px; margin-bottom:20px; }
.note { font-size:12.5px; color:var(--text-secondary); margin:0 0 10px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
thead th { text-align:left; padding:6px 8px; color:var(--text-secondary); font-weight:600;
           border-bottom:1px solid var(--border-ring); cursor:pointer; white-space:nowrap; user-select:none; }
thead th:hover { color:var(--text-primary); }
thead th.sorted::after { content:" " attr(data-arrow); }
tbody td { padding:5px 8px; border-bottom:1px solid var(--grid-line); font-variant-numeric:tabular-nums;
           white-space:nowrap; }
tbody tr:hover { background:rgba(128,128,128,0.06); }
.table-wrap { overflow-x:auto; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600;
         color:#fff; white-space:nowrap; }
.badge.d7 { background:var(--d7-color); }
.badge.w1 { background:var(--w1-color); }
.badge.tie { background:var(--tie-color); }
.badge.na { background:var(--text-muted); }
.select-box { font:inherit; font-size:13px; padding:6px 10px; border-radius:8px; border:1px solid var(--border-ring);
              background:var(--surface-1); color:var(--text-primary); }
.controls-row { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
.origin-row { display:grid; grid-template-columns:82px 1fr 90px; align-items:center; gap:8px; margin-bottom:3px; font-size:12px; }
.origin-label { color:var(--text-secondary); font-variant-numeric:tabular-nums; }
.origin-value { text-align:right; font-variant-numeric:tabular-nums; color:var(--text-secondary); }
.signed-track { position:relative; height:13px; background:var(--grid-line); border-radius:3px; }
.signed-zero { position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--baseline); }
.signed-fill { position:absolute; top:1px; bottom:1px; border-radius:2px; }
.pair-track { position:relative; height:11px; background:var(--grid-line); border-radius:3px; margin-bottom:2px; }
.pair-fill { position:absolute; top:0; bottom:0; left:0; border-radius:3px; }
.cal-track { position:relative; height:13px; background:var(--grid-line); border-radius:3px; }
.cal-fill { position:absolute; top:1px; bottom:1px; left:0; border-radius:2px; }
.cal-target { position:absolute; top:-2px; bottom:-2px; width:1px; background:var(--text-primary); opacity:0.5; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.class-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:14px; margin-top:10px; }
.class-tile { border:1px solid var(--border-ring); border-radius:8px; padding:12px 14px; }
.class-tile .title { font-size:12px; color:var(--text-secondary); margin-bottom:6px; font-weight:600; }
.class-tile .n-line { font-size:11px; color:var(--text-muted); margin-top:6px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--text-secondary); margin-bottom:10px; }
.legend-item { display:inline-flex; align-items:center; gap:6px; }
.legend-swatch { width:10px; height:10px; border-radius:2px; display:inline-block; }
footer.card { font-size:12px; color:var(--text-secondary); line-height:1.6; }
footer.card code { background:var(--grid-line); padding:1px 5px; border-radius:4px; }
.no-data { color:var(--text-muted); font-size:13px; }
.verdict-toggle { background:none; border:none; padding:0; margin:0; cursor:pointer; font:inherit;
                  display:inline-flex; align-items:center; gap:5px; }
.expand-arrow { font-size:9px; color:var(--text-muted); }
tr.detail-row td { background:rgba(128,128,128,0.05); padding:12px 16px; white-space:normal; }
.report-basis { font-size:12.5px; color:var(--text-secondary); margin:0 0 8px; line-height:1.5; }
.report-kpi-table { width:100%; font-size:12px; border-collapse:collapse; margin-top:4px; }
.report-kpi-table th, .report-kpi-table td { padding:4px 6px; border-bottom:1px solid var(--grid-line);
                                             text-align:left; white-space:normal; vertical-align:top; }
.report-arbitrage { font-size:12.5px; margin:8px 0 0; line-height:1.5; }
.report-arbitrage.conflict { font-weight:600; color:var(--text-primary); }
.agg-report { border-top:1px dashed var(--border-ring); margin-top:8px; padding-top:8px; }
.agg-report p { margin:4px 0; }

/* ── Refonte clarté : cartes niveau 1, blocs Détails niveau 3 ──────────────── */
.card-grid-3 { display:grid; grid-template-columns:repeat(auto-fit, minmax(250px,1fr)); gap:16px; margin-bottom:26px; }
details.verdict-card { background:var(--surface-1); border:1px solid var(--border-ring); border-radius:10px;
                       box-shadow:var(--card-shadow); padding:16px 18px; }
details.verdict-card summary { cursor:pointer; list-style:revert; }
details.verdict-card summary::marker { color:var(--text-muted); }
.vc-title { font-size:12px; color:var(--text-muted); font-weight:600; text-transform:uppercase;
           letter-spacing:0.03em; margin-bottom:6px; }
.vc-headline { font-size:16.5px; font-weight:700; line-height:1.35; margin:2px 0 8px; display:inline; }
.vc-gauge { font-size:12.5px; color:var(--text-secondary); margin:6px 0 2px; }
.vc-oneliner { font-size:13px; color:var(--text-secondary); margin:2px 0 0; line-height:1.5; }
.vc-hint { font-size:11px; color:var(--text-muted); margin-top:10px; }
.vc-explanation { margin-top:14px; padding-top:12px; border-top:1px dashed var(--border-ring);
                 font-size:13.5px; line-height:1.6; }
.vc-explanation p { margin:0 0 9px; }
details.exact-numbers { margin-top:8px; }
details.exact-numbers summary { font-size:11.5px; color:var(--text-muted); cursor:pointer; }
details.section-card { background:var(--surface-1); border:1px solid var(--border-ring); border-radius:10px;
                       box-shadow:var(--card-shadow); padding:18px 20px; margin-bottom:20px; }
details.section-card > summary { font-size:15px; font-weight:600; cursor:pointer; list-style:revert; }
details.section-card > summary::marker { color:var(--text-muted); }
details.section-card[open] > summary { margin-bottom:14px; }
.section-body { padding-top:2px; }
.toggle-btn { font:inherit; font-size:12px; padding:5px 10px; border-radius:6px; border:1px solid var(--border-ring);
             background:var(--page-plane); color:var(--text-primary); cursor:pointer; margin-bottom:10px; }
.extra-col { display:none; }
.table-wrap.show-extra .extra-col { display:table-cell; }
dfn { font-style:normal; border-bottom:1px dotted var(--text-muted); }
</style>
</head>
<body>

<h1 id="header-question"></h1>
<p class="header-answer" id="header-answer"></p>

<div class="card-grid-3" id="card-grid"></div>

<details class="section-card" id="details-table">
  <summary>Tableau détaillé (pour les curieux)</summary>
  <div class="section-body">
    <p class="note">Une ligne par (modèle, actif). Verdict et fiabilité en clair par défaut ; le bouton
       "Tout afficher" ajoute les colonnes techniques (précision/fiabilité exactes de chaque côté,
       <dfn title="Winkler / Interval Score : fiabilité de la fourchette de prix">Winkler</dfn>,
       <dfn title="à quelle fréquence le vrai prix tombe dans la fourchette (on vise 95%)">Cov95</dfn>,
       <dfn title="quantité de recul réellement exploitable, après prise en compte de l'autocorrélation entre semaines">effective_n</dfn>…).
       Cliquer un en-tête trie ; cliquer un verdict déplie son explication.</p>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch" style="background:var(--d7-color)"></span>Le quotidien est meilleur</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--w1-color)"></span>L'hebdomadaire est meilleur</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--tie-color)"></span>Match nul</span>
    </div>
    <div class="table-wrap" id="cell-table-wrap">
      <button class="toggle-btn" id="toggle-extra-cols" type="button">Tout afficher (colonnes techniques)</button>
      <table id="cell-table"><thead></thead><tbody></tbody></table>
    </div>
  </div>
</details>

<details class="section-card">
  <summary>Semaine par semaine, qui a gagné ?</summary>
  <div class="section-body">
    <p class="note">Pour un (modèle, actif) choisi, l'écart de précision et la largeur de la fourchette de
       prix, semaine après semaine.</p>
    <div class="controls-row">
      <label for="cell-select" class="note" style="margin:0;">Modèle / actif :</label>
      <select id="cell-select" class="select-box"></select>
    </div>
    <div class="two-col">
      <div>
        <h3>Qui a été le plus précis, semaine par semaine ?</h3>
        <p class="note" style="margin-top:-4px;">Bleu = l'hebdomadaire a été plus précis cette semaine-là · orange = le quotidien a été plus précis.</p>
        <div id="traj-sqerror"></div>
      </div>
      <div>
        <h3>Largeur de la fourchette de prix, semaine par semaine</h3>
        <p class="note" style="margin-top:-4px;">Barre orange = quotidien, barre bleue = hebdomadaire (échelle propre à la ligne choisie).</p>
        <div id="traj-piwidth"></div>
      </div>
    </div>
  </div>
</details>

<details class="section-card">
  <summary>Les fourchettes sont-elles bien calibrées ?</summary>
  <div class="section-body">
    <p class="note">On vise une fourchette qui contient le vrai prix 95% du temps (trait vertical). Une barre
       plus courte que 95% = fourchette trop optimiste (trop souvent hors cible) ; une barre qui va bien
       au-delà de 95% = fourchette probablement trop large.</p>
    <div id="calibration-panel"></div>
  </div>
</details>

<details class="section-card">
  <summary>Chiffres agrégés détaillés (les 4 groupes)</summary>
  <div class="section-body">
    <p class="note">Mêmes 4 groupes que les cartes du haut (+ "Global" toutes classes confondues), avec les
       chiffres exacts du test statistique de chaque axe (précision / fiabilité de la fourchette).</p>
    <div class="class-grid" id="aggregate-grid"></div>
  </div>
</details>

<details class="section-card" id="details-methodo">
  <summary>Comment on a calculé ça ? (définitions)</summary>
  <div class="section-body">
    <p class="note" id="footer-meta"></p>
    <p><b>Comparaison.</b> Un même modèle est entraîné deux fois : une fois en <b>quotidien</b> (il apprend
       sur des données jour par jour, puis on regarde sa prévision à 1 semaine), une fois en
       <b>hebdomadaire</b> (il apprend directement sur des données semaine par semaine). On compare les deux
       sur exactement les mêmes semaines de test, jamais vues à l'entraînement.</p>
    <p><b>Précision (<dfn title="Root Mean Squared Error : racine de l'erreur quadratique moyenne">RMSE</dfn>).</b>
       À quel point le prix prévu est proche du prix réel, en moyenne sur toutes les semaines de test. Plus
       petit = meilleur. C'est ce chiffre qui décide du verdict affiché par défaut.</p>
    <p><b>Fiabilité de la fourchette (<dfn title="Winkler / Interval Score (Gneiting &amp; Raftery 2007) : pénalise une fourchette large, et beaucoup plus encore une fourchette qui rate le vrai prix">Winkler</dfn> +
       <dfn title="à quelle fréquence le vrai prix tombe dans la fourchette annoncée ; la cible est 95%">Cov95</dfn>).</b>
       Une bonne fourchette contient le vrai prix environ 95% du temps, <i>sans</i> être inutilement large.
       Le Winkler condense ces deux exigences (largeur + taux de réussite) en un seul chiffre ; plus petit = meilleur.</p>
    <p><b>Recul réellement exploitable (<dfn title="nombre de blocs de 3 semaines consécutives quasi indépendants, après prise en compte du chevauchement des semaines de test">effective_n</dfn>).</b>
       On a <span id="footer-n-weeks"></span> semaines de test, mais elles se chevauchent partiellement (une
       semaine influence un peu la suivante) : le recul <i>réellement</i> indépendant est plus proche de
       <code id="footer-eff-n-example"></code> "blocs" de semaines -- d'où la jauge 🔴 (&lt;15) 🟠 (15-40) 🟢 (&gt;40)
       plutôt qu'un simple nombre de semaines.</p>
    <p><b>Modèle de référence (marche aléatoire).</b> Les groupes "Crypto / Actions / Obligations" comparent en
       fait un <i>score par rapport à une prévision naïve</i> ("le prix ne bouge pas", ajusté à l'incertitude
       historiquement observée) plutôt que l'erreur brute -- ça permet de regrouper des actifs à des échelles
       de prix très différentes (BTC ~ 100 000 $, ZN ~ 100 $) dans un même verdict. ZN=F et TLT (deux façons de
       parier sur les taux américains, très corrélées) sont fusionnées en une seule voix "obligations" pour ne
       pas compter deux fois le même pari.</p>
    <p><b>Comment le verdict est tranché.</b> On ne déclare un gagnant que si l'écart, testé statistiquement
       (test bootstrap par blocs, qui tient compte du chevauchement entre semaines), exclut clairement le
       "zéro différence". Sinon c'est un <b>match nul</b> assumé -- même si un chiffre penche légèrement d'un
       côté, ce n'est pas un manque de rigueur, c'est le résultat honnête.</p>
    <p><b>Limites.</b> Sur les groupes Crypto/Actions/Obligations, un seul modèle très divergent d'un côté peut
       influencer le verdict du groupe entier (moyenne par semaine sur tous les modèles) -- croiser avec le
       tableau détaillé par modèle si un verdict de groupe surprend. La corrélation entre BTC-USD et ETH-USD
       n'est pas neutralisée (contrairement à la paire ZN=F/TLT).</p>
  </div>
</details>

<script>
const DATA = __DATA_JSON__;

function fmtNum(v, d) { return (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d); }
function fmtPct(v) { return (v === null || v === undefined) ? "—" : (Number(v) * 100).toFixed(1) + "%"; }

// Le test par cellule (comparison_3_daily_vs_weekly) et le test poolé (run_pooled_test,
// dashboard_d7_w1.py) ne nomment pas le verdict "daily gagne" à l'identique
// ("daily_multistep_significantly_better" vs "daily_significantly_better") -- les deux
// formes sont donc acceptées ici pour que le badge par cellule ne retombe pas à tort
// sur "indistinguable".
function isDailyWin(verdict) { return verdict === "daily_significantly_better" || verdict === "daily_multistep_significantly_better"; }
function isWeeklyWin(verdict) { return verdict === "weekly_native_significantly_better"; }

function verdictBadge(status, verdict) {
  if (status !== "tested") return '<span class="badge na">n insuffisant</span>';
  if (isDailyWin(verdict)) return '<span class="badge d7">Daily significativement meilleur</span>';
  if (isWeeklyWin(verdict)) return '<span class="badge w1">Weekly natif significativement meilleur</span>';
  return '<span class="badge tie">indistinguable</span>';
}

function leaningBadgeClass(leaning) { return leaning === "daily" ? "d7" : leaning === "weekly" ? "w1" : "tie"; }
function leaningLabel(leaning) { return leaning === "daily" ? "Daily" : leaning === "weekly" ? "Weekly natif" : "Quasi identique"; }

// Chiffres exacts (technique) par cellule -- l'ancien rapport détaillé §6bis, inchangé,
// désormais replié derrière "voir les chiffres exacts" (niveau 3).
function renderCellExactNumbersHTML(r) {
  const rep = r.report;
  if (!rep) return '<p class="no-data">Pas de rapport disponible.</p>';
  const kpiRows = rep.kpi_readings.map(k => `<tr>
      <td>${k.label}</td><td>${k.value_daily_display}</td><td>${k.value_weekly_display}</td>
      <td><span class="badge ${leaningBadgeClass(k.leaning)}">${leaningLabel(k.leaning)}</span></td>
      <td>${k.note}</td>
    </tr>`).join("");
  return `
    <p class="report-basis">${rep.verdict_basis.text}</p>
    <table class="report-kpi-table"><thead><tr>
      <th>KPI</th><th>Daily</th><th>Weekly natif</th><th>Penche</th><th>Lecture</th>
    </tr></thead><tbody>${kpiRows}</tbody></table>
    <p class="report-arbitrage${rep.arbitrage.conflict ? " conflict" : ""}">${rep.arbitrage.text}</p>
  `;
}

// Contenu affiché au clic sur un verdict de ligne : explication en clair (niveau 2/3)
// + toggle natif "voir les chiffres exacts" (niveau 3, technique, ancien rapport §6bis).
function renderCellDetailHTML(r) {
  const plain = r.plain;
  if (!plain) return '<p class="no-data">Pas d\'explication disponible.</p>';
  const paras = plain.explanation.map(s => `<p>${s}</p>`).join("");
  return `<div class="vc-explanation">
    ${paras}
    <details class="exact-numbers">
      <summary>Voir les chiffres exacts</summary>
      ${renderCellExactNumbersHTML(r)}
    </details>
  </div>`;
}

// ---- En-tête (niveau 1 : question + réponse en clair) ----
(function renderHeader() {
  document.getElementById("header-question").textContent = DATA.plain.question;
  document.getElementById("header-answer").textContent = DATA.plain.answer;
  const gen = new Date(DATA.generated_at);
  document.getElementById("footer-meta").textContent =
    `Source : ${DATA.db_path} · horizon W+1 · généré le ${gen.toLocaleString("fr-FR")} · graine aléatoire (reproductibilité) : test global=${DATA.seed_pooled}, test par ligne=${DATA.seed_cell_tests}.`;
  document.getElementById("footer-n-weeks").textContent = (DATA.cells[0] && DATA.cells[0].n) || "—";
  document.getElementById("footer-eff-n-example").textContent = (DATA.cells[0] && DATA.cells[0].effective_n) || "—";
})();

// ---- Niveau 1 : grille de 3 cartes (crypto / actions / obligations) ----
function gaugeLine(gauge) {
  return `${gauge.emoji} ${gauge.label}`;
}
function renderVerdictCard(groupKey, agg) {
  if (!agg || agg.status !== "tested" || !agg.plain) {
    return `<details class="verdict-card"><summary>
      <div class="vc-title">${groupKey}</div>
      <span class="vc-headline">Pas assez de recul pour trancher</span>
    </summary></details>`;
  }
  const p = agg.plain;
  const exact = agg.report ? `<details class="exact-numbers">
      <summary>Voir les chiffres exacts</summary>
      ${renderAggregateExactNumbersHTML(agg.report)}
    </details>` : "";
  return `<details class="verdict-card">
    <summary>
      <div class="vc-title">${p.title}</div>
      <span class="vc-headline">${p.headline}</span>
      <div class="vc-oneliner">${p.one_liner}</div>
      <div class="vc-gauge">${gaugeLine(p.gauge)}</div>
      <div class="vc-hint">Cliquer pour le détail (pourquoi, à quel point on est sûr)</div>
    </summary>
    <div class="vc-explanation">
      ${p.explanation.map(s => `<p>${s}</p>`).join("")}
      ${exact}
    </div>
  </details>`;
}
(function renderCardGrid() {
  const g = DATA.aggregate;
  const html = renderVerdictCard("crypto", g.crypto) + renderVerdictCard("index", g.index) + renderVerdictCard("bond", g.bond);
  document.getElementById("card-grid").innerHTML = html;
})();

// ---- Niveau 3 : table triable (réduite par défaut, "tout afficher" pour le reste) ----
const CELL_COLUMNS = [
  {key:"model", label:"Modèle", extra:false},
  {key:"asset", label:"Actif", extra:false},
  {key:"__verdict", label:"Verdict", extra:false},
  {key:"__fiabilite", label:"Fiabilité", extra:false},
  {key:"asset_class", label:"Classe", extra:true},
  {key:"rmse_daily", label:"Précision Daily (RMSE)", extra:true},
  {key:"rmse_weekly", label:"Précision Weekly (RMSE)", extra:true},
  {key:"winkler_daily", label:"Fiabilité fourchette Daily (Winkler)", extra:true},
  {key:"winkler_weekly", label:"Fiabilité fourchette Weekly (Winkler)", extra:true},
  {key:"cov95_daily", label:"Couverture 95% Daily (Cov95)", extra:true},
  {key:"cov95_weekly", label:"Couverture 95% Weekly (Cov95)", extra:true},
  {key:"pi_width_daily", label:"Largeur fourchette Daily", extra:true},
  {key:"pi_width_weekly", label:"Largeur fourchette Weekly", extra:true},
  {key:"direction_daily", label:"Direction correcte Daily", extra:true},
  {key:"direction_weekly", label:"Direction correcte Weekly", extra:true},
  {key:"mean_diff", label:"Écart moyen testé", extra:true},
  {key:"p_value", label:"p (significativité)", extra:true},
  {key:"n", label:"n (semaines de test)", extra:true},
  {key:"effective_n", label:"effective_n (recul réel)", extra:true},
];
let cellSort = {key:"model", dir:1};
const expandedCells = new Set();   // clés "model||asset" dépliées -- préservées d'un tri à l'autre

function sortValue(row, key) {
  if (key === "__verdict") return row.p_value ?? 2;
  if (key === "__fiabilite") return row.effective_n ?? -1;
  return row[key];
}

function renderCellTable() {
  const thead = document.querySelector("#cell-table thead");
  thead.innerHTML = "<tr>" + CELL_COLUMNS.map(c => {
    const arrow = cellSort.key === c.key ? (cellSort.dir === 1 ? "▲" : "▼") : "";
    const cls = (cellSort.key === c.key ? "sorted " : "") + (c.extra ? "extra-col" : "");
    return `<th data-key="${c.key}" class="${cls}" data-arrow="${arrow}">${c.label}</th>`;
  }).join("") + "</tr>";
  thead.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-key");
      if (cellSort.key === key) { cellSort.dir *= -1; } else { cellSort = {key, dir:1}; }
      renderCellTable();
    });
  });

  const rows = DATA.cells.slice().sort((a, b) => {
    const va = sortValue(a, cellSort.key), vb = sortValue(b, cellSort.key);
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "string") return cellSort.dir * va.localeCompare(vb);
    return cellSort.dir * ((va > vb) - (va < vb));
  });

  const tbody = document.querySelector("#cell-table tbody");
  tbody.innerHTML = rows.map(r => {
    const key = `${r.model}||${r.asset}`;
    const expanded = expandedCells.has(key);
    const plain = r.plain || {headline: "—", gauge: {emoji:"⚪", label:"—"}};
    const outcomeCls = isDailyWin(r.verdict) ? "d7" : isWeeklyWin(r.verdict) ? "w1"
                       : (r.status === "tested" ? "tie" : "na");
    return `<tr>
      <td>${r.model}</td><td>${r.asset}</td>
      <td><button class="verdict-toggle" data-key="${key}" title="Déplier l'explication"><span class="badge ${outcomeCls}">${plain.headline}</span><span class="expand-arrow">${expanded ? "▲" : "▼"}</span></button></td>
      <td>${plain.gauge.emoji} ${plain.gauge.label.split(" -- ")[0]}</td>
      <td class="extra-col">${DATA.asset_class_label[r.asset_class] || r.asset_class}</td>
      <td class="extra-col">${fmtNum(r.rmse_daily,2)}</td><td class="extra-col">${fmtNum(r.rmse_weekly,2)}</td>
      <td class="extra-col">${fmtNum(r.winkler_daily,2)}</td><td class="extra-col">${fmtNum(r.winkler_weekly,2)}</td>
      <td class="extra-col">${fmtPct(r.cov95_daily)}</td><td class="extra-col">${fmtPct(r.cov95_weekly)}</td>
      <td class="extra-col">${fmtNum(r.pi_width_daily,2)}</td><td class="extra-col">${fmtNum(r.pi_width_weekly,2)}</td>
      <td class="extra-col">${fmtPct(r.direction_daily)}</td><td class="extra-col">${fmtPct(r.direction_weekly)}</td>
      <td class="extra-col">${fmtNum(r.mean_diff,3)}</td><td class="extra-col">${fmtNum(r.p_value,4)}</td>
      <td class="extra-col">${r.n}</td><td class="extra-col">${r.effective_n ?? "—"}</td>
    </tr>
    <tr class="detail-row" data-key="${key}" style="display:${expanded ? "table-row" : "none"};">
      <td colspan="${CELL_COLUMNS.length}">${expanded ? renderCellDetailHTML(r) : ""}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll(".verdict-toggle").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-key");
      if (expandedCells.has(key)) { expandedCells.delete(key); } else { expandedCells.add(key); }
      renderCellTable();
    });
  });
}
renderCellTable();

(function wireExtraColsToggle() {
  const wrap = document.getElementById("cell-table-wrap");
  const btn = document.getElementById("toggle-extra-cols");
  btn.addEventListener("click", () => {
    const show = wrap.classList.toggle("show-extra");
    btn.textContent = show ? "Réduire aux colonnes essentielles" : "Tout afficher (colonnes techniques)";
  });
})();

// ---- Niveau 3 : chiffres agrégés détaillés (4 tuiles, ancien contenu §6bis) ----
function verdictLabel(v) {
  if (v === "daily_significantly_better") return "Daily significativement meilleur";
  if (v === "weekly_native_significantly_better") return "Weekly natif significativement meilleur";
  return "indistinguable";
}
function badgeClass(v) {
  if (v === "daily_significantly_better") return "d7";
  if (v === "weekly_native_significantly_better") return "w1";
  return "tie";
}
function renderAggregateExactNumbersHTML(rep) {
  if (!rep) return "";
  const conflictCls = /Arbitrage/.test(rep.synthesis) ? " conflict" : "";
  return `<div class="agg-report">
    <p class="report-basis"><b>Base — skill RMSE.</b> ${rep.skill_sqerror.text}</p>
    <p class="report-basis"><b>Base — skill Winkler.</b> ${rep.skill_winkler.text}</p>
    <p class="report-arbitrage${conflictCls}">${rep.synthesis}</p>
  </div>`;
}
function renderAggregateTile(label, agg) {
  if (!agg || agg.status !== "tested") {
    return `<div class="class-tile"><div class="title">${label}</div>
      <div class="no-data">n insuffisant (n_origines=${agg ? agg.n_origins : 0})</div></div>`;
  }
  const sq = agg.skill_sqerror, wk = agg.skill_winkler;
  return `<div class="class-tile">
    <div class="title">${label}</div>
    <div>skill RMSE : <span class="badge ${badgeClass(sq.verdict)}">${verdictLabel(sq.verdict)}</span></div>
    <div class="note" style="margin:2px 0 0;">mean_diff=${fmtNum(sq.mean_diff,4)} · p=${fmtNum(sq.p_value,4)} · IC95=[${fmtNum(sq.ci95_lo,4)}, ${fmtNum(sq.ci95_hi,4)}]</div>
    <div style="margin-top:6px;">skill Winkler : <span class="badge ${badgeClass(wk.verdict)}">${verdictLabel(wk.verdict)}</span></div>
    <div class="note" style="margin:2px 0 0;">mean_diff=${fmtNum(wk.mean_diff,4)} · p=${fmtNum(wk.p_value,4)} · IC95=[${fmtNum(wk.ci95_lo,4)}, ${fmtNum(wk.ci95_hi,4)}]</div>
    <div class="n-line">n_origines=${agg.n_origins} (${agg.n_contributions} contributions pré-moyennage) · effective_n=${sq.effective_n}</div>
    ${renderAggregateExactNumbersHTML(agg.report)}
  </div>`;
}
(function renderAggregate() {
  const g = DATA.aggregate;
  const html = renderAggregateTile("Global (toutes classes)", g.global)
    + renderAggregateTile("Crypto (BTC-USD, ETH-USD)", g.crypto)
    + renderAggregateTile("Actions (SPY)", g.index)
    + renderAggregateTile("Obligations / taux (ZN=F + TLT, dédoublonnées)", g.bond);
  document.getElementById("aggregate-grid").innerHTML = html;
})();

// ---- Niveau 3 : semaine par semaine (cellule sélectionnée) ----
const cellKeys = Object.keys(DATA.trajectories).sort();
(function populateCellSelect() {
  const sel = document.getElementById("cell-select");
  sel.innerHTML = cellKeys.map(k => {
    const [model, asset] = k.split("||");
    return `<option value="${k}">${model} — ${asset}</option>`;
  }).join("");
  sel.addEventListener("change", () => renderTrajectory(sel.value));
})();

function renderSignedBars(containerId, points, valueKey) {
  const el = document.getElementById(containerId);
  if (!points || points.length === 0) { el.innerHTML = '<p class="no-data">Pas de données.</p>'; return; }
  const maxAbs = Math.max(1e-9, ...points.map(p => Math.abs(p[valueKey])));
  el.innerHTML = points.map(p => {
    const v = p[valueKey];
    const pct = Math.min(50, Math.abs(v) / maxAbs * 50);
    const fill = v >= 0
      ? `<div class="signed-fill" style="left:50%; width:${pct}%; background:var(--w1-color);"></div>`
      : `<div class="signed-fill" style="right:50%; width:${pct}%; background:var(--d7-color);"></div>`;
    return `<div class="origin-row">
      <span class="origin-label">${p.cutoff_date}</span>
      <div class="signed-track"><div class="signed-zero"></div>${fill}</div>
      <span class="origin-value">${fmtNum(v,1)}</span>
    </div>`;
  }).join("");
}

function renderPairedBars(containerId, points) {
  const el = document.getElementById(containerId);
  if (!points || points.length === 0) { el.innerHTML = '<p class="no-data">Pas de données.</p>'; return; }
  const maxW = Math.max(1e-9, ...points.map(p => Math.max(p.pi_width_daily, p.pi_width_weekly)));
  el.innerHTML = points.map(p => `<div class="origin-row" style="grid-template-columns:82px 1fr 1fr;">
      <span class="origin-label">${p.cutoff_date}</span>
      <div class="pair-track"><div class="pair-fill" style="width:${p.pi_width_daily/maxW*100}%; background:var(--d7-color);"></div></div>
      <div class="pair-track"><div class="pair-fill" style="width:${p.pi_width_weekly/maxW*100}%; background:var(--w1-color);"></div></div>
    </div>`).join("");
}

function renderTrajectory(key) {
  const points = DATA.trajectories[key] || [];
  renderSignedBars("traj-sqerror", points, "sq_error_diff");
  renderPairedBars("traj-piwidth", points);
}
if (cellKeys.length) {
  document.getElementById("cell-select").value = cellKeys[0];
  renderTrajectory(cellKeys[0]);
}

// ---- Niveau 3 : calibration ----
(function renderCalibration() {
  const el = document.getElementById("calibration-panel");
  const rows = DATA.cells.slice().sort((a,b) => a.model.localeCompare(b.model) || a.asset.localeCompare(b.asset));
  el.innerHTML = rows.map(r => `
    <div class="origin-row" style="grid-template-columns:170px 1fr 1fr;">
      <span class="origin-label">${r.model} — ${r.asset}</span>
      <div class="cal-track"><div class="cal-fill" style="width:${Math.min(100,r.cov95_daily*100)}%; background:var(--d7-color);"></div><div class="cal-target" style="left:95%;"></div></div>
      <div class="cal-track"><div class="cal-fill" style="width:${Math.min(100,r.cov95_weekly*100)}%; background:var(--w1-color);"></div><div class="cal-target" style="left:95%;"></div></div>
    </div>`).join("");
})();
</script>
</body>
</html>
"""
