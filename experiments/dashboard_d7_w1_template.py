"""
dashboard_d7_w1_template.py — le HTML/CSS/JS statique de experiments/dashboard_d7_w1.py.

Séparé du générateur pour lisibilité seulement (aucune logique ici, aucun calcul).
100% autonome : pas de <script src>, pas de fetch(), rien qui requiert le réseau
pour ouvrir la page en file://. Volontairement SANS librairie de graphes
(pas de Plotly) : les graphiques (barres signées par origine, calibration) sont
de simples <div> dimensionnés en CSS, générés/mis à jour par un JS vanille
minimal, sans qu'aucun graphique n'écoute un autre -- élimine structurellement
la classe de bug du CORRECTIF_dashboard_v4_boucle_infinie.md.

__DATA_JSON__ est remplacé par le générateur avec le payload JSON complet
(cellules, trajectoires, agrégats, config).
"""

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Daily vs Weekly natif — horizon W+1</title>
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
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:15px; margin:0 0 12px; }
h3 { font-size:13px; margin:0 0 8px; color:var(--text-secondary); }
.subtitle { color:var(--text-secondary); font-size:13px; margin:0 0 4px; }
.meta-line { color:var(--text-muted); font-size:12px; margin:2px 0 20px; }
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
</style>
</head>
<body>

<h1>Daily vs Weekly natif — horizon W+1</h1>
<p class="subtitle">Pour prévoir 1 semaine, vaut-il mieux le modèle daily (régime B, évalué à son horizon
   natif W+1) ou le modèle weekly natif (régime C) ? Comparaison intra-modèle, appariée exactement
   (même origine, même cible), sur les 6 modèles × 5 actifs.</p>
<p class="meta-line" id="header-meta"></p>

<div class="card">
  <h2>Verdict par cellule (model × asset)</h2>
  <p class="note">Test = différence d'erreur quadratique (RMSE), bootstrap par blocs, seed=<span id="seed-cell"></span>.
     Winkler/Cov95/largeur PI/direction affichés à titre descriptif. <code>n</code>/<code>effective_n</code>
     indiqués par cellule. Cliquer un en-tête pour trier.</p>
  <div class="legend">
    <span class="legend-item"><span class="legend-swatch" style="background:var(--d7-color)"></span>Daily significativement meilleur</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--w1-color)"></span>Weekly natif significativement meilleur</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--tie-color)"></span>indistinguable (p ≥ 0,05)</span>
  </div>
  <div class="table-wrap"><table id="cell-table"><thead></thead><tbody></tbody></table></div>
</div>

<div class="card">
  <h2>Agrégat poolé — skill-score sans échelle vs baseline RW</h2>
  <p class="note">skill = 1 − score_modèle / score_RW (RW : point = dernier close, PI = quantiles empiriques
     des rendements cumulés à l'horizon W+1, fenêtre ≤ origine). Diff pooled = skill(daily) − skill(weekly),
     moyennée par origine, testée par bootstrap par blocs. ZN=F et TLT (corrélées) sont dédoublonnées en une
     contribution "taux" avant pooling. Jamais de RMSE/Winkler absolu comparé entre actifs, seul le skill
     sans échelle est poolé.</p>
  <div class="class-grid" id="aggregate-grid"></div>
</div>

<div class="card">
  <h2>Trajectoires par origine</h2>
  <div class="controls-row">
    <label for="cell-select" class="note" style="margin:0;">Cellule :</label>
    <select id="cell-select" class="select-box"></select>
  </div>
  <div class="two-col">
    <div>
      <h3>Différence d'erreur quadratique par origine (daily − weekly)</h3>
      <p class="note" style="margin-top:-4px;">Positif (bleu) = weekly natif a une erreur plus faible à cette origine · négatif (orange) = daily a une erreur plus faible.</p>
      <div id="traj-sqerror"></div>
    </div>
    <div>
      <h3>Largeur de l'intervalle 95% par origine</h3>
      <p class="note" style="margin-top:-4px;">Barre orange = daily, barre bleue = weekly natif (échelle propre à la cellule).</p>
      <div id="traj-piwidth"></div>
    </div>
  </div>
</div>

<div class="card">
  <h2>Calibration — Cov95 réelle vs cible 0,95</h2>
  <p class="note">Trait vertical = cible 0,95.</p>
  <div id="calibration-panel"></div>
</div>

<footer class="card">
  <h2>Formules, définitions, limites</h2>
  <p><b>Appariement.</b> Regime B (frequence=daily, modèle daily évalué à son horizon natif W+1) vs
     regime C (frequence=weekly, weekly natif), apparié par <code>target_date</code> — les deux côtés
     partagent le même <code>cutoff_date</code> par construction (vérifié 100%), donc pas d'approximation
     d'horizon ni de restriction aux origines-vendredi : ~30 paires/cellule, <code>effective_n</code>~10
     (bootstrap par blocs, block_length=3). Comparaison intra-modèle (même modèle des deux côtés).</p>
  <p><b>Winkler / Interval Score @95%</b> (Gneiting &amp; Raftery 2007) : pour une cible <code>y</code>,
     un intervalle <code>[l, u]</code> et <code>alpha=0,05</code> :</p>
  <p style="font-family:ui-monospace,monospace; font-size:12px;">
    IS = (u − l) + (2/alpha)·(l − y) si y &lt; l<br>
    IS = (u − l) + (2/alpha)·(y − u) si y &gt; u<br>
    IS = (u − l) sinon (y dans l'intervalle)
  </p>
  <p><b>Baseline random walk (RW).</b> Point = dernier close au cutoff (persistance). PI 95% = dernier
     close × (1 + quantile [2,5%, 97,5%]) des rendements cumulés observés à l'horizon réel de la cible,
     sur tout l'historique de prix disponible à la date ≤ cutoff (fenêtre expansive, aucune fuite de
     future). Nécessite au moins <span id="footer-min-rw"></span> rendements historiques valides ;
     historique de prix commençant au plus tôt le <span id="footer-price-start"></span> (yfinance).</p>
  <p><b>Pas de vrai CRPS.</b> La DB ne stocke que <code>(y_pred, y_lower, y_upper, y_true)</code>, pas
     d'échantillons : le Winkler est la métrique probabiliste utilisée ici.</p>
  <p><b>Limites.</b> Le pooling inter-actifs/inter-modèles moyenne par origine ; un modèle très divergent
     d'un côté sur une longue période peut influencer le verdict global/par-classe — croiser avec la table
     par cellule. La corrélation résiduelle entre BTC-USD/ETH-USD n'est pas neutralisée (seule la paire
     ZN=F/TLT l'est).</p>
  <p>Source : <code id="footer-db-path"></code> · généré le <span id="footer-generated-at"></span> ·
     seed test poolé = <span id="footer-seed-pooled"></span> · seed test par cellule = <span id="footer-seed-cell"></span>
     (fixe, hérité de <code>comparison_3_daily_vs_weekly</code>).</p>
</footer>

<script>
const DATA = __DATA_JSON__;

function fmtNum(v, d) { return (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d); }
function fmtPct(v) { return (v === null || v === undefined) ? "—" : (Number(v) * 100).toFixed(1) + "%"; }

function verdictBadge(status, verdict) {
  if (status !== "tested") return '<span class="badge na">n insuffisant</span>';
  if (verdict === "daily_significantly_better") return '<span class="badge d7">Daily significativement meilleur</span>';
  if (verdict === "weekly_native_significantly_better") return '<span class="badge w1">Weekly natif significativement meilleur</span>';
  return '<span class="badge tie">indistinguable</span>';
}

// ---- En-tête ----
(function renderHeader() {
  const gen = new Date(DATA.generated_at);
  document.getElementById("header-meta").textContent =
    `Source ${DATA.db_path} · horizon ${DATA.horizon_unit} · généré le ${gen.toLocaleString("fr-FR")} · seed poolé=${DATA.seed_pooled} · seed cellule=${DATA.seed_cell_tests}`;
  document.getElementById("seed-cell").textContent = DATA.seed_cell_tests;
  document.getElementById("footer-min-rw").textContent = DATA.min_rw_quantile_samples;
  document.getElementById("footer-price-start").textContent = DATA.price_history_start;
  document.getElementById("footer-db-path").textContent = DATA.db_path;
  document.getElementById("footer-generated-at").textContent = gen.toLocaleString("fr-FR");
  document.getElementById("footer-seed-pooled").textContent = DATA.seed_pooled;
  document.getElementById("footer-seed-cell").textContent = DATA.seed_cell_tests;
})();

// ---- Panneau 2 : table triable ----
const CELL_COLUMNS = [
  {key:"model", label:"Modèle"}, {key:"asset", label:"Actif"}, {key:"asset_class", label:"Classe"},
  {key:"rmse_daily", label:"RMSE Daily"}, {key:"rmse_weekly", label:"RMSE Weekly"},
  {key:"winkler_daily", label:"Winkler Daily"}, {key:"winkler_weekly", label:"Winkler Weekly"},
  {key:"cov95_daily", label:"Cov95 Daily"}, {key:"cov95_weekly", label:"Cov95 Weekly"},
  {key:"pi_width_daily", label:"Larg. PI Daily"}, {key:"pi_width_weekly", label:"Larg. PI Weekly"},
  {key:"direction_daily", label:"Dir. Daily"}, {key:"direction_weekly", label:"Dir. Weekly"},
  {key:"mean_diff", label:"mean_diff"}, {key:"p_value", label:"p"},
  {key:"n", label:"n"}, {key:"effective_n", label:"eff_n"}, {key:"verdict", label:"Verdict"},
];
let cellSort = {key:"model", dir:1};

function renderCellTable() {
  const thead = document.querySelector("#cell-table thead");
  thead.innerHTML = "<tr>" + CELL_COLUMNS.map(c => {
    const arrow = cellSort.key === c.key ? (cellSort.dir === 1 ? "▲" : "▼") : "";
    const cls = cellSort.key === c.key ? ' class="sorted"' : "";
    return `<th data-key="${c.key}"${cls} data-arrow="${arrow}">${c.label}</th>`;
  }).join("") + "</tr>";
  thead.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-key");
      if (cellSort.key === key) { cellSort.dir *= -1; } else { cellSort = {key, dir:1}; }
      renderCellTable();
    });
  });

  const rows = DATA.cells.slice().sort((a, b) => {
    const va = a[cellSort.key], vb = b[cellSort.key];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "string") return cellSort.dir * va.localeCompare(vb);
    return cellSort.dir * ((va > vb) - (va < vb));
  });

  const tbody = document.querySelector("#cell-table tbody");
  tbody.innerHTML = rows.map(r => `<tr>
    <td>${r.model}</td><td>${r.asset}</td><td>${DATA.asset_class_label[r.asset_class] || r.asset_class}</td>
    <td>${fmtNum(r.rmse_daily,2)}</td><td>${fmtNum(r.rmse_weekly,2)}</td>
    <td>${fmtNum(r.winkler_daily,2)}</td><td>${fmtNum(r.winkler_weekly,2)}</td>
    <td>${fmtPct(r.cov95_daily)}</td><td>${fmtPct(r.cov95_weekly)}</td>
    <td>${fmtNum(r.pi_width_daily,2)}</td><td>${fmtNum(r.pi_width_weekly,2)}</td>
    <td>${fmtPct(r.direction_daily)}</td><td>${fmtPct(r.direction_weekly)}</td>
    <td>${fmtNum(r.mean_diff,3)}</td><td>${fmtNum(r.p_value,4)}</td>
    <td>${r.n}</td><td>${r.effective_n ?? "—"}</td>
    <td>${verdictBadge(r.status, r.verdict)}</td>
  </tr>`).join("");
}
renderCellTable();

// ---- Panneau 3 : agrégat poolé ----
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

// ---- Panneau 4 : trajectoires par origine (cellule sélectionnée) ----
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

// ---- Panneau 5 : calibration ----
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
