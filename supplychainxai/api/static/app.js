/* SupplyChainXAI dashboard — vanilla JS, ECharts. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (n, d = 0) => Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const charts = {};
function chart(id) {
  if (charts[id]) return charts[id];
  charts[id] = echarts.init($("#" + id), "dark");
  window.addEventListener("resize", () => charts[id].resize());
  return charts[id];
}
const axisStyle = { axisLine: { lineStyle: { color: "#2a3247" } }, axisLabel: { color: "#8b93a7" }, splitLine: { lineStyle: { color: "#1c2333" } } };

/* ================= tabs ================= */
$$(".nav-btn").forEach((b) => b.addEventListener("click", () => {
  $$(".nav-btn").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  $$(".tab").forEach((t) => t.classList.remove("active"));
  $("#tab-" + b.dataset.tab).classList.add("active");
  Object.values(charts).forEach((c) => c.resize());
  if (b.dataset.tab === "simulator" && !simLoaded) runSim();
}));

/* ================= overview ================= */
async function loadOverview() {
  const [kpis, models, risk, inv] = await Promise.all([
    api("/api/kpis"), api("/api/models"), api("/api/risk"), api("/api/inventory")]);

  $("#snapshot").textContent = "snapshot " + kpis.snapshot_date;
  const alerts = risk.alerts;
  const crit = alerts.filter((a) => a.severity === "CRITICAL").length;

  $("#kpi-grid").innerHTML = [
    kpi("Forecast demand (30d)", fmt(kpis.forecast_30d_demand) + " u", "all SKUs"),
    kpi("Inventory value", "$" + fmt(kpis.inventory_value), "on-hand at cost"),
    kpi("Buy actions", kpis.buy_actions, fmt(kpis.buy_units) + " u · $" + fmt(kpis.buy_cost)),
    kpi("Supplier OTD", pct(kpis.supplier_on_time_pct / 100), "delivered POs"),
    kpi("Open POs", kpis.open_po_count, "$" + fmt(kpis.open_po_spend) + " committed"),
    kpi("Risk alerts", alerts.length, crit + " critical", crit ? "bad" : "ok"),
  ].join("");

  const m = models.by_model.slice().reverse();
  chart("chart-models").setOption({
    backgroundColor: "transparent",
    grid: { left: 150, right: 30, top: 10, bottom: 26 },
    xAxis: { type: "value", ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: "{value}%" } },
    yAxis: { type: "category", data: m.map((x) => x.model), ...axisStyle },
    tooltip: { trigger: "axis" },
    series: [{ type: "bar", data: m.map((x) => x.mean_wape), barWidth: 16,
      itemStyle: { color: (p) => (p.dataIndex === m.length - 1 ? "#2dd4a7" : "#3d465e"), borderRadius: 4 },
      label: { show: true, position: "right", color: "#8b93a7", formatter: "{c}%" } }],
  });

  const invRows = inv.items;
  const byCat = {};
  invRows.forEach((r) => { byCat[r.category] = (byCat[r.category] || 0) + r.inventory_value; });
  const cats = Object.entries(byCat).sort((a, b) => b[1] - a[1]);
  chart("chart-spend").setOption({
    backgroundColor: "transparent",
    grid: { left: 130, right: 40, top: 10, bottom: 26 },
    xAxis: { type: "value", ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v) => "$" + (v / 1000) + "k" } },
    yAxis: { type: "category", data: cats.map((c) => c[0]), ...axisStyle },
    tooltip: { trigger: "axis", valueFormatter: (v) => "$" + fmt(v) },
    series: [{ type: "bar", data: cats.map((c) => Math.round(c[1])), barWidth: 16,
      itemStyle: { color: "#f5b342", borderRadius: 4 } }],
  });

  $("#model-narrative").textContent = models.narrative.narrative;
  $("#overview-alerts").innerHTML = alerts.slice(0, 4).map(alertCard).join("");
}
const kpi = (label, value, sub, cls = "") =>
  `<div class="kpi"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub}</div></div>`;

/* ================= forecast ================= */
let skuList = [];
async function loadForecast() {
  if (!skuList.length) {
    const inv = await api("/api/inventory");
    skuList = inv.items.map((i) => i.sku);
    $("#sku-select").innerHTML = skuList.map((s) => `<option>${s}</option>`).join("");
    $("#sku-select").addEventListener("change", () => renderForecast($("#sku-select").value));
  }
  renderForecast($("#sku-select").value || skuList[0]);
}

async function renderForecast(sku) {
  const [fc, expl] = await Promise.all([api(`/api/forecast/${sku}`), api(`/api/recommendations/${sku}/explain`)]);

  const histDates = fc.history.map((h) => h.date);
  const lastHist = histDates[histDates.length - 1];
  const pad = new Array(histDates.length - 1).fill(null);   // align to x categories
  const fPred = [...pad, fc.history.at(-1).units, ...fc.forecast.map((f) => f.prediction)];
  const fUp = [...pad, null, ...fc.forecast.map((f) => f.upper_80)];
  const fLo = [...pad, null, ...fc.forecast.map((f) => f.lower_80)];

  chart("chart-forecast").setOption({
    backgroundColor: "transparent",
    grid: { left: 55, right: 25, top: 30, bottom: 50 },
    legend: { data: ["actual", "forecast"], textStyle: { color: "#8b93a7" }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: [...histDates, ...fc.forecast.map((f) => f.date)], ...axisStyle },
    yAxis: { type: "value", name: "units/day", nameTextStyle: { color: "#8b93a7" }, ...axisStyle },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 8, borderColor: "#2a3247" }],
    series: [
      { name: "actual", type: "line", data: fc.history.map((h) => h.units), symbol: "none",
        lineStyle: { color: "#8b93a7", width: 1.4 } },
      { name: "forecast", type: "line", data: fPred, symbol: "none",
        lineStyle: { color: "#2dd4a7", width: 2, type: "dashed" },
        areaStyle: { color: "rgba(45,212,167,0.06)" } },
      { name: "80% PI upper", type: "line", data: fUp, symbol: "none", stack: "pi",
        lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false } },
      { name: "80% PI band", type: "line", data: fLo, symbol: "none", stack: "pi",
        lineStyle: { opacity: 0 }, areaStyle: { color: "rgba(45,212,167,0.12)" },
        tooltip: { show: false } },
    ],
  });

  const best = fc.model_comparison[0];
  $("#table-models").innerHTML = `
    <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>WAPE</th></tr></thead>
    <tbody>${fc.model_comparison.map((m, i) => `
      <tr class="${i === 0 ? "best" : ""}"><td>${esc(m.model)}</td>
      <td class="num">${m.MAE}</td><td class="num">${m.RMSE}</td>
      <td class="num">${m.MAPE}%</td><td class="num">${m.WAPE}%</td></tr>`).join("")}
    </tbody>`;

  const e = expl.forecast_explanation || {};
  $("#forecast-explain").innerHTML = `
    <p class="narrative">${esc(e.narrative || fc.model + " selected by hold-out WAPE.")}</p>
    ${e.top_features?.length ? `<div class="components" style="margin-top:10px">
      ${e.top_features.map((f) => `
        <div class="comp"><span>${esc(f.feature)}</span>
          <div class="cbar"><i style="width:${Math.min(100, f.share * 100)}%"></i></div>
          <span class="num">${pct(f.share, 0)}</span></div>`).join("")}
    </div>` : ""}`;
}

/* ================= inventory ================= */
async function loadInventory() {
  const inv = await api("/api/inventory");
  const maxDoh = Math.max(...inv.items.map((i) => i.days_of_cover), 30);
  const pill = (s) => ({ OK: "pill-ok", LOW: "pill-low", CRITICAL: "pill-crit", OVERSTOCK: "pill-over" }[s]);
  $("#table-inventory").innerHTML = `
    <thead><tr><th>SKU</th><th>Product</th><th>Category</th><th>On hand</th><th>On order</th>
    <th>Daily fc</th><th>Days of cover</th><th></th><th>Value</th><th>Status</th></tr></thead>
    <tbody>${inv.items.map((i) => `
      <tr><td><b>${i.sku}</b></td><td>${esc(i.name)}</td><td>${esc(i.category)}</td>
      <td class="num">${fmt(i.on_hand)}</td><td class="num">${fmt(i.on_order)}</td>
      <td class="num">${i.daily_forecast}</td>
      <td class="num">${i.days_of_cover}</td>
      <td class="bar-cell"><div class="bar" style="width:${Math.min(100, (i.days_of_cover / maxDoh) * 100)}%;
        background:${i.status === "CRITICAL" ? "#f0616d" : i.status === "LOW" ? "#f5b342" : i.status === "OVERSTOCK" ? "#8b93a7" : "#2dd4a7"}"></div></td>
      <td class="num">$${fmt(i.inventory_value)}</td>
      <td><span class="pill ${pill(i.status)}">${i.status}</span></td></tr>`).join("")}
    </tbody>`;
}

/* ================= suppliers ================= */
async function loadSuppliers() {
  const data = await api("/api/suppliers");
  $("#supplier-cards").innerHTML = data.suppliers.map((s) => `
    <div class="sup-card">
      <div class="head"><span class="name">${esc(s.name)} <span class="cty">(${s.supplier_id} · ${esc(s.country)})</span></span>
        <span class="pill ${s.on_time_pct == null ? "pill-over" : s.on_time_pct >= 70 ? "pill-ok" : s.on_time_pct >= 55 ? "pill-low" : "pill-crit"}">${s.on_time_pct == null ? "no POs yet" : "OTD " + s.on_time_pct + "%"}</span></div>
      <div class="sup-stats">
        <div><div class="k">POs</div><div class="v">${fmt(s.pos)}</div></div>
        <div><div class="k">Avg lead</div><div class="v">${s.avg_lead_days == null ? "—" : s.avg_lead_days + "d"}</div></div>
        <div><div class="k">Max drift</div><div class="v ${s.max_drift_pct >= 20 ? "delta-down" : ""}">+${s.max_drift_pct}%</div></div>
        <div><div class="k">Spend</div><div class="v">$${fmt(s.spend / 1000)}k</div></div>
      </div>
    </div>`).join("");

  chart("chart-supplier-drift").setOption({
    backgroundColor: "transparent",
    grid: { left: 110, right: 30, top: 30, bottom: 26 },
    legend: { textStyle: { color: "#8b93a7" }, top: 0 },
    tooltip: { trigger: "axis", valueFormatter: (v) => v + "d" },
    xAxis: { type: "value", ...axisStyle },
    yAxis: { type: "category", data: data.suppliers.map((s) => s.supplier_id), ...axisStyle },
    series: [
      { name: "baseline lead", type: "bar", barWidth: 12, itemStyle: { color: "#3d465e", borderRadius: 3 },
        data: data.suppliers.map((s) => s.avg_lead_baseline) },
      { name: "recent lead", type: "bar", barWidth: 12, itemStyle: { color: "#f5b342", borderRadius: 3 },
        data: data.suppliers.map((s) => s.avg_lead_days) },
    ],
  });
}

/* ================= risk ================= */
const alertCard = (a) => `
  <div class="alert ${a.severity === "CRITICAL" ? "crit" : "warn"}">
    <div class="head"><span class="type">${esc(a.type)}</span><span class="type">${esc(a.sku || a.supplier_id || "")}</span>
      <span class="pill ${a.severity === "CRITICAL" ? "pill-crit" : "pill-low"}">${a.severity}</span></div>
    <div class="msg">${esc(a.message)}</div>
    <div class="data">${Object.entries(a.data || {}).slice(0, 5)
      .map(([k, v]) => `<span>${esc(k)}: ${esc(typeof v === "number" ? Math.round(v * 1000) / 1000 : v)}</span>`).join("")}</div>
  </div>`;

async function loadRisk() {
  const risk = await api("/api/risk");
  $("#risk-list").innerHTML = risk.alerts.map(alertCard).join("") || "<p class='hint'>No active alerts.</p>";
}

/* ================= recommendations ================= */
async function loadRecs() {
  const data = await api("/api/recommendations");
  $("#rec-list").innerHTML = data.recommendations.map((r, idx) => `
    <div class="rec" data-sku="${r.sku}">
      <div class="row1">
        <span class="pill ${r.action === "HOLD" ? "pill-over" : r.action === "EXPEDITE" ? "pill-crit" : "pill-ok"}">${r.action}</span>
        <span class="sku">${r.sku} · ${esc(r.product_name)}</span>
        ${r.action !== "HOLD" ? `<span class="qty">BUY ${fmt(r.quantity)} u</span>
        <span class="meta">from <b>${esc(r.supplier_id)}</b> @ $${r.unit_price} · total $${fmt(r.total_cost)} · order by ${r.order_by}</span>
        <span class="meta">stockout ~${r.expected_stockout_date} · cover ${r.coverage_days}d</span>` :
        `<span class="meta">coverage ${r.coverage_days}d — no action needed</span>`}
      </div>
      ${r.narrative ? `<div class="why">${esc(r.narrative)}</div>` : ""}
      ${r.components?.length ? `<div class="components">${r.components.slice(0, 4).map((c) => `
        <div class="comp ${c.direction === "decrease" ? "dec" : ""}">
          <span>${esc(c.factor)}</span>
          <div class="cbar"><i style="width:${Math.min(100, c.impact_pct)}%"></i></div>
          <span class="num">${c.units > 0 ? "+" : ""}${fmt(c.units)} u · ${c.impact_pct}%</span></div>`).join("")}
      </div>` : ""}
    </div>`).join("");
  $$("#rec-list .rec").forEach((el) => el.addEventListener("click", () => showRecDetail(el.dataset.sku)));
}

async function showRecDetail(sku) {
  const expl = await api(`/api/recommendations/${sku}/explain`);
  $("#rec-detail-card").hidden = false;
  $("#rec-detail-title").textContent = `Why — ${sku}`;
  $("#rec-detail").innerHTML = `
    <p class="narrative">${esc(expl.explanation.narrative || "")}</p>
    ${expl.feature_importance?.length ? `<h3 style="margin-top:12px">Forecast model drivers (permutation importance)</h3>
      <div class="components">${expl.feature_importance.map((f) => `
        <div class="comp"><span>${esc(f.feature)}</span>
        <div class="cbar"><i style="width:${Math.min(100, f.importance * 100)}%"></i></div>
        <span class="num">${f.importance.toFixed(3)}</span></div>`).join("")}</div>` : ""}`;
  $("#rec-detail-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ================= simulator ================= */
let simLoaded = false, simTimer = null;
function runSim() {
  const p = {
    demand_pct: +$("#sim-demand").value / 100,
    lead_time_pct: +$("#sim-lead").value / 100,
    inventory_pct: +$("#sim-inv").value / 100,
    service_level: +$("#sim-sl").value,
  };
  $("#out-demand").textContent = (p.demand_pct >= 0 ? "+" : "") + Math.round(p.demand_pct * 100) + "%";
  $("#out-lead").textContent = (p.lead_time_pct >= 0 ? "+" : "") + Math.round(p.lead_time_pct * 100) + "%";
  $("#out-inv").textContent = (p.inventory_pct >= 0 ? "+" : "") + Math.round(p.inventory_pct * 100) + "%";
  $("#out-sl").textContent = Math.round(p.service_level * 100) + "%";

  clearTimeout(simTimer);
  $("#sim-summary").innerHTML = `<div class="kpi"><div class="label">recomputing plan…</div></div>`;
  simTimer = setTimeout(async () => {
    const res = await post("/api/simulate", p);
    const s = res.summary;
    $("#sim-summary").innerHTML = [
      kpi("Procurement required", fmt(s.procurement_units) + " u", "scenario total"),
      kpi("Procurement cost", "$" + fmt(s.procurement_cost), "at scenario prices"),
      kpi("Buy actions", s.buy_actions, "SKUs below reorder point"),
      kpi("Avg stockout prob.", pct(s.avg_stockout_probability), "portfolio, over lead time",
        s.avg_stockout_probability > 0.4 ? "bad" : s.avg_stockout_probability > 0.2 ? "warn" : "ok"),
      kpi("Avg coverage", s.avg_coverage_days + "d", "post-replenishment"),
    ].join("");

    $("#table-sim").innerHTML = `
      <thead><tr><th>SKU</th><th>Action</th><th>Qty</th><th>Cost</th><th>Supplier</th>
      <th>Stockout P.</th><th>Cover</th></tr></thead>
      <tbody>${res.per_sku.map((r) => {
        const dp = r.stockout_probability - r.stockout_probability_baseline;
        return `<tr><td><b>${r.sku}</b></td>
        <td><span class="pill ${r.action === "HOLD" ? "pill-over" : "pill-ok"}">${r.action}</span></td>
        <td class="num">${fmt(r.quantity)}</td><td class="num">$${fmt(r.cost)}</td>
        <td>${esc(r.supplier_id || "—")}${r.supplier_id_baseline && r.supplier_id !== r.supplier_id_baseline
          ? ` <span class="delta-up">(was ${esc(r.supplier_id_baseline)})</span>` : ""}</td>
        <td class="num">${pct(r.stockout_probability)} ${dp > 0.02 ? `<span class="delta-down">+${pct(dp)}</span>`
          : dp < -0.02 ? `<span class="delta-up">${pct(dp)}</span>` : ""}</td>
        <td class="num">${r.coverage_days}d</td></tr>`;
      }).join("")}</tbody>`;
    simLoaded = true;
  }, 220);
}
["sim-demand", "sim-lead", "sim-inv", "sim-sl"].forEach((id) =>
  $("#" + id).addEventListener("input", runSim));
$$(".sim-presets button").forEach((b) => b.addEventListener("click", () => {
  const spec = b.dataset.p;
  const set = (id, val) => { $("#sim-" + id).value = val; };
  if (spec === "reset") {
    set("demand", 0); set("lead", 0); set("inv", 0); $("#sim-sl").value = "0.95";
  } else {
    spec.split("|").forEach((part) => {
      const [key, val] = part.split(":");
      const v = Math.round(parseFloat(val) * 100);
      if (key === "demand") set("demand", v);
      if (key === "lead") set("lead", v);
      if (key === "inv") set("inv", v);
    });
  }
  runSim();
}));

/* ================= copilot ================= */
function addBubble(cls, text, sources) {
  const div = document.createElement("div");
  div.className = "bubble " + cls;
  div.textContent = text;
  if (sources?.length) {
    const s = document.createElement("span");
    s.className = "src";
    s.textContent = "grounded in → " + sources.join(" | ");
    div.appendChild(s);
  }
  $("#chat").appendChild(div);
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

async function askQuestion(q) {
  addBubble("user", q);
  addBubble("bot", "…");
  const bubbles = $$("#chat .bubble");
  try {
    const res = await post("/api/copilot", { question: q });
    bubbles.at(-1).remove();
    addBubble("bot", res.answer, res.sources);
  } catch (e) {
    bubbles.at(-1).remove();
    addBubble("bot", "Copilot unavailable: " + e.message);
  }
}

$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#chat-input").value.trim();
  if (q) { $("#chat-input").value = ""; askQuestion(q); }
});
$$(".chat-suggest button").forEach((b) => b.addEventListener("click", () => askQuestion(b.textContent)));

/* ================= boot ================= */
(async function boot() {
  try {
    const h = await api("/health");
    $("#llm-mode").textContent = "copilot: " + h.llm_mode.replace("grounded-", "grounded ");
  } catch (_) { /* noop */ }

  await loadOverview();

  // lazy-load tab data on first open; each tab keeps its data afterwards
  const loaded = new Set(["overview"]);
  const fns = {
    forecast: loadForecast, inventory: loadInventory, suppliers: loadSuppliers,
    risk: loadRisk, recommendations: loadRecs,
  };
  $$(".nav-btn").forEach((b) => b.addEventListener("click", () => {
    const t = b.dataset.tab;
    if (fns[t] && !loaded.has(t)) { loaded.add(t); fns[t]().catch(console.error); }
  }));
})();
