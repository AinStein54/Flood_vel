const API_BASE = "https://flood-vel.onrender.com";

// ── State ─────────────────────────────────────────────────────────────────────
let selectedTown    = null;
let LAT = 0, LON = 0;
let forecastData    = [];
let predictionData  = [];
let explanationData = [];

// ── Fallback coords  ─────────────────
const FALLBACK_COORDS = {
  "Leeds":               { lat: 53.8008, lon: -1.5491 },
  "Sheffield":           { lat: 53.3811, lon: -1.4701 },
  "Bradford":            { lat: 53.7938, lon: -1.7529 },
  "Doncaster":           { lat: 53.5228, lon: -1.1288 },
  "Newcastle upon Tyne": { lat: 54.9783, lon: -1.6178 },
  "York":                { lat: 53.9590, lon: -1.0815 },
  "Rotherham":           { lat: 53.4326, lon: -1.3635 },
  "Barnsley":            { lat: 53.5527, lon: -1.4797 },
  "Middlesbrough":       { lat: 54.5742, lon: -1.2348 },
  "Sunderland":          { lat: 54.9069, lon: -1.3838 },
};

// ── Load towns from API and build the town grid ───────────────────────────────
async function loadTowns() {
  let towns = [], coords = {};
  try {
    const res  = await fetch(`${API_BASE}/towns`);
    const data = await res.json();
    towns  = data.towns;
    coords = data.coords;
  } catch {
    // API offline — use fallback list
    towns  = Object.keys(FALLBACK_COORDS);
    coords = FALLBACK_COORDS;
    console.warn("Could not reach /towns endpoint — using fallback list");
  }
  buildTownGrid(towns, coords);
}

// ── Build clickable town buttons ──────────────────────────────────────────────
function buildTownGrid(towns, coords) {
  const grid = document.getElementById("town-grid");
  grid.innerHTML = "";

  towns.forEach(name => {
    const c   = coords[name] || FALLBACK_COORDS[name] || { lat: 54, lon: -1.5 };
    const btn = document.createElement("button");
    btn.className   = "town-btn";
    btn.id          = `town-btn-${name.replace(/ /g, "_")}`;
    btn.innerHTML   = `<span class="town-name">${name}</span>`;
    btn.onclick     = () => selectTown(name, c.lat, c.lon);
    grid.appendChild(btn);
  });
}

function selectTown(name, lat, lon) {
  // Highlight selected
  document.querySelectorAll(".town-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById(`town-btn-${name.replace(/ /g, "_")}`);
  if (btn) btn.classList.add("active");

  selectedTown = name;
  LAT = lat; LON = lon;

  document.getElementById("loc-name").textContent   = name;
  document.getElementById("loc-coords").textContent =
    `${lat.toFixed(4)}°N, ${Math.abs(lon).toFixed(4)}°${lon < 0 ? "W" : "E"}`;
  document.getElementById("location-card").style.display = "block";

  // Reset previous results when switching towns
  ["today-section","chart-section","picker-section",
   "prediction-section","xai-section"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });
  document.getElementById("prediction-result").innerHTML = "";

  fetchWeather(name);
}
// ── Fetch 7-day forecast from Open-Meteo ──────────────────────────────────────
async function fetchWeather(townName) {
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${LAT}&longitude=${LON}` +
    `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,` +
    `wind_speed_10m_max,wind_gusts_10m_max,` +
    `soil_moisture_0_to_100cm_mean` +
    `&hourly=relative_humidity_2m` +
    `&timezone=Europe%2FLondon`;
  try {
    const res  = await fetch(url);
    if (!res.ok) throw new Error("Weather fetch failed");
    const data = await res.json();

    // Daily humidity mean from hourly data
    const n = data.daily.time.length;
    const humidityDaily = [];
    for (let d = 0; d < n; d++) {
      const slice = (data.hourly.relative_humidity_2m || []).slice(d * 24, d * 24 + 24).filter(v => v != null);
      humidityDaily.push(slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : 80);
    }
    forecastData = data.daily.time.map((date, i) => ({
      date,
      day:          shortDay(date),
      fullDate:     longDate(date),
      town:         townName,
      temp_max:     data.daily.temperature_2m_max[i]           ?? 12,
      temp_min:     data.daily.temperature_2m_min[i]           ?? 5,
      precip:       data.daily.precipitation_sum[i]            ?? 0,
      wind_speed:   data.daily.wind_speed_10m_max[i]           ?? 10,
      wind_gusts:   data.daily.wind_gusts_10m_max[i]           ?? 15,
      soil_moisture:data.daily.soil_moisture_0_to_100cm_mean[i]?? 0.35,
      humidity:     humidityDaily[i] ?? 80,
    }));

    renderToday(forecastData[0]);
    fillDayDropdown();
    drawForecastChart();
    showSections(["today-section","chart-section","picker-section","prediction-section"]);
  } catch (err) {
    showError("today-grid", err.message);
  }
}

// ── Today weather cards ───────────────────────────────────────────────────────
function renderToday(d) {
  document.getElementById("today-date").textContent = `${d.fullDate} — ${d.town}`;
  document.getElementById("today-grid").innerHTML = `
    ${wcard("🌡️", "Max Temp",     `${d.temp_max.toFixed(1)} °C`, "")}
    ${wcard("🌡️", "Min Temp",     `${d.temp_min.toFixed(1)} °C`, "")}
    ${wcard("🌧️", "Precipitation",`${d.precip.toFixed(1)} mm`,   precipClass(d.precip))}
    ${wcard("💨", "Wind Speed",   `${d.wind_speed.toFixed(1)} km/h`, "")}
    ${wcard("🌬️", "Wind Gusts",   `${d.wind_gusts.toFixed(1)} km/h`, "")}
    ${wcard("🌫️", "Humidity",     `${d.humidity.toFixed(0)} %`,  "")}
    ${wcard("💧", "Soil Moisture",`${(d.soil_moisture * 100).toFixed(1)} %`, moistClass(d.soil_moisture))}`;
}

function wcard(icon, label, value, cls) {
  return `<div class="wcard ${cls}">
    <div class="wcard-icon">${icon}</div>
    <div class="wcard-label">${label}</div>
    <div class="wcard-value">${value}</div>
  </div>`;
}
function precipClass(p) { return p >= 20 ? "alert-high" : p >= 10 ? "alert-med" : p >= 3 ? "alert-low" : ""; }
function moistClass(m)  { return m >= 0.45 ? "alert-high" : m >= 0.38 ? "alert-med" : ""; }

// ── Day picker dropdown ───────────────────────────────────────────────────────
function fillDayDropdown() {
  const sel = document.getElementById("day-dropdown");
  sel.innerHTML = '<option value="">— Choose a day —</option>';
  forecastData.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${d.day}  ${d.date.slice(5)} — ${d.town}`;
    sel.appendChild(o);
  });
}

async function showSelectedDay() {
  const idx = document.getElementById("day-dropdown").value;
  const box = document.getElementById("day-detail-box");
  if (idx === "") { box.style.display = "none"; return; }
  const day = forecastData[parseInt(idx)];
  box.innerHTML = `<p class="loading">⏳ Fetching prediction…</p>`;
  box.style.display = "block";
  try {
    const res = await postJSON(`${API_BASE}/predict`, { days: [prepareDayForAPI(day)] });
    const p   = res.predictions[0];
    box.innerHTML = `
      <div class="detail-header">
        <h3>${day.fullDate} — ${day.town}</h3>
        <div class="risk-pill ${p.is_high_risk ? "pill-flood" : "pill-safe"}">
          ${p.is_high_risk ? "🚨" : "✅"} ${p.prediction}
        </div>
      </div>
      <div class="risk-verdict ${getRiskClass(p.flood_prob)}">
        <div class="risk-verdict-icon">${riskIcon(p.flood_prob)}</div>
        <div>
          <div class="risk-verdict-label">${p.risk_level}</div>
        </div>
      </div>
      <div class="detail-grid">
        <div class="detail-stat"><span>📍 Town</span><strong>${day.town}</strong></div>
        <div class="detail-stat"><span>🌡️ Temp</span><strong>${day.temp_max.toFixed(1)} / ${day.temp_min.toFixed(1)} °C</strong></div>
        <div class="detail-stat"><span>🌧️ Precip</span><strong>${day.precip.toFixed(1)} mm</strong></div>
        <div class="detail-stat"><span>💨 Wind</span><strong>${day.wind_speed.toFixed(1)} km/h</strong></div>
        <div class="detail-stat"><span>💧 Soil</span><strong>${(day.soil_moisture * 100).toFixed(1)} %</strong></div>
        <div class="detail-stat"><span>🌫️ Humidity</span><strong>${day.humidity.toFixed(0)} %</strong></div>
      </div>`;
  } catch (err) { box.innerHTML = `<p class="error-msg">${err.message}</p>`; }
}

function riskIcon(p) { return p >= 0.75 ? "🔴" : p >= 0.50 ? "🟠" : p >= 0.25 ? "🔵" : "🟢"; }
function getRiskClass(p) { return p >= 0.75 ? "verdict-high" : p >= 0.50 ? "verdict-med" : p >= 0.25 ? "verdict-low" : "verdict-vlow"; }

// ── 7-day prediction ──────────────────────────────────────────────────────────
async function run7DayPrediction() {
  const btn    = document.getElementById("predict-btn");
  const result = document.getElementById("prediction-result");
  btn.textContent = "⏳ Running…"; btn.disabled = true;
  result.innerHTML = `<p class="loading">Sending data to model…</p>`;
  try {
    const data = await postJSON(`${API_BASE}/predict`, { days: forecastData.map(prepareDayForAPI) });
    predictionData = data.predictions;
    let html = `<div class="pred-grid">`;
    data.predictions.forEach((p, i) => {
      const fd = forecastData[i];
      html += `<div class="pred-card ${p.is_high_risk ? "pred-flood" : "pred-safe"}">
        <div class="pred-day">${fd.day}</div>
        <div class="pred-date">${fd.date.slice(5)}</div>
        <div class="pred-icon">${p.is_high_risk ? "🚨" : "✅"}</div>
        <div class="pred-label" style="color:${p.risk_colour}">${p.risk_level}</div>
      </div>`;
    });
    html += `</div>`;
    result.innerHTML = html;
    fetchSHAP({ days: forecastData.map(prepareDayForAPI) });
  } catch (err) {
    result.innerHTML = `<p class="error-msg">❌ ${err.message}</p>`;
  } finally {
    btn.textContent = "🔮 Run Prediction"; btn.disabled = false;
  }
}

// ── SHAP explanations ─────────────────────────────────────────────────────────
async function fetchSHAP(payload) {
  const section = document.getElementById("xai-section");
  section.style.display = "block";
  setTimeout(() => section.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
  document.getElementById("shap-bullets").innerHTML = "<li class='loading'>Loading SHAP…</li>";
  try {
    const data = await postJSON(`${API_BASE}/explain`, payload);
    explanationData = data.explanations;
    document.getElementById("xai-method-label").textContent = `Method: ${data.method}`;
    buildXAITabs();
    renderXAIDay(0);
  } catch (err) {
    document.getElementById("shap-bullets").innerHTML =
      `<li class="error-msg">SHAP failed: ${err.message}</li>`;
  }
}

function buildXAITabs() {
  const tabs = document.getElementById("xai-day-tabs");
  tabs.innerHTML = "";
  forecastData.forEach((fd, i) => {
    const btn = document.createElement("button");
    btn.className   = `tab-btn ${i === 0 ? "active" : ""}`;
    btn.textContent = `${fd.day} ${fd.date.slice(5)}`;
    btn.onclick = () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderXAIDay(i);
    };
    tabs.appendChild(btn);
  });
}

function renderXAIDay(idx) {
  const exp = explanationData[idx];
  if (!exp) return;
  const riskCol = predictionData[idx]?.risk_colour ?? "#3b82f6";
  drawGauge(exp.flood_prob, riskCol);
  document.getElementById("gauge-pct").textContent  = `${(exp.flood_prob * 100).toFixed(1)}%`;
  document.getElementById("gauge-risk").textContent = predictionData[idx]?.risk_level ?? "";
  document.getElementById("gauge-risk").style.color = riskCol;
  const ul = document.getElementById("shap-bullets");
  ul.innerHTML = exp.bullets.length
    ? exp.bullets.map(b => `<li>${b}</li>`).join("")
    : `<li>Multiple weather factors contribute equally to this prediction.</li>`;
  drawSHAPChart(exp.top_features);
}

// ── D3 Forecast chart ─────────────────────────────────────────────────────────
function drawForecastChart() {
  const svg = d3.select("#d3-chart"); svg.selectAll("*").remove();
  const w = Math.min(960, window.innerWidth - 40), h = 300;
  const m = { top: 25, right: 20, bottom: 55, left: 50 };
  svg.attr("width", w).attr("height", h);
  const x  = d3.scaleBand().domain(forecastData.map(d => d.date)).range([m.left, w - m.right]).padding(0.3);
  const yT = d3.scaleLinear().domain([0, d3.max(forecastData, d => d.temp_max) + 5]).range([h - m.bottom, m.top]);
  const yP = d3.scaleLinear().domain([0, Math.max(15, d3.max(forecastData, d => d.precip) + 5)]).range([h - m.bottom, m.top]);
  svg.selectAll(".pb").data(forecastData).join("rect")
    .attr("x", d => x(d.date)).attr("y", d => yP(d.precip))
    .attr("width", x.bandwidth()).attr("height", d => h - m.bottom - yP(d.precip))
    .attr("fill", d => d.precip >= 10 ? "#6366f1" : "#818cf8").attr("opacity", 0.7).attr("rx", 3);
  const line = d3.line().x(d => x(d.date) + x.bandwidth() / 2).y(d => yT(d.temp_max)).curve(d3.curveMonotoneX);
  svg.append("path").datum(forecastData).attr("fill","none").attr("stroke","#60a5fa").attr("stroke-width",2.5).attr("d",line);
  svg.selectAll(".lbl").data(forecastData).join("text")
    .attr("x", d => x(d.date) + x.bandwidth() / 2).attr("y", d => yT(d.temp_max) - 7)
    .attr("text-anchor","middle").attr("fill","#e2e8f0").attr("font-size","11px")
    .text(d => d.temp_max.toFixed(0) + "°");
  svg.append("g").attr("transform",`translate(0,${h - m.bottom})`)
    .call(d3.axisBottom(x).tickFormat(d => { const f = forecastData.find(fd => fd.date === d); return `${f.day}\n${d.slice(5)}`; }))
    .selectAll("text").attr("text-anchor","middle").attr("dy","1.4em").attr("font-size","12px").attr("fill","#94a3b8");
  svg.append("g").attr("transform",`translate(${m.left},0)`)
    .call(d3.axisLeft(yT).ticks(5).tickFormat(d => d + "°"))
    .selectAll("text").attr("fill","#60a5fa").attr("font-size","10px");
}

// ── D3 SHAP chart ─────────────────────────────────────────────────────────────
function drawSHAPChart(features) {
  const el = d3.select("#shap-chart"); el.selectAll("*").remove();
  if (!features?.length) return;
  const data = features.slice(0, 10);
  const w = Math.min(680, window.innerWidth - 60);
  const barH = 36, m = { top: 8, right: 85, bottom: 8, left: 200 };
  const h = data.length * barH + m.top + m.bottom;
  el.attr("width", w).attr("height", h);
  const maxAbs = d3.max(data, d => d.abs_shap) || 0.01;
  const xS = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([m.left, w - m.right]);
  const yS = d3.scaleBand().domain(data.map((_, i) => i)).range([m.top, h - m.bottom]).padding(0.2);
  const zero = xS(0);
  el.append("line").attr("x1",zero).attr("x2",zero).attr("y1",m.top).attr("y2",h-m.bottom)
    .attr("stroke","#475569").attr("stroke-width",1.5);
  el.selectAll(".sb").data(data).join("rect")
    .attr("x", d => d.shap_value >= 0 ? zero : xS(d.shap_value))
    .attr("y", (_, i) => yS(i))
    .attr("width", d => Math.abs(xS(d.shap_value) - zero))
    .attr("height", yS.bandwidth())
    .attr("fill", d => d.shap_value >= 0 ? "#ef4444" : "#3b82f6")
    .attr("rx", 3).attr("opacity", 0.85);
  el.selectAll(".sl").data(data).join("text")
    .attr("x", m.left - 8).attr("y", (_, i) => yS(i) + yS.bandwidth() / 2 + 4)
    .attr("text-anchor","end").attr("fill","#e2e8f0").attr("font-size","12px")
    .text(d => trunc(d.display_name, 24));
  el.selectAll(".sv").data(data).join("text")
    .attr("x", d => d.shap_value >= 0 ? xS(d.shap_value) + 5 : xS(d.shap_value) - 5)
    .attr("y", (_, i) => yS(i) + yS.bandwidth() / 2 + 4)
    .attr("text-anchor", d => d.shap_value >= 0 ? "start" : "end")
    .attr("fill", d => d.shap_value >= 0 ? "#fca5a5" : "#93c5fd")
    .attr("font-size","11px")
    .text(d => (d.shap_value >= 0 ? "+" : "") + d.shap_value.toFixed(3));
}

// ── Gauge ─────────────────────────────────────────────────────────────────────
function drawGauge(prob, color) {
  const svg = d3.select("#gauge-svg"); svg.selectAll("*").remove();
  const cx = 110, cy = 115, r = 90, sw = 18, sa = -Math.PI, ea = 0;
  const angle = sa + prob * Math.PI;
  const arc = (s, e) => {
    const lg = (e - s) > Math.PI ? 1 : 0;
    return `M ${cx + r * Math.cos(s)} ${cy + r * Math.sin(s)} A ${r} ${r} 0 ${lg} 1 ${cx + r * Math.cos(e)} ${cy + r * Math.sin(e)}`;
  };
  svg.append("path").attr("d", arc(sa, ea)).attr("fill","none").attr("stroke","#1e293b").attr("stroke-width",sw);
  if (prob > 0) svg.append("path").attr("d", arc(sa, angle)).attr("fill","none").attr("stroke",color).attr("stroke-width",sw).attr("stroke-linecap","round");
  [0, 0.5, 1].forEach(v => {
    const a = sa + v * Math.PI, ir = r - sw / 2 - 2, or_ = r + sw / 2 + 4;
    svg.append("line").attr("x1",cx+ir*Math.cos(a)).attr("y1",cy+ir*Math.sin(a)).attr("x2",cx+or_*Math.cos(a)).attr("y2",cy+or_*Math.sin(a)).attr("stroke","#334155").attr("stroke-width",2);
    svg.append("text").attr("x",cx+(r+sw/2+18)*Math.cos(a)).attr("y",cy+(r+sw/2+18)*Math.sin(a)).attr("text-anchor","middle").attr("dominant-baseline","middle").attr("fill","#64748b").attr("font-size","10px").text(v===0?"0%":v===0.5?"50%":"100%");
  });
}

// ── API payload — matches this pipeline's feature names exactly ───────────────
function prepareDayForAPI(day) {
  return {
    town:               day.town,
    temp_max:           day.temp_max,
    temp_min:           day.temp_min,
    precipitation_sum:  day.precip,
    wind_speed_max:     day.wind_speed,
    wind_gusts_max:     day.wind_gusts ?? day.wind_speed * 1.4,
    humidity_mean:      day.humidity  ?? 80,
    soil_moisture_mean: day.soil_moisture,
    pub_month:          new Date(day.date).getMonth() + 1,
  };
}

// ── Utilities ─────────────────────────────────────────────────────────────────
async function postJSON(url, body) {
  const res = await fetch(url, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
function showSections(ids) { ids.forEach(id => { const el = document.getElementById(id); if (el) el.style.display = "block"; }); }
function showError(id, msg) { const el = document.getElementById(id); if (el) el.innerHTML = `<p class="error-msg">⚠️ ${msg}</p>`; }
function shortDay(s) { return new Date(s).toLocaleDateString("en-GB", { weekday:"short" }); }
function longDate(s)  { return new Date(s).toLocaleDateString("en-GB", { weekday:"long", day:"numeric", month:"long", year:"numeric" }); }
function trunc(s, n)  { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  loadTowns();   // fetch towns from API, then build the grid
});
