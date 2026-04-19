// RL-UAV Lab — single-page dashboard client.
// Polls /api/status, /api/runs, /api/runs/<id>/tail; renders Chart.js charts.

const REWARDS = [
  "dist_normalized","completion_ratio","battery_aware",
  "time_pressure","regret_based","shaped_potential",
];
const ALGOS = ["Heuristic","PPO","A2C","RecurrentPPO"];

const state = {
  selectedReward: new Set(["completion_ratio"]),
  selectedAlgo: new Set(["Heuristic","PPO"]),
  activeRunId: null,
  runs: [],
  chartSince: 0,
  charts: {},
};

// ───────── chip builders ─────────
function buildChipset(parentId, items, selectedSet){
  const el = document.getElementById(parentId);
  el.innerHTML = "";
  items.forEach(name => {
    const c = document.createElement("div");
    c.className = "opt" + (selectedSet.has(name) ? " on" : "");
    c.textContent = name;
    c.onclick = () => {
      selectedSet.has(name) ? selectedSet.delete(name) : selectedSet.add(name);
      c.classList.toggle("on");
    };
    el.appendChild(c);
  });
}
buildChipset("rewards", REWARDS, state.selectedReward);
buildChipset("algos", ALGOS, state.selectedAlgo);

// ───────── workers slider ─────────
const workersEl = document.getElementById("workers");
const workersVal = document.getElementById("workers-val");
function updWorkers(){ workersVal.textContent = workersEl.value + " cores"; }
workersEl.oninput = updWorkers; updWorkers();

// ───────── chart helpers ─────────
function mkChart(id, label, color = "#06b6d4"){
  const ctx = document.getElementById(id).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {labels: [], datasets: [{
      label, data: [], borderColor: color, backgroundColor: color + "22",
      borderWidth: 2, pointRadius: 0, tension: 0.25, fill: true,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false, resizeDelay: 50,
      plugins:{legend:{display:false},tooltip:{backgroundColor:"#000",borderColor:"#333",borderWidth:1}},
      scales:{
        x:{ticks:{color:"rgba(255,255,255,.35)",font:{size:9}},grid:{color:"rgba(255,255,255,.05)"}},
        y:{ticks:{color:"rgba(255,255,255,.35)",font:{size:9}},grid:{color:"rgba(255,255,255,.05)"}},
      },
    },
  });
}
state.charts.ret  = mkChart("c-return",  "return",  "#06b6d4");
state.charts.eval = mkChart("c-eval",    "eval",    "#67e8f9");
state.charts.succ = mkChart("c-success", "success", "#22c55e");
state.charts.srv  = mkChart("c-served",  "served",  "#a5f3fc");

function resetAllCharts(){
  for(const k of Object.keys(state.charts)){
    const c = state.charts[k];
    c.data.labels = []; c.data.datasets[0].data = [];
    c.update("none");
  }
  state.chartSince = 0;
  document.getElementById("v-return").textContent = "–";
  document.getElementById("v-eval").textContent = "–";
  document.getElementById("v-success").textContent = "–";
  document.getElementById("v-served").textContent = "–";
}

function pushChart(chart, x, y){
  chart.data.labels.push(x);
  chart.data.datasets[0].data.push(y);
  chart.update("none");
}
function lastVal(arr){ return arr.length ? arr[arr.length-1] : null; }

// ───────── API ─────────
async function api(path, opts){
  const res = await fetch(path, opts);
  if(!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

// ───────── poll: status + runs list ─────────
async function pollStatus(){
  try{
    const s = await api("/api/status");
    document.getElementById("chip-cores").textContent = `${s.cpu_cores} cores`;
    document.getElementById("chip-scheduled").textContent = `scheduled ${s.counts.scheduled}`;
    document.getElementById("chip-done").textContent = `done ${s.counts.done}`;
    document.getElementById("chip-failed").textContent = `failed ${s.counts.failed}`;
    const statusEl = document.getElementById("chip-status");
    if(s.busy){
      statusEl.textContent = "running";
      statusEl.className = "chip chip-run";
    }else{
      statusEl.textContent = "idle";
      statusEl.className = "chip";
    }
  }catch(e){ /* swallow */ }
}

async function pollRuns(){
  try{
    const runs = await api("/api/runs");
    state.runs = runs;
    renderRunList();
    renderHeatmap();
  }catch(e){ /* swallow */ }
}

function renderRunList(){
  const q = (document.getElementById("filter-runs").value || "").toLowerCase();
  const sortMode = document.getElementById("sort-runs").value;
  let arr = state.runs.slice();
  if(q) arr = arr.filter(r => (r.run_id||"").toLowerCase().includes(q));
  if(sortMode === "return")  arr.sort((a,b)=>((b.summary||{}).mean_return||-1e9)-((a.summary||{}).mean_return||-1e9));
  if(sortMode === "success") arr.sort((a,b)=>((b.summary||{}).success_rate||0)-((a.summary||{}).success_rate||0));

  const listEl = document.getElementById("run-list");
  listEl.innerHTML = "";
  arr.forEach(r => {
    const div = document.createElement("div");
    div.className = "run-item" + (r.run_id === state.activeRunId ? " active" : "");
    const sclass = "s-" + (r.status || "scheduled");
    const sumR = ((r.summary||{}).mean_return);
    const sumS = ((r.summary||{}).success_rate);
    div.innerHTML = `
      <div class="run-item-head">
        <div class="run-item-title">${r.run_id}</div>
        <span class="status-dot ${sclass}" title="${r.status}"></span>
      </div>
      <div class="run-item-meta">
        <span>${r.reward || ""} · ${r.algo || ""} · s${r.seed ?? "?"}</span>
        <span>R=<b>${sumR==null?"–":sumR.toFixed(1)}</b> · S=<b>${sumS==null?"–":(sumS*100).toFixed(0)+"%"}</b></span>
      </div>`;
    div.onclick = () => selectRun(r.run_id);
    listEl.appendChild(div);
  });
}
document.getElementById("filter-runs").oninput = renderRunList;
document.getElementById("sort-runs").onchange = renderRunList;
document.getElementById("btn-refresh").onclick = () => { pollRuns(); pollStatus(); };

// ───────── selected run: detail + live tail ─────────
async function selectRun(runId){
  state.activeRunId = runId;
  state.chartSince = 0;
  resetAllCharts();
  renderRunList();
  // fetch full metrics first
  try{
    const data = await api(`/api/runs/${runId}`);
    const cfg = data.config || {};
    document.getElementById("detail-title").textContent = runId;
    const badges = [];
    if(cfg.reward_key) badges.push(`<span class="badge">reward <strong>${cfg.reward_key}</strong></span>`);
    if(cfg.algo) badges.push(`<span class="badge">algo <strong>${cfg.algo}</strong></span>`);
    if(cfg.seed != null) badges.push(`<span class="badge">seed <strong>${cfg.seed}</strong></span>`);
    if(data.summary && data.summary.mean_return != null){
      badges.push(`<span class="badge">return <strong>${data.summary.mean_return.toFixed(1)}</strong></span>`);
      badges.push(`<span class="badge">success <strong>${(data.summary.success_rate*100).toFixed(0)}%</strong></span>`);
    }
    document.getElementById("detail-badges").innerHTML = badges.join("");
    document.getElementById("detail-sub").textContent = cfg.reward_key
      ? `${cfg.algo} on ${cfg.reward_key}` : "";
    (data.metrics || []).forEach(m => appendMetricToCharts(m));
  }catch(e){
    document.getElementById("detail-title").textContent = runId;
    document.getElementById("detail-sub").textContent = "(failed to load)";
  }
}

function appendMetricToCharts(m){
  const x = m.iter;
  if(m.mean_return != null){
    pushChart(state.charts.ret, x, m.mean_return);
    document.getElementById("v-return").textContent = m.mean_return.toFixed(1);
  }
  if(m.eval_return != null){
    pushChart(state.charts.eval, x, m.eval_return);
    document.getElementById("v-eval").textContent = m.eval_return.toFixed(1);
  }
  if(m.success_rate != null){
    pushChart(state.charts.succ, x, m.success_rate);
    document.getElementById("v-success").textContent = (m.success_rate*100).toFixed(0) + "%";
  }
  if(m.mean_served != null){
    pushChart(state.charts.srv, x, m.mean_served);
    document.getElementById("v-served").textContent = m.mean_served.toFixed(2);
  }
  state.chartSince = Math.max(state.chartSince, x);
}

async function pollActiveRunTail(){
  if(!state.activeRunId) return;
  try{
    const data = await api(`/api/runs/${state.activeRunId}/tail?since=${state.chartSince}`);
    (data.metrics || []).forEach(m => appendMetricToCharts(m));
  }catch(e){}
}

// ───────── heatmap ─────────
async function renderHeatmap(){
  try{
    const data = await api("/api/heatmap");
    const el = document.getElementById("heatmap");
    el.innerHTML = "";
    if(!Object.keys(data).length){
      el.innerHTML = '<div class="muted small" style="padding:.5rem">No runs yet — launch one from the left panel.</div>';
      document.getElementById("heat-best").textContent = "–";
      return;
    }
    const cols = ALGOS;
    // header row
    const header = document.createElement("div");
    header.className = "heat-row";
    header.style.gridTemplateColumns = `160px repeat(${cols.length},1fr)`;
    header.innerHTML = `<div></div>` + cols.map(c=>`<div class="heat-label" style="justify-content:center;color:var(--white-60);padding:.35rem">${c}</div>`).join("");
    el.appendChild(header);
    let best = {val:-1, cell:""};
    REWARDS.forEach(rw => {
      if(!data[rw]) return;
      const row = document.createElement("div");
      row.className = "heat-row";
      row.style.gridTemplateColumns = `160px repeat(${cols.length},1fr)`;
      row.innerHTML = `<div class="heat-label">${rw}</div>` +
        cols.map(al => {
          const v = data[rw][al];
          if(!v || v.success == null) return `<div class="heat-cell heat-empty">–</div>`;
          const s = v.success; // 0..1
          const bg = `rgba(6,182,212,${Math.max(.08, s)})`;
          const fg = s > 0.5 ? "#001519" : "var(--white-80)";
          if(s > best.val){best = {val:s, cell:`${rw} · ${al}`};}
          return `<div class="heat-cell" style="background:${bg};color:${fg}" title="return ${v.return?.toFixed(1)} · n=${v.n}">${(s*100).toFixed(0)}%</div>`;
        }).join("");
      el.appendChild(row);
    });
    document.getElementById("heat-best").textContent =
      best.val >= 0 ? `best ${best.cell} (${(best.val*100).toFixed(0)}%)` : "–";
  }catch(e){}
}

// ───────── launch / stop ─────────
document.getElementById("btn-start").onclick = async () => {
  const rewards = [...state.selectedReward];
  const algos = [...state.selectedAlgo];
  const seeds = (document.getElementById("seeds").value || "0")
    .split(",").map(x=>parseInt(x.trim(),10)).filter(x=>!Number.isNaN(x));
  const workers = parseInt(workersEl.value, 10);
  if(!rewards.length || !algos.length){alert("pick at least one reward and one algo");return;}
  try{
    const res = await api("/api/start",{
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({reward:rewards, algo:algos, seeds, workers}),
    });
    if(!res.ok){alert(res.error || "failed to start");return;}
    await pollStatus();
    await pollRuns();
  }catch(e){ alert(e.message); }
};
document.getElementById("btn-stop").onclick = async () => {
  await api("/api/stop", {method:"POST"});
  await pollStatus();
};

// ───────── poll loop ─────────
setInterval(pollStatus, 1500);
setInterval(pollRuns, 2500);
setInterval(pollActiveRunTail, 1500);
pollStatus(); pollRuns();
