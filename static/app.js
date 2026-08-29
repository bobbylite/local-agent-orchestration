/* ═══════════════════════════════════════════════════════════════════════════
   Orchestration dashboard client.

   One SSE connection carries four frame types: `snapshot` (full state on
   connect), `event` (a pipeline event), `ollama` (poller output) and `notice`
   (server-synthesised alerts). Pipeline events are folded into the local run
   index exactly as the server folds them, so the UI updates from the delta
   rather than re-fetching — that is what makes token streaming feel immediate.
   ═══════════════════════════════════════════════════════════════════════════ */

const $ = (id) => document.getElementById(id);

const STAGE_LABEL = {
  route: "Route", work: "Worker", judge: "Judge", escalate: "Escalate",
  architect: "Architect", builder: "Builder",
};
const DEFAULT_PIPELINE = ["route", "work", "judge", "escalate"];
const TOAST_LIMIT = 5;
const TOAST_MS = 4200;

const state = {
  runs: new Map(),
  order: [],
  selected: null,
  config: {},
  ollama: { reachable: false, models: [], running: [] },
  charsWindow: [],
  rate: [],
};

/* ── run folding (mirrors dashboard.py's Hub.apply) ─────────────────────── */

function ensureRun(id) {
  let run = state.runs.get(id);
  if (!run) {
    run = { id, tool: "quick_question", request: "", pipeline: [], started: Date.now() / 1000,
            status: "running", source: "", seconds: null, worker_model: "", attempts: 0,
            score: null, error: "", answer: "", origin: "dashboard", stages: [], pending_approval: null };
    state.runs.set(id, run);
    state.order.push(id);
    while (state.order.length > 60) state.runs.delete(state.order.shift());
  }
  return run;
}

const activeStage = (run, name) => {
  for (let i = run.stages.length - 1; i >= 0; i--) {
    if (run.stages[i].name === name && run.stages[i].state === "running") return run.stages[i];
  }
  return null;
};

function applyEvent(frame) {
  const { run_id, kind, ts, payload } = frame;
  const run = ensureRun(run_id);
  const name = payload.stage || "";
  let stage = activeStage(run, name);

  switch (kind) {
    case "run_started":
      Object.assign(run, {
        request: payload.request || "", tool: payload.tool || "quick_question",
        pipeline: payload.pipeline || [], started: ts, status: "running",
      });
      break;
    case "stage_started":
      if (!stage) {
        stage = { name, title: "", model: "", state: "running", started: ts,
                  seconds: null, chars: 0, attempt: null, score: null, verdict: "", tools: [], text: "" };
        run.stages.push(stage);
      }
      if (payload.title) stage.title = payload.title;
      if (payload.model) stage.model = payload.model;
      if (payload.attempt != null) { stage.attempt = payload.attempt; run.attempts = Math.max(run.attempts, payload.attempt); }
      if (stage.model && name === "work") run.worker_model = stage.model;
      break;
    case "token":
      if (stage) { stage.text += payload.text || ""; countChars((payload.text || "").length); }
      break;
    case "tool_call":
      (stage || run.stages.at(-1))?.tools.push({ tool: payload.tool, detail: payload.detail });
      break;
    case "verdict":
      run.score = payload.score;
      { const s = stage || run.stages.at(-1);
        if (s) { s.score = payload.score; s.verdict = payload.verdict; } }
      break;
    case "stage_finished":
      if (stage) {
        stage.state = stage.verdict === "retry" ? "failed" : "done";
        if (payload.seconds != null) stage.seconds = payload.seconds;
        if (payload.chars) stage.chars = payload.chars;
        if (payload.chose) stage.model = payload.chose;
      }
      break;
    case "approval_requested":
      run.pending_approval = { run_id, path: payload.path, bytes: payload.bytes, content: payload.content };
      break;
    case "approval_resolved":
      run.pending_approval = null;
      break;
    case "run_finished":
      Object.assign(run, {
        status: "done", finished: ts, source: payload.source || "", seconds: payload.seconds,
        worker_model: payload.worker_model || run.worker_model,
        attempts: payload.attempts || run.attempts, answer: payload.answer || "", pending_approval: null,
      });
      run.stages.forEach((s) => { if (s.state === "running") s.state = "done"; });
      break;
    case "run_failed":
      Object.assign(run, { status: "failed", finished: ts, error: payload.error || "", pending_approval: null });
      run.stages.forEach((s) => { if (s.state === "running") s.state = "failed"; });
      break;
  }
  return run;
}

/* ── notifications ──────────────────────────────────────────────────────── */

function toast(level, title, body) {
  const box = $("toasts");
  while (box.children.length >= TOAST_LIMIT) dismiss(box.firstElementChild);

  const el = document.createElement("div");
  el.className = "toast";
  el.dataset.level = level;
  el.innerHTML = `<div class="toast-title"><i></i><span></span></div>${body ? '<div class="toast-body"></div>' : ""}<div class="toast-bar"></div>`;
  el.querySelector(".toast-title span").textContent = title;
  if (body) el.querySelector(".toast-body").textContent = body;
  box.appendChild(el);

  const bar = el.querySelector(".toast-bar");
  bar.animate([{ transform: "scaleX(1)" }, { transform: "scaleX(0)" }], { duration: TOAST_MS, easing: "linear", fill: "forwards" });
  const timer = setTimeout(() => dismiss(el), TOAST_MS);
  el.addEventListener("click", () => { clearTimeout(timer); dismiss(el); });
}

function dismiss(el) {
  if (!el || el.classList.contains("out")) return;
  el.classList.add("out");
  el.addEventListener("animationend", () => el.remove(), { once: true });
  setTimeout(() => el.remove(), 600);
}

function notifyForEvent(frame, run) {
  const p = frame.payload;
  const where = run.origin === "cli" ? " · from terminal" : "";
  switch (frame.kind) {
    case "run_started":
      toast("info", "Run dispatched" + where, p.request); break;
    case "stage_started":
      if (p.stage === "work" && p.attempt > 1) toast("warn", `Retrying on ${p.model}`, `attempt ${p.attempt}`);
      else if (p.stage === "escalate") toast("warn", "Escalating to Claude", "local attempts exhausted");
      else if (p.model) toast("muted", `${STAGE_LABEL[p.stage] || p.stage} started`, p.model);
      break;
    case "tool_call":
      toast("info", `tool · ${p.tool}`, p.detail); break;
    case "verdict":
      toast(p.verdict === "accept" ? "success" : "warn",
            p.score == null ? "Judge returned no score" : `Judge scored ${p.score}/5`,
            p.verdict === "accept" ? "accepted" : "sending back for a retry");
      break;
    case "run_finished":
      toast("success", `Answered via ${p.source}`, `${p.seconds}s · ${p.worker_model || ""}`); break;
    case "run_failed":
      toast("error", "Run failed", p.error); break;
  }
}

/* ── rendering ──────────────────────────────────────────────────────────── */

const fmtBytes = (n) => (n > 1e9 ? (n / 1e9).toFixed(1) + " GB" : (n / 1e6).toFixed(0) + " MB");
const ago = (ts) => {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
};

function tweenNum(el, target, decimals = 0, suffix = "") {
  const from = parseFloat(el.dataset.v || "0");
  if (from === target) return;
  el.dataset.v = String(target);
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 620);
    const eased = 1 - Math.pow(1 - k, 3);
    const value = from + (target - from) * eased;
    el.firstChild ? (el.firstChild.nodeType === 3 ? el.firstChild.textContent = value.toFixed(decimals) : el.textContent = value.toFixed(decimals) + suffix)
                  : (el.textContent = value.toFixed(decimals) + suffix);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderOllama() {
  const { reachable, models, running } = state.ollama;
  $("pillOllama").dataset.state = reachable ? "up" : "down";
  $("ollamaState").textContent = reachable ? "online" : "offline";
  tweenNum($("statModels"), models.length);
  const vram = running.reduce((sum, m) => sum + (m.size_vram || 0), 0);
  tweenNum($("statVram"), vram / 1e9, 1);

  const residentNames = new Set(running.map((m) => m.name));
  const roles = state.config.worker_models || {};
  const roleOf = (name) => {
    const base = name.replace(/:latest$/, "");
    if (base === state.config.judge || name === state.config.judge) return "judge";
    if (name === state.config.router) return "router";
    if (name === state.config.bump_worker) return "escalated worker";
    if (roles[name]) return "worker";
    return null;
  };

  const sorted = [...models].sort((a, b) => {
    const live = residentNames.has(b.name) - residentNames.has(a.name);
    return live || (roleOf(b.name) ? 1 : 0) - (roleOf(a.name) ? 1 : 0) || a.name.localeCompare(b.name);
  });

  $("rosterCount").textContent = `${residentNames.size}/${models.length} resident`;
  const host = $("roster");
  if (!sorted.length) { host.innerHTML = '<div class="empty">waiting for ollama…</div>'; return; }

  host.innerHTML = sorted.map((m, i) => {
    const live = residentNames.has(m.name);
    const resident = running.find((r) => r.name === m.name);
    const role = roleOf(m.name);
    const caps = (m.details && m.capabilities) || m.capabilities || [];
    const pct = resident && m.size ? Math.min(100, (resident.size_vram / m.size) * 100) : 0;
    return `<div class="agent" data-live="${live ? 1 : 0}" style="animation-delay:${i * 28}ms">
      <div class="agent-top">
        <span class="agent-name">${esc(m.name)}</span>
        <span class="agent-size">${m.details?.parameter_size || fmtBytes(m.size || 0)}</span>
      </div>
      <div class="agent-role">${role ? esc(roles[m.name] || role) : esc(m.details?.family || "installed")}</div>
      <div class="chips">
        ${live ? '<span class="chip chip-live">resident</span>' : ""}
        ${role ? `<span class="chip chip-role">${esc(role)}</span>` : ""}
        ${caps.includes("tools") ? '<span class="chip">tools</span>' : ""}
        ${m.details?.quantization_level ? `<span class="chip">${esc(m.details.quantization_level)}</span>` : ""}
      </div>
      ${live ? `<div class="vram-bar"><i style="width:${pct}%"></i></div>` : ""}
    </div>`;
  }).join("");
}

function renderConfig() {
  const c = state.config;
  if (!c.router) return;
  $("config").innerHTML = [
    ["router", c.router], ["default worker", c.default_worker], ["retry worker", c.bump_worker],
    ["judge", c.judge], ["escalation", c.escalation],
    ["local attempts", c.max_attempts], ["accept at", `≥ ${c.accept_threshold}/5`],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("");
}

function currentRun() {
  return state.selected ? state.runs.get(state.selected) : null;
}

function renderGraph() {
  const run = currentRun();
  const pipeline = run?.pipeline?.length ? run.pipeline : DEFAULT_PIPELINE;
  const host = $("nodes");

  const stageFor = (name) => {
    const matches = run ? run.stages.filter((s) => s.name === name) : [];
    return matches.at(-1) || null;
  };

  host.innerHTML = pipeline.map((name, i) => {
    const s = stageFor(name);
    const count = run ? run.stages.filter((x) => x.name === name).length : 0;
    const meta = s ? (s.state === "running" ? "running…"
                    : s.score != null ? `${s.score}/5`
                    : s.seconds != null ? `${s.seconds}s` : "—") : "—";
    return `${i ? `<div class="connector" data-active="${s?.state === "running" ? 1 : 0}"></div>` : ""}
      <div class="node-slot"><div class="node" data-state="${s ? s.state : "idle"}" data-stage="${name}">
        ${count > 1 ? `<span class="node-badge">×${count}</span>` : ""}
        <div class="node-name">${esc(STAGE_LABEL[name] || name)}</div>
        <div class="node-model">${esc(s?.model || defaultModelFor(name) || "—")}</div>
        <div class="node-meta">${esc(meta)}</div>
      </div></div>`;
  }).join("");

  const retrying = run ? run.stages.filter((s) => s.name === "work").length > 1 : false;
  $("graph").dataset.retry = retrying ? "1" : "0";
  $("graphStatus").textContent = !run ? "idle"
    : run.status === "running" ? `running · ${run.request.slice(0, 42)}`
    : `${run.status} · ${run.seconds ?? "?"}s`;
  layoutArc();
}

function defaultModelFor(name) {
  const c = state.config;
  return { route: c.router, work: c.default_worker, judge: c.judge, escalate: c.escalation }[name] || "";
}

/* The retry edge is drawn under the row, measured from the live node boxes so it
   stays glued to them through every reflow. */
function layoutArc() {
  const graph = $("graph"), svg = $("graphArc");
  const work = graph.querySelector('.node[data-stage="work"]');
  const judge = graph.querySelector('.node[data-stage="judge"]');
  if (!work || !judge) { $("arcPath").removeAttribute("d"); $("arcFlow").removeAttribute("d"); return; }

  const base = graph.getBoundingClientRect();
  const a = judge.getBoundingClientRect(), b = work.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${base.width} ${base.height}`);
  const x1 = a.left + a.width / 2 - base.left, x2 = b.left + b.width / 2 - base.left;
  const y = a.bottom - base.top, dip = base.height - 6;
  const d = `M ${x1} ${y} C ${x1} ${dip}, ${x2} ${dip}, ${x2} ${y}`;
  $("arcPath").setAttribute("d", d);
  $("arcFlow").setAttribute("d", d);
}

/* ── transcript ─────────────────────────────────────────────────────────── */

const streamCache = { runId: null, stages: [] };

function renderStream(force = false) {
  const run = currentRun(), host = $("stream");
  if (!run) { host.innerHTML = '<div class="empty">Dispatch a question above to watch it stream.</div>'; return; }

  if (force || streamCache.runId !== run.id) {
    host.innerHTML = "";
    streamCache.runId = run.id;
    streamCache.stages = [];
  }

  run.stages.forEach((stage, i) => {
    let cached = streamCache.stages[i];
    if (!cached) {
      const head = document.createElement("div");
      head.className = "stream-head";
      const body = document.createElement("div");
      body.className = "stream-body";
      host.append(head, body);
      cached = streamCache.stages[i] = { head, body, len: 0, tools: 0 };
    }
    const label = STAGE_LABEL[stage.name] || stage.name;
    cached.head.innerHTML = `<span>${esc(label)}</span><em>${esc(stage.model || "")}${
      stage.seconds != null ? ` · ${stage.seconds}s` : ""}${stage.score != null ? ` · ${stage.score}/5` : ""}</em>`;

    while (cached.tools < stage.tools.length) {
      const t = stage.tools[cached.tools++];
      const chip = document.createElement("div");
      chip.className = "stream-tool";
      chip.textContent = `🔧 ${t.tool} — ${t.detail || ""}`;
      cached.body.appendChild(chip);
    }
    if (stage.text.length < cached.len) { cached.body.textContent = stage.text; cached.len = stage.text.length; }
    else if (stage.text.length > cached.len) {
      const span = document.createElement("span");
      span.className = "tok";
      span.textContent = stage.text.slice(cached.len);
      cached.body.appendChild(span);
      cached.len = stage.text.length;
    }
  });

  host.querySelector(".stream-caret")?.remove();
  if (run.status === "running") {
    const caret = document.createElement("span");
    caret.className = "stream-caret";
    host.appendChild(caret);
  }

  const last = run.stages.at(-1);
  $("streamMeta").textContent = run.status === "running"
    ? `${STAGE_LABEL[last?.name] || "…"} · ${run.origin === "cli" ? "terminal run" : "dashboard run"}`
    : `${run.status} · ${run.stages.length} stages`;

  if ($("followBtn").dataset.on === "1") host.scrollTop = host.scrollHeight;
}

/* ── telemetry ──────────────────────────────────────────────────────────── */

function renderTelemetry() {
  const runs = state.order.map((id) => state.runs.get(id)).filter(Boolean);
  const done = runs.filter((r) => r.status === "done");
  tweenNum($("mRuns"), runs.length);
  tweenNum($("statRuns"), runs.length);

  const firstPass = done.filter((r) => r.attempts <= 1 && r.source === "local").length;
  setSuffixed($("mFirstPass"), done.length ? Math.round((firstPass / done.length) * 100) : 0, "%");

  const times = done.map((r) => r.seconds).filter((s) => typeof s === "number").sort((a, b) => a - b);
  const median = times.length ? times[Math.floor(times.length / 2)] : 0;
  setSuffixed($("mAvg"), Math.round(median), "s");
  tweenNum($("mEsc"), done.filter((r) => r.source === "claude").length);

  drawSpark(done.slice(-26).map((r) => r.seconds || 0));
  refreshInstruments();

  const buckets = [0, 0, 0, 0, 0];
  runs.forEach((r) => r.stages.forEach((s) => { if (s.score >= 1 && s.score <= 5) buckets[s.score - 1]++; }));
  const peak = Math.max(1, ...buckets);
  const threshold = state.config.accept_threshold ?? 4;
  $("scoreBars").innerHTML = buckets.map((n, i) =>
    `<div style="height:${Math.max(3, (n / peak) * 100)}%" data-pass="${i + 1 >= threshold ? 1 : 0}" title="${n} at ${i + 1}/5"><span>${i + 1}</span></div>`
  ).join("");
}

function setSuffixed(el, value, suffix) {
  const from = parseFloat(el.dataset.v || "0");
  el.dataset.v = String(value);
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 620);
    const v = Math.round(from + (value - from) * (1 - Math.pow(1 - k, 3)));
    el.innerHTML = `${v}<i>${suffix}</i>`;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function drawSpark(values) {
  const W = 240, H = 56, pad = 4;
  if (values.length < 2) {
    $("sparkLine").setAttribute("points", "");
    $("sparkArea").removeAttribute("d");
    $("sparkDot").setAttribute("r", "0");
    $("sparkNote").textContent = values.length ? `${values[0].toFixed(1)}s` : "no data yet";
    return;
  }
  const max = Math.max(...values), min = Math.min(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => [
    pad + (i / (values.length - 1)) * (W - pad * 2),
    H - pad - ((v - min) / span) * (H - pad * 2),
  ]);
  $("sparkLine").setAttribute("points", pts.map((p) => p.join(",")).join(" "));
  $("sparkArea").setAttribute("d", `M ${pts[0][0]} ${H} L ${pts.map((p) => p.join(" ")).join(" L ")} L ${pts.at(-1)[0]} ${H} Z`);
  const dot = $("sparkDot");
  dot.setAttribute("cx", pts.at(-1)[0]); dot.setAttribute("cy", pts.at(-1)[1]); dot.setAttribute("r", "3.2");
  $("sparkNote").textContent = `${min.toFixed(1)}s – ${max.toFixed(1)}s`;
}

/* ── history ────────────────────────────────────────────────────────────── */

function renderHistory() {
  const runs = state.order.map((id) => state.runs.get(id)).filter(Boolean).reverse();
  $("historyCount").textContent = runs.length ? `${runs.length}` : "";
  const host = $("history");
  if (!runs.length) { host.innerHTML = '<div class="empty">no runs yet</div>'; return; }

  host.innerHTML = runs.map((r, i) => `
    <div class="run" data-id="${r.id}" data-status="${r.status}" data-selected="${r.id === state.selected ? 1 : 0}" style="animation-delay:${Math.min(i, 10) * 26}ms">
      <div class="run-top">
        <span class="run-dot"></span>
        <span class="run-q">${esc(r.request || "(no prompt)")}</span>
        <span class="run-time">${r.seconds != null ? r.seconds + "s" : ago(r.started)}</span>
      </div>
      <div class="run-bottom">
        ${r.stages.map((s) => `<i class="pip" data-state="${s.state}" title="${esc(s.name)}"></i>`).join("")}
        ${r.origin === "cli" ? '<span class="tag tag-cli">cli</span>' : ""}
        ${r.source ? `<span class="tag tag-${r.source}">${esc(r.source)}</span>` : ""}
      </div>
    </div>`).join("");

  host.querySelectorAll(".run").forEach((el) =>
    el.addEventListener("click", () => { state.selected = el.dataset.id; renderAll(); }));
}

function renderAll() {
  renderGraph();
  renderStream();
  renderTelemetry();
  renderHistory();
  renderAnswer();
}

/* ── approval modal ─────────────────────────────────────────────────────── */

function showApproval(pending) {
  $("apvPath").textContent = pending.path;
  $("apvBytes").textContent = `${pending.bytes} bytes`;
  $("apvContent").textContent = pending.content || "";
  $("approvalShade").hidden = false;
  $("approvalShade").dataset.runId = pending.run_id;
  toast("warn", "Write permission requested", pending.path);
}

async function resolveApproval(allow) {
  const runId = $("approvalShade").dataset.runId;
  $("approvalShade").hidden = true;
  try {
    await fetch(`/api/approve/${runId}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ allow }),
    });
    toast(allow ? "success" : "muted", allow ? "Write allowed" : "Write denied", "");
  } catch (err) {
    toast("error", "Could not send the decision", String(err));
  }
}

/* ── instruments ────────────────────────────────────────────────────────────
   Each gauge is a 240° arc with its own needle. Values are eased toward the
   target every frame rather than snapped, so a reading that jumps still reads
   as a physical movement — and the needle keeps a faint idle tremor so the
   panel never looks frozen when the pipeline is quiet.
   ──────────────────────────────────────────────────────────────────────────── */

const SWEEP_START = -210, SWEEP_END = 30;           // degrees, clockwise from 3 o'clock
const polar = (cx, cy, r, deg) => {
  const rad = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
};
const arcPath = (cx, cy, r, from, to) => {
  const [x1, y1] = polar(cx, cy, r, from), [x2, y2] = polar(cx, cy, r, to);
  return `M ${x1} ${y1} A ${r} ${r} 0 ${Math.abs(to - from) > 180 ? 1 : 0} 1 ${x2} ${y2}`;
};

function makeGauge(host, { label, unit = "", max = 100, decimals = 0, majors = 5 }) {
  const W = 132, H = 104, cx = 66, cy = 66, r = 45;
  const ticks = [];
  for (let i = 0; i <= majors * 2; i++) {
    const deg = SWEEP_START + ((SWEEP_END - SWEEP_START) * i) / (majors * 2);
    const major = i % 2 === 0;
    const [ax, ay] = polar(cx, cy, r - 11, deg), [bx, by] = polar(cx, cy, r - (major ? 17 : 14.5), deg);
    ticks.push(`<line class="g-tick${major ? " g-tick-major" : ""}" x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}"/>`);
  }
  host.innerHTML = `
    <div class="gauge-wrap">
    <svg class="gauge" viewBox="0 0 ${W} ${H}">
      <defs><linearGradient id="gaugeGrad" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="var(--blue-deep)"/>
        <stop offset="55%" stop-color="var(--blue)"/>
        <stop offset="100%" stop-color="var(--cyan)"/>
      </linearGradient></defs>
      <path class="g-track" d="${arcPath(cx, cy, r, SWEEP_START, SWEEP_END)}"/>
      <path class="g-value" d="${arcPath(cx, cy, r, SWEEP_START, SWEEP_END)}"/>
      ${ticks.join("")}
      <line class="g-needle" x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy - r + 15}"/>
      <circle class="g-hub" cx="${cx}" cy="${cy}" r="6.5"/>
      <circle class="g-hub-dot" cx="${cx}" cy="${cy}" r="2.4"/>
    </svg>
    <b class="inst-value">0${unit ? `<i>${unit}</i>` : ""}</b>
    </div>
    <em class="inst-label">${label}</em>`;

  const valueArc = host.querySelector(".g-value");
  const needle = host.querySelector(".g-needle");
  const readout = host.querySelector(".inst-value");
  const len = valueArc.getTotalLength();
  valueArc.style.strokeDasharray = len;
  valueArc.style.strokeDashoffset = len;

  const gauge = { target: 0, shown: 0, max, decimals, unit, valueArc, needle, readout, len, host };
  instruments.push(gauge);
  return gauge;
}

const instruments = [];

function paintGauge(g) {
  const frac = Math.max(0, Math.min(1, g.shown / g.max));
  g.valueArc.style.strokeDashoffset = g.len * (1 - frac);
  const deg = SWEEP_START + (SWEEP_END - SWEEP_START) * frac;
  g.needle.setAttribute("transform", `rotate(${deg + 90} 66 66)`);
  g.readout.innerHTML = g.shown.toFixed(g.decimals) + (g.unit ? `<i>${g.unit}</i>` : "");
  g.host.dataset.live = g.shown > g.max * 0.02 ? "1" : "0";
}

/* One rAF loop drives every needle: ease toward target, plus a tremor small
   enough to read as instrument noise rather than data. */
function animateInstruments(now) {
  for (const g of instruments) {
    const tremor = g.target > 0 ? Math.sin(now / 420 + g.max) * g.max * 0.004 : 0;
    g.shown += (g.target + tremor - g.shown) * 0.12;
    if (Math.abs(g.shown) < 1e-4) g.shown = 0;
    paintGauge(g);
  }
  requestAnimationFrame(animateInstruments);
}

/* ── oscilloscope ───────────────────────────────────────────────────────── */

const scope = { history: new Array(160).fill(0), phase: 0 };

function drawScope() {
  const canvas = $("scope");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 620, h = canvas.clientHeight || 104;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const blue = css.getPropertyValue("--blue").trim() || "#4f8ff7";
  const edge = css.getPropertyValue("--edge").trim() || "#28313d";

  ctx.strokeStyle = edge; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  scope.phase = (scope.phase + 0.6) % 26;
  for (let x = -scope.phase; x < w; x += 26) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const peak = Math.max(12, ...scope.history);
  const pts = scope.history.map((v, i) => [(i / (scope.history.length - 1)) * w, h - 6 - (v / peak) * (h - 14)]);

  const fill = ctx.createLinearGradient(0, 0, 0, h);
  fill.addColorStop(0, blue + "55"); fill.addColorStop(1, blue + "00");
  ctx.beginPath(); ctx.moveTo(0, h);
  pts.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.lineTo(w, h); ctx.closePath(); ctx.fillStyle = fill; ctx.fill();

  ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = blue; ctx.lineWidth = 1.7; ctx.lineJoin = "round";
  ctx.shadowColor = blue; ctx.shadowBlur = 9; ctx.stroke(); ctx.shadowBlur = 0;

  const [hx, hy] = pts.at(-1);
  ctx.beginPath(); ctx.arc(hx, hy, 3.1, 0, Math.PI * 2);
  ctx.fillStyle = blue; ctx.shadowColor = blue; ctx.shadowBlur = 11; ctx.fill(); ctx.shadowBlur = 0;
  requestAnimationFrame(drawScope);
}

/* ── demo sweep ─────────────────────────────────────────────────────────────
   Drives the instruments through their range so they can be shown off with no
   run in flight. Clearly labelled, and it restores real readings on release. */
let sweeping = false;
async function demoSweep() {
  if (sweeping) return;
  sweeping = true;
  $("sweepBtn").dataset.on = "1";
  $("scopeNote").textContent = "demo sweep";
  const steps = [0.15, 0.55, 0.32, 0.88, 0.62, 1, 0.4, 0.08];
  for (const k of steps) {
    instruments.forEach((g) => (g.target = g.max * k));
    scope.history.push(k * 60); scope.history.shift();
    await new Promise((done) => setTimeout(done, 260));
  }
  sweeping = false;
  $("sweepBtn").dataset.on = "0";
  refreshInstruments();
}

/* ── tabs ───────────────────────────────────────────────────────────────── */

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("is-active", p.dataset.panel === name));
  const active = document.querySelector(".tab.is-active");
  const ink = $("tabInk"), tabs = document.querySelector(".tabs");
  if (active && ink) {
    ink.style.left = active.offsetLeft - tabs.scrollLeft + "px";
    ink.style.width = active.offsetWidth + "px";
  }
  if (name === "transcript") $("tabDot").dataset.on = "0";
  if (name === "answer") renderAnswer();
  if (name === "transcript") renderStream(true);
}

function renderAnswer() {
  const run = currentRun(), host = $("answer");
  if (!run || !run.answer) {
    host.innerHTML = '<div class="empty">the final answer of the selected run appears here</div>';
    return;
  }
  host.textContent = run.answer;
}

/* Point the gauges at the current real readings (no-op while a demo sweep
   owns them, so the sweep is not fought frame by frame). */
function refreshInstruments() {
  if (sweeping || !gTok) return;
  const runs = state.order.map((id) => state.runs.get(id)).filter(Boolean);
  const done = runs.filter((r) => r.status === "done");
  const vram = (state.ollama.running || []).reduce((sum, m) => sum + (m.size_vram || 0), 0) / 1e9;
  const firstPass = done.filter((r) => r.attempts <= 1 && r.source === "local").length;
  const scores = runs.flatMap((r) => r.stages.map((x) => x.score)).filter((n) => n >= 1 && n <= 5);

  gVram.target = vram;
  gAccept.target = done.length ? (firstPass / done.length) * 100 : 0;
  gJudge.target = scores.length ? scores.at(-1) : 0;
  $("scopeNote").textContent = runs.some((r) => r.status === "running")
    ? "streaming" : done.length ? `${done.length} runs` : "standby";
}

/* ── token rate ekg ─────────────────────────────────────────────────────── */

function countChars(n) { state.charsWindow.push([performance.now(), n]); }

function tickRate() {
  const cutoff = performance.now() - 1000;
  state.charsWindow = state.charsWindow.filter(([t]) => t > cutoff);
  const chars = state.charsWindow.reduce((sum, [, n]) => sum + n, 0);
  const tokens = Math.round(chars / 4);            // ~4 chars per token, close enough for a gauge
  tweenNum($("statRate"), tokens);

  if (!sweeping && gTok) gTok.target = tokens;
  scope.history.push(tokens);
  scope.history.shift();

  state.rate.push(tokens);
  if (state.rate.length > 40) state.rate.shift();
  const peak = Math.max(8, ...state.rate);
  $("ekgLine").setAttribute("points", state.rate.map((v, i) =>
    `${(i / 39) * 120},${26 - (v / peak) * 22}`).join(" "));
}

/* ── stream connection ──────────────────────────────────────────────────── */

let source = null;

function connect() {
  source = new EventSource("/api/stream");

  source.onmessage = (message) => {
    const frame = JSON.parse(message.data);
    if (frame.type === "snapshot") {
      state.runs = new Map(frame.data.runs.map((r) => [r.id, r]));
      state.order = frame.data.runs.map((r) => r.id);
      state.config = frame.data.config;
      state.ollama = frame.data.ollama;
      if (!state.selected || !state.runs.has(state.selected)) state.selected = state.order.at(-1) || null;
      renderConfig(); renderOllama(); renderAll(); renderStream(true);
      const pending = state.order.map((id) => state.runs.get(id)).find((r) => r?.pending_approval);
      if (pending) showApproval(pending.pending_approval);
    } else if (frame.type === "ollama") {
      state.ollama = frame.data;
      renderOllama();
    } else if (frame.type === "notice") {
      toast(frame.data.level, frame.data.title, frame.data.body);
    } else if (frame.type === "event") {
      const run = applyEvent(frame.data);
      if (frame.data.kind === "run_started") { state.selected = run.id; renderStream(true); }
      notifyForEvent(frame.data, run);
      if (frame.data.kind === "approval_requested") showApproval(run.pending_approval);
      if (frame.data.kind === "approval_resolved") $("approvalShade").hidden = true;
      if (frame.data.kind === "token") {
        renderStream();
        if (run.id === state.selected) renderGraph();
        if (!document.querySelector('.panel[data-panel="transcript"]').classList.contains("is-active")) {
          $("tabDot").dataset.on = "1";
        }
      } else renderAll();
    }
  };

  source.onerror = () => {
    $("pillOllama").dataset.state = "down";
    $("ollamaState").textContent = "no server";
    source.close();
    setTimeout(connect, 2500);
  };
}

/* ── wiring ─────────────────────────────────────────────────────────────── */

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const SUGGESTIONS = [
  "Explain what this project's judge stage actually does",
  "Compare mistral-nemo:12b and qwen2.5:14b-instruct for tool use",
  "What is the difference between SSE and WebSockets?",
  "Summarise the README in five bullet points",
];

let gTok = null, gVram = null, gAccept = null, gJudge = null;

function init() {
  document.documentElement.dataset.theme = localStorage.getItem("theme") || "slate";

  gTok    = makeGauge($("instTok"),    { label: "tokens / sec", max: 60,  decimals: 0 });
  gVram   = makeGauge($("instVram"),   { label: "vram resident", unit: "GB", max: 24, decimals: 1 });
  gAccept = makeGauge($("instAccept"), { label: "first-pass accept", unit: "%", max: 100, decimals: 0 });
  gJudge  = makeGauge($("instJudge"),  { label: "last judge score", max: 5, decimals: 1, majors: 5 });
  requestAnimationFrame(animateInstruments);
  requestAnimationFrame(drawScope);

  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => selectTab(tab.dataset.tab)));
  selectTab("dispatch");
  window.addEventListener("resize", () => selectTab(document.querySelector(".tab.is-active").dataset.tab));
  $("sweepBtn").addEventListener("click", demoSweep);
  $("themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "slate" ? "paper" : "slate";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
    toast("muted", next === "paper" ? "Cold-press paper" : "Graphite paper", "");
  });

  $("suggestions").innerHTML = SUGGESTIONS.map((s) => `<button class="sug" type="button">${esc(s)}</button>`).join("");
  $("suggestions").querySelectorAll(".sug").forEach((b) =>
    b.addEventListener("click", () => { $("askInput").value = b.textContent; $("askInput").focus(); }));

  $("askForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = $("askInput").value.trim();
    if (!question) return;
    $("askInput").value = "";
    $("askBtn").disabled = true;
    selectTab("transcript");
    setTimeout(() => ($("askBtn").disabled = false), 900);
    try {
      const response = await fetch("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) toast("error", "Dispatch rejected", await response.text());
    } catch (err) {
      toast("error", "Dispatch failed", String(err));
    }
  });

  $("followBtn").addEventListener("click", (event) => {
    const on = event.target.dataset.on === "1" ? "0" : "1";
    event.target.dataset.on = on;
  });

  $("apvAllow").addEventListener("click", () => resolveApproval(true));
  $("apvDeny").addEventListener("click", () => resolveApproval(false));
  // Escape denies rather than merely closing: leaving the modal without an
  // answer would strand the run until the server's approval timeout.
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("approvalShade").hidden) resolveApproval(false);
  });

  new ResizeObserver(layoutArc).observe($("graph"));
  window.addEventListener("resize", layoutArc);
  setInterval(tickRate, 500);
  setInterval(renderHistory, 15000);   // keeps the relative timestamps honest

  renderGraph();
  connect();
}

init();
