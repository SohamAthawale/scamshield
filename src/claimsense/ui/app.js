// ClaimSense dashboard: vanilla JS, one SSE stream, role-based views.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const money = (v) => "$" + Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
const short = (t) => (t ? t.slice(0, 9) + "…" : "–");
const LABEL = { PAY: "Pay", RETURN_WITH_FIX: "Return with fix", HOLD_FOR_REVIEW: "Hold for review" };
const chip = (a) => `<span class="chip ${a}">${LABEL[a] || a}</span>`;

let session = null;
try { session = JSON.parse(sessionStorage.getItem("cs_session") || "null"); } catch (_) { session = null; }
let source = null;
let chart = null;
const notesCache = new Map(); // claim_token -> {kind: note}

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json", Authorization: `Bearer ${session.token}`, ...(opts.headers || {}) } });
  if (r.status === 401) { logout(); throw new Error("signed out"); }
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

// ------------------------------------------------------------------ login
$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  $("login-error").textContent = "";
  try {
    const r = await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: f.get("username"), password: f.get("password") }) });
    if (!r.ok) throw new Error("Wrong username or password");
    const d = await r.json();
    session = { token: d.access_token, role: d.role, provider: d.provider_token, user: f.get("username") };
    try { sessionStorage.setItem("cs_session", JSON.stringify(session)); } catch (_) { /* private mode */ }
    start();
  } catch (err) { $("login-error").textContent = err.message; }
});

function logout() {
  try { sessionStorage.removeItem("cs_session"); } catch (_) { /* ignore */ }
  session = null;
  if (source) source.close();
  location.reload();
}
$("logout").addEventListener("click", logout);

// ------------------------------------------------------------------ boot
async function start() {
  $("login-view").hidden = true;
  $("who").hidden = false;
  $("role-badge").textContent = `${session.user} · ${session.role}`;
  fetch("/health").then((r) => r.json()).then((h) => {
    const b = $("backend-badge");
    b.textContent = h.anomaly_backend === "torch_nnpa" ? "Telum II NNPA" : h.anomaly_backend;
    b.title = `anomaly model: ${h.anomaly_backend}, provider model: ${h.provider_backend}`;
    if (h.anomaly_backend === "torch_nnpa") b.classList.add("nnpa");
  });
  const role = session.role;
  $("kpis").hidden = role === "provider";
  $(`${role}-view`).hidden = false;
  if (role === "investigator") { initChart(); loadCases(); loadLeader(); setInterval(loadLeader, 10000); }
  if (role === "provider") loadReturns();
  if (role === "auditor") { loadAudit(); setInterval(loadAudit, 5000); }
  connect();
}

function connect() {
  source = new EventSource(`/stream?token=${encodeURIComponent(session.token)}`);
  source.addEventListener("metrics", (e) => renderMetrics(JSON.parse(e.data)));
  source.addEventListener("claim", (e) => onClaim(JSON.parse(e.data)));
  source.addEventListener("note", (e) => onNote(JSON.parse(e.data)));
  source.onerror = () => { /* EventSource reconnects by itself */ };
}

// ------------------------------------------------------------------ metrics
function renderMetrics(m) {
  $("k-total").textContent = m.claims_total.toLocaleString();
  $("k-rate").textContent = m.claims_per_sec;
  $("k-lat").textContent = `${m.latency_p50_ms} / ${m.latency_p99_ms}`;
  const d = m.decisions;
  $("k-mix").textContent = `${(d.PAY || 0).toLocaleString()} · ${(d.RETURN_WITH_FIX || 0).toLocaleString()} · ${(d.HOLD_FOR_REVIEW || 0).toLocaleString()}`;
  $("k-held").textContent = money(m.amount_held);
  if (chart) {
    const t = new Date().toLocaleTimeString();
    chart.data.labels.push(t);
    chart.data.datasets[0].data.push(m.claims_per_sec);
    chart.data.datasets[1].data.push(m.latency_p99_ms);
    if (chart.data.labels.length > 90) { chart.data.labels.shift(); chart.data.datasets.forEach((s) => s.data.shift()); }
    chart.update("none");
  }
}

function initChart() {
  if (!window.Chart) return;
  const css = getComputedStyle(document.documentElement);
  chart = new Chart($("rate-chart"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "claims/sec", data: [], borderColor: css.getPropertyValue("--brand").trim(), yAxisID: "y", pointRadius: 0, tension: .25 },
      { label: "p99 latency (ms)", data: [], borderColor: css.getPropertyValue("--ret").trim(), yAxisID: "y1", pointRadius: 0, tension: .25 },
    ] },
    options: { animation: false, maintainAspectRatio: false, interaction: { intersect: false },
      plugins: { legend: { labels: { boxWidth: 10, color: css.getPropertyValue("--muted").trim() } } },
      scales: { x: { display: false }, y: { beginAtZero: true, position: "left", ticks: { color: css.getPropertyValue("--muted").trim() } },
        y1: { beginAtZero: true, position: "right", grid: { drawOnChartArea: false }, ticks: { color: css.getPropertyValue("--muted").trim() } } } },
  });
}

// ------------------------------------------------------------------ stream events
function onClaim(c) {
  if (session.role === "investigator") {
    const tr = document.createElement("tr");
    if (c.action !== "PAY") tr.className = "flash";
    tr.innerHTML = `<td class="tok">${esc(short(c.claim_token))}</td><td class="tok">${esc(short(c.provider_token))}</td>
      <td class="num">${money(c.amount)}</td><td>${chip(c.action)}</td><td class="num">${Number(c.latency_ms).toFixed(1)}</td>`;
    const feed = $("feed");
    feed.prepend(tr);
    while (feed.children.length > 60) feed.lastChild.remove();
    if (c.action === "HOLD_FOR_REVIEW") debounce("cases", loadCases, 800);
  }
  if (session.role === "provider" && c.action !== "PAY") debounce("returns", loadReturns, 800);
}

function onNote(n) {
  const entry = notesCache.get(n.claim_token) || {};
  entry[n.kind] = n;
  notesCache.set(n.claim_token, entry);
  const el = document.querySelector(`[data-note="${n.claim_token}:${n.kind}"]`);
  if (el) el.innerHTML = noteHtml(n);
}

const timers = {};
function debounce(key, fn, ms) { clearTimeout(timers[key]); timers[key] = setTimeout(fn, ms); }

function noteHtml(n) {
  if (!n) return `<span class="src">explanation pending</span>`;
  const src = n.source === "granite" ? "written by IBM Granite on LinuxONE" : "template (Granite still working or unavailable)";
  return `<span class="src">${src}</span>${esc(n.text)}`;
}

function reasonList(r) {
  const items = [];
  (r.rules || []).forEach((x) => items.push(`<li><b>${esc(x.title)}</b>. ${esc(x.fix)}</li>`));
  (r.claim || []).forEach((x) => items.push(`<li>${esc(x.label)}: <b>${esc(x.value)}</b> (${esc(x.direction)})</li>`));
  (r.provider || []).forEach((x) => items.push(`<li>Provider ${esc(x.label)}: <b>${esc(x.value)}</b>${x.peer_percentile != null ? ` (${Math.round(x.peer_percentile)}th pct of peers)` : ""}</li>`));
  return items.length ? `<ul class="reasons">${items.join("")}</ul>` : "";
}

// ------------------------------------------------------------------ investigator
async function loadCases() {
  const cases = await api("/cases?status=open");
  $("case-count").textContent = cases.length;
  $("cases").innerHTML = cases.map((c) => `
    <li data-token="${esc(c.claim_token)}">
      <div class="case-head"><span class="tok">${esc(short(c.claim_token))} · provider ${esc(short(c.provider_token))}</span>
        <span>${money(c.amount)} · risk <b>${(c.provider_risk ?? 0).toFixed(2)}</b></span></div>
      ${reasonList(c.reasons)}
    </li>`).join("");
  document.querySelectorAll("#cases li[data-token]").forEach((li) => li.addEventListener("click", () => openCase(li.dataset.token)));
}

async function openCase(token) {
  const d = await api(`/cases/${encodeURIComponent(token)}`);
  const brief = (notesCache.get(token) || {}).brief || (d.notes.brief ? { ...d.notes.brief, claim_token: token, kind: "brief" } : null);
  const p = d.payload || {};
  const kv = [
    ["Decision", chip(d.action)], ["Amount", money(d.amount)], ["Service date", esc(d.service_date)],
    ["Provider (decrypted)", esc(p.Provider)], ["Claim id (decrypted)", esc(p.ClaimID)],
    ["Attending physician", esc(p.AttendingPhysician || "missing")], ["Primary diagnosis", esc(p.ClmDiagnosisCode_1 || "–")],
    ["Provider risk", (d.provider_risk ?? 0).toFixed(3)], ["Claim anomaly score", d.anomaly.toFixed(2)],
    ["Provider claims seen", d.provider_claims], ["Model backend", esc(d.backend)],
  ].map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("");
  $("drawer-body").innerHTML = `
    <h2>Case ${esc(short(token))}</h2>
    <p class="muted">Opening this case decrypted the claim and wrote an audit entry.</p>
    <div class="kv">${kv}</div>
    <h3>Why it was held</h3>${reasonList(d.reasons)}
    <h3>Investigator brief</h3><div class="note" data-note="${esc(token)}:brief">${noteHtml(brief)}</div>
    ${d.case && d.case.status !== "open" ? `<p><b>Resolved:</b> ${esc(d.case.status)} by ${esc(d.case.resolved_by)}</p>` : `
    <div class="actions">
      <button class="release" data-s="released">Release payment</button>
      <button class="more" data-s="needs_more_info">Request records</button>
      <button class="confirm" data-s="confirmed_issue">Confirm issue</button>
    </div>`}`;
  $("drawer").hidden = false;
  document.querySelectorAll("#drawer .actions button").forEach((b) => b.addEventListener("click", async () => {
    await api(`/cases/${encodeURIComponent(token)}/resolve`, { method: "POST", body: JSON.stringify({ status: b.dataset.s }) });
    $("drawer").hidden = true;
    loadCases();
  }));
}
$("drawer-close").addEventListener("click", () => { $("drawer").hidden = true; });

async function loadLeader() {
  const rows = await api("/providers/leaderboard");
  $("leader").innerHTML = rows.map((r) => `<tr><td class="tok">${esc(short(r.provider))}</td><td class="num">${r.risk.toFixed(2)}</td>
    <td class="num">${r.claims}</td><td>${(r.reasons || []).map((x) => `${esc(x.label)} ${esc(x.value)}${x.peer_percentile != null ? ` (p${Math.round(x.peer_percentile)})` : ""}`).join(" · ")}</td></tr>`).join("");
}

// ------------------------------------------------------------------ provider
async function loadReturns() {
  const d = await api("/my/claims");
  $("returns").innerHTML = d.claims.map((c) => {
    const note = (notesCache.get(c.claim_token) || {}).correction || (c.note ? { ...c.note, claim_token: c.claim_token, kind: "correction" } : null);
    const body = c.action === "HOLD_FOR_REVIEW"
      ? `<p class="muted">This claim is being checked by a reviewer. No action is needed from you right now.</p>`
      : `${reasonList(c.reasons)}<div class="note" data-note="${esc(c.claim_token)}:correction">${noteHtml(note)}</div>`;
    return `<li><div class="case-head"><span>${chip(c.action)} <span class="tok">${esc(short(c.claim_token))}</span></span>
      <span>${money(c.amount)} · ${esc(c.service_date)}</span></div>${body}</li>`;
  }).join("") || `<li class="muted">Nothing to fix. All your claims were paid.</li>`;
}

// ------------------------------------------------------------------ auditor
$("verify").addEventListener("click", async () => {
  const el = $("verify-result");
  el.textContent = "Verifying…"; el.className = "verify";
  const r = await api("/audit/verify");
  el.className = "verify " + (r.ok ? "ok" : "bad");
  el.textContent = r.ok ? `✔ Chain intact: ${r.checked.toLocaleString()} entries verified in ${r.seconds}s`
    : `✘ Chain broken at entry #${r.first_bad_id}: ${r.reason}`;
  loadAudit();
});

async function loadAudit() {
  const [tail, agg] = await Promise.all([api("/audit/tail?limit=40"), api("/audit/aggregates")]);
  $("audit").innerHTML = tail.map((a) => `<tr><td>${a.id}</td><td>${esc(a.ts.slice(11, 23))}</td><td>${esc(a.actor)}</td>
    <td>${esc(a.action)}</td><td class="mono">${esc(a.hash.slice(0, 12))}…</td></tr>`).join("");
  $("aggregates").textContent = JSON.stringify(agg, null, 2);
}

if (session) start();
