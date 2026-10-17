// ScamShield dashboard: vanilla JS, one SSE stream, role-based views.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const money = (v) => "$" + Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
const short = (t) => (t ? t.slice(0, 9) + "…" : "–");
const LABEL = { ALLOW: "Allow", WARN: "Warn customer", HOLD: "Hold for call-back" };
const chip = (a) => `<span class="chip ${a}">${LABEL[a] || a}</span>`;

let session = null;
try { session = JSON.parse(sessionStorage.getItem("ss_session") || "null"); } catch (_) { session = null; }
let source = null;
let chart = null;
const notesCache = new Map(); // payment_token -> {kind: note}

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json", Authorization: `Bearer ${session.token}`, ...(opts.headers || {}) } });
  if (r.status === 401) { logout(); throw new Error("signed out"); }
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  $("login-error").textContent = "";
  try {
    const r = await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: f.get("username"), password: f.get("password") }) });
    if (!r.ok) throw new Error("Wrong username or password");
    const d = await r.json();
    session = { token: d.access_token, role: d.role, account: d.account_token, user: f.get("username") };
    try { sessionStorage.setItem("ss_session", JSON.stringify(session)); } catch (_) { /* private mode */ }
    start();
  } catch (err) { $("login-error").textContent = err.message; }
});

function logout() {
  try { sessionStorage.removeItem("ss_session"); } catch (_) { /* ignore */ }
  session = null;
  if (source) source.close();
  location.reload();
}
$("logout").addEventListener("click", logout);

async function start() {
  $("login-view").hidden = true;
  $("who").hidden = false;
  $("role-badge").textContent = `${session.user} · ${session.role}`;
  fetch("/health").then((r) => r.json()).then((h) => {
    const b = $("backend-badge");
    b.textContent = h.anomaly_backend === "torch_nnpa" ? "Telum II NNPA ready" : h.anomaly_backend;
    b.title = `anomaly model: ${h.anomaly_backend}, risk model: ${h.risk_backend}, history: ${h.history_payments} payments`;
    if (h.anomaly_backend === "torch_nnpa") b.classList.add("nnpa");
  });
  const role = session.role;
  $("kpis").hidden = role === "customer";
  $(`${role}-view`).hidden = false;
  if (role === "analyst") { initChart(); loadCases(); loadWatch(); setInterval(loadWatch, 8000); }
  if (role === "customer") loadReturns();
  if (role === "auditor") { loadAudit(); setInterval(loadAudit, 5000); }
  connect();
}

function connect() {
  source = new EventSource(`/stream?token=${encodeURIComponent(session.token)}`);
  source.addEventListener("metrics", (e) => renderMetrics(JSON.parse(e.data)));
  source.addEventListener("payment", (e) => onPayment(JSON.parse(e.data)));
  source.addEventListener("note", (e) => onNote(JSON.parse(e.data)));
}

function renderMetrics(m) {
  $("k-total").textContent = m.payments_total.toLocaleString();
  $("k-rate").textContent = m.payments_per_sec;
  $("k-lat").textContent = `${m.latency_p50_ms} / ${m.latency_p99_ms}`;
  const d = m.decisions;
  $("k-mix").textContent = `${(d.ALLOW || 0).toLocaleString()} · ${(d.WARN || 0).toLocaleString()} · ${(d.HOLD || 0).toLocaleString()}`;
  $("k-held").textContent = money(m.usd_held);
  // the accelerator only takes calls of nnpa_min_batch or more, so say how many payments really used it
  $("k-nnpa").textContent = m.nnpa_min_batch ? `${(100 * (m.nnpa_share || 0)).toFixed(1)}%` : "off";
  $("k-nnpa-hint").textContent = m.nnpa_min_batch
    ? `${(m.payments_on_nnpa || 0).toLocaleString()} payments; calls of ${m.nnpa_min_batch.toLocaleString()}+ (largest batch ${m.max_batch_size || 0})`
    : "CPU only";
  if (chart) {
    chart.data.labels.push(new Date().toLocaleTimeString());
    chart.data.datasets[0].data.push(m.payments_per_sec);
    chart.data.datasets[1].data.push(m.latency_p99_ms);
    if (chart.data.labels.length > 90) { chart.data.labels.shift(); chart.data.datasets.forEach((s) => s.data.shift()); }
    chart.update("none");
  }
}

function initChart() {
  if (!window.Chart) return;
  const css = getComputedStyle(document.documentElement);
  const muted = css.getPropertyValue("--muted").trim();
  chart = new Chart($("rate-chart"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "payments/sec", data: [], borderColor: css.getPropertyValue("--brand").trim(), yAxisID: "y", pointRadius: 0, tension: .25 },
      { label: "p99 latency (ms)", data: [], borderColor: css.getPropertyValue("--ret").trim(), yAxisID: "y1", pointRadius: 0, tension: .25 },
    ] },
    options: { animation: false, maintainAspectRatio: false, interaction: { intersect: false },
      plugins: { legend: { labels: { boxWidth: 10, color: muted } } },
      scales: { x: { display: false }, y: { beginAtZero: true, position: "left", ticks: { color: muted } },
        y1: { beginAtZero: true, position: "right", grid: { drawOnChartArea: false }, ticks: { color: muted } } } },
  });
}

function onPayment(p) {
  if (session.role === "analyst") {
    const tr = document.createElement("tr");
    if (p.action !== "ALLOW") tr.className = "flash";
    tr.innerHTML = `<td class="tok">${esc(short(p.payment_token))}</td><td class="tok">${esc(short(p.sender_token))} → ${esc(short(p.receiver_token))}</td>
      <td class="num">${money(p.amount_usd)}</td><td>${esc(p.payment_format)}</td><td>${chip(p.action)}</td><td class="num">${Number(p.latency_ms).toFixed(1)}</td>`;
    const feed = $("feed");
    feed.prepend(tr);
    while (feed.children.length > 60) feed.lastChild.remove();
    if (p.action === "HOLD") debounce("cases", loadCases, 800);
  }
  if (session.role === "customer" && p.action !== "ALLOW") debounce("returns", loadReturns, 500);
}

function onNote(n) {
  const entry = notesCache.get(n.payment_token) || {};
  entry[n.kind] = n;
  notesCache.set(n.payment_token, entry);
  const el = document.querySelector(`[data-note="${n.payment_token}:${n.kind}"]`);
  if (el) el.innerHTML = noteHtml(n);
}

const timers = {};
function debounce(key, fn, ms) { clearTimeout(timers[key]); timers[key] = setTimeout(fn, ms); }

function noteHtml(n) {
  if (!n) return `<span class="src">message pending</span>`;
  const src = n.source === "granite" ? "written by IBM Granite on LinuxONE" : "instant message (Granite rewrite pending or unavailable)";
  return `<span class="src">${src}</span>${esc(n.text)}`;
}

function reasonList(r) {
  const items = [];
  (r.rules || []).forEach((x) => items.push(`<li><b>${esc(x.title)}</b>. ${esc(x.advice)}</li>`));
  (r.customer || []).forEach((x) => items.push(`<li>Unusual for this customer: ${esc(x.label)}${x.value ? ` <b>${esc(x.value)}</b>` : ""} (${esc(x.direction)})</li>`));
  (r.drivers || []).forEach((x) => items.push(`<li>Risk driver: ${esc(x.label)}${x.value ? ` <b>${esc(x.value)}</b>` : ""}</li>`));
  return items.length ? `<ul class="reasons">${items.join("")}</ul>` : "";
}

async function loadCases() {
  const cases = await api("/cases?status=open");
  $("case-count").textContent = cases.length;
  $("cases").innerHTML = cases.map((c) => `
    <li data-token="${esc(c.payment_token)}">
      <div class="case-head"><span class="tok">${esc(short(c.payment_token))} · ${esc(c.payment_format)}</span>
        <span>${money(c.amount_usd)} · risk <b>${c.risk.toFixed(2)}</b></span></div>
      ${reasonList(c.reasons)}
    </li>`).join("");
  document.querySelectorAll("#cases li[data-token]").forEach((li) => li.addEventListener("click", () => openCase(li.dataset.token)));
}

async function openCase(token) {
  const d = await api(`/cases/${encodeURIComponent(token)}`);
  const brief = (notesCache.get(token) || {}).brief || (d.notes.brief ? { ...d.notes.brief, payment_token: token, kind: "brief" } : null);
  const p = d.payload || {};
  const kv = [
    ["Decision", chip(d.action)], ["Amount", `${money(d.amount_usd)} (${esc(p.amount_paid)} ${esc(p.currency_paid)})`],
    ["Format", esc(d.payment_format)], ["Paid at", esc(d.paid_at)],
    ["From (decrypted)", `${esc(p.from_bank)} / ${esc(p.from_acct)}`], ["To (decrypted)", `${esc(p.to_bank)} / ${esc(p.to_acct)}`],
    ["Risk score", d.risk.toFixed(3)], ["Customer anomaly", d.anomaly.toFixed(2)], ["Model backend", esc(d.backend)],
  ].map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("");
  $("drawer-body").innerHTML = `
    <h2>Paused payment ${esc(short(token))}</h2>
    <p class="muted">Opening this case decrypted the payment and wrote an audit entry.</p>
    <div class="kv">${kv}</div>
    <h3>Why it was paused</h3>${reasonList(d.reasons)}
    <h3>Analyst brief</h3><div class="note" data-note="${esc(token)}:brief">${noteHtml(brief)}</div>
    ${d.case && d.case.status !== "open" ? `<p><b>Resolved:</b> ${esc(d.case.status)} by ${esc(d.case.resolved_by)}</p>` : `
    <div class="actions">
      <button class="release" data-s="released">Customer confirmed: release</button>
      <button class="more" data-s="contact_customer">Can't reach customer yet</button>
      <button class="stop" data-s="stopped">Scam confirmed: stop payment</button>
    </div>`}`;
  $("drawer").hidden = false;
  document.querySelectorAll("#drawer .actions button").forEach((b) => b.addEventListener("click", async () => {
    await api(`/cases/${encodeURIComponent(token)}/resolve`, { method: "POST", body: JSON.stringify({ status: b.dataset.s }) });
    $("drawer").hidden = true;
    loadCases();
  }));
}
$("drawer-close").addEventListener("click", () => { $("drawer").hidden = true; });

async function loadWatch() {
  const rows = await api("/accounts/watchlist");
  $("watch").innerHTML = rows.map((r) => `<tr><td class="tok">${esc(short(r.account))}</td><td class="num">${r.mean_risk.toFixed(2)}</td>
    <td class="num">${r.max_risk.toFixed(2)}</td><td class="num">${r.payments}</td><td class="num">${r.flagged}</td></tr>`).join("");
}

async function loadReturns() {
  const d = await api("/my/payments");
  $("returns").innerHTML = d.payments.map((c) => {
    const note = (notesCache.get(c.payment_token) || {}).warning || (c.message ? { ...c.message, payment_token: c.payment_token, kind: "warning" } : null);
    return `<li><div class="case-head"><span>${chip(c.action)} <span class="tok">${esc(short(c.payment_token))}</span></span>
      <span>${money(c.amount_usd)} · ${esc(c.payment_format)} · ${esc(c.paid_at)}</span></div>
      <div class="note" data-note="${esc(c.payment_token)}:warning">${noteHtml(note)}</div></li>`;
  }).join("") || `<li class="muted">Nothing to check. All your payments went through.</li>`;
}

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
