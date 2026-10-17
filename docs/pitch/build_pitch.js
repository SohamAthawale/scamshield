// ScamShield pitch deck (pptx) with a word-for-word recording script in the speaker notes.
// node build.js <out.pptx> <chart.png> <skill-dir>
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const [OUT, CHART, SKILL] = process.argv.slice(2);
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const HEX = {
  navy: "0E1A2B", white: "FFFFFF", grey: "46505E", paper: "F5F6F2", blue: "2456C9", orange: "B04E14",
  ink: "14213D", line: "DDE1D8", sky: "8EB1FF", card: "1C2D45", muted: "6B7482", soft: "B8C4D6",
};
const THEME = {
  name: "ScamShield",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.navy, lt1: HEX.white, dk2: HEX.grey, lt2: HEX.paper,
    accent1: HEX.blue, accent2: HEX.orange, accent3: HEX.ink, accent4: HEX.line, accent5: HEX.sky, accent6: HEX.card,
    hlink: HEX.blue, folHlink: HEX.muted,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "ScamShield — IBM Z Datathon 2026";
pres.author = "Team IBMBoB";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
// C.text1 navy, C.text2 grey, C.background1 white, C.background2 paper,
// C.accent1 blue, accent2 orange, accent3 ink, accent4 line, accent5 sky, accent6 dark card

const W = 13.333, M = 0.6, CW = W - 2 * M;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background2 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.42, w: CW, h: 0.36, fontSize: 13, bold: true,
      color: C.accent1, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.78, w: CW, h: 0.8, fontSize: 36, bold: true,
      color: C.text1, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "source", type: "body", x: M, y: 6.95, w: CW - 0.8, h: 0.3, fontSize: 11,
      color: C.text2, margin: 0, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.6, y: 6.95, w: 0.6, h: 0.3, fontSize: 11, color: C.text2, align: "right" },
});
pres.defineSlideMaster({
  title: "Dark",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "source", type: "body", x: M, y: 6.95, w: CW - 0.8, h: 0.3, fontSize: 11,
      color: C.accent5, margin: 0, valign: "middle" }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
async function icon(Comp, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function card(slide, x, y, w, h, name, fill = C.background1) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: C.accent4, width: 1 }, objectName: name });
}
function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, valign: "top", fontSize: 16, color: C.text2, ...o });
}
function notes(time, body) {
  return `TIME ${time}\n\n${body.trim()}\n`;
}

(async () => {
  const I = {
    users: await icon(fa.FaUserFriends, HEX.blue), search: await icon(fa.FaSearch, HEX.orange),
    check: await icon(fa.FaCheckCircle, HEX.blue), warn: await icon(fa.FaExclamationTriangle, HEX.orange),
    key: await icon(fa.FaKey, HEX.blue), lock: await icon(fa.FaLock, HEX.blue), roles: await icon(fa.FaUserShield, HEX.blue),
    chain: await icon(fa.FaLink, HEX.orange), send: await icon(fa.FaPaperPlane, HEX.grey),
    chart: await icon(fa.FaChartLine, HEX.sky), user: await icon(fa.FaUser, HEX.sky), audit: await icon(fa.FaClipboardCheck, HEX.sky),
  };

  // ================================================================== 1 cover
  pres.addSection({ title: "Why" });
  let s = pres.addSlide({ masterName: "Dark", sectionTitle: "Why" });
  text(s, "IBM Z DATATHON 2026 · THEME 1: REAL-TIME AI FOR CRITICAL DECISIONS",
    { x: M, y: 1.35, w: CW, h: 0.4, fontSize: 14, bold: true, color: C.accent5, charSpacing: 2 });
  text(s, "ScamShield", { x: M, y: 1.95, w: CW, h: 1.3, fontSize: 72, bold: true, color: C.background1 });
  text(s, [{ text: "Every payment checked " }, { text: "before the money moves", options: { bold: true } },
    { text: ", in milliseconds, on IBM LinuxONE." }],
    { x: M, y: 3.35, w: 10.5, h: 0.95, fontSize: 26, color: C.background1 });
  text(s, "Risky payments get a clear warning. Mule-account patterns are paused for a human call-back.",
    { x: M, y: 4.35, w: 10.5, h: 0.5, fontSize: 18, color: C.accent5 });
  ["IBM z17 · Telum II NNPA", "PyTorch + XGBoost + IBM Granite", "100% of payments checked"].forEach((t, i) => {
    const w = [3.0, 3.75, 3.2][i], x = M + [0, 3.2, 7.15][i];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.2, w, h: 0.5, rectRadius: 0.25, fill: { color: C.accent6 },
      line: { color: C.accent6 }, objectName: `pill ${i + 1}` });
    text(s, t, { x, y: 5.2, w, h: 0.5, fontSize: 14, color: C.background1, align: "center", valign: "middle" });
  });
  s.addText("Live on the IBM LinuxONE Community Cloud · Team IBMBoB", { placeholder: "source" });
  s.addNotes(notes("0:00–0:15", `
SAY:
"Hi, we're Team IBMBoB, and this is ScamShield. It checks every payment before the money moves, in milliseconds, running live on an IBM z17 in the LinuxONE Community Cloud. We're in Theme 1: Real-Time AI for Critical Decisions."
`));

  // ================================================================== 2 problem
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Why" });
  s.addText("THE PROBLEM", { placeholder: "kicker" });
  s.addText("Scam money is gone in minutes", { placeholder: "title" });
  text(s, "$2.03B", { x: M, y: 1.95, w: 5.6, h: 1.1, fontSize: 66, bold: true, color: C.accent2 });
  text(s, [{ text: "lost by Australians to scams " }, { text: "in 2024", options: { bold: true } },
    { text: ", across 494,732 reports." }], { x: M, y: 3.1, w: 5.6, h: 0.7, fontSize: 18, color: C.accent3 });
  text(s, [{ text: "The money moves through " }, { text: "mule accounts", options: { bold: true, color: C.text1 } },
    { text: " within minutes and is rarely recovered. The " },
    { text: "Scams Prevention Framework", options: { bold: true, color: C.text1 } },
    { text: " (2025) requires banks to detect, disrupt and respond." }],
    { x: M, y: 3.95, w: 5.6, h: 1.3, fontSize: 16 });
  text(s, [{ text: "ScamShield's measurable job: " }, { text: "break those mule chains", options: { bold: true, color: C.text1 } },
    { text: " before the money moves." }], { x: M, y: 5.4, w: 5.6, h: 0.8, fontSize: 16 });
  [[I.users, "PRIMARY PERSONA", "Bank customer", "Being talked into sending money. Needs a clear warning before it leaves.", C.accent1],
   [I.search, "SECONDARY PERSONA", "Bank scam analyst", "Limited caseload. Needs the few risky payments, with evidence.", C.accent2]]
    .forEach(([ic, k, h, b, col], i) => {
      const y = 1.95 + i * 2.25;
      card(s, 6.9, y, 5.83, 2.0, `persona ${i + 1}`);
      s.addImage({ data: ic, x: 7.2, y: y + 0.32, w: 0.6, h: 0.6, objectName: `persona icon ${i + 1}` });
      text(s, k, { x: 8.1, y: y + 0.28, w: 4.4, h: 0.3, fontSize: 12, bold: true, color: col, charSpacing: 1 });
      text(s, h, { x: 8.1, y: y + 0.6, w: 4.4, h: 0.5, fontSize: 24, bold: true, color: C.text1 });
      text(s, b, { x: 8.1, y: y + 1.15, w: 4.4, h: 0.7, fontSize: 15 });
    });
  s.addText("Source: National Anti-Scam Centre / ACCC, Targeting Scams 2024 · proven on IBM AMLworld synthetic payments",
    { placeholder: "source" });
  s.addNotes(notes("0:15–0:45", `
SAY:
"Australians lost 2.03 billion dollars to scams in 2024. The money moves through mule accounts within minutes, and once it's gone, it's rarely recovered. The new Scams Prevention Framework says banks must detect, disrupt and respond.

We built for two people: a customer being talked into sending money, and a scam analyst who needs the few risky payments, with evidence.

One honest note: our public data labels laundering, not scams, so what we measure is breaking the mule chains scam money flows through."
`));

  // ================================================================== 3 solution
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Why" });
  s.addText("THE SOLUTION", { placeholder: "kicker" });
  s.addText("Three actions, decided in milliseconds", { placeholder: "title" });
  [["ALLOW", "98.26%", "Nothing unusual. Goes straight through.", "Nobody needs to act.", C.text2],
   ["WARN", "1.66%", "Risky, out of character, or a big first payment to a new payee.", "Customer gets a specific warning. They decide.", C.accent1],
   ["HOLD", "0.074%", "Strong mule-account match; borderline ones only from $5,000.", "Paused; an analyst calls back on a known number. A human decides.", C.accent2]]
    .forEach(([a, pct, what, who, col], i) => {
      const w = (CW - 0.6) / 3, x = M + i * (w + 0.3), y = 1.85;
      card(s, x, y, w, 3.95, `${a} card`);
      text(s, a, { x: x + 0.35, y: y + 0.3, w: w - 0.7, h: 0.4, fontSize: 18, bold: true, color: col, charSpacing: 2 });
      text(s, pct, { x: x + 0.35, y: y + 0.75, w: w - 0.7, h: 0.85, fontSize: 48, bold: true, color: C.text1 });
      text(s, "of test-day payments", { x: x + 0.35, y: y + 1.6, w: w - 0.7, h: 0.3, fontSize: 13, color: C.text2 });
      text(s, what, { x: x + 0.35, y: y + 2.05, w: w - 0.7, h: 0.9, fontSize: 17, color: C.accent3 });
      text(s, who, { x: x + 0.35, y: y + 3.0, w: w - 0.7, h: 0.75, fontSize: 14, color: C.text2 });
    });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.05, w: CW, h: 0.62, rectRadius: 0.12, fill: { color: C.text1 },
    line: { color: C.text1 }, objectName: "never block strip" });
  text(s, [{ text: "Warnings never block. ", options: { bold: true } },
    { text: "Nothing is reported, frozen or refused automatically." }],
    { x: M + 0.35, y: 6.05, w: CW - 0.7, h: 0.62, fontSize: 17, color: C.background2, valign: "middle" });
  s.addText("Decision mix on unseen test days 9–10 Sep; thresholds tuned on validation days only (docs/MODEL_CARD.md)",
    { placeholder: "source" });
  s.addNotes(notes("0:45–1:05", `
SAY:
"Every payment gets one of three actions.

Allow: nothing unusual, it goes straight through. That's 98 percent of payments.

Warn: it's risky or out of character, so the customer gets a specific warning, and they decide. About one and a half percent.

Hold: a strong mule-account match, or a borderline one of five thousand dollars or more. The payment waits while the bank calls the customer back, and a human decides. That's fewer than one payment in a thousand.

And warnings never block."
`));

  // ================================================================== 4 architecture
  pres.addSection({ title: "How" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "How" });
  s.addText("TECHNICAL DESIGN", { placeholder: "kicker" });
  s.addText("Architecture on IBM LinuxONE", { placeholder: "title" });
  card(s, M, 1.85, 2.3, 3.9, "payment stream");
  s.addImage({ data: I.send, x: M + 0.3, y: 2.25, w: 0.5, h: 0.5, objectName: "stream icon" });
  text(s, "Payment stream", { x: M + 0.3, y: 2.9, w: 1.8, h: 0.75, fontSize: 18, bold: true, color: C.text1 });
  text(s, "Core banking (simulated)", { x: M + 0.3, y: 3.7, w: 1.8, h: 0.7, fontSize: 14 });
  text(s, "POST /payments\n/payments/batch", { x: M + 0.3, y: 4.55, w: 1.9, h: 0.7, fontSize: 12, color: C.accent1, fontFace: "Courier New" });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 3.05, y: 3.6, w: 0.45, h: 0.32, fill: { color: "9AA3AF" }, line: { color: "9AA3AF" }, objectName: "arrow 1" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.6, y: 1.85, w: 5.3, h: 3.9, rectRadius: 0.12, fill: { color: C.background1 },
    line: { color: C.text1, width: 2 }, objectName: "scoring api" });
  text(s, "Scoring API · FastAPI", { x: 3.9, y: 2.05, w: 4.8, h: 0.45, fontSize: 18, bold: true, color: C.text1 });
  const steps = [
    "Pseudonymise account numbers (HMAC-SHA256)",
    "One streaming feature engine: payment, payer and receiving-account (mule) history",
    "Customer-behaviour autoencoder: PyTorch on Telum II NNPA (≥1,024 per call), else CPU",
    "Risk: XGBoost with SHAP drivers",
    "Rule: big first payment to a new payee",
    "Fusion → ALLOW | WARN | HOLD (borderline holds only from $5,000)",
  ];
  const sy = [2.6, 2.95, 3.6, 4.25, 4.6, 4.95];
  steps.forEach((t, i) => {
    if (i === 2) s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.82, y: sy[i] - 0.06, w: 4.92, h: 0.66, rectRadius: 0.08,
      fill: { color: "E3EBFB" }, line: { color: "E3EBFB" }, objectName: "nnpa highlight" });
    text(s, [{ text: `${i + 1}  `, options: { bold: true, color: C.text1 } }, { text: t }],
      { x: 3.9, y: sy[i], w: 4.75, h: i === 1 || i === 2 || i === 5 ? 0.6 : 0.32, fontSize: 13, color: C.accent3 });
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 9.0, y: 3.6, w: 0.45, h: 0.32, fill: { color: "9AA3AF" }, line: { color: "9AA3AF" }, objectName: "arrow 2" });
  [["Encrypted SQLite (WAL)", "AES-256-GCM payloads + hash-chained audit"],
   ["IBM Granite worker", "granite-3.3-2b-instruct, llama.cpp + zDNN, async"],
   ["Live dashboard (SSE)", "Analyst · customer · auditor"]].forEach(([h, b], i) => {
    const y = 1.85 + i * 1.33;
    card(s, 9.55, y, 3.18, 1.2, `store ${i + 1}`);
    text(s, h, { x: 9.8, y: y + 0.18, w: 2.8, h: 0.35, fontSize: 15, bold: true, color: C.text1 });
    text(s, b, { x: 9.8, y: y + 0.55, w: 2.8, h: 0.55, fontSize: 12 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.0, w: CW, h: 0.62, rectRadius: 0.12, fill: { color: C.text1 },
    line: { color: C.text1 }, objectName: "deploy strip" });
  text(s, "Docker Compose · every container built FROM the official L1CC image · non-root · read-only FS · API only on 127.0.0.1 · IBM z17 (type 9175), 3 vCPU, 8 GB",
    { x: M + 0.35, y: 6.0, w: CW - 0.7, h: 0.62, fontSize: 13, color: C.background2, valign: "middle" });
  s.addText("Frameworks: PyTorch 2.11 + torch_nnpa 1.5 · XGBoost 3.4.1 · llama_cpp_python 0.3.36 (zDNN)", { placeholder: "source" });
  s.addNotes(notes("1:05–1:35", `
SAY (on this slide):
"Account numbers become keyed tokens the moment a payment arrives. One streaming feature engine reads the payer's and the receiving account's history, which is where mule behaviour shows. A PyTorch autoencoder asks: is this unusual for this customer? XGBoost scores mule risk, and IBM Granite writes the analyst's brief, off the hot path."

DO: Switch to the VM terminal. Run, one at a time:
  uname -m
  grep Type /proc/sysinfo
  lscpu | grep -io nnpa
  docker compose ps

SAY (while they print):
"This is the real machine: machine type 9175, a z17, with the NNPA flag for the Telum II accelerator, and our containers built from the official LinuxONE image."
`));

  // ================================================================== 5 live demo divider
  s = pres.addSlide({ masterName: "Dark", sectionTitle: "How" });
  text(s, "LIVE DEMO", { x: M, y: 1.5, w: CW, h: 0.4, fontSize: 14, bold: true, color: C.accent5, charSpacing: 2 });
  text(s, "Live on the z17", { x: M, y: 2.0, w: CW, h: 1.0, fontSize: 54, bold: true, color: C.background1 });
  text(s, "Three views of the same stream, each seeing only what its role needs.",
    { x: M, y: 3.05, w: CW, h: 0.5, fontSize: 20, color: C.accent5 });
  [[I.chart, "Analyst", "Live feed, paused payments, Granite brief, mule watchlist"],
   [I.user, "Customer", "Own payments and an instant, specific message"],
   [I.audit, "Auditor", "Aggregates and the tamper-evident audit chain"]].forEach(([ic, h, b], i) => {
    const w = (CW - 0.6) / 3, x = M + i * (w + 0.3), y = 4.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.9, rectRadius: 0.12, fill: { color: C.accent6 },
      line: { color: C.accent6 }, objectName: `view ${h}` });
    s.addImage({ data: ic, x: x + 0.35, y: y + 0.35, w: 0.5, h: 0.5, objectName: `view icon ${h}` });
    text(s, h, { x: x + 1.05, y: y + 0.38, w: w - 1.3, h: 0.45, fontSize: 22, bold: true, color: C.background1 });
    text(s, b, { x: x + 0.35, y: y + 1.05, w: w - 0.7, h: 0.7, fontSize: 15, color: C.accent5 });
  });
  s.addText("Dashboard over an SSH tunnel · every payment scored before it moves", { placeholder: "source" });
  s.addNotes(notes("1:35–3:05  (live demo: analyst, then customer)", `
DO: Switch to the ANALYST browser window.

SAY:
"This is the analyst dashboard. Payments stream in at about four milliseconds each, and every one is checked. Single payments run on the CPU, which is faster for small batches."

DO: Click a PAUSED payment to open it.

SAY:
"This one is paused: a first ACH payment to this payee, with its main risk drivers. Opening it wrote an audit entry. IBM Granite, running here on LinuxONE, wrote this brief from pseudonymised facts only, and it says plainly it's not a finding of fraud."

DO: Click "Customer confirmed: release" (or "Scam confirmed: stop payment"). Point at the mule watchlist.

SAY:
"The analyst makes the call. And this watchlist ranks receiving accounts that look like mules."

DO: Switch to the CUSTOMER browser window.

SAY:
"The customer only sees their own payments, and an instant, specific message: your first payment to this payee, much larger than usual, so check them through a contact you already trust. No scores, no accusations; they decide."
`));

  // ================================================================== 6 security
  pres.addSection({ title: "Trust" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Trust" });
  s.addText("PLATFORM SECURITY", { placeholder: "kicker" });
  s.addText("Security in the core, not bolted on", { placeholder: "title" });
  [[I.key, "Pseudonymisation", "Keyed HMAC-SHA256 tokens for account numbers and payment IDs, on arrival. Models, logs, UI and Granite see only tokens."],
   [I.lock, "AES-256-GCM at rest, on CPACF", "The full raw payment is sealed, with a random 96-bit nonce and key id per record. OpenSSL uses IBM Z's on-core CPACF instructions."],
   [I.roles, "Role-based access (JWT)", "Customer: own payments and messages. Analyst: paused payments, decrypted on open. Auditor: aggregates and audit only."],
   [I.chain, "Tamper-evident audit chain", "Each row stores sha256(prev_hash + row). Edit one row in SQLite and verification reports \"Chain broken at entry #50\"."]]
    .forEach(([ic, h, b], i) => {
      const w = (CW - 0.3) / 2, x = M + (i % 2) * (w + 0.3), y = 1.85 + Math.floor(i / 2) * 1.95;
      card(s, x, y, w, 1.75, `control ${i + 1}`);
      s.addImage({ data: ic, x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, objectName: `control icon ${i + 1}` });
      text(s, h, { x: x + 1.05, y: y + 0.28, w: w - 1.3, h: 0.45, fontSize: 19, bold: true, color: C.text1 });
      text(s, b, { x: x + 1.05, y: y + 0.78, w: w - 1.3, h: 0.85, fontSize: 14 });
    });
  card(s, M, 5.85, CW, 0.85, "also", "E9ECE5");
  text(s, [{ text: "Also: ", options: { bold: true, color: C.accent3 } },
    { text: "non-root, read-only containers · secrets as files, never in images or git · Jupyter never exposed · all AI on-premises.", options: { breakLine: true } },
    { text: "Not claimed: ", options: { bold: true, color: C.accent3 } },
    { text: "quantum-safe signatures (the host OpenSSL 3.0.13 has no ML-DSA)." }],
    { x: M + 0.3, y: 5.93, w: CW - 0.6, h: 0.7, fontSize: 14, valign: "middle" });
  s.addText("docs/SECURITY.md · tests/test_security.py, test_audit.py, test_api.py", { placeholder: "source" });
  s.addNotes(notes("3:05–3:40  (live: auditor view + tamper test)", `
DO: Switch to the AUDITOR browser window. Click "Verify audit chain".

SAY:
"Security is in the core. Every decision, login and case view is hash-chained. Verify: intact."

DO: In the VM terminal (in ~/scamshield), paste and run the tamper command:
  docker compose exec api python -c "import sqlite3; c=sqlite3.connect('/data/scamshield.db'); c.execute('DROP TRIGGER audit_no_update'); c.execute(\\"UPDATE audit SET details='{}' WHERE id=50\\"); c.commit()"
Then click "Verify audit chain" again.

SAY:
"Now I edit one row directly in the database, going around the protections. Verify again: broken, at that exact entry.

Account numbers are pseudonymised on arrival, payloads are AES-256-GCM encrypted on the z17's CPACF crypto, and each role only sees what it needs."
`));

  // ================================================================== 7 results
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("RESULTS", { placeholder: "kicker" });
  s.addText("Two unseen days, scored on the z17", { placeholder: "title" });
  [["73%", "of paused payments are laundering: 0.074% of payments paused, about 660× the base rate", C.accent1],
   ["147/174", "laundering schemes broken (at least one payment paused)", C.accent2],
   ["82%", "of laundering payments warned or paused", C.text1],
   ["$269M", "of $424M laundered, paused before it moved, though 95% of it is ten very large transfers", C.accent2],
   ["0.984", "ROC-AUC, risk model (PR-AUC 0.564, laundering base rate 0.11%)", C.accent1],
   ["4.4 ms", "p50 server latency, live at 274 payments/s on 3 vCPUs", C.text1]].forEach(([n, t, col], i) => {
    const w = (CW - 0.6) / 3, x = M + (i % 3) * (w + 0.3), y = 1.85 + Math.floor(i / 3) * 1.95;
    card(s, x, y, w, 1.75, `stat ${i + 1}`);
    text(s, n, { x: x + 0.3, y: y + 0.2, w: w - 0.6, h: 0.75, fontSize: 40, bold: true, color: col });
    text(s, t, { x: x + 0.3, y: y + 0.95, w: w - 0.6, h: 0.72, fontSize: 13.5 });
  });
  text(s, [{ text: "What this measures: ", options: { bold: true, color: C.text1 } },
    { text: "the data labels laundering, not scams, so these numbers are about breaking the mule-account chains scam money flows through. Thresholds were tuned on two validation days; the test days were scored once." }],
    { x: M, y: 5.85, w: CW, h: 0.8, fontSize: 15 });
  s.addText("z17 VM · test days 9–10 Sep, 862,792 payments · docs/MODEL_CARD.md, docs/evidence/value_hold, docs/evidence/claims",
    { placeholder: "source" });
  s.addNotes(notes("3:40–4:00", `
SAY:
"So, the results. On two unseen days, 862,000 payments, we paused fewer than one payment in a thousand, and three in four of those were laundering. 147 of 174 laundering schemes were broken, and 82 percent of laundering payments were warned or paused.

The money paused is 269 million dollars. To be straight with you, most of that is ten very large transfers, which is why we lead with precision and schemes broken."

IF ASKED (don't read on camera): borderline payments are only paused from $5,000. Versus a plain risk threshold: 641 vs 936 pauses, 73% vs 53% precision, the same $269M, 147 vs 150 schemes. Precision fell from validation to test (81.5% to 73%), so two test days is a short evaluation.
`));

  // ================================================================== 8 hardware
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("TELUM II ON-CHIP AI", { placeholder: "kicker" });
  s.addText("What the Telum II speeds up", { placeholder: "title" });
  card(s, M, 1.85, 7.0, 4.8, "chart frame");
  s.addImage({ path: CHART, x: M + 0.15, y: 1.85 + (4.8 - 6.7 * 560 / 1540) / 2, w: 6.7, h: 6.7 * 560 / 1540,
    altText: "Payment autoencoder on IBM z17: payments per second and p50 time per batch, CPU versus Telum II NNPA, batch 1 to 16,384. CPU faster up to 256; NNPA faster from 1,024.",
    objectName: "nnpa chart" });
  [["The model: up to 1.4×", [{ text: "1.2×", options: { bold: true, color: C.text1 } }, { text: " at 1,024 per call, " },
     { text: "1.4×", options: { bold: true, color: C.text1 } }, { text: " at 4,096. Below ~256 the CPU wins, so single payments stay on the CPU." }]],
   ["Live: bulk files on the NNPA", [{ text: "A 2,048-payment bulk file runs in one call: " },
     { text: "40,960 of 40,960", options: { bold: true, color: C.text1 } }, { text: " payments scored on the Telum II." }]],
   ["End to end: no gain yet", [{ text: "1,562 vs 1,610 payments/s, inside the ±20% noise. " },
     { text: "Next:", options: { bold: true, color: C.text1 } }, { text: " the tree model on the accelerator." }]]]
    .forEach(([h, b], i) => {
      const x = 7.9, y = 1.85 + i * 1.65, w = W - M - x;
      card(s, x, y, w, 1.5, `hw ${i + 1}`);
      text(s, h, { x: x + 0.3, y: y + 0.18, w: w - 0.6, h: 0.4, fontSize: 18, bold: true, color: C.text1 });
      text(s, b, { x: x + 0.3, y: y + 0.62, w: w - 0.6, h: 0.8, fontSize: 14 });
    });
  s.addText("z17 VM, PyTorch 2.11 + torch_nnpa 1.5 · docs/evidence/bench (bench_autoencoder.csv, bulk_file_nnpa_vs_cpu.txt)",
    { placeholder: "source" });
  s.addNotes(notes("4:00–4:20  (live: bulk file)", `
DO: In the VM terminal, run the prepared bulk-file command (DEMO_SCRIPT step 6):
  docker compose --profile demo run --rm --no-deps simulator python -m scamshield.simulator --url http://api:8000 --offset 88000 --limit 4096 --rate 0 --concurrency 1 --inject 0 --file-size 2048
Then show the analyst dashboard's "On Telum II NNPA" tile.

SAY:
"Banks also receive bulk payment files. A 2,048-payment file is scored in one call, and the tile jumps: those payments ran on the Telum II, where our model is up to 1.4 times faster than the CPU.

End to end, honestly, that gain is lost in the rest of the pipeline. So moving the tree model onto the accelerator is our next step."

DON'T claim the whole service is accelerated, or that NNPA speeds up single payments.
`));

  // ================================================================== 9 responsible
  pres.addSection({ title: "Responsible AI" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Responsible AI" });
  s.addText("RESPONSIBLE AI", { placeholder: "kicker" });
  s.addText("Human-led, transparent, honest", { placeholder: "title" });
  const colW = (CW - 0.6) / 3;
  // column 1
  card(s, M, 1.85, colW, 4.8, "human in the loop");
  text(s, "Human in the loop", { x: M + 0.3, y: 2.05, w: colW - 0.6, h: 0.45, fontSize: 19, bold: true, color: C.text1 });
  [["A warning never blocks.", " The customer decides."],
   ["A hold always goes to a human,", " who calls back on a number the bank already has."],
   ["Reasons on every decision.", " Granite sees pseudonymised facts only."]].forEach(([b, t], i) => {
    const y = 2.7 + i * 1.25;
    s.addImage({ data: I.check, x: M + 0.3, y: y + 0.03, w: 0.32, h: 0.32, objectName: `check ${i + 1}` });
    text(s, [{ text: b, options: { bold: true, color: C.text1 } }, { text: t }],
      { x: M + 0.8, y, w: colW - 1.1, h: 1.1, fontSize: 14 });
  });
  // column 2
  const x2 = M + colW + 0.3;
  card(s, x2, 1.85, colW, 4.8, "fairness");
  text(s, "False alarms by group", { x: x2 + 0.3, y: 2.05, w: colW - 0.6, h: 0.45, fontSize: 19, bold: true, color: C.text1 });
  const th = { bold: true, color: HEX.navy, fontSize: 13, fill: { color: HEX.paper } };
  const td = { color: HEX.ink, fontSize: 13 };
  s.addTable([
    [{ text: "Group", options: th }, { text: "Honest payments", options: th }],
    [{ text: "Fiat currencies", options: td }, { text: "1.6–2.3% each", options: td }],
    [{ text: "ACH format", options: td }, { text: "10.4%", options: td }],
    [{ text: "Smallest banks", options: td }, { text: "2.6%", options: td }],
    [{ text: "Biggest banks", options: td }, { text: "0.74% (3.4× less)", options: td }],
  ], { x: x2 + 0.3, y: 2.65, w: colW - 0.6, colW: [1.7, colW - 2.3], rowH: 0.36,
    border: { type: "solid", color: HEX.line, pt: 1 }, fontFace: "Calibri", objectName: "fairness table" });
  text(s, [{ text: "Measured fix: ", options: { bold: true, color: C.text1 } },
    { text: "an ACH-only warning cap brings ACH to 6.9% and small banks to 1.2%, for 82% → 79% of laundering flagged." }],
    { x: x2 + 0.3, y: 4.7, w: colW - 0.6, h: 1.4, fontSize: 14 });
  // column 3
  const x3 = M + 2 * (colW + 0.3);
  card(s, x3, 1.85, colW, 4.8, "limitations");
  text(s, "Known limitations", { x: x3 + 0.3, y: 2.05, w: colW - 0.6, h: 0.45, fontSize: 19, bold: true, color: C.text1 });
  [["Victim or honest?", " Here a payment to a mule looks like an honest big first payment, so both get the same check."],
   ["Short evaluation:", " 2 validation and 2 test days; the $ figure rests on ~10 transfers."],
   ["Synthetic data", " (IBM AMLSim) labels laundering, not scam victims. Granite takes 1–2 min, so customers get the instant template."]]
    .forEach(([b, t], i) => {
      const y = 2.7 + i * 1.25;
      s.addImage({ data: I.warn, x: x3 + 0.3, y: y + 0.03, w: 0.32, h: 0.32, objectName: `warn ${i + 1}` });
      text(s, [{ text: b, options: { bold: true, color: C.text1 } }, { text: t }],
        { x: x3 + 0.8, y, w: colW - 1.1, h: 1.15, fontSize: 13.5 });
    });
  s.addText("No demographics in the data, so fairness is audited by proxy · docs/MODEL_CARD.md, docs/evidence/claims",
    { placeholder: "source" });
  s.addNotes(notes("4:20–4:45", `
SAY:
"Warnings never block, and every hold reaches a human. Every decision carries its reasons.

We audited fairness by currency, format and bank size. Honest ACH transfers and small-bank customers get more false alarms, so we measured a fix: an ACH-specific threshold cuts them by about a third, for a small loss in coverage.

And one limit: in this data, a victim's payment looks like an honest large first payment, so both get the same protective check."
`));

  // ================================================================== 10 impact + team
  pres.addSection({ title: "Impact" });
  s = pres.addSlide({ masterName: "Dark", sectionTitle: "Impact" });
  text(s, "IMPACT · AI FOR GOOD", { x: M, y: 0.42, w: CW, h: 0.36, fontSize: 13, bold: true, color: C.accent5, charSpacing: 2, valign: "middle" });
  text(s, "Every payment checked before it leaves", { x: M, y: 0.78, w: CW, h: 0.8, fontSize: 36, bold: true, color: C.background1 });
  [["Customers", "A clear, specific warning before the money leaves. No scores, no accusations; they decide."],
   ["Scam analysts", "Only 0.074% of payments paused; 73% of them laundering, about 660× the base rate."],
   ["The community", "Goes after the mule chains behind the $2.03B Australians lost to scams in 2024: 147 of 174 broken on test days."]]
    .forEach(([h, b], i) => {
      const x = M + i * (colW + 0.3), y = 1.85;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: colW, h: 1.85, rectRadius: 0.12, fill: { color: C.accent6 },
        line: { color: C.accent6 }, objectName: `impact ${h}` });
      text(s, h, { x: x + 0.3, y: y + 0.25, w: colW - 0.6, h: 0.45, fontSize: 19, bold: true, color: C.background1 });
      text(s, b, { x: x + 0.3, y: y + 0.75, w: colW - 0.6, h: 1.0, fontSize: 14, color: HEX.soft });
    });
  text(s, [{ text: "Next: ", options: { bold: true, color: "F2A65A" } },
    { text: "add Confirmation of Payee mismatches, shared mule reports (e.g. AFCX) and account age; months of history, recalibration and drift monitoring." }],
    { x: M, y: 3.95, w: CW, h: 0.7, fontSize: 15, color: HEX.soft });
  text(s, "TEAM", { x: M, y: 4.85, w: CW, h: 0.35, fontSize: 13, bold: true, color: C.accent5, charSpacing: 2 });
  [["Soham Athawale", "Team lead · [what Soham built]"], ["Vaishnavi Joshi", "[what Vaishnavi built]"]].forEach(([n, r], i) => {
    const w = (CW - 0.3) / 2, x = M + i * (w + 0.3), y = 5.3;
    s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: C.accent5, width: 2 }, objectName: `team rule ${i + 1}` });
    text(s, n, { x, y: y + 0.15, w, h: 0.45, fontSize: 20, bold: true, color: C.background1 });
    text(s, r, { x, y: y + 0.65, w, h: 0.6, fontSize: 15, color: HEX.soft });
  });
  s.addText("Code written during the datathon (Oct 17–18) with help from Claude Code, an AI coding assistant · public data", { placeholder: "source" });
  s.addNotes(notes("4:45–5:00", `
SAY:
"ScamShield checks every payment before it leaves and breaks the mule chain early. Customers get a clear warning, and analysts spend their time where three in four cases are real.

I'm Soham Athawale, the team lead, and I [say what you built]. My teammate Vaishnavi Joshi [say what Vaishnavi built].

Thanks for watching."

BEFORE RECORDING: replace the two [ ] role lines on this slide and in this script.
`));

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
