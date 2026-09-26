#!/usr/bin/env node
/**
 * Browser smoke test for the Laya DSH RAG GUI.
 *
 * The Python suite covers the server side of the GUI: that the assets ship and that
 * mounting it at "/" does not shadow the API routes. It cannot cover what the page
 * actually does once a browser parses it, which is where the layout, the token flow
 * and the abstention rendering are decided. This drives a real headless Chrome over
 * the DevTools Protocol and asserts on the rendered DOM.
 *
 * Skips (exit 0, with a printed reason) when Node lacks a global WebSocket, when no
 * Chrome can be found, or when the engine is not answering — the same way the other
 * integration checks in this repository skip instead of failing on missing setup.
 *
 * Usage:
 *   node tools/gui-smoke.mjs
 *   LAYA_DSH_RAG_BASE_URL=http://127.0.0.1:8079 LAYA_DSH_RAG_API_KEY=secret \
 *     node tools/gui-smoke.mjs
 *   node tools/gui-smoke.mjs --screenshots /tmp/gui
 *
 * Environment:
 *   LAYA_DSH_RAG_BASE_URL  engine to test (default http://127.0.0.1:8077)
 *   LAYA_DSH_RAG_API_KEY   bearer token, when the engine requires one
 *   CHROME                 path to a Chrome/Chromium binary
 */
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = (process.env.LAYA_DSH_RAG_BASE_URL || "http://127.0.0.1:8077").replace(/\/+$/, "");
const TOKEN = process.env.LAYA_DSH_RAG_API_KEY || "";
const SHOTS = (() => {
  const at = process.argv.indexOf("--screenshots");
  return at === -1 ? null : (process.argv[at + 1] || "gui-screenshots");
})();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function skip(reason) {
  console.log(`SKIP: ${reason}`);
  process.exit(0);
}

/* ── prerequisites ───────────────────────────────────────────────────── */

if (typeof WebSocket === "undefined") {
  skip("this Node has no global WebSocket (needs Node 22+); cannot speak the DevTools Protocol");
}

function findChrome() {
  const candidates = [
    process.env.CHROME,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  return candidates.find((path) => existsSync(path)) || null;
}

const chromePath = findChrome();
if (!chromePath) skip("no Chrome/Chromium found; set CHROME to a browser binary");

async function engineReachable() {
  try {
    const headers = TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};
    const response = await fetch(`${BASE}/health`, { headers });
    return response.ok;
  } catch (_) {
    return false;
  }
}

if (!(await engineReachable())) {
  skip(`no engine answering at ${BASE}/health — start it with \`.venv/bin/laya-dsh-rag-serve\``);
}

/* ── chrome + CDP plumbing ───────────────────────────────────────────── */

const debugPort = 9500 + Math.floor(Math.random() * 400);
const profile = mkdtempSync(join(tmpdir(), "laya-gui-smoke-"));
const chrome = spawn(chromePath, [
  "--headless=new",
  // Chrome's own sandbox cannot initialise under a restricted file sandbox (and in
  // most CI containers), and the GUI needs no isolation from itself.
  "--no-sandbox",
  "--disable-gpu",
  "--disable-dev-shm-usage",
  "--disable-crash-reporter",
  "--no-first-run",
  "--no-default-browser-check",
  "--hide-scrollbars",
  `--remote-debugging-port=${debugPort}`,
  `--user-data-dir=${profile}`,
  "about:blank",
], { stdio: "ignore" });

let ws = null;
function cleanup() {
  try { ws?.close(); } catch (_) { /* ignore */ }
  try { chrome.kill("SIGKILL"); } catch (_) { /* ignore */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch (_) { /* ignore */ }
}
process.on("exit", cleanup);
process.on("SIGINT", () => { cleanup(); process.exit(130); });

async function browserTarget() {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
      const page = targets.find((t) => t.type === "page");
      if (page?.webSocketDebuggerUrl) return page;
    } catch (_) { /* not up yet */ }
    await sleep(250);
  }
  throw new Error("headless Chrome did not expose a page target");
}

const target = await browserTarget();
ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve, { once: true });
  ws.addEventListener("error", () => reject(new Error("could not attach to the browser")), { once: true });
});

let seq = 0;
const pending = new Map();
const consoleErrors = [];
ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
    return;
  }
  if (message.method === "Runtime.exceptionThrown") {
    consoleErrors.push(message.params?.exceptionDetails?.exception?.description || "uncaught exception");
  } else if (message.method === "Runtime.consoleAPICalled" && message.params?.type === "error") {
    consoleErrors.push((message.params.args || []).map((a) => a.value ?? a.description).join(" "));
  }
});

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
    setTimeout(() => {
      if (pending.has(id)) { pending.delete(id); reject(new Error(`timed out in ${method}`)); }
    }, 120000);
  });
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression, awaitPromise: true, returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error("page threw: " +
      (result.exceptionDetails.exception?.description || result.exceptionDetails.text));
  }
  return result.result.value;
}

async function waitFor(expression, label, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(`(() => { try { return !!(${expression}); } catch (e) { return false; } })()`)) return;
    await sleep(200);
  }
  throw new Error(`timed out waiting for ${label}`);
}

async function screenshot(name) {
  if (!SHOTS) return;
  mkdirSync(SHOTS, { recursive: true });
  const { data } = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(SHOTS, name), Buffer.from(data, "base64"));
}

/* ── the checks ──────────────────────────────────────────────────────── */

const results = [];
async function check(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`  ok   ${name}`);
  } catch (error) {
    results.push({ name, ok: false, error: error.message });
    console.log(`  FAIL ${name}\n         ${error.message}`);
  }
}

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", {
  width: 1560, height: 1000, deviceScaleFactor: 1, mobile: false,
});

console.log(`GUI smoke test against ${BASE}${TOKEN ? " (with a bearer token)" : ""}\n`);

await send("Page.navigate", { url: `${BASE}/` });
await waitFor(`document.readyState === 'complete'`, "the page to load");

/* Deterministic starting state. A composition preference or token remembered from a
   previous run would otherwise make these assertions depend on history. */
await evaluate(`(() => {
  try {
    localStorage.setItem('laya.compose', '0');
    ${TOKEN ? `localStorage.setItem('laya.token', ${JSON.stringify(TOKEN)});` : ""}
  } catch (e) {}
  return true;
})()`);
await send("Page.navigate", { url: `${BASE}/` });
await waitFor(`document.readyState === 'complete'`, "the page to reload with a known state");

/* A running engine that does not serve the GUI is almost always one started before
   the GUI existed: it answers /health but has no / route, so every later check would
   fail with a null dereference. Say so plainly instead. */
const servingGui = await evaluate(`!!document.getElementById('status-strip')`);
if (!servingGui) {
  const pageStatus = await evaluate(
    `(async () => (await fetch('/', { cache: 'no-store' })).status)()`);
  console.log(`FAIL: ${BASE}/ did not serve the GUI (HTTP ${pageStatus} for the page itself).`);
  console.log("      The engine is answering /health, so it is running — but a process started");
  console.log("      before the GUI shipped serves the JSON API only. Restart it and retry:");
  console.log("        .venv/bin/laya-dsh-rag-serve");
  cleanup();
  process.exit(1);
}

await check("the page loads and reports the engine as ready", async () => {
  await waitFor(`document.querySelectorAll('#status-strip .pill').length > 1`, "status pills");
  const pills = await evaluate(`
    Array.from(document.querySelectorAll('#status-strip .pill')).map(p => p.textContent.trim())`);
  assert(pills.some((p) => p.includes("ready")),
    `expected an "engine ready" pill, saw: ${pills.join(", ")}`);
});

const config = await evaluate(`(() => {
  const p = Array.from(document.querySelectorAll('#status-strip .pill'))
    .map(x => x.textContent.trim());
  return { pills: p };
})()`);
const chunkPill = config.pills.find((p) => p.startsWith("index"));
const indexed = chunkPill ? !chunkPill.includes("empty") : false;
console.log(`  ..   corpus: ${indexed ? chunkPill : "empty (query checks will be skipped)"}`);

await check("the query controls are seeded from the engine configuration", async () => {
  const values = await evaluate(`({
    topK: document.getElementById('ctl-topk').value,
    retrieveK: document.getElementById('ctl-retrievek').value,
    relevance: document.getElementById('ctl-relevance').value,
    labels: document.getElementById('controls-summary').textContent.trim(),
  })`);
  assert(Number(values.topK) >= 1 && Number(values.retrieveK) >= 1,
    `controls hold unusable values: ${JSON.stringify(values)}`);
  assert(values.labels === "engine defaults",
    `expected "engine defaults" before any change, got "${values.labels}"`);
});

await check("the layout keeps the footer out of the panes and scrolls internally", async () => {
  const layout = await evaluate(`(() => {
    const pane = document.querySelector('.pane-query');
    const footer = document.querySelector('.footer');
    return {
      overflow: document.documentElement.scrollHeight - window.innerHeight,
      footerTop: footer.getBoundingClientRect().top,
      paneBottom: pane.getBoundingClientRect().bottom,
    };
  })()`);
  assert(layout.overflow <= 2, `the page overflows the viewport by ${layout.overflow}px`);
  assert(layout.footerTop >= layout.paneBottom - 1, "the footer overlaps the query pane");
});

await check("the theme control cycles and persists", async () => {
  const before = await evaluate(`document.documentElement.dataset.theme`);
  await evaluate(`document.getElementById('theme-btn').click()`);
  const after = await evaluate(`document.documentElement.dataset.theme`);
  assert(before !== after, `theme did not change from "${before}"`);
});

/* Composition is the only feature whose output nobody has verified, so the switch
   and the banner must move together: on means the page says so, off means it does not. */
await check("the compose switch drives the footer warning in both directions", async () => {
  const read = () => evaluate(`(() => ({
    disabled: document.getElementById('compose-toggle').disabled,
    hint: document.getElementById('compose-hint').textContent.trim(),
    footer: document.getElementById('footer-note').textContent.trim(),
    warning: document.getElementById('footer').classList.contains('is-warning'),
    switchOn: document.getElementById('compose-switch').classList.contains('is-on'),
  }))()`);
  const toggle = (on) => evaluate(`(() => {
    const box = document.getElementById('compose-toggle');
    box.checked = ${on};
    box.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()`);

  const off = await read();
  assert(!off.warning, "the footer started as a warning while composition was off");
  // Match the warning phrase, not the word "hallucinate": the honest footer says
  // "nothing to hallucinate", so a loose match would flag the wrong state.
  assert(!/may contain hallucinations/i.test(off.footer),
    `the footer warned about hallucination while composition was off: ${off.footer}`);

  if (off.disabled) {
    // No model configured on this engine: the switch must be off and explain itself.
    assert(/COMPOSE_MODEL|composition model/i.test(off.hint),
      `a disabled switch has to say why: ${off.hint}`);
    console.log("  ..   no composition model configured; testing the disabled path only");
    return;
  }

  await toggle(true);
  await sleep(300);
  const on = await read();
  assert(on.warning, "the footer did not become a warning when composition was turned on");
  assert(/may contain hallucinations/i.test(on.footer),
    `the warning does not say the answer may contain hallucinations: ${on.footer}`);
  assert(on.switchOn, "the switch did not render as on");

  await toggle(false);
  await sleep(300);
  const back = await read();
  assert(!back.warning, "the footer stayed a warning after composition was turned off");
  assert(/nothing to hallucinate/i.test(back.footer),
    `the honest footer did not come back: ${back.footer}`);
});
/* Panes must actually receive their elevation, and the primary action must be
   legible: wrapping a box-shadow shorthand in light-dark() silently voids it, and a
   pale accent with white text fails WCAG AA. Check both themes. */
async function tokensForCurrentTheme() {
  return evaluate(`(() => {
    function luminance(colour) {
      const parts = (colour.match(/[\\d.]+/g) || []).slice(0, 3).map(Number).map((v) => {
        v /= 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2];
    }
    const pane = getComputedStyle(document.querySelector('.pane'));
    const button = getComputedStyle(document.querySelector('.btn-primary'));
    const a = luminance(button.color), b = luminance(button.backgroundColor);
    return {
      theme: document.documentElement.dataset.theme,
      shadow: pane.boxShadow,
      ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05),
      buttonText: button.color,
      buttonBackground: button.backgroundColor,
    };
  })()`);
}

for (const theme of ["dark", "light"]) {
  await evaluate(`document.documentElement.dataset.theme = ${JSON.stringify(theme)}`);
  await check(`the ${theme} theme elevates its panes and keeps the primary action legible`, async () => {
    const tokens = await tokensForCurrentTheme();
    assert(tokens.shadow && tokens.shadow !== "none",
      "the pane box-shadow resolved to none (is --shadow still wrapping a whole shorthand in light-dark()?)");
    assert(tokens.ratio >= 4.5,
      `primary button contrast is ${tokens.ratio.toFixed(2)}:1 ` +
      `(${tokens.buttonText} on ${tokens.buttonBackground})`);
  });
}

if (indexed) {
  await check("the corpus browser lists indexed files with chunk counts", async () => {
    await waitFor(`document.querySelectorAll('#corpus-list .doc-row').length > 0`, "corpus rows");
    const rows = await evaluate(`Array.from(document.querySelectorAll('#corpus-list .doc-row'))
      .slice(0, 3).map(r => r.textContent.trim())`);
    assert(rows.length > 0, "no corpus rows rendered");
    assert(rows.every((r) => /chunk/.test(r)), `a row omitted its chunk count: ${rows[0]}`);
    assert(rows.every((r) => !/\b1 chunks\b/.test(r)), `bad pluralisation: ${rows.join(" | ")}`);
  });

  await check("opening a document renders its chunks", async () => {
    await evaluate(`document.querySelectorAll('#corpus-list .doc-row')[0].click()`);
    await waitFor(`document.querySelectorAll('#chunk-viewer .chunk').length > 0`, "chunk viewer");
    const chunks = await evaluate(`document.querySelectorAll('#chunk-viewer .chunk').length`);
    assert(chunks > 0, "the chunk viewer stayed empty");
  });

  await check("an answerable query renders a measured verdict and its evidence", async () => {
    await evaluate(`(() => {
      document.getElementById('query-input').value = 'How do I run Laya in Docker?';
      document.getElementById('query-form').requestSubmit();
      return true;
    })()`);
    await waitFor(`document.querySelector('#query-output .running')`, "the running indicator", 15000);
    await waitFor(
      `document.querySelector('#query-output .verdict') && !document.querySelector('#query-output .running')`,
      "the verdict", 120000);

    const verdict = await evaluate(`(() => {
      const card = document.querySelector('#query-output .verdict');
      return {
        classes: card.className,
        label: card.querySelector('.verdict-label').textContent.trim(),
        hasMeter: !!card.querySelector('.meter'),
        meterWidth: card.querySelector('.meter-fill')?.style.width || null,
        tick: !!card.querySelector('.meter-tick'),
        passages: document.querySelectorAll('#query-output .passage').length,
        highlights: document.querySelectorAll('#query-output .passage-text mark').length,
      };
    })()`);

    if (verdict.classes.includes("is-ok")) {
      assert(verdict.hasMeter, "an answerable verdict drew no probability meter");
      assert(verdict.tick, "an answerable verdict drew no threshold marker");
      assert(verdict.passages > 0, "an answerable verdict returned no passages");
      assert(verdict.highlights > 0, "query terms were not highlighted in the evidence");
    } else {
      // The corpus may legitimately not cover the probe; the honest path still has to
      // render, and must not dress a default 0.0 up as a measurement.
      assert(!verdict.hasMeter, "a gated-out verdict drew an unmeasured probability meter");
    }
  });

  await check("a question the corpus cannot answer abstains without faking a probability", async () => {
    await evaluate(`(() => {
      document.getElementById('query-input').value = 'What is the capital of Peru?';
      document.getElementById('query-form').requestSubmit();
      return true;
    })()`);
    await waitFor(
      `document.querySelector('#query-output .verdict') && !document.querySelector('#query-output .running') &&
       document.querySelector('#query-output .passage') === null`,
      "the abstention", 120000);

    const abstain = await evaluate(`(() => {
      const card = document.querySelector('#query-output .verdict');
      return {
        label: card.querySelector('.verdict-label').textContent.trim(),
        score: card.querySelector('.verdict-score').textContent.trim(),
        hasMeter: !!card.querySelector('.meter'),
        passages: document.querySelectorAll('#query-output .passage').length,
      };
    })()`);
    assert(/Not answerable/.test(abstain.label), `expected an abstention, got "${abstain.label}"`);
    assert(abstain.passages === 0, "an abstention rendered passages");
    assert(!abstain.hasMeter,
      "the engine's short-circuited answerability default was drawn as a measured probability");
  });

  /* Opt-in, because it calls a real model and takes tens of seconds:
       GUI_SMOKE_COMPOSE=1 node tools/gui-smoke.mjs
     Everything else in this file is deterministic and fast. */
  if (process.env.GUI_SMOKE_COMPOSE === "1") {
    await check("a composed answer streams in, badged as unverified, with evidence intact", async () => {
      const available = await evaluate(`!document.getElementById('compose-toggle').disabled`);
      if (!available) {
        console.log("       (engine has no composition model configured; nothing to verify)");
        return;
      }
      await evaluate(`(() => {
        const box = document.getElementById('compose-toggle');
        box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true }));
        document.getElementById('query-input').value = 'How do I run Laya in Docker?';
        document.getElementById('query-form').requestSubmit();
        return true;
      })()`);

      await waitFor(`document.querySelector('.answer-body')`, "the answer panel", 30000);
      // Wait for a terminal state, not a length threshold: the panel is created
      // before the first token arrives, so anything weaker reads a partial answer
      // (or an empty one) and quietly under-tests the feature.
      await waitFor(
        `document.querySelector('.answer[data-state="done"], .answer[data-state="empty"], .answer[data-state="error"], .answer[data-state="truncated"]')`,
        "the composed answer to finish streaming", 240000);

      const answer = await evaluate(`(() => {
        const body = document.querySelector('.answer-body');
        return {
          text: body.dataset.text || "",
          badge: document.querySelector('.answer-badge')?.textContent.trim() || "",
          model: document.querySelector('.answer-model')?.textContent.trim() || "",
          caveat: document.querySelector('.answer-caveat')?.textContent.trim() || "",
          footerWarning: document.getElementById('footer').classList.contains('is-warning'),
          footerText: document.getElementById('footer-note').textContent.trim(),
          passages: document.querySelectorAll('#query-output .passage').length,
          state: document.querySelector('.answer')?.dataset.state || "",
          errors: Array.from(document.querySelectorAll('.answer-error')).map(e => e.textContent.trim()),
          warnings: Array.from(document.querySelectorAll('.answer-warn')).map(e => e.textContent.trim()),
        };
      })()`);

      assert(answer.text.length > 150,
        `a finished answer should be a paragraph, got ${answer.text.length} chars ` +
        `(state=${answer.state}, errors=${JSON.stringify(answer.errors)})`);
      assert(/Composed/i.test(answer.badge), `the answer is not badged as composed: ${answer.badge}`);
      assert(answer.model.length > 0, "the answer does not name the model that wrote it");
      assert(/not been verified|not a Laya output/i.test(answer.caveat),
        `the answer lacks an unverified caveat: ${answer.caveat}`);
      assert(answer.footerWarning && /may contain hallucinations/i.test(answer.footerText),
        "the footer was not warning about hallucination while composing");
      assert(answer.passages > 0, "evidence was dropped when composition was enabled");
      // A truncation is a legitimate outcome reported in its own right; a real
      // failure is not.
      assert(answer.errors.length === 0,
        `composition failed: ${JSON.stringify(answer.errors)}`);
      if (answer.warnings.length) {
        console.log(`       (the model truncated its answer: ${answer.warnings[0].slice(0, 60)}…)`);
      }
      console.log(`       composed ${answer.text.length} chars via ${answer.model}`);

      // Layer 1 runs after the answer lands, so wait for it to settle before reading.
      await waitFor(
        `document.querySelector('.cite-check[data-state="clean"], .cite-check[data-state="flagged"], .cite-check[data-state="error"]')`,
        "the citation check", 60000);
      const cite = await evaluate(`(() => {
        const box = document.querySelector('.cite-check');
        return {
          state: box.dataset.state,
          summary: box.querySelector('.cite-summary')?.textContent.trim() || "",
          counts: box.querySelector('.cite-counts')?.textContent.trim() || "",
          caveat: box.querySelector('.cite-caveat')?.textContent.trim() || "",
          flagged: box.querySelectorAll('.cite-details li').length,
        };
      })()`);
      assert(cite.state !== "error", `the citation check failed to run: ${cite.summary}`);
      assert(/^Citation check:/i.test(cite.summary),
        `the citation check produced no summary: ${JSON.stringify(cite)}`);
      assert(/\d+ claims/.test(cite.counts), `no claim counts rendered: ${cite.counts}`);
      // The wording has to bound the claim: this is about citations, not truth.
      assert(/does not check whether the claims are true/i.test(cite.caveat),
        `the check's caveat is missing or overclaims: ${cite.caveat}`);
      console.log(`       citation check: ${cite.state} (${cite.flagged} flagged) — ${cite.summary}`);
    });
  }
}

await check("the tabs and passage disclosures are keyboard-operable and associated", async () => {
  const a11y = await evaluate(`(() => {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const heads = Array.from(document.querySelectorAll('.passage-head[role="button"]'));
    return {
      tabCount: tabs.length,
      tabStops: tabs.filter(t => t.tabIndex === 0).length,
      headCount: heads.length,
      unassociated: heads.filter(h => {
        const id = h.getAttribute('aria-controls');
        return !id || !document.getElementById(id);
      }).length,
    };
  })()`);
  assert(a11y.tabCount === 3, `expected 3 tabs, got ${a11y.tabCount}`);
  assert(a11y.tabStops === 1,
    `the ARIA tabs pattern wants exactly one tab stop, found ${a11y.tabStops}`);
  assert(a11y.unassociated === 0,
    `${a11y.unassociated} of ${a11y.headCount} passage headers lack a resolvable aria-controls`);
});

await check("the page raised no console errors or uncaught exceptions", () => {
  assert(consoleErrors.length === 0, consoleErrors.join(" | "));
});

await screenshot("gui-smoke.png");

/* ── report ──────────────────────────────────────────────────────────── */

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  console.log("failed:\n" + failed.map((f) => `  - ${f.name}: ${f.error}`).join("\n"));
  process.exit(1);
}
console.log("GUI OK");
process.exit(0);
