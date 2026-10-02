/* ============================================================================
   Laya RAG — evidence console
   Plain ES2020, no dependencies, no build step. Every corpus-derived string is
   inserted with textContent or createTextNode; nothing from the engine or the
   filesystem is ever parsed as HTML.
   ========================================================================== */
"use strict";

(function () {
  /* ── tiny helpers ───────────────────────────────────────────────────── */

  const $ = (id) => document.getElementById(id);

  /** Create an element. `props.text` is always set as a text node. */
  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === "text") node.textContent = String(value);
        else if (key === "class") node.className = value;
        else if (key === "html") throw new Error("html is not permitted; use text");
        else if (key === "dataset") Object.assign(node.dataset, value);
        else if (key.startsWith("on") && typeof value === "function") {
          node.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value === true) node.setAttribute(key, "");
        else node.setAttribute(key, String(value));
      }
    }
    for (const child of children || []) {
      if (child === null || child === undefined || child === false) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  function fmtSeconds(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    return value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(2)} s`;
  }

  const fmtInt = (value) =>
    typeof value === "number" ? value.toLocaleString() : "—";

  function fmtProb(value) {
    return typeof value === "number" && isFinite(value) ? value.toFixed(3) : "—";
  }

  function basename(path) {
    const parts = String(path).split("/");
    return parts[parts.length - 1] || String(path);
  }

  function dirname(path) {
    const parts = String(path).split("/");
    parts.pop();
    return parts.join("/") || "./";
  }

  /* localStorage is unavailable in some privacy modes; never let it break the app. */
  const store = {
    get(key, fallback) {
      try {
        const raw = window.localStorage.getItem(`laya.${key}`);
        return raw === null ? fallback : raw;
      } catch (_) { return fallback; }
    },
    set(key, value) {
      try { window.localStorage.setItem(`laya.${key}`, value); } catch (_) { /* ignore */ }
    },
    del(key) {
      try { window.localStorage.removeItem(`laya.${key}`); } catch (_) { /* ignore */ }
    },
  };

  /* ── API client ─────────────────────────────────────────────────────── */

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  function detailOf(data, status) {
    const detail = data && data.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          const where = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
          return where ? `${where}: ${item.msg}` : item.msg;
        })
        .join("; ");
    }
    if (status === 0) return "could not reach the engine";
    return `request failed with HTTP ${status}`;
  }

  async function api(path, options) {
    const opts = options || {};
    const headers = {};
    if (opts.body !== undefined) headers["Content-Type"] = "application/json";
    if (state.token) headers["Authorization"] = `Bearer ${state.token}`;

    let response;
    try {
      response = await fetch(path, {
        method: opts.method || "GET",
        headers,
        body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
        cache: "no-store",
        signal: opts.signal,
      });
    } catch (error) {
      if (error && error.name === "AbortError") throw error;
      throw new ApiError("could not reach the engine — is `laya-rag-serve` running?", 0);
    }

    let data = null;
    const type = response.headers.get("content-type") || "";
    if (type.includes("application/json")) {
      try { data = await response.json(); } catch (_) { data = null; }
    }
    if (!response.ok) throw new ApiError(detailOf(data, response.status), response.status);
    if (data === null) {
      // A 2xx that is not decodable JSON would otherwise yield null and fail later
      // with a confusing TypeError somewhere far from the cause.
      throw new ApiError("the engine returned a response this page could not read", response.status);
    }
    return data;
  }

  /* ── state ──────────────────────────────────────────────────────────── */

  const state = {
    token: store.get("token", ""),
    theme: store.get("theme", "auto"),
    health: null,
    stats: null,
    documents: [],
    activeDoc: null,
    activeChunks: null,
    chunkError: null,
    lastResult: null,
    lastRequest: null,
    defaults: null,
    history: [],
    filter: "",
    busy: false,
    authBlocked: false,
    engineError: null,
    // Composition is remembered across reloads, but the footer banner always reflects
    // the current state, so a remembered "on" is never quiet about itself.
    compose: store.get("compose", "0") === "1",
    composeMeta: null,
    composeAbort: null,
    inFlight: null,
  };

  try {
    const parsed = JSON.parse(store.get("history", "[]"));
    // Anything could be in local storage, so validate rather than trusting it: a
    // non-array here would throw inside renderHistory and leave the UI unwired.
    state.history = Array.isArray(parsed)
      ? parsed.filter((item) => typeof item === "string").slice(0, 6)
      : [];
  } catch (_) {
    state.history = [];
  }

  const ui = {};

  /* ── theme ──────────────────────────────────────────────────────────── */

  const THEMES = ["auto", "light", "dark"];

  function applyTheme() {
    document.documentElement.dataset.theme = state.theme;
    ui.themeBtn.textContent = state.theme[0].toUpperCase() + state.theme.slice(1);
  }

  /* ── status strip ───────────────────────────────────────────────────── */

  function pill(label, value, tone, title) {
    const classes = ["pill"];
    if (tone) classes.push(`is-${tone}`);
    const node = el("span", { class: classes.join(" "), title: title || "" }, [
      el("span", { class: "dot" }),
    ]);
    node.appendChild(document.createTextNode(label));
    if (value !== undefined && value !== null) node.appendChild(el("b", { text: String(value) }));
    return node;
  }

  function renderStatus() {
    const strip = ui.status;
    clear(strip);

    if (state.authBlocked) {
      strip.appendChild(pill("engine", "token required", "warn",
        "This engine has LAYA_RAG_API_KEY set; supply the bearer token under Connection"));
      return;
    }
    if (!state.health && !state.stats) {
      // "unreachable" and "the engine answered with an error" are different problems
      // and need different fixes, so do not collapse them into one pill.
      const unreachable = !state.engineError || state.engineError.status === 0;
      strip.appendChild(pill("engine", unreachable ? "unreachable" : "error", "bad",
        state.engineError ? state.engineError.message : ""));
      return;
    }

    const health = state.health || {};
    const stats = state.stats || {};
    const config = stats.config || {};
    const index = stats.index || {};
    const model = stats.model || {};

    const degraded = health.status && health.status !== "ok";
    strip.appendChild(
      pill("engine", degraded ? "degraded" : "ready", degraded ? "warn" : "ok",
        health.error || "GET /health")
    );

    strip.appendChild(pill("checkpoint", health.checkpoint || config.checkpoint || "—", null,
      "Laya checkpoint in use"));
    strip.appendChild(pill("device", config.device || "—", null, "auto resolves to CUDA → MPS → CPU"));
    const embedDim = model.embedding_dim || index.dim;
    strip.appendChild(
      pill("model", health.model_loaded ? "loaded" : "not loaded",
        health.model_loaded ? "ok" : "warn",
        embedDim ? `embedding dim ${embedDim}` : (model.models_dir || "GET /health"))
    );

    const chunks = index.chunks || 0;
    strip.appendChild(
      pill("index",
        chunks ? `${fmtInt(chunks)} chunks` : "empty",
        chunks ? "ok" : "warn",
        stats.index_dir || "")
    );
    if (chunks) {
      strip.appendChild(pill("docs", fmtInt(index.documents || 0)));
      strip.appendChild(pill("files", fmtInt(index.paths || 0)));
      strip.appendChild(pill("dim", fmtInt(index.dim || 0)));
    }
    strip.appendChild(
      pill("lexical", stats.lexical_present ? "on" : "off",
        stats.lexical_present ? "ok" : "warn",
        stats.lexical_present
          ? "BM25 channel fused with dense recall by reciprocal rank fusion"
          : "missing — recall is dense-only and will miss exact-term matches; re-index to rebuild")
    );
  }

  /* ── global alert ───────────────────────────────────────────────────── */

  function showAlert(kind, title, body) {
    const box = ui.alert;
    clear(box);
    box.className = `alert${kind ? ` is-${kind}` : ""}`;
    box.hidden = false;
    box.appendChild(el("div", {}, [
      el("strong", { text: title }),
      body ? el("span", { text: body }) : null,
    ]));
    box.appendChild(el("button", {
      class: "btn btn-ghost btn-sm",
      type: "button",
      text: "Dismiss",
      onclick: hideAlert,
    }));
  }

  function hideAlert() { ui.alert.hidden = true; }

  /* ── connection panel ───────────────────────────────────────────────── */

  function renderConnection() {
    const stats = state.stats || {};
    const config = stats.config || {};
    // /v1/stats is itself authenticated, so its config is unavailable precisely when
    // we most need to know a token is required — fall back to the 401 we just saw.
    const needed = config.api_key === "<redacted>" || state.authBlocked;
    const tokenState = state.token
      ? (state.authBlocked ? "configured, but rejected" : "configured")
      : needed ? "required, missing" : "not required";

    clear(ui.connMeta);
    const rows = [
      ["endpoint", window.location.origin],
      ["token", tokenState],
    ];
    if (stats.index_dir) rows.push(["index dir", stats.index_dir]);
    for (const [key, value] of rows) {
      ui.connMeta.appendChild(el("dt", { text: key }));
      ui.connMeta.appendChild(el("dd", { text: value }));
    }
  }

  function openConnection(message) {
    ui.connPanel.hidden = false;
    ui.connBtn.setAttribute("aria-expanded", "true");
    if (message) showAlert("error", "Engine refused the request", message);
  }

  function closeConnection() {
    ui.connPanel.hidden = true;
    ui.connBtn.setAttribute("aria-expanded", "false");
  }

  function handleAuthError(error) {
    if (error && error.status === 401) {
      state.authBlocked = true;
      openConnection(
        state.token
          ? "The bearer token was rejected. Check it against LAYA_RAG_API_KEY."
          : "This engine has LAYA_RAG_API_KEY set. Enter the token to continue."
      );
      renderConnection();
      return true;
    }
    return false;
  }

  /* ── query: rendering ───────────────────────────────────────────────── */

  function queryTerms(query) {
    const words = String(query).toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || [];
    const seen = new Set();
    const out = [];
    for (const word of words) {
      if (seen.has(word)) continue;
      seen.add(word);
      out.push(word);
      if (out.length >= 16) break;
    }
    return out.sort((a, b) => b.length - a.length);
  }

  const escapeRe = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  /** Build a fragment with query terms wrapped in <mark>, safely. */
  function highlighted(text, terms) {
    const fragment = document.createDocumentFragment();
    if (!terms.length) {
      fragment.appendChild(document.createTextNode(text));
      return fragment;
    }
    const pattern = new RegExp(`(${terms.map(escapeRe).join("|")})`, "gi");
    let cursor = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match[0].length === 0) { pattern.lastIndex += 1; continue; }
      if (match.index > cursor) {
        fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      }
      fragment.appendChild(el("mark", { text: match[0] }));
      cursor = match.index + match[0].length;
    }
    if (cursor < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(cursor)));
    }
    return fragment;
  }

  function meter(fillRatio, tickRatio) {
    const track = el("div", { class: "meter" }, [
      el("i", { class: "meter-fill", style: `width:${Math.max(0, Math.min(1, fillRatio)) * 100}%` }),
    ]);
    if (typeof tickRatio === "number") {
      track.appendChild(el("span", {
        class: "meter-tick",
        style: `left:calc(${Math.max(0, Math.min(1, tickRatio)) * 100}% - 1px)`,
        title: `decision threshold ${fmtProb(tickRatio)}`,
      }));
    }
    return track;
  }

  /**
   * Did Laya actually render an answerability decision?
   *
   * When no passage clears the relevance bar, LayaReranker.rerank short-circuits and
   * reports `answerability = 0.0` *without* asking Laya the answerability question.
   * That 0.0 is a plumbing default, not a calibrated measurement, so no rendering
   * path — card, meter or copied markdown — may present it as one.
   */
  function answerabilityIsMeasured(result) {
    if (typeof result.answerability !== "number") return false;
    if (result.answerable === true) return true;
    return (result.kept || 0) > 0;
  }

  /** True when nothing was retrieved at all: an empty index, or no candidates. */
  function nothingRetrieved(result) {
    return (result.kept || 0) === 0 && (result.considered || 0) === 0;
  }

  function verdictCard(result, request) {
    const answerable = result.answerable === true;
    const probability = typeof result.answerability === "number" ? result.answerability : null;
    const threshold = request.answerability_threshold;
    const relevanceThreshold = request.relevance_threshold;
    const kept = result.kept || 0;

    const measured = answerabilityIsMeasured(result);
    const gatedOut = !answerable && kept === 0 && !nothingRetrieved(result);
    const empty = !answerable && nothingRetrieved(result);

    const card = el("div", { class: `verdict ${answerable ? "is-ok" : "is-no"}` });

    const score = measured
      ? el("span", { class: "verdict-score" }, [
          el("span", { class: "num", text: fmtProb(probability) }),
          el("span", { class: "cap", text: "Laya P(answerable)" }),
        ])
      : el("span", { class: "verdict-score" }, [
          el("span", { class: "not-measured", text: "gate not reached" }),
          el("span", { class: "cap", text: "answerability" }),
        ]);

    card.appendChild(el("div", { class: "verdict-top" }, [
      el("span", { class: "verdict-label" }, [
        el("span", { class: "verdict-icon", text: answerable ? "✓" : "⌀", "aria-hidden": "true" }),
        document.createTextNode(answerable ? "Answerable" : "Not answerable"),
      ]),
      score,
    ]));

    const relevanceText = typeof relevanceThreshold === "number"
      ? ` (relevance threshold ${fmtProb(relevanceThreshold)})` : "";
    const why = answerable
      ? "Laya's calibrated answerability decision cleared the threshold for this evidence set."
      : empty
        ? "Nothing was retrieved, so Laya had no candidates to judge. The index is empty or no passage matched the recall stage."
        : gatedOut
          ? `No passage cleared the relevance bar${relevanceText}, so Laya never reached its ` +
            "answerability gate: it had nothing it was willing to call evidence."
          : "Laya judged the retrieved passages insufficient to answer the question. The corpus does not cover it — do not answer from memory.";
    card.appendChild(el("p", { class: "verdict-why", text: why }));

    if (measured) {
      card.appendChild(meter(probability, threshold));
      card.appendChild(el("div", { class: "meter-legend" }, [
        el("span", { text: "0.0" }),
        el("span", { text: typeof threshold === "number" ? `threshold ${fmtProb(threshold)}` : "" }),
        el("span", { text: "1.0" }),
      ]));
    }

    const timings = result.timings || {};
    card.appendChild(el("div", { class: "verdict-stats" }, [
      el("span", {}, [document.createTextNode("considered "), el("b", { text: fmtInt(result.considered) })]),
      el("span", {}, [document.createTextNode("kept "), el("b", { text: fmtInt(kept) })]),
      el("span", {}, [document.createTextNode("below relevance "), el("b", { text: fmtInt(result.dropped_below_threshold) })]),
      el("span", {}, [document.createTextNode("retrieve "), el("b", { text: fmtSeconds(timings.retrieval_seconds) })]),
      el("span", {}, [document.createTextNode("rerank "), el("b", { text: fmtSeconds(timings.rerank_seconds) })]),
      el("span", {}, [document.createTextNode("total "), el("b", { text: fmtSeconds(timings.total_seconds) })]),
      result.checkpoint ? el("span", {}, [document.createTextNode("checkpoint "), el("b", { text: result.checkpoint })]) : null,
    ]));

    if (result.note) {
      card.appendChild(el("p", { class: "note" }, [
        el("span", { text: "⚠", "aria-hidden": "true" }),
        el("span", { text: result.note }),
      ]));
    }
    return card;
  }

  /* ── evidence grouping ──────────────────────────────────────────────── */

  /* Retrieval works in chunks, so one file can arrive as several passages. Every
     consumer of that evidence — the composer's numbered context, the numbered cards
     here, and the citation check — has to agree on what "[n]" means, or a citation
     stops pointing at what it names. So the evidence is grouped once, here: one entry
     per document, at the position of its first passage, with its chunks joined in rank
     order and the chunker's overlap trimmed. Passages with no path stay separate,
     since no document is known to contain them.

     The engine applies the same rule in `laya_rag.evidence`, so a caller that speaks
     to the API directly gets the same grouping. This copy exists so the page displays
     and sends exactly the grouped evidence rather than a second opinion about it. */
  const MIN_OVERLAP_CHARS = 40;

  function joinWithoutOverlap(earlier, later) {
    const limit = Math.min(earlier.length, later.length);
    for (let size = limit; size >= MIN_OVERLAP_CHARS; size -= 1) {
      if (earlier.endsWith(later.slice(0, size))) return earlier + later.slice(size);
    }
    return `${earlier}\n\n${later}`;
  }

  function mergeInto(group, passage) {
    group.text = joinWithoutOverlap(group.text, passage.text || "");
    group.chunks += 1;
    if (typeof passage.relevance === "number") {
      group.relevance = typeof group.relevance === "number"
        ? Math.max(group.relevance, passage.relevance) : passage.relevance;
    }
    if (typeof passage.recall_score === "number") {
      group.recall_score = typeof group.recall_score === "number"
        ? Math.max(group.recall_score, passage.recall_score) : passage.recall_score;
    }
    if (typeof passage.start_char === "number") {
      group.start_char = typeof group.start_char === "number"
        ? Math.min(group.start_char, passage.start_char) : passage.start_char;
    }
    if (typeof passage.end_char === "number") {
      group.end_char = typeof group.end_char === "number"
        ? Math.max(group.end_char, passage.end_char) : passage.end_char;
    }
    const channels = Array.isArray(group.channels) ? group.channels.slice() : [];
    (Array.isArray(passage.channels) ? passage.channels : []).forEach((name) => {
      if (!channels.includes(name)) channels.push(name);
    });
    group.channels = channels;
  }

  function groupPassages(passages) {
    const order = [];
    const groups = new Map();
    let pathless = 0;
    (passages || []).forEach((passage) => {
      const path = (passage.path || "").trim();
      if (!path) {
        pathless += 1;
        const key = `\u0000pathless\u0000${pathless}`;
        groups.set(key, { ...passage, path: "", text: passage.text || "", chunks: 1 });
        order.push(key);
        return;
      }
      const existing = groups.get(path);
      if (existing) {
        mergeInto(existing, passage);
      } else {
        groups.set(path, { ...passage, text: passage.text || "", chunks: 1 });
        order.push(path);
      }
    });
    return order.map((key) => groups.get(key));
  }

  /* Grouped once per result and kept off the result object, so "Copy JSON" stays
     exactly the response the engine sent rather than a second, grouped payload. */
  const groupedEvidence = new WeakMap();

  function evidenceFor(result) {
    if (!groupedEvidence.has(result)) {
      groupedEvidence.set(result, groupPassages(result.passages || []));
    }
    return groupedEvidence.get(result);
  }

  function passageCard(passage, rank, terms, lowConfidence) {
    const relevance = typeof passage.relevance === "number" ? passage.relevance : 0;
    const channels = Array.isArray(passage.channels) ? passage.channels : [];
    const card = el("article", { class: `passage${lowConfidence ? " is-low" : ""}` });
    const bodyId = `passage-body-${rank}`;

    const header = el("header", {
      class: "passage-head",
      role: "button",
      tabindex: "0",
      "aria-expanded": "false",
      "aria-controls": bodyId,
      title: "Expand or collapse this passage",
    }, [
      el("span", { class: "passage-rank", text: String(rank) }),
      el("span", { class: "passage-path", text: passage.path, title: passage.path }),
      el("span", { class: "passage-relevance" }, [
        el("span", { class: "relevance-meter" }, [
          el("i", { style: `width:${Math.max(0, Math.min(1, relevance)) * 100}%` }),
        ]),
        el("span", { class: "val", text: fmtProb(relevance) }),
      ]),
      el("span", { class: "channels" }, channels.length
        ? channels.map((name) => el("span", {
            class: `chan chan-${name}`,
            text: name,
            title: name === "dense"
              ? "found by dense cosine recall over Laya embeddings"
              : "found by BM25 lexical recall",
          }))
        : [el("span", { class: "chan", text: "unknown" })]),
    ]);

    const text = el("div", { class: "passage-text" });
    text.appendChild(highlighted(passage.text || "", terms));

    const body = el("div", { class: "passage-body", id: bodyId }, [
      text,
      el("div", { class: "passage-foot" }, [
        el("span", { text: `recall ${typeof passage.recall_score === "number" ? passage.recall_score.toFixed(4) : "—"}`,
          title: "fused recall score (reciprocal rank fusion) — a ranking key, not a probability" }),
        el("span", { text: `chars ${fmtInt(passage.start_char)}–${fmtInt(passage.end_char)}` }),
        passage.chunks > 1
          ? el("span", {
              text: `${passage.chunks} chunks merged`,
              title: "Several retrieved chunks of this document form one evidence entry, so this number, "
                + "the composer's context and the citation check all agree",
            })
          : null,
        el("button", {
          class: "btn btn-ghost btn-sm copy",
          type: "button",
          text: "Copy",
          title: "Copy this passage",
          onclick: (event) => {
            event.stopPropagation();
            copyText(passage.text || "", event.currentTarget);
          },
        }),
      ]),
    ]);

    const toggle = () => {
      const open = card.classList.toggle("is-open");
      header.setAttribute("aria-expanded", open ? "true" : "false");
    };
    header.addEventListener("click", toggle);
    header.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggle(); }
    });

    card.appendChild(header);
    card.appendChild(body);
    return card;
  }

  function evidenceMarkdown(result) {
    const lines = [`Laya RAG evidence for: ${result.query}`];
    // Same gate as the verdict card: this text is what a downstream reader (often an
    // LLM) treats as the measurement, so a short-circuited 0.0 must never appear here.
    const measured = answerabilityIsMeasured(result);
    const suffix = measured
      ? ` (Laya answerability ${result.answerability.toFixed(3)})`
      : (result.kept || 0) === 0 ? " (answerability gate not reached)" : "";
    lines.push(`Verdict: ${result.answerable ? "answerable" : "NOT answerable"}${suffix}`);
    if (result.note) lines.push(`Note: ${result.note}`);
    lines.push("");
    evidenceFor(result).forEach((passage, index) => {
      const channels = (passage.channels || []).join("+") || "unknown";
      const chunks = passage.chunks > 1 ? `, ${passage.chunks} chunks` : "";
      lines.push(
        `[${index + 1}] ${passage.path} (relevance ${fmtProb(passage.relevance)}, ` +
        `recall ${typeof passage.recall_score === "number" ? passage.recall_score.toFixed(4) : "—"}, ${channels}${chunks})`
      );
      lines.push(passage.text || "");
      lines.push("");
    });
    if (!evidenceFor(result).length) {
      lines.push(nothingRetrieved(result)
        ? "(no candidates were retrieved: the index is empty or nothing matched at recall)"
        : "(no passage cleared the relevance threshold)");
    }
    return lines.join("\n");
  }

  function copyText(text, button) {
    const announce = (label) => {
      if (!button) return;
      if (!button.dataset.label) button.dataset.label = button.textContent;
      button.textContent = label;
      setTimeout(() => { button.textContent = button.dataset.label; }, 1200);
    };
    const succeed = () => announce("Copied");
    const fail = () => announce("Copy failed");
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(succeed, () => fallbackCopy(text, succeed, fail));
    } else {
      fallbackCopy(text, succeed, fail);
    }
  }

  function fallbackCopy(text, succeed, fail) {
    const area = el("textarea", { style: "position:fixed;top:-1000px;opacity:0", readonly: true });
    area.value = text;
    document.body.appendChild(area);
    area.select();
    let copied = false;
    // execCommand reports failure by returning false, not by throwing, so the return
    // value is the only way to know whether the clipboard actually changed.
    try { copied = document.execCommand("copy"); } catch (_) { copied = false; }
    document.body.removeChild(area);
    if (copied) succeed(); else fail();
  }

  function renderResult() {
    // Results are rebuilt inside #results; the compose control and its output sit after
    // it (see index.html), so a re-render never disturbs them and an unverified answer
    // always follows the evidence it was written from.
    const output = ui.results;
    clear(output);
    clear(ui.answerSlot);
    const result = state.lastResult;
    if (!result) return;

    output.appendChild(verdictCard(result, state.lastRequest || {}));

    const passages = evidenceFor(result);
    if (!passages.length) {
      // The verdict card above already rendered result.note, so printing it again
      // here would say the same thing twice; only add a remedy it does not contain.
      if (!result.note) {
        output.appendChild(el("p", {
          class: "hint empty-hint",
          text: nothingRetrieved(result)
            ? "Index some documents on the Index tab, then ask again."
            : "Nothing scored above the relevance threshold. Lower it, widen the candidate pool, or index more of the corpus.",
        }));
      }
      return;
    }

    if (state.compose && composeIsAvailable()) {
      composeAnswer(result, ui.answerSlot);
    } else if (state.compose) {
      ui.answerSlot.appendChild(el("p", {
        class: "hint empty-hint",
        text: composeUnavailableReason(),
      }));
    }

    const terms = queryTerms(result.query);
    const totalChunks = (result.passages || []).length;
    const docLabel = `${passages.length} document${passages.length === 1 ? "" : "s"}`;
    const head = el("div", { class: "evidence-head" }, [
      el("h3", {
        text: totalChunks > passages.length
          ? `${docLabel} from ${totalChunks} passages`
          : docLabel,
      }),
      el("span", { class: "rule" }),
      el("span", { class: "actions" }, [
        el("button", {
          class: "btn btn-ghost btn-sm", type: "button", text: "Copy markdown",
          title: "Copy the evidence bundle as markdown. Evidence only — the composed answer is never included.",
          onclick: (event) => copyText(evidenceMarkdown(result), event.currentTarget),
        }),
        el("button", {
          class: "btn btn-ghost btn-sm", type: "button", text: "Copy JSON",
          title: "Copy the raw query response",
          onclick: (event) => copyText(JSON.stringify(result, null, 2), event.currentTarget),
        }),
      ]),
    ]);
    output.appendChild(head);

    const list = el("div", { class: "passages" });
    const relevanceThreshold = (state.lastRequest || {}).relevance_threshold;
    passages.forEach((passage, index) => {
      const low = typeof relevanceThreshold === "number" && passage.relevance < relevanceThreshold;
      list.appendChild(passageCard(passage, index + 1, terms, low));
    });
    output.appendChild(list);
  }

  function renderRunning(message, elapsedMs) {
    clear(ui.results);
    clear(ui.answerSlot);
    ui.results.appendChild(el("div", { class: "running" }, [
      el("span", { class: "spinner", "aria-hidden": "true" }),
      el("span", { class: "running-text" }, [
        el("b", { text: message }),
        document.createTextNode(`  ${(elapsedMs / 1000).toFixed(1)}s`),
      ]),
    ]));
  }

  function renderQueryError(error) {
    clear(ui.results);
    clear(ui.answerSlot);
    if (error.status === 409) {
      ui.results.appendChild(el("div", { class: "empty-state" }, [
        el("p", { class: "empty-title", text: "No index yet" }),
        el("p", { class: "empty-body", text: error.message }),
        el("button", {
          class: "btn btn-primary", type: "button", text: "Go to Index",
          onclick: () => showTab("index"),
        }),
      ]));
      return;
    }
    const card = el("div", { class: "verdict is-error" }, [
      el("div", { class: "verdict-top" }, [
        el("span", { class: "verdict-label" }, [
          el("span", { class: "verdict-icon", text: "!", "aria-hidden": "true" }),
          document.createTextNode("Query failed"),
        ]),
      ]),
      el("p", { class: "verdict-why", text: error.message }),
    ]);
    ui.results.appendChild(card);
  }

  /* ── query: execution ───────────────────────────────────────────────── */

  const STAGES = [
    [0, "Retrieving candidates"],
    [900, "Laya is judging relevance"],
    [2600, "Laya is gating answerability"],
  ];

  function controlValues() {
    const values = {
      top_k: Number(ui.topK.value),
      retrieve_k: Number(ui.retrieveK.value),
      relevance_threshold: Number(ui.relevance.value),
      answerability_threshold: Number(ui.answerability.value),
    };
    // Only send a threshold the user actually changed. The engine resolves its own
    // defaults for the configured backend — Laya's relevance bar, or RERANK_THRESHOLD for
    // a substituted cross-encoder — and a slider that always sent its default would
    // override them.
    if (state.defaults) {
      if (values.relevance_threshold === state.defaults.relevance_threshold) {
        delete values.relevance_threshold;
      }
      if (values.answerability_threshold === state.defaults.answerability_threshold) {
        delete values.answerability_threshold;
      }
    }
    const tenant = ui.tenant ? ui.tenant.value.trim() : "";
    if (tenant) values.tenant = tenant;
    return values;
  }

  function pushHistory(query) {
    const trimmed = query.trim();
    if (!trimmed) return;
    state.history = [trimmed, ...state.history.filter((item) => item !== trimmed)].slice(0, 6);
    store.set("history", JSON.stringify(state.history));
    renderHistory();
  }

  function renderHistory() {
    clear(ui.history);
    state.history.forEach((query) => {
      ui.history.appendChild(el("button", {
        class: "chip", type: "button", text: query, title: query,
        onclick: () => {
          ui.queryInput.value = query;
          ui.queryInput.focus();
        },
      }));
    });
  }

  /** Cancel a retrieval in flight. Deliberately does not go through runQuery's
   *  `state.busy` guard, which is exactly what would block a cancellation. */
  function cancelQuery() {
    if (state.inFlight) state.inFlight.abort();
  }

  async function runQuery(event) {
    if (event) event.preventDefault();
    const query = ui.queryInput.value.trim();
    if (!query || state.busy) return;

    const controller = new AbortController();
    state.inFlight = controller;
    state.busy = true;
    ui.runBtn.disabled = true;
    ui.cancelBtn.hidden = false;
    // A composition in flight belongs to the previous result; stop it before its
    // answer slot is torn down.
    cancelCompose();
    hideAlert();

    const body = Object.assign({ query }, controlValues());
    state.lastRequest = body;
    pushHistory(query);

    const started = performance.now();
    renderRunning(STAGES[0][1], 0);
    ui.srStatus.textContent = "Query running";
    const timer = setInterval(() => {
      const elapsed = performance.now() - started;
      let message = STAGES[0][1];
      for (const [at, text] of STAGES) if (elapsed >= at) message = text;
      renderRunning(message, elapsed);
    }, 120);

    try {
      const result = await api("/v1/query", { method: "POST", body, signal: controller.signal });
      state.lastResult = result;
      renderResult();
      ui.srStatus.textContent = result.answerable
        ? `Answerable, Laya probability ${fmtProb(result.answerability)}, ${result.kept} passages`
        : "Not answerable from the indexed corpus";
      refreshPanels({ quiet: true });
    } catch (error) {
      if (error && error.name === "AbortError") {
        // Replace the spinner, or the panel would sit on "Laya is judging…" forever.
        clear(ui.results);
        clear(ui.answerSlot);
        ui.results.appendChild(el("p", {
          class: "hint empty-hint",
          text: "Query cancelled. The engine keeps working on it in the background; run it again to see the evidence.",
        }));
        ui.srStatus.textContent = "Query cancelled";
        return;
      }
      // Render the failure in the output panel even when it was also an auth problem,
      // otherwise the spinner stays on screen with the message only in the alert.
      handleAuthError(error);
      renderQueryError(error);
      ui.srStatus.textContent = "Query failed";
    } finally {
      clearInterval(timer);
      state.busy = false;
      ui.runBtn.disabled = false;
      ui.cancelBtn.hidden = true;
      if (state.inFlight === controller) state.inFlight = null;
    }
  }

  /* ── composed answers (opt-in, generative, unverified) ──────────────── */

  function composeIsAvailable() {
    return (state.composeMeta || {}).available === true;
  }

  function composeUnavailableReason() {
    return (state.composeMeta || {}).reason
      || "Composition is disabled on the engine, so answers cannot be composed.";
  }

  /**
   * The footer banner. It is the one thing that must never be wrong: while
   * composition is on, the engine is returning generated text that nobody has
   * verified, and the page has to say so.
   */
  function renderFooter() {
    const on = state.compose && composeIsAvailable();
    ui.footer.classList.toggle("is-warning", on);
    clear(ui.footerNote);
    if (!on) {
      ui.footerNote.appendChild(document.createTextNode(
        "Laya is a non-autoregressive decision engine. It cannot generate text, so there is nothing to hallucinate."));
      return;
    }
    const model = (state.composeMeta || {}).model || "a local model";
    ui.footerNote.appendChild(el("b", { text: "Composed answers may contain hallucinations." }));
    ui.footerNote.appendChild(document.createTextNode(
      ` The prose is written by ${model}, not by Laya, and is not verified against the corpus. ` +
      "Laya's verdict and evidence above are calibrated; the answer is not — check every claim against the cited passages."));
  }

  function renderComposeSwitch() {
    const available = composeIsAvailable();
    ui.composeToggle.disabled = !available;
    ui.composeSwitch.classList.toggle("is-disabled", !available);
    if (!available) {
      // Never leave the switch on when there is nothing behind it.
      state.compose = false;
      ui.composeToggle.checked = false;
      ui.composeSwitch.classList.remove("is-on");
      ui.composeHint.textContent = composeUnavailableReason();
      renderFooter();
      return;
    }
    const meta = state.composeMeta || {};
    ui.composeToggle.checked = state.compose;
    ui.composeSwitch.classList.toggle("is-on", state.compose);
    ui.composeHint.textContent =
      `${meta.model} at ${meta.base_url} · unverified prose, not a Laya measurement`;
    renderFooter();
  }

  /**
   * Stream a composed answer into `slot`.
   *
   * The engine sends newline-delimited JSON: `delta` while the model writes, then one
   * terminal `done` or `error`. An `error` can arrive after some deltas, so any text
   * already on screen is marked incomplete rather than silently kept.
   */
  async function composeAnswer(result, slot) {
    if (state.composeAbort) state.composeAbort.abort();
    const controller = new AbortController();
    state.composeAbort = controller;

    const meta = state.composeMeta || {};
    // Never ask the model to answer nothing: a blank question wastes a model call and
    // comes back as prose asking for the question, which reads like a real answer.
    if (!(result.query || "").trim()) {
      clear(slot);
      slot.appendChild(el("p", {
        class: "hint empty-hint",
        text: "Cannot compose: the query this evidence came from is empty.",
      }));
      return;
    }
    const body = el("div", { class: "answer-body is-waiting", text: "Waiting for the model…" });
    const copyBtn = el("button", {
      class: "btn btn-ghost btn-sm", type: "button", text: "Copy", disabled: true,
      title: "Copy the composed answer",
      onclick: (event) => copyText(body.dataset.text || "", event.currentTarget),
    });
    const stopBtn = el("button", {
      class: "btn btn-ghost btn-sm", type: "button", text: "Stop",
      title: "Stop generating. Any text already written is kept.",
      onclick: () => controller.abort(),
    });    // Filled in once the answer is complete: a truncation warning, then the
    // deterministic citation check.
    const warnSlot = el("div");
    const checkSlot = el("div");
    const adjSlot = el("div");
    // Naming the question the answer was written from makes a mismatch impossible to
    // miss: an answer that asks for the question is then obviously a model failure
    // rather than something the reader has to guess about.
    const asked = (result.query || "").trim();
    const panel = el("section", { class: "answer", dataset: { state: "streaming" } }, [
      el("div", { class: "answer-head" }, [
        el("span", { class: "answer-badge" }, [
          el("span", { text: "✎", "aria-hidden": "true" }),
          document.createTextNode("Composed"),
        ]),
        el("span", { class: "answer-model", text: meta.model || "" }),
        el("span", { class: "actions" }, [copyBtn, stopBtn]),
      ]),
      el("p", { class: "answer-asked" }, [
        el("span", { class: "answer-asked-label", text: "Question" }),
        el("span", { class: "answer-asked-text", text: asked || "(none)" }),
      ]),
      body,
      warnSlot,
      checkSlot,
      adjSlot,
      el("div", {
        class: "answer-caveat",
        text: "Written by a generative model from the evidence below. This is not a Laya output and has not been verified.",
      }),
    ]);
    clear(slot);
    slot.appendChild(panel);

    let text = "";
    let finished = false;
    let failed = false;
    let truncated = false;

    const paint = () => {
      body.classList.toggle("is-waiting", !text);
      body.textContent = text || "Waiting for the model…";
      if (text && !finished) body.appendChild(el("span", { class: "answer-caret", "aria-hidden": "true" }));
      body.dataset.text = text;
    };

    const addError = (message) => {
      failed = true;
      // An error can arrive after some deltas, so the state distinguishes "never
      // produced anything" from "produced something and then broke".
      panel.dataset.state = "error";
      panel.appendChild(el("div", { class: "answer-error" }, [
        el("b", { text: text ? "Answer incomplete: " : "No answer: " }),
        document.createTextNode(message),
      ]));
    };

    try {
      const headers = { "Content-Type": "application/json" };
      if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
      const response = await fetch("/v1/compose", {
        method: "POST",
        headers,
        signal: controller.signal,
        cache: "no-store",
        body: JSON.stringify({
          query: result.query,
          // Exactly the evidence on screen: the answer is never written from
          // passages the reader cannot see. Grouping by document here is what makes
          // the numbering the model sees match the numbered cards below.
          passages: evidenceFor(result).map((p) => ({ text: p.text, path: p.path })),
        }),
      });

      if (!response.ok) {
        if (response.status === 401) handleAuthError(new ApiError("unauthorised", 401));
        throw new ApiError(await detailFrom(response), response.status);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, cut).trim();
          buffer = buffer.slice(cut + 1);
          if (!line) continue;
          let event;
          try { event = JSON.parse(line); } catch (_) { continue; }
          if (event.type === "delta") {
            text += event.text || "";
            paint();
          } else if (event.type === "error") {
            addError(event.message || "the model stopped unexpectedly");
          } else if (event.type === "done") {
            ui.srStatus.textContent = `Composed answer ready from ${event.model || meta.model}`;
            if (event.finish_reason === "length") {
              // A truncated answer otherwise looks exactly like a finished one. This is
              // a warning about completeness, not a failure, so it is not an `.answer-error`.
              truncated = true;
              warnSlot.appendChild(el("div", { class: "answer-warn" }, [
                el("b", { text: "Answer truncated: " }),
                document.createTextNode(
                  "the model hit its token limit before finishing. The text above stops mid-thought — " +
                  "raise LAYA_RAG_COMPOSE_MAX_TOKENS for a longer answer."),
              ]));
            }
          }
        }
      }
    } catch (error) {
      if (error && error.name === "AbortError") {
        addError("generation was stopped before it finished.");
      } else {
        if (error && error.status === 401) handleAuthError(error);
        addError(error && error.message ? error.message : "the model could not be reached");
      }
    } finally {
      finished = true;
      stopBtn.remove();
      copyBtn.disabled = !text;
      paint();
      if (!failed) panel.dataset.state = truncated ? "truncated" : text ? "done" : "empty";
      if (state.composeAbort === controller) state.composeAbort = null;
      // Check citations only on an answer that actually finished: a partial answer
      // would report missing citations that simply had not been written yet.
      if (!failed && text) {
        checkCitations(text, result, checkSlot);
        adjudicateAnswer(text, adjSlot);
      }
    }
  }

  /**
   * Layer 1: deterministic citation and quote checks.
   *
   * No model and no threshold are involved, so there is no probability here to
   * misread — which is exactly why the result can be shown at all. It says whether
   * the answer cites the evidence and whether what it quotes is really there; it
   * says nothing about whether the claims are true, and the wording has to keep
   * that distinction visible.
   */
  async function checkCitations(answerText, result, slot) {
    clear(slot);
    slot.appendChild(el("div", { class: "cite-check", dataset: { state: "running" } }, [
      el("span", { class: "spinner", "aria-hidden": "true" }),
      el("span", { text: "Checking citations…" }),
    ]));

    let report;
    try {
      report = await api("/v1/verify", {
        method: "POST",
        body: {
          answer: answerText,
          // Pass the question so an answer that quotes it back is not flagged as a
          // fabricated quote.
          query: result.query || "",
          passages: evidenceFor(result).map((p) => ({ text: p.text, path: p.path })),
        },
      });
    } catch (error) {
      if (error && error.status === 401) handleAuthError(error);
      clear(slot);
      slot.appendChild(el("div", { class: "cite-check", dataset: { state: "error" } }, [
        el("span", { text: `Citation check unavailable: ${error && error.message ? error.message : "request failed"}` }),
      ]));
      return;
    }
    renderCitationCheck(slot, report);
  }

  function renderCitationCheck(slot, report) {
    const counts = report.counts || {};
    const flagged = (report.claims || []).filter((claim) => (claim.issues || []).length);
    const missingQuotes = (report.quotes || []).filter((quote) => quote.status === "not_found");
    const problems = flagged.length + missingQuotes.length;

    const box = el("div", {
      class: `cite-check${report.clean ? "" : " is-flagged"}`,
      dataset: { state: report.clean ? "clean" : "flagged" },
    });

    box.appendChild(el("div", { class: "cite-head" }, [
      el("span", { class: "cite-badge", text: "Citation check" }),
      el("span", {
        class: "cite-counts",
        text: `${fmtInt(counts.claims)} claims · ${fmtInt(counts.cited)} cited · ` +
              `${fmtInt(counts.uncited)} uncited · ${fmtInt(counts.quotes_found)}/${fmtInt(counts.quotes_checked)} quotes found`,
      }),
    ]));
    box.appendChild(el("p", { class: "cite-summary", text: report.summary }));

    if (problems) {
      const details = el("details", { class: "cite-details" }, [
        el("summary", { text: `${problems} item${problems === 1 ? "" : "s"} to check` }),
      ]);
      const list = el("ul");
      flagged.forEach((claim) => {
        list.appendChild(el("li", {}, [
          el("span", { class: "cite-issue", text: claim.issues.join(", ") }),
          document.createTextNode(claim.text),
        ]));
      });
      missingQuotes.forEach((quote) => {
        list.appendChild(el("li", {}, [
          el("span", { class: "cite-issue", text: "quote not found" }),
          document.createTextNode(`“${quote.text}”`),
        ]));
      });
      details.appendChild(list);
      box.appendChild(details);
    }

    box.appendChild(el("p", {
      class: "cite-caveat",
      text: "Checks that claims cite the evidence and that quoted text appears in it. It does not check whether the claims are true.",
    }));

    clear(slot);
    slot.appendChild(box);
  }

  /**
   * Layer 3 + variant 2a: does the corpus actually support each claim?
   *
   * This re-retrieves every claim across the whole index and adjudicates it with
   * Laya's calibrated decisions, reusing the already-measured answerability question.
   * It costs real time (seconds, not milliseconds), so it shows elapsed progress
   * rather than a bare spinner.
   */
  async function adjudicateAnswer(answerText, slot) {
    clear(slot);
    const started = performance.now();
    const label = el("span", { text: "Adjudicating claims against the corpus… 0.0s" });
    slot.appendChild(el("div", { class: "adj", dataset: { state: "running" } }, [
      el("span", { class: "spinner", "aria-hidden": "true" }),
      label,
    ]));
    const timer = setInterval(() => {
      label.textContent =
        `Adjudicating claims against the corpus… ${((performance.now() - started) / 1000).toFixed(1)}s`;
    }, 200);

    try {
      const report = await api("/v1/adjudicate", { method: "POST", body: { answer: answerText } });
      renderAdjudication(slot, report);
    } catch (error) {
      if (error && error.status === 401) handleAuthError(error);
      clear(slot);
      // No index is a normal state, not a failure — say which it is.
      const noIndex = error && error.status === 409;
      slot.appendChild(el("div", { class: "adj", dataset: { state: "unavailable" } }, [
        el("span", {
          text: noIndex
            ? "Corpus support: nothing is indexed yet, so there is no corpus to check against."
            : `Corpus support unavailable: ${error && error.message ? error.message : "request failed"}`,
        }),
      ]));
    } finally {
      clearInterval(timer);
    }
  }

  function renderAdjudication(slot, report) {
    const total = report.total || 0;
    clear(slot);
    if (!total) {
      slot.appendChild(el("div", { class: "adj", dataset: { state: "unavailable" } }, [
        el("span", { text: report.note || "Nothing in the answer was checkable as a claim." }),
      ]));
      return;
    }

    const bad = (report.unsupported || 0) + (report.not_measured || 0);
    const box = el("div", { class: `adj${bad ? " is-flagged" : ""}`, dataset: { state: "done" } });

    const counts = [
      `${fmtInt(total)} claims`,
      `${fmtInt(report.supported)} supported`,
      report.unsupported ? `${fmtInt(report.unsupported)} unsupported` : null,
      report.not_measured ? `${fmtInt(report.not_measured)} not measured` : null,
    ].filter(Boolean).join(" · ");

    box.appendChild(el("div", { class: "adj-head" }, [
      el("span", { class: "adj-badge", text: "Corpus support" }),
      el("span", { class: "adj-counts", text: counts }),
    ]));

    const list = el("ul", { class: "adj-list" });
    (report.claims || []).forEach((claim) => {
      // "not measured" is not a low score: it means no passage cleared the relevance
      // bar, so Laya was never asked. Printing 0.000 there would invent a decision.
      const mark = claim.supported ? "✓" : claim.measured ? "✗" : "·";
      const score = claim.measured ? claim.support_probability.toFixed(3) : "not measured";
      const sources = (claim.sources || []).map((source) => basename(source.path));
      list.appendChild(el("li", { class: `adj-item${claim.supported ? "" : " is-bad"}` }, [
        el("span", { class: "adj-mark", text: mark, "aria-hidden": "true" }),
        el("span", { class: "adj-p", text: score }),
        el("span", { class: "adj-text", text: claim.text }),
        sources.length
          ? el("span", { class: "adj-src", text: sources.join(", "), title: claim.sources.map((s) => s.path).join("\n") })
          : el("span", { class: "adj-src", text: "no match in the corpus" }),
      ]));
    });
    box.appendChild(list);

    box.appendChild(el("p", {
      class: "adj-caveat",
      text: `Reuses Laya's measured answerability question (“${report.question}”) at P ≥ ${fmtProb(report.answerability_threshold)}. `
        + "That threshold was benchmarked for questions, not for claim support, so read it as a strong signal rather than a verdict. "
        + "“not measured” means no passage in the corpus cleared the relevance bar.",
    }));

    slot.appendChild(box);
  }

  /** Pull a FastAPI `detail` out of a failed streaming response. */
  async function detailFrom(response) {
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === "string") return payload.detail;
    } catch (_) { /* not JSON */ }
    return `the engine refused to compose (HTTP ${response.status})`;
  }

  function cancelCompose() {
    if (state.composeAbort) state.composeAbort.abort();
  }

  /* ── corpus panel ───────────────────────────────────────────────────── */

  function renderCorpus() {
    const list = ui.corpusList;
    clear(list);
    const filter = state.filter.toLowerCase();
    const documents = state.documents.filter(
      (doc) => !filter || String(doc.path).toLowerCase().includes(filter)
    );

    ui.corpusCount.textContent = state.documents.length
      ? `${documents.length} / ${state.documents.length}`
      : "";

    if (!state.documents.length) {
      list.appendChild(el("div", { class: "empty-state" }, [
        el("p", { class: "empty-title", text: "Corpus is empty" }),
        el("p", { class: "empty-body", text: "Index a directory on the Index tab to populate it." }),
      ]));
      renderChunkViewer();
      return;
    }

    if (!documents.length) {
      list.appendChild(el("p", { class: "hint", text: "No path matches that filter." }));
      renderChunkViewer();
      return;
    }

    for (const doc of documents) {
      list.appendChild(el("button", {
        class: `doc-row${state.activeDoc === doc.path ? " is-active" : ""}`,
        type: "button",
        title: doc.path,
        onclick: () => selectDoc(doc.path),
      }, [
        el("span", { class: "doc-path", text: basename(doc.path), title: doc.path }),
        el("span", { class: "doc-meta" }, [
          el("span", { text: dirname(doc.path) }),
          el("span", { text: `${fmtInt(doc.chunks)} chunk${doc.chunks === 1 ? "" : "s"}` }),
          el("span", { text: `${fmtInt(doc.chars)} chars` }),
        ]),
      ]));
    }
    renderChunkViewer();
  }

  async function selectDoc(path) {
    if (state.activeDoc === path) {
      state.activeDoc = null;
      renderCorpus();
      return;
    }
    state.activeDoc = path;
    state.activeChunks = null;
    state.chunkError = null;
    renderCorpus();
    try {
      const payload = await api(`/v1/chunks?path=${encodeURIComponent(path)}`);
      if (state.activeDoc !== path) return; // a different document was opened meanwhile
      state.activeChunks = payload.chunks || [];
    } catch (error) {
      handleAuthError(error);
      if (state.activeDoc !== path) return;
      state.activeChunks = [];
      state.chunkError = error.message;
    }
    renderChunkViewer();
  }

  function renderChunkViewer() {
    const viewer = ui.chunkViewer;
    clear(viewer);
    if (!state.activeDoc) { viewer.hidden = true; return; }
    viewer.hidden = false;

    viewer.appendChild(el("header", {}, [
      el("span", { class: "doc-path", text: state.activeDoc }),
      el("button", {
        class: "btn btn-ghost btn-sm", type: "button", text: "Close",
        onclick: () => { state.activeDoc = null; renderCorpus(); },
      }),
    ]));

    if (state.chunkError) {
      viewer.appendChild(el("p", { class: "hint", text: state.chunkError }));
      return;
    }
    if (!state.activeChunks) {
      viewer.appendChild(el("p", { class: "hint", text: "Loading chunks…" }));
      return;
    }
    for (const chunk of state.activeChunks) {
      viewer.appendChild(el("div", { class: "chunk" }, [
        el("div", { class: "chunk-meta" }, [
          el("span", { text: chunk.chunk_id }),
          el("span", { text: `chars ${fmtInt(chunk.start_char)}–${fmtInt(chunk.end_char)}` }),
          el("span", { text: `${fmtInt(chunk.chars)} chars` }),
        ]),
        el("div", { class: "chunk-text", text: chunk.text }),
      ]));
    }
  }

  /* ── index panel ────────────────────────────────────────────────────── */

  function renderIndexReport(report) {
    const output = ui.indexOutput;
    clear(output);
    const box = el("div", { class: "report" }, [el("h4", { text: "Index report" })]);

    const rows = [
      ["documents read", fmtInt(report.documents)],
      ["chunks created", fmtInt(report.chunks)],
      ["stale chunks removed", fmtInt(report.removed_chunks)],
      ["chunks in index", fmtInt(report.total_chunks)],
      ["files in index", fmtInt(report.paths)],
      ["embed", fmtSeconds(report.embed_seconds)],
      ["save", fmtSeconds(report.save_seconds)],
      ["total", fmtSeconds(report.total_seconds)],
    ];
    const grid = el("dl", { class: "report-grid" });
    for (const [key, value] of rows) {
      grid.appendChild(el("dt", { text: key }));
      grid.appendChild(el("dd", { text: value }));
    }
    box.appendChild(grid);

    if (report.note) {
      box.appendChild(el("p", { class: "note" }, [
        el("span", { text: "⚠", "aria-hidden": "true" }),
        el("span", { text: report.note }),
      ]));
    }

    const skipped = report.skipped || [];
    if (skipped.length) {
      const details = el("details", { class: "skipped" }, [
        el("summary", { text: `${skipped.length} path${skipped.length === 1 ? "" : "s"} skipped` }),
      ]);
      const list = el("ul");
      for (const item of skipped) {
        list.appendChild(el("li", { text: `${item.path} — ${item.reason}` }));
      }
      details.appendChild(list);
      box.appendChild(details);
    }
    output.appendChild(box);
  }

  /**
   * Index over the streaming endpoint, painting running counts as they arrive.
   *
   * Indexing is one long request, so the plain JSON call can only show a spinner that
   * says nothing while a large corpus embeds. The stream emits a counter event per file
   * read, per document chunked and per embedding batch, so the wait is legible.
   */
  async function indexWithProgress(body, scores) {
    const headers = { "Content-Type": "application/json" };
    if (state.token) headers["Authorization"] = `Bearer ${state.token}`;

    let response;
    try {
      response = await fetch("/v1/index/stream", {
        method: "POST", headers, cache: "no-store", body: JSON.stringify(body),
      });
    } catch (_) {
      throw new ApiError("could not reach the engine — is `laya-rag-serve` running?", 0);
    }
    if (!response.ok) {
      let detail = null;
      try { detail = await response.json(); } catch (_) { detail = null; }
      throw new ApiError(detailOf(detail, response.status), response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let report = null;
    let failure = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, cut).trim();
        buffer = buffer.slice(cut + 1);
        if (!line) continue;
        let event;
        try { event = JSON.parse(line); } catch (_) { continue; }
        if (event.type === "progress") paintIndexProgress(scores, event);
        else if (event.type === "done") report = event.report;
        else if (event.type === "error") failure = event;
      }
    }
    if (failure) throw new ApiError(failure.message || "indexing failed", failure.status || 0);
    // A stream that closes without a report is a failure, not a silent success: the
    // index may or may not have been written, and saying nothing would hide that.
    if (!report) throw new ApiError("the index stream ended without a report", 0);
    return report;
  }

  /** One line of running counters for the current indexing stage. */
  function paintIndexProgress(node, event) {
    const parts = [];
    if (event.stage === "reading") {
      parts.push(`${fmtInt(event.documents)} file${event.documents === 1 ? "" : "s"} read`);
      if (event.skipped) parts.push(`${fmtInt(event.skipped)} skipped`);
    } else if (event.stage === "chunking") {
      if (event.documents_total) {
        parts.push(`${fmtInt(event.documents)}/${fmtInt(event.documents_total)} documents chunked`);
      }
      parts.push(`${fmtInt(event.chunks)} chunks`);
    } else if (event.stage === "embedding") {
      const total = event.chunks || 0;
      const pct = total ? Math.floor((event.embedded / total) * 100) : 0;
      parts.push(`embedded ${fmtInt(event.embedded)}/${fmtInt(total)} chunks (${pct}%)`);
    } else if (event.stage === "saving") {
      parts.push(`${fmtInt(event.chunks)} chunks`);
      parts.push("saving index…");
    }
    node.textContent = parts.join(" · ");
  }

  async function runIndex(event) {
    event.preventDefault();
    if (state.busy) return;

    const paths = ui.indexPaths.value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (!paths.length) {
      showAlert("error", "Nothing to index", "Enter at least one file or directory path.");
      ui.indexPaths.focus();
      return;
    }

    const extensions = ui.indexExt.value.trim();
    const body = { paths, rebuild: ui.indexRebuild.checked };
    if (extensions) {
      body.extensions = extensions.split(",").map((item) => item.trim()).filter(Boolean);
    }

    state.busy = true;
    ui.indexBtn.disabled = true;
    ui.resetBtn.disabled = true;
    ui.indexBtn.textContent = "Indexing…";
    hideAlert();
    clear(ui.indexOutput);
    const scores = el("span", { class: "running-scores" });
    ui.indexOutput.appendChild(el("div", { class: "running" }, [
      el("span", { class: "spinner", "aria-hidden": "true" }),
      el("span", { class: "running-text", text: "Reading, chunking and embedding…" }),
      scores,
    ]));

    try {
      const report = await indexWithProgress(body, scores);
      renderIndexReport(report);
      ui.srStatus.textContent = `Indexed ${report.documents} documents, ${report.total_chunks} chunks total`;
      await refreshPanels({ quiet: true });
    } catch (error) {
      clear(ui.indexOutput);
      if (!handleAuthError(error)) {
        showAlert("error", "Indexing failed", error.message);
        ui.indexOutput.appendChild(el("p", { class: "hint", text: error.message }));
      }
    } finally {
      state.busy = false;
      ui.indexBtn.disabled = false;
      ui.resetBtn.disabled = false;
      ui.indexBtn.textContent = "Index";
    }
  }

  async function resetIndex() {
    // Deleting the index while an index build or a query is in flight races wipe()
    // against the engine's own save/read and leaves disk and memory disagreeing.
    if (state.busy) return;
    if (!window.confirm("Delete the persisted index? The corpus on disk is not touched.")) return;

    state.busy = true;
    ui.resetBtn.disabled = true;
    ui.indexBtn.disabled = true;
    cancelCompose();
    try {
      await api("/v1/reset", { method: "POST" });
      clear(ui.indexOutput);
      state.documents = [];
      state.activeDoc = null;
      state.activeChunks = null;
      state.chunkError = null;
      state.lastResult = null;
      clear(ui.results);
      clear(ui.answerSlot);
      ui.results.appendChild(ui.queryEmpty);
      ui.queryEmpty.hidden = false;
      await refreshPanels();
      ui.srStatus.textContent = "Index deleted";
    } catch (error) {
      if (!handleAuthError(error)) showAlert("error", "Reset failed", error.message);
    } finally {
      state.busy = false;
      ui.resetBtn.disabled = false;
      ui.indexBtn.disabled = false;
    }
  }

  /* ── engine panel ───────────────────────────────────────────────────── */

  function kvSection(title, pairs) {
    const fragment = document.createDocumentFragment();
    fragment.appendChild(el("h3", { class: "section-title", text: title }));
    const list = el("dl", { class: "kv" });
    for (const [key, value] of pairs) {
      if (value === undefined || value === null || value === "") continue;
      list.appendChild(el("dt", { text: key }));
      list.appendChild(el("dd", { text: String(value) }));
    }
    fragment.appendChild(list);
    return fragment;
  }

  function renderEngine() {
    const output = ui.engineOutput;
    clear(output);
    if (!state.stats) {
      output.appendChild(el("p", { class: "hint", text: "Engine state unavailable." }));
      return;
    }
    const stats = state.stats;
    const config = stats.config || {};
    const model = stats.model || {};
    const index = stats.index || {};
    const health = state.health || {};

    output.appendChild(kvSection("Service", [
      ["status", health.status || "—"],
      ["health error", health.error],
      ["index dir", stats.index_dir],
      ["index present", stats.index_present ? "yes" : "no"],
      ["lexical channel", stats.lexical_present ? "present" : "missing"],
      ["index error", stats.index_error],
    ]));

    output.appendChild(kvSection("Corpus", [
      ["chunks", fmtInt(index.chunks)],
      ["documents", fmtInt(index.documents)],
      ["files", fmtInt(index.paths)],
      ["embedding dim", fmtInt(index.dim)],
    ]));

    // LayaModel.info() reports these keys; it has no "model" name field, so the panel
    // shows the encoder budget and load facts instead of a row that never exists.
    output.appendChild(kvSection("Model", [
      ["checkpoint", config.checkpoint],
      ["device", config.device],
      ["loaded", health.model_loaded ? "yes" : "no"],
      ["embedding dim", model.embedding_dim],
      ["state max_len", model.max_len],
      ["head max_len", model.head_max_len],
      ["load seconds", model.load_seconds],
      ["models dir", model.models_dir || config.models_dir],
    ]));

    output.appendChild(kvSection("Retrieval funnel", [
      ["retrieve_k", config.retrieve_k],
      ["rerank_k", config.rerank_k],
      ["relevance threshold", config.relevance_threshold],
      ["answerability threshold", config.answerability_threshold],
      ["chunk chars", config.chunk_chars],
      ["chunk overlap", config.chunk_overlap_chars],
      ["max_len", config.max_len],
      ["embed max length", config.embed_max_length],
    ]));

    output.appendChild(kvSection("API key", [
      ["state", config.api_key === "<redacted>" ? "configured (never disclosed)" : "not configured"],
    ]));
  }

  /* ── controls ───────────────────────────────────────────────────────── */

  function applyDefaults() {
    const config = (state.stats || {}).config || {};
    // With a cross-encoder configured the engine cuts its scores at RERANK_THRESHOLD, not
    // at Laya's RELEVANCE_THRESHOLD, so the slider should show the value actually used.
    const usesRerank = String(config.rerank_model || "").trim() !== "";
    const relevanceDefault = usesRerank && isFinite(Number(config.rerank_threshold))
      ? Number(config.rerank_threshold)
      : Number(config.relevance_threshold);
    const defaults = {
      top_k: Number(config.rerank_k) || 5,
      retrieve_k: Number(config.retrieve_k) || 20,
      relevance_threshold: relevanceDefault,
      answerability_threshold: Number(config.answerability_threshold),
    };
    if (!isFinite(defaults.relevance_threshold)) defaults.relevance_threshold = usesRerank ? 0 : 0.5;
    if (!isFinite(defaults.answerability_threshold)) defaults.answerability_threshold = 0.5;

    ui.topK.value = defaults.top_k;
    ui.retrieveK.value = defaults.retrieve_k;
    ui.relevance.value = defaults.relevance_threshold;
    ui.answerability.value = defaults.answerability_threshold;
    if (ui.tenant) ui.tenant.value = "";
    state.defaults = defaults;
    syncControlLabels();
  }

  function syncControlLabels() {
    ui.topKVal.textContent = ui.topK.value;
    ui.retrieveKVal.textContent = ui.retrieveK.value;
    ui.relevanceVal.textContent = Number(ui.relevance.value).toFixed(2);
    ui.answerabilityVal.textContent = Number(ui.answerability.value).toFixed(2);

    const changed = state.defaults && (
      Number(ui.topK.value) !== state.defaults.top_k ||
      Number(ui.retrieveK.value) !== state.defaults.retrieve_k ||
      Number(ui.relevance.value) !== state.defaults.relevance_threshold ||
      Number(ui.answerability.value) !== state.defaults.answerability_threshold ||
      (ui.tenant && ui.tenant.value.trim() !== "")
    );
    ui.controlsSummary.textContent = changed ? "customised" : "engine defaults";
  }

  /* ── tabs ───────────────────────────────────────────────────────────── */

  const TABS = [
    { name: "corpus", tab: "tabCorpus", panel: "panelCorpus" },
    { name: "index", tab: "tabIndex", panel: "panelIndex" },
    { name: "engine", tab: "tabEngine", panel: "panelEngine" },
  ];

  function showTab(name) {
    for (const entry of TABS) {
      const active = entry.name === name;
      ui[entry.tab].classList.toggle("is-active", active);
      ui[entry.tab].setAttribute("aria-selected", active ? "true" : "false");
      // Roving tabindex, per the ARIA tabs pattern: only the selected tab is a tab stop.
      ui[entry.tab].tabIndex = active ? 0 : -1;
      ui[entry.panel].classList.toggle("is-active", active);
    }
  }

  /** Move tab focus with the arrow keys, and jump with Home/End. */
  function moveTabFocus(entry, event) {
    const index = TABS.indexOf(entry);
    let next = null;
    if (event.key === "ArrowRight") next = TABS[(index + 1) % TABS.length];
    else if (event.key === "ArrowLeft") next = TABS[(index - 1 + TABS.length) % TABS.length];
    else if (event.key === "Home") next = TABS[0];
    else if (event.key === "End") next = TABS[TABS.length - 1];
    if (!next) return;
    event.preventDefault();
    showTab(next.name);
    ui[next.tab].focus();
  }

  /* ── refresh ────────────────────────────────────────────────────────── */

  async function refreshPanels(options) {
    const opts = options || {};
    if (!opts.quiet) ui.refreshBtn.disabled = true;
    state.authBlocked = false;

    try {
      const health = await api("/health");
      const stats = await api("/v1/stats");
      state.health = health;
      state.stats = stats;
      state.composeMeta = stats.compose || null;
      state.engineError = null;
      hideAlert();
      if (!state.defaults) applyDefaults();
    } catch (error) {
      state.health = null;
      state.stats = null;
      state.engineError = error;
      if (handleAuthError(error)) {
        // Every other endpoint needs the same token, so stop here rather than
        // painting an "unreachable" state that misdescribes the problem.
        state.authBlocked = true;
        state.documents = [];
        state.composeMeta = null;
        renderStatus();
        renderConnection();
        renderEngine();
        renderCorpus();
        renderComposeSwitch();
        ui.refreshBtn.disabled = false;
        return;
      }
      if (error.status === 0) {
        showAlert(
          "error",
          "Cannot reach the engine",
          "Start it with `.venv/bin/laya-rag-serve`, then refresh. The GUI is served by the same process."
        );
      } else {
        showAlert("error", "Engine reported a problem", error.message);
      }
    }

    try {
      const payload = await api("/v1/documents");
      state.documents = payload.documents || [];
    } catch (_) {
      state.documents = [];
    }

    renderStatus();
    renderConnection();
    renderCorpus();
    renderEngine();
    renderComposeSwitch();
    if (!opts.quiet) ui.refreshBtn.disabled = false;
  }

  /* ── wiring ─────────────────────────────────────────────────────────── */

  function cacheElements() {
    Object.assign(ui, {
      status: $("status-strip"),
      alert: $("alert"),
      connPanel: $("conn-panel"),
      connBtn: $("conn-btn"),
      connMeta: $("conn-meta"),
      tokenInput: $("token-input"),
      tokenSave: $("token-save"),
      tokenClear: $("token-clear"),
      refreshBtn: $("refresh-btn"),
      themeBtn: $("theme-btn"),
      queryForm: $("query-form"),
      queryInput: $("query-input"),
      results: $("results"),
      queryEmpty: $("query-empty"),
      answerSlot: $("answer-slot"),
      composeToggle: $("compose-toggle"),
      composeSwitch: $("compose-switch"),
      composeHint: $("compose-hint"),
      footer: $("footer"),
      footerNote: $("footer-note"),
      runBtn: $("run-btn"),
      cancelBtn: $("cancel-btn"),
      clearBtn: $("clear-btn"),
      history: $("history"),
      controls: $("controls"),
      controlsSummary: $("controls-summary"),
      topK: $("ctl-topk"), topKVal: $("ctl-topk-val"),
      retrieveK: $("ctl-retrievek"), retrieveKVal: $("ctl-retrievek-val"),
      relevance: $("ctl-relevance"), relevanceVal: $("ctl-relevance-val"),
      answerability: $("ctl-answerability"), answerabilityVal: $("ctl-answerability-val"),
      tenant: $("ctl-tenant"),
      ctlReset: $("ctl-reset"),
      corpusFilter: $("corpus-filter"),
      corpusList: $("corpus-list"),
      corpusCount: $("corpus-count"),
      chunkViewer: $("chunk-viewer"),
      indexForm: $("index-form"),
      indexPaths: $("index-paths"),
      indexExt: $("index-ext"),
      indexRebuild: $("index-rebuild"),
      indexBtn: $("index-btn"),
      indexOutput: $("index-output"),
      resetBtn: $("reset-btn"),
      engineOutput: $("engine-output"),
      srStatus: $("sr-status"),
      tabCorpus: $("tab-corpus"),
      tabIndex: $("tab-index"),
      tabEngine: $("tab-engine"),
      panelCorpus: $("panel-corpus"),
      panelIndex: $("panel-index"),
      panelEngine: $("panel-engine"),
    });
  }

  function wire() {
    ui.refreshBtn.addEventListener("click", () => refreshPanels());
    ui.themeBtn.addEventListener("click", () => {
      state.theme = THEMES[(THEMES.indexOf(state.theme) + 1) % THEMES.length];
      store.set("theme", state.theme);
      applyTheme();
    });

    ui.connBtn.addEventListener("click", () => {
      if (ui.connPanel.hidden) openConnection(); else closeConnection();
    });
    ui.tokenSave.addEventListener("click", () => {
      state.token = ui.tokenInput.value.trim();
      if (state.token) store.set("token", state.token); else store.del("token");
      ui.tokenInput.value = "";
      closeConnection();
      hideAlert();
      refreshPanels();
    });
    ui.tokenClear.addEventListener("click", () => {
      state.token = "";
      store.del("token");
      ui.tokenInput.value = "";
      renderConnection();
      refreshPanels();
    });

    ui.queryForm.addEventListener("submit", runQuery);
    ui.queryInput.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runQuery(event);
    });
    ui.clearBtn.addEventListener("click", () => {
      cancelCompose();
      ui.queryInput.value = "";
      state.lastResult = null;
      clear(ui.results);
      clear(ui.answerSlot);
      ui.results.appendChild(ui.queryEmpty);
      ui.queryEmpty.hidden = false;
      ui.queryInput.focus();
    });

    ui.composeToggle.addEventListener("change", () => {
      state.compose = ui.composeToggle.checked && composeIsAvailable();
      store.set("compose", state.compose ? "1" : "0");
      renderComposeSwitch();
      // Toggling with a result already on screen composes it, or drops the answer,
      // without making the user run the query again.
      if (state.lastResult) renderResult();
    });

    for (const control of [ui.topK, ui.retrieveK, ui.relevance, ui.answerability, ui.tenant]) {
      if (control) control.addEventListener("input", syncControlLabels);
    }
    ui.ctlReset.addEventListener("click", () => { applyDefaults(); });

    ui.corpusFilter.addEventListener("input", () => {
      state.filter = ui.corpusFilter.value;
      renderCorpus();
    });

    ui.indexForm.addEventListener("submit", runIndex);
    ui.resetBtn.addEventListener("click", resetIndex);

    for (const entry of TABS) {
      ui[entry.tab].addEventListener("click", () => showTab(entry.name));
      ui[entry.tab].addEventListener("keydown", (event) => moveTabFocus(entry, event));
    }

    ui.cancelBtn.addEventListener("click", cancelQuery);

    /* Auto-refresh while the tab is visible, but never during a run. */
    let tick = 0;
    setInterval(() => {
      if (document.hidden || state.busy) return;
      tick += 1;
      if (tick >= 5) {
        tick = 0;
        refreshPanels({ quiet: true }).catch(() => { /* a render error must not kill the timer */ });
      }
    }, 4000);
  }

  async function init() {
    cacheElements();
    applyTheme();
    // Wire first: if anything below throws, the interface must still be operable.
    wire();
    showTab("corpus");
    renderHistory();
    ui.tokenInput.value = "";
    // The range inputs carry sane values in the markup, so the labels are correct
    // even if /v1/stats never answers and applyDefaults() never runs.
    syncControlLabels();
    try {
      await refreshPanels();
    } catch (_) {
      showAlert("error", "The interface failed to start", "Reload the page; if it persists, check the browser console.");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
