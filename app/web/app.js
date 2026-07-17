/* Redaction Studio — vanilla JS (no framework, no CDN, no build).
   Streams the staged pipeline over SSE (/api/mask/stream): the document is shown
   the instant it's read, then the parse sample, PII screen, bounding boxes and
   masked result reveal as each stage finishes — so the ~minute wait is live. */
(function () {
  "use strict";

  var CATS = [
    { key: "logos", label: "Logos", hint: "brand marks & wordmarks", color: "#ff3c3c" },
    { key: "faces", label: "Faces", hint: "people / headshots", color: "#00c8ff" },
    { key: "text", label: "Text / PII", hint: "names, addresses, IDs", color: "#ffd200" },
    { key: "signatures", label: "Signatures", hint: "handwritten sign-offs", color: "#c850ff" },
  ];
  var SRC_BORDER = { logo: "#ff3c3c", face: "#00c8ff", text: "#ffd200", signature: "#c850ff" };

  // Inline (no-CDN) stage icons — Material Design path data.
  var P = {
    crack: 'M15.5 12c2.5 0 4.5 2 4.5 4.5c0 .88-.25 1.71-.69 2.4l3.08 3.1L21 23.39l-3.12-3.07c-.69.43-1.51.68-2.38.68c-2.5 0-4.5-2-4.5-4.5s2-4.5 4.5-4.5m0 2a2.5 2.5 0 0 0-2.5 2.5a2.5 2.5 0 0 0 2.5 2.5a2.5 2.5 0 0 0 2.5-2.5a2.5 2.5 0 0 0-2.5-2.5M5 3h14c1.11 0 2 .89 2 2v8.03c-.5-.8-1.19-1.49-2-2.03V5H5v14h4.5c.31.75.76 1.42 1.31 2H5c-1.11 0-2-.89-2-2V5c0-1.11.89-2 2-2m2 4h10v2H7zm0 4h5.03c-.8.5-1.49 1.19-2.03 2H7zm0 4h2.17c-.11.5-.17 1-.17 1.5v.5H7z',
    text_id: 'm19.31 18.9l3.08 3.1L21 23.39l-3.12-3.07c-.69.43-1.51.68-2.38.68c-2.5 0-4.5-2-4.5-4.5s2-4.5 4.5-4.5s4.5 2 4.5 4.5c0 .88-.25 1.71-.69 2.4m-3.81.1a2.5 2.5 0 0 0 0-5a2.5 2.5 0 0 0 0 5M21 4v2H3V4zM3 16v-2h6v2zm0-5V9h18v2h-2.03c-1.01-.63-2.2-1-3.47-1s-2.46.37-3.47 1z',
    text_mask: 'M18.5 1.15c-.53 0-1.04.19-1.43.58l-5.81 5.82l5.65 5.65l5.82-5.81c.77-.78.77-2.04 0-2.83l-2.84-2.83c-.39-.39-.89-.58-1.39-.58M10.3 8.5l-5.96 5.96c-.78.78-.78 2.04.02 2.85C3.14 18.54 1.9 19.77.67 21h5.66l.86-.86c.78.76 2.03.75 2.81-.02l5.95-5.96',
    img_detect: 'M15.5 9c.7 0 1.29-.24 1.77-.73c.49-.48.73-1.07.73-1.77c0-.67-.24-1.27-.73-1.77c-.48-.5-1.07-.73-1.77-.73c-.67 0-1.27.23-1.77.73S13 5.83 13 6.5c0 .7.23 1.29.73 1.77c.5.49 1.1.73 1.77.73m3.81-.09l3.1 3.09L21 13.41l-3.14-3.1c-.78.47-1.58.69-2.39.69c-1.25 0-2.31-.42-3.17-1.3c-.85-.87-1.3-1.93-1.3-3.2c0-1.23.45-2.3 1.33-3.17C13.2 2.45 14.27 2 15.5 2c1.27 0 2.33.45 3.2 1.33c.88.87 1.3 1.94 1.3 3.17c0 .83-.22 1.63-.69 2.41M16.5 18h-11l2.75-3.5l1.97 2.33l2.72-3.52zm1.5-5l2 2v5c0 .55-.19 1-.59 1.4c-.41.39-.88.6-1.41.6H4c-.55 0-1-.21-1.4-.6c-.39-.4-.6-.85-.6-1.4V6c0-.53.21-1 .6-1.41C3 4.19 3.45 4 4 4h5.5c-.3.64-.47 1.31-.5 2H4v14h14z',
    boxes: 'M2 2h6v2h8V2h6v6h-2v8h2v6h-6v-2H8v2H2v-6h2V8H2zm14 6V6H8v2H6v8h2v2h8v-2h2V8zM4 4v2h2V4zm14 0v2h2V4zM4 18v2h2v-2zm14 0v2h2v-2z',
    img_mask: 'M21.8 16v-1.5c0-1.4-1.4-2.5-2.8-2.5s-2.8 1.1-2.8 2.5V16c-.6 0-1.2.6-1.2 1.2v3.5c0 .7.6 1.3 1.2 1.3h5.5c.7 0 1.3-.6 1.3-1.2v-3.5c0-.7-.6-1.3-1.2-1.3m-1.3 0h-3v-1.5c0-.8.7-1.3 1.5-1.3s1.5.5 1.5 1.3zM5 3c-1.1 0-2 .9-2 2v14a2 2 0 0 0 2 2h8.03c-.03-.1-.03-.2-.03-.3V19H5V5h14v5c.69 0 1.37.16 2 .42V5a2 2 0 0 0-2-2zm8.96 9.29l-2.75 3.54l-1.96-2.36L6.5 17H13c.08-.86.46-1.54.96-2.04c.07-.07.17-.11.24-.17v-.29c0-.55.1-1.06.27-1.53z',
    final: 'M.41 13.41L6 19l1.41-1.42L1.83 12m20.41-6.42L11.66 16.17L7.5 12l-1.43 1.41L11.66 19l12-12M18 7l-1.41-1.42l-6.35 6.35l1.42 1.41z',
  };
  function svg(id) { return '<svg viewBox="0 0 24 24"><path fill="currentColor" d="' + (P[id] || "") + '"/></svg>'; }

  var state = {
    file: null, fileName: "", isImage: false,
    opt: { logos: true, faces: true, text: true, signatures: true,
           face_style: "blur", other_style: "black", keep_databricks: true },
    phase: "idle", mock: false, err: "", dragOver: false, showOverlay: false,
    stages: [], img: {}, parse: null, textpii: null, detections: [], counts: null, timing: 0,
    model: { state: "cold" },
  };

  var MODEL_UI = {
    warm:    { cls: "ok",   txt: "Models ready" },
    warming: { cls: "warn", txt: "Warming models…" },
    cold:    { cls: "idle", txt: "Models idle" },
    error:   { cls: "err",  txt: "Models unavailable" },
  };

  var root = document.getElementById("root");
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ── streaming ────────────────────────────────────────────────────────────
  function handle(ev) {
    switch (ev.event) {
      case "meta": state.mock = !!ev.mock; break;
      case "stages": state.stages = ev.stages.map(function (s) {
        return { id: s.id, label: s.label, status: "pending" }; }); break;
      case "stage": state.stages.forEach(function (s) { if (s.id === ev.id) s.status = ev.status; }); break;
      case "render": state.img.original = ev.original; break;
      case "parse": state.parse = { n: ev.n, sample: ev.sample }; break;
      case "text_pii": state.textpii = { flagged: ev.flagged, screened: ev.screened, items: ev.items || [] }; break;
      case "vision": state.detections = (state.detections || []).concat(ev.detections); break;
      case "overlay": state.img.overlay = ev.overlay; state.detections = ev.detections; break;
      case "masked": state.img.masked = ev.masked; break;
      case "done": state.timing = ev.timing_s; state.counts = ev.counts;
        state.detections = ev.detections; state.phase = "done"; break;
      case "error": state.err = ev.msg; break;
    }
  }

  function run() {
    if (!state.file) { state.err = "Upload a document or image first."; render(); return; }
    state.phase = "running"; state.err = ""; state.showOverlay = false;
    state.stages = []; state.img = {}; state.parse = null; state.textpii = null;
    state.detections = []; state.counts = null; render();

    fetch("/api/mask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: state.file, filename: state.fileName, options: state.opt }),
    }).then(function (resp) {
      if (!resp.ok || !resp.body) throw new Error("stream failed (" + resp.status + ")");
      var reader = resp.body.getReader(), dec = new TextDecoder(), buf = "";
      (function pump() {
        return reader.read().then(function (res) {
          if (res.done) { if (state.phase === "running") state.phase = "done"; render(); return; }
          buf += dec.decode(res.value, { stream: true });
          var parts = buf.split("\n\n"); buf = parts.pop();
          parts.forEach(function (frame) {
            var line = frame.replace(/^data: /, "").trim();
            if (line) { try { handle(JSON.parse(line)); } catch (e) { /* skip */ } }
          });
          render();
          return pump();
        });
      })();
    }).catch(function (e) {
      state.err = String(e.message || e); state.phase = state.img.masked ? "done" : "idle"; render();
    });
  }

  // ── view pieces ──────────────────────────────────────────────────────────
  function figure(cap, src, cls) {
    return '<figure><figcaption' + (cls ? ' class="' + cls + '"' : "") + ">" + cap + "</figcaption>" +
      (src ? '<img src="' + src + '">' : '<div class="ph"><div class="spinner sm"></div></div>') + "</figure>";
  }

  function stageStrip() {
    if (!state.stages.length) return "";
    return '<div class="stages">' + state.stages.map(function (s) {
      return '<div class="stg ' + s.status + '"><span class="stg-ic">' + svg(s.id) +
        '<i class="stg-mark"></i></span><span class="stg-lb">' + esc(s.label) + "</span></div>";
    }).join("") + "</div>";
  }

  function docView() {
    var im = state.img;
    if (state.phase === "done" && im.masked) {
      var after = state.showOverlay ? (im.overlay || im.masked) : im.masked;
      return '<div class="compare">' + figure("BEFORE", im.original) +
        figure(state.showOverlay ? "DETECTED (boxes)" : "AFTER (masked)", after,
               state.showOverlay ? "ov" : "after") + "</div>";
    }
    var right = im.masked ? { cap: "MASKED", src: im.masked, cls: "after" }
      : im.overlay ? { cap: "DETECTED (boxes)", src: im.overlay, cls: "ov" }
      : { cap: "WORKING…", src: null };
    return '<div class="compare">' + figure("DOCUMENT", im.original) +
      figure(right.cap, right.src, right.cls) + "</div>";
  }

  function infoPanel() {
    var out = "";
    if (state.parse) {
      var flagged = (state.textpii && state.textpii.items) || [];
      var rows = state.parse.sample.map(function (s) {
        var pii = flagged.some(function (it) { return s.text.indexOf(it.text.slice(0, 24)) === 0; });
        return '<li' + (pii ? ' class="pii"' : "") + ">" + esc(s.text) + (pii ? " 🔒" : "") + "</li>";
      }).join("");
      var more = state.parse.n - state.parse.sample.length;
      out += '<div class="card"><h4>Parsed text' +
        (state.textpii ? " · " + state.textpii.flagged + " PII" : " · sample") + '</h4>' +
        '<ul class="parse">' + rows + "</ul>" +
        (more > 0 ? '<small>+' + more + " more elements</small>" : "") + "</div>";
    }
    if (state.detections && state.detections.length) {
      var chips = state.detections.map(function (d) {
        return '<span class="chip' + (d.mask ? "" : " kept") + '" style="border-color:' +
          (SRC_BORDER[d.source] || "#ccc") + '">' + (d.mask ? "" : "kept · ") +
          esc(d.source) + " · " + esc(d.label) + " " + d.score.toFixed(2) + "</span>";
      }).join("");
      out += '<div class="card"><h4>Detections</h4><div class="dets">' + chips + "</div></div>";
    }
    return out;
  }

  function render() {
    var o = state.opt;
    var cats = CATS.map(function (c) {
      return '<label class="cat' + (o[c.key] ? " on" : "") + '" data-cat="' + c.key + '">' +
        '<span class="dot" style="background:' + c.color + '"></span>' +
        '<span class="cat-label">' + c.label + "<small>" + c.hint + "</small></span></label>";
    }).join("");
    var seg = function (field, opts) {
      return opts.map(function (s) {
        return '<button data-seg="' + field + '" data-val="' + s + '"' +
          (o[field] === s ? ' class="on"' : "") + ">" + s + "</button>";
      }).join("");
    };

    var stage;
    if (state.phase === "idle" && !state.file) {
      stage = '<div class="empty"><h2>Upload a document to see the before / after</h2>' +
        "<p>Pick categories on the left, then <b>Redact document</b>.</p></div>";
    } else if (state.phase === "idle") {
      // file chosen but not run yet — show it immediately if it's an image
      stage = '<div class="result-head"><div class="stat">Ready — ' + esc(state.fileName) + "</div></div>" +
        '<div class="compare">' +
        (state.isImage ? figure("DOCUMENT", state.file)
          : '<figure><figcaption>DOCUMENT</figcaption><div class="ph">📄 ' + esc(state.fileName) +
            "<small>preview after redact</small></div></figure>") +
        '<figure><figcaption>AFTER</figcaption><div class="ph">press <b>Redact</b></div></figure></div>';
    } else {
      var head = state.phase === "done"
        ? '<div class="result-head"><div class="stat">' + (state.counts ? state.counts.masked + " masked" +
            (state.counts.kept ? " · " + state.counts.kept + " kept" : "") : "") + " · " + state.timing + "s</div>" +
            (state.img.overlay ? '<label class="check inline"><input type="checkbox" id="ovl"' +
              (state.showOverlay ? " checked" : "") + "> Show boxes</label>" : "") + "</div>"
        : '<div class="result-head"><div class="stat">Working… live preview below</div>' +
            '<div class="spinner sm"></div></div>';
      stage = head + stageStrip() + '<div class="work">' + docView() +
        '<aside class="info">' + infoPanel() + "</aside></div>";
    }

    var busy = state.phase === "running";
    var mu = MODEL_UI[state.mock ? "warm" : state.model.state] || MODEL_UI.cold;
    var visionNeeded = o.logos || o.faces;
    var notReady = visionNeeded && !state.mock && state.model.state !== "warm";
    var runDisabled = busy || !state.file || notReady;
    var runLabel = busy ? "Redacting…" : notReady ? "Waiting for models…" : "Redact document";
    root.innerHTML =
      '<div class="app"><header class="topbar">' +
        '<div class="brand"><span class="brick"></span> Redaction Studio' +
          '<span class="sub">Image Masking v5</span></div>' +
        '<div class="tbright"><span class="mstat ' + mu.cls + '" id="mstat">' +
            '<span class="mdot"></span>' + (state.mock ? "mock — ready" : mu.txt) + "</span>" +
          '<span class="' + (state.mock ? "mock-badge" : "live-badge") + '">' +
            (state.mock ? "mock backend — illustrative boxes" : "live · Databricks pipeline") +
          "</span></div></header>" +
      '<div class="layout"><aside class="panel">' +
        "<h3>1 · Upload</h3>" +
        '<div class="drop' + (state.dragOver ? " over" : "") + '" id="drop">' +
          '<input type="file" id="file" hidden accept=".pdf,.jpg,.jpeg,.png,.tif,.tiff,.doc,.docx,.ppt,.pptx,image/*,application/pdf">' +
          (state.fileName
            ? '<div class="file-ok">📄 ' + esc(state.fileName) + "<small>click to replace</small></div>"
            : '<div class="drop-hint">Drag &amp; drop or <u>browse</u>' +
              "<small>PDF · JPG · PNG · TIFF · DOC/DOCX · PPT/PPTX · max 100MB / 500 pages</small></div>") +
        "</div>" +
        '<h3>2 · What to mask</h3><div class="cats">' + cats + "</div>" +
        "<h3>3 · Options</h3>" +
        '<div class="opt-row"><span>Faces</span><div class="seg">' + seg("face_style", ["blur", "black"]) + "</div></div>" +
        '<div class="opt-row"><span>Logos / text / signatures</span><div class="seg">' + seg("other_style", ["black", "blur"]) + "</div></div>" +
        '<label class="check"><input type="checkbox" id="keepdb"' + (o.keep_databricks ? " checked" : "") +
          "> Keep Databricks' own logo (allowlist)</label>" +
        '<button class="run" id="run"' + (runDisabled ? " disabled" : "") + ">" + runLabel + "</button>" +
        (notReady ? '<div class="hint">GPU vision endpoint is spinning up — this can take ~2 min on first use. The button enables when it\'s ready.</div>' : "") +
        (state.err ? '<div class="err">' + esc(state.err) + "</div>" : "") +
      '</aside><main class="stage">' + stage + "</main></div></div>";

    wire();
  }

  function setFile(f) {
    if (!f) return;
    state.err = ""; state.phase = "idle"; state.fileName = f.name;
    state.isImage = /\.(png|jpe?g|tif?f|bmp|webp|gif)$/i.test(f.name) || /^image\//.test(f.type);
    state.img = {}; state.parse = null; state.textpii = null; state.detections = []; state.stages = [];
    var rd = new FileReader();
    rd.onload = function () { state.file = rd.result; render(); };
    rd.readAsDataURL(f);
  }

  function wire() {
    var drop = document.getElementById("drop");
    var file = document.getElementById("file");
    if (drop) {
      drop.onclick = function () { file.click(); };
      drop.ondragover = function (e) { e.preventDefault(); if (!state.dragOver) { state.dragOver = true; render(); } };
      drop.ondragleave = function () { if (state.dragOver) { state.dragOver = false; render(); } };
      drop.ondrop = function (e) { e.preventDefault(); state.dragOver = false; setFile(e.dataTransfer.files[0]); };
    }
    if (file) file.onchange = function (e) { setFile(e.target.files[0]); };
    Array.prototype.forEach.call(document.querySelectorAll("[data-cat]"), function (el) {
      el.onclick = function () { var k = el.getAttribute("data-cat"); state.opt[k] = !state.opt[k]; render(); };
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-seg]"), function (el) {
      el.onclick = function () { state.opt[el.getAttribute("data-seg")] = el.getAttribute("data-val"); render(); };
    });
    var keepdb = document.getElementById("keepdb");
    if (keepdb) keepdb.onchange = function (e) { state.opt.keep_databricks = e.target.checked; };
    var runBtn = document.getElementById("run");
    if (runBtn) runBtn.onclick = run;
    var ovl = document.getElementById("ovl");
    if (ovl) ovl.onchange = function (e) { state.showOverlay = e.target.checked; render(); };
  }

  // Poll the GPU vision endpoint's warm-state; trigger a warm-up while it's cold so
  // its ~150s cold start overlaps the user's upload/options, and re-render the
  // status dot + Redact gate on each transition.
  function pollStatus() {
    fetch("/api/status").then(function (r) { return r.json(); }).then(function (d) {
      var s = d.state || "cold", mock = d.endpoint === "mock";
      if (s !== state.model.state || mock !== state.mock) {
        state.model.state = s; state.mock = mock; render();
      }
      if (!mock && (s === "cold" || s === "error")) fetch("/api/warm", { method: "POST" }).catch(function () {});
    }).catch(function () {});
  }

  render();
  pollStatus();
  setInterval(pollStatus, 3000);
})();
