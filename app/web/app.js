/* Redaction Studio — self-contained vanilla JS (no framework, no CDN, no build).
   Renders the full UX off localhost only. Same /api/mask contract the React
   production app will use. */
(function () {
  "use strict";

  var CATS = [
    { key: "logos", label: "Logos", hint: "brand marks & wordmarks", color: "#ff3c3c" },
    { key: "faces", label: "Faces", hint: "people / headshots", color: "#00c8ff" },
    { key: "text", label: "Text / PII", hint: "names, addresses, IDs", color: "#ffd200" },
    { key: "signatures", label: "Signatures", hint: "handwritten sign-offs", color: "#c850ff" },
  ];
  var SRC_BORDER = { logo: "#ff3c3c", face: "#00c8ff", text: "#ffd200", signature: "#c850ff" };

  var state = {
    file: null, fileName: "",
    opt: { logos: true, faces: true, text: true, signatures: true,
           face_style: "blur", other_style: "black", keep_databricks: true },
    result: null, busy: false, err: "", showOverlay: false, dragOver: false,
  };

  var root = document.getElementById("root");
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function render() {
    var o = state.opt;
    var cats = CATS.map(function (c) {
      return '<label class="cat' + (o[c.key] ? " on" : "") + '" data-cat="' + c.key + '">' +
        '<span class="dot" style="background:' + c.color + '"></span>' +
        '<span class="cat-label">' + c.label + '<small>' + c.hint + '</small></span></label>';
    }).join("");

    var seg = function (field, opts) {
      return opts.map(function (s) {
        return '<button data-seg="' + field + '" data-val="' + s + '"' +
          (o[field] === s ? ' class="on"' : "") + ">" + s + "</button>";
      }).join("");
    };

    var stage;
    if (state.busy) {
      stage = '<div class="progress"><div class="spinner"></div>' +
        "<p>Running detectors… (real pipeline ~15–40s via Model Serving)</p></div>";
    } else if (!state.result) {
      stage = '<div class="empty"><h2>Upload a document to see the before / after</h2>' +
        "<p>Select categories on the left, then <b>Redact document</b>.</p></div>";
    } else {
      var r = state.result;
      var maskN = r.detections.filter(function (d) { return d.mask; }).length;
      var keptN = r.detections.length - maskN;
      var after = state.showOverlay ? r.overlay : r.masked;
      var chips = r.detections.map(function (d) {
        return '<span class="chip' + (d.mask ? "" : " kept") + '" style="border-color:' +
          (SRC_BORDER[d.source] || "#ccc") + '">' + (d.mask ? "" : "kept · ") +
          esc(d.source) + " · " + esc(d.label) + " " + d.score.toFixed(2) + "</span>";
      }).join("");
      stage =
        '<div class="results"><div class="result-head">' +
          '<div class="stat">' + maskN + " masked" + (keptN ? " · " + keptN + " kept" : "") +
            " · " + r.timing_s + "s</div>" +
          '<label class="check inline"><input type="checkbox" id="ovl"' +
            (state.showOverlay ? " checked" : "") + "> Show debug overlay</label></div>" +
        '<div class="compare">' +
          '<figure><figcaption>BEFORE</figcaption><img src="' + r.original + '"></figure>' +
          '<figure><figcaption class="' + (state.showOverlay ? "ov" : "after") + '">' +
            (state.showOverlay ? "DEBUG OVERLAY" : "AFTER (masked)") + "</figcaption>" +
            '<img src="' + after + '"></figure></div>' +
        '<div class="dets">' + chips + "</div></div>";
    }

    root.innerHTML =
      '<div class="app"><header class="topbar">' +
        '<div class="brand"><span class="brick"></span> Redaction Studio' +
          '<span class="sub">Image Masking v5</span></div>' +
        '<span class="mock-badge">mock backend — illustrative boxes</span></header>' +
      '<div class="layout"><aside class="panel">' +
        "<h3>1 · Upload</h3>" +
        '<div class="drop' + (state.dragOver ? " over" : "") + '" id="drop">' +
          '<input type="file" id="file" hidden accept=".pdf,.jpg,.jpeg,.png,.tif,.tiff,.doc,.docx,.ppt,.pptx,image/*,application/pdf">' +
          (state.fileName
            ? '<div class="file-ok">📄 ' + esc(state.fileName) + "<small>click to replace</small></div>"
            : '<div class="drop-hint">Drag &amp; drop or <u>browse</u>' +
              "<small>PDF · JPG · PNG · TIFF · DOC/DOCX · PPT/PPTX · max 100MB / 500 pages</small></div>") +
        "</div>" +
        "<h3>2 · What to mask</h3><div class=\"cats\">" + cats + "</div>" +
        "<h3>3 · Options</h3>" +
        '<div class="opt-row"><span>Faces</span><div class="seg">' + seg("face_style", ["blur", "black"]) + "</div></div>" +
        '<div class="opt-row"><span>Logos / text / signatures</span><div class="seg">' + seg("other_style", ["black", "blur"]) + "</div></div>" +
        '<label class="check"><input type="checkbox" id="keepdb"' + (o.keep_databricks ? " checked" : "") +
          "> Keep Databricks' own logo (allowlist)</label>" +
        '<button class="run" id="run"' + (state.busy || !state.file ? " disabled" : "") + ">" +
          (state.busy ? "Redacting…" : "Redact document") + "</button>" +
        (state.err ? '<div class="err">' + esc(state.err) + "</div>" : "") +
      '</aside><main class="stage">' + stage + "</main></div></div>";

    wire();
  }

  function setFile(f) {
    if (!f) return;
    state.err = ""; state.result = null; state.fileName = f.name;
    var rd = new FileReader();
    rd.onload = function () { state.file = rd.result; render(); };
    rd.readAsDataURL(f);
  }

  function run() {
    if (!state.file) { state.err = "Upload a document or image first."; render(); return; }
    state.busy = true; state.err = ""; state.result = null; render();
    fetch("/api/mask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: state.file, filename: state.fileName, options: state.opt }),
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) throw new Error(d.error);
      state.result = d; state.showOverlay = false;
    }).catch(function (e) { state.err = String(e.message || e); })
      .finally(function () { state.busy = false; render(); });
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

  render();
})();
