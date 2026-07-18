"""Local mock backend for the redaction app — ZERO external dependencies.

Why stdlib instead of FastAPI: this dev environment has no PyPI/npm access, so
we can't `uv add fastapi` or `npm install`. This server uses only the Python
standard library + Pillow (already installed) and serves both the static React
frontend (web/) and a mock `/api/mask` JSON endpoint.

The API contract here is exactly what the production FastAPI endpoint will
expose, so the React frontend is unchanged when we port:

    POST /api/mask
      body  (JSON): { image: "data:image/...;base64,...",
                      options: { logos, faces, text, signatures,   # bools
                                 face_style: "blur"|"black",
                                 other_style: "blur"|"black",
                                 keep_databricks: bool } }
      reply (JSON): { original, masked, overlay,   # data URLs
                      detections: [ {source,label,score,box:[x1,y1,x2,y2],mask} ],
                      timing_s }

DEPLOYMENT NOTE: in production the masking is done by the real MaskingPipeline
(src_v5/pipeline/) running behind a Databricks **Model Serving endpoint** (the
app container on Databricks Apps has no GPU). The app calls it **async
(submit-then-poll)** because the Apps ingress has a ~120s timeout. Here we just
draw illustrative boxes with Pillow so the UX is fully clickable.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

# Make `pipeline` (grandparent src_v5/) and `_stream` (sibling) importable, then
# pull in the SHARED staged orchestration so this dev server and the deployed
# FastAPI app run the exact same pipeline order / PII logic / masking.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # src_v5 → pipeline
sys.path.insert(0, _HERE)                     # app    → _stream
from _stream import STAGES, run_stages, sse, aggregate, to_data_url  # noqa: E402
from pipeline.boxes import Detection                                  # noqa: E402

WEB = os.path.join(_HERE, "web")
PORT = int(os.environ.get("PORT", "8000"))


# ── Mock detectors (Detection objects → feed the real shared orchestrator) ────
# Plausible, deterministic boxes so the staged UX is fully exercisable with no
# GPU / Databricks. Includes a KEPT logo + a non-PII text line to show that
# masking is *selective*, not mask-everything.
def mock_vision(img, opt):
    w, h = img.size; out = []
    if opt.get("logos"):
        out.append(Detection(box=[w*.06, h*.05, w*.30, h*.13], source="logo", label="brand logo", score=0.74))
        if opt.get("keep_databricks"):
            out.append(Detection(box=[w*.70, h*.88, w*.94, h*.96], source="logo",
                                 label="databricks (kept)", score=0.91, mask=False))
    if opt.get("faces"):
        cx = w * .5
        out.append(Detection(box=[cx-w*.07, h*.30, cx+w*.07, h*.46], source="face", label="face", score=0.88))
    return out


def mock_text(img):
    w, h = img.size; out = []
    lines = ["John Smith — 123 Main St", "Acct #4471-9982", "DOB 04/12/1981", "Value to Databricks"]
    for i, txt in enumerate(lines):
        y = h * (0.20 + i * 0.10)
        out.append(Detection(box=[w*.10, y, w*.46, y + h*.045], source="text", label=txt[:40],
                             score=0.99, meta={"content": txt, "elem_type": "text"}))
    return out


def mock_pii(contents):
    # everything that isn't the generic "Value to Databricks" line is PII
    return set(i for i, c in enumerate(contents) if "Databricks" not in c)


# Formats ai_parse_document accepts: PDF, JPG/JPEG, PNG, TIFF/TIF, DOC/DOCX, PPT/PPTX
# (max 100 MB / 500 pages). Images load directly; PDF is rasterized (page 1) for
# the visual detectors; DOC/PPT need server-side rendering (LibreOffice), which
# isn't available in this local dev app — on the deployed app ai_parse_document
# handles all of them and a rasterizer feeds the image detectors.
IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"}
OFFICE_EXTS = {"doc", "docx", "ppt", "pptx"}

def load_input_image(payload):
    """Decode the uploaded file (data URL) → a single RGB PIL image to detect on."""
    header, b64 = payload["image"].split(",", 1)
    raw = base64.b64decode(b64)
    name = (payload.get("filename") or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    mime = header.split(";")[0].replace("data:", "")

    if mime.startswith("image/") or ext in IMAGE_EXTS:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if ext == "pdf" or mime == "application/pdf":
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(raw); pdf = f.name
        png = pdf[:-4] + ".png"
        subprocess.run(["sips", "-s", "format", "png", pdf, "--out", png],
                       capture_output=True, check=True)  # page 1
        return Image.open(png).convert("RGB")
    if ext in OFFICE_EXTS:
        raise ValueError(
            "DOC/PPT are parsed by ai_parse_document server-side; local visual "
            "detection needs rasterization (LibreOffice) not available in this dev "
            "app. Locally, use PDF or an image; the deployed app handles all types.")
    raise ValueError("Unsupported file. Accepts: PDF, JPG, PNG, TIFF, DOC/DOCX, PPT/PPTX.")


# ── Real pipeline mode (REDACT_REAL=1) ────────────────────────────────────────
# When enabled, the actual MaskingPipeline (Grounding DINO + CLIP + YuNet, and
# the ai_parse_document text phase if REDACT_TEXT=1) runs locally so boxes are
# REAL and aligned — not the mock placeholders. Heavy models load once at start.
REAL = os.environ.get("REDACT_REAL") == "1"
TEXT_CAPABLE = os.environ.get("REDACT_TEXT") == "1"  # text phase needs Databricks
_PIPE = None  # single cached MaskingPipeline (models load once, reused per request)

def _get_pipeline():
    """Build/cache ONE MaskingPipeline so heavy models load only once."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from pipeline.run import MaskingPipeline, PipelineConfig
    spark = endpoint = None
    profile = os.environ.get("DATABRICKS_PROFILE", "e2-demo-west")
    if TEXT_CAPABLE:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()
        # Endpoint host + model are env-driven so the app isn't pinned to a dead
        # workspace. Default to the same workspace as DATABRICKS_PROFILE.
        host = os.environ.get("DATABRICKS_HOST", "https://e2-demo-west.cloud.databricks.com").rstrip("/")
        model = os.environ.get("CLAUDE_ENDPOINT", "databricks-claude-sonnet-5")
        endpoint = f"{host}/serving-endpoints/{model}/invocations"
    cfg = PipelineConfig(do_logos=True, do_faces=True, do_text=TEXT_CAPABLE,
                         use_clip_gate=True,
                         text_mode=os.environ.get("TEXT_MODE", "pii_only"))
    _PIPE = MaskingPipeline(cfg, spark=spark, claude_endpoint=endpoint, profile=profile)
    return _PIPE


# ── Backends: where do the VISION detectors run? ──────────────────────────────
#   VISION_BACKEND=local   → Grounding DINO + CLIP + YuNet in-process (default)
#   VISION_BACKEND=serving → call the Databricks GPU Model Serving endpoint
# The text/PII phase always runs in-app (ai_parse_document + Claude) since serving
# endpoints have no Spark.
VISION_BACKEND = os.environ.get("VISION_BACKEND", "local")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "image-masking-vision")

def _serving_vision(img, opt):
    """Get logo+face detections from the GPU Model Serving endpoint."""
    import json as J
    from databricks.sdk import WorkspaceClient
    from pipeline.boxes import Detection
    w = WorkspaceClient(profile=os.environ.get("DATABRICKS_PROFILE", "fevm-lsryzn"))
    buf = io.BytesIO(); img.save(buf, "PNG")
    rec = {"image_b64": base64.b64encode(buf.getvalue()).decode(),
           "options": J.dumps({"logos": bool(opt.get("logos", True)),
                                "faces": bool(opt.get("faces", True)),
                                "keep_databricks": bool(opt.get("keep_databricks", True))})}
    resp = w.serving_endpoints.query(name=VISION_ENDPOINT, dataframe_records=[rec])
    data = J.loads(resp.predictions[0])
    return [Detection(box=d["box"], source=d["source"], label=d["label"],
                      score=d["score"], mask=d["mask"]) for d in data["detections"]]


def _local_text(img, opt):
    """Run only the text/PII phase locally (ai_parse_document + Claude)."""
    pipe = _get_pipeline()
    pipe.cfg.do_logos = False; pipe.cfg.do_faces = False; pipe.cfg.do_text = True
    import tempfile
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tf.name)
    return pipe.detect(img, tf.name)


def process(payload):
    img = load_input_image(payload)
    opt = payload.get("options", {})
    t0 = time.time()

    if REAL:
        from pipeline.overlay import render_overlay as real_overlay
        from pipeline.masking import apply_masks as real_apply
        want_text = bool(opt.get("text")); want_sig = bool(opt.get("signatures"))
        want_vision = bool(opt.get("logos")) or bool(opt.get("faces"))
        do_text = (want_text or want_sig) and TEXT_CAPABLE

        dets_obj = []
        if VISION_BACKEND == "serving":
            # Vision via the GPU Model Serving endpoint; text in-app.
            if want_vision:
                dets_obj += _serving_vision(img, opt)
            if do_text:
                dets_obj += _local_text(img, opt)
        else:
            # All local: one cached MaskingPipeline, toggles per request.
            pipe = _get_pipeline()
            pipe.cfg.do_logos = bool(opt.get("logos"))
            pipe.cfg.do_faces = bool(opt.get("faces"))
            pipe.cfg.do_text = do_text
            image_path = None
            if do_text:
                import tempfile
                tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tf.name); image_path = tf.name
            dets_obj = pipe.detect(img, image_path)

        # Respect the per-category toggles
        keep_src = set()
        if opt.get("logos"): keep_src.add("logo")
        if opt.get("faces"): keep_src.add("face")
        if want_text: keep_src.add("text")
        if want_sig: keep_src.add("signature")
        dets_obj = [d for d in dets_obj if d.source in keep_src]
        styles = {"face": opt.get("face_style", "blur"), "logo": opt.get("other_style", "black"),
                  "text": opt.get("other_style", "black"), "signature": opt.get("other_style", "black")}
        masked = real_apply(img, dets_obj, style_overrides=styles)
        overlay = real_overlay(img, dets_obj)
        dets = [{"source": d.source, "label": d.label, "score": round(float(d.score), 3),
                 "box": [round(float(v), 1) for v in d.box], "mask": bool(d.mask)} for d in dets_obj]
        return {"original": to_data_url(img), "masked": to_data_url(masked),
                "overlay": to_data_url(overlay), "detections": dets,
                "timing_s": round(time.time() - t0, 2), "mock": False}

    # mock path (default): same shared orchestrator, mock detectors
    r = aggregate(run_stages(img, {**opt, "_filename": payload.get("filename", "")},
                             vision_fn=mock_vision, text_fn=mock_text, pii_fn=mock_pii))
    r["mock"] = True
    return r


# ── Streaming (SSE) — the staged UX ───────────────────────────────────────────
def _replay_stages(result):
    """REAL mode computes everything up front; reveal it through the same staged
    events so the panel still animates (boxes/masked appear at the end)."""
    yield {"event": "stages", "stages": STAGES}
    yield {"event": "render", "original": result["original"]}
    for sid in ("crack", "text_id", "img_detect"):
        yield {"event": "stage", "id": sid, "status": "done"}
    yield {"event": "overlay", "overlay": result["overlay"], "detections": result["detections"]}
    yield {"event": "stage", "id": "boxes", "status": "done"}
    yield {"event": "masked", "masked": result["masked"]}
    for sid in ("text_mask", "img_mask", "final"):
        yield {"event": "stage", "id": sid, "status": "done"}
    dets = result["detections"]
    yield {"event": "done", "timing_s": result["timing_s"], "detections": dets,
           "counts": {"masked": sum(d["mask"] for d in dets),
                      "kept": sum(not d["mask"] for d in dets)}}


def stream_events(payload):
    """Generator of staged events. Mock = live stages; REAL = compute then replay."""
    yield {"event": "meta", "mock": not REAL}
    try:
        img = load_input_image(payload)
    except Exception as e:  # noqa: BLE001
        yield {"event": "error", "msg": str(e)}; return
    opt = {**payload.get("options", {}), "_filename": payload.get("filename", "")}
    if REAL:
        yield from _replay_stages(process(payload))
    else:
        yield from run_stages(img, opt, vision_fn=mock_vision, text_fn=mock_text,
                              pii_fn=mock_pii, step_delay=0.45)


# ── HTTP server (static + /api) ───────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/api/warm":
            return self._send(200, b'{"state":"warm"}')   # mock: no real endpoint
        if path == "/api/mask/stream":
            return self._stream()
        if path != "/api/mask":
            return self._send(404, b'{"error":"not found"}')
        try:
            self._send(200, json.dumps(process(self._read_payload())).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

    def _stream(self):
        """Server-Sent Events: write one frame per staged event, flushing each."""
        try:
            payload = self._read_payload()
        except Exception as e:
            return self._send(400, json.dumps({"error": str(e)}).encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for e in stream_events(payload):
                self.wfile.write(sse(e).encode()); self.wfile.flush()
        except Exception as e:  # noqa: BLE001
            try:
                self.wfile.write(sse({"event": "error", "msg": str(e)}).encode()); self.wfile.flush()
            except Exception:
                pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            return self._send(200, b'{"state":"warm","endpoint":"mock"}')
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        fp = os.path.normpath(os.path.join(WEB, rel))
        if not fp.startswith(WEB) or not os.path.isfile(fp):
            fp = os.path.join(WEB, "index.html")  # SPA fallback
        ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css"}.get(
            fp.rsplit(".", 1)[-1], "application/octet-stream")
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    def log_message(self, *a):  # quieter logs
        pass


if __name__ == "__main__":
    print(f"Redaction app (mock backend) → http://localhost:{PORT}")
    print("  Ctrl-C to stop.  This is a MOCK — boxes are illustrative.")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
