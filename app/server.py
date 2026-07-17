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
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFilter

# Make the v5 `pipeline` package importable (src_v5/ is this file's grandparent),
# so REAL mode can `from pipeline.run import MaskingPipeline`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PORT = int(os.environ.get("PORT", "8000"))

SRC_COLOR = {"logo": (255, 60, 60), "face": (0, 200, 255),
             "text": (255, 210, 0), "signature": (200, 80, 255)}
KEPT_COLOR = (60, 220, 60)


# ── Mock detection ──────────────────────────────────────────────────────────
def mock_detections(w, h, opt):
    """Produce plausible, deterministic-ish boxes per enabled category.
    Stands in for the real pipeline so the UI can be exercised."""
    rnd = random.Random(w * 7919 + h)  # stable per image size
    dets = []
    if opt.get("logos"):
        dets.append({"source": "logo", "label": "brand logo", "score": 0.74,
                     "box": [int(w * 0.06), int(h * 0.05), int(w * 0.30), int(h * 0.13)],
                     "mask": True})
        if opt.get("keep_databricks"):
            dets.append({"source": "logo", "label": "databricks(kept)", "score": 0.91,
                         "box": [int(w * 0.70), int(h * 0.88), int(w * 0.94), int(h * 0.96)],
                         "mask": False})
    if opt.get("faces"):
        cx = w * 0.5
        dets.append({"source": "face", "label": "face", "score": 0.88,
                     "box": [int(cx - w * 0.07), int(h * 0.30), int(cx + w * 0.07), int(h * 0.46)],
                     "mask": True})
    if opt.get("text"):
        for i in range(3):
            y = h * (0.20 + i * 0.10)
            dets.append({"source": "text", "label": "PII text", "score": 0.99,
                         "box": [int(w * 0.10), int(y), int(w * (0.45 + rnd.random() * 0.25)), int(y + h * 0.045)],
                         "mask": True})
    if opt.get("signatures"):
        dets.append({"source": "signature", "label": "signature", "score": 0.80,
                     "box": [int(w * 0.10), int(h * 0.82), int(w * 0.42), int(h * 0.88)],
                     "mask": True})
    return dets


def apply_masks(img, dets, opt):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for d in dets:
        if not d["mask"]:
            continue
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        style = opt.get("face_style", "blur") if d["source"] == "face" else opt.get("other_style", "black")
        if style == "blur" and x2 > x1 and y2 > y1:
            region = out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(max(8, (x2 - x1) // 4)))
            out.paste(region, (x1, y1))
        else:
            draw.rectangle([x1, y1, x2, y2], fill="black")
    return out


def render_overlay(img, dets):
    base = img.convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for det in dets:
        c = KEPT_COLOR if not det["mask"] else SRC_COLOR.get(det["source"], (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        d.rectangle([x1, y1, x2, y2], fill=c + (45,), outline=c + (255,), width=3)
        tag = ("KEPT " if not det["mask"] else "") + f'{det["source"]}:{det["label"][:16]} {det["score"]:.2f}'
        d.rectangle([x1, max(0, y1 - 18), x1 + 8 * len(tag), y1], fill=c + (230,))
        d.text((x1 + 3, max(0, y1 - 17)), tag, fill=(0, 0, 0, 255))
    return Image.alpha_composite(base, ov).convert("RGB")


def to_data_url(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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
    if TEXT_CAPABLE:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.profile(
            os.environ.get("DATABRICKS_PROFILE", "e2-field-eng-west")).serverless(True).getOrCreate()
        endpoint = ("https://e2-demo-field-eng.cloud.databricks.com"
                    "/serving-endpoints/databricks-claude-sonnet-4/invocations")
    cfg = PipelineConfig(do_logos=True, do_faces=True, do_text=TEXT_CAPABLE,
                         use_clip_gate=True, text_mode="pii_only")
    _PIPE = MaskingPipeline(cfg, spark=spark, claude_endpoint=endpoint)
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

    # mock path (default): illustrative placeholder boxes
    dets = mock_detections(img.width, img.height, opt)
    masked = apply_masks(img, dets, opt)
    overlay = render_overlay(img, dets)
    time.sleep(0.6)  # simulate latency so the progress UI is visible
    return {"original": to_data_url(img), "masked": to_data_url(masked),
            "overlay": to_data_url(overlay), "detections": dets,
            "timing_s": round(time.time() - t0, 2), "mock": True}


# ── HTTP server (static + /api) ───────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") != "/api/mask":
            return self._send(404, b'{"error":"not found"}')
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            result = process(payload)
            self._send(200, json.dumps(result).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

    def do_GET(self):
        path = self.path.split("?", 1)[0]
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
