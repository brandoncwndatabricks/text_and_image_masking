"""FastAPI app for Databricks Apps — the deployed redaction service.

Light container (NO torch/transformers/opencv): the GPU vision models run on a
Model Serving endpoint; the text/PII phase uses Databricks services directly:
  - vision (logos+faces) : Model Serving endpoint (VISION_ENDPOINT)
  - text extraction      : ai_parse_document via SQL Statement Execution
                           (SQL_WAREHOUSE_ID) — unbase64(<image>) → elements
  - PII classification    : Claude serving endpoint (CLAUDE_ENDPOINT)
  - masking / overlay     : Pillow (pipeline.masking / overlay) — torch-free
Auth is the app's injected service principal (WorkspaceClient()); locally pass
DATABRICKS_PROFILE. Serves the web/ SPA.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# pipeline importable both locally (src_v5/pipeline, the parent) and when bundled
# into the app dir for deploy (app/pipeline).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from pipeline.boxes import Detection, merge_sources            # noqa: E402
from pipeline.masking import apply_masks                       # noqa: E402
from pipeline.overlay import render_overlay                    # noqa: E402
from pipeline import text_pii                                  # noqa: E402 (regex/prompt/flag_signatures — pure)

VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "image-masking-vision")
SQL_WAREHOUSE_ID = os.environ.get("SQL_WAREHOUSE_ID", "")
CLAUDE_ENDPOINT = os.environ.get("CLAUDE_ENDPOINT", "databricks-claude-sonnet-4")
AI_PARSE_MAX_DIM = 1800   # cap image sent to ai_parse (SQL literal size); boxes scaled back

app = FastAPI(title="Redaction Studio — Image Masking v5")

_ws = None
def ws():
    global _ws
    if _ws is None:
        from databricks.sdk import WorkspaceClient
        prof = os.environ.get("DATABRICKS_PROFILE")
        _ws = WorkspaceClient(profile=prof) if prof else WorkspaceClient()
    return _ws


class MaskOptions(BaseModel):
    logos: bool = True; faces: bool = True; text: bool = True; signatures: bool = True
    face_style: str = "blur"; other_style: str = "black"; keep_databricks: bool = True


class MaskRequest(BaseModel):
    image: str
    filename: str = ""
    options: MaskOptions = MaskOptions()


# ── input ─────────────────────────────────────────────────────────────────────
def load_image(data_url: str, filename: str) -> Image.Image:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf" or data_url.startswith("data:application/pdf"):
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(raw)
        pil = pdf[0].render(scale=2.0).to_pil().convert("RGB")
        pdf.close()
        return pil
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ── vision via Model Serving ──────────────────────────────────────────────────
def vision_detect(img: Image.Image, opt: MaskOptions):
    buf = io.BytesIO(); img.save(buf, "PNG")
    rec = {"image_b64": base64.b64encode(buf.getvalue()).decode(),
           "options": json.dumps({"logos": opt.logos, "faces": opt.faces,
                                  "keep_databricks": opt.keep_databricks})}
    resp = ws().serving_endpoints.query(name=VISION_ENDPOINT, dataframe_records=[rec])
    data = json.loads(resp.predictions[0])
    return [Detection(box=d["box"], source=d["source"], label=d["label"],
                      score=d["score"], mask=d["mask"]) for d in data["detections"]]


# ── text via ai_parse_document (SQL) ──────────────────────────────────────────
def text_detect(img: Image.Image):
    w0, h0 = img.size
    send = img.copy(); send.thumbnail((AI_PARSE_MAX_DIM, AI_PARSE_MAX_DIM))
    sx, sy = w0 / send.width, h0 / send.height          # scale boxes back to original
    buf = io.BytesIO(); send.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    sql = f"SELECT CAST(ai_parse_document(unbase64('{b64}')) AS STRING) AS parsed"
    r = ws().statement_execution.execute_statement(
        warehouse_id=SQL_WAREHOUSE_ID, statement=sql, wait_timeout="50s")
    if not (r.result and r.result.data_array):
        return []
    parsed = json.loads(r.result.data_array[0][0])
    from pipeline.boxes import clip_to_image, area_frac, pad_box
    dets = []
    for elem in parsed.get("document", {}).get("elements", []):
        etype = (elem.get("type", "") or "").lower()
        content = (elem.get("content", "") or "").strip()
        is_fig = etype in text_pii.FIGURE_TYPES
        if not content and not is_fig:
            continue
        for b in elem.get("bbox", []):
            c = b.get("coord", [])
            if len(c) != 4:
                continue
            box = [c[0] * sx, c[1] * sy, c[2] * sx, c[3] * sy]
            box = clip_to_image(box, w0, h0)
            if area_frac(box, w0, h0) > 0.30:
                continue
            box = pad_box(box, 0.08, w0, h0)
            dets.append(Detection(box=box, source="text", label=content[:40], score=1.0,
                                  meta={"content": content, "elem_type": etype}))
    return dets


def claude_pii_indices(text_dets):
    texts = "\n".join(f'{i}: {d.meta.get("content", d.label)}' for i, d in enumerate(text_dets))
    prompt = (f"{text_pii.SENSITIVE_TEXT_PROMPT}\n\nText items to review:\n{texts}\n\n"
              "Return a JSON array of index numbers (0-based) for items that should be masked.\n"
              "Return ONLY the JSON array. Example: [0, 2, 5]")
    try:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        resp = ws().serving_endpoints.query(
            name=CLAUDE_ENDPOINT,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)])
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return set(int(i) for i in json.loads(raw) if isinstance(i, int))
    except Exception:
        return None  # fall back to regex-only


@app.post("/api/mask")
def mask(req: MaskRequest):
    import time
    t0 = time.time()
    try:
        img = load_image(req.image, req.filename)
    except Exception as e:
        return JSONResponse({"error": f"could not read input: {e}"}, status_code=400)
    opt = req.options
    dets = []
    if opt.logos or opt.faces:
        dets += vision_detect(img, opt)
    want_text, want_sig = opt.text, opt.signatures
    if want_text or want_sig:
        tds = text_detect(img)
        text_pii.flag_signatures(tds, img_h=img.size[1])
        # regex always; Claude refines
        claude_idx = claude_pii_indices([d for d in tds if d.source == "text"]) if want_text else None
        text_only = [d for d in tds if d.source == "text"]
        for i, d in enumerate(text_only):
            content = d.meta.get("content", d.label)
            d.mask = bool(want_text and (text_pii.regex_is_sensitive(content)
                                         or (claude_idx is not None and i in claude_idx)))
        dets += tds
    # respect toggles
    keep = set()
    if opt.logos: keep.add("logo")
    if opt.faces: keep.add("face")
    if want_text: keep.add("text")
    if want_sig: keep.add("signature")
    dets = [d for d in merge_sources(dets) if d.source in keep and d.mask]
    styles = {"face": opt.face_style, "logo": opt.other_style,
              "text": opt.other_style, "signature": opt.other_style}

    def url(im):
        b = io.BytesIO(); im.convert("RGB").save(b, "PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    return {"original": url(img), "masked": url(apply_masks(img, dets, style_overrides=styles)),
            "overlay": url(render_overlay(img, dets)),
            "detections": [{"source": d.source, "label": d.label, "score": round(float(d.score), 3),
                            "box": [round(float(v), 1) for v in d.box], "mask": bool(d.mask)} for d in dets],
            "timing_s": round(time.time() - t0, 2), "mock": False}


# ── serve the SPA ─────────────────────────────────────────────────────────────
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.isdir(WEB):
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        target = os.path.join(WEB, full_path)
        if full_path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(WEB, "index.html"))
