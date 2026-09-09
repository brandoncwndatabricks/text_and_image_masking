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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

# pipeline importable both locally (src_v5/pipeline, the parent) and when bundled
# into the app dir for deploy (app/pipeline).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from pipeline.boxes import Detection                           # noqa: E402
from pipeline import text_pii                                  # noqa: E402 (regex/prompt/flag_signatures — pure)
from _stream import run_stages, sse, aggregate                 # noqa: E402 (shared staged orchestration)

# Config resolution: env var → config.yaml (repo root or app/) → default.
# On Databricks Apps, app.yaml `env:` sets the env vars (they win). config.yaml
# is for local runs and is git-ignored. See config.example.yaml.
def _load_config_file():
    for d in (os.path.dirname(_HERE), _HERE):
        p = os.path.join(d, "config.yaml")
        if os.path.isfile(p):
            try:
                import yaml
                with open(p) as f:
                    return yaml.safe_load(f) or {}
            except Exception:      # yaml missing or unparsable → env/defaults only
                return {}
    return {}

_CFG = _load_config_file()
def cfg(name, default=""):
    return os.environ.get(name) or _CFG.get(name) or default

VISION_ENDPOINT = cfg("VISION_ENDPOINT", "image-masking-vision")
SQL_WAREHOUSE_ID = cfg("SQL_WAREHOUSE_ID", "")
CLAUDE_ENDPOINT = cfg("CLAUDE_ENDPOINT", "databricks-claude-sonnet-5")
TEXT_MODE = cfg("TEXT_MODE", "pii_only")
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


# ── SQL warehouse resolution (configured id, with runtime fallback) ────────────
# The configured SQL_WAREHOUSE_ID can go stale — a warehouse recreated in the
# workspace gets a new id, which silently breaks ai_parse_document. So we resolve
# lazily: use the configured id when it still exists, otherwise scan the workspace
# and pick a usable warehouse (prefer one already RUNNING, then serverless — cheap
# and auto-starts). A warehouse that fails at query time is blacklisted and we
# re-pick. The resolved id is cached across requests.
_WH = {"id": None, "bad": set()}

def _pick_warehouse(exclude=()):
    """Best available warehouse id: prefer RUNNING, then serverless, then any."""
    try:
        cands = [w for w in ws().warehouses.list() if w.id and w.id not in exclude]
    except Exception:
        return None
    if not cands:
        return None
    def rank(w):
        running = str(getattr(w, "state", "")).upper().endswith("RUNNING")
        serverless = bool(getattr(w, "enable_serverless_compute", False))
        return (running, serverless)
    cands.sort(key=rank, reverse=True)
    return cands[0].id

def resolve_warehouse_id(force=False):
    """Return a usable SQL warehouse id, with caching + runtime fallback."""
    if not force and _WH["id"]:
        return _WH["id"]
    wid = SQL_WAREHOUSE_ID if (SQL_WAREHOUSE_ID and SQL_WAREHOUSE_ID not in _WH["bad"]) else ""
    if wid:
        try:
            ws().warehouses.get(wid)          # configured id still exists?
        except Exception:
            wid = ""                           # gone → fall back to discovery
    if not wid:
        wid = _pick_warehouse(exclude=_WH["bad"]) or ""
    _WH["id"] = wid
    return wid

def run_sql(sql, wait_timeout="50s"):
    """Execute SQL via the Statement Execution API, resolving the warehouse and
    retrying once on a freshly-picked warehouse if the chosen one is unusable
    (missing, stopped-and-unreachable, or no CAN_USE for the app principal)."""
    last = None
    for attempt in range(2):
        wid = resolve_warehouse_id(force=attempt > 0)
        if not wid:
            raise RuntimeError("no usable SQL warehouse available in this workspace")
        try:
            return ws().statement_execution.execute_statement(
                warehouse_id=wid, statement=sql, wait_timeout=wait_timeout)
        except Exception as e:                 # noqa: BLE001
            msg = str(e).lower()
            # Only fall back for warehouse/permission problems — a genuine SQL
            # error must not blacklist an otherwise-good warehouse.
            if not any(k in msg for k in ("warehouse", "permission", "does not exist", "not found")):
                raise
            last = e
            _WH["bad"].add(wid); _WH["id"] = None
    raise last


class MaskOptions(BaseModel):
    logos: bool = True; faces: bool = True; text: bool = True; signatures: bool = True
    sensitive: bool = False   # optional Claude-vision lane: in-image PII + sensitive items
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
    r = run_sql(sql)
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


def claude_pii_indices(contents):
    """Ask Claude which of the parsed text strings are PII → set of 0-based indices."""
    texts = "\n".join(f"{i}: {c}" for i, c in enumerate(contents))
    prompt = (f"{text_pii.SENSITIVE_TEXT_PROMPT}\n\nText items to review:\n{texts}\n\n"
              "Return a JSON array of index numbers (0-based) for items that should be masked.\n"
              "Return ONLY the JSON array. Example: [0, 2, 5]")
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    last_err = "unknown error"
    for attempt in range(2):  # try, then retry once
        try:
            resp = ws().serving_endpoints.query(
                name=CLAUDE_ENDPOINT,
                messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)])
            content = resp.choices[0].message.content
            # Reasoning models (e.g. databricks-claude-sonnet-5) return content as
            # a list of blocks ([{"type":"reasoning",...},{"type":"text",...}]);
            # older models return a plain string. Handle both.
            if isinstance(content, list):
                parts = []
                for b in content:
                    btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                    if btype == "text":
                        parts.append((b.get("text") if isinstance(b, dict) else getattr(b, "text", "")) or "")
                content = "".join(parts)
            raw = (content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            return set(int(i) for i in json.loads(raw) if isinstance(i, int))
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    # Both attempts failed → signal unavailable so the caller can surface the
    # error and apply the fail-safe (mask-all) instead of silently under-masking.
    raise text_pii.ClassifierUnavailable(last_err)


def sensitive_detect(img):
    """Optional VLM lane: Claude vision → boxes for sensitive items + in-image PII.

    Uses the app's service-principal auth to POST a multimodal request to the same
    Claude serving endpoint the PII classifier uses (CLAUDE_ENDPOINT). Raises on
    failure so the orchestrator surfaces a warn — the torch-free App has no local
    Grounding DINO fallback (that path lives in the notebook pipeline).
    """
    import urllib.request
    from pipeline import vlm_detect

    b64, sw, sh, W, H = vlm_detect.prepare_image(img)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": vlm_detect.build_prompt(sw, sh)},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        "max_tokens": 1500}
    cfg = ws().config
    headers = dict(cfg.authenticate()); headers["Content-Type"] = "application/json"
    url = cfg.host.rstrip("/") + f"/serving-endpoints/{CLAUDE_ENDPOINT}/invocations"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.loads(r.read())
    raw = vlm_detect.extract_text(body["choices"][0]["message"]["content"])
    return vlm_detect.boxes_from_text(raw, sw, sh, W, H)


def _events(req: MaskRequest):
    """Build the staged event generator for one request (load + run_stages).

    Detection backends are injected: vision → GPU Model Serving, text →
    ai_parse_document (SQL), PII → Claude. Load errors surface as an SSE event.
    """
    try:
        img = load_image(req.image, req.filename)
    except Exception as e:                                  # noqa: BLE001
        def fail():
            yield {"event": "error", "msg": f"could not read input: {e}"}
        return fail()
    opt = req.options.model_dump()
    opt["_filename"] = req.filename
    return run_stages(
        img, opt,
        vision_fn=lambda im, _o: vision_detect(im, req.options) if (req.options.logos or req.options.faces) else [],
        text_fn=lambda im: text_detect(im),
        pii_fn=claude_pii_indices,
        sensitive_fn=sensitive_detect,
    )


# ── GPU vision endpoint warm-state ────────────────────────────────────────────
# A scale-to-zero GPU endpoint reports READY even with 0 replicas; the only proof
# it's actually warm is a request that a replica answered. So we track warmth by
# firing a tiny inference and recording when it returns. The UI polls /api/status
# and gates Redact on "warm" so a request never fires into a ~150s cold start
# (which would blow the Apps ~120s ingress timeout).
import threading                                          # noqa: E402
import time as _time                                      # noqa: E402

_WARM = {"state": "cold", "ts": 0.0, "msg": ""}           # cold|warming|warm|error
_WARM_LOCK = threading.Lock()
_WARM_TTL = 15 * 60   # a warm replica may scale back to zero after idle → re-warm


def _do_warm():
    try:
        vision_detect(Image.new("RGB", (8, 8), "white"), MaskOptions())
        _WARM.update(state="warm", ts=_time.time(), msg="")
    except Exception as e:                                 # noqa: BLE001
        _WARM.update(state="error", msg=str(e))


def _trigger_warm():
    with _WARM_LOCK:
        if _WARM["state"] in ("cold", "error"):
            _WARM.update(state="warming", msg="")
            threading.Thread(target=_do_warm, daemon=True).start()


@app.post("/api/warm")
def warm():
    _trigger_warm()
    return {"state": _WARM["state"]}


@app.get("/api/status")
def status():
    # a warm replica can scale back to zero after idle → expire so the UI re-warms
    if _WARM["state"] == "warm" and _time.time() - _WARM["ts"] > _WARM_TTL:
        _WARM.update(state="cold")
    return {"state": _WARM["state"], "endpoint": VISION_ENDPOINT, "msg": _WARM.get("msg", "")}


@app.post("/api/mask/stream")
def mask_stream(req: MaskRequest):
    """Server-Sent Events: render → parse → text PII → vision → boxes → masked → done."""
    def gen():
        try:
            for e in _events(req):
                yield sse(e)
        except Exception as e:                             # noqa: BLE001
            yield sse({"event": "error", "msg": str(e)})
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/mask")
def mask(req: MaskRequest):
    """Legacy single-shot reply (drains the same staged pipeline)."""
    out = aggregate(_events(req))
    if out.get("error"):
        return JSONResponse({"error": out["error"]}, status_code=400)
    return out


# ── serve the SPA ─────────────────────────────────────────────────────────────
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_NOCACHE = {"Cache-Control": "no-cache, must-revalidate"}  # always revalidate so a
# redeploy is picked up immediately (index.html must never be served stale, else
# the ?v= asset busting never reaches the browser).
if os.path.isdir(WEB):
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        target = os.path.join(WEB, full_path)
        if full_path and os.path.isfile(target):
            return FileResponse(target, headers=_NOCACHE)
        return FileResponse(os.path.join(WEB, "index.html"), headers=_NOCACHE)
