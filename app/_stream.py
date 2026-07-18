"""Staged redaction orchestration — shared by app.py (deployed) and server.py (dev).

ONE place owns the pipeline order, the PII decision, masking, overlay and the
data-url plumbing; the two HTTP layers just adapt it to their server (FastAPI
SSE vs stdlib SSE) and inject *where* detection runs:

    vision_fn(img, opt) -> [Detection]   # logos+faces  (GPU serving / local / mock)
    text_fn(img)        -> [Detection]   # ai_parse_document elements (text source)
    pii_fn(contents)    -> set[int]|None # indices to mask (Claude); None = regex-only

`run_stages` is a generator of small JSON-able dict events. It renders the doc
immediately, cracks + screens text while vision runs concurrently (vision is the
cold-start path), then reveals bounding boxes and the masked result — so the
~minute wait shows continuous progress instead of one long spinner.
"""

from __future__ import annotations

import base64
import io
import threading
import time

from pipeline.boxes import Detection, merge_sources   # noqa: E402  (torch-free)
from pipeline.masking import apply_masks               # noqa: E402
from pipeline.overlay import render_overlay            # noqa: E402
from pipeline import text_pii                          # noqa: E402

# Stage panel definition — id order == display order. Sent to the client once so
# the frontend renders the panel generically (icons live in the frontend).
STAGES = [
    {"id": "crack",      "label": "Cracking document"},
    {"id": "text_id",    "label": "Identifying text PII"},
    {"id": "text_mask",  "label": "Masking text PII"},
    {"id": "img_detect", "label": "Detecting image PII"},
    {"id": "boxes",      "label": "Drawing bounding boxes"},
    {"id": "img_mask",   "label": "Masking image PII"},
    {"id": "final",      "label": "Finalizing"},
]


def to_data_url(im) -> str:
    b = io.BytesIO(); im.convert("RGB").save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _det_json(d: Detection) -> dict:
    return {"source": d.source, "label": d.label, "score": round(float(d.score), 3),
            "box": [round(float(v), 1) for v in d.box], "mask": bool(d.mask)}


def run_stages(img, opt, *, vision_fn, text_fn, pii_fn, step_delay: float = 0.0):
    """Yield staged redaction events for one image. See module docstring."""
    t0 = time.time()
    W, H = img.size
    want_vision = bool(opt.get("logos") or opt.get("faces"))
    want_text = bool(opt.get("text"))
    want_sig = bool(opt.get("signatures"))
    do_text = want_text or want_sig

    yield {"event": "stages", "stages": STAGES}
    yield {"event": "render", "original": to_data_url(img), "w": W, "h": H,
           "filename": opt.get("_filename", "")}

    # Vision (logos+faces) is the slow / GPU-cold-start path → start it now and
    # let the text phase run while it warms up.
    vbox = {"dets": [], "err": None}
    vthread = None
    if want_vision and vision_fn:
        yield {"event": "stage", "id": "img_detect", "status": "active"}

        def _vis():
            try:
                vbox["dets"] = vision_fn(img, opt) or []
            except Exception as e:                          # noqa: BLE001
                vbox["err"] = str(e)
        vthread = threading.Thread(target=_vis, daemon=True); vthread.start()
    else:
        yield {"event": "stage", "id": "img_detect", "status": "skip"}
        yield {"event": "stage", "id": "img_mask", "status": "skip"}

    # ── Text: crack → screen for PII ──────────────────────────────────────────
    text_dets = []
    if do_text and text_fn:
        yield {"event": "stage", "id": "crack", "status": "active"}
        if step_delay: time.sleep(step_delay)
        try:
            text_dets = text_fn(img) or []
        except Exception as e:                              # noqa: BLE001
            yield {"event": "warn", "where": "parse", "msg": str(e)}
        text_pii.flag_signatures(text_dets, img_h=H)
        text_only = [d for d in text_dets if d.source == "text"]
        yield {"event": "parse", "n": len(text_only),
               "sample": [{"type": d.meta.get("elem_type", "text"),
                           "text": (d.meta.get("content", d.label) or "")[:90]}
                          for d in text_only[:8]]}
        yield {"event": "stage", "id": "crack", "status": "done"}

        yield {"event": "stage", "id": "text_id", "status": "active"}
        if step_delay: time.sleep(step_delay)
        contents = [d.meta.get("content", d.label) for d in text_only]
        # Classify. pii_fn retries once internally, then raises
        # ClassifierUnavailable. On failure we FAIL SAFE (mask all text) and
        # surface the error to the UI rather than silently under-masking.
        claude_idx = None
        classifier_error = None
        if want_text and pii_fn:
            try:
                claude_idx = pii_fn(contents)
            except text_pii.ClassifierUnavailable as e:
                classifier_error = str(e)
                yield {"event": "warn", "where": "pii_classifier",
                       "msg": f"PII classifier unavailable ({e}); masking all text as a fail-safe."}
        for i, d in enumerate(text_only):
            if not want_text:
                d.mask = False
            elif text_pii.regex_is_sensitive(contents[i]):
                d.mask = True
            elif claude_idx is not None:
                d.mask = i in claude_idx
            else:
                # classifier failed → mask-all fail-safe
                d.mask = classifier_error is not None
                if classifier_error:
                    d.meta["classifier_error"] = classifier_error
        flagged = [d for d in text_only if d.mask]
        yield {"event": "text_pii", "flagged": len(flagged), "screened": len(text_only),
               "error": classifier_error,
               "items": [{"text": (d.meta.get("content", d.label) or "")[:60]}
                         for d in flagged][:12]}
        yield {"event": "stage", "id": "text_id", "status": "done"}
    else:
        for sid in ("crack", "text_id", "text_mask"):
            yield {"event": "stage", "id": sid, "status": "skip"}

    # ── Join vision ───────────────────────────────────────────────────────────
    # Vision can be a long GPU cold start (~150s). Emit heartbeats while waiting
    # so the SSE connection keeps flowing bytes and the Databricks Apps ingress
    # (~120s idle timeout) doesn't drop the stream.
    vdets = []
    if vthread:
        while vthread.is_alive():
            vthread.join(timeout=5.0)
            if vthread.is_alive():
                yield {"event": "heartbeat", "at": "img_detect"}
        if vbox["err"]:
            yield {"event": "warn", "where": "vision", "msg": vbox["err"]}
        vdets = vbox["dets"]
        yield {"event": "vision", "detections": [_det_json(d) for d in vdets]}
        yield {"event": "stage", "id": "img_detect", "status": "done"}

    # ── Merge + filter by toggles ─────────────────────────────────────────────
    keep = set()
    if opt.get("logos"): keep.add("logo")
    if opt.get("faces"): keep.add("face")
    if want_text: keep.add("text")
    if want_sig: keep.add("signature")
    shown = [d for d in merge_sources(list(vdets) + list(text_dets)) if d.source in keep]
    final = [d for d in shown if d.mask]

    # ── Bounding boxes ────────────────────────────────────────────────────────
    yield {"event": "stage", "id": "boxes", "status": "active"}
    yield {"event": "overlay", "overlay": to_data_url(render_overlay(img, shown)),
           "detections": [_det_json(d) for d in shown]}
    yield {"event": "stage", "id": "boxes", "status": "done"}

    # ── Masking ───────────────────────────────────────────────────────────────
    if do_text: yield {"event": "stage", "id": "text_mask", "status": "active"}
    if want_vision: yield {"event": "stage", "id": "img_mask", "status": "active"}
    styles = {"face": opt.get("face_style", "blur"), "logo": opt.get("other_style", "black"),
              "text": opt.get("other_style", "black"), "signature": opt.get("other_style", "black")}
    yield {"event": "masked", "masked": to_data_url(apply_masks(img, final, style_overrides=styles))}
    if do_text: yield {"event": "stage", "id": "text_mask", "status": "done"}
    if want_vision: yield {"event": "stage", "id": "img_mask", "status": "done"}

    # ── Done ──────────────────────────────────────────────────────────────────
    yield {"event": "stage", "id": "final", "status": "active"}
    yield {"event": "stage", "id": "final", "status": "done"}
    yield {"event": "done", "timing_s": round(time.time() - t0, 2),
           "detections": [_det_json(d) for d in shown],
           "counts": {"masked": len(final), "kept": len(shown) - len(final)}}


# ── adapters shared by both servers ────────────────────────────────────────────
def sse(event: dict) -> str:
    """Format one event dict as an SSE frame."""
    import json
    return "data: " + json.dumps(event) + "\n\n"


def aggregate(events) -> dict:
    """Drain a run_stages generator into the legacy single-shot /api/mask reply."""
    out = {"original": None, "masked": None, "overlay": None,
           "detections": [], "timing_s": 0.0, "mock": False}
    for e in events:
        ev = e.get("event")
        if ev == "render":      out["original"] = e["original"]
        elif ev == "overlay":   out["overlay"] = e["overlay"]; out["detections"] = e["detections"]
        elif ev == "masked":    out["masked"] = e["masked"]
        elif ev == "done":      out["timing_s"] = e["timing_s"]
        elif ev == "error":     out["error"] = e["msg"]
    return out
