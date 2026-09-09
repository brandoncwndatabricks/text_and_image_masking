# Redaction Studio — app

A web UI for the v5 masking pipeline: upload a document, pick **what** to mask
(logos / faces / text-PII / signatures / sensitive items), and see the **before / after**
side-by-side with an optional debug overlay.

**Target architecture:** React (Vite) frontend + FastAPI backend on **Databricks
Apps**, with the heavy ML pipeline behind a **Model Serving endpoint** (Apps
containers have no GPU) called **async submit-then-poll** (≈120s ingress limit).

## Run locally (now — zero installs)

This dev environment has **no PyPI/npm access**, so the runnable local version
uses only the standard library + Pillow (already installed) and loads React from
a CDN via an import map — **no `npm install`, no `uv add`, no build step**:

```bash
cd src_v5/app
/Users/brandon.cowen/Projects/image-masking/.venv/bin/python server.py
# open http://localhost:8000
```

`server.py` serves the `web/` UI and a `/api/mask` endpoint. By default it's a
**mock** (Pillow draws placeholder boxes — fast, no models). This is the UX
shell: toggles, mask-style controls, the Databricks-logo allowlist switch,
side-by-side render, the debug-overlay toggle, and the progress indicator.

### Real detections (aligned boxes)

The mock's boxes are placeholders and won't sit on real content. To run the
**actual MaskingPipeline** locally (the venv has torch/transformers/opencv):

```bash
# Logos + faces (local models only; fast after a ~11s first-call model load):
REDACT_REAL=1 .venv/bin/python server.py

# Also enable the text/PII + signature phase (needs Databricks: ai_parse_document
# + Claude). Slower per doc (~10-40s); the deploy version uses async submit-poll.
REDACT_REAL=1 REDACT_TEXT=1 DATABRICKS_PROFILE=e2-field-eng-west .venv/bin/python server.py
```

In real mode the pipeline (Grounding DINO + CLIP gate, YuNet, ai_parse + Claude)
produces accurate boxes; the UI response includes `"mock": false`. Models load
once and are cached across requests.

## Supported input formats

Matches `ai_parse_document`: **PDF, JPG/JPEG, PNG, TIFF/TIF, DOC/DOCX, PPT/PPTX**
(max 100 MB / 500 pages). Locally: images and PDF (rasterized page 1 via macOS
`sips`) run through the full pipeline; DOC/PPT return a clear message because
they need server-side rendering (LibreOffice) not present in this dev sandbox —
the **deployed** app parses all of them via `ai_parse_document` with a rasterizer
feeding the visual detectors.

## Files

| File | Role |
| ---- | ---- |
| `server.py` | **Local mock backend** — stdlib `http.server` + Pillow, zero deps. Runnable now. |
| `web/index.html`, `web/app.js`, `web/styles.css` | **React frontend** (React via CDN import map; `app.js` is plain React + htm). |
| `app.py` | **FastAPI deploy target** — same `/api/mask` contract; swap-in point for the real pipeline / Model Serving. Not runnable in the sandbox (needs fastapi). |
| `app.yaml`, `pyproject.toml` | Databricks Apps config + deploy deps. |

## `/api/mask` contract (mock and prod identical)

```jsonc
POST /api/mask
{ "image": "data:image/png;base64,…",
  "options": { "logos": true, "faces": true, "text": true, "signatures": true,
               "sensitive": false,
               "face_style": "blur", "other_style": "black", "keep_databricks": true } }
→ { "original": "data:…", "masked": "data:…", "overlay": "data:…",
    "detections": [ { "source": "logo|face|text|signature|sensitive", "label": "...",
                      "score": 0.0, "box": [x1,y1,x2,y2], "mask": true } ],
    "timing_s": 0.0 }
```

## Porting to the Vite + FastAPI deploy

1. `npm create vite@latest frontend -- --template react` and move `web/app.js`
   into `frontend/src/App.jsx` (replace the CDN import map with normal
   `import React` — the component code is unchanged; it already uses standard
   React hooks).
2. Implement `app.py`'s `/api/mask` against the **MaskingPipeline Model Serving
   endpoint** (see the swap-in comment). Switch to submit-then-poll for batch.
3. `npm run build` → `frontend/dist`; FastAPI serves it via `StaticFiles`.
4. Deploy with the `databricks apps` workflow (see the `databricks-apps` skill).

> Why not Vite locally now? `npm install` can't reach the registry in this
> sandbox. The CDN approach gives a real, runnable React UI today; the Vite
> structure above is the build/deploy path once a network is available.
