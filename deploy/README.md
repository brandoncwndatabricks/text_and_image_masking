# Deploy — two ways to run the redaction app

The app (`src_v5/app/`) gets its **vision** detections (logos + faces) from one
of two backends, selected by the `VISION_BACKEND` env var. The **text/PII**
phase (`ai_parse_document` + Claude) always runs in the app, because Model
Serving endpoints have no Spark.

## Option A — Local

Everything runs in-process (the venv has torch/transformers/opencv):

```bash
cd src_v5/app
# logos + faces only:
REDACT_REAL=1 VISION_BACKEND=local python server.py
# + text/PII + signatures (needs a Databricks profile for ai_parse + Claude):
REDACT_REAL=1 REDACT_TEXT=1 VISION_BACKEND=local \
  DATABRICKS_PROFILE=fevm-lsryzn python server.py
```

Best for development and demos. No GPU endpoint, no cost; first request loads
models (~11s), then logos+faces ~1.5s.

## Option B — Databricks Model Serving (GPU)

The vision models (Grounding DINO + CLIP + YuNet) run on a **GPU Model Serving
endpoint**; the app calls it. Best for a shared/deployed app where the client
has no GPU.

### One-time: create the endpoint (run IN the workspace)

MLflow isn't installable in the offline dev sandbox, so packaging runs in the
FEVM workspace via a serverless job:

```bash
# 1. stage the model code + assets and sync to the workspace
#    (pipeline/, deploy/serving/, models/yunet, refs/, run_deploy.py)
databricks sync <staged_dir> \
  /Workspace/Users/<you>/image_masking_v5deploy -p fevm-lsryzn

# 2. run the packaging notebook as a one-time serverless job
databricks jobs submit -p fevm-lsryzn --no-wait --json '{
  "run_name": "deploy-vision-endpoint",
  "tasks": [{"task_key": "deploy", "notebook_task": {
    "notebook_path": "/Workspace/Users/<you>/image_masking_v5deploy/run_deploy"}}]}'
```

`run_deploy.py` (notebook) pip-installs mlflow, then `deploy_vision_endpoint.py`
logs `VisionRedactor` (pyfunc), registers it to Unity Catalog
(`serverless_stable_lsryzn_catalog.image_masking.vision_redactor`), and creates a
scale-to-zero **GPU_SMALL** endpoint `image-masking-vision`. The endpoint then
**builds for ~20–60 min** (image build + model download); watch the Serving UI.

### Status (FEVM)

**Live + verified.** Endpoint `image-masking-vision` (UC model
`serverless_stable_lsryzn_catalog.image_masking.vision_redactor`) is
`DEPLOYMENT_READY` on GPU_SMALL scale-to-zero. A query with the Audi image
returns the real grille-emblem box (~0.54); ~18s round-trip when scaling from
zero. The app's `VISION_BACKEND=serving` path was validated end-to-end
(`mock=false`).

> Recall note: `vision_model.py` builds `LogoDetector()` with default thresholds
> (box 0.30 / text 0.25) — slightly more conservative than local dev (0.25/0.20),
> so very faint wordmarks may be missed. Lower the thresholds in
> `vision_model.load_context` and redeploy to match local recall.

### Point the app at the endpoint

```bash
cd src_v5/app
REDACT_REAL=1 REDACT_TEXT=1 VISION_BACKEND=serving \
  VISION_ENDPOINT=image-masking-vision DATABRICKS_PROFILE=fevm-lsryzn \
  python server.py
```

The app queries the endpoint for logo/face boxes and runs text/PII locally.

## Files

| File | Role |
| ---- | ---- |
| `serving/vision_model.py` | MLflow pyfunc — vision detectors, JSON in/out |
| `serving/deploy_vision_endpoint.py` | log → register (UC) → create GPU endpoint |
| `serving/run_deploy.py` | notebook wrapper (pip install + run) for the serverless job |

## The app on Databricks Apps (LIVE)

`app/app.py` is a light FastAPI app (no torch) deployed to Databricks Apps. It
serves the SPA and orchestrates: **vision** → Model Serving endpoint;
**text** → `ai_parse_document` via SQL Statement Execution (`SQL_WAREHOUSE_ID`);
**PII** → Claude serving endpoint; **mask/overlay** → Pillow.

Live: **https://image-masking-redactor-7474647639299343.aws.databricksapps.com**
(verified end-to-end: a document returns logo + 10 PII text masks, `mock=false`).

### Deploy steps

```bash
# bundle = app/ files + the torch-free pipeline subset (boxes, masking, overlay, text_pii)
databricks apps create image-masking-redactor -p fevm-lsryzn
databricks sync <bundle> /Workspace/Users/<you>/image_masking_app -p fevm-lsryzn
databricks apps deploy image-masking-redactor \
  --source-code-path /Workspace/Users/<you>/image_masking_app -p fevm-lsryzn
# grant the app's service principal:
#   CAN_QUERY on the vision endpoint (by endpoint ID) and the Claude endpoint
#   CAN_USE on the SQL warehouse
```

`app.yaml` sets `VISION_ENDPOINT`, `SQL_WAREHOUSE_ID`, `CLAUDE_ENDPOINT`.

### Operational notes
- **Cold start:** the GPU endpoint is scale-to-zero; a cold first request (~150s)
  exceeds the Apps ~120s ingress timeout. Keep it warm, or implement async
  submit-then-poll. Warm requests: logos ~14s, a full document ~28–37s.
- The Claude call must use `ChatMessage` objects (not dicts) with the SDK
  `serving_endpoints.query`.
