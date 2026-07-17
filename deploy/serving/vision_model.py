"""MLflow pyfunc wrapping the VISION half of the masking pipeline.

Hosts ONLY the GPU-bound, self-contained detectors — Grounding DINO (logos) +
CLIP gate + YuNet (faces) — for a Databricks GPU Model Serving endpoint. The
text/PII phase (ai_parse_document + Claude) stays in the app because serving
endpoints have no Spark; the app merges vision + text results.

I/O contract (Model Serving JSON):
  input  : records with {"image_b64": "<base64 PNG/JPG>", "options": "<json>"}
           options = {"logos": bool, "faces": bool, "keep_databricks": bool}
  output : JSON string per row: {"detections": [{source,label,score,box,mask}],
                                 "width": w, "height": h}

Logged/registered/served by deploy_vision_endpoint.py (run IN the workspace —
mlflow + GPU aren't available in the local dev sandbox).
"""

from __future__ import annotations

import base64
import io
import json

import mlflow.pyfunc


class VisionRedactor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import sys, os
        # the v5 `pipeline` package is shipped via code_paths
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pipeline.detectors import LogoDetector, FaceDetector
        from pipeline.verify import ClipGate

        # All weights are bundled as artifacts (serving containers have no
        # internet egress, so from_pretrained(hub_id) would fail at load).
        a = context.artifacts
        self._logo = LogoDetector(model_id=a["gdino"])
        self._face = FaceDetector(model_path=a["yunet"])
        self._clip = ClipGate(model_id=a["clip"], refs_dir=a.get("refs"))

    def predict(self, context, model_input):
        from PIL import Image
        rows = (model_input.to_dict("records")
                if hasattr(model_input, "to_dict") else model_input)
        out = []
        for row in rows:
            opt = row.get("options")
            opt = json.loads(opt) if isinstance(opt, str) else (opt or {})
            img = Image.open(io.BytesIO(base64.b64decode(row["image_b64"]))).convert("RGB")
            dets = []
            if opt.get("logos", True):
                logo = self._logo.detect(img)
                logo = self._clip.filter_logos(img, logo)
                dets += logo
            if opt.get("faces", True):
                dets += self._face.detect(img)
            out.append(json.dumps({
                "width": img.width, "height": img.height,
                "detections": [{"source": d.source, "label": d.label,
                                 "score": round(float(d.score), 3),
                                 "box": [round(float(v), 1) for v in d.box],
                                 "mask": bool(d.mask)} for d in dets],
            }))
        return out
