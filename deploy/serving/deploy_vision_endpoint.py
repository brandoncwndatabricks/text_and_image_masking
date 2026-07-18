"""Package the vision pipeline as an MLflow model and create a GPU Model Serving
endpoint. RUN THIS INSIDE THE FEVM WORKSPACE (notebook or job) — mlflow, GPU
serving, and Unity Catalog are not available in the local dev sandbox.

It logs VisionRedactor (Grounding DINO + CLIP + YuNet) with the v5 `pipeline`
package as code, the YuNet ONNX + Databricks reference logos as artifacts,
registers it to Unity Catalog, and creates a scale-to-zero GPU endpoint.

Assumes this file sits in <root>/deploy/serving/ alongside the synced src_v5
tree (pipeline/, models/, refs/). Adjust CATALOG/SCHEMA to a UC location you can
write to in FEVM.
"""

import os
import mlflow
from mlflow.models import infer_signature
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V5 = os.path.dirname(os.path.dirname(HERE))           # .../src_v5
import sys; sys.path.insert(0, SRC_V5)

# ── config (edit to your workspace) ──────────────────────────────────────────
CATALOG = os.environ.get("UC_CATALOG", "main")
SCHEMA = os.environ.get("UC_SCHEMA", "image_masking")
MODEL_NAME = os.environ.get("MODEL_NAME_OVERRIDE", f"{CATALOG}.{SCHEMA}.vision_redactor")
ENDPOINT = os.environ.get("ENDPOINT_NAME", "image-masking-vision")
WORKLOAD_TYPE = os.environ.get("WORKLOAD_TYPE", "GPU_SMALL")  # T4-class; scale-to-zero

# Import as TOP-LEVEL `vision_model` so the pickled python_model references
# module 'vision_model' (which is shipped via code_paths), not
# 'deploy.serving.vision_model' (no 'deploy' package exists in the serving
# container → ModuleNotFoundError at load).
sys.path.insert(0, HERE)
import vision_model  # noqa: E402
VisionRedactor = vision_model.VisionRedactor


def main():
    mlflow.set_registry_uri("databricks-uc")
    spark_sql = None
    try:  # ensure the schema exists
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
    except Exception as e:
        print(f"[schema] {type(e).__name__}: {e} (continuing — may already exist)")

    # Download model weights now (the job has internet; the serving container
    # does NOT) and bundle them as artifacts so load is fully offline.
    from huggingface_hub import snapshot_download
    import tempfile
    print("downloading GDINO + CLIP weights to bundle as artifacts...")
    # local_dir → real file copies (not the HF cache's symlinks, which MLflow
    # rejects as 'outside the artifact directory'). GDINO ships safetensors so
    # we drop its *.bin; CLIP (clip-vit-base-patch32) ships ONLY pytorch_model.bin,
    # so we must keep .bin there or no weights get bundled.
    misc = ["*.msgpack", "*.h5", "*.ot"]
    gdino_dir = snapshot_download("IDEA-Research/grounding-dino-base",
                                  local_dir=tempfile.mkdtemp(prefix="gdino_"),
                                  ignore_patterns=misc + ["*.bin"])
    clip_dir = snapshot_download("openai/clip-vit-base-patch32",
                                 local_dir=tempfile.mkdtemp(prefix="clip_"),
                                 ignore_patterns=misc)

    import base64, io
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (64, 64), "white").save(buf, "PNG")
    example = pd.DataFrame([{"image_b64": base64.b64encode(buf.getvalue()).decode(),
                             "options": '{"logos": true, "faces": true}'}])
    signature = infer_signature(example, ["{}"])  # output: JSON string per row

    with mlflow.start_run(run_name="vision_redactor"):
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=VisionRedactor(),
            code_paths=[os.path.join(SRC_V5, "pipeline"),
                        os.path.join(SRC_V5, "deploy", "serving", "vision_model.py")],
            artifacts={
                "yunet": os.path.join(SRC_V5, "models", "face_detection_yunet_2023mar.onnx"),
                "refs": os.path.join(SRC_V5, "refs"),
                "gdino": gdino_dir,   # bundled Grounding DINO weights
                "clip": clip_dir,     # bundled CLIP weights
            },
            input_example=example,
            signature=signature,
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "mlflow", "torch", "torchvision", "transformers>=4.51",
                "opencv-python-headless", "Pillow", "numpy",
            ],
        )
    print("logged + registered:", MODEL_NAME, "uri:", info.model_uri)

    # latest registered version
    from mlflow.tracking import MlflowClient
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    version = max(int(v.version) for v in versions)
    print("model version:", version)

    # create / update the GPU serving endpoint
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType)
    w = WorkspaceClient()
    served = [ServedEntityInput(
        entity_name=MODEL_NAME, entity_version=str(version),
        workload_type=ServingModelWorkloadType(WORKLOAD_TYPE),  # enum, not str
        workload_size="Small", scale_to_zero_enabled=True)]
    existing = [e.name for e in (w.serving_endpoints.list() or [])]
    if ENDPOINT in existing:
        print("updating endpoint config:", ENDPOINT)
        w.serving_endpoints.update_config(name=ENDPOINT, served_entities=served)
    else:
        print("creating endpoint:", ENDPOINT)
        w.serving_endpoints.create(name=ENDPOINT,
                                   config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=served))
    print("DONE. Endpoint:", ENDPOINT, "→ poll readiness in the Serving UI.")


if __name__ == "__main__":
    main()
