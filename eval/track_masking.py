"""Runnable MLflow tracking workflow for the masking pipeline.

Lightweight entry point you can run as a Databricks job, a notebook cell, or
locally (dry-run). It evaluates the pipeline over a labelled image set and logs
one MLflow run — or a sweep of runs — capturing model quality + workflow
performance. See pipeline/tracking.py for what gets logged.

Examples
--------
# Single run (logos + faces locally; dry-run if MLflow absent):
python track_masking.py --images-dir ../test_set --gt eval/ground_truth.json \
    --experiment /Shared/image-masking-v5

# Include the text phase (needs a Databricks session + Claude endpoint):
python track_masking.py --do-text --experiment /Shared/image-masking-v5

# Threshold sweep — one MLflow run per logo threshold:
python track_masking.py --sweep --experiment /Shared/image-masking-v5

On a Databricks notebook MLflow is auto-configured. Running via
databricks-connect from elsewhere, first `export MLFLOW_TRACKING_URI=databricks`
(or call mlflow.set_tracking_uri("databricks")).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

SRC_V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_V5)

from pipeline.run import MaskingPipeline, PipelineConfig          # noqa: E402
from pipeline.tracking import evaluate_and_log, sweep_and_log     # noqa: E402

CLAUDE_ENDPOINT = (
    "https://e2-demo-field-eng.cloud.databricks.com"
    "/serving-endpoints/databricks-claude-sonnet-5/invocations"
)
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


def discover(images_dir: str):
    return sorted(
        p for e in EXTS for p in glob.glob(os.path.join(images_dir, e))
        if "_masked" not in os.path.basename(p) and "_overlay" not in os.path.basename(p)
    )


def maybe_spark(do_text: bool, profile: str):
    if not do_text:
        return None
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", default=os.path.join(SRC_V5, "test_set"))
    ap.add_argument("--gt", default=os.path.join(SRC_V5, "eval", "ground_truth.json"))
    ap.add_argument("--experiment", default=None, help="MLflow experiment path, e.g. /Shared/image-masking-v5")
    ap.add_argument("--run-name", default="masking-eval")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--do-text", action="store_true", help="enable ai_parse_document text phase (needs Databricks)")
    ap.add_argument("--profile", default="e2-field-eng-west")
    ap.add_argument("--no-images", action="store_true", help="skip logging overlay/masked artifacts")
    ap.add_argument("--sweep", action="store_true", help="run a logo-threshold sweep instead of a single run")
    args = ap.parse_args()

    image_paths = discover(args.images_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images_dir}")
    gt = args.gt if os.path.exists(args.gt) else None
    if gt is None:
        print(f"[track] No ground truth at {args.gt} — logging workflow metrics only "
              "(no precision/recall/coverage).")
    spark = maybe_spark(args.do_text, args.profile)

    base = dict(do_logos=True, do_faces=True, do_text=args.do_text, use_clip_gate=True)

    def make_pipeline(overrides):
        cfg = PipelineConfig(**{**base, **{k: v for k, v in overrides.items() if not k.startswith("_")}})
        return MaskingPipeline(cfg, spark=spark,
                               claude_endpoint=CLAUDE_ENDPOINT if args.do_text else None)

    if args.sweep:
        configs = [
            {"logo_box_threshold": 0.20, "_run_name": "logo_thr_0.20"},
            {"logo_box_threshold": 0.25, "_run_name": "logo_thr_0.25"},
            {"logo_box_threshold": 0.30, "_run_name": "logo_thr_0.30"},
            {"logo_box_threshold": 0.40, "_run_name": "logo_thr_0.40"},
        ]
        sweep_and_log(make_pipeline, configs, image_paths, gt,
                      experiment=args.experiment, iou_thr=args.iou, log_images=False)
    else:
        pipe = make_pipeline({})
        evaluate_and_log(pipe, image_paths, gt,
                         experiment=args.experiment, run_name=args.run_name,
                         iou_thr=args.iou, log_images=not args.no_images)


if __name__ == "__main__":
    main()
