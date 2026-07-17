"""Lightweight MLflow tracking for masking model + workflow performance.

Wraps an evaluation of the pipeline over a labelled image set and records, per
MLflow run:

  params   — the PipelineConfig (thresholds, mask styles), each detector's model
             repo id, IoU threshold, device, git commit.
  metrics  — MODEL quality (per category: precision / recall / coverage @ IoU)
             and WORKFLOW behaviour (end-to-end latency, detections per source,
             mean confidence, masked-area fraction = over-masking proxy).
  artifacts— debug overlays, masked outputs, the scored report JSON, ground truth.

Design: MLflow is imported lazily and optionally. On Databricks (MLflow native)
it logs a real run; with no MLflow installed it runs in DRY-RUN mode — it still
computes and returns the identical payload and writes report.json, so the metric
logic is testable locally (PyPI is firewalled in dev). One config = one run, so
threshold sweeps / model swaps are directly comparable in the MLflow UI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from PIL import Image

_SRC_V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_V5 not in sys.path:
    sys.path.insert(0, _SRC_V5)

from pipeline.boxes import area_frac          # noqa: E402
from pipeline.overlay import render_overlay    # noqa: E402
from pipeline.masking import apply_masks        # noqa: E402
from eval.score import score_set                # noqa: E402


def _mlflow():
    """Return the mlflow module, or None if unavailable (dry-run mode)."""
    try:
        import mlflow
        return mlflow
    except Exception:
        return None


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_SRC_V5,
        ).stdout.strip() or None
    except Exception:
        return None


def _flatten_params(pipeline, iou_thr: float) -> Dict[str, object]:
    """Pipeline config + model ids → a flat params dict for MLflow."""
    from pipeline.detectors import pick_device

    cfg = pipeline.cfg
    params: Dict[str, object] = {f"cfg.{k}": v for k, v in asdict(cfg).items()}
    params["iou_threshold"] = iou_thr
    params["device"] = pick_device()
    # Model repo ids (only for detectors actually enabled / instantiated)
    params["model.logo"] = "IDEA-Research/grounding-dino-base" if cfg.do_logos else None
    params["model.clip"] = "openai/clip-vit-base-patch32" if cfg.use_clip_gate else None
    params["model.face"] = "opencv/YuNet (face_detection_yunet_2023mar)" if cfg.do_faces else None
    params["model.text"] = "databricks/ai_parse_document" if cfg.do_text else None
    sha = _git_sha()
    if sha:
        params["git_sha"] = sha
    return {k: v for k, v in params.items() if v is not None}


def _collect(pipeline, image_paths: List[str], out_dir: str,
             log_images: bool) -> tuple[Dict[str, List], Dict[str, object]]:
    """Run the pipeline over the set; return (results_by_file, workflow_stats)."""
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, List] = {}
    latencies, mask_fracs = [], []
    src_counts: Dict[str, int] = {}
    src_scores: Dict[str, List[float]] = {}

    for path in image_paths:
        fname = os.path.basename(path)
        image = Image.open(path).convert("RGB")
        w, h = image.size
        t0 = time.time()
        dets = pipeline.detect(image, path)
        latencies.append(time.time() - t0)
        results[fname] = dets

        masked_boxes = [d.box for d in dets if d.mask]
        frac = float(sum(area_frac(b, w, h) for b in masked_boxes))  # upper-bound (overlaps double-count)
        mask_fracs.append(min(1.0, frac))
        for d in dets:
            src_counts[d.source] = src_counts.get(d.source, 0) + (1 if d.mask else 0)
            src_scores.setdefault(d.source, []).append(float(d.score))

        if log_images:
            stem = os.path.splitext(fname)[0]
            render_overlay(image, dets).save(os.path.join(out_dir, f"{stem}_overlay.jpg"))
            apply_masks(image, dets).save(os.path.join(out_dir, f"{stem}_masked.jpg"), quality=92)

    n = max(1, len(image_paths))
    stats = {
        "workflow.images": len(image_paths),
        "workflow.latency_s_mean": round(sum(latencies) / n, 3),
        "workflow.latency_s_total": round(sum(latencies), 3),
        "workflow.masked_area_frac_mean": round(sum(mask_fracs) / n, 4),
        "workflow.masks_total": int(sum(src_counts.values())),
    }
    for src, c in src_counts.items():
        stats[f"count.{src}_masked"] = c
    for src, scores in src_scores.items():
        if scores:
            stats[f"score.{src}_mean"] = round(sum(scores) / len(scores), 3)
    return results, stats


def evaluate_and_log(
    pipeline,
    image_paths: List[str],
    gt_path: Optional[str] = None,
    *,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    iou_thr: float = 0.5,
    log_images: bool = True,
    extra_params: Optional[dict] = None,
    extra_tags: Optional[dict] = None,
) -> dict:
    """Evaluate the pipeline over image_paths and log one MLflow run.

    Returns the payload {params, metrics, report} regardless of MLflow presence.
    If gt_path is given, model-quality metrics (precision/recall/coverage) are
    scored against it; otherwise only workflow metrics are recorded.
    """
    out_dir = output_dir or os.path.join(_SRC_V5, "outputs", "tracking")
    params = _flatten_params(pipeline, iou_thr)
    if extra_params:
        params.update(extra_params)

    results, workflow = _collect(pipeline, image_paths, out_dir, log_images)
    metrics: Dict[str, float] = dict(workflow)

    report = None
    if gt_path and os.path.exists(gt_path):
        report = score_set(results, gt_path, iou_thr)
        for cat, m in report["summary"].items():
            for key in ("precision", "recall", "coverage"):
                if m.get(key) is not None:
                    metrics[f"{cat}.{key}"] = m[key]
            metrics[f"{cat}.tp"] = m["tp"]
            metrics[f"{cat}.fp"] = m["fp"]
            metrics[f"{cat}.fn"] = m["fn"]
        report_path = os.path.join(out_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    payload = {"params": params, "metrics": metrics, "report": report}

    mlflow = _mlflow()
    if mlflow is None:
        print("[tracking] MLflow not installed — DRY RUN (metrics computed, not logged).")
        print(json.dumps({"params": params, "metrics": metrics}, indent=2, default=str))
        return payload

    if experiment:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        tags = {"pipeline": "src_v5", "component": "masking"}
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)
        if os.path.isdir(out_dir):
            mlflow.log_artifacts(out_dir, artifact_path="eval")
        if gt_path and os.path.exists(gt_path):
            mlflow.log_artifact(gt_path, artifact_path="ground_truth")
    print(f"[tracking] Logged MLflow run — {len(params)} params, {len(metrics)} metrics.")
    return payload


def sweep_and_log(
    make_pipeline,
    configs: List[dict],
    image_paths: List[str],
    gt_path: Optional[str] = None,
    *,
    experiment: Optional[str] = None,
    iou_thr: float = 0.5,
    log_images: bool = False,
) -> List[dict]:
    """Run a threshold/model sweep — one MLflow run per config.

    make_pipeline(overrides: dict) -> MaskingPipeline. Each `configs` entry is a
    dict of overrides (e.g. {"logo_box_threshold": 0.3}); it also names the run.
    Returns the list of payloads. Compare runs side-by-side in the MLflow UI.
    """
    payloads = []
    for i, overrides in enumerate(configs):
        pipe = make_pipeline(overrides)
        name = overrides.get("_run_name") or f"sweep_{i:02d}_" + "_".join(
            f"{k}={v}" for k, v in overrides.items() if not k.startswith("_")
        )
        payloads.append(evaluate_and_log(
            pipe, image_paths, gt_path,
            experiment=experiment, run_name=name, iou_thr=iou_thr,
            log_images=log_images,
            extra_params={f"sweep.{k}": v for k, v in overrides.items() if not k.startswith("_")},
        ))
    return payloads
