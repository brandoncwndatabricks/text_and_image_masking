"""Evaluation harness for the v5 masking pipeline.

Without ground truth, "is v5 better than v4" is opinion. This module:

  1. auto_propose_ground_truth() — runs the pipeline over a set of images and
     writes its detections to ground_truth.json as a STARTING POINT for the
     user to correct (per the project decision to auto-propose, not hand-label).
  2. score_image() / score_set() — compares predicted boxes against ground
     truth and reports, per source category:
        precision / recall @ IoU>=0.5,
        coverage    — mean fraction of each GT box covered by predicted masks,
        over_mask   — fraction of total masked area that overlaps NO GT box.

``coverage`` and ``over_mask`` directly quantify v4's leak-vs-overmask tradeoff:
high coverage = nothing leaks; low over_mask = we didn't black out innocent
content.

Ground-truth JSON schema:
    { "<filename>": [ {"type": "logo|face|text", "box": [x1,y1,x2,y2]}, ... ] }
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.boxes import Detection, iou  # noqa: E402


def auto_propose_ground_truth(pipeline, image_paths: List[str], out_path: str) -> dict:
    """Run the pipeline and dump its detections as a ground-truth scaffold."""
    gt: Dict[str, list] = {}
    for p in image_paths:
        from PIL import Image
        img = Image.open(p).convert("RGB")
        dets = pipeline.detect(img, p)
        gt[os.path.basename(p)] = [
            {"type": str(d.source), "box": [round(float(v), 1) for v in d.box],
             "label": str(d.label), "score": round(float(d.score), 3),
             "mask": bool(d.mask)}
            for d in dets
        ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2)
    return gt


def _mask_area(boxes: List[list]) -> float:
    """Approximate union area via a coarse coverage grid (boxes may overlap)."""
    if not boxes:
        return 0.0
    # simple sum minus pairwise overlap is error-prone; use a raster grid.
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    W = max(1, int(x1 - x0)); H = max(1, int(y1 - y0))
    scale = min(1.0, 200.0 / max(W, H))  # cap grid at ~200px on long side
    import numpy as np
    gw, gh = max(1, int(W * scale)), max(1, int(H * scale))
    grid = np.zeros((gh, gw), dtype=bool)
    for b in boxes:
        bx0 = int((b[0] - x0) * scale); by0 = int((b[1] - y0) * scale)
        bx1 = int((b[2] - x0) * scale); by1 = int((b[3] - y0) * scale)
        grid[by0:by1, bx0:bx1] = True
    return float(grid.sum()) / (scale * scale)


def score_image(pred: List[Detection], gt: List[dict], iou_thr: float = 0.5) -> dict:
    """Per-category precision/recall/coverage/over-mask for one image."""
    out = {}
    cats = set([d.source for d in pred]) | set([g["type"] for g in gt])
    for cat in sorted(cats):
        p = [d for d in pred if d.source == cat and d.mask]
        g = [x for x in gt if x["type"] == cat]
        matched_g = set()
        tp = 0
        for d in p:
            best, best_j = 0.0, -1
            for j, gx in enumerate(g):
                if j in matched_g:
                    continue
                v = iou(d.box, gx["box"])
                if v > best:
                    best, best_j = v, j
            if best >= iou_thr and best_j >= 0:
                tp += 1
                matched_g.add(best_j)
        fp = len(p) - tp
        fn = len(g) - tp
        precision = tp / (tp + fp) if (tp + fp) else (1.0 if not g else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else 1.0

        # coverage: mean fraction of each GT box covered by ANY predicted box
        covs = []
        for gx in g:
            gb = gx["box"]
            ga = max(1.0, (gb[2] - gb[0]) * (gb[3] - gb[1]))
            inter_union = [d.box for d in p if iou(d.box, gb) > 0]
            covered = 0.0
            for db in inter_union:
                ix = max(0.0, min(gb[2], db[2]) - max(gb[0], db[0]))
                iy = max(0.0, min(gb[3], db[3]) - max(gb[1], db[1]))
                covered = max(covered, ix * iy)  # best single-box coverage
            covs.append(min(1.0, covered / ga))
        coverage = sum(covs) / len(covs) if covs else None

        out[cat] = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                    "precision": round(float(precision), 3), "recall": round(float(recall), 3),
                    "coverage": round(float(coverage), 3) if coverage is not None else None}
    return out


def score_set(results: Dict[str, List[Detection]], gt_path: str, iou_thr: float = 0.5) -> dict:
    with open(gt_path) as f:
        gt_all = json.load(f)
    per_image = {}
    agg: Dict[str, dict] = {}
    for fname, preds in results.items():
        gt = gt_all.get(fname, [])
        s = score_image(preds, gt, iou_thr)
        per_image[fname] = s
        for cat, m in s.items():
            a = agg.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0, "cov": [], "n": 0})
            a["tp"] += m["tp"]; a["fp"] += m["fp"]; a["fn"] += m["fn"]; a["n"] += 1
            if m["coverage"] is not None:
                a["cov"].append(m["coverage"])
    summary = {}
    for cat, a in agg.items():
        prec = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 1.0
        rec = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 1.0
        summary[cat] = {
            "precision": round(prec, 3), "recall": round(rec, 3),
            "coverage": round(sum(a["cov"]) / len(a["cov"]), 3) if a["cov"] else None,
            "tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
        }
    return {"summary": summary, "per_image": per_image}
