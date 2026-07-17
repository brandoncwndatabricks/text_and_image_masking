"""Shared bounding-box utilities for the v5 masking pipeline.

This module owns ALL coordinate handling so the localization logic lives in
exactly one place. Boxes are represented internally as axis-aligned pixel
corners ``[x1, y1, x2, y2]`` (floats) on the ORIGINAL image's coordinate frame.

A ``Detection`` carries the box plus metadata (source detector, label, score)
so the debug overlay and the eval harness can reason about *why* a box exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# ── Core type ──────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One detected region on the original-image coordinate frame.

    box:    [x1, y1, x2, y2] pixel corners (floats).
    source: 'logo' | 'face' | 'text' — drives mask style and overlay color.
    label:  human-readable label (query / OCR string / 'face').
    score:  detector confidence in [0, 1].
    mask:   whether this detection should actually be masked (False = kept,
            e.g. an allowlisted Databricks logo, but still shown in overlay).
    meta:   free-form extras (clip_score, ocr_text, is_pii, ...).
    """

    box: List[float]
    source: str
    label: str = ""
    score: float = 1.0
    mask: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def xywh(self) -> List[float]:
        x1, y1, x2, y2 = self.box
        return [x1, y1, x2 - x1, y2 - y1]

    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ── Coordinate post-processing (the v4 bug fix) ─────────────────────────────────

def square_target_sizes(img_w: int, img_h: int) -> Tuple[int, int]:
    """Target size for OWLv2-style processors that pad to a square.

    OWLv2 pads the image to a ``max(H, W)`` square (bottom/right) before
    inference and returns boxes normalized to that PADDED square. The correct
    rescale target is therefore ``(side, side)`` with ``side = max(H, W)`` —
    NOT the original ``(H, W)``. Using ``(H, W)`` is the v4 bug that pushed
    masks into the wrong location on non-square images.

    Returns ``(side, side)`` suitable for ``post_process_object_detection``'s
    ``target_sizes=[...]`` argument (which expects (height, width)).
    """
    side = max(img_w, img_h)
    return (side, side)


def clip_to_image(box: Sequence[float], img_w: int, img_h: int) -> List[float]:
    """Clamp an [x1,y1,x2,y2] box to the image bounds."""
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), img_w)
    y1 = min(max(0.0, y1), img_h)
    x2 = min(max(0.0, x2), img_w)
    y2 = min(max(0.0, y2), img_h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def pad_box(box: Sequence[float], frac: float, img_w: int, img_h: int) -> List[float]:
    """Expand a box by ``frac`` of its size on each side, clipped to image.

    frac=0.15 → grow ~15% (e.g. faces, to cover hairline/chin; wordmarks, to
    cover colored logo chrome around the text that tight OCR boxes miss).
    """
    if frac <= 0:
        return clip_to_image(box, img_w, img_h)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * frac, h * frac
    return clip_to_image([x1 - dx, y1 - dy, x2 + dx, y2 + dy], img_w, img_h)


# ── IoU / NMS / merge ───────────────────────────────────────────────────────────

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(dets: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
    """Greedy non-maximum suppression over a list of Detections (by score)."""
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    keep: List[int] = []
    suppressed = [False] * len(dets)
    for idx, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order[idx + 1:]:
            if not suppressed[j] and iou(dets[i].box, dets[j].box) > iou_threshold:
                suppressed[j] = True
    return [dets[i] for i in keep]


def merge_sources(
    detections: List[Detection],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    """Cross-source dedupe: drop a box that heavily overlaps a higher-scoring
    box of the SAME source. Boxes from different sources (logo vs face vs text)
    are intentionally kept even if they overlap — they mask for different
    reasons. NMS is applied within each source.
    """
    by_source: dict = {}
    for d in detections:
        by_source.setdefault(d.source, []).append(d)
    out: List[Detection] = []
    for src, group in by_source.items():
        out.extend(nms(group, iou_threshold))
    return out


def area_frac(box: Sequence[float], img_w: int, img_h: int) -> float:
    x1, y1, x2, y2 = box
    return (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / float(img_w * img_h)
