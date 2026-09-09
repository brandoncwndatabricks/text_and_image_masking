"""Specialist detectors for the v5 masking pipeline.

Each detector returns a list of ``boxes.Detection`` on the ORIGINAL image
coordinate frame, with coordinates already correctly rescaled — the v4
localization bug is fixed at the source here, not patched downstream.

Detectors:
  - LogoDetector   : Grounding DINO (open-vocab) + optional CLIP verification.
  - FaceDetector   : RetinaFace (added in Phase 2).
  - TextDetector   : PaddleOCR PP-OCRv5 (added in Phase 4).

Models are loaded lazily so importing this module is cheap and a pipeline can
use only the detectors it needs.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .boxes import Detection, area_frac, clip_to_image, nms, pad_box


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Logos: Grounding DINO ───────────────────────────────────────────────────────

# Grounding DINO expects a SHORT prompt of lowercase phrases separated by " . ".
# Keep it tight — long query lists hurt precision. CLIP verification (Phase 3)
# filters the false positives instead of relying on many narrow prompts.
DEFAULT_LOGO_PROMPT = "logo . brand logo . company emblem . brand mark"


class LogoDetector:
    """Open-vocabulary logo detector built on Grounding DINO.

    Returns pixel-accurate [x1,y1,x2,y2] boxes. Grounding DINO's
    ``post_process_grounded_object_detection`` is padding-aware as long as
    ``target_sizes`` is given in (height, width) order — this is the correct
    counterpart to OWLv2's square-padding bug.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        max_area_frac: float = 0.35,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or pick_device()
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_area_frac = max_area_frac
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_LOGO_PROMPT,
        iou_threshold: float = 0.4,
    ) -> List[Detection]:
        image = image.convert("RGB")
        img_w, img_h = image.size

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # (H, W) order is REQUIRED here — the processor rescales padding-aware.
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

        dets: List[Detection] = []
        labels = results.get("text_labels", results.get("labels", []))
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            box = [float(v) for v in box.tolist()]
            box = clip_to_image(box, img_w, img_h)
            if area_frac(box, img_w, img_h) > self.max_area_frac:
                continue  # whole-object false positive (e.g. entire truck body)
            dets.append(Detection(
                box=box,
                source="logo",
                label=str(label),
                score=float(score),
            ))
        return nms(dets, iou_threshold)


# ── Generic sensitive objects: Grounding DINO open-vocab ─────────────────────────
# Open-vocabulary fallback for "obviously sensitive" items beyond faces/logos.
# Same GDINO weights as LogoDetector, different prompt; NOT gated by the CLIP logo
# verifier (these aren't brand marks). Noisier than face detection — tune the
# prompt/threshold per deployment. Used as the fallback when the VLM backend
# (vlm_detect) is unavailable.
SENSITIVE_OBJECT_PROMPT = (
    "identity card . passport . driver license . license plate . credit card . "
    "bank card . barcode . qr code . name badge . computer screen . "
    "phone screen . cheque"
)


class SensitiveObjectDetector(LogoDetector):
    """Detect obviously-sensitive objects via open-vocabulary Grounding DINO.

    Reuses the LogoDetector GDINO model; returns Detections with
    ``source='sensitive'`` so they mask (black) and skip the logo CLIP gate.
    A higher ``max_area_frac`` default (an ID scan can fill most of the frame).
    """

    def __init__(self, *args, max_area_frac: float = 0.9, **kwargs):
        super().__init__(*args, max_area_frac=max_area_frac, **kwargs)

    def detect(self, image: Image.Image, prompt: str = SENSITIVE_OBJECT_PROMPT,
               iou_threshold: float = 0.4) -> List[Detection]:
        dets = super().detect(image, prompt=prompt, iou_threshold=iou_threshold)
        for d in dets:
            d.source = "sensitive"
        return dets


# ── Faces: YuNet (OpenCV DNN, no extra deps) ────────────────────────────────────

_DEFAULT_YUNET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_detection_yunet_2023mar.onnx",
)


class FaceDetector:
    """Face detector using OpenCV's bundled YuNet model.

    Runs through OpenCV's own DNN backend (no onnxruntime / TensorFlow needed),
    is fast on CPU, and returns pixel-accurate boxes. Boxes are padded ~15% so
    masks cover hairline/chin. Chosen over RetinaFace/SCRFD because PyPI is not
    reachable in this environment but the YuNet ONNX is a tiny GitHub download.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_YUNET,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        pad_frac: float = 0.15,
    ):
        import cv2

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. Download from the "
                "OpenCV Zoo (face_detection_yunet_2023mar.onnx)."
            )
        self.cv2 = cv2
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.pad_frac = pad_frac
        # input size is set per-image in detect()
        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )

    def detect(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")
        img_w, img_h = image.size
        bgr = self.cv2.cvtColor(np.array(image), self.cv2.COLOR_RGB2BGR)
        self._det.setInputSize((img_w, img_h))
        _, faces = self._det.detect(bgr)

        dets: List[Detection] = []
        if faces is None:
            return dets
        for f in faces:
            x, y, w, h = f[0], f[1], f[2], f[3]
            score = float(f[-1])
            box = clip_to_image([x, y, x + w, y + h], img_w, img_h)
            box = pad_box(box, self.pad_frac, img_w, img_h)
            dets.append(Detection(box=box, source="face", label="face", score=score))
        return dets


# ── Text: DBNet (OpenCV DNN, no extra deps) ──────────────────────────────────────

_DEFAULT_DBNET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "text_detection_DB_TD500_resnet18_2021sep.onnx",
)


class TextDetector:
    """Scene/document text detector using OpenCV's DBNet model.

    Same detection architecture PaddleOCR uses; returns rotated quad boxes which
    we convert to axis-aligned rects (+ small pad so colored logo chrome around
    wordmarks is covered — fixes v4's TAGCYBER/TechVision leak). Runs through
    OpenCV's DNN backend (no paddle/onnxruntime), chosen because PyPI is not
    reachable here. Detection only — pair with a recogniser/Claude for PII
    filtering, or mask all detected text for full redaction.

    The model input side must be a multiple of 32; larger = better recall on
    small text but slower.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_DBNET,
        bin_threshold: float = 0.3,
        poly_threshold: float = 0.5,
        input_side: int = 736,
        pad_frac: float = 0.08,
    ):
        import cv2

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DBNet model not found at {model_path}. Download from the "
                "OpenCV Zoo (text_detection_DB_*.onnx)."
            )
        self.cv2 = cv2
        self.model_path = model_path
        self.input_side = (input_side // 32) * 32
        self.pad_frac = pad_frac
        self._det = cv2.dnn_TextDetectionModel_DB(model_path)
        self._det.setBinaryThreshold(bin_threshold)
        self._det.setPolygonThreshold(poly_threshold)
        self._det.setMaxCandidates(400)
        self._det.setUnclipRatio(2.0)
        # DBNet normalisation (ImageNet mean, BGR), per OpenCV Zoo defaults.
        self._det.setInputParams(
            scale=1.0 / 255.0,
            size=(self.input_side, self.input_side),
            mean=(122.67891434, 116.66876762, 104.00698793),
        )

    def detect(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")
        img_w, img_h = image.size
        bgr = self.cv2.cvtColor(np.array(image), self.cv2.COLOR_RGB2BGR)
        quads, scores = self._det.detect(bgr)

        dets: List[Detection] = []
        if quads is None:
            return dets
        for quad, score in zip(quads, scores):
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            box = clip_to_image([min(xs), min(ys), max(xs), max(ys)], img_w, img_h)
            box = pad_box(box, self.pad_frac, img_w, img_h)
            dets.append(Detection(box=box, source="text", label="text", score=float(score)))
        return nms(dets, 0.5)
