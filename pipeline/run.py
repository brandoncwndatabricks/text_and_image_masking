"""End-to-end orchestration for the v5 masking pipeline.

Ties the specialist detectors together behind one ``MaskingPipeline`` object:

    logos  : Grounding DINO  → CLIP verify (drop icons, keep Databricks)
    faces  : YuNet           → pad
    text   : ai_parse_document (optional, needs Spark) → regex/Claude PII

    → merge across sources → debug overlay + masked output.

Each stage is optional so the pipeline can run logo+face locally (no Spark/auth)
and add text on the Databricks deployment. Coordinate handling lives entirely in
``boxes`` / the detectors, so everything here works on the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from .boxes import Detection, merge_sources
from .detectors import FaceDetector, LogoDetector, TextDetector
from .masking import apply_masks
from .overlay import render_overlay
from .verify import ClipGate


@dataclass
class PipelineConfig:
    do_logos: bool = True
    do_faces: bool = True
    do_text: bool = False          # needs Spark/ai_parse_document
    do_sensitive: bool = False     # generic sensitive items + text baked into images
    sensitive_backend: str = "vlm" # 'vlm' (Claude vision, selective) | 'objects' (GDINO open-vocab)
    use_clip_gate: bool = True
    text_mode: str = "pii_only"    # 'pii_only' | 'all_text'
    logo_box_threshold: float = 0.25
    logo_text_threshold: float = 0.20
    face_pad_frac: float = 0.15
    merge_iou: float = 0.5


class MaskingPipeline:
    def __init__(self, config: PipelineConfig = None, spark=None,
                 claude_endpoint: Optional[str] = None, profile: str = "e2-demo-west"):
        self.cfg = config or PipelineConfig()
        self.spark = spark
        self.claude_endpoint = claude_endpoint
        self.profile = profile          # used to mint the token for the Claude REST call
        self._logo = self._face = self._text = self._clip = self._sensitive_obj = None

    # lazy loaders so we only pay for the detectors we use
    @property
    def logo(self):
        if self._logo is None:
            self._logo = LogoDetector(
                box_threshold=self.cfg.logo_box_threshold,
                text_threshold=self.cfg.logo_text_threshold,
            )
        return self._logo

    @property
    def face(self):
        if self._face is None:
            self._face = FaceDetector(pad_frac=self.cfg.face_pad_frac)
        return self._face

    @property
    def text(self):
        if self._text is None:
            self._text = TextDetector()
        return self._text

    @property
    def clip(self):
        if self._clip is None:
            self._clip = ClipGate()
        return self._clip

    @property
    def sensitive_obj(self):
        # open-vocab Grounding DINO fallback for the sensitive-items phase
        if self._sensitive_obj is None:
            from .detectors import SensitiveObjectDetector
            self._sensitive_obj = SensitiveObjectDetector(
                box_threshold=self.cfg.logo_box_threshold,
                text_threshold=self.cfg.logo_text_threshold,
            )
        return self._sensitive_obj

    def detect(self, image: Image.Image, image_path: str = None) -> List[Detection]:
        image = image.convert("RGB")
        dets: List[Detection] = []

        if self.cfg.do_logos:
            logo_dets = self.logo.detect(image)
            if self.cfg.use_clip_gate:
                logo_dets = self.clip.filter_logos(image, logo_dets)
            dets += logo_dets

        if self.cfg.do_faces:
            dets += self.face.detect(image)

        if self.cfg.do_text:
            from . import text_pii
            if self.spark is not None and image_path is not None:
                text_dets = text_pii.parse_document_text_elements(self.spark, image_path)
            else:
                text_dets = self.text.detect(image)  # local DBNet fallback
            text_dets = text_pii.filter_sensitive(
                text_dets, mode=self.cfg.text_mode,
                claude_endpoint=self.claude_endpoint,
                profile=self.profile,
                img_h=image.size[1],
            )
            dets += text_dets

        if self.cfg.do_sensitive:
            # Primary: Claude vision (selective — covers text baked into images +
            # generic sensitive objects). Fallback: open-vocab Grounding DINO.
            sens = None
            if self.cfg.sensitive_backend == "vlm" and self.claude_endpoint:
                try:
                    from . import vlm_detect
                    sens = vlm_detect.detect_sensitive_vlm(
                        image, self.claude_endpoint, self.profile)
                except vlm_detect.VLMUnavailable:
                    sens = None  # fall through to the open-vocab detector
            if sens is None:
                sens = self.sensitive_obj.detect(image)
            dets += sens

        return merge_sources(dets, self.cfg.merge_iou)

    def run(self, image_path: str):
        """Return (detections, overlay_image, masked_image)."""
        image = Image.open(image_path).convert("RGB")
        dets = self.detect(image, image_path)
        return dets, render_overlay(image, dets), apply_masks(image, dets)
