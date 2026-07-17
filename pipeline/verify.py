"""CLIP-based logo verification gate + Databricks brand allowlist.

Grounding DINO is recall-oriented: it fires on decorative icons, chart glyphs,
and abstract shapes as well as true brand logos. CLIP has no spatial head so it
cannot fix box *position*, but it is an excellent *classifier* of a cropped
region — so we use it purely as a precision filter on top of Grounding DINO's
(already correctly localized) boxes.

Two jobs:
  1. is_logo(crop)        — keep only crops that read as a real brand logo,
                            not a decorative icon / photo / chart element.
  2. is_databricks(crop)  — per the project decision, Databricks' OWN logos are
                            KEPT (not masked). Flag those so the pipeline sets
                            ``Detection.mask = False`` (still shown in overlay).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import torch
from PIL import Image

from .boxes import Detection
from .detectors import pick_device

_REFS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refs"
)

# Prompt sets. CLIP picks the best-matching caption per crop; we read the
# argmax / relative probabilities to decide.
LOGO_POSITIVE = [
    "a company brand logo",
    "a corporate wordmark or trademark",
]
LOGO_NEGATIVE = [
    "a decorative icon or pictogram",
    "an abstract colored shape",
    "a photograph of people or objects",
    "a chart, diagram, or graph element",
    "plain background",
]

class ClipGate:
    """CLIP gate. Two distinct mechanisms by design:

    - is_logo()       : TEXT-prompt zero-shot (logo vs icon/photo/chart). Robust
                        because the classes are visually generic.
    - is_databricks() : IMAGE-to-IMAGE cosine similarity against reference logos
                        in ``refs/``. Text-prompt brand matching proved fragile
                        (it wrongly tagged the "Conde Nast" serif title and an
                        Audi emblem as Databricks); reference-image similarity is
                        far more reliable for a SPECIFIC brand.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = None,
        logo_min_prob: float = 0.55,
        databricks_min_sim: float = 0.80,
        refs_dir: str = _REFS_DIR,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or pick_device()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.logo_min_prob = logo_min_prob
        self.databricks_min_sim = databricks_min_sim
        self.ref_embeds = self._load_reference_embeds(refs_dir)

    @torch.no_grad()
    def _embed_image(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)
        # transformers >=5 wraps the (already projected) CLIP image embedding in
        # a model-output object whose pooler_output is the joint-space vector.
        # Older versions return the tensor directly.
        if not isinstance(feat, torch.Tensor):
            feat = feat.pooler_output
        return torch.nn.functional.normalize(feat, dim=-1)  # [1, D]

    def _load_reference_embeds(self, refs_dir: str) -> Optional[torch.Tensor]:
        if not os.path.isdir(refs_dir):
            return None
        paths = sorted(
            sum([glob.glob(os.path.join(refs_dir, e))
                 for e in ("*.jpg", "*.jpeg", "*.png")], [])
        )
        if not paths:
            return None
        embeds = [self._embed_image(Image.open(p)) for p in paths]
        return torch.cat(embeds, dim=0)  # [N, D], L2-normalized

    @torch.no_grad()
    def _probs(self, crop: Image.Image, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor(
            text=prompts, images=crop, return_tensors="pt", padding=True
        ).to(self.device)
        logits = self.model(**inputs).logits_per_image  # [1, len(prompts)]
        return logits.softmax(dim=-1)[0]

    def is_logo(self, crop: Image.Image) -> float:
        """Return P(logo) = summed prob mass of positive captions vs negatives."""
        prompts = LOGO_POSITIVE + LOGO_NEGATIVE
        probs = self._probs(crop, prompts)
        return float(probs[: len(LOGO_POSITIVE)].sum())

    def is_databricks(self, crop: Image.Image) -> float:
        """Max cosine similarity of the crop to any reference Databricks logo."""
        if self.ref_embeds is None:
            return 0.0
        emb = self._embed_image(crop)               # [1, D]
        sims = (emb @ self.ref_embeds.T).squeeze(0)  # [N]
        return float(sims.max())

    def filter_logos(self, image: Image.Image, dets: List[Detection]) -> List[Detection]:
        """Verify logo detections. Mutates each detection's meta with clip
        scores; drops non-logos; marks Databricks logos as keep (mask=False).
        Returns the surviving detections (Databricks ones included, but with
        mask=False so the overlay shows them as KEPT).
        """
        image = image.convert("RGB")
        out: List[Detection] = []
        for d in dets:
            if d.source != "logo":
                out.append(d)
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in d.box]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            p_logo = self.is_logo(crop)
            d.meta["clip_logo"] = round(p_logo, 3)
            if p_logo < self.logo_min_prob:
                continue  # decorative icon / photo / chart — not a brand logo
            p_db = self.is_databricks(crop)
            d.meta["clip_databricks"] = round(p_db, 3)
            if p_db >= self.databricks_min_sim:
                d.mask = False  # keep Databricks' own logo
                d.label = "databricks(kept)"
            out.append(d)
        return out
