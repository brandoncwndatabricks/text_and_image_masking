"""Apply masks to an image from a list of Detections.

Two mask styles, selectable per-detection-source via ``MASK_STYLE``:
  - 'black' : solid black rectangle (default for PII/text/logos — true redaction)
  - 'blur'  : heavy Gaussian blur (default for faces — less destructive)

Only detections with ``mask=True`` are applied; allowlisted regions (e.g.
Databricks' own logos) carry ``mask=False`` and are left untouched.
"""

from __future__ import annotations

from typing import Dict, List

from PIL import Image, ImageDraw, ImageFilter

from .boxes import Detection

# Default mask style per source. Faces → blur, everything else → black.
MASK_STYLE: Dict[str, str] = {
    "face": "blur",
    "logo": "black",
    "text": "black",
    "signature": "black",
}

# Blur radius as a fraction of the box's smaller side (so big faces blur as
# heavily as small ones, proportionally). Clamped to a sane minimum.
BLUR_FRAC = 0.30
BLUR_MIN_RADIUS = 8


def _blur_region(image: Image.Image, box) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = image.crop((x1, y1, x2, y2))
    radius = max(BLUR_MIN_RADIUS, int(min(x2 - x1, y2 - y1) * BLUR_FRAC))
    region = region.filter(ImageFilter.GaussianBlur(radius=radius))
    image.paste(region, (x1, y1))


def apply_masks(
    pil_image: Image.Image,
    detections: List[Detection],
    style_overrides: Dict[str, str] = None,
) -> Image.Image:
    """Return a new image with all ``mask=True`` detections masked.

    style_overrides: per-source style overrides, e.g. {'face': 'black'}.
    """
    styles = dict(MASK_STYLE)
    if style_overrides:
        styles.update(style_overrides)

    result = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    for d in detections:
        if not d.mask:
            continue
        style = styles.get(d.source, "black")
        if style == "blur":
            _blur_region(result, d.box)
        else:  # black
            x1, y1, x2, y2 = [int(round(v)) for v in d.box]
            draw.rectangle([x1, y1, x2, y2], fill="black")
    return result
