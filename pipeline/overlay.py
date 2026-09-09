"""Debug overlay renderer.

Draws every Detection as a colored, labelled, semi-transparent box onto a copy
of the image WITHOUT destroying content — so we can see exactly what each model
found and where. This is the tool that separates *detection* errors from
*localization* errors at a glance (the thing v4 made impossible).

Color by source:
  logo → red, face → cyan, text → yellow.
Allowlisted (mask=False) boxes are drawn dashed/green to show "found but kept".
"""

from __future__ import annotations

from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from .boxes import Detection

SOURCE_COLOR: Dict[str, tuple] = {
    "logo": (255, 60, 60),
    "face": (0, 200, 255),
    "text": (255, 210, 0),
    "signature": (200, 80, 255),
    "sensitive": (255, 140, 0),   # VLM / open-vocab sensitive items (cards, IDs, plates, screens…)
}
KEPT_COLOR = (60, 220, 60)  # allowlisted / not masked


def _font(size: int = 16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_overlay(pil_image: Image.Image, detections: List[Detection]) -> Image.Image:
    base = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(max(12, base.width // 70))

    for d in detections:
        color = KEPT_COLOR if not d.mask else SOURCE_COLOR.get(d.source, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]

        # translucent fill + solid border
        draw.rectangle([x1, y1, x2, y2], fill=color + (50,), outline=color + (255,), width=3)

        tag = f"{d.source}:{d.label[:18]} {d.score:.2f}"
        if not d.mask:
            tag = "KEPT " + tag
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color + (230,))
        draw.text((x1 + 3, ly + 2), tag, fill=(0, 0, 0, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")
