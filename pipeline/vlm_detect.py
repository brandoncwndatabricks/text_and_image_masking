"""VLM-based sensitive-content detection (Claude vision).

Sends the whole image to a multimodal Claude serving endpoint and asks for
bounding boxes of sensitive content — text PII **and** non-text items (ID/tax
documents, payment cards, license plates, barcodes/QR, screens showing data,
signatures). This is *selective* (the model reads and judges) and covers what
the other lanes miss:
  - ``ai_parse_document`` only reads document layout — it extracts nothing from
    photographic scenes (text baked into a photo, a badge, a screen);
  - the face/logo detectors handle only faces and brand marks.

Boxes come back in absolute pixels of the (possibly downscaled) image sent to
the model; we map them back to the original frame, clamp to bounds, and drop
degenerate / whole-image boxes. Reliability was validated (IoU ~0.95–1.0 on a
synthetic objects image; tight boxes on real document PII lines; screens found
in an office scene). On any transport/parse failure this raises so the caller
can fall back to the open-vocab ``SensitiveObjectDetector``.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import List

from PIL import Image

from .boxes import Detection, area_frac, clip_to_image, pad_box
from .text_pii import _get_token   # reuse the CLI OAuth-token minter

SENSITIVE_VLM_PROMPT = """\
You are redacting an image for a professional-services firm. Find EVERY region \
that contains sensitive content, of any of these kinds:
- identity or tax documents (ID card, passport, driver license, W-9, 1099, SSN card)
- payment / bank cards, cheques, account or routing numbers
- license plates, barcodes, QR codes
- computer / phone / TV screens that show data or text
- signatures and stamps / seals
- any TEXT that reveals a person or company name, email, phone, postal address, \
or a government / financial identifier (SSN, EIN, IBAN, account, card, case no.)
Do NOT flag generic decorative graphics, section headings, or boilerplate that \
identifies no one."""


def _extract_text(content) -> str:
    """Claude reasoning models return a list of blocks; older models a string."""
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content or ""


def _rescale_box(b, sw, sh):
    """Map a model box onto the SENT-image pixel frame, tolerating the three
    conventions models use: normalized 0-1, 0-1000 grid, or absolute pixels."""
    mx = max(abs(float(v)) for v in b)
    if mx <= 1.5:                                   # normalized 0-1
        return [b[0] * sw, b[1] * sh, b[2] * sw, b[3] * sh]
    if mx > max(sw, sh) + 4 and mx <= 1000 + 1:     # 0-1000 grid
        return [b[0] / 1000 * sw, b[1] / 1000 * sh, b[2] / 1000 * sw, b[3] / 1000 * sh]
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]   # absolute pixels


class VLMUnavailable(RuntimeError):
    """Raised when the vision endpoint could not be reached/parsed."""


def detect_sensitive_vlm(
    image: Image.Image,
    endpoint: str,
    profile: str = "e2-demo-west",
    max_dim: int = 1600,
    max_area_frac: float = 0.6,
    pad_frac: float = 0.04,
    timeout: int = 90,
    prompt: str = SENSITIVE_VLM_PROMPT,
) -> List[Detection]:
    """Return sensitive-content Detections from a Claude vision endpoint.

    Raises ``VLMUnavailable`` on any transport/parse failure (so the caller can
    fall back to the open-vocab detector). Returns ``[]`` only when the model
    genuinely reports nothing sensitive.
    """
    import requests

    image = image.convert("RGB")
    W, H = image.size
    send = image.copy()
    send.thumbnail((max_dim, max_dim))
    sw, sh = send.size
    sx, sy = W / sw, H / sh                          # sent-frame → original
    buf = io.BytesIO(); send.save(buf, "JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    full_prompt = (
        f"{prompt}\n\nThe image is {sw} pixels wide and {sh} pixels tall. "
        "Return ONLY a JSON array; each item "
        '{"label":"<what it is>","box":[x0,y0,x1,y1]} with coordinates in '
        f"ABSOLUTE PIXELS (x in 0..{sw}, y in 0..{sh}), origin top-left. "
        "No prose, no code fence."
    )
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 1500,
    }
    try:
        token = _get_token(profile)
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        if resp.status_code != 200:
            raise VLMUnavailable(f"HTTP {resp.status_code}: {resp.text[:200]}")
        raw = _extract_text(resp.json()["choices"][0]["message"]["content"]).strip()
    except VLMUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        raise VLMUnavailable(f"{type(e).__name__}: {e}")

    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise VLMUnavailable(f"no JSON array in response: {raw[:160]}")
    try:
        items = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        raise VLMUnavailable(f"bad JSON: {e}")

    dets: List[Detection] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        b = it.get("box")
        if not (isinstance(b, list) and len(b) == 4):
            continue
        try:
            b = _rescale_box(b, sw, sh)
        except Exception:  # noqa: BLE001
            continue
        box = clip_to_image([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], W, H)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue                                 # degenerate
        if area_frac(box, W, H) > max_area_frac:
            continue                                 # whole-image false positive
        box = pad_box(box, pad_frac, W, H)
        dets.append(Detection(
            box=box, source="sensitive",
            label=str(it.get("label", "sensitive"))[:40], score=0.9, mask=True,
            meta={"backend": "vlm"},
        ))
    return dets
