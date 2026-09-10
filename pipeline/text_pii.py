"""Text extraction + PII sensitivity filtering (Phase 4).

Text localization was NOT v4's weakness — ``ai_parse_document`` returned good
pixel boxes (e.g. logo1.jpg masked correctly). v4's text problems were:
  1. Over-masking: when Claude's guardrail blocked a payload, v4 fell back to
     masking ALL text → whole-paragraph blackout.
  2. Tight boxes: colored logo chrome around wordmarks leaked (TAGCYBER/TechVision).

This module keeps ``ai_parse_document`` as the Databricks-native text backend
(works on the serverless GPU deployment target) and fixes both problems:
  - pad text boxes a little so chrome is covered (handled at detection time);
  - a deterministic regex PII pre-filter runs FIRST and always redacts obvious
    PII (emails, phones, URLs, SSNs, card-like numbers). The Claude classifier
    refines the rest; if Claude is unavailable, we fall back to the regex hits
    ONLY (plus an optional mask-all switch) instead of blindly masking everything.

Two masking modes:
  - 'pii_only'  (default): mask only text the regex/Claude flags as sensitive.
  - 'all_text'           : mask every detected text box (full redaction).

This module is import-safe without a Spark session; the Databricks calls are
made lazily so the rest of the pipeline runs locally without auth.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import List, Optional

from PIL import Image

from .boxes import Detection, area_frac, clip_to_image, pad_box

# ── Deterministic PII patterns (always redacted, no model needed) ───────────────
# Patterns are intentionally structure-specific: a loose "any digits" phone or a
# bare 5-digit ZIP fires on years, amounts, and IDs, which (combined with masking)
# is a big source of over-redaction. Phone requires a real 3-3-4 shape; ZIP
# requires a state prefix or the ZIP+4 form (Claude still catches looser addresses).
PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"\b(?:https?://|www\.)[^\s]+", re.I),
    "phone": re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "zip": re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b|\b\d{5}-\d{4}\b"),
}

# ai_parse_document element types that represent non-text graphics.
FIGURE_TYPES = {"figure", "picture", "image"}

# Sign-off context cues. A content-less figure sitting next to one of these (or a
# "Name, Role" caption) is almost certainly a signature — this is how we catch
# illegible handwritten scrawls that OCR can't read into a name (so the name-PII
# path never fires). For legible signatures, ai_parse reads the name and the
# normal PII filter already masks it.
SIGNOFF_CUES = [
    "sincerely", "regards", "yours truly", "signature", "signed", "/s/",
    "approved by", "reviewed by", "prepared by", "authorized", "witness",
    "engagement partner", "partner", "per:", "by:",
]
# "Surname, Role" caption directly under a signature (e.g. "Dara Okafor, Partner")
NAME_ROLE_CAPTION = re.compile(r"^[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3},\s+[A-Z]")

SENSITIVE_TEXT_PROMPT = """\
You are redacting documents for a professional-services firm (consulting, audit, \
tax, advisory) that handles confidential client engagements. Err on the side of \
masking: flag ANY text element that identifies a client, a person, an \
organization, a brand, an engagement, or that is otherwise sensitive. Mask an \
item if it contains ANY of:
- Customer / client names and any customer entity (subsidiaries, affiliates, \
  fund or deal names, ticker symbols)
- Company, organization, or firm names (the firm's own name and third parties)
- Brand names, product names, or brand wordmarks rendered as text \
  (e.g. "AUDI", "Gartner", "Salesforce") — treat a standalone brand word as sensitive
- Employee / person names — partners, staff, signatories, contacts \
  (first, last, or full name), and titles tied to a named person
- Project, engagement, matter, or codename identifiers (e.g. "Project Atlas", \
  engagement numbers, matter IDs)
- Signature blocks and "signed by / approved by" names
- Street addresses, cities, postal / ZIP codes, countries tied to a party
- Phone, fax, email addresses, and website URLs
- Government / financial identifiers: SSN, EIN/TIN, routing/account numbers, \
  IBAN, card numbers, case numbers
- Monetary amounts, fees, or figures tied to a specific client or engagement
- Any other personally identifiable or client-confidential information

Do NOT flag text that identifies no one — generic narrative sentences, section \
headings and titles, quotations, common nouns, generic legal disclaimers, page \
numbers, or dates with no other identifying context — even if they sit near \
sensitive lines. Flag an element only when it actually contains one of the \
identifiers above; a sentence is not sensitive merely because it discusses the \
client's business in general terms."""


def regex_is_sensitive(text: str) -> bool:
    """Deterministic check — true if text contains any obvious PII pattern."""
    for pat in PII_PATTERNS.values():
        if pat.search(text):
            return True
    return False


# ── Text detection via ai_parse_document (Databricks-native) ────────────────────

def parse_document_text_elements(
    spark,
    image_path: str,
    max_area_frac: float = 0.30,
    pad_frac: float = 0.08,
) -> List[Detection]:
    """Run ai_parse_document and return text Detections with padded boxes.

    Requires an active DatabricksSession ``spark`` with serverless access.
    """
    from pyspark.sql.functions import expr

    with open(image_path, "rb") as f:
        binary_image = f.read()
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    df = spark.createDataFrame([(binary_image,)], ["image_data"])
    result = df.select(
        expr("cast(ai_parse_document(image_data) as string)").alias("parsed")
    ).collect()[0]["parsed"]

    parsed = json.loads(result)
    dets: List[Detection] = []
    for elem in parsed.get("document", {}).get("elements", []):
        etype = (elem.get("type", "") or "").lower()
        content = (elem.get("content", "") or "").strip()
        is_figure = etype in FIGURE_TYPES
        # Keep figures even with NO content — an illegible signature scrawl comes
        # back as a content-less figure and must survive for flag_signatures().
        if not content and not is_figure:
            continue
        for b in elem.get("bbox", []):
            coords = b.get("coord", [])
            if len(coords) != 4:
                continue
            x, y, x2, y2 = coords
            box = clip_to_image([x, y, x2, y2], img_w, img_h)
            if area_frac(box, img_w, img_h) > max_area_frac:
                continue
            box = pad_box(box, pad_frac, img_w, img_h)
            dets.append(Detection(box=box, source="text", label=content[:40], score=1.0,
                                  meta={"content": content, "elem_type": etype}))
    return split_into_lines(dets)


def split_into_lines(dets: List[Detection]) -> List[Detection]:
    """Split a multi-line text element into one detection PER LINE.

    ``ai_parse_document`` returns one coarse box per element — a whole paragraph
    or contact block is a single box — so masking the element blacks out generic
    text around one PII token (the main source of over-redaction). For a
    multi-line element we partition its box into equal vertical bands (one per
    content line) so the downstream regex/Claude filter can mask only the
    sensitive lines.

    SAFETY: equal bands only line up when each content line is one *visual* line.
    A long sentence wraps to several visual lines, so an equal split would leave
    part of it (possibly the PII) OUTSIDE its band — under-redaction. We therefore
    split ONLY stacked SHORT lines that cannot wrap (contact blocks, signature /
    address blocks); any element with a long (wrappable) line, a table, or a
    figure is left whole (masked in full — safe). Bands overlap slightly so a
    kept split line is never left half-visible either.
    """
    SHORT_LINE_MAX = 45   # chars; longer lines can wrap → unsafe to band-split
    out: List[Detection] = []
    for d in dets:
        content = d.meta.get("content", d.label) or ""
        etype = d.meta.get("elem_type", "")
        lines = content.split("\n")
        non_empty = [ln for ln in lines if ln.strip()]
        if (d.source != "text" or etype in FIGURE_TYPES
                or content.strip().startswith("<table") or len(non_empty) < 2
                or max(len(ln.strip()) for ln in non_empty) > SHORT_LINE_MAX):
            out.append(d)
            continue
        x1, y1, x2, y2 = d.box
        n = len(lines)
        h = (y2 - y1) / n
        vpad = 0.12 * h
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            sy1 = max(y1, y1 + i * h - vpad)
            sy2 = min(y2, y1 + (i + 1) * h + vpad)
            out.append(Detection(box=[x1, sy1, x2, sy2], source="text",
                                  label=ln.strip()[:40], score=d.score,
                                  meta={"content": ln.strip(), "elem_type": etype, "split": True}))
    return out


# ── PII sensitivity filter ──────────────────────────────────────────────────────

def _get_token(profile: str) -> str:
    result = subprocess.run(
        ["databricks", "auth", "token", "--profile", profile],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["access_token"]


# ── Signature heuristic (no extra model) ────────────────────────────────────────

def _h_overlap_frac(a, b) -> float:
    """Horizontal overlap as a fraction of the narrower box's width."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return ix / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def _v_gap(a, b) -> float:
    """Vertical gap between two boxes (0 if they overlap vertically)."""
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def _is_signoff_cue(text: str) -> bool:
    t = text.lower()
    if any(cue in t for cue in SIGNOFF_CUES):
        return True
    return bool(NAME_ROLE_CAPTION.match(text.strip()))


def flag_signatures(dets: List[Detection], img_h: float = None) -> List[Detection]:
    """Promote ai_parse 'figure' elements in a sign-off zone to source='signature'.

    Closes the gap where a signature is an illegible scrawl: it comes back as a
    content-less figure (no name → not caught by the PII-name path). We mask it
    when it sits horizontally aligned with, and vertically near, an element that
    reads as a sign-off cue or a "Name, Role" caption. Legible signatures still
    get masked by the normal name-PII path; this is purely the fallback.
    """
    # A handwritten signature is WIDE and SHORT. Two shape guards reject the
    # figures that can sit in a sign-off/caption zone but are NOT signatures:
    #   - aspect (w/h) >= 1.8  → excludes headshots (square) & portraits (tall)
    #   - height <= 8% of page → excludes charts/diagrams (wide but tall)
    # Real signatures are ~2-4% of page height and aspect >> 1, so they pass.
    SIG_MIN_ASPECT = 1.8
    if img_h is None:  # estimate page height from the lowest detection
        img_h = max((d.box[3] for d in dets), default=1.0)
    sig_max_h = 0.08 * img_h
    figures = []
    for d in dets:
        if d.meta.get("elem_type") not in FIGURE_TYPES:
            continue
        w = d.box[2] - d.box[0]; h = max(1.0, d.box[3] - d.box[1])
        if w / h >= SIG_MIN_ASPECT and h <= sig_max_h:
            figures.append(d)
    cues = [d for d in dets
            if d.meta.get("elem_type") not in FIGURE_TYPES
            and _is_signoff_cue(d.meta.get("content", d.label) or "")]
    for f in figures:
        fh = f.box[3] - f.box[1]
        for c in cues:
            if _h_overlap_frac(f.box, c.box) > 0.30 and _v_gap(f.box, c.box) < 1.5 * fh:
                f.source = "signature"
                f.label = "signature"
                f.mask = True
                f.meta["signature_cue"] = (c.meta.get("content", c.label) or "")[:40]
                break
    return dets


def filter_sensitive(
    dets: List[Detection],
    mode: str = "pii_only",
    claude_endpoint: Optional[str] = None,
    profile: str = "e2-field-eng-west",
    prompt: str = SENSITIVE_TEXT_PROMPT,
    max_retries: int = 2,
    img_h: float = None,
    on_error: str = "mask_all",
) -> List[Detection]:
    """Decide which text Detections to mask.

    mode='all_text' → mask every text box.
    mode='pii_only' → regex PII always masked; Claude refines the remainder.

    Fail-safe: if the Claude classifier is unavailable after one retry, this
    does NOT silently drop to regex-only (which let brand wordmarks slip
    through). Instead it applies ``on_error``:
      - 'mask_all' (default): mask every text box — never under-mask a
        confidential document when the classifier is down;
      - 'regex_only': mask only deterministic regex PII hits.
    Either way the failure is recorded on each affected detection's
    ``meta['classifier_error']`` so the caller can surface it to the user.
    Sets ``Detection.mask`` accordingly and returns the full list (so the
    overlay can show kept text too).
    """
    # Promote sign-off-zone figures to 'signature' first (catches illegible
    # scrawls that have no readable name). These are then masked regardless of
    # the PII verdict and excluded from the text-PII pass below.
    flag_signatures(dets, img_h=img_h)

    text_dets = [d for d in dets if d.source == "text"]
    if mode == "all_text":
        for d in text_dets:
            d.mask = True
        return dets

    # regex pass — deterministic, always on
    regex_flags = []
    for d in text_dets:
        content = d.meta.get("content", d.label)
        hit = regex_is_sensitive(content)
        d.meta["regex_pii"] = hit
        regex_flags.append(hit)

    # Claude pass — refine the non-regex items. Retries once inside
    # _claude_sensitive_indices; on hard failure it raises and we apply the
    # fail-safe policy (default: mask everything) + record the error.
    claude_indices = None
    classifier_error = None
    if claude_endpoint:
        try:
            claude_indices = _claude_sensitive_indices(
                text_dets, claude_endpoint, profile, prompt, max_retries
            )
        except ClassifierUnavailable as e:
            classifier_error = str(e)

    for i, d in enumerate(text_dets):
        if regex_flags[i]:
            d.mask = True
        elif claude_indices is not None:
            d.mask = i in claude_indices
        elif classifier_error is not None:
            # Fail-safe: classifier down → don't silently under-mask.
            d.mask = (on_error == "mask_all")
            d.meta["classifier_error"] = classifier_error
        else:
            # No endpoint configured at all → regex hits only.
            d.mask = False
    return dets


class ClassifierUnavailable(RuntimeError):
    """Raised when the Claude PII classifier could not be reached/parsed.

    Carries the last underlying error so callers can surface it to the user
    (instead of silently under-masking) and apply a fail-safe policy.
    """


def _claude_sensitive_indices(text_dets, endpoint, profile, prompt, max_retries=2):
    """Return the set of indices Claude flags as sensitive.

    Tries the call, and on failure retries ONCE more (max_retries=2 total). If
    it still fails, raises ClassifierUnavailable with the last error — the
    caller decides the fail-safe (this module no longer silently drops to
    regex-only, which is what let brand wordmarks like "AUDI" slip through).
    """
    import requests

    texts_str = "\n".join(f'{i}: {d.meta.get("content", d.label)}' for i, d in enumerate(text_dets))
    full_prompt = (
        f"{prompt}\n\nText items to review:\n{texts_str}\n\n"
        "Return a JSON array of index numbers (0-based) for items that should be "
        "masked. Return ONLY the JSON array. Example: [0, 2, 5]"
    )
    last_err = "unknown error"
    for attempt in range(max_retries):
        try:
            token = _get_token(profile)
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"messages": [{"role": "user", "content": full_prompt}], "temperature": 0},
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Reasoning models (e.g. claude-sonnet-5) return `content` as a
                # list of blocks [{"type":"reasoning",...},{"type":"text",...}];
                # older models return a plain string. Handle both.
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                raw = content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1].lstrip("json").strip()
                return set(int(i) for i in json.loads(raw) if isinstance(i, int))
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries - 1:
            time.sleep(3 * (attempt + 1))
    raise ClassifierUnavailable(last_err)
