# src_v5 — Redesign Plan

> Companion to [`EVALUATION.md`](./EVALUATION.md). This plan turns the v4
> failure analysis into a concrete v5 architecture: which Hugging Face models
> to use for each content type, how to fix the bounding-box localization that
> is v4's central weakness, and how to make the whole thing *measurable*.

**Target environment (confirmed):** Databricks **serverless GPU (A10)** for
inference; **internal use** — AGPL / non-commercial-research model weights are
acceptable. Models should still be able to fall back to CPU for local dev.

---

## 1. Design principles

1. **Localization is the product.** v4 could find things but put boxes in the
   wrong place. Every model choice and the entire eval harness are organized
   around *box accuracy*, not just detection recall.
2. **One specialist per content type.** Stop forcing a single open-vocab
   detector to do logos + decoration + everything. Use a purpose-built model
   per category (faces, text, logos) and merge their boxes.
3. **Measure, don't eyeball.** Ship a debug-overlay renderer and a small
   labelled eval set from day one so every threshold change is scored.
4. **Verify before masking.** A detection becomes a mask only after a
   confidence gate and (for logos) a CLIP verification pass — this kills the
   "blacked-out the whole infographic" failure.
5. **Fail safe but visible.** Keep a conservative fallback, but log/flag it so
   we know when an image was over-masked due to a guardrail block.

---

## 2. Proposed architecture

```text
                          ┌─────────────────────────────────────┐
   Original image ───────►│  Pad-to-square preprocessor (shared) │
                          └───────────────┬─────────────────────┘
            ┌──────────────────────────────┼──────────────────────────────┐
            ▼                              ▼                              ▼
   ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
   │  TEXT detector  │          │  FACE detector  │          │  LOGO detector  │
   │  PaddleOCR v5   │          │  RetinaFace /   │          │  Grounding DINO │
   │  (word/line box)│          │  YOLO-face      │          │  + CLIP verify  │
   └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
            │ text boxes + strings        │ face boxes                 │ logo boxes
            ▼                            │                            │
   ┌─────────────────┐                   │                            │
   │ Claude / regex   │                  │                            │
   │ PII sensitivity  │                  │                            │
   │ filter (on text) │                  │                            │
   └────────┬────────┘                   │                            │
            └───────────────┬────────────┴───────────────┬────────────┘
                            ▼                             ▼
                  ┌───────────────────────────────────────────┐
                  │  Box post-process: correct rescale, clip,   │
                  │  pad, merge/NMS across all sources          │
                  └───────────────────┬─────────────────────────┘
                          ┌───────────┴───────────┐
                          ▼                       ▼
               ┌────────────────────┐   ┌────────────────────┐
               │  DEBUG OVERLAY      │   │  apply_masks        │
               │  (colored boxes +   │   │  (black / blur)     │
               │   labels + scores)  │   │  → *_masked.*       │
               └────────────────────┘   └────────────────────┘
```

Key change vs v4: **three parallel specialist detectors**, a **shared correct
coordinate post-processing stage**, a **CLIP verification gate on logos**, and a
**debug-overlay branch** that renders detections *without* destroying the image
so we can see exactly what each model found and where.

---

## 3. Model selection (per content type)

All picks are runnable on the A10 and have CPU fallbacks. License column notes
the constraint even though internal use clears them — recorded for future
reuse / productization.

### 3.1 Logos — **Grounding DINO (base)**, replacing OWLv2

| | |
| --- | --- |
| **Repo** | `IDEA-Research/grounding-dino-base` (A/B vs `openmmlab-community/mm_grounding_dino_large_all`) |
| **Why** | Best open-weight open-vocab detector; **returns pixel-accurate corner boxes** through `post_process_grounded_object_detection`. Apache-2.0. Far better small-object localization than OWLv2-base, and the post-processing is padding-aware (unlike OWLv2). |
| **HW** | ~233M params; comfortable on A10. CPU works but slow. ONNX tiny build exists for CPU dev. |
| **Notes** | Prompt with a *short* list of logo phrases (`"logo. brand logo. company emblem."`), not 18 queries. Add **image tiling** for tiny corner logos. CLIP rerank (below) filters false positives. |

**Why not OWLv2:** its non-square coordinate bug is exactly v4's central defect
and the post-processing API is error-prone. Grounding DINO's processor handles
padding correctly when given `target_sizes=(H, W)`.

**Why not YOLO-World / dedicated YOLO logo models:** AGPL (acceptable internally
but lower absolute accuracy) and trained on fixed brand sets. Keep as a possible
speed-optimized fallback, not the primary.

**Logo false-positive gate (fixes the "whole infographic blacked out" bug):**
Run a CLIP pass (`openai/clip-vit-base-patch32` or SigLIP) on each candidate
crop, scoring it against `"a brand logo"` vs `"a decorative shape / icon / photo"`.
Only keep crops that score as a logo. CLIP has no spatial head so it does **not**
fix box *position* — it is purely a precision filter on top of Grounding DINO's
boxes.

### 3.2 Faces — **RetinaFace** (add the missing capability)

| | |
| --- | --- |
| **Repo / pkg** | `retina-face` (PyPI, serengil) — MIT, turnkey. Alt: `insightface` **SCRFD** for best tiny-face accuracy; YOLO-face (AGPL) for speed. |
| **Why** | Best accuracy-to-effort balance; `RetinaFace.detect_faces(img)` → pixel boxes + 5-pt landmarks. MobileNet backbone runs on CPU; ResNet-50 on A10 for max accuracy. |
| **Notes** | Internal use clears SCRFD's non-commercial weights — if RetinaFace recall on small/occluded faces is insufficient, swap to **SCRFD** (`insightface` `FaceAnalysis`), which leads WIDER FACE hard set. Mask faces with a slightly expanded box (faces benefit from ~15% padding to cover hairline/chin). |

### 3.3 Text / PII — **PaddleOCR PP-OCRv5**, replacing `ai_parse_document` (with fallback)

| | |
| --- | --- |
| **Repo / pkg** | `paddleocr` / `PaddlePaddle/PP-OCRv5*` — Apache-2.0. |
| **Why** | Deterministic (fixes v4's run-to-run variance), returns **precise word/line quadrilateral boxes**, runs locally (no Spark/REST round-trip), 100+ languages. Mobile model on CPU, server model on A10. |
| **PII filtering** | Keep a Claude sensitivity classifier on the extracted strings (it worked well in v4), but add a **deterministic regex/Presidio pre-filter** for emails/phones/URLs/addresses so the pipeline still redacts obvious PII when Claude's guardrail blocks the payload — replacing the blunt "mask everything" fallback. |
| **Notes** | Convert PaddleOCR polygons → axis-aligned rect + small pad so colored logo chrome around wordmarks (the TAGCYBER/TechVision leak) is covered. Keep `ai_parse_document` available as an alternate text backend behind a flag for comparison. |

### Summary table

| Category | v4 | v5 | Primary win |
| -------- | --- | --- | --- |
| Logos | OWLv2-base (buggy coords) | Grounding DINO base + CLIP gate + tiling | Correct boxes, fewer false positives |
| Faces | — none — | RetinaFace (→ SCRFD if needed) | New capability |
| Text/PII | ai_parse_document + Claude | PaddleOCR v5 + Claude + regex/Presidio fallback | Deterministic, local, leak-resistant |

---

## 4. The localization fixes (the heart of v5)

### 4.1 Correct OWLv2-style / open-vocab coordinate rescale

If we keep *any* OWLv2 code path, rescale against the **padded square side**,
not the original non-square dims:

```python
# OWLv2 normalizes boxes to a square of side max(H, W) (padded bottom/right).
W, H = image.size                 # PIL (width, height)
side = max(H, W)
target_sizes = torch.tensor([(side, side)])      # square — NOT (H, W)
results = processor.post_process_object_detection(
    outputs, threshold=thr, target_sizes=target_sizes)[0]
boxes = results["boxes"].clamp(min=0)
boxes[:, 0::2] = boxes[:, 0::2].clamp(max=W)      # clip to real image
boxes[:, 1::2] = boxes[:, 1::2].clamp(max=H)
```

### 4.2 Correct Grounding DINO post-process (primary path)

```python
results = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids,
    threshold=0.35, text_threshold=0.30,
    target_sizes=[(image.height, image.width)],   # (H, W) order — critical
)
# results[0]["boxes"] are pixel (x0, y0, x1, y1) — ready to mask.
```

The two classic Grounding DINO mistakes to avoid: passing `(W, H)` instead of
`(H, W)`, and bypassing the processor (raw outputs are normalized **cx,cy,w,h**
center format, not corner pixels).

### 4.3 Shared box pipeline

One module owns: rescale → clip to bounds → per-type padding (faces +15%,
wordmarks +small) → cross-source merge + NMS → optional mask-style (black box
vs. Gaussian blur). All three detectors feed into it so coordinate handling
lives in exactly one place.

Sources for the coordinate fixes: HF transformers
[#27705](https://github.com/huggingface/transformers/issues/27705),
[#18553](https://github.com/huggingface/transformers/issues/18553);
[Grounding DINO HF docs](https://huggingface.co/docs/transformers/main/model_doc/grounding-dino);
[GroundingDINO issue #259](https://github.com/IDEA-Research/GroundingDINO/issues/259).

---

## 5. Evaluation harness (so we can prove improvement)

Without this, v5 is just opinion. Build:

1. **Labelled eval set** — hand-annotate ground-truth boxes for ~15–20 images
   already in `images/` (logos, faces, PII text), stored as a simple JSON
   (`eval/ground_truth.json`: filename → list of `{type, bbox}`).
2. **Metrics** — per category: **IoU-based precision/recall @ IoU 0.5**, plus a
   **"coverage"** metric (does the mask fully cover the GT box?) and an
   **"over-mask"** metric (fraction of masked area that wasn't sensitive). This
   directly quantifies the leak-vs-overmask tradeoff from EVALUATION §3.3–3.4.
3. **Debug overlay** — render every detection as a *colored, labelled,
   semi-transparent* box (color by source: logo/face/text; label = score +
   query) onto a copy of the image, saved as `*_debug.*`. This makes
   localization vs. detection errors visible at a glance.
4. **Regression run** — a notebook cell that scores the whole eval set and
   prints a before/after table vs. v4's outputs.

---

## 6. Proposed deliverables in `src_v5/`

```text
src_v5/
├── EVALUATION.md              ✅ done — v4 failure analysis
├── PLAN.md                    ✅ done — this file
├── masking_pipeline.ipynb     ▢ v5 pipeline (3 detectors + shared box stage + overlay)
├── pipeline/                  ▢ optional: extracted .py modules
│   ├── detectors.py           ▢ logo (GDINO+CLIP), face (RetinaFace), text (PaddleOCR)
│   ├── boxes.py               ▢ shared rescale/clip/pad/merge/NMS
│   ├── pii_filter.py          ▢ Claude + regex/Presidio sensitivity gate
│   └── overlay.py             ▢ debug overlay renderer
├── eval/
│   ├── ground_truth.json      ▢ hand-labelled boxes for ~15-20 images
│   └── score.py               ▢ IoU precision/recall/coverage/over-mask
└── requirements.txt           ▢ transformers, paddleocr, retina-face, torch, clip
```

---

## 7. Phased implementation roadmap

| Phase | Scope | Exit criterion |
| ----- | ----- | -------------- |
| **0 — Eval foundation** | Labelled eval set + debug overlay + scoring on v4 outputs | We have baseline numbers for v4 |
| **1 — Fix localization** | Swap OWLv2 → Grounding DINO with correct post-process; shared box module | Audi wordmark + emblem masked correctly; box-IoU up vs. v4 |
| **2 — Add faces** | RetinaFace integration | Faces in consulting/success-story slides masked |
| **3 — Logo precision** | CLIP verification gate + tiling | Infographic over-masking eliminated; logo precision up |
| **4 — Text/PII** | PaddleOCR v5 + regex/Presidio fallback | Deterministic text boxes; no more whole-paragraph blackout |
| **5 — Tune & document** | Threshold sweep against eval harness; final README | v5 beats v4 on every category metric |

---

## 8. Open questions / decisions for you

1. **Mask style** — hard black box (current) or **Gaussian blur**? Blur is less
   destructive and increasingly expected for "redacted but still readable as a
   layout." Recommend offering both via a flag; default to black for true PII,
   blur for faces. *(Your call.)*
2. **Logo scope** — mask *all* logos, or only third-party/competitor logos
   (keeping Databricks' own)? v4 masks everything. This changes the CLIP prompts
   and possibly adds a brand-allowlist.
3. **Deployment shape** — stay a notebook, or wrap as a Databricks job / model
   serving endpoint for batch redaction? Affects how much we extract into `.py`
   modules in Phase 1.
4. **Eval labelling** — happy for me to auto-propose ground-truth boxes (run the
   new detectors, you correct them) rather than hand-labelling from scratch?

---

## 9. TL;DR

v4's weakness is **box placement**, and the single highest-impact fix is
correcting the open-vocab detector's coordinate post-processing — likely
recovering most of the Audi-style failures on its own. On top of that, v5:
swaps OWLv2 → **Grounding DINO** (accurate boxes), **adds RetinaFace** for the
entirely-missing face category, swaps `ai_parse_document` → **PaddleOCR v5** for
deterministic local text boxes, adds a **CLIP verification gate** to stop
over-masking, and — critically — ships a **debug overlay + IoU eval harness** so
every change is measured instead of eyeballed.
