# src_v4 Evaluation — Effectiveness Analysis

> Goal of this doc: honestly assess how well the `src_v4` pipeline masks
> sensitive content (logos, faces, text/PII) in real images, identify the
> concrete failure modes, and pin down their root causes. The companion
> [`PLAN.md`](./PLAN.md) proposes the `src_v5` redesign that addresses them.

---

## 1. What src_v4 does (recap)

A two-phase pipeline (`logo_masking_combined.ipynb`):

| Phase | Engine | Job |
| ----- | ------ | --- |
| **1 — Text** | Databricks `ai_parse_document` → Claude Sonnet sensitivity filter → `apply_masks` | Extract every text element with pixel bboxes, ask Claude which are sensitive, black those out. |
| **2 — Logos** | OWLv2 (`google/owlv2-base-patch16-finetuned`, local, CPU) → NMS → `apply_masks` | Open-vocabulary detect graphical logos on the text-masked image, black those out. |

There is **no face detection** anywhere in the pipeline.

---

## 2. Method

Evaluated by visually diffing original vs. `*_masked` outputs across the three
representative categories the pipeline targets:

- **Photographic + graphical logo:** `audi_image.jpg` (wordmark + grille emblem)
- **Document / wordmark logos:** `logo1.jpg` (15 research-firm logos on white)
- **Slide with embedded photo of people:** `consulting-slides-2.jpg`
- **Customer success-story slide (the real use case):** `db-success-stories-2.jpg`

---

## 3. Findings by failure mode

### 3.1 🔴 CRITICAL — Logo bounding boxes land in the wrong location (OWLv2 non-square bug)

**Evidence — `audi_image.jpg` (1014×1078, near-square but not exactly):**
- The large red **"AUDI" wordmark** → **not masked**.
- The **four-rings grille emblem** → **not masked**.
- Instead a small black box was painted into **empty sky** in the upper-middle of the image — content that needed no masking at all.

This is the single most damaging defect and it matches your report exactly:
the model *detects* a logo, but the mask is *placed in the wrong spot*.

**Root cause — confirmed.** OWLv2's image processor pads every image to a
**square** (zeros added to the right and bottom) before resizing to the model's
input resolution. The model returns box coordinates normalized to that *padded
square*. The pipeline's `detect_visual_logos()` then rescales those coordinates
using `target_sizes=[(img_h, img_w)]` — the **original, non-square** dimensions.
The result: boxes are stretched/offset along the padded axis and land in the
wrong place. On a perfectly square image the padding is zero and the bug is
invisible — which is why some test images "work" and others don't.

This is a long-standing, documented `transformers` issue, not a bug in our code
logic per se — it's a misuse of the post-processing API:
- HF transformers issue [#27705](https://github.com/huggingface/transformers/issues/27705) — "Bug in the code of owlv2 algorithm" (padding not accounted for in post-processing).
- HF transformers issue [#18553](https://github.com/huggingface/transformers/issues/18553) — "OWL-ViT outputs are offset for non-square images".
- HF forum: [Owl-v2 bounding box misalignment problem](https://discuss.huggingface.co/t/owl-v2-bounding-box-misalignment-problem/66181).

**The fix** (detailed in PLAN.md §4.1): rescale against `max(H, W)` (the padded
square side), not the per-axis original dimensions, then clip to image bounds.

### 3.2 🔴 CRITICAL — Faces are never detected

**Evidence — `consulting-slides-2.jpg`:** the central circular photo shows two
people having a meeting. In the masked output their **faces remain fully
visible**. The pipeline has no face model, so any human face in a slide photo,
headshot, or team photo passes through unmasked. For a redaction tool this is a
correctness gap, not just a quality one.

### 3.3 🟠 HIGH — Severe over-masking of graphical content (OWLv2 false positives)

**Evidence — `consulting-slides-2.jpg`:** the entire colored petal/wheel diagram
(8 large colored shapes) was blacked out. None of those shapes are logos — they
are decorative infographic elements. The generic queries (`'company logo'`,
`'brand mark'`, `'product label'`, etc.) at a low `OWL_SCORE_THRESHOLD = 0.20`
cause OWLv2 to fire on any colorful, logo-shaped region.

The `OWL_MAX_AREA_FRAC = 0.08` guard is a blunt instrument: it drops boxes >8%
of the image, which both (a) fails to stop a swarm of medium-sized false
positives and (b) would *also* drop a genuinely large logo. It trades one error
for another rather than solving either.

### 3.4 🟠 HIGH — Text masking over-/under-covers and leaks

**Evidence — `logo1.jpg`:** mostly works (text-based logos get masked) but
**TAGCYBER's blue box leaks** (mask doesn't cover the full mark) and
**"TechVision" peeks through** at the bottom — tight `ai_parse_document` boxes
don't include the logo's graphical/colored background, so colored logo chrome
survives around the masked text.

**Evidence — `db-success-stories-2.jpg`:** paragraph-level over-masking — whole
body paragraphs blacked out while the "CONDÉ NAST" wordmark on the right *is*
covered but inconsistently, and "Confidential and Proprietary" + the Databricks
bug logo at the bottom survive. The Claude filter operates on whatever text
elements `ai_parse_document` returns, and that segmentation is coarse and
non-deterministic.

### 3.5 🟡 MEDIUM — Non-determinism & conservative fallback

Per v4's own README: `ai_parse_document` returns a varying number of elements
across runs (e.g. 2 vs 13 for the same image), and when Claude's guardrail
blocks a business-sensitive payload the pipeline **falls back to masking ALL
text**. That fallback is safe-by-default but produces the paragraph-blackout
behavior seen in 3.4 and makes outputs unpredictable run-to-run.

### 3.6 🟡 MEDIUM — Architecture & operational friction

- **Mask box is the wrong primitive for evaluation.** Hard black rectangles
  make it impossible to tell *detection* errors from *localization* errors
  without re-running — there's no debug overlay showing scores/labels/source.
- **OWLv2 on CPU** is slow and we have no objective accuracy metric — tuning is
  done by eyeballing outputs.
- **No ground truth / eval harness** — every change is judged subjectively.
- **Two auth paths** (Spark serverless + REST token via CLI) add fragility.

---

## 4. Scorecard

| Category | Detection | Localization | Net effectiveness |
| -------- | --------- | ------------ | ----------------- |
| Wordmark / document text logos | Good (`ai_parse_document`) | Fair — tight boxes leak colored chrome | 🟠 Usable, leaks |
| Graphical logos (emblems, badges) | Poor — misses real ones, fires on decor | **Broken on non-square images** | 🔴 Unreliable |
| Faces | **None** | — | 🔴 Missing |
| General PII text (names, addresses) | Good when Claude responds | Fair | 🟠 Non-deterministic |

**Overall:** v4 is a solid *proof of concept* for text-driven redaction on
square, document-like images, but it is **not dependable for the stated goal**
because (1) graphical-logo localization is broken on non-square images, (2)
faces are entirely unhandled, and (3) it swings between leaking and
over-masking with no objective way to tune the balance.

---

## 5. The core insight

> Your observation is correct and the evidence backs it: the pipeline is
> **decent at *finding* problematic content but unreliable at *placing the mask
> in the right location***.

The biggest single win is **not** a better detector — it's fixing the OWLv2
coordinate post-processing so detected boxes land where they belong. After that,
the priorities are: **add face detection**, **replace the blunt area-fraction
false-positive filter with a proper confidence + CLIP-verification gate**, and
**build a debug-overlay + eval harness** so localization can be measured rather
than eyeballed. These are the backbone of [`PLAN.md`](./PLAN.md).
