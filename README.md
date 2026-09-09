# Text & Image Masking

A redaction pipeline that automatically finds and masks **sensitive content
embedded in images and documents** — company **logos**, human **faces**, and
sensitive **text / PII** — and writes back a redacted copy plus a non-destructive
debug overlay showing exactly what was found and where.

It's built for consulting / advisory / audit / tax-style material (engagement
letters, financial statements, slide decks, business cards, scanned PDFs) where a
single page often mixes all three: a client logo in the header, headshots in a
team slide, and names / emails / account numbers in the body.

## What it looks like

Each image is run through the pipeline and produces a redacted copy. Below,
**before** is the original and **after** is what the pipeline emits.

**All three phases at once — advisory report.** The "Meridian" logo is blacked
out (logo phase), the executive headshot is Gaussian-blurred (face phase), and
the direct phone line, personal-assistant email, and the CEO name/email caption
are masked (text/PII phase) — while the title and body narrative are preserved.

![Three-phase report before/after](docs/img/three_phase_report_before_after.jpg)

**Team slide — 4 faces blurred.** Every headshot is detected and blurred while
the layout, names, and roles are preserved.

![Team slide before/after](docs/img/slide_engagement_team_before_after.jpg)

**Business card — logo masked.**

![Business card before/after](docs/img/business_card_1_before_after.jpg)

**Financial statement — logo masked.**

![Financial statement before/after](docs/img/financial_statement_1_before_after.jpg)

> Regenerate these (and overlays for the whole sample set) by running the
> pipeline over `test_set/` — see [Running](#running).

## How it works

Three detectors run over each image, each feeding a shared coordinate stage that
rescales, clips, pads, merges, and de-duplicates boxes before anything is drawn
or masked:

<p align="center">
  <img src="docs/img/pipeline_flow.svg" alt="Pipeline flow: one document fans into three detector lanes — logos (Grounding DINO finds boxes → crop each box → a flat-fill pre-filter and a CLIP gate discard non-marks → an allowlist keeps your own brand), faces (YuNet detect → pad), and text/PII (ai_parse_document elements → regex pre-filter → Claude classifier → signature heuristic). All three feed a shared coordinate stage (rescale · clip · pad · merge · NMS) that emits a non-destructive debug overlay and a redacted copy (faces blurred, logos and text blacked)." width="900">
</p>

<sub>Each lane is a funnel: **Grounding DINO / YuNet / `ai_parse_document`** find *where* content might be, then the per-lane gates decide *what* actually gets masked — the CLIP gate (with a flat-fill pre-filter) for logos, the regex + Claude classifier for text. Everything lands on one shared coordinate frame, which emits a non-destructive **overlay** (for auditing) and the **redacted copy**. See [How logo detection works](#how-logo-detection-works-grounding-dino--crop--clip) below for the crop → CLIP detail. **Scope note (amber band):** the text lane relies on `ai_parse_document`, which reads *document-like* layout — so on a pure photograph only faces and logos are masked; text baked into a photo (scene text, badges, on-screen data) and other visual PII (license plates, ID-card fields, barcodes/QR, stamps, handwriting) are **not** detected yet — see [Known limitations](#known-limitations).</sub>

- **Logos** — [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)
  open-vocabulary detection with pixel-accurate `(H, W)` post-processing, then a
  **CLIP verification gate** that drops decorative icons / photos / charts so the
  pipeline masks real brand marks, not infographics. An **allowlist** (CLIP
  reference-image similarity against crops in `refs/`) can *keep* a chosen brand
  (e.g. your own logo) while masking every third-party mark.
- **Faces** — OpenCV **YuNet** (bundled DNN backend, tiny ONNX in `models/`), then
  **Gaussian blur** rather than a hard box — less destructive, still de-identifying.
- **Text / PII** — Databricks **`ai_parse_document`** localizes text elements
  (including wordmark logos like Gartner / Forrester that the graphical detector
  ignores), then a **regex PII pre-filter + Claude classifier** decides what to
  mask (`pii_only` vs `all_text`) so body copy isn't blindly blacked out.

Masks: **faces → blur, logos & text → black box.** Every run also emits a colored,
labelled **debug overlay** so you can tell detection errors from localization
errors at a glance.

### How logo detection works: Grounding DINO → crop → CLIP

Logo detection is a **two-model funnel**, and it helps to know that the two
models look at *different things*:

1. **Grounding DINO looks at the whole image and answers "where?"** It's an
   open-vocabulary detector: given a prompt like `"logo . brand logo . company
   emblem"`, it returns a list of **bounding boxes** — rectangles that *might* be
   logos (e.g. "something logo-like at pixels `[105, 66, 296, 129]`"). It is
   deliberately **recall-oriented** (a low score threshold), so it over-fires: it
   will happily box chart bars, colored shapes, and icons alongside real marks.
   Catching those false positives is the next stage's job.

2. **CLIP looks at one box at a time and answers "is this really a logo?"** CLIP
   is a strong *classifier* but has **no sense of location** — you cannot ask it
   to "point to the logo." So for each box Grounding DINO proposed, the pipeline
   **cuts that rectangle out of the image into a small standalone picture — the
   "crop" — and hands only that crop to CLIP.** CLIP never sees the whole page;
   it sees a sequence of little rectangles and, for each, scores it against
   captions like *"a company brand logo"* vs *"a decorative icon"*, *"a chart or
   graph element"*, *"a photograph"*. Crops that don't read as a logo are dropped.
   You can see this in `pipeline/verify.py`: `crop = image.crop(box)` then
   `is_logo(crop)`.

   ```text
   whole image ──▶ Grounding DINO ──▶ boxes ──▶ for each box: image.crop(box)
                     (where?)                              │
                                                           ▼
                                            CLIP judges each crop in isolation
                                              (is THIS little rectangle a logo?)
                                                           │
                                              keep brand marks, drop the rest
   ```

3. **A structural pre-filter runs before CLIP.** Because CLIP judges each crop
   *in isolation*, a crop that is just a flat colored rectangle (a chart bar, a
   solid shape, a colored table-header fill) can score as a minimalist "logo" —
   there's no surrounding context to tell CLIP it's part of a chart. So before
   the CLIP call, `ClipGate.is_flat_fill()` rejects any crop that is **both**
   near-single-color **and** edge-poor (few internal edges). A real logo — even a
   wordmark on a plain letterhead — is always edge-rich from its text strokes, so
   requiring *both* conditions means this never drops a genuine mark; it only
   removes solid fills.

4. **The allowlist is also CLIP, but image-to-image.** To *keep* a chosen brand
   (e.g. your own logo) while masking third-party marks, each surviving logo crop
   is compared by **CLIP image-embedding cosine similarity** to reference crops in
   `refs/`. Above `databricks_min_sim` (0.80) the mark is kept (`mask=False`).
   Reference-image similarity is used instead of a text prompt because zero-shot
   "is this the X logo?" proved unreliable (it mis-tagged a serif title and a car
   emblem).

The division of labor, in one line: **Grounding DINO decides *where*, the
structural pre-filter and CLIP decide *whether it's actually a brand mark*, and
the allowlist decides *whether to keep or mask* it.**

## Layout

```text
text_and_image_masking/
├── masking_pipeline.ipynb   # entry point — run, visualise, evaluate
├── pipeline/
│   ├── boxes.py             # Detection type + ALL coordinate handling
│   ├── detectors.py         # LogoDetector (GDINO), FaceDetector (YuNet), TextDetector (DBNet)
│   ├── verify.py            # CLIP gate: icon filter (text-prompt) + brand allowlist (image-sim)
│   ├── text_pii.py          # ai_parse_document text + regex/Claude PII filter
│   ├── masking.py           # apply_masks: black box / Gaussian blur per source
│   ├── overlay.py           # debug overlay renderer
│   ├── tracking.py          # MLflow instrumentation (dry-run if no MLflow)
│   └── run.py               # MaskingPipeline orchestrator + PipelineConfig
├── eval/
│   ├── score.py             # auto-propose GT + IoU precision/recall/coverage
│   ├── track_masking.py     # runnable MLflow workflow (single run or threshold sweep)
│   └── ground_truth.json    # auto-proposed scaffold — CORRECT BY HAND before trusting scores
├── app/                     # torch-free FastAPI Databricks App (vanilla-JS UI, SSE streaming)
├── deploy/                  # GPU Model Serving endpoint (vision) deploy scripts
├── refs/                    # reference logo crops for the brand allowlist
├── models/                  # YuNet ONNX (+ optional DBNet) — see requirements.txt
├── test_set/                # sample input images + build scripts + manifest
├── sample_data/             # signed-document samples (signature handling)
├── docs/img/                # before/after showcase composites (this README)
├── EVALUATION.md            # detection/localization failure analysis
├── PLAN.md                  # design rationale
├── NEXT_STEPS.md            # roadmap: specialized models (signatures, tables, financial PII)
└── requirements.txt

# outputs/ (*_overlay.jpg + *_masked.jpg per image) are generated at run time.
```

## Running

```python
from pipeline.run import MaskingPipeline, PipelineConfig

pipe = MaskingPipeline(PipelineConfig(do_logos=True, do_faces=True, do_text=False))
dets, overlay, masked = pipe.run("test_set/doc_report_with_photo.png")
overlay.save("debug.jpg")   # what was found + where (colored, labelled)
masked.save("out.jpg")      # the redacted image
```

- **Logos + faces run fully locally** (CPU / Apple MPS / GPU; auto-detected). The
  Grounding DINO + CLIP weights auto-download from Hugging Face on first run; the
  YuNet ONNX ships in `models/`.
- **Text phase** needs a Databricks serverless session for `ai_parse_document`.
  Set `do_text=True` and pass a `spark` session + `claude_endpoint` (see the
  notebook). It catches wordmark logos and PII the graphical detector leaves alone.

### Tuning the brand allowlist

Drop reference crops of a logo you want to **keep** (white-on-dark, mark-only,
on-white, …) into `refs/`. The allowlist keeps any detected logo whose CLIP
image-embedding cosine similarity to *any* reference exceeds `databricks_min_sim`
(default 0.80). Reference-image similarity is used instead of text prompts because
zero-shot "is this the X logo?" wrongly tagged a serif title and a car emblem.

## Tracking performance (MLflow)

`pipeline/tracking.py` captures each evaluation as an MLflow run — **params**
(config, model ids, git sha), **metrics** (per-category precision / recall /
coverage @ IoU + workflow latency, mask-area fraction, detection counts, mean
confidence), and **artifacts** (overlays, masked images, scored report). One
config = one run, so threshold sweeps and model swaps compare directly in the UI.

```bash
# single run over the sample set
python eval/track_masking.py --experiment /Shared/image-masking
# logo-threshold sweep (one run per threshold)
python eval/track_masking.py --sweep --experiment /Shared/image-masking
# include the ai_parse_document text phase
python eval/track_masking.py --do-text --experiment /Shared/image-masking
```

MLflow is native on Databricks; with no MLflow installed it runs in **dry-run**
mode (metrics computed and printed, not logged), so the metric logic is testable
offline. The notebook has equivalent cells (`evaluate_and_log` / `sweep_and_log`).

## Known limitations

- **Text phase requires Databricks** — `ai_parse_document` is a platform function;
  the logo + face phases are fully local. Run with `do_text=True` for full coverage
  (wordmark logos and PII depend on it). Validated live on a Databricks serverless
  session with `databricks-claude-sonnet-5` as the entity/PII classifier.
- **Text/PII detection is document-oriented — pure photos get only faces + logos.**
  `ai_parse_document` targets *document* layout (scans, forms, slides, PDFs,
  business cards). In testing it extracted **no** text from photographic scenes,
  so text/PII *baked into a photo* (signage, badges, on-screen data) and other
  non-text visual PII (**license plates, ID-card fields, barcodes/QR, stamps &
  seals, handwriting**) are **not** detected today — on a pure photograph only
  faces and logos are masked. This is the biggest coverage gap; see
  [`NEXT_STEPS.md`](./NEXT_STEPS.md) (“The gap” → document-layout + scene OCR) for
  the v6 plan.
- **PII classifier selectivity varies doc-to-doc** — in `text_mode="pii_only"`
  the Claude classifier is a judgment call, so exactly which lines it masks (names,
  fees, addresses) is not perfectly consistent across documents. Use
  `text_mode="all_text"` when you need to guarantee every text element is masked.
- **`ground_truth.json` is auto-proposed** — correct it before trusting eval
  numbers (scores against an uncorrected scaffold are meaningless by construction).
- **Thresholds** (`logo_box_threshold`, `databricks_min_sim`, face score) are
  reasonable defaults; sweep them against the corrected eval set per image type.

See [`EVALUATION.md`](./EVALUATION.md) for the detection/localization analysis,
[`PLAN.md`](./PLAN.md) for design rationale, and [`NEXT_STEPS.md`](./NEXT_STEPS.md)
for the roadmap (signatures, table cells, structured financial PII).
