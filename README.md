# Text & Image Masking

A redaction pipeline that automatically finds and masks **sensitive content
embedded in images and documents** — company **logos**, human **faces**, and
sensitive **text / PII** — and writes back a redacted copy plus a non-destructive
debug overlay showing exactly what was found and where.

It's built for consulting / advisory / audit / tax-style material (engagement
letters, financial statements, slide decks, business cards, scanned PDFs) where a
single page often mixes all three: a client logo in the header, headshots in a
team slide, and names / emails / account numbers in the body.

An optional fourth lane (**`do_sensitive`**) adds a **Claude-vision** pass for
sensitive content *baked into an image* that the document-text lane can't read —
ID / tax documents, payment cards, license plates, barcodes / QR, on-screen data,
signatures, and text inside photographs — with an open-vocabulary Grounding DINO
fallback. The logo, face, and text lanes are unchanged and still use their own
models; the sensitive lane is additive and off by default.

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

Three detectors run over each image (plus an optional fourth — the Claude-vision
sensitive-items lane), each feeding a shared coordinate stage that rescales,
clips, pads, merges, and de-duplicates boxes before anything is drawn or masked:

<p align="center">
  <img src="docs/img/pipeline_flow.svg" alt="Pipeline flow: one document fans into three detector lanes — logos (Grounding DINO finds boxes → crop each box → a flat-fill pre-filter and a CLIP gate discard non-marks → an allowlist keeps your own brand), faces (YuNet detect → pad), and text/PII (ai_parse_document elements → regex pre-filter → Claude classifier → signature heuristic). All three feed a shared coordinate stage (rescale · clip · pad · merge · NMS) that emits a non-destructive debug overlay and a redacted copy (faces blurred, logos and text blacked)." width="900">
</p>

<sub>Every lane is the same two-step funnel: **① FIND** where content might be (Grounding DINO / YuNet / `ai_parse_document` / Claude vision), then **② DECIDE** what actually gets masked (the flat-fill + CLIP gate for logos, always-blur for faces, the regex + Claude classifier for text). **③ MERGE** lands every box on one aligned coordinate frame; **④ OUTPUT** is a non-destructive labelled **overlay** and the **redacted copy**. Lanes run in parallel and toggle independently — logos + faces are on by default, text/PII needs Databricks, and the **sensitive-items lane (dashed = optional, off by default)** adds a Claude-vision pass for in-image PII the document-text lane can't read. See [How logo detection works](#how-logo-detection-works-grounding-dino--crop--clip) for the crop → CLIP detail and [Known limitations](#known-limitations) for the document-vs-photo scope.</sub>

**A real document through those stages** — the same advisory profile at each step: the detectors find regions (colored boxes; green = detected but kept), then the client logo is blacked, the executive headshot is Gaussian-blurred, and the CEO name / direct line / assistant email are masked while the title and narrative are preserved.

<p align="center">
  <img src="docs/img/pipeline_walkthrough.png" alt="Step-by-step: (1) original advisory document; (2) detector boxes drawn over it — logo, face, and text/PII regions, with kept regions in green; (3) the redacted copy with the logo blacked, the headshot blurred, and the sensitive text blacked while the title and body remain." width="960">
</p>

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
  mask (`pii_only` vs `all_text`) so body copy isn't blindly blacked out. Because
  `ai_parse_document` returns one coarse box per block, multi-line elements are
  **split into per-line boxes** so only the sensitive lines are masked, not the
  whole paragraph (tables are still masked whole — they're dense PII by nature).
- **Sensitive items & in-image PII** *(optional, `do_sensitive=True`)* — a
  **Claude vision** pass returns boxes for sensitive content the other lanes
  miss: **text baked into a photo** (signage, badges, on-screen data) and
  **non-text items** (ID/tax documents, payment cards, license plates,
  barcodes/QR, signatures). It's *selective* — the model reads and judges — and
  field-accurate (validated IoU ≈ 0.95–1.0 on known boxes; it localized a W-9's
  SSN/EIN, invoice bank details, and office screens). If the vision endpoint is
  unavailable it **falls back** to open-vocabulary **Grounding DINO** object
  detection (`sensitive_backend="objects"`) — local and fast, but noisier and
  not selective (it can't read the content).

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
│   ├── detectors.py         # LogoDetector (GDINO), FaceDetector (YuNet), TextDetector (DBNet), SensitiveObjectDetector (GDINO open-vocab)
│   ├── verify.py            # CLIP gate: flat-fill pre-filter + icon filter (text-prompt) + brand allowlist (image-sim)
│   ├── text_pii.py          # ai_parse_document text + regex/Claude PII filter
│   ├── vlm_detect.py        # Claude-vision sensitive-items lane (in-image PII + objects), open-vocab fallback
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
- **Sensitive-items phase** *(optional)* — set `do_sensitive=True` and pass a
  `claude_endpoint` (a **multimodal** Claude endpoint). It masks in-image PII and
  sensitive objects the other lanes miss. It runs the Claude-vision backend by
  default; pass `sensitive_backend="objects"` to force the local Grounding DINO
  open-vocabulary fallback instead (no endpoint needed, but noisier / not
  selective). Independent of the logo/face/text toggles.

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
- **The base text lane is document-oriented; in-image PII needs the optional
  sensitive-items lane.** `ai_parse_document` targets *document* layout (scans,
  forms, slides, PDFs, business cards) and extracts **no** text from photographic
  scenes — so with only faces + logos + text enabled, a pure photograph gets just
  faces and logos masked. Enable **`do_sensitive=True`** (Claude-vision, above) to
  also catch text baked into photos and non-text items (IDs, cards, plates,
  barcodes/QR, screens, signatures). Caveats for that lane: it needs a multimodal
  Claude serving endpoint, adds a vision call's latency, and — while box
  localization tested well — VLM coordinates can drift on cluttered photos or very
  small text; the open-vocab fallback is noisier still. See
  [`NEXT_STEPS.md`](./NEXT_STEPS.md) for the broader v6 plan.
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
