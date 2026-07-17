# src_v5 — Image Masking Pipeline

A redaction pipeline that detects and masks **logos, faces, and sensitive text**
in images — rebuilt to fix v4's central defect: **masks landing in the wrong
location**. See [`EVALUATION.md`](./EVALUATION.md) for the v4 failure analysis
and [`PLAN.md`](./PLAN.md) for the full design rationale. For the roadmap of
specialized models to add next (signatures, stamps, tables, financial PII for a
consulting/tax/audit corpus), see [`NEXT_STEPS.md`](./NEXT_STEPS.md).

## What changed from v4 (and why)

v4 could *find* sensitive content but often *placed the mask in the wrong spot*
— most visibly the Audi image, where the "AUDI" wordmark and four-rings emblem
were left unmasked while a stray box landed in empty sky. Root cause: **OWLv2
pads images to a square before inference and returns box coordinates relative to
that padded square, but v4 rescaled them against the original non-square
dimensions.** ([transformers #27705](https://github.com/huggingface/transformers/issues/27705),
[#18553](https://github.com/huggingface/transformers/issues/18553))

| | v4 | v5 |
| --- | --- | --- |
| **Logos** | OWLv2 (buggy non-square coords) | **Grounding DINO** + correct `(H,W)` post-process → pixel-accurate boxes |
| **Logo false positives** | blunt area-fraction cutoff (blacked out whole infographics) | **CLIP verification gate** drops decorative icons/photos/charts |
| **Databricks' own logo** | masked like any other | **kept** via CLIP reference-image similarity (allowlist) |
| **Faces** | *not detected at all* | **YuNet** face detector, **Gaussian-blurred** (less destructive) |
| **Text / PII** | `ai_parse_document` + Claude, mask-everything fallback | same detector (good localization) + **padded boxes** + **regex PII pre-filter** replacing the mask-everything fallback |
| **Debuggability** | hard masks only — can't tell detection vs localization errors | **debug overlay** (colored, labelled, non-destructive) + **IoU eval harness** |

### Proof points (local run, logos + faces)

- **Audi:** both the wordmark and the grille emblem are now masked in the
  correct location (v4 masked neither). Scores precision/recall **1.0**,
  coverage **0.99** against ground truth.
- **Consulting slide:** the two faces (missed entirely by v4) are detected and
  blurred; the decorative petal icons that v4 blacked the whole diagram for are
  correctly **filtered out** by the CLIP gate.
- **Success-story slide:** the "Conde Nast" customer name (title + wordmark) is
  masked; the Databricks logo is **kept**; body text preserved for the text phase.

## Architecture

```text
                 ┌──────────────── one image ────────────────┐
                 ▼                  ▼                          ▼
        Grounding DINO          YuNet                 ai_parse_document
        (graphical logos)       (faces)               (text / wordmarks / PII)
                 │                  │                          │
          CLIP gate:          pad ~15%               regex PII pre-filter
          drop icons,                                + Claude classifier
          keep Databricks                            (pii_only | all_text)
                 └──────────┬───────┴───────────┬──────────────┘
                            ▼                   ▼
                   shared box stage      debug overlay (non-destructive)
                   (rescale·clip·pad·            │
                    merge·NMS)                    ▼
                            ▼              apply_masks
                            └────────────► faces→blur, logos/text→black
                                                  → *_masked.*
```

## Layout

```text
text_and_image_masking/
├── masking_pipeline.ipynb   # entry point — run, visualise, evaluate
├── pipeline/
│   ├── boxes.py             # Detection type + ALL coordinate handling (the v4 fix)
│   ├── detectors.py         # LogoDetector (GDINO), FaceDetector (YuNet), TextDetector (DBNet)
│   ├── verify.py            # CLIP gate: icon filter (text-prompt) + Databricks allowlist (image-sim)
│   ├── text_pii.py          # ai_parse_document text + regex/Claude PII filter
│   ├── masking.py           # apply_masks: black box / Gaussian blur per source
│   ├── overlay.py           # debug overlay renderer
│   ├── tracking.py          # MLflow instrumentation (model + workflow metrics; dry-run if no MLflow)
│   └── run.py               # MaskingPipeline orchestrator + PipelineConfig
├── eval/
│   ├── score.py             # auto-propose GT + IoU precision/recall/coverage
│   ├── track_masking.py     # runnable MLflow workflow (single run or threshold sweep)
│   └── ground_truth.json    # auto-proposed scaffold — CORRECT BY HAND before trusting scores
├── app/                     # torch-free FastAPI Databricks App (vanilla-JS UI, SSE streaming)
├── deploy/                  # GPU Model Serving endpoint (vision) deploy scripts
├── refs/                    # reference Databricks logo crops for the allowlist
├── models/                  # YuNet ONNX (+ optional DBNet) — see requirements.txt
├── test_set/               # sample input images + build scripts + manifest
├── sample_data/            # signed-document samples (signature handling)
├── EVALUATION.md            # v4 failure analysis
├── PLAN.md                  # design rationale
├── NEXT_STEPS.md            # v6 roadmap (specialized models)
└── requirements.txt

# outputs/ (*_overlay.jpg + *_masked.jpg per image) are generated at run time.
```

## Running

```python
from pipeline.run import MaskingPipeline, PipelineConfig

pipe = MaskingPipeline(PipelineConfig(do_logos=True, do_faces=True, do_text=False))
dets, overlay, masked = pipe.run("path/to/image.jpg")
overlay.save("debug.jpg")   # what was found + where (colored, labelled)
masked.save("out.jpg")      # the redacted image
```

- **Logos + faces run fully locally** (CPU / Apple MPS / GPU; auto-detected).
- **Text phase** needs a Databricks serverless session for `ai_parse_document`.
  Set `do_text=True` and pass a `spark` session + `claude_endpoint` (see the
  notebook). The text phase catches **wordmark logos** (Gartner, Forrester, …)
  and PII that the graphical-logo detector intentionally leaves alone.

### Tuning the Databricks allowlist

Drop more reference crops of the Databricks logo (white-on-dark, brick-only,
on-white, etc.) into `refs/`. The allowlist keeps any detected logo whose CLIP
image-embedding cosine similarity to *any* reference exceeds
`databricks_min_sim` (default 0.80). Reference-image similarity is used instead
of text prompts because zero-shot "is this the Databricks logo?" wrongly tagged
a serif title and a car emblem as Databricks.

## Tracking performance (MLflow)

`pipeline/tracking.py` captures each evaluation as an MLflow run — **params**
(config, model ids, git sha), **metrics** (per-category precision / recall /
coverage @ IoU + workflow latency, mask-area fraction, detection counts, mean
confidence), and **artifacts** (overlays, masked images, scored report). One
config = one run, so threshold sweeps and model swaps compare directly in the
MLflow UI.

```bash
# single run
python eval/track_masking.py --experiment /Shared/image-masking-v5
# logo-threshold sweep (one run per threshold)
python eval/track_masking.py --sweep --experiment /Shared/image-masking-v5
# include the ai_parse_document text phase
python eval/track_masking.py --do-text --experiment /Shared/image-masking-v5
```

MLflow is native on Databricks; with no MLflow installed it runs in **dry-run**
mode (metrics computed and printed, not logged), so the metric logic is testable
offline. The notebook has equivalent cells (`evaluate_and_log` / `sweep_and_log`).

## Known limitations / next steps

- **Text phase is validated on Databricks, not in the offline sandbox** (PyPI
  firewalled → no PaddleOCR; Databricks auth was expired locally). On-platform
  it uses the same `ai_parse_document` that localized text well in v4.
- **Wordmark logos depend on the text phase** — the graphical-logo detector
  deliberately ignores pure text. Run with `do_text=True` for full coverage.
- **`ground_truth.json` is auto-proposed** — correct it before trusting eval
  numbers (scores against an uncorrected scaffold are meaningless by construction).
- **Thresholds** (`logo_box_threshold`, `databricks_min_sim`, face score) are
  reasonable defaults; sweep them against the corrected eval set per image type.
