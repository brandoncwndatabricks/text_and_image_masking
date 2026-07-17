# Next Steps — Recommended Specialized Models (v6 candidates)

> Roadmap for extending the v5 pipeline beyond **logos / faces / text-PII** to
> the full range of sensitive content found in a consulting / advisory / tax /
> audit document corpus (engagement letters, audit workpapers, financial
> statements, tax returns, KYC packets, contracts, scanned PDFs, slide decks).
>
> All models below are **Hugging Face / GitHub downloadable** (PyPI is firewalled
> in our env), fit a **single A10 GPU or CPU**, and are cleared for **internal
> use** (AGPL / restricted weights acceptable — flagged where relevant). See
> [`PLAN.md`](./PLAN.md) and [`EVALUATION.md`](./EVALUATION.md) for the v5 design.

---

## The gap

v5 masks logos, faces, and sensitive text. A Big-4-style corpus also contains
**signatures, stamps/seals, financial figures inside tables, structured
financial identifiers (EIN/routing/IBAN/account #), ID-document fields, and
barcodes/QR** — none of which the current stack handles.

---

## Recommended additions (ranked by leverage)

### 1. 🥇 Document layout analysis — *biggest win, build first*
Segments each page into zones (header / footer / title / figure / table /
key-value). This **routes redaction policy by document type**, tells every other
detector **where to look**, and lets us suppress boilerplate (footer logos, page
numbers). We have nothing like this today; it should become the **first stage**
of a v6 pipeline.

| Model | Repo | License | Notes |
| ----- | ---- | ------- | ----- |
| **Docling layout (Heron)** ⭐ | `ds4sd/docling-layout-heron` | Apache-2.0 | RT-DETRv2 ~43M; 17 classes incl. explicit page-header/footer/key-value |
| DocLayout-YOLO | `juliozhao/DocLayout-YOLO-DocStructBench` | code AGPL / weights tagged Apache (conflict — resolve) | ~14× faster than DiT; coarser header/footer |
| _avoid_ DiT-PubLayNet / LayoutLMv3 | `microsoft/dit-base` / `microsoft/layoutlmv3-base` | MIT / CC-BY-NC | dit-base is a backbone (no boxes); LayoutLMv3 needs OCR boxes + is non-commercial |

### 2. ✅ Signature detection — *mostly free via `ai_parse_document` (DONE), model only as fallback*
**Tested, not assumed.** Running `ai_parse_document` on the
[`sample_data/`](./sample_data) signed docs showed it already **localizes
signatures with accurate boxes** and usually **reads the signer's name**:

- Engagement letter → signature returned as a `figure`, "Jonathan R. Albright" transcribed.
- Audit sign-off → both signatures returned as `text`, names transcribed.

Two consequences, both now handled in `pipeline/text_pii.py` with **no extra model**:

1. **Legible signatures** come back as `text` with the name → the existing
   **name-PII filter** (Claude/regex) masks them already.
2. **Illegible scrawls** come back as a **content-less `figure`** (no name → PII
   path can't fire). `flag_signatures()` closes this gap: it promotes any
   `figure` in a **sign-off zone** (horizontally aligned with, and vertically
   near, a "Sincerely/Approved by/…" cue or a "Name, Role" caption) to a
   `signature` detection and masks it. Validated on the engagement letter at
   **IoU 0.89** vs ground truth.

**So: no signature model needed for now.** Add the dedicated detector below
**only if** real-world testing shows illegible handwritten signatures slipping
through as untyped figures *outside* a recognizable sign-off zone.

| Fallback model | Repo | License | Notes |
| -------------- | ---- | ------- | ----- |
| YOLOv8s signature | `tech4humans/yolov8s-signature-detector` | weights AGPL-3.0, code Apache | ~11M, mAP@50 94.5%, Tobacco800-trained |
| Conditional-DETR signature | `tech4humans/conditional-detr-50-signature-detector` | Apache-2.0 | 43.5M, mAP@50 93.7% — use if a permissive license is required |

### 3. 🥉 Table detection + structure (cell-level boxes)
Financial statements and audit schedules are the core artifact — sensitive
figures (balances, salaries, account totals) live **in cells**. Cell boxes let
us redact a single column without destroying the table.

| Model | Repo | License | Cell boxes |
| ----- | ---- | ------- | ---------- |
| **TATR structure (financial)** ⭐ | `microsoft/table-transformer-structure-recognition-v1.1-fin` | MIT | rows+cols → cells via intersection (handle spanning cells); FinTabNet-tuned |
| TATR detection (stage 1) | `microsoft/table-transformer-detection` | MIT | table locator |
| PP-Structure / SLANet | `PaddlePaddle/PaddleOCR` | Apache-2.0 | **native cell coords** if adopting the Paddle stack |
| UniTable | `poloclub/unitable` (GitHub) | MIT | permissive cell-box alternative to Surya |

### 4. Structured financial PII NER — *augments, not replaces, the Claude/regex step*
Claude is good at names/orgs but unreliable on rigid numeric IDs
(**EIN, ABA routing #, IBAN, account #**). GLiNER gives zero-shot custom entity
types + character spans locally; pair it with **deterministic checksum
validators** (IBAN mod-97, ABA, TIN format) for the numeric IDs.

| Model | Repo | License | Notes |
| ----- | ---- | ------- | ----- |
| **GLiNER PII** ⭐ | `urchade/gliner_multi_pii-v1` / `knowledgator/gliner-pii-large-v1.0` | Apache-2.0 | define custom labels at inference; returns char spans |
| Piiranha | `iiiorg/piiranha-v1-detect-personal-information` | CC-BY-NC-ND (flag) | 17 fixed types; no routing/IBAN/balance, no derivatives |
| _re-implement_ Presidio recognizers | (Presidio is pip-only) | MIT | port the regex/checksum validators for SSN/TIN/EIN/ABA/IBAN |

> ⚠️ Don't let any single ML model be the **sole compliance gate**. Layer
> GLiNER (contextual) + deterministic validators (structured IDs) + existing
> Claude pass.

---

## Cheap, high-value runner-ups

| Need | Model | Repo | License |
| ---- | ----- | ---- | ------- |
| **Stamps / seals** (notary, approval, received) | YOLOv8 stamp | `stamps-labs/yolov8-finetuned` | MIT (card lacks explicit tag — confirm) |
| **ID / KYC — MRZ bbox** | MRZ detector | `passport-exn5h/mrz-detection-29c4e-d5trv` | CC-BY-4.0 (trained on 111 imgs — validate) |
| **ID field-level boxes** | PP-Structure KIE | `PaddlePaddle/PaddleOCR` | Apache-2.0 |
| **Barcodes / QR** (encode client IDs) | built-in | `cv2.QRCodeDetector`, `cv2.barcode.BarcodeDetector` | already have OpenCV; decode w/ zbar/pyzbar |

---

## Consolidation option: a box-emitting document VLM

If the specialist stack grows unwieldy, one VLM can do layout + tables + text +
reading order in a single pass on the A10 — good as a **fallback for messy scans**
that break the specialist detectors.

| Model | Repo | Params | License | Boxes |
| ----- | ---- | ------ | ------- | ----- |
| **dots.ocr** ⭐ | `rednote-hilab/dots.ocr` | 3B | MIT | JSON layout: bbox + category + text + reading order; A10G reference target |
| **Qwen2.5-VL-7B** ⭐ | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Apache-2.0 | native grounding bboxes/points (use **7B**, not the non-commercial 3B) |
| Granite-Docling | `ibm-granite/granite-docling-258M` | 258M | Apache-2.0 | DocTags `<loc_>` coords (validate on scans) |

---

## Skip / lower priority (redundant with v5 or weak for redaction)

- **Doc-type classification** (`microsoft/dit-base-finetuned-rvlcdip`) — routing
  convenience, not a redaction localizer. Add only if routing logic needs it.
- **Donut / Nougat / GOT-OCR / olmOCR** — no reliable boxes; some non-commercial;
  overlap `ai_parse_document`.
- **TrOCR handwriting** — recognition only; useless until a handwriting-**region**
  detector exists (none off-the-shelf). Defer unless handwritten annotations are
  a real surface.

---

## Licensing flags for a Big-4 context

- **Surya** (tables/layout) — weights OpenRAIL-M with a **~$5M revenue cap**;
  prefer TATR / PP-Structure / UniTable.
- **YOLO-lineage weights** (signature, barcode, DocLayout-YOLO) are **AGPL** —
  fine internally, but get legal sign-off before anything customer-facing.
- Several model cards (stamp repos, `dit-base-finetuned-rvlcdip`) **lack explicit
  license tags** — confirm before production.

---

## Proposed v6 architecture (layout-first)

```text
Page image
   │
   ▼
[1] Docling layout  ──►  zones: header / footer / title / body / figure / table / key-value
   │                         │
   │            route redaction policy by zone + doc type
   ▼                         ▼
[2] Per-zone specialist detectors (run only where relevant)
      body/figure → logos (GDINO+CLIP), faces (YuNet)
      table       → TATR/PP-Structure cell boxes → redact sensitive columns
      any text    → ai_parse_document → GLiNER + checksum validators + Claude
      signature   → ai_parse figures + flag_signatures()  (YOLOv8 only as fallback)
      stamp       → YOLOv8 stamp
      ID region   → MRZ + PP-Structure KIE       barcode → OpenCV QR/barcode
   │
   ▼
[3] shared box stage (rescale·clip·pad·merge·NMS)  →  debug overlay + apply_masks
```

**Suggested order of work:** (1) layout → (2) tables → (3) GLiNER+validators,
validating each against the IoU eval harness on a labelled tax/audit sample set
before moving on. *(Signatures are already handled by `ai_parse_document` +
`flag_signatures()` — see §2.)*
