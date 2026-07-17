# test_set — held-out images for evaluating the masking pipeline

30 images the pipeline has **never been tuned on**, for measuring real-world
performance (and feeding the MLflow tracking workflow). Every image is **freely
shareable**: 18 real photos under the **Unsplash License** (free to download,
use, and distribute, incl. commercial, no attribution required — via Unsplash
and Lorem Picsum), and 12 **synthetic** business documents generated locally
with fabricated PII (no real personal data).

Regenerate anytime with `python build_test_set.py`. `manifest.json` records each
image's source URL and license.

## Composition

| Category | Count | What it tests | Source / license |
| -------- | ----- | ------------- | ---------------- |
| `face_*`  | 4  | face detection + blur (single portraits + a multi-person meeting) | Unsplash License |
| `scene_*` | 7  | logos & faces in business scenes (offices, handshakes, signage, buildings) | Unsplash License |
| `neg_*`   | 7  | **negatives** — generic real photos that should mask **nothing** (over-masking / false-positive check) | Unsplash License (via Lorem Picsum) |
| `gen_*`   | 12 | full-stack PII on realistic docs (see below) | generated locally (fabricated PII) |

### Synthetic documents (`gen_*`)

Cover the consulting/tax/audit content the pipeline targets, with known PII:

- **3 business cards** — logo + name + title + phone + email + address
- **2 invoices** — EIN, bank routing #, account #, amounts, client name
- **financial statement** — figures in a table, org names, EIN, "Confidential"
- **W-9** + **1099-NEC** — SSN, EIN/TIN, names, addresses (+ signature on the W-9)
- **engagement letter** + **audit memo** — names, fees, contact emails/phones, cursive signature
- **bank statement** — account #, routing #, transactions
- **NDA page** — party names, signatory names, legal contact emails

## Notes

- **No ground-truth labels** ship with these — they're for qualitative review
  and **workflow metrics** (latency, mask counts, masked-area fraction). To get
  precision/recall/coverage, label a subset (`auto_propose_ground_truth` →
  correct) and point the harness at it.
- Synthetic docs contain **fabricated** names/numbers — no real PII. Real photos
  are people/scenes under the Unsplash License; see `manifest.json` for URLs.

## Quick start

```bash
# run the pipeline + log workflow metrics over the whole set (dry-run if no MLflow)
python ../eval/track_masking.py --images-dir . --run-name test_set_baseline \
    --experiment /Shared/image-masking-v5
```
