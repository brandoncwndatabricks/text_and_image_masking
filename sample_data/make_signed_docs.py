"""Generate synthetic signed documents for testing signature detection.

Produces realistic-looking consulting/audit documents that each contain one or
more handwritten-style signatures rendered with macOS script fonts. Synthetic
(not real signatures) so there are no privacy concerns, and because we draw the
signatures ourselves we know their exact bounding boxes — written to
``ground_truth_signatures.json`` for use with the eval harness.

Run:  python make_signed_docs.py
Output: signed_engagement_letter.png, signed_audit_signoff.png,
        ground_truth_signatures.json   (all in this directory)
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/System/Library/Fonts/Supplemental"

SCRIPT_FONTS = [
    "SnellRoundhand.ttc",
    "Savoye LET.ttc",
    "Brush Script.ttf",
    "Zapfino.ttf",
]


def font(name: str, size: int):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def text_bbox(draw, xy, s, fnt):
    """Return the [x1,y1,x2,y2] a string occupies when drawn at xy."""
    b = draw.textbbox(xy, s, font=fnt)
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


def draw_signature(draw, x, y, name, font_file, size, color=(20, 30, 90)):
    """Draw a cursive signature; return its bounding box (padded a little)."""
    fnt = font(font_file, size)
    box = text_bbox(draw, (x, y), name, fnt)
    draw.text((x, y), name, fill=color, font=fnt)
    # pad ~8% so the recorded box covers ascenders/descenders/flourishes
    w, h = box[2] - box[0], box[3] - box[1]
    return [box[0] - w * 0.04, box[1] - h * 0.12, box[2] + w * 0.06, box[3] + h * 0.12]


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def para(draw, x, y, text, fnt, max_w, leading, fill=(30, 30, 30)):
    for line in wrap(draw, text, fnt, max_w):
        draw.text((x, y), line, fill=fill, font=fnt)
        y += leading
    return y


# ── Document 1: engagement letter (single signature) ────────────────────────────

def engagement_letter():
    W, H = 1275, 1650  # ~US Letter at 150 DPI
    img = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(img)
    m = 110
    title = font("Times New Roman Bold.ttf", 30)
    h2 = font("Times New Roman Bold.ttf", 19)
    body = font("Times New Roman.ttf", 19)
    small = font("Arial.ttf", 15)

    d.text((m, 70), "MERIDIAN ADVISORY PARTNERS LLP", fill=(25, 40, 75), font=title)
    d.text((m, 112), "Audit  •  Tax  •  Advisory", fill=(110, 110, 110), font=small)
    d.line([(m, 150), (W - m, 150)], fill=(180, 180, 180), width=2)

    y = 200
    d.text((m, y), "March 14, 2026", fill=(60, 60, 60), font=body); y += 50
    y = para(d, m, y, "Ms. Eleanor Whitcombe", body, W - 2 * m, 28);
    y = para(d, m, y, "Chief Financial Officer, Northgate Industries Inc.", body, W - 2 * m, 28); y += 20

    d.text((m, y), "Re: Engagement for Fiscal Year 2025 Audit Services", fill=(25, 40, 75), font=h2); y += 50
    y = para(d, m, y, "Dear Ms. Whitcombe,", body, W - 2 * m, 30); y += 10
    y = para(d, m, y,
             "We are pleased to confirm our understanding of the services we are to provide "
             "to Northgate Industries Inc. for the year ending December 31, 2025. We will audit "
             "the consolidated financial statements, which comprise the balance sheet and the "
             "related statements of income, changes in equity, and cash flows.", body, W - 2 * m, 30)
    y += 14
    y = para(d, m, y,
             "Our fees are based on the time required by the individuals assigned to the "
             "engagement, plus direct expenses. We estimate total fees for this engagement at "
             "$245,000, billed monthly as work progresses.", body, W - 2 * m, 30)
    y += 14
    y = para(d, m, y,
             "Please sign below to indicate your acknowledgement of, and agreement with, the "
             "terms of this engagement.", body, W - 2 * m, 30)
    y += 60

    # signature block
    d.text((m, y), "Sincerely,", fill=(40, 40, 40), font=body); y += 70
    sig1 = draw_signature(d, m + 10, y, "Jonathan R. Albright", "SnellRoundhand.ttc", 46)
    y += 90
    d.line([(m, y), (m + 360, y)], fill=(80, 80, 80), width=2); y += 8
    d.text((m, y), "Jonathan R. Albright, Engagement Partner", fill=(40, 40, 40), font=small)
    d.text((m, y + 22), "Meridian Advisory Partners LLP", fill=(90, 90, 90), font=small)

    return img, [sig1]


# ── Document 2: audit sign-off / approval (two signatures) ───────────────────────

def audit_signoff():
    W, H = 1275, 1650
    img = Image.new("RGB", (W, H), (255, 255, 253))
    d = ImageDraw.Draw(img)
    m = 110
    title = font("Times New Roman Bold.ttf", 28)
    h2 = font("Times New Roman Bold.ttf", 19)
    body = font("Times New Roman.ttf", 18)
    small = font("Arial.ttf", 15)
    mono = font("Arial.ttf", 16)

    d.text((m, 64), "WORKPAPER REVIEW & APPROVAL", fill=(20, 20, 20), font=title)
    d.text((m, 104), "Engagement: Northgate Industries Inc. — FY2025 Audit", fill=(90, 90, 90), font=small)
    d.line([(m, 138), (W - m, 138)], fill=(170, 170, 170), width=2)

    y = 180
    rows = [
        ("Workpaper reference:", "A-3200  Revenue Recognition"),
        ("Prepared by:", "Priya Nair, Senior Associate"),
        ("Date prepared:", "February 27, 2026"),
        ("Review status:", "Approved — no exceptions noted"),
    ]
    for label, val in rows:
        d.text((m, y), label, fill=(60, 60, 60), font=h2)
        d.text((m + 320, y), val, fill=(25, 25, 25), font=body)
        y += 40
    y += 20

    d.text((m, y), "Reviewer notes", fill=(20, 20, 20), font=h2); y += 36
    y = para(d, m, y,
             "Substantive testing of the revenue cycle was performed over a sample of 60 "
             "transactions. Cut-off procedures and analytical review did not identify material "
             "misstatement. Management's estimates regarding variable consideration appear "
             "reasonable and adequately supported.", body, W - 2 * m, 28)
    y += 50

    # Two-column signature block: preparer + partner
    d.line([(m, y), (W - m, y)], fill=(200, 200, 200), width=1); y += 40
    d.text((m, y), "Reviewed by (Manager)", fill=(40, 40, 40), font=small)
    d.text((W // 2 + 30, y), "Approved by (Partner)", fill=(40, 40, 40), font=small)
    y += 40

    sig_a = draw_signature(d, m + 10, y, "Marcus Lindqvist", "Savoye LET.ttc", 50, color=(15, 20, 60))
    sig_b = draw_signature(d, W // 2 + 40, y, "D. Okafor", "Brush Script.ttf", 44, color=(10, 10, 70))
    y += 86
    d.line([(m, y), (m + 380, y)], fill=(80, 80, 80), width=2)
    d.line([(W // 2 + 30, y), (W // 2 + 30 + 360, y)], fill=(80, 80, 80), width=2)
    y += 8
    d.text((m, y), "Marcus Lindqvist, Audit Manager", fill=(40, 40, 40), font=small)
    d.text((W // 2 + 30, y), "Dara Okafor, Engagement Partner", fill=(40, 40, 40), font=small)

    d.text((m, H - 70), "Meridian Advisory Partners LLP   —   Confidential", fill=(150, 150, 150), font=mono)

    return img, [sig_a, sig_b]


def main():
    out = {}
    img1, sigs1 = engagement_letter()
    p1 = os.path.join(HERE, "signed_engagement_letter.png")
    img1.save(p1)
    out["signed_engagement_letter.png"] = [
        {"type": "signature", "box": [round(v, 1) for v in b]} for b in sigs1
    ]

    img2, sigs2 = audit_signoff()
    p2 = os.path.join(HERE, "signed_audit_signoff.png")
    img2.save(p2)
    out["signed_audit_signoff.png"] = [
        {"type": "signature", "box": [round(v, 1) for v in b]} for b in sigs2
    ]

    gt_path = os.path.join(HERE, "ground_truth_signatures.json")
    with open(gt_path, "w") as f:
        json.dump(out, f, indent=2)

    print("Wrote:")
    for p in (p1, p2, gt_path):
        print("  ", p)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
