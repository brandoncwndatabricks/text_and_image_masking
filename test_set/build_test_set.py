"""Build a held-out, FREELY SHAREABLE test set (~30 images) for the pipeline.

Licensing is the priority here — every image is safe to commit and share:
  - Real photos come only from **Unsplash** (Unsplash License: free to download,
    use, and DISTRIBUTE, incl. commercial, no attribution required) and **Lorem
    Picsum** (which serves Unsplash photos). No OpenCV/OCR-repo samples — those
    (e.g. lena.jpg, press photos) have non-free or unclear licensing.
  - Document/PII images are **synthetic**, generated locally with fabricated
    names/numbers (no real PII), so they are fully ours to share.

Faces are intentionally limited to 4 images. URL lists are overprovisioned;
each download is validated with PIL and skipped on failure. manifest.json records
each image's source, URL, and license.

Run:  python build_test_set.py
"""

from __future__ import annotations

import io
import json
import os
import urllib.request

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/System/Library/Fonts/Supplemental"
UA = {"User-Agent": "Mozilla/5.0 (test-set-builder)"}

# (category, license, url). Overprovisioned; validated downloads are kept.
# Faces capped at 4 (see FACE_CAP). All freely shareable.
REAL_SOURCES = [
    # ── faces / people (Unsplash License) ───────────────────────────────────
    ("face", "Unsplash License", "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?w=800"),  # team
    ("face", "Unsplash License", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800"),  # man
    ("face", "Unsplash License", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800"),  # woman
    ("face", "Unsplash License", "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=800"),  # businesswoman
    ("face", "Unsplash License", "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=800"),     # backup
    ("face", "Unsplash License", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800"),  # backup
    # ── business scenes (logos / signage / offices) — Unsplash License ──────
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?w=800"),
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?w=800"),
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1556761175-5973dc0f32e7?w=800"),
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=800"),
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800"),
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800"),
    ("scene", "Unsplash License", "https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=800"),
    # ── negatives / generic real photos (Lorem Picsum → Unsplash) ───────────
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/ledger11/800/600"),
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/harbor27/800/600"),
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/forest33/800/600"),
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/skyline48/800/600"),
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/desk52/800/600"),
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/coast64/800/600"),
    ("neg", "Unsplash License (via Lorem Picsum)", "https://picsum.photos/seed/market77/800/600"),
]
FACE_CAP = 4


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def fetch(url, timeout=30):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img if (img.width >= 120 and img.height >= 120) else None
    except Exception as e:
        print(f"  skip ({type(e).__name__}) {url}")
        return None


# ── synthetic generators (fabricated PII; fully shareable) ───────────────────

def _wrap(d, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if d.textlength((cur + " " + w).strip(), font=fnt) <= max_w:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def gen_card(name, title, company, phone, email, addr, color):
    W, H = 1050, 600
    img = Image.new("RGB", (W, H), (250, 250, 248)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 18, H], fill=color)
    d.ellipse([60, 60, 130, 130], fill=color); d.rectangle([78, 78, 112, 112], fill=(255, 255, 255))
    d.text((150, 78), company, fill=color, font=font("Arial Bold.ttf", 40))
    d.text((150, 130), "Audit • Tax • Advisory", fill=(120, 120, 120), font=font("Arial.ttf", 20))
    d.line([(60, 200), (W - 60, 200)], fill=(210, 210, 210), width=2)
    d.text((60, 250), name, fill=(25, 25, 25), font=font("Georgia Bold.ttf", 34))
    d.text((60, 300), title, fill=(90, 90, 90), font=font("Georgia Italic.ttf", 22))
    y = 380
    for lab, val in [("M", phone), ("E", email), ("A", addr)]:
        d.text((60, y), f"{lab}   {val}", fill=(40, 40, 40), font=font("Arial.ttf", 22)); y += 42
    return img


def gen_statement(title, org, meta_rows, items, total_label, footer_rows, accent=(20, 40, 90)):
    W, H = 1000, 1300
    img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
    d.text((60, 50), title, fill=accent, font=font("Arial Bold.ttf", 42))
    d.text((60, 112), org, fill=accent, font=font("Arial Bold.ttf", 22))
    y = 150
    for r in meta_rows:
        d.text((60, y), r, fill=(80, 80, 80), font=font("Arial.ttf", 18)); y += 26
    y += 30
    d.rectangle([60, y, W - 60, y + 36], fill=accent)
    d.text((70, y + 7), "Description", fill=(255, 255, 255), font=font("Arial Bold.ttf", 18))
    d.text((760, y + 7), "Amount", fill=(255, 255, 255), font=font("Arial Bold.ttf", 18))
    y += 50
    for desc, amt in items:
        d.text((70, y), desc, fill=(30, 30, 30), font=font("Arial.ttf", 18))
        d.text((760, y), amt, fill=(30, 30, 30), font=font("Arial.ttf", 18)); y += 40
    d.line([(60, y + 8), (W - 60, y + 8)], fill=(180, 180, 180), width=1)
    d.text((600, y + 26), total_label, fill=accent, font=font("Arial Bold.ttf", 22)); y += 90
    for r in footer_rows:
        d.text((60, y), r, fill=(40, 40, 40), font=font("Arial.ttf", 18)); y += 30
    return img


def gen_form(form_title, subtitle, rows, signer=None):
    W, H = 1000, 1280
    img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
    d.rectangle([40, 40, W - 40, 112], outline=(0, 0, 0), width=2)
    d.text((52, 50), form_title, fill=(0, 0, 0), font=font("Times New Roman Bold.ttf", 34))
    d.text((52, 92), subtitle, fill=(0, 0, 0), font=font("Times New Roman.ttf", 16))
    y = 150
    for lab, val in rows:
        d.rectangle([40, y, W - 40, y + 58], outline=(120, 120, 120), width=1)
        d.text((52, y + 5), lab, fill=(90, 90, 90), font=font("Times New Roman.ttf", 14))
        d.text((60, y + 26), val, fill=(15, 15, 15), font=font("Times New Roman.ttf", 22)); y += 58
    if signer:
        y += 40
        d.text((52, y), "Signature", fill=(90, 90, 90), font=font("Times New Roman.ttf", 16))
        d.text((300, y - 10), signer, fill=(15, 20, 70), font=font("Savoye LET.ttc", 44))
        d.line([(300, y + 44), (720, y + 44)], fill=(80, 80, 80), width=2)
    return img


def gen_letter(recipient, subject, paras, signer, role, company):
    W, H = 1275, 1650
    img = Image.new("RGB", (W, H), (252, 252, 250)); d = ImageDraw.Draw(img)
    m = 110
    d.text((m, 70), company, fill=(25, 40, 75), font=font("Times New Roman Bold.ttf", 30))
    d.text((m, 112), "Audit • Tax • Advisory", fill=(110, 110, 110), font=font("Arial.ttf", 15))
    d.line([(m, 150), (W - m, 150)], fill=(180, 180, 180), width=2)
    y = 200
    for line in recipient:
        d.text((m, y), line, fill=(60, 60, 60), font=font("Times New Roman.ttf", 19)); y += 28
    y += 20
    d.text((m, y), subject, fill=(25, 40, 75), font=font("Times New Roman Bold.ttf", 19)); y += 50
    for p in paras:
        for line in _wrap(d, p, font("Times New Roman.ttf", 19), W - 2 * m):
            d.text((m, y), line, fill=(30, 30, 30), font=font("Times New Roman.ttf", 19)); y += 30
        y += 16
    y += 50
    d.text((m, y), "Sincerely,", fill=(40, 40, 40), font=font("Times New Roman.ttf", 19)); y += 70
    d.text((m + 10, y), signer, fill=(20, 30, 90), font=font("SnellRoundhand.ttc", 46)); y += 90
    d.line([(m, y), (m + 380, y)], fill=(80, 80, 80), width=2)
    d.text((m, y + 8), f"{signer}, {role}", fill=(40, 40, 40), font=font("Arial.ttf", 15))
    return img


def build_synthetic():
    out = []
    out.append(("business_card_1", gen_card(
        "Dara Okafor", "Engagement Partner", "Meridian", "+1 (415) 555-0192",
        "d.okafor@meridian-ap.com", "500 Market St, San Francisco, CA 94105", (180, 30, 45))))
    out.append(("business_card_2", gen_card(
        "Priya Nair", "Senior Tax Associate", "Brightwater", "+1 (312) 555-0148",
        "priya.nair@brightwater-cg.com", "742 Evergreen Terrace, Chicago, IL 60601", (20, 90, 140))))
    out.append(("business_card_3", gen_card(
        "Marcus Lindqvist", "Audit Manager", "Northstar", "+1 (206) 555-0173",
        "m.lindqvist@northstar-llp.com", "1180 Lakeshore Dr, Seattle, WA 98101", (30, 110, 70))))
    out.append(("invoice_1", gen_statement(
        "INVOICE", "Brightwater Consulting Group",
        ["742 Evergreen Terrace, Chicago, IL 60601",
         "billing@brightwater-cg.com  •  (312) 555-0148",
         "Invoice #: INV-20260615   •   EIN: 47-1839204",
         "Bill To: Northgate Industries Inc. — Attn: Eleanor Whitcombe, CFO"],
        [("FY2025 Audit fieldwork", "$182,000.00"), ("Tax advisory (Q1-Q2)", "$ 54,500.00"),
         ("Transfer-pricing study", "$ 28,750.00")],
        "Total Due: $265,250.00",
        ["Remit to: Brightwater Consulting Group",
         "Bank: First National • Routing 071000013 • Acct 5829104471"])))
    out.append(("invoice_2", gen_statement(
        "INVOICE", "Meridian Advisory Partners LLP",
        ["500 Market St, San Francisco, CA 94105",
         "ar@meridian-ap.com  •  (415) 555-0192",
         "Invoice #: MAP-2026-0412   •   EIN: 81-4470925",
         "Bill To: Cascade Retail Holdings — Attn: T. Alvarez, Controller"],
        [("Internal controls review", "$ 96,200.00"), ("SOC 1 readiness", "$ 41,000.00")],
        "Total Due: $137,200.00",
        ["Remit to: Meridian Advisory Partners LLP",
         "Bank: Pacific Union • Routing 121000358 • Acct 4471092250"], accent=(120, 30, 40))))
    out.append(("financial_statement_1", gen_statement(
        "STATEMENT OF INCOME", "Northgate Industries Inc.  —  FY2025 (USD)",
        ["Prepared by Meridian Advisory Partners LLP", "Confidential — Draft for discussion"],
        [("Revenue", "48,210,000"), ("Cost of goods sold", "(29,640,000)"),
         ("Operating expenses", "(11,205,000)"), ("Interest expense", "(1,340,000)"),
         ("Income tax", "(1,560,000)")],
        "Net income: 4,465,000",
        ["Auditor: Dara Okafor, Engagement Partner",
         "Engagement: A-3200 • EIN: 47-1102938"])))
    out.append(("w9_form_1", gen_form(
        "Form W-9", "Request for Taxpayer Identification Number and Certification",
        [("1  Name", "Marcus T. Lindqvist"), ("2  Business name", "Lindqvist Advisory LLC"),
         ("5  Address", "1180 Lakeshore Dr, Suite 400"), ("6  City, state, ZIP", "Seattle, WA 98101"),
         ("Social security number", "412-55-9930"), ("Employer ID number (EIN)", "91-2048817")],
        signer="Marcus Lindqvist")))
    out.append(("form_1099_1", gen_form(
        "Form 1099-NEC", "Nonemployee Compensation — Tax Year 2025",
        [("PAYER  Brightwater Consulting Group", "742 Evergreen Terrace, Chicago, IL 60601"),
         ("PAYER TIN", "47-1839204"), ("RECIPIENT  Dara Okafor", "500 Market St, San Francisco, CA"),
         ("RECIPIENT TIN", "528-91-0447"), ("1  Nonemployee compensation", "$ 96,000.00")],
        signer=None)))
    out.append(("engagement_letter_1", gen_letter(
        ["Ms. Eleanor Whitcombe", "Chief Financial Officer, Northgate Industries Inc."],
        "Re: Engagement for Fiscal Year 2025 Audit Services",
        ["We are pleased to confirm our understanding of the services we are to provide to "
         "Northgate Industries Inc. for the year ending December 31, 2025, including the audit "
         "of the consolidated financial statements.",
         "Our estimated fees for this engagement are $245,000, billed monthly. Please sign below "
         "to indicate your agreement with the terms set out in this letter."],
        "Jonathan R. Albright", "Engagement Partner", "MERIDIAN ADVISORY PARTNERS LLP")))
    out.append(("audit_memo_1", gen_letter(
        ["To: File", "From: Priya Nair, Senior Associate", "Engagement: Cascade Retail Holdings"],
        "Memo: Revenue Recognition — Substantive Testing Conclusions",
        ["Substantive testing of the revenue cycle was performed over a sample of 60 transactions. "
         "Cut-off procedures and analytical review did not identify material misstatement.",
         "Management contact: T. Alvarez (t.alvarez@cascade-retail.com, 206-555-0241). "
         "Recommend partner sign-off prior to issuing the opinion."],
        "Priya Nair", "Senior Associate", "MERIDIAN ADVISORY PARTNERS LLP")))
    out.append(("bank_statement_1", gen_statement(
        "ACCOUNT STATEMENT", "Pacific Union Bank — Business Checking",
        ["Account holder: Lindqvist Advisory LLC", "Account: 4471-0922-50  •  Routing: 121000358",
         "Statement period: May 1 – May 31, 2026"],
        [("05/03  Client wire — Cascade Retail", "+ $41,000.00"),
         ("05/12  Payroll", "- $18,420.00"), ("05/20  Office lease", "- $6,500.00")],
        "Closing balance: $112,840.00",
        ["Questions? (800) 555-0117 • support@pacificunion.example"], accent=(15, 70, 60))))
    out.append(("nda_page_1", gen_letter(
        ["Between: Meridian Advisory Partners LLP", "And: Northgate Industries Inc."],
        "MUTUAL NON-DISCLOSURE AGREEMENT",
        ["This Agreement is entered into as of June 1, 2026 by and between the parties identified "
         "above. Each party may disclose Confidential Information to the other solely for the "
         "purpose of the FY2025 audit engagement.",
         "Authorized signatories: Dara Okafor (Partner) and Eleanor Whitcombe (CFO). Notices to "
         "legal@meridian-ap.com and e.whitcombe@northgate.example."],
        "Dara Okafor", "Engagement Partner", "MERIDIAN ADVISORY PARTNERS LLP")))
    return out


def main():
    manifest = []
    counts, faces = {}, 0
    for cat, lic, url in REAL_SOURCES:
        if cat == "face" and faces >= FACE_CAP:
            continue
        img = fetch(url)
        if img is None:
            continue
        if cat == "face":
            faces += 1
        counts[cat] = counts.get(cat, 0) + 1
        fname = f"{cat}_{counts[cat]:02d}.jpg"
        img.save(os.path.join(HERE, fname), quality=90)
        manifest.append({"file": fname, "category": cat, "source": "real",
                         "license": lic, "url": url, "size": list(img.size)})
        print(f"  ok   {fname:18s} <- {url}")

    for name, img in build_synthetic():
        fname = f"{name}.png"
        img.save(os.path.join(HERE, fname))
        manifest.append({"file": fname, "category": "gen", "source": "synthetic",
                         "license": "generated locally (fabricated PII) — freely shareable",
                         "size": list(img.size)})
        print(f"  gen  {fname}")

    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    real_n = sum(1 for m in manifest if m["source"] == "real")
    gen_n = len(manifest) - real_n
    print(f"\nTotal {len(manifest)} images ({real_n} real / {gen_n} synthetic). "
          f"Faces: {faces}. manifest.json written.")


if __name__ == "__main__":
    main()
