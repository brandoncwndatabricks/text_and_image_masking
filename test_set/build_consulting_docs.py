"""Generate synthetic consulting slides + one-page text documents (freely shareable).

Adds document-style test content like the original samples:
  - 4 consulting SLIDES (landscape): title, engagement team (with headshot photos),
    key findings (with bar chart), transformation roadmap.
  - 5 one-page TEXT DOCS (portrait, also exported as PDF): advisory letter,
    due-diligence memo, tax opinion letter, a report WITH AN EMBEDDED PHOTO, and
    an audit-findings page with an embedded chart.

Headshots are real Unsplash photos (Unsplash License — free to use/share); all
names/numbers are fabricated. Output written into this directory.
"""
from __future__ import annotations
import io, os, urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
FD = "/System/Library/Fonts/Supplemental"
def F(name, s): return ImageFont.truetype(os.path.join(FD, name), s)
A = lambda s: F("Arial.ttf", s); AB = lambda s: F("Arial Bold.ttf", s)
G = lambda s: F("Georgia.ttf", s); GB = lambda s: F("Georgia Bold.ttf", s)
TR = lambda s: F("Times New Roman.ttf", s); TRB = lambda s: F("Times New Roman Bold.ttf", s)
SIG = lambda s: F("SnellRoundhand.ttc", s)
NAVY=(20,34,68); RED=(176,30,45); Gray=(90,90,90); LGray=(150,150,150)
UA={"User-Agent":"Mozilla/5.0"}

HEADSHOTS = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500",
    "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=500",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=500",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=500",
    "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=500",
]
def fetch(url):
    return Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())).convert("RGB")

def circle(img, d):
    s = min(img.size); img = img.crop(((img.width-s)//2,(img.height-s)//2,(img.width+s)//2,(img.height+s)//2)).resize((d,d), Image.LANCZOS)
    mask = Image.new("L",(d,d),0); ImageDraw.Draw(mask).ellipse([0,0,d,d],fill=255)
    out = Image.new("RGB",(d,d),(255,255,255)); out.paste(img,(0,0),mask); return out, mask

def logo(d, x, y, color=RED, label="Meridian"):
    d.ellipse([x,y,x+46,y+46], fill=color); d.rectangle([x+13,y+13,x+33,y+33], fill=(255,255,255))
    d.text((x+58,y+4), label, fill=color, font=AB(30))
    d.text((x+58,y+38), "Advisory Partners", fill=Gray, font=A(14))

def wrap(d, t, fnt, w):
    out=[]; cur=""
    for word in t.split():
        if d.textlength((cur+" "+word).strip(), font=fnt)<=w: cur=(cur+" "+word).strip()
        else: out.append(cur); cur=word
    if cur: out.append(cur)
    return out

def paras(d, x, y, plist, fnt, w, lh, fill=(35,35,35)):
    for p in plist:
        for ln in wrap(d,p,fnt,w):
            d.text((x,y),ln,fill=fill,font=fnt); y+=lh
        y+=14
    return y

def save(img, name, pdf=False):
    img.save(os.path.join(HERE, name+".png"))
    if pdf:
        img.convert("RGB").save(os.path.join(HERE, name+".pdf"), "PDF", resolution=150)
    print("  gen", name + (".png/.pdf" if pdf else ".png"))

# ── Consulting slides (1280x720) ─────────────────────────────────────────────
def slide_title():
    W,H=1600,900; img=Image.new("RGB",(W,H),(248,249,251)); d=ImageDraw.Draw(img)
    d.rectangle([0,0,W,14], fill=RED)
    logo(d, 90, 80)
    d.text((90,300),"Operating Model Review", fill=NAVY, font=AB(72))
    d.text((90,390),"Cascade Retail Holdings — FY2026", fill=Gray, font=A(38))
    d.line([(90,500),(620,500)], fill=RED, width=4)
    d.text((90,540),"Prepared by: Dara Okafor, Engagement Partner", fill=(60,60,60), font=A(28))
    d.text((90,580),"d.okafor@meridian-ap.com  •  June 15, 2026", fill=Gray, font=A(24))
    d.text((90,H-60),"Strictly Private & Confidential", fill=LGray, font=A(22))
    save(img,"slide_strategy_title")

def slide_team():
    W,H=1600,900; img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img)
    logo(d,90,60); d.text((90,150),"Engagement Team", fill=NAVY, font=AB(56))
    people=[("Dara Okafor","Engagement Partner","d.okafor@meridian-ap.com"),
            ("Priya Nair","Senior Manager","p.nair@meridian-ap.com"),
            ("Marcus Lindqvist","Manager","m.lindqvist@meridian-ap.com"),
            ("Elena Whitcombe","Senior Associate","e.whitcombe@meridian-ap.com")]
    heads=[fetch(HEADSHOTS[i]) for i in range(4)]
    cw=320; x0=90; y=300
    for i,(name,role,email) in enumerate(people):
        x=x0+i*cw; c,m=circle(heads[i],200); img.paste(c,(x+50,y),m)
        d.text((x+150,y+230),name, fill=(20,20,20), font=AB(26), anchor="ma")
        d.text((x+150,y+265),role, fill=RED, font=A(20), anchor="ma")
        d.text((x+150,y+295),email, fill=Gray, font=A(16), anchor="ma")
    d.text((90,H-60),"Meridian Advisory Partners LLP  —  Confidential", fill=LGray, font=A(22))
    save(img,"slide_engagement_team")

def slide_findings():
    W,H=1600,900; img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img)
    logo(d,90,60); d.text((90,150),"Key Findings — Cost Structure", fill=NAVY, font=AB(52))
    # bar chart
    bx,by,bw,bh=110,300,620,420; d.line([(bx,by),(bx,by+bh)],fill=(120,120,120),width=2); d.line([(bx,by+bh),(bx+bw,by+bh)],fill=(120,120,120),width=2)
    bars=[("FY23",0.55),("FY24",0.72),("FY25",0.63),("FY26E",0.88)]
    bwid=110; gap=40
    for i,(lab,v) in enumerate(bars):
        x=bx+40+i*(bwid+gap); top=by+bh-int(bh*v)
        d.rectangle([x,top,x+bwid,by+bh], fill=RED if i==3 else NAVY)
        d.text((x+bwid//2,by+bh+10),lab, fill=(60,60,60), font=A(20), anchor="ma")
        d.text((x+bwid//2,top-26),f"{int(v*100)}", fill=(40,40,40), font=AB(20), anchor="ma")
    # bullets
    tx=820
    d.text((tx,300),"Observations", fill=NAVY, font=AB(30))
    bl=["Overhead grew 33% over 3 years, outpacing revenue.",
        "Top 2 cost centers account for 61% of spend.",
        "Vendor consolidation could yield $4.2M annually.",
        "FY26 trajectory exceeds budget by 12%."]
    y=360
    for b in bl:
        d.ellipse([tx,y+8,tx+12,y+20],fill=RED)
        for ln in wrap(d,b,A(24),W-tx-110): d.text((tx+28,y),ln,fill=(40,40,40),font=A(24)); y+=34
        y+=14
    d.text((90,H-60),"Cascade Retail Holdings  —  Strictly Confidential", fill=LGray, font=A(22))
    save(img,"slide_key_findings")

def slide_roadmap():
    W,H=1600,900; img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img)
    logo(d,90,60); d.text((90,150),"Transformation Roadmap", fill=NAVY, font=AB(56))
    phases=[("Phase 1","Diagnostic","Q3 2026"),("Phase 2","Redesign","Q4 2026"),
            ("Phase 3","Pilot","Q1 2027"),("Phase 4","Scale","Q2 2027")]
    x0=110; y=340; w=320; h=240
    for i,(p,t,q) in enumerate(phases):
        x=x0+i*(w+20); col=NAVY if i%2==0 else RED
        d.rounded_rectangle([x,y,x+w,y+h], radius=18, fill=col)
        d.text((x+w//2,y+50),p, fill=(255,255,255), font=AB(34), anchor="ma")
        d.text((x+w//2,y+120),t, fill=(255,255,255), font=A(28), anchor="ma")
        d.text((x+w//2,y+170),q, fill=(220,220,220), font=A(22), anchor="ma")
    d.text((90,H-60),"Prepared for Cascade Retail Holdings by Meridian Advisory Partners LLP", fill=LGray, font=A(22))
    save(img,"slide_roadmap")

# ── One-page text docs (1000x1300) ───────────────────────────────────────────
def letterhead(d, m=110):
    logo(d, m, 70); d.line([(m,150),(1275-m,150)], fill=(180,180,180), width=2)

def doc_advisory_letter():
    W,H=1275,1650; img=Image.new("RGB",(W,H),(252,252,250)); d=ImageDraw.Draw(img); m=110
    letterhead(d,m)
    d.text((m,200),"June 15, 2026", fill=(60,60,60), font=TR(19)); y=250
    y=paras(d,m,y,["Mr. Thomas Alvarez","Controller, Cascade Retail Holdings","880 Harbor Blvd, Long Beach, CA 90802"],TR(19),W-2*m,28)
    y+=10; d.text((m,y),"Re: Advisory Recommendations — Working Capital Optimization", fill=NAVY, font=TRB(19)); y+=46
    y=paras(d,m,y,["Dear Mr. Alvarez,",
      "Following our review of Cascade Retail Holdings' working capital position, we recommend a phased program to release approximately $4.2M in trapped cash over the next two quarters, primarily through vendor-term renegotiation and inventory rationalization.",
      "Our analysis indicates days-payable-outstanding can be extended from 38 to 52 days without disrupting key supplier relationships. We further recommend consolidating the current 14 logistics vendors to 4 preferred partners.",
      "We would be pleased to discuss these recommendations at your convenience. Please contact the engagement partner directly at d.okafor@meridian-ap.com or (415) 555-0192."],TR(19),W-2*m,30)
    y+=50; d.text((m,y),"Sincerely,", fill=(40,40,40), font=TR(19)); y+=70
    d.text((m+10,y),"Dara Okafor", fill=NAVY, font=SIG(46)); y+=92
    d.line([(m,y),(m+360,y)], fill=(80,80,80), width=2); d.text((m,y+8),"Dara Okafor, Engagement Partner", fill=(40,40,40), font=A(15))
    d.text((m,H-70),"Meridian Advisory Partners LLP  —  Strictly Private & Confidential", fill=LGray, font=A(14))
    save(img,"doc_advisory_letter", pdf=True)

def doc_diligence_memo():
    W,H=1275,1650; img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img); m=110
    d.text((m,70),"CONFIDENTIAL MEMORANDUM", fill=NAVY, font=TRB(30)); d.line([(m,120),(W-m,120)],fill=(170,170,170),width=2)
    rows=[("To:","Engagement File — Project Cascade"),("From:","Priya Nair, Senior Manager"),
          ("Date:","June 12, 2026"),("Re:","Financial Due Diligence — Quality of Earnings")]
    y=160
    for k,v in rows: d.text((m,y),k,fill=Gray,font=TRB(18)); d.text((m+120,y),v,fill=(20,20,20),font=TR(18)); y+=34
    y+=20
    y=paras(d,m,y,["We performed a quality-of-earnings analysis over the trailing twelve months ended April 30, 2026. Reported EBITDA of $11.4M includes $1.8M of non-recurring items that we have normalized out.",
      "Key adjustments: (i) one-time legal settlement of $0.9M; (ii) owner compensation above market of $0.6M; (iii) deferred revenue cut-off of $0.3M. Adjusted EBITDA is $9.6M.",
      "Customer concentration remains a risk: the top customer (Northgate Industries) represents 22% of revenue. Contact: e.whitcombe@northgate.example.",
      "No material weaknesses in internal controls were identified. We recommend proceeding to confirmatory diligence."],TR(18),W-2*m,28)
    d.text((m,H-70),"Meridian Advisory Partners LLP  —  Privileged & Confidential", fill=LGray, font=A(14))
    save(img,"doc_diligence_memo", pdf=True)

def doc_tax_opinion():
    W,H=1275,1650; img=Image.new("RGB",(W,H),(252,252,250)); d=ImageDraw.Draw(img); m=110
    letterhead(d,m)
    d.text((m,200),"June 9, 2026", fill=(60,60,60), font=TR(19)); y=250
    y=paras(d,m,y,["Lindqvist Advisory LLC","1180 Lakeshore Dr, Suite 400, Seattle, WA 98101","EIN: 91-2048817"],TR(19),W-2*m,28)
    y+=10; d.text((m,y),"Re: Tax Opinion — Section 1031 Like-Kind Exchange", fill=NAVY, font=TRB(19)); y+=46
    y=paras(d,m,y,["To Whom It May Concern,",
      "Based on the facts presented and applicable provisions of the Internal Revenue Code, it is our opinion that the proposed exchange of the Seattle warehouse property qualifies as a like-kind exchange under Section 1031, provided the replacement property is identified within 45 days and acquired within 180 days.",
      "This opinion is rendered solely for Lindqvist Advisory LLC (taxpayer ID 412-55-9930) and may not be relied upon by any other party. It is based on current law, which is subject to change."],TR(19),W-2*m,30)
    y+=50; d.text((m,y),"Respectfully,", fill=(40,40,40), font=TR(19)); y+=70
    d.text((m+10,y),"Marcus Lindqvist", fill=NAVY, font=SIG(44)); y+=88
    d.line([(m,y),(m+380,y)], fill=(80,80,80), width=2); d.text((m,y+8),"Marcus Lindqvist, Tax Partner", fill=(40,40,40), font=A(15))
    d.text((m,H-70),"Meridian Advisory Partners LLP  —  Confidential Tax Advice", fill=LGray, font=A(14))
    save(img,"doc_tax_opinion", pdf=True)

def doc_report_with_photo():
    W,H=1275,1650; img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img); m=110
    letterhead(d,m)
    d.text((m,200),"Client Spotlight — Leadership Profile", fill=NAVY, font=TRB(30)); y=270
    y=paras(d,m,y,["As part of our FY2026 operating model review, we conducted in-depth interviews with the executive team at Cascade Retail Holdings. This profile summarizes our discussion with the Chief Executive Officer."],TR(19),W-2*m,30)
    # embedded photo with caption
    photo=fetch(HEADSHOTS[1]); ph=photo.resize((300,360),Image.LANCZOS)
    img.paste(ph,(W-m-300,y)); d.rectangle([W-m-300,y,W-m,y+360],outline=(150,150,150),width=2)
    d.text((W-m-300,y+368),"Eleanor Whitcombe, CEO", fill=(40,40,40), font=AB(16))
    d.text((W-m-300,y+392),"e.whitcombe@cascade-retail.com", fill=Gray, font=A(14))
    y=paras(d,m,y,["Ms. Whitcombe outlined three strategic priorities: margin recovery, supply-chain resilience, and a digital storefront refresh. She noted that working capital remains the binding constraint on growth.",
      "\"We have the demand,\" she said, \"what we need is the operating discipline to fund it from within.\" Our recommendations in the accompanying report are calibrated to that objective.",
      "Direct line: (562) 555-0173. Personal assistant: j.romero@cascade-retail.com."],TR(19),(W-2*m)-340,30)
    d.text((m,H-70),"Meridian Advisory Partners LLP  —  Strictly Private & Confidential", fill=LGray, font=A(14))
    save(img,"doc_report_with_photo", pdf=True)

def doc_audit_findings():
    W,H=1275,1650; img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img); m=110
    d.text((m,70),"Audit Findings Summary", fill=NAVY, font=TRB(30)); d.line([(m,120),(W-m,120)],fill=(170,170,170),width=2)
    d.text((m,150),"Northgate Industries Inc. — FY2025  •  Engagement A-3200", fill=Gray, font=A(16)); y=210
    y=paras(d,m,y,["Our audit identified three findings requiring management attention. Severity is summarized in the chart below; none rise to the level of a material weakness."],TR(18),W-2*m,28)
    # embedded chart
    bx,by,bw,bh=m,y,W-2*m,260; d.line([(bx,by+bh),(bx+bw,by+bh)],fill=(120,120,120),width=2)
    cats=[("Revenue cut-off",0.4),("Inventory count",0.7),("IT access review",0.95)]
    bwid=180; gap=120
    for i,(lab,v) in enumerate(cats):
        x=bx+80+i*(bwid+gap); top=by+bh-int(bh*v)
        d.rectangle([x,top,x+bwid,by+bh], fill=[(0,150,90),(255,170,0),RED][i])
        for j,ln in enumerate(wrap(d,lab,A(16),bwid+40)): d.text((x+bwid//2,by+bh+10+j*20),ln,fill=(60,60,60),font=A(16),anchor="ma")
    y=by+bh+90
    y=paras(d,m,y,["Recommendations have been discussed with the CFO, Eleanor Whitcombe, and the Controller, Thomas Alvarez (t.alvarez@northgate.example). Remediation is targeted for Q3 2026.",
      "Prepared by Priya Nair; reviewed and approved by Dara Okafor, Engagement Partner."],TR(18),W-2*m,28)
    d.text((m,H-70),"Meridian Advisory Partners LLP  —  Confidential", fill=LGray, font=A(14))
    save(img,"doc_audit_findings", pdf=True)

if __name__ == "__main__":
    print("Generating consulting slides + text docs...")
    slide_title(); slide_team(); slide_findings(); slide_roadmap()
    doc_advisory_letter(); doc_diligence_memo(); doc_tax_opinion(); doc_report_with_photo(); doc_audit_findings()
    print("Done.")
