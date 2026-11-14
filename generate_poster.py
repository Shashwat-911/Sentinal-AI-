import os
import sys
from reportlab.lib.pagesizes import A3
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "SENTINEL_AI_Poster_A3.pdf")
PNG_PATH = os.path.join(BASE_DIR, "SENTINEL_AI_Poster_A3.png")
PREVIEW_PATH = os.path.join(BASE_DIR, "SENTINEL_AI_Poster_A3_preview.png")
CREST_PATH = os.path.join(BASE_DIR, "dsatm_crest.png")

# Page Dimensions (A3 Portrait in points)
PAGE_WIDTH, PAGE_HEIGHT = A3  # 841.89 x 1190.55

# Color Palette matching reference poster
C_DARK_BLUE   = HexColor("#1E3A8A")   # College title / deep headers
C_TEXT_MAIN   = HexColor("#1E293B")   # Main body text
C_TEXT_MUTED  = HexColor("#475569")   # Subtitles and muted text
C_TEAL_DARK   = HexColor("#0E7060")   # Department and accent headers
C_TEAL_MED    = HexColor("#0D9488")   # Teal accents
C_TEAL_LIGHT  = HexColor("#E6FFFA")   # Card fill accents
C_GREEN_PILL  = HexColor("#DCFCE7")   # Green pill fill
C_GREEN_TEXT  = HexColor("#15803D")   # Green pill text / borders
C_BLUE_PILL   = HexColor("#DBEAFE")   # Blue pill fill
C_BLUE_TEXT   = HexColor("#1D4ED8")   # Blue pill text / borders
C_CARD_BG     = HexColor("#FFFFFF")   # Pure white cards
C_CARD_ALT    = HexColor("#F8FAFC")   # Subtle card gray
C_BORDER_GRAY = HexColor("#CBD5E1")   # Card borders
C_BORDER_TEAL = HexColor("#99F6E4")   # Architecture borders
C_WAVE_TEAL   = HexColor("#8CE0D0")   # Top/bottom wave shapes
C_WAVE_LIGHT  = HexColor("#D7F5EF")   # Light wave

def draw_rounded_card(c, x, y, w, h, radius=8, fill_color=C_CARD_BG, border_color=C_BORDER_GRAY, border_width=1):
    c.saveState()
    c.setFillColor(fill_color)
    c.setStrokeColor(border_color)
    c.setLineWidth(border_width)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    c.restoreState()

def draw_pill(c, x, y, w, h, text, fill_color, text_color, font_size=10, icon_type=None):
    c.saveState()
    r = h / 2.0
    c.setFillColor(fill_color)
    c.setStrokeColor(text_color)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1)
    
    # Optional Icon
    tx = x + r
    if icon_type == "target":
        c.setStrokeColor(text_color)
        c.setLineWidth(1.2)
        c.circle(x + 12, y + h/2, 4.5, fill=0, stroke=1)
        c.circle(x + 12, y + h/2, 1.8, fill=1, stroke=0)
        tx = x + 24
    elif icon_type == "bulb":
        c.setFillColor(text_color)
        c.circle(x + 12, y + h/2 + 1, 3.5, fill=1, stroke=0)
        c.rect(x + 10.5, y + h/2 - 4, 3, 2.5, fill=1, stroke=0)
        tx = x + 24
    elif icon_type == "chart":
        c.setFillColor(text_color)
        c.rect(x + 8,  y + h/2 - 4, 2.2, 5, fill=1, stroke=0)
        c.rect(x + 11, y + h/2 - 4, 2.2, 7.5, fill=1, stroke=0)
        c.rect(x + 14, y + h/2 - 4, 2.2, 10, fill=1, stroke=0)
        tx = x + 24
    elif icon_type == "gear":
        c.setStrokeColor(text_color)
        c.setLineWidth(1.2)
        c.circle(x + 12, y + h/2, 4, fill=0, stroke=1)
        c.circle(x + 12, y + h/2, 1.5, fill=1, stroke=0)
        tx = x + 24
    elif icon_type == "chip":
        c.setStrokeColor(text_color)
        c.setLineWidth(1.0)
        c.rect(x + 8, y + h/2 - 4, 8, 8, fill=0, stroke=1)
        c.setFillColor(text_color)
        c.circle(x + 12, y + h/2, 1.8, fill=1, stroke=0)
        tx = x + 24
    elif icon_type == "layers":
        c.setStrokeColor(text_color)
        c.setLineWidth(1.1)
        c.rect(x + 8, y + h/2 - 3, 8, 6, fill=0, stroke=1)
        tx = x + 24
    else:
        tx = x + w / 2.0
    
    c.setFont("Helvetica-Bold", font_size)
    c.setFillColor(text_color)
    if icon_type:
        c.drawString(tx, y + (h - font_size) / 2 + 1, text)
    else:
        c.drawCentredString(tx, y + (h - font_size) / 2 + 1, text)
    c.restoreState()

def draw_triangle(c, p1, p2, p3, fill=1, stroke=1):
    p = c.beginPath()
    p.moveTo(p1[0], p1[1])
    p.lineTo(p2[0], p2[1])
    p.lineTo(p3[0], p3[1])
    p.close()
    c.drawPath(p, fill=fill, stroke=stroke)

def draw_top_waves(c):
    c.saveState()
    c.setFillColor(C_WAVE_LIGHT)
    p = c.beginPath()
    p.moveTo(0, PAGE_HEIGHT)
    p.lineTo(260, PAGE_HEIGHT)
    p.curveTo(200, PAGE_HEIGHT - 35, 120, PAGE_HEIGHT - 65, 0, PAGE_HEIGHT - 60)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    
    p2 = c.beginPath()
    p2.moveTo(PAGE_WIDTH, PAGE_HEIGHT)
    p2.lineTo(PAGE_WIDTH - 240, PAGE_HEIGHT)
    p2.curveTo(PAGE_WIDTH - 180, PAGE_HEIGHT - 35, PAGE_WIDTH - 100, PAGE_HEIGHT - 60, PAGE_WIDTH, PAGE_HEIGHT - 50)
    p2.close()
    c.drawPath(p2, fill=1, stroke=0)
    
    c.setFillColor(C_WAVE_TEAL)
    p3 = c.beginPath()
    p3.moveTo(0, PAGE_HEIGHT)
    p3.lineTo(150, PAGE_HEIGHT)
    p3.curveTo(110, PAGE_HEIGHT - 20, 60, PAGE_HEIGHT - 40, 0, PAGE_HEIGHT - 35)
    p3.close()
    c.drawPath(p3, fill=1, stroke=0)
    c.restoreState()

def draw_bottom_waves(c):
    c.saveState()
    c.setFillColor(C_WAVE_LIGHT)
    p = c.beginPath()
    p.moveTo(0, 0)
    p.lineTo(300, 0)
    p.curveTo(220, 35, 120, 55, 0, 50)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    
    p2 = c.beginPath()
    p2.moveTo(PAGE_WIDTH, 0)
    p2.lineTo(PAGE_WIDTH - 300, 0)
    p2.curveTo(PAGE_WIDTH - 220, 35, PAGE_WIDTH - 120, 55, PAGE_WIDTH, 50)
    p2.close()
    c.drawPath(p2, fill=1, stroke=0)
    
    c.setFillColor(C_WAVE_TEAL)
    p3 = c.beginPath()
    p3.moveTo(0, 0)
    p3.lineTo(180, 0)
    p3.curveTo(130, 20, 70, 35, 0, 30)
    p3.close()
    c.drawPath(p3, fill=1, stroke=0)
    
    p4 = c.beginPath()
    p4.moveTo(PAGE_WIDTH, 0)
    p4.lineTo(PAGE_WIDTH - 180, 0)
    p4.curveTo(PAGE_WIDTH - 130, 20, PAGE_WIDTH - 70, 35, PAGE_WIDTH, 30)
    p4.close()
    c.drawPath(p4, fill=1, stroke=0)
    c.restoreState()

def build_pdf():
    c = canvas.Canvas(PDF_PATH, pagesize=A3)
    c.setTitle("SENTINEL AI - Project Exhibition Poster")

    # 1. Background Waves
    draw_top_waves(c)
    draw_bottom_waves(c)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. COLLEGE HEADER BLOCK (Y: 1080 - 1175)
    # ═══════════════════════════════════════════════════════════════════════════
    # DSI / DSATM Crest Logo
    if os.path.exists(CREST_PATH):
        c.drawImage(CREST_PATH, 42, 1098, width=74, height=74, mask='auto')

    # College Title Text
    c.setFont("Helvetica-Bold", 19)
    c.setFillColor(C_DARK_BLUE)
    c.drawString(132, 1152, "DAYANANDA SAGAR ACADEMY OF TECHNOLOGY & MANAGEMENT")

    c.setFont("Helvetica", 10)
    c.setFillColor(C_TEXT_MUTED)
    c.drawString(132, 1136, "(Autonomous Institute under VTU) | Udayapura, Kanakapura Road, Bengaluru - 560082")

    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(C_TEAL_DARK)
    c.drawString(132, 1120, "DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING")

    # Event Header Pill
    draw_pill(c, 100, 1082, 642, 22, "OPEN DAY 2026   |   FINAL YEAR PROJECT EXHIBITION   |   13 – 14 NOVEMBER",
              fill_color=HexColor("#E0E7FF"), text_color=HexColor("#1E3A8A"), font_size=9.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. PROJECT TITLE BLOCK (Y: 980 - 1068)
    # ═══════════════════════════════════════════════════════════════════════════
    # Left Icon Badge (Surveillance Camera / Shield)
    c.saveState()
    c.setFillColor(HexColor("#0D9488"))
    c.circle(76, 1024, 30, fill=1, stroke=0)
    c.setFillColor(HexColor("#CCFBF1"))
    c.circle(76, 1024, 27, fill=1, stroke=0)
    c.setFillColor(HexColor("#0F766E"))
    c.roundRect(62, 1016, 20, 15, 2, fill=1, stroke=0)
    c.setFillColor(HexColor("#CCFBF1"))
    c.circle(72, 1023.5, 4.5, fill=1, stroke=0)
    c.setFillColor(HexColor("#0F766E"))
    c.circle(72, 1023.5, 2.2, fill=1, stroke=0)
    c.rect(79, 1020, 8, 7, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#0F766E"))
    c.setLineWidth(1.8)
    c.arc(88, 1014, 100, 1034, -45, 90)
    c.arc(92, 1009, 108, 1039, -45, 90)
    c.restoreState()

    # Main Project Title (sized to avoid overlap with right graphic)
    c.setFont("Helvetica-Bold", 19.5)
    c.setFillColor(C_DARK_BLUE)
    c.drawString(124, 1040, "SENTINEL AI: AI-Based Intelligent Surveillance & Emergency")
    c.drawString(124, 1016, "Threat Detection System for Public Safety & Smart Spaces")

    # Sub-pill
    draw_pill(c, 124, 990, 410, 20, "(Edge Detection, 17-Keypoint Pose Tracking & Multimodal Visual Reasoning)",
              fill_color=HexColor("#F1F5F9"), text_color=HexColor("#334155"), font_size=8.5)

    # Right Smart City / Shield Graphic
    c.saveState()
    c.setFillColor(HexColor("#0E7060"))
    c.rect(740, 995, 16, 42, fill=1, stroke=0)
    c.rect(760, 1005, 14, 32, fill=1, stroke=0)
    c.rect(778, 988, 18, 49, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    for by in range(1002, 1032, 7):
        c.rect(744, by, 3, 4, fill=1, stroke=0)
        c.rect(749, by, 3, 4, fill=1, stroke=0)
    for by in range(1012, 1032, 7):
        c.rect(764, by, 3, 4, fill=1, stroke=0)
    for by in range(996, 1032, 7):
        c.rect(782, by, 3, 4, fill=1, stroke=0)
        c.rect(788, by, 3, 4, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#0D9488"))
    c.setLineWidth(1.8)
    c.arc(715, 1020, 735, 1040, 30, 90)
    c.arc(710, 1015, 740, 1045, 30, 90)
    c.restoreState()

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. TEAM MEMBERS & FACULTY GUIDE STRIP (Y: 922 - 976)
    # ═══════════════════════════════════════════════════════════════════════════
    draw_rounded_card(c, 34, 922, 774, 54, radius=6, fill_color=C_CARD_ALT, border_color=C_BORDER_GRAY, border_width=0.8)

    students = [
        ("SHASHWAT YADAV", "1DT23AI051"),
        ("AVIRAL SAHU", "1DT23AI011"),
        ("DHEERAJ A U", "1DT23AI016"),
        ("KUSHAL N SETTY", "1DT23AI026")
    ]
    col_w = 774 / 4.0
    for idx, (name, usn) in enumerate(students):
        cx = 34 + idx * col_w + col_w / 2.0
        c.setFont("Helvetica-Bold", 10.5)
        c.setFillColor(C_TEXT_MAIN)
        c.drawCentredString(cx, 955, name)
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(C_TEAL_DARK)
        c.drawCentredString(cx, 942, usn)
        
        if idx > 0:
            c.setStrokeColor(C_BORDER_GRAY)
            c.setLineWidth(0.6)
            c.line(34 + idx * col_w, 938, 34 + idx * col_w, 968)

    c.setFont("Helvetica", 8.5)
    c.setFillColor(C_TEXT_MUTED)
    guide_str = "Under the guidance of Prof. Indu B, Asst. Professor, Dept. of AIML   |   HOD: Dr. Sandhya N   |   Principal: Dr. M. Ravishankar"
    c.drawCentredString(PAGE_WIDTH / 2.0, 928, guide_str)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. ROW 1: THE PROBLEM | OUR APPROACH | RESULTS (Y: 785 - 912)
    # ═══════════════════════════════════════════════════════════════════════════
    card_w = (774 - 24) / 3.0
    r1_y = 785
    r1_h = 127

    # 5A. THE PROBLEM
    draw_rounded_card(c, 34, r1_y, card_w, r1_h, radius=8)
    draw_pill(c, 44, r1_y + r1_h - 18, 140, 24, "THE PROBLEM", C_GREEN_PILL, C_GREEN_TEXT, font_size=9.5, icon_type="target")
    
    p_lines = [
        "• Operators experience severe fatigue monitoring",
        "  multiple CCTV feeds, missing active threats.",
        "• Legacy motion triggers produce >75% false",
        "  alarms on benign tasks (door checking, pacing).",
        "• Rigid heuristics lack visual semantic context,",
        "  causing delayed dispatch in break-ins & falls."
    ]
    c.setFont("Helvetica", 8.8)
    c.setFillColor(C_TEXT_MAIN)
    curr_y = r1_y + r1_h - 36
    for line in p_lines:
        c.drawString(44, curr_y, line)
        curr_y -= 13.5

    # 5B. OUR APPROACH
    draw_rounded_card(c, 34 + card_w + 12, r1_y, card_w, r1_h, radius=8)
    draw_pill(c, 44 + card_w + 12, r1_y + r1_h - 18, 150, 24, "OUR APPROACH", C_BLUE_PILL, C_BLUE_TEXT, font_size=9.5, icon_type="bulb")
    
    a_lines = [
        "• Hybrid two-tier pipeline uniting real-time edge",
        "  YOLOv9s detection with local multimodal VLM.",
        "• 17-Keypoint YOLOv8-Pose tracking captures",
        "  postural dynamics, falls, and abnormal gaits.",
        "• Automated 80% AI / 20% Rules Decision Fusion",
        "  overrides false alarms while preserving safety."
    ]
    c.setFont("Helvetica", 8.8)
    c.setFillColor(C_TEXT_MAIN)
    curr_y = r1_y + r1_h - 36
    for line in a_lines:
        c.drawString(44 + card_w + 12, curr_y, line)
        curr_y -= 13.5

    # 5C. RESULTS
    draw_rounded_card(c, 34 + (card_w + 12) * 2, r1_y, card_w, r1_h, radius=8)
    draw_pill(c, 44 + (card_w + 12) * 2, r1_y + r1_h - 18, 120, 24, "RESULTS", C_GREEN_PILL, C_GREEN_TEXT, font_size=9.5, icon_type="chart")

    results_data = [
        ("Video Inference Throughput", "30.0 FPS"),
        ("Person Detection (mAP@50)", "91.50 %"),
        ("Posture / Fall Latency", "< 0.40 s"),
        ("False Alarm Reduction", "> 75.0 %"),
        ("AI Discrepancy Resolution", "100 %"),
        ("System Response Time", "Real-Time")
    ]
    curr_y = r1_y + r1_h - 36
    for k, v in results_data:
        c.setFont("Helvetica", 8.5)
        c.setFillColor(C_TEXT_MUTED)
        c.drawString(44 + (card_w + 12) * 2, curr_y, k)
        c.setFont("Helvetica-Bold", 9.2)
        c.setFillColor(HexColor("#0F766E") if "%" in v or "FPS" in v or "Real" in v else C_TEXT_MAIN)
        c.drawRightString(34 + (card_w + 12) * 2 + card_w - 12, curr_y, v)
        curr_y -= 14

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. MIDDLE SECTION: ARCHITECTURE & FLOW (LEFT) | FACTORS & AI (RIGHT)
    # ═══════════════════════════════════════════════════════════════════════════
    left_w = 466
    right_w = 296
    
    # ---------------------------------------------------------------------------
    # 6A. SYSTEM ARCHITECTURE (Left Top, Y: 475 - 770, H: 295)
    # ---------------------------------------------------------------------------
    arch_y = 475
    arch_h = 295
    draw_rounded_card(c, 34, arch_y, left_w, arch_h, radius=8)
    draw_pill(c, 44, arch_y + arch_h - 18, 205, 24, "SYSTEM ARCHITECTURE", C_BLUE_PILL, C_BLUE_TEXT, font_size=9.5, icon_type="layers")

    # Layer 1: Input Video Stream
    draw_rounded_card(c, 46, arch_y + 224, left_w - 24, 28, radius=4, fill_color=HexColor("#EFF6FF"), border_color=HexColor("#93C5FD"))
    c.setFont("Helvetica-Bold", 8.8)
    c.setFillColor(HexColor("#1E3A8A"))
    c.drawCentredString(34 + left_w/2.0, arch_y + 233, "VIDEO INGESTION LAYER (Live IP / USB RTSP Cameras  •  Benchmark Surveillance Video Datasets)")

    # Down arrow
    c.setStrokeColor(HexColor("#64748B"))
    c.setLineWidth(1.2)
    c.line(34 + left_w/2.0, arch_y + 224, 34 + left_w/2.0, arch_y + 212)
    draw_triangle(c, (34 + left_w/2.0 - 3, arch_y + 215), (34 + left_w/2.0 + 3, arch_y + 215), (34 + left_w/2.0, arch_y + 210), fill=1, stroke=1)

    # Layer 2: Split Edge Processing (Detection + Pose)
    sub_box_w = (left_w - 36) / 2.0
    draw_rounded_card(c, 46, arch_y + 144, sub_box_w, 64, radius=4, fill_color=HexColor("#F0FDF4"), border_color=HexColor("#86EFAC"))
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(HexColor("#166534"))
    c.drawCentredString(46 + sub_box_w/2.0, arch_y + 192, "YOLOv9s Person Detection & Tracking")
    c.setFont("Helvetica", 7.8)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(46 + sub_box_w/2.0, arch_y + 178, "• Deep bounding box localization (mAP 91.5%)")
    c.drawCentredString(46 + sub_box_w/2.0, arch_y + 165, "• IoU Hungarian centroid matching")
    c.drawCentredString(46 + sub_box_w/2.0, arch_y + 152, "• Trajectory meander & velocity tracking")

    draw_rounded_card(c, 46 + sub_box_w + 12, arch_y + 144, sub_box_w, 64, radius=4, fill_color=HexColor("#FEF3C7"), border_color=HexColor("#FDE68A"))
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(HexColor("#92400E"))
    c.drawCentredString(46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 192, "YOLOv8-Pose 17-Keypoint Estimation")
    c.setFont("Helvetica", 7.8)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 178, "• Real-time skeletal joint orientation")
    c.drawCentredString(46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 165, "• Multi-frame fall verification window")
    c.drawCentredString(46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 152, "• Posture states (Standing, Sitting, Fallen)")

    # Arrows converging to Multimodal AI Context Engine
    c.setStrokeColor(HexColor("#64748B"))
    c.line(46 + sub_box_w/2.0, arch_y + 144, 46 + sub_box_w/2.0, arch_y + 134)
    c.line(46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 144, 46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 134)
    c.line(46 + sub_box_w/2.0, arch_y + 134, 46 + sub_box_w + 12 + sub_box_w/2.0, arch_y + 134)
    c.line(34 + left_w/2.0, arch_y + 134, 34 + left_w/2.0, arch_y + 122)
    draw_triangle(c, (34 + left_w/2.0 - 3, arch_y + 125), (34 + left_w/2.0 + 3, arch_y + 125), (34 + left_w/2.0, arch_y + 120), fill=1, stroke=1)

    # Layer 3: Multimodal Vision AI Context
    draw_rounded_card(c, 46, arch_y + 72, left_w - 24, 48, radius=4, fill_color=HexColor("#F5F3FF"), border_color=HexColor("#DDD6FE"))
    c.setFont("Helvetica-Bold", 8.8)
    c.setFillColor(HexColor("#5B21B6"))
    c.drawCentredString(34 + left_w/2.0, arch_y + 106, "LOCAL MULTIMODAL VISION AI ENGINE (Qwen2.5-VL 4B via Ollama)")
    c.setFont("Helvetica", 7.8)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(34 + left_w/2.0, arch_y + 92, "• Multi-frame temporal scene reasoning  •  Ambient lighting & environmental context")
    c.drawCentredString(34 + left_w/2.0, arch_y + 80, "• Behavioral intent classification: Visiting / Waiting vs. Forced Tampering / Hostile Breach")

    # Down arrow
    c.setStrokeColor(HexColor("#64748B"))
    c.line(34 + left_w/2.0, arch_y + 72, 34 + left_w/2.0, arch_y + 60)
    draw_triangle(c, (34 + left_w/2.0 - 3, arch_y + 63), (34 + left_w/2.0 + 3, arch_y + 63), (34 + left_w/2.0, arch_y + 58), fill=1, stroke=1)

    # Layer 4: 80% AI / 20% Rules Decision Fusion & Dispatch
    draw_rounded_card(c, 46, arch_y + 10, left_w - 24, 48, radius=4, fill_color=HexColor("#ECFDF5"), border_color=HexColor("#6EE7B7"))
    c.setFont("Helvetica-Bold", 8.8)
    c.setFillColor(HexColor("#065F46"))
    c.drawCentredString(34 + left_w/2.0, arch_y + 44, "DECISION FUSION & SURVEILLANCE CYBER-HUD DISPATCH")
    c.setFont("Helvetica", 7.8)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(34 + left_w/2.0, arch_y + 30, "• 80% Visual AI + 20% Rule Fusion Core  •  Automated Discrepancy Resolution")
    c.drawCentredString(34 + left_w/2.0, arch_y + 18, "• Cyber-HUD Telemetry Overlay  •  Incident Dossiers  •  Acoustic Alarms  •  PDF Compliance Export")

    # ---------------------------------------------------------------------------
    # 6B. RESPONSE FLOW (Left Bottom, Y: 218 - 465, H: 247)
    # ---------------------------------------------------------------------------
    flow_y = 218
    flow_h = 247
    draw_rounded_card(c, 34, flow_y, left_w, flow_h, radius=8)
    draw_pill(c, 44, flow_y + flow_h - 18, 175, 24, "RESPONSE FLOW", C_GREEN_PILL, C_GREEN_TEXT, font_size=9.5, icon_type="target")

    # Start Box
    draw_rounded_card(c, 46, flow_y + 172, 64, 28, radius=14, fill_color=HexColor("#DCFCE7"), border_color=HexColor("#166534"))
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(HexColor("#166534"))
    c.drawCentredString(78, flow_y + 182, "▶ START")

    # Arrow 1 to Detect & Track
    c.setStrokeColor(HexColor("#64748B"))
    c.line(110, flow_y + 186, 126, flow_y + 186)
    draw_triangle(c, (123, flow_y + 189), (123, flow_y + 183), (128, flow_y + 186), fill=1, stroke=1)

    # Box 2: Detect & Track
    draw_rounded_card(c, 128, flow_y + 168, 142, 36, radius=4, fill_color=C_CARD_ALT, border_color=C_BORDER_GRAY)
    c.setFont("Helvetica-Bold", 8.0)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(199, flow_y + 189, "Video Ingestion & Tracking")
    c.setFont("Helvetica", 7.2)
    c.setFillColor(C_TEXT_MUTED)
    c.drawCentredString(199, flow_y + 176, "YOLOv9s Detect + Centroid Track")

    # Arrow 2 to Pose Analysis
    c.line(270, flow_y + 186, 286, flow_y + 186)
    draw_triangle(c, (283, flow_y + 189), (283, flow_y + 183), (288, flow_y + 186), fill=1, stroke=1)

    # Box 3: Pose Analysis
    draw_rounded_card(c, 288, flow_y + 168, 142, 36, radius=4, fill_color=C_CARD_ALT, border_color=C_BORDER_GRAY)
    c.setFont("Helvetica-Bold", 8.0)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(359, flow_y + 189, "Posture & Gait Analysis")
    c.setFont("Helvetica", 7.2)
    c.setFillColor(C_TEXT_MUTED)
    c.drawCentredString(359, flow_y + 176, "YOLOv8-Pose (17 Keypoints)")

    # Down arrow from Pose Analysis to Diamond
    c.line(359, flow_y + 168, 359, flow_y + 144)
    draw_triangle(c, (356, flow_y + 147), (362, flow_y + 147), (359, flow_y + 142), fill=1, stroke=1)

    # Diamond: Anomaly Triggered? (Accurate position inside card)
    diam_cx = 359
    diam_cy = flow_y + 118
    diam_rx = 48
    diam_ry = 22
    c.saveState()
    c.setFillColor(HexColor("#FEF3C7"))
    c.setStrokeColor(HexColor("#D97706"))
    c.setLineWidth(1)
    p_diam = c.beginPath()
    p_diam.moveTo(diam_cx, diam_cy + diam_ry)
    p_diam.lineTo(diam_cx + diam_rx, diam_cy)
    p_diam.lineTo(diam_cx, diam_cy - diam_ry)
    p_diam.lineTo(diam_cx - diam_rx, diam_cy)
    p_diam.close()
    c.drawPath(p_diam, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 7.6)
    c.setFillColor(HexColor("#92400E"))
    c.drawCentredString(diam_cx, diam_cy + 3, "Anomaly Flagged?")
    c.setFont("Helvetica", 6.8)
    c.drawCentredString(diam_cx, diam_cy - 7, "(Dwell > 30s / Motion)")
    c.restoreState()

    # Branch: NO -> Continue monitoring
    c.setStrokeColor(HexColor("#16A34A"))
    c.line(diam_cx + diam_rx, diam_cy, diam_cx + diam_rx + 20, diam_cy)
    draw_triangle(c, (diam_cx + diam_rx + 17, diam_cy + 3), (diam_cx + diam_rx + 17, diam_cy - 3), (diam_cx + diam_rx + 22, diam_cy), fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(HexColor("#16A34A"))
    c.drawString(diam_cx + diam_rx + 3, diam_cy + 5, "No")

    draw_rounded_card(c, diam_cx + diam_rx + 24, flow_y + 104, 58, 28, radius=4, fill_color=HexColor("#F0FDF4"), border_color=HexColor("#86EFAC"))
    c.setFont("Helvetica", 7.0)
    c.setFillColor(HexColor("#166534"))
    c.drawCentredString(diam_cx + diam_rx + 53, flow_y + 120, "Telemetry Log")
    c.drawCentredString(diam_cx + diam_rx + 53, flow_y + 110, "& Continue")

    # Branch: YES -> Down to Qwen-VL Reasoning
    c.setStrokeColor(HexColor("#DC2626"))
    c.line(diam_cx, diam_cy - diam_ry, diam_cx, flow_y + 68)
    draw_triangle(c, (diam_cx - 3, flow_y + 71), (diam_cx + 3, flow_y + 71), (diam_cx, flow_y + 66), fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(HexColor("#DC2626"))
    c.drawString(diam_cx + 5, diam_cy - diam_ry - 8, "Yes")

    # Box: Multimodal AI Context Reasoning
    draw_rounded_card(c, 160, flow_y + 36, 260, 30, radius=4, fill_color=HexColor("#F5F3FF"), border_color=HexColor("#DDD6FE"))
    c.setFont("Helvetica-Bold", 8.0)
    c.setFillColor(HexColor("#5B21B6"))
    c.drawCentredString(290, flow_y + 53, "Extract Keyframes ➔ Qwen2.5-VL Context Analysis")
    c.setFont("Helvetica", 7.0)
    c.setFillColor(C_TEXT_MAIN)
    c.drawCentredString(290, flow_y + 42, "Semantic intent analysis: Visiting / Waiting vs. Forced Tampering")

    # Arrow left & down to 80/20 Decision Fusion
    c.setStrokeColor(HexColor("#64748B"))
    c.line(160, flow_y + 51, 130, flow_y + 51)
    c.line(130, flow_y + 51, 130, flow_y + 30)
    draw_triangle(c, (127, flow_y + 33), (133, flow_y + 33), (130, flow_y + 28), fill=1, stroke=1)

    # Box: 80% AI / 20% Decision Fusion & Alert Dispatch
    draw_rounded_card(c, 46, flow_y + 6, 442, 22, radius=4, fill_color=HexColor("#ECFDF5"), border_color=HexColor("#6EE7B7"))
    c.setFont("Helvetica-Bold", 8.0)
    c.setFillColor(HexColor("#065F46"))
    c.drawCentredString(46 + 442/2.0, flow_y + 13, "Weighted 80% AI / 20% Rule Decision Fusion ➔ Risk Tier (Normal / Med / High / Critical) ➔ Dossier Saved")

    # ---------------------------------------------------------------------------
    # 6C. KEY FACTORS (Right Top, Y: 500 - 770, H: 270)
    # ---------------------------------------------------------------------------
    draw_rounded_card(c, 34 + left_w + 12, 500, right_w, 270, radius=8)
    draw_pill(c, 44 + left_w + 12, 500 + 270 - 18, 160, 24, "KEY FACTORS", C_GREEN_PILL, C_GREEN_TEXT, font_size=9.5, icon_type="gear")

    kf_items = [
        ("• Dual-Tier Detection Pipeline: ", "Blends ultra-fast edge tracking with deep multimodal semantic reasoning."),
        ("• 17-Keypoint Skeletal Analysis: ", "Instant detection of falls, sitting posture, and physical distress without latency."),
        ("• Trajectory Telemetry: ", "Continuous tracking of dwell timers, meander ratios, and grid revisit counts."),
        ("• 80% AI / 20% Rule Fusion: ", "Resolves motion rule discrepancies, eliminating false alarms on normal door access."),
        ("• Non-Modal Incident Audit: ", "Immutable incident table with linked snapshot coordinates and forensic logs."),
        ("• Cyber-HUD Overlay: ", "Real-time confidence histogram, active targets list, and global threat severity gauge.")
    ]
    curr_y = 500 + 270 - 36
    for title, desc in kf_items:
        c.setFont("Helvetica-Bold", 8.2)
        c.setFillColor(HexColor("#065F46"))
        c.drawString(44 + left_w + 12, curr_y, title)
        c.setFont("Helvetica", 7.8)
        c.setFillColor(C_TEXT_MAIN)
        words = desc.split()
        l1, l2 = "", ""
        for w in words:
            if len(l1 + w) < 36: l1 += w + " "
            else: l2 += w + " "
        c.drawString(44 + left_w + 12, curr_y - 10, l1.strip())
        if l2:
            c.drawString(44 + left_w + 12, curr_y - 19, l2.strip())
            curr_y -= 9
        curr_y -= 23

    # ---------------------------------------------------------------------------
    # 6D. USE OF AI (Right Bottom, Y: 218 - 490, H: 272)
    # ---------------------------------------------------------------------------
    draw_rounded_card(c, 34 + left_w + 12, 218, right_w, 272, radius=8)
    draw_pill(c, 44 + left_w + 12, 218 + 272 - 18, 140, 24, "USE OF AI", C_BLUE_PILL, C_BLUE_TEXT, font_size=9.5, icon_type="chip")

    ai_items = [
        ("• YOLOv9s Deep Neural Network: ", "State-of-the-art bounding box localization, human shape tracking, and deduplication."),
        ("• YOLOv8-Pose Landmark Model: ", "Maps 17 anatomical keypoints for instant fall detection and postural classification."),
        ("• Qwen2.5-VL Multimodal Model: ", "Performs multi-frame temporal reasoning on video evidence to assess behavioral context."),
        ("• Automated Hypothesis Formulation: ", "Scores plausible task scenarios (e.g. visiting, waiting vs. break-in tampering)."),
        ("• Decision Fusion & Discrepancy: ", "Overrides rigid movement heuristics when visual AI confirms normal social or visitor intent."),
        ("• Automated Incident Dossier: ", "Generates plain-language forensic summaries ready for operator review and audit.")
    ]
    curr_y = 218 + 272 - 36
    for title, desc in ai_items:
        c.setFont("Helvetica-Bold", 8.2)
        c.setFillColor(HexColor("#1E40AF"))
        c.drawString(44 + left_w + 12, curr_y, title)
        c.setFont("Helvetica", 7.8)
        c.setFillColor(C_TEXT_MAIN)
        words = desc.split()
        l1, l2 = "", ""
        for w in words:
            if len(l1 + w) < 36: l1 += w + " "
            else: l2 += w + " "
        c.drawString(44 + left_w + 12, curr_y - 10, l1.strip())
        if l2:
            c.drawString(44 + left_w + 12, curr_y - 19, l2.strip())
            curr_y -= 9
        curr_y -= 23

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. BOTTOM SECTION: SURVEILLANCE & DEPLOYMENT USE CASES (Y: 60 - 208)
    # ═══════════════════════════════════════════════════════════════════════════
    use_y = 60
    use_h = 148
    draw_rounded_card(c, 34, use_y, 774, use_h, radius=8)
    draw_pill(c, 44, use_y + use_h - 18, 275, 24, "SURVEILLANCE & DEPLOYMENT USE CASES", C_BLUE_PILL, C_BLUE_TEXT, font_size=9.5, icon_type="layers")

    use_box_w = (774 - 36) / 2.0

    # Box 1: Public & Campus Safety
    draw_rounded_card(c, 46, use_y + 12, use_box_w, 98, radius=6, fill_color=C_CARD_ALT, border_color=C_BORDER_GRAY)
    c.saveState()
    c.setFillColor(HexColor("#0D9488"))
    c.roundRect(58, use_y + 44, 38, 38, 4, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    p_roof = c.beginPath()
    p_roof.moveTo(58 + 19, use_y + 76)
    p_roof.lineTo(58 + 32, use_y + 63)
    p_roof.lineTo(58 + 6,  use_y + 63)
    p_roof.close()
    c.drawPath(p_roof, fill=1, stroke=0)
    c.rect(58 + 11, use_y + 49, 16, 14, fill=1, stroke=0)
    c.restoreState()

    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColor(HexColor("#0F766E"))
    c.drawString(106, use_y + 84, "Smart Campus & Public Infrastructure")
    c.setFont("Helvetica", 8.2)
    c.setFillColor(C_TEXT_MAIN)
    c.drawString(106, use_y + 69, "• University campuses, academic hallways, & dormitory perimeters")
    c.drawString(106, use_y + 55, "• Pedestrian bridges, transit walkways, and open public plazas")
    c.drawString(106, use_y + 41, "• Real-time fall detection for elderly care & physical emergency response")
    c.drawString(106, use_y + 27, "• Social crowd behavior monitoring without false panic alarms")

    # Box 2: Commercial & High-Security
    draw_rounded_card(c, 46 + use_box_w + 12, use_y + 12, use_box_w, 98, radius=6, fill_color=C_CARD_ALT, border_color=C_BORDER_GRAY)
    c.saveState()
    c.setFillColor(HexColor("#1E40AF"))
    c.roundRect(46 + use_box_w + 24, use_y + 44, 38, 38, 4, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    p_sh = c.beginPath()
    p_sh.moveTo(46 + use_box_w + 43, use_y + 76)
    p_sh.lineTo(46 + use_box_w + 54, use_y + 68)
    p_sh.lineTo(46 + use_box_w + 54, use_y + 55)
    p_sh.curveTo(46 + use_box_w + 54, use_y + 48, 46 + use_box_w + 43, use_y + 46, 46 + use_box_w + 43, use_y + 46)
    p_sh.curveTo(46 + use_box_w + 43, use_y + 46, 46 + use_box_w + 32, use_y + 48, 46 + use_box_w + 32, use_y + 55)
    p_sh.lineTo(46 + use_box_w + 32, use_y + 68)
    p_sh.close()
    c.drawPath(p_sh, fill=1, stroke=0)
    c.restoreState()

    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColor(HexColor("#1E3A8A"))
    c.drawString(46 + use_box_w + 72, use_y + 84, "Commercial & Industrial Workplaces")
    c.setFont("Helvetica", 8.2)
    c.setFillColor(C_TEXT_MAIN)
    c.drawString(46 + use_box_w + 72, use_y + 69, "• Entryway access control & residential door inspection verification")
    c.drawString(46 + use_box_w + 72, use_y + 55, "• High-security data centers, bank vaults, & restricted facility perimeters")
    c.drawString(46 + use_box_w + 72, use_y + 41, "• Retail store theft detection & warehouse loitering discrimination")
    c.drawString(46 + use_box_w + 72, use_y + 27, "• Tampering vs. authorized maintenance classification via multimodal AI")

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. BOTTOM FOOTER PILL (Y: 22 - 46)
    # ═══════════════════════════════════════════════════════════════════════════
    draw_pill(c, PAGE_WIDTH/2.0 - 150, 24, 300, 22, "INCLUSIVE   •   AUTONOMOUS   •   REAL-TIME   •   SECURE",
              fill_color=HexColor("#E0E7FF"), text_color=HexColor("#1E3A8A"), font_size=9.5)

    c.showPage()
    c.save()
    print(f"Successfully generated Vector PDF Poster: {PDF_PATH}")

def convert_pdf_to_high_res_png():
    doc = fitz.open(PDF_PATH)
    page = doc[0]
    zoom = 300.0 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(PNG_PATH)
    print(f"Successfully rendered 300 DPI Print PNG: {PNG_PATH} ({pix.width}x{pix.height})")
    
    # Also save 1200px width preview for instant viewing
    zoom_prev = 1200.0 / float(page.rect.width)
    mat_prev = fitz.Matrix(zoom_prev, zoom_prev)
    pix_prev = page.get_pixmap(matrix=mat_prev, alpha=False)
    pix_prev.save(PREVIEW_PATH)
    print(f"Successfully rendered Preview PNG: {PREVIEW_PATH} ({pix_prev.width}x{pix_prev.height})")

if __name__ == "__main__":
    build_pdf()
    convert_pdf_to_high_res_png()
