import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "report_screenshots")
DOCX_PATH = os.path.join(BASE_DIR, "SENTINEL_AI_Interface_Walkthrough.docx")
PDF_PATH = os.path.join(BASE_DIR, "SENTINEL_AI_Interface_Walkthrough.pdf")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, text_paragraphs, border_color="0E7060", bg_color="F0FDF4"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.8)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}>'
                        f'<w:top w:val="none"/>'
                        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
                        f'<w:bottom w:val="none"/>'
                        f'<w:right w:val="none"/>'
                        f'</w:tcBorders>')
    tcPr.append(borders)
    
    cell.paragraphs[0].text = "" # clear default
    for i, tp in enumerate(text_paragraphs):
        p = cell.add_paragraph() if i > 0 else cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(tp[0])
        run.font.name = "Segoe UI"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
        if len(tp) > 1 and tp[1]: # bold
            run.font.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_heading(p, text, font_size=14, bold=True, color_rgb=(0x0E, 0x70, 0x60), space_before=14, space_after=6):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Segoe UI"
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color_rgb)
    return run

def add_body_p(doc, text, bold_prefix="", space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Segoe UI"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    run = p.add_run(text)
    run.font.name = "Segoe UI"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
    return p

def add_bullet_item(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    r_pre = p.add_run(bold_prefix)
    r_pre.font.name = "Segoe UI"
    r_pre.font.size = Pt(9.5)
    r_pre.font.bold = True
    r_pre.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    run = p.add_run(text)
    run.font.name = "Segoe UI"
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
    return p

def add_image_figure(doc, img_name, caption_text, width_inches=6.5):
    img_path = os.path.join(SCREENSHOTS_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"Warning: image {img_path} not found")
        return
    
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    
    p_img.add_run().add_picture(img_path, width=Inches(width_inches))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    
    r_num = p_cap.add_run("Figure: ")
    r_num.font.name = "Segoe UI"
    r_num.font.size = Pt(8.5)
    r_num.font.bold = True
    r_num.font.color.rgb = RGBColor(0x0E, 0x70, 0x60)
    
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = "Segoe UI"
    r_cap.font.size = Pt(8.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

def build_document():
    doc = Document()
    
    # Set page margins to 0.7 inches
    for sec in doc.sections:
        sec.top_margin = Inches(0.7)
        sec.bottom_margin = Inches(0.7)
        sec.left_margin = Inches(0.7)
        sec.right_margin = Inches(0.7)
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # TITLE & HEADER BLOCK
    # ═══════════════════════════════════════════════════════════════════════════
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    r_t1 = p_title.add_run("SENTINEL AI")
    r_t1.font.name = "Segoe UI"
    r_t1.font.size = Pt(26)
    r_t1.font.bold = True
    r_t1.font.color.rgb = RGBColor(0x0F, 0x14, 0x1B)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(10)
    r_s = p_sub.add_run("Public Safety Surveillance System — Interface & Operational Walkthrough Report")
    r_s.font.name = "Segoe UI"
    r_s.font.size = Pt(13)
    r_s.font.bold = True
    r_s.font.color.rgb = RGBColor(0x0E, 0x70, 0x60)

    # Metadata Banner Table
    meta_table = doc.add_table(rows=2, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    meta_table.columns[0].width = Inches(2.3)
    meta_table.columns[1].width = Inches(2.4)
    meta_table.columns[2].width = Inches(2.3)
    
    cells_data = [
        [("System Version", True), ("Primary Detector", True), ("Visual Context Engine", True)],
        [("Sentinel AI v2.0 (Release)", False), ("YOLOv9s + YOLOv8-Pose (17 Pts)", False), ("Qwen2.5-VL 4B (Local Ollama)", False)],
    ]
    for r_idx, row in enumerate(meta_table.rows):
        for c_idx, cell in enumerate(row.cells):
            set_cell_background(cell, "F9FAFB" if r_idx == 1 else "111827")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            t, b = cells_data[r_idx][c_idx]
            run = p.add_run(t)
            run.font.name = "Segoe UI"
            run.font.size = Pt(8.5)
            run.font.bold = b
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r_idx == 0 else RGBColor(0x1F, 0x29, 0x37)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Executive Overview
    add_callout(doc, [
        ("Executive Overview & Architecture:", True),
        ("Sentinel AI is a hybrid intelligent surveillance platform integrating high-throughput deep learning detectors (YOLOv9s object recognition, 17-keypoint skeleton pose estimation, spatial trajectory tracking) with local multimodal Vision-Language Models (Qwen2.5-VL via Ollama). By employing an automated 80% Visual AI / 20% Rule decision fusion, the system prevents false alarms while preserving critical threat detection across high-stakes environments.", False)
    ], border_color="0E7060", bg_color="F0FDF4")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: OPERATIONS DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "1. Operations Dashboard (Overview)", 14, True, (0x0F, 0x14, 0x1B), 14, 4)
    add_body_p(doc, "The Operations Dashboard serves as the central mission control interface for surveillance operators, providing an aggregated, real-time snapshot of the overall surveillance environment. It presents live operational metrics, active video streams, detection throughput (FPS), tracked entities, and a non-modal stream of recent security incidents.")

    add_image_figure(doc, "01_overview_dashboard.png", "Figure 1: Sentinel AI Operations Dashboard displaying real-time metrics cards, recent security event logs, and diagnostic telemetry.")

    add_bullet_item(doc, "Status Metric Cards: ", "Instant visibility into live stream status, processing throughput (30.0 FPS real-time execution), active tracked individuals, and cumulative recorded incidents.")
    add_bullet_item(doc, "Recent Events Telemetry Stream: ", "Real-time chronological log tracking detected behavioral deviations including dwell time, meander ratios, grid revisits, and severity ratings.")
    add_bullet_item(doc, "Runtime Diagnostics Pane: ", "Live inspection of underlying hardware acceleration, model weights, local Ollama engine availability, and active fusion parameters.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: VIDEO DEMO SURVEILLANCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "2. Video Demo — AI Surveillance Analysis", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "The Video Demo module is the core investigative environment of Sentinel AI. It executes deep inference on recorded video footage, displaying bounding boxes, posture skeletons, confidence distribution histograms, Cyber-HUD active targets, and local vision context.")

    # Scenario 2A
    format_heading(doc.add_paragraph(), "Scenario 2A: Entryway Access & Doorway Inspection", 11.5, True, (0x0E, 0x70, 0x60), 10, 2)
    add_body_p(doc, "In this scenario, a visitor approaches a residential doorway and checks the door handle before stepping back. Traditional motion heuristics triggered a loitering warning due to extended dwell (>30s). However, the multimodal Visual AI verified benign intent (no tools, no physical force), and the 80% AI / 20% Rules fusion correctly aligned the threat level to NORMAL (AI) / Medium Risk (45%).")

    add_image_figure(doc, "02a_video_demo_entryway_inspection.png", "Figure 2A: Video Demo analyzing residential entryway access, showing green AI-override bounding box, posture skeleton, and cyber-HUD sidebar.")

    # Scenario 2B
    format_heading(doc.add_paragraph(), "Scenario 2B: Pedestrian Walkway & Bridge Interaction", 11.5, True, (0x0E, 0x70, 0x60), 10, 2)
    add_body_p(doc, "In public pedestrian areas, motion heuristics often produce false alerts due to repeated pacing or social conversations. Here, two individuals converse on a walkway; one exits while the other stands calmly. The visual AI confirms peaceful public interaction, resolving the discrepancy and downgrading global risk to NORMAL (20%).")

    add_image_figure(doc, "02b_video_demo_bridge_interaction.png", "Figure 2B: Video Demo evaluating pedestrian bridge scenario with automated threat resolution and real-time posture estimation.")

    add_bullet_item(doc, "17-Keypoint Posture Skeleton: ", "Real-time human pose estimation mapping joints to identify sitting, standing, bending, or fallen states with confidence estimation.")
    add_bullet_item(doc, "Confidence Distribution Histogram: ", "Dynamic 10-bin histogram displaying detector confidence spread across active frames.")
    add_bullet_item(doc, "Cyber-HUD Sidebar: ", "Comprehensive per-person tracking telemetry (Track ID, dwell time, meander ratio, revisit count, risk tier, and active threat tags).")
    add_bullet_item(doc, "Local Vision Smart Context: ", "Real-time AI reasoning pane summarizing observed physical actions and the combined fusion decision.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: DETAILED BEHAVIORAL ASSESSMENT REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "3. Multimodal AI Behavioral Assessment Dossier", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "Clicking 'Full report' within the Video Context interface launches the dedicated Detailed Behavioral Assessment modal window. This document provides a forensic, plain-language breakdown of all evaluated multi-frame visual evidence, environmental conditions, and task hypotheses.")

    add_image_figure(doc, "03_video_context_full_report.png", "Figure 3: Forensic Behavioral Assessment Report modal displaying action sequences, ambient lighting, task hypotheses, and classification justification.")

    add_bullet_item(doc, "Session Review & Voting Breakdown: ", "Explicit record of the 80% AI / 20% Rules weighting and evidence window sampling coverage.")
    add_bullet_item(doc, "Chronological Action Sequence: ", "Timestamped step-by-step description of subject movements and interactions.")
    add_bullet_item(doc, "Environmental & Illumination Context: ", "Quantitative image brightness score (0–255), day/night appearance, and visible scene fixtures.")
    add_bullet_item(doc, "Supported Task Hypotheses: ", "Probability estimates for benign hypotheses (e.g. Visiting/Door Checking 85%, Waiting 75%) versus criminal intent (<5%).")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: LIVE CAMERA MONITORING
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "4. Live Camera Monitoring", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "The Live Monitoring interface connects directly to live CCTV devices, RTSP video streams, and USB cameras for real-time security surveillance. Operators can dynamically adjust detection confidence thresholds, monitor processing latency, and receive immediate visual and acoustic alerts upon event triggers.")

    add_image_figure(doc, "04_live_monitoring.png", "Figure 4: Live CCTV Camera Monitoring interface with camera selector, confidence slider, and telemetry HUD.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: INCIDENTS MANAGEMENT & AUDIT TRAIL
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "5. Incident Management & Audit Trail", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "The Incidents view delivers a comprehensive, non-modal audit log of all security events recorded during active monitoring shifts. Every security event is immutably logged with timestamp, category, severity, subject ID, acknowledgment status, and snapshot file paths.")

    add_image_figure(doc, "05_incident_management.png", "Figure 5: Incident Management table featuring multi-column sorting, category/severity filtering, and incident inspection pane.")

    add_bullet_item(doc, "Multi-Column Sorting & Filtering: ", "Filter events by category (e.g. REPEATED_MOVEMENT, LONG_DWELL, SUDDEN_RUNNING) and severity (LOW to CRITICAL).")
    add_bullet_item(doc, "Operator Workflow Actions: ", "Dedicated buttons to Acknowledge Selected incidents, Dismiss false positives, or open full evidence reports.")
    add_bullet_item(doc, "Forensic Detail Panel: ", "Direct inspection of exact dwell duration, meander values, revisit counts, and linked snapshot file paths.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: FORENSIC INCIDENT EVIDENCE DOSSIER
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "6. Forensic Incident Evidence Dossier", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "Selecting an incident and clicking 'View Report' opens a standalone forensic dossier window. This document combines the original heuristic trigger telemetry, video timestamp, snapshot path, and the matching visual AI scene assessment for evidentiary compliance.")

    add_image_figure(doc, "06_incident_report_modal.png", "Figure 6: Standalone Incident Evidence Dossier modal displaying forensic telemetry and matching AI scene assessments.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: SESSION STATISTICS & THREAT ANALYTICS
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "7. Session Statistics & Threat Analytics", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "The Session Statistics dashboard provides security supervisors with aggregate performance and threat metrics across monitoring shifts. It summarizes total tracked subjects, specific behavioral alerts, and provides one-click PDF export functionality.")

    add_image_figure(doc, "07_session_statistics.png", "Figure 7: Session Statistics dashboard displaying large KPI metric cards, alert history feed, and export controls.")

    add_bullet_item(doc, "KPI Counter Cards: ", "Real-time summary counters for Persons Tracked, Loitering Alerts, Rapid Movement, Repeated Pacing, and Possible Falls.")
    add_bullet_item(doc, "Alert History Feed: ", "Chronological history of all triggered security anomalies with exact timestamps and spatial metrics.")
    add_bullet_item(doc, "Executive Export: ", "One-click 'Export PDF Report' functionality generating publication-ready compliance summaries.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: DATASET MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "8. Dataset Management & Version Browser", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "The Dataset interface allows computer vision engineers and operators to review surveillance datasets, annotation splits, and class distributions (Person, Backpack, Handbag, Suitcase) across multiple dataset versions (v1i through v4i).")

    add_image_figure(doc, "08_dataset_management.png", "Figure 8: Dataset Management tab displaying dataset version browser, class labels, and train/val/test splits.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 9: MODEL TRAINING & CONVERGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "9. Model Training & Convergence Analytics", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "Sentinel AI includes an integrated model fine-tuning suite allowing in-application training of custom YOLO models. Users can configure architectures, training hyperparameters, and monitor live convergence curves in real time.")

    add_image_figure(doc, "09_model_training.png", "Figure 9: Model Training interface with hyperparameter configuration, live loss & mAP@50 curves, and training logs.")

    add_bullet_item(doc, "Hyperparameter Configuration: ", "Flexible controls for Model Size (yolov9s.pt, yolov8n.pt), Epochs (50), Image Size (640), and Batch Size (16).")
    add_bullet_item(doc, "Real-Time Matplotlib Curves: ", "Dual live charts tracking Train Loss convergence and Validation mAP@50 performance exceeding 0.915 mAP.")
    add_bullet_item(doc, "Streaming Execution Log: ", "Detailed console output recording GPU memory consumption, per-epoch loss components, precision, recall, and model checkpoint paths.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 10: SYSTEM SETTINGS & RUNTIME CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "10. System Settings & Runtime Configuration", 14, True, (0x0F, 0x14, 0x1B), 16, 4)
    add_body_p(doc, "The Settings page exposes the active configuration thresholds governing the heuristic detection engine, allowing operators to calibrate sensitivity to specific deployment environments.")

    add_image_figure(doc, "10_system_settings.png", "Figure 10: System Settings tab displaying active detection thresholds, dwell times, and model paths.")

    # ═══════════════════════════════════════════════════════════════════════════
    # APPENDIX: SUMMARY TABLE OF ASSETS
    # ═══════════════════════════════════════════════════════════════════════════
    format_heading(doc.add_paragraph(), "Appendix: Summary Table of Generated Interface Visuals", 14, True, (0x0F, 0x14, 0x1B), 18, 6)
    
    table_data = [
        ("Filename", "Resolution", "Module", "Primary Content / Functional Role"),
        ("01_overview_dashboard.png", "1456 × 884", "Dashboard", "System status cards, FPS, active incidents, runtime telemetry"),
        ("02a_video_demo_entryway_inspection.png", "1456 × 884", "Video Demo", "Entryway access analysis, pose skeleton, 80/20 AI fusion override"),
        ("02b_video_demo_bridge_interaction.png", "1456 × 884", "Video Demo", "Pedestrian bridge scenario, pose overlay, normalized risk gauge"),
        ("03_video_context_full_report.png", "916 × 759", "Visual AI", "Detailed behavioral assessment modal, action sequence, hypotheses"),
        ("04_live_monitoring.png", "1456 × 884", "Live CCTV", "Live camera controls, confidence slider, target details HUD"),
        ("05_incident_management.png", "1456 × 884", "Incidents", "Sortable incident audit table, category/severity filters, telemetry"),
        ("06_incident_report_modal.png", "916 × 719", "Forensics", "Standalone incident evidence dossier modal and forensic audit logs"),
        ("07_session_statistics.png", "1456 × 884", "Analytics", "Large KPI cards, alert history feed, automated PDF export button"),
        ("08_dataset_management.png", "1456 × 884", "Dataset", "Surveillance dataset version browser, class splits, annotation counts"),
        ("09_model_training.png", "1456 × 884", "Training", "Custom YOLO training, loss & mAP@50 convergence plots, log stream"),
        ("10_system_settings.png", "1456 × 884", "Settings", "Detection thresholds, dwell limits, fall frames, alert cooldowns")
    ]
    
    app_table = doc.add_table(rows=len(table_data), cols=4)
    app_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    app_table.autofit = False
    app_table.columns[0].width = Inches(2.2)
    app_table.columns[1].width = Inches(0.9)
    app_table.columns[2].width = Inches(0.9)
    app_table.columns[3].width = Inches(3.0)
    
    for r_idx, row in enumerate(app_table.rows):
        is_hdr = (r_idx == 0)
        for c_idx, cell in enumerate(row.cells):
            set_cell_background(cell, "0F141B" if is_hdr else ("F9FAFB" if r_idx % 2 == 1 else "FFFFFF"))
            set_cell_margins(cell, top=50, bottom=50, left=70, right=70)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx in (0, 3) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            t = table_data[r_idx][c_idx]
            run = p.add_run(t)
            run.font.name = "Segoe UI"
            run.font.size = Pt(8.0)
            run.font.bold = is_hdr or (c_idx == 0)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if is_hdr else RGBColor(0x1F, 0x29, 0x37)

    doc.save(DOCX_PATH)
    print(f"Successfully generated Word document: {DOCX_PATH}")

def convert_to_pdf():
    # Use PowerShell COM automation to invoke Microsoft Word and save as PDF
    ps_cmd = f"""
    $docx = "{DOCX_PATH}";
    $pdf = "{PDF_PATH}";
    $word = New-Object -ComObject Word.Application;
    $word.Visible = $false;
    $doc = $word.Documents.Open($docx);
    $doc.SaveAs([ref]$pdf, [ref]17); # 17 = wdFormatPDF
    $doc.Close();
    $word.Quit();
    """
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
    if os.path.exists(PDF_PATH):
        print(f"Successfully converted to PDF: {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")
    else:
        print(f"PDF conversion error: {res.stderr}")

if __name__ == "__main__":
    build_document()
    convert_to_pdf()

