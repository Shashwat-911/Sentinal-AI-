# 🛡️ SENTINEL AI — Intelligent Public Safety Surveillance System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv9](https://img.shields.io/badge/YOLO-v9-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![mAP@50](https://img.shields.io/badge/mAP%4050-93.07%25-00FF88?style=flat&logo=target)](https://github.com/Shashwat-911/Sentinal-AI-)
[![Precision](https://img.shields.io/badge/Precision-94.96%25-00FF88?style=flat)](https://github.com/Shashwat-911/Sentinal-AI-)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SENTINEL AI** is an advanced, real-time computer vision and deep learning surveillance system designed for public transit hubs, airports, train stations, shopping malls, and enterprise facilities. Powered by a **dual-model YOLOv9 inference pipeline**, it detects unattended baggage, monitors suspicious loitering, computes 4-tier risk severity scores, and employs intelligent proximity algorithms to automatically disarm false alarms when owners return.

---

## 📌 Table of Contents

1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [Empirical Performance Metrics](#-empirical-performance-metrics)
4. [Project Structure](#-project-structure)
5. [Hardware & System Requirements](#-hardware--system-requirements)
6. [Step-by-Step Installation & Setup](#-step-by-step-installation--setup)
7. [How to Run](#-how-to-run)
8. [GUI Tabs & Operational Workflow](#-gui-tabs--operational-workflow)
9. [Tunable Parameters](#-tunable-parameters)
10. [Automated Reporting & Incident Logs](#-automated-reporting--incident-logs)
11. [License & Acknowledgements](#-license--acknowledgements)

---

## 🚀 Key Features

- **⚡ Dual-Model Deep Inference Pipeline**:
  - **Custom Fine-Tuned Luggage Model (`best.pt`)**: Detects backpacks, suitcases, handbags, and left luggage with high precision.
  - **Pretrained YOLOv9s Person Model (`yolov9s.pt`)**: Detects and tracks individuals simultaneously in crowded scenes.
- **📍 Centroid Multi-Object Tracking & Smoothing**:
  - Assigns persistent IDs to objects and individuals using Euclidean spatial matching with Exponential Moving Average (EMA) bounding box smoothing and graceful stale track eviction.
- **⏱️ Temporal Dwell-Time Tracking & 4-Tier Risk Scoring**:
  - Continuous timer tracking for every stationary item:
    - 🟢 **LOW (0–33%)**: Normal stationary object
    - 🟠 **MEDIUM (34–66%)**: Approaching unattended threshold
    - 🔴 **HIGH (67–99%)**: Imminent security threat
    - 🚨 **CRITICAL (100%)**: Threat confirmed; triggers audio siren, visual flashing HUD alert, and automated snapshot logging.
- **🧠 Smart Owner-Return Detection (False-Alarm Suppression)**:
  - Continuously measures the Euclidean distance between unattended bags and nearby people. If an owner re-enters the proximity boundary (default `150px`), the alert is disarmed automatically, saving security personnel from false alarms.
- **🚶 Loitering & Suspicious Behavior Detection**:
  - Monitors person dwell time in restricted or surveillance zones, flagging individuals lingering beyond safety thresholds.
- **🖥️ Tactical Cyber-HUD GUI (Tkinter + Matplotlib)**:
  - Dark-mode control room interface featuring live video playback, webcam stream, training monitors with real-time loss/mAP curves, dataset verifier, and statistics dashboard.
- **📄 Instant Auditable Reports (PDF & Excel)**:
  - Automated PDF report generator (`reportlab`) containing timestamped critical incident snapshots and forensic metadata.
  - Structured Excel session logging (`openpyxl`) for continuous audit trails.

---

## 🏛️ System Architecture

```
                                  [ Surveillance Video / RTSP / Live Camera ]
                                                      │
                                                      ▼
                                       [ Frame Acquisition & Preprocessing ]
                                                      │
                         ┌────────────────────────────┴────────────────────────────┐
                         ▼                                                         ▼
        [ Custom YOLOv9 Luggage Detector ]                        [ Pretrained YOLOv9s Person Detector ]
              (models/best.pt)                                              (yolov9s.pt)
                         │                                                         │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
                                       [ Centroid Multi-Object Tracker ]
                                       - Euclidean Distance Association
                                       - Track ID Assignment & Smoothing
                                                      │
                         ┌────────────────────────────┴────────────────────────────┐
                         ▼                                                         ▼
         [ Temporal Dwell-Time Engine ]                            [ Proximity Owner-Return Engine ]
         - Real-time stationary timer                              - Distance between Bag & Person
         - 4-Tier Risk Severity (0-100%)                           - Auto-Cancellation if Dist <= 150px
                         │                                                         │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
                                    [ Tactical HUD & Alert Dispatcher ]
                                    - Bounding Box & Risk Badges
                                    - Flashing Critical Banner
                                    - Audio Siren (winsound)
                                                      │
                         ┌────────────────────────────┴────────────────────────────┐
                         ▼                                                         ▼
            [ Incident Snapshot Capture ]                              [ Audit & Reporting Engine ]
              Saved to incidents/*.jpg                                   - Excel Logs (openpyxl)
                                                                         - PDF Incident Reports (reportlab)
```

---

## 📊 Empirical Performance Metrics

The custom abandoned luggage detection model was trained on annotated public safety surveillance datasets over **30 epochs** using **YOLOv9**.

### 📈 Final Benchmark Summary

| Metric | Score | Performance Implication |
| :--- | :---: | :--- |
| **mAP@50** | **93.07%** | Outstanding detection accuracy under standard IoU threshold |
| **Precision** | **94.96%** | Extremely low false positive rate in complex environments |
| **Recall** | **84.58%** | High capture rate of unattended items across occlusions |
| **mAP@50-95** | **72.86%** | High bounding box localization precision |
| **Train Box Loss** | **0.5878** | Robust spatial convergence |
| **Train Class Loss**| **0.3621** | Distinct object feature separation |
| **Train DFL Loss** | **1.2644** | Fine-grained distribution focal loss convergence |

### 🔬 Epoch Progression Highlights

```
Epoch 01/30 ─── mAP@50: 65.42% │ Precision: 74.53% │ Recall: 60.54% │ Box Loss: 1.4192
Epoch 10/30 ─── mAP@50: 90.36% │ Precision: 90.10% │ Recall: 84.58% │ Box Loss: 1.0716
Epoch 20/30 ─── mAP@50: 93.43% │ Precision: 92.76% │ Recall: 85.26% │ Box Loss: 0.8701
Epoch 30/30 ─── mAP@50: 93.07% │ Precision: 94.96% │ Recall: 84.58% │ Box Loss: 0.5878
```

*(All training curves, confusion matrices, F1 curves, and PR curves are stored in [`models/train/`](models/train/))*

---

## 📂 Project Structure

```
SENTINEL-AI/
│
├── app.py                             # Main application entry point (Tkinter GUI + Pipeline)
├── start_training.py                  # Standalone YOLOv9 training script with live Excel logging
├── test_detection.py                  # Headless benchmark & test harness for video feeds
├── create_presentation.py             # Automated script to generate presentation slides (.pptx)
├── data.yaml                          # YOLO dataset configuration (classes & split paths)
├── requirements.txt                   # Project Python dependencies
├── run.bat                            # Windows 1-click execution launcher
├── .gitignore                         # Git ignore rules for virtualenvs & large media
├── README.md                          # Comprehensive documentation
│
├── models/                            # Trained weights & evaluation artifacts
│   ├── best.pt                        # Best luggage detection model weights
│   ├── train_training_log.xlsx        # Epoch-by-epoch training logs
│   └── train/                         # Ultralytics training run outputs
│       ├── BoxF1_curve.png            # F1 score vs confidence curve
│       ├── BoxPR_curve.png            # Precision-Recall curve
│       ├── BoxP_curve.png             # Precision vs confidence curve
│       ├── BoxR_curve.png             # Recall vs confidence curve
│       ├── confusion_matrix.png       # Normalized confusion matrix
│       ├── results.csv                # Complete numerical training metrics
│       ├── results.png                # Combined training loss and mAP plots
│       └── weights/
│           ├── best.pt                # Checkpoint weights (highest mAP)
│           └── last.pt                # Final epoch checkpoint
│
├── yolov9s.pt                         # Pretrained YOLOv9s model for person detection
│
├── incidents/                         # Auto-captured critical incident snapshots
│   ├── incident_BAG1_CRITICAL_*.jpg
│   └── ...
│
├── test_videos/                       # Sample surveillance footage for evaluation
│   ├── LeftBag.mpg                    # Benchmark test video for left luggage detection
│   └── sample_frame_*.jpg             # Sample frame visual evaluations
│
└── dataset/                           # Dataset structure & configuration
    └── data.yaml                      # Dataset definition
```

---

## 💻 Hardware & System Requirements

### Recommended Hardware
- **OS**: Windows 10 / 11, Ubuntu 20.04+, or macOS
- **CPU**: Intel Core i5 / i7 (8th Gen+) or AMD Ryzen 5 / 7
- **GPU**: NVIDIA GeForce GTX 1660 / RTX 2060 / RTX 3060 or higher (CUDA 11.8 / 12.x supported)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Camera**: Standard USB Webcam or IP RTSP Camera Stream

---

## 🛠️ Step-by-Step Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shashwat-911/Sentinal-AI-.git
cd Sentinal-AI-
```

### 2. Create a Virtual Environment
It is recommended to use Python 3.10, 3.11, or 3.12:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(If utilizing NVIDIA GPU with CUDA, ensure the appropriate PyTorch CUDA build is installed from [pytorch.org](https://pytorch.org/get-started/locally/))*:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎯 How to Run

### 1. Launch Main Application (GUI)
```bash
python app.py
```
*Or on Windows, simply double-click **`run.bat`**.*

### 2. Run Headless Detection Test (CLI Benchmark)
To test the detection pipeline on a video file without launching the GUI:
```bash
python test_detection.py
```

### 3. Retrain the Luggage Detection Model
```bash
python start_training.py
```

### 4. Generate Project Presentation Deck
```bash
python create_presentation.py
```

---

## 🖥️ GUI Tabs & Operational Workflow

### 1. `[ DATASET ]` Tab
- Inspects dataset directories and versions.
- Validates `data.yaml` formatting, class mappings, and image/label alignment.
- Performs integrity checks on training, validation, and test splits.

### 2. `[ TRAIN ]` Tab
- Launches background model training directly from the UI.
- Live progress bar, real-time loss curves, and mAP@50 graph updates.
- Real-time stream of terminal output and automatic Excel log creation.

### 3. `[ VIDEO DEMO ]` Tab
- Upload and analyze surveillance footage (e.g. `test_videos/LeftBag.mpg`).
- Dual inference with real-time bounding boxes, threat timers, risk badges, and flashing critical banners.
- Adjustable confidence thresholds, dwell time triggers, and loitering timers.

### 4. `[ LIVE CAMERA ]` Tab
- Connects to local webcam or security camera stream for live real-time surveillance.
- Interactive alarm dismissal and live incident logging.

### 5. `[ STATS ]` Tab
- Comprehensive summary dashboard of model performance and incident telemetry.
- One-click export of structured Excel incident sheets and professional PDF safety audit reports.

---

## ⚙️ Tunable Parameters

You can adjust detection thresholds either directly in the GUI sidebar or programmatically:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `CONF_THRESHOLD` | `0.25` | Minimum confidence score for luggage detection |
| `PERSON_CONF` | `0.40` | Minimum confidence score for person detection |
| `ALERT_THRESHOLD_SEC`| `15s` | Stationary dwell time before triggering CRITICAL abandoned bag alert |
| `LOITER_THRESHOLD_SEC`| `20s` | Dwell time before triggering person loitering alarm |
| `RETURN_RADIUS_PX` | `150px` | Proximity radius to verify owner presence and auto-disarm alerts |

---

## 📑 Automated Reporting & Incident Logs

When a critical incident occurs (e.g., luggage left unattended for >15s):
1. **Snapshot Captured**: Saved immediately to `incidents/incident_<BAG_ID>_CRITICAL_<TIMESTAMP>.jpg`.
2. **Audio Siren**: Beeper alert sounded to immediately notify nearby security desk operators.
3. **Session Audit Log**: Added to internal incident log registry with timestamp, dwell duration, and coordinates.
4. **PDF Incident Report**: Exportable directly from the **STATS** tab, formatting all incident timestamps, risk metrics, and snapshot frames into an audit-ready PDF document.

---

## 📄 License & Acknowledgements

- **License**: Distributed under the MIT License.
- **Object Detection Framework**: [Ultralytics YOLOv9](https://github.com/ultralytics/ultralytics)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **UI Framework**: [Tkinter](https://docs.python.org/3/library/tkinter.html)

---

<p align="center">
  <b>Developed for Intelligent Public Safety & Automated Surveillance</b>
</p>
