"""
SENTINEL AI — Configuration & Tunable Parameters
Centralized, validated configuration thresholds for detection, tracking,
behavioral loitering analysis, and risk scoring.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SentinelConfig:
    # ── Operational Mode ──
    DEBUG_DETECTION = True               # Enable diagnostic logging for detection/model origins

    # ── Person Detection Thresholds ──
    PERSON_CONF_DEFAULT = 0.35          # Default confidence for person detector (COCO class 0 only)

    # ── Multi-Object Tracker Parameters ──
    TRACK_MATCH_THRESHOLD = 100.0       # Max pixel distance for centroid matching
    MAX_MISSED_FRAMES = 30              # Frames to retain missing track (~1.0s at 30fps)
    MAX_TRACK_HISTORY = 600             # Max points in trajectory history (~20s at 30fps)
    BBOX_SMOOTHING_ALPHA = 0.75         # EMA factor for bbox smoothing (0.75 curr + 0.25 prev)
    VELOCITY_SMOOTHING_ALPHA = 0.30     # EMA factor for image-plane velocity (0.30 curr + 0.70 prev)
    TRACK_TIMEOUT = 3.0                 # Seconds to keep temporarily missing tracks

    # ── Behavioral Loitering & Suspicious Movement Analysis ──
    DWELL_TIME_THRESHOLD = 45.0         # Baseline Tdwell (seconds) before person dwell signal fires
    LOITER_THRESHOLD = 45.0             # Backward-compatible alias
    TRAJECTORY_WINDOW_SECONDS = 60.0    # Trajectory analysis time window in seconds (baseline 60s)
    MEANDER_RATIO_THRESHOLD = 2.5       # Path length / net displacement ratio for wandering/pacing
    MEANDER_MIN_PATH_LENGTH = 30.0      # Minimum trajectory path length (px) before evaluating meander
    MEANDER_EPSILON_DISP = 1.0          # Minimum displacement floor (px) to prevent division by zero / instability
    GRID_ROWS = 6                       # Grid rows for spatial revisit tracking
    GRID_COLS = 8                       # Grid columns for spatial revisit tracking
    GRID_MEMORY_LENGTH = 50             # Max past cells to remember in revisit history
    REVISIT_COUNT_THRESHOLD = 3         # Number of cell returns needed to flag repetitive behavior
    LOITER_SPATIAL_RATIO = 0.20         # Fraction of frame diagonal for stationary loiter spread
    SPEED_ABSOLUTE_MIN_RATIO = 0.08     # Minimum speed floor fraction of frame diagonal / sec
    SPEED_RATIO_THRESHOLD = 2.5         # Ratio to past speed for sudden running detection

    # ── Multi-Signal Suspicion Scoring Weights (Sum to 100) ──
    WEIGHT_DWELL = 35                   # Contribution of excessive dwell time
    WEIGHT_MEANDER = 35                 # Contribution of high meander ratio (repetitive wandering)
    WEIGHT_REVISIT = 30                 # Contribution of frequent spatial revisits

    # ── Suspicion Score Level Thresholds (0-100 scale) ──
    SUSPICION_THRESH_OBSERVING = 15     # Mild observation threshold
    SUSPICION_THRESH_LOW = 30           # Low suspicion
    SUSPICION_THRESH_MEDIUM = 50        # Medium suspicion (e.g. dwell + repetitive movement)
    SUSPICION_THRESH_HIGH = 70          # High suspicion (loitering + revisits)
    SUSPICION_THRESH_CRITICAL = 85      # Critical suspicion (sustained loitering + pacing/revisits)

    # ── Temporal Stability & Hysteresis ──
    SUSPICION_PERSISTENCE_FRAMES = 15   # Sustained frames before escalating to ALERT state
    SUSPICION_COOLDOWN_FRAMES = 30      # Frames of normal behavior before de-escalating

    # ── Human Posture Estimation Parameters ──
    POSTURE_MODEL_PATH = "yolov8n-pose.pt"
    POSTURE_CONF_THRESHOLD = 0.35
    POSTURE_FALL_CONFIRMATION_FRAMES = 25  # ~1.0s sustained confirmation for possible fall
    POSTURE_FALL_COOLDOWN = 15.0           # Seconds between repeated possible fall alerts

    # ── Alerting & System Cooldowns ──
    ALERT_COOLDOWN = 10.0               # Seconds between repeated alerts for same entity + event
