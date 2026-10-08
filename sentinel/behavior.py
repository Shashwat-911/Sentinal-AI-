"""
SENTINEL AI — Behavioral Analysis Module
Implements true trajectory-based human behavior analysis:
- Dwell Time Analysis (Tdwell >= 45s baseline signal, never suspicious alone)
- Trajectory & Meander Ratio Analysis (rho = Path Length / Net Displacement with windowing)
- Spatial Grid Revisit Analysis (N x M spatial grid cell returns)
- Restricted Zone Event Separation (zone violation distinct from personal loitering)
- Multi-Signal Explainable Suspicion Scoring (transparent, explainable evidence breakdown)
- Temporal Stability & Hysteresis State Machine (NORMAL -> OBSERVING -> LOW -> MEDIUM -> HIGH -> ALERT / COOLING DOWN)
"""

import math
import time
import cv2
import numpy as np
from sentinel.config import SentinelConfig
from sentinel.detection import is_person, is_bag_object


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DWELL TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dwell_time(track, current_time=None):
    """
    Compute how many seconds a track has been persistently visible.
    Dwell time is an accumulating behavioral signal, but NEVER flags
    a person as suspicious on its own.
    """
    if current_time is None:
        current_time = time.time()
    first_seen = track.get("first_seen", current_time)
    return max(0.0, float(current_time - first_seen))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAJECTORY & MEANDER RATIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trajectory_metrics(history, window_seconds=None):
    """
    Compute cumulative path length and net displacement from trajectory history.
    history: list of (x, y, timestamp)
    window_seconds: optional window to restrict analysis to recent trajectory.

    Returns: (total_path, net_displacement, meander_ratio)

    Meander ratio: rho = total_path / net_displacement
    - ~1.0: direct, purposeful linear movement
    - moderately higher: turning, wandering
    - significantly higher (>= 2.5): pacing, circling, repetitive movement

    Safe handling:
    - Guaranteed non-zero divisor via MEANDER_EPSILON_DISP
    - Disregards micro-jitter if path length is below MEANDER_MIN_PATH_LENGTH
      (e.g., stationary standing does NOT explode meander ratio)
    """
    if len(history) < 2:
        return 0.0, 0.0, 1.0

    # Filter by analysis window if requested
    if window_seconds is not None and window_seconds > 0:
        latest_t = history[-1][2]
        window_pts = [pt for pt in history if pt[2] >= (latest_t - window_seconds)]
        if len(window_pts) >= 2:
            pts = window_pts
        else:
            pts = history
    else:
        pts = history

    total_path = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total_path += math.hypot(dx, dy)

    x0, y0 = pts[0][:2]
    xn, yn = pts[-1][:2]
    net_disp = math.hypot(xn - x0, yn - y0)

    min_path = getattr(SentinelConfig, "MEANDER_MIN_PATH_LENGTH", 30.0)
    eps_disp = getattr(SentinelConfig, "MEANDER_EPSILON_DISP", 3.0)

    # If the target has barely moved overall (e.g. standing stationary with camera noise),
    # clamp meander ratio to 1.0 to prevent false-positive division explosion.
    # If history has only 2 points and net_disp > 0, direct motion ratio is exact.
    if total_path < min_path and len(pts) > 2 and net_disp < eps_disp:
        return total_path, net_disp, 1.0

    meander_ratio = total_path / max(net_disp, eps_disp)
    return total_path, net_disp, meander_ratio


def count_direction_reversals(history, min_step=10.0):
    """
    Count significant direction reversals (> 100 degrees) along smoothed trajectory.
    Filters out micro-jitter by only considering displacements >= min_step.
    A single turn-around produces 1 reversal; repeated pacing produces >= 2 reversals.
    """
    if len(history) < 3:
        return 0

    # Downsample to points separated by at least min_step
    filtered = [history[0][:2]]
    for pt in history[1:]:
        dx = pt[0] - filtered[-1][0]
        dy = pt[1] - filtered[-1][1]
        if math.hypot(dx, dy) >= min_step:
            filtered.append(pt[:2])

    if len(filtered) < 3:
        return 0

    reversals = 0
    for i in range(1, len(filtered) - 1):
        v1x = filtered[i][0] - filtered[i-1][0]
        v1y = filtered[i][1] - filtered[i-1][1]
        v2x = filtered[i+1][0] - filtered[i][0]
        v2y = filtered[i+1][1] - filtered[i][1]
        m1 = math.hypot(v1x, v1y)
        m2 = math.hypot(v2x, v2y)
        if m1 > 0 and m2 > 0:
            cos_a = (v1x * v2x + v1y * v2y) / (m1 * m2)
            if cos_a < -0.2:  # Angle > ~101 degrees
                reversals += 1
    return reversals


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPATIAL GRID REVISIT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def update_grid_revisits(track, cx, cy, w, h, rows=None, cols=None):
    """
    Track spatial grid transitions and returns.
    Increments revisit_count when a target leaves a grid cell and later returns to it.
    Does NOT count continuous movement inside the same cell as repeated revisits.
    """
    if rows is None:
        rows = getattr(SentinelConfig, "GRID_ROWS", 6)
    if cols is None:
        cols = getattr(SentinelConfig, "GRID_COLS", 8)

    if w <= 0 or h <= 0:
        return track.get("revisit_count", 0)

    # Determine current cell index
    col = min(max(int((cx / w) * cols), 0), cols - 1)
    row = min(max(int((cy / h) * rows), 0), rows - 1)
    current_cell = (row, col)

    last_cell = track.get("last_grid_cell", None)
    visited_cells = list(track.get("visited_cells", []))
    revisit_count = int(track.get("revisit_count", 0))

    if last_cell is None:
        track["last_grid_cell"] = current_cell
        track["visited_cells"] = [current_cell]
        track["revisit_count"] = 0
        return 0

    if current_cell != last_cell:
        # Check if person has visited current_cell previously (excluding the cell just left)
        if current_cell in visited_cells[:-1]:
            revisit_count += 1

        visited_cells.append(current_cell)
        max_mem = getattr(SentinelConfig, "GRID_MEMORY_LENGTH", 50)
        if len(visited_cells) > max_mem:
            visited_cells.pop(0)

        track["last_grid_cell"] = current_cell
        track["visited_cells"] = visited_cells
        track["revisit_count"] = revisit_count

    return revisit_count


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPLAINABLE MULTI-SIGNAL SUSPICION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_person_behavior(p_info, w, h, current_time=None):
    """
    Evaluates behavioral state for a single person track independently.
    Guarantees:
    - PERSON #1 state is strictly independent from PERSON #2.
    - Long dwell time alone is treated as one signal, NEVER suspicious loitering alone.
    - Combines:
      Signal A: Dwell time
      Signal B: Trajectory / Meander ratio
      Signal C: Spatial revisits
      Signal D: Restricted zone entry
    - Computes an explainable suspicion score (0-100) and evidence breakdown.
    - Uses temporal stability / hysteresis to prevent single-frame flickering alerts.
    """
    if current_time is None:
        current_time = time.time()

    pcx, pcy = p_info["centroid"]
    history = p_info.get("history", [])

    # Ensure active_threats is a set
    if "active_threats" not in p_info or not isinstance(p_info["active_threats"], set):
        p_info["active_threats"] = set(p_info.get("active_threats", []))

    # ── Signal A: Dwell Time ──
    dwell_time = compute_dwell_time(p_info, current_time)
    p_info["dwell_time"] = dwell_time
    # ── Signal B: Trajectory Metrics & Meander Ratio ──
    traj_window = getattr(SentinelConfig, "TRAJECTORY_WINDOW_SECONDS", 60.0)
    path_len, net_disp, meander_ratio = compute_trajectory_metrics(history, window_seconds=traj_window)
    p_info["path_length"] = path_len
    p_info["net_displacement"] = net_disp
    p_info["meander_ratio"] = meander_ratio

    # ── Signal C: Spatial Grid Revisits ──
    revisit_count = update_grid_revisits(p_info, pcx, pcy, w, h)
    p_info["revisit_count"] = revisit_count
    revisit_thresh = getattr(SentinelConfig, "REVISIT_COUNT_THRESHOLD", 3)
    revisit_flag = revisit_count >= revisit_thresh

    frame_diag = math.sqrt(w * w + h * h) if (w > 0 and h > 0) else 1000.0
    loiter_spread = getattr(SentinelConfig, "LOITER_SPATIAL_RATIO", 0.20) * frame_diag

    # Trajectory direct transit check:
    # A person with continuous directional displacement across the frame (meander < 1.5, 0 revisits, net_disp >= loiter_spread)
    # is transiting across the scene, not dwelling in an area.
    is_transiting = (meander_ratio < 1.5) and (revisit_count == 0) and (net_disp >= loiter_spread)

    dwell_thresh = getattr(SentinelConfig, "DWELL_TIME_THRESHOLD", 45.0)
    dwell_flag = (dwell_time >= dwell_thresh) and (not is_transiting)

    meander_thresh = getattr(SentinelConfig, "MEANDER_RATIO_THRESHOLD", 2.5)
    min_path = getattr(SentinelConfig, "MEANDER_MIN_PATH_LENGTH", 30.0)
    # A meander flag requires non-trivial movement so stationary standing doesn't trigger it
    meander_flag = (meander_ratio >= meander_thresh) and (path_len >= min_path)

    # ── Signal D: Restricted Surveillance Zone ──
    inside_zone = False
    if getattr(SentinelConfig, "RESTRICTED_ZONE_ENABLED", True):
        poly_pts = np.array([
            [int(pt[0] * w), int(pt[1] * h)] for pt in SentinelConfig.RESTRICTED_ZONE_COORDS
        ], dtype=np.int32)
        dist_to_poly = cv2.pointPolygonTest(poly_pts, (pcx, pcy), False)
        if dist_to_poly >= 0:
            inside_zone = True

    p_info["in_restricted_zone"] = inside_zone
    min_zone_frames = getattr(SentinelConfig, "RESTRICTED_ZONE_MIN_FRAMES", 15)

    if inside_zone:
        p_info["consecutive_restricted_frames"] = p_info.get("consecutive_restricted_frames", 0) + 1
        restricted_flag = p_info["consecutive_restricted_frames"] >= min_zone_frames
    else:
        p_info["consecutive_restricted_frames"] = 0
        restricted_flag = False

    # ── Behavioral Combinations ──
    reversals = count_direction_reversals(history)
    p_info["direction_reversals_count"] = reversals

    # Repetitive movement formulation:
    # A single direction change or walking back once (reversals <= 1, revisit_count <= 2) is NORMAL.
    # Repetitive movement requires either:
    # 1. Spatial revisit threshold reached (revisit_flag)
    # 2. High meander with multiple reversals (meander_flag and reversals >= 2)
    # 3. High meander with multiple cell returns (meander_flag and revisit_count >= 2)
    is_repetitive = revisit_flag or (meander_flag and (reversals >= 2 or revisit_count >= 2))
    # Suspicious loitering requires BOTH persistent presence AND repetitive movement
    is_loitering = dwell_flag and is_repetitive

    # ── Multi-Signal Explainable Suspicion Score (0-100) ──
    w_dwell = getattr(SentinelConfig, "WEIGHT_DWELL", 25)
    w_meander = getattr(SentinelConfig, "WEIGHT_MEANDER", 30)
    w_revisit = getattr(SentinelConfig, "WEIGHT_REVISIT", 25)
    w_restricted = getattr(SentinelConfig, "WEIGHT_RESTRICTED", 20)

    score = 0.0

    # 1. Dwell contribution: up to w_dwell (never enough alone to reach MEDIUM or HIGH)
    if dwell_flag:
        excess_ratio = min(1.0, (dwell_time - dwell_thresh) / max(1.0, dwell_thresh))
        score += w_dwell * (0.70 + 0.30 * excess_ratio)
    elif dwell_time > 10.0:
        score += w_dwell * 0.40 * (dwell_time / dwell_thresh)

    # 2. Meander contribution:
    if meander_flag:
        m_ratio = min(1.0, (meander_ratio - meander_thresh) / max(1.0, meander_thresh * 2.0))
        score += w_meander * (0.75 + 0.25 * m_ratio)

    # 3. Revisit contribution:
    if revisit_flag:
        r_ratio = min(1.0, (revisit_count - revisit_thresh) / max(1.0, revisit_thresh))
        score += w_revisit * (0.75 + 0.25 * r_ratio)
    elif revisit_count > 0:
        score += w_revisit * 0.30 * (revisit_count / max(1, revisit_thresh))

    # 4. Restricted zone contribution:
    if restricted_flag:
        score += w_restricted

    suspicion_score = int(min(100, max(0, round(score))))
    p_info["suspicion_score"] = suspicion_score

    # ── Target Level Determination ──
    if is_loitering and restricted_flag:
        target_level = "CRITICAL"
    elif is_loitering:
        target_level = "HIGH"
    elif is_repetitive and not dwell_flag:
        target_level = "MEDIUM" if (meander_flag and revisit_flag) else "LOW"
    elif dwell_flag and not is_repetitive:
        # Long dwell alone is ONLY observation / low, NEVER suspicious loitering!
        target_level = "OBSERVING"
    elif restricted_flag and not is_repetitive and not dwell_flag:
        # Restricted entry alone is a zone warning, not suspicious loitering
        target_level = "LOW"
    else:
        target_level = "NORMAL"

    # ── Temporal Stability & Hysteresis State Machine ──
    # Prevents 1-frame spikes from firing an alarm and prevents flickering
    persistence_thresh = getattr(SentinelConfig, "SUSPICION_PERSISTENCE_FRAMES", 15)
    cooldown_thresh = getattr(SentinelConfig, "SUSPICION_COOLDOWN_FRAMES", 30)

    sustained_frames = p_info.get("sustained_suspicious_frames", 0)
    cooldown_frames = p_info.get("cooldown_frames", 0)
    current_state = p_info.get("behavior_state", "NORMAL")

    if target_level in ("HIGH", "CRITICAL"):
        sustained_frames += 1
        cooldown_frames = 0
        if sustained_frames >= persistence_thresh:
            stable_state = "ALERT" if target_level == "HIGH" else "CRITICAL"
        else:
            stable_state = "OBSERVING"
    elif target_level in ("OBSERVING", "LOW", "MEDIUM"):
        sustained_frames = max(0, sustained_frames - 1)
        if current_state in ("ALERT", "HIGH", "CRITICAL", "COOLING_DOWN"):
            cooldown_frames += 1
            if cooldown_frames < cooldown_thresh:
                stable_state = "COOLING_DOWN"
            else:
                stable_state = target_level
                cooldown_frames = 0
        else:
            stable_state = target_level
            cooldown_frames = 0
    else: # target_level is NORMAL
        sustained_frames = 0
        if current_state in ("ALERT", "HIGH", "CRITICAL", "COOLING_DOWN"):
            cooldown_frames += 1
            if cooldown_frames < cooldown_thresh:
                stable_state = "COOLING_DOWN"
            else:
                stable_state = "NORMAL"
                cooldown_frames = 0
        else:
            stable_state = "NORMAL"
            cooldown_frames = 0

    p_info["sustained_suspicious_frames"] = sustained_frames
    p_info["cooldown_frames"] = cooldown_frames
    p_info["behavior_state"] = stable_state
    p_info["suspicion_level"] = stable_state

    prior_threats = set(p_info["active_threats"])
    loiter_confirmed = (
        is_loitering and stable_state in ("ALERT", "CRITICAL")
    ) or (
        stable_state == "COOLING_DOWN"
        and bool(prior_threats & {"SUSPICIOUS_LOITERING", "LOITERING"})
    )
    high_conf_confirmed = (
        loiter_confirmed and restricted_flag and stable_state in ("ALERT", "CRITICAL", "COOLING_DOWN")
    )

    # ── Evidence Breakdown (Fully Explainable) ──
    cell_r, cell_c = p_info.get("last_grid_cell", (0, 0))
    grid_label = f"R{cell_r}C{cell_c}"
    evidence_strings = [
        f"Dwell: {int(dwell_time)}s",
        f"Meander ratio: {meander_ratio:.1f}",
        f"Revisits: {revisit_count}",
        f"Restricted zone: {'YES' if inside_zone else 'NO'}"
    ]

    p_info["suspicion_evidence"] = {
        "dwell_time": round(dwell_time, 1),
        "dwell_flag": dwell_flag,
        "path_length": round(path_len, 1),
        "net_displacement": round(net_disp, 1),
        "meander_ratio": round(meander_ratio, 2),
        "meander_flag": meander_flag,
        "revisit_count": revisit_count,
        "revisit_flag": revisit_flag,
        "inside_zone": inside_zone,
        "restricted_flag": restricted_flag,
        "grid_cell": grid_label,
        "summary": evidence_strings
    }

    # ── Threat Set Management ──
    # 1. Restricted zone entry
    if restricted_flag:
        p_info["active_threats"].add("RESTRICTED_ZONE")
    else:
        p_info["active_threats"].discard("RESTRICTED_ZONE")

    # 2. Long dwell (warning only, if not loitering)
    # Keep LONG_DWELL active once dwell threshold is reached, even if repetitive movement starts
    # This prevents flickering and allows LONG_DWELL + REPEATED_MOVEMENT to coexist
    if dwell_flag:
        p_info["active_threats"].add("LONG_DWELL")
    else:
        p_info["active_threats"].discard("LONG_DWELL")

    # 3. Repeated movement (warning only, if dwell not yet reached)
    # Keep REPEATED_MOVEMENT active once repetitive behavior is detected
    # This allows it to coexist with LONG_DWELL and prevents flickering
    if is_repetitive:
        p_info["active_threats"].add("REPEATED_MOVEMENT")
    else:
        p_info["active_threats"].discard("REPEATED_MOVEMENT")

    # 4. Suspicious loitering — requires dwell + repetitive movement AND temporal persistence
    if loiter_confirmed:
        p_info["active_threats"].add("SUSPICIOUS_LOITERING")
        p_info["active_threats"].add("LOITERING")  # for backward compatibility
    else:
        p_info["active_threats"].discard("SUSPICIOUS_LOITERING")
        p_info["active_threats"].discard("LOITERING")

    # 5. Critical threat: persistent loitering inside restricted zone
    if high_conf_confirmed:
        p_info["active_threats"].add("HIGH_CONFIDENCE_SUSPICIOUS")
    else:
        p_info["active_threats"].discard("HIGH_CONFIDENCE_SUSPICIOUS")

    # ── Kinematic Sudden Running Check (Preserved) ──
    speed = p_info.get("speed", 0.0)
    frame_diag = math.sqrt(w * w + h * h)
    speed_abs_min = getattr(SentinelConfig, "SPEED_ABSOLUTE_MIN_RATIO", 0.08) * frame_diag
    past_pts = [pt for pt in history if 2.0 <= current_time - pt[2] <= 7.0]

    if len(past_pts) >= 10:
        past_speeds = []
        for i in range(1, len(past_pts)):
            dx = past_pts[i][0] - past_pts[i-1][0]
            dy = past_pts[i][1] - past_pts[i-1][1]
            dt = past_pts[i][2] - past_pts[i-1][2]
            if dt > 0:
                past_speeds.append(math.hypot(dx, dy) / dt)
        if past_speeds:
            avg_past_speed = sum(past_speeds) / len(past_speeds)
            speed_ratio_thresh = getattr(SentinelConfig, "SPEED_RATIO_THRESHOLD", 2.5)
            if speed > speed_abs_min and speed >= speed_ratio_thresh * max(avg_past_speed, speed_abs_min * 0.125):
                p_info["consecutive_running_frames"] = p_info.get("consecutive_running_frames", 0) + 1
                if p_info["consecutive_running_frames"] >= 8:
                    p_info["active_threats"].add("SUDDEN_RUNNING")
            else:
                p_info["consecutive_running_frames"] = 0
                p_info["active_threats"].discard("SUDDEN_RUNNING")
    else:
        p_info["consecutive_running_frames"] = 0
        p_info["active_threats"].discard("SUDDEN_RUNNING")
