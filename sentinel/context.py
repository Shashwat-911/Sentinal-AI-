"""Bounded local visual context from sampled frames and existing telemetry."""
import base64
import copy
import json
import queue
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta


from sentinel.context_report import SCHEMA, PROMPT, parse_recording_start, temporal_context, validate_report, review_output

MODEL = "qwen3.5:4b"
MODELS = (MODEL, "qwen3.5:2b")


def assess(packet, model=None):
    model = model or packet.get("model", MODEL)
    if model not in MODELS:
        raise ValueError("Unsupported local visual model")
    images = packet.get("images", [])
    payload = {key: value for key, value in packet.items() if key not in ("images", "model")}
    message = {"role": "user", "content": json.dumps(payload) + "\n\n"
               "Assess the evidence above. Clothing and hidden identity must not appear in "
               "concern factors, task support or classification reasoning. Low visibility is "
               "not evidence of nighttime. Describe observable contact/leaning; force, permission "
               "and intent are unknown unless demonstrated. Compare ordinary access/maintenance "
               "with concerning possibilities without assuming either. Use uncertain when these "
               "cannot be distinguished. Do not list being alone, clothing or absence of tools "
               "as concern factors. Missing information should concern actions, objects and "
               "environment, never the person's identity. Empty concern lists are valid. "
               "Return the detailed JSON report."}
    if images:
        message["images"] = images
    response_schema = copy.deepcopy(SCHEMA)
    response_schema["properties"]["evidence_ids"]["minItems"] = 1
    response_schema["properties"]["evidence_ids"]["items"]["enum"] = [item["id"] for item in packet["evidence"]]
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({"model": model, "stream": False, "think": False,
                         "format": response_schema,
                         "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1800},
                         "messages": [{"role": "system", "content": PROMPT},
                                      message]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(json.loads(response.read())["message"]["content"])
    validate(result, packet)
    return review_output(result)


def validate(result, packet):
    validate_report(result, packet)


def capture_time(source, timestamp, recording_start):
    if source == "cam":
        return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")
    if recording_start:
        return (parse_recording_start(recording_start) + timedelta(seconds=timestamp)).isoformat(timespec="seconds")
    return "unknown"


class ContextEngine:
    def __init__(self, reasoner=assess):
        self.reasoner = reasoner
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=1)
        self.priority = queue.Queue(maxsize=8)
        self.results = queue.Queue(maxsize=8)
        self.closed = threading.Event()
        self.generation = 0
        self.busy = None
        self.history = deque(maxlen=30)
        self.frames = deque(maxlen=30)
        self.last_sample = self.last_submit = -float("inf")
        self.last_processed = -float("inf")
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.history.clear()
            self.frames.clear()
            self.last_sample = self.last_submit = -float("inf")
            self.last_processed = -float("inf")
            while not self.priority.empty():
                try:
                    self.priority.get_nowait()
                except queue.Empty:
                    break
            while not self.pending.empty():
                try:
                    self.pending.get_nowait()
                except queue.Empty:
                    break
            return self.generation

    def close(self):
        self.closed.set()
        self.reset()

    def observe(self, generation, source, timestamp, tracks, brightness, settings, frame=None, force=False, urgent=False):
        with self.lock:
            if generation != self.generation or self.closed.is_set() or not settings["enabled"]:
                return
            if timestamp - self.last_sample < 1 and not force:
                return
            if force and self.history and self.history[-1]["id"] == f"E{timestamp:.3f}":
                self.history.pop()
                if self.frames and self.frames[-1][1] == f"E{timestamp:.3f}":
                    self.frames.pop()
            self.last_sample = timestamp
            people = []
            for track in tracks.values():
                if track.get("missed_frames", 0):
                    continue
                people.append({"track_id": str(track["track_id"]),
                               "posture": track.get("posture", "UNKNOWN"),
                               "dwell_seconds": round(track.get("dwell_time", 0), 1),
                               "revisits": track.get("revisit_count", 0),
                               "meander_ratio": round(track.get("meander_ratio", 0), 2),
                               "direction_reversals": track.get("direction_reversals_count", 0),
                               "speed_px_per_second": round(track.get("speed", 0), 1),
                               "path_length_px": round(track.get("path_length", 0), 1),
                               "pose_confidence": round(track.get("pose_confidence", 0), 2),
                               "detection_confidence": round(track.get("confidence", 0), 2),
                               "behavior": track.get("behavior_state", "NORMAL"),
                               "signals": sorted(track.get("active_threats", []))})
            self.history.append({"id": f"E{timestamp:.3f}", "time": round(timestamp, 3),
                                 "people": people[:12], "visible_count": len(people),
                                 "brightness_0_255": round(brightness)})
            if frame is not None:
                import cv2
                height, width = frame.shape[:2]
                scale = min(1.0, 512 / max(width, height))
                small = cv2.resize(frame, (max(1, round(width*scale)), max(1, round(height*scale))))
                ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.frames.append((timestamp, f"E{timestamp:.3f}", base64.b64encode(encoded).decode()))
            while self.frames and timestamp - self.frames[0][0] > 30:
                self.frames.popleft()
            while self.history and timestamp - self.history[0]["time"] > 30:
                self.history.popleft()
            if (timestamp - self.history[0]["time"] < 2 or timestamp - self.last_submit < 6) and not force:
                return
            history = list(self.history)
            window_people = [person for sample in history for person in sample["people"]]
            review = any(set(p["signals"]) & {"SUSPICIOUS_LOITERING", "HIGH_CONFIDENCE_SUSPICIOUS", "REPEATED_MOVEMENT"} for p in window_people)
            safety = any("POSSIBLE_FALL" in p["signals"] for p in window_people)
            indices = sorted({round(i * (len(history)-1) / max(1, min(9, len(history)-1))) for i in range(min(10, len(history)))})
            captured = capture_time(source, timestamp, settings["recording_start"])
            packet = {"source": source, "scene": settings["scene"],
                      "model": settings.get("model", MODEL),
                      "temporal_context": temporal_context(captured, settings.get("usual_hours", "")),
                      "capture_time": captured,
                      "activity_expected": settings["expected"],
                      "rule_assessment": "review" if review else "observe",
                      "safety_event": safety,
                      "interpretation": "Review visual interactions independently of baseline dwell/loiter thresholds",
                      "evidence": copy.deepcopy([history[i] for i in indices])}
            # Crowded single-camera scenes must still fit the small model context.
            while len(packet["evidence"]) > 1 and len(json.dumps(packet)) > 7000:
                packet["evidence"].pop(0)
            # Context can raise review priority only alongside actual behavior evidence.
            if settings["expected"] == "No" and any(p["signals"] for p in people):
                packet["rule_assessment"] = "review"
            available = [f for f in self.frames if f[1] in {e["id"] for e in packet["evidence"]}]
            if available:
                indices = sorted({round(i * (len(available)-1) / max(1, min(4, len(available)-1))) for i in range(min(5, len(available)))})
                chosen = [available[i] for i in indices]
                packet["visual_frames"] = [{"evidence_id": f[1], "video_seconds": f[0]} for f in chosen]
                packet["images"] = [f[2] for f in chosen]
            job = (generation, timestamp, packet)
            if urgent:
                try:
                    self.priority.put_nowait(job)
                except queue.Full:
                    if not self.results.full():
                        self.results.put_nowait(dict(generation=generation, timestamp=timestamp, packet=packet,
                            latency=0, error="Incident analysis queue full; ongoing scene analysis continues."))
            else:
                try:
                    self.pending.get_nowait()
                except queue.Empty:
                    pass
                self.pending.put_nowait(job)
            self.last_submit = timestamp

    def has_pending(self):
        return not self.pending.empty() or not self.priority.empty()

    def _run(self):
        while not self.closed.is_set():
            try:
                try:
                    generation, timestamp, packet = self.priority.get_nowait()
                    priority_job = True
                except queue.Empty:
                    generation, timestamp, packet = self.pending.get(timeout=.2)
                    priority_job = False
            except queue.Empty:
                continue
            with self.lock:
                if generation != self.generation:
                    continue
                if not priority_job and timestamp < self.last_processed:
                    continue
            started = time.monotonic()
            self.busy = (generation, started)
            try:
                result = {"assessment": self.reasoner(packet)}
            except Exception as exc:
                result = {"error": f"Context unavailable: {type(exc).__name__}: {exc}"}
            finally:
                self.busy = None
            result.update(generation=generation, timestamp=timestamp, packet=packet,
                          latency=round(time.monotonic() - started, 2))
            with self.lock:
                if generation != self.generation or self.closed.is_set():
                    continue
                self.last_processed = max(self.last_processed, timestamp)
                if self.results.full():
                    self.results.get_nowait()
                self.results.put_nowait(result)
