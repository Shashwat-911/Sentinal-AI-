"""Manual local-VLM review of the reported demo windows (not an accuracy benchmark).

Run from the project root:
    python evaluation/review_demo_context.py --assess --model qwen3.5:4b
Without --assess, only extract contact sheets. No source names/labels reach the model.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentinel.context import MODEL, MODELS, assess, select_visual_frames, chronological_sheet, frame_change
from sentinel.context_report import format_report, temporal_context

CASES = {
    "normal": ("normal/Normal_Videos_003_x264.mp4", [3, 9, 16, 26, 32]),
    "interaction": ("context_safety/Arrest003_x264.mp4", [7, 14, 17, 23, 28.5]),
    "door_early": ("suspicious/Burglary005_x264.mp4", [16, 22, 29, 38, 44.8]),
    "door_later": ("suspicious/Burglary005_x264.mp4", [150, 155, 160, 165, 170]),
    "vehicle_26": ("suspicious/Vandalism007_x264.mp4", [0, 13, 14, 15, 26]),
    "vehicle_end": ("suspicious/Vandalism007_x264.mp4", [9, 16, 23, 30, 38]),
}


def make_packet(path, times, adaptive=False, short_window=False):
    requested_window = {'start': times[0], 'end': times[-1]}
    if adaptive:
        start, end = max(times[0], times[-1]-(8 if short_window else 29)), times[-1]
        times = sorted(set([start+i for i in range(int(end-start)+1)] + [end]))
    cap = cv2.VideoCapture(str(path))
    evidence, images = [], []
    candidates, previous = [], None
    boundary = None
    try:
        for timestamp in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {path} at {timestamp}s")
            evidence.append(dict(id=f"E{timestamp:.3f}", time=timestamp,
                brightness_0_255=round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())),
                people=[]))
            # Match ContextEngine's production image transport. Sending full
            # 1080p/4K frames can exceed Ollama's 8192-token context before any
            # vision tokens are accounted for, making every report fail with 400.
            height, width = frame.shape[:2]
            scale = min(1.0, 512 / max(width, height))
            frame = cv2.resize(frame, (max(1, round(width*scale)), max(1, round(height*scale))))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            images.append(base64.b64encode(encoded).decode())
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
            motion, view_change = frame_change(previous, gray)
            previous = gray
            if adaptive and view_change:
                evidence = evidence[-1:]
                images = images[-1:]
                candidates.clear()
                boundary = timestamp
                motion = 0.0
            candidates.append((timestamp, evidence[-1]['id'], images[-1], motion))
    finally:
        cap.release()
    if adaptive:
        if short_window:
            selected = [candidates[round(i*(len(candidates)-1)/4)] for i in range(min(5,len(candidates)))] if len(candidates)>5 else candidates
        else:
            selected = select_visual_frames(candidates)
        ids = {f[1] for f in selected}
        indices = [i for i,e in enumerate(evidence) if e['id'] in ids]
        evidence = [evidence[i] for i in indices]
        images = [images[i] for i in indices]
    packet = dict(source="vid", scene="Unknown", capture_time="unknown",
                  requested_window=requested_window,
                  scene_boundary_seconds=boundary,
                  temporal_context=temporal_context("unknown"), rule_assessment="observe",
                  evidence=evidence, images=images,
                  visual_frames=[dict(evidence_id=e["id"], video_seconds=e["time"]) for e in evidence])
    sheet = cv2.imdecode(np.frombuffer(base64.b64decode(chronological_sheet(packet)[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    return packet, sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assess", action="store_true")
    parser.add_argument("--adaptive", action="store_true", help="Use production view-boundary and temporal/motion sampling")
    parser.add_argument("--model", choices=MODELS, default=MODEL)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    output = ROOT / "evaluation" / "context_review" / (args.model.replace(":", "_") + ('_adaptive' if args.adaptive else ''))
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (relative, times) in CASES.items():
        if args.case and name != args.case:
            continue
        packet, sheet = make_packet(ROOT / "demo_videos" / relative, times, args.adaptive)
        cv2.imwrite(str(output / f"{name}.jpg"), sheet)
        if not args.assess:
            continue
        started = time.monotonic()
        report = assess(packet, args.model)
        result = dict(case=name, model=args.model, elapsed_seconds=round(time.monotonic()-started, 2),
                      assessment=report, packet={k:v for k,v in packet.items() if k != "images"})
        (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / f"{name}.txt").write_text(format_report(report, packet, times[-1]), encoding="utf-8")
        results.append(dict(case=name, classification=report["classification"],
                            lighting=report["lighting_assessment"]["appearance"],
                            elapsed_seconds=result["elapsed_seconds"]))
        print(json.dumps(results[-1]), flush=True)
    (output / "run.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
