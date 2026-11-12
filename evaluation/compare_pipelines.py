"""Prepare independent evidence and run project inference; no dataset labels enter prompts."""
import json
import sys
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.review_demo_context import make_packet
from evaluation.review_suite import inventory
from sentinel.context import assess
from sentinel.voting import classify_window
from sentinel.context_report import format_report
from sentinel.detection import load_person_model, detect_persons, iou_deduplicate
from sentinel.tracking import match_entities
from sentinel.risk import analyze_behaviors
from sentinel.posture import load_pose_model, estimate_postures
from sentinel.config import SentinelConfig


def main():
    windows = json.loads((ROOT/'evaluation/demo_windows.json').read_text()) + json.loads((ROOT/'evaluation/additional_demo_windows.json').read_text())
    out = ROOT/'evaluation/context_review/paired_25_v2'
    out.mkdir(parents=True, exist_ok=True)
    # Independent reviewer gets whole-clip contact sheets, not the model output.
    inventory(out, windows)
    SentinelConfig.DEBUG_DETECTION = False
    torch.set_num_threads(2)
    model, status = load_person_model(str(ROOT))
    pose = load_pose_model(str(ROOT))
    if isinstance(pose, tuple): pose = pose[0]
    if model is None or pose is None: raise RuntimeError(status)
    for window in windows:
        path = ROOT/'demo_videos'/window['file']
        target = out/(path.stem+'.json')
        if target.exists(): continue
        packet, sheet = make_packet(path, [window['start'], window['end']], adaptive=True)
        cap = cv2.VideoCapture(str(path))
        tracks = {}
        history = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, round(fps/5))
        index = 0
        while index/fps <= window['end']:
            ok, frame = cap.read()
            if not ok: break
            t = index/fps
            if index % stride == 0:
                detections = detect_persons(frame, model, device='cpu')
                tracks = match_entities(iou_deduplicate(detections), tracks, current_time=t)
                estimate_postures(frame, tracks, pose, current_time=t)
                analyze_behaviors(tracks, frame.shape[1], frame.shape[0], current_time=t)
                people = [dict(track_id=p['track_id'], signals=sorted(p.get('active_threats', [])), behavior=p.get('behavior_state', 'NORMAL')) for p in tracks.values() if not p.get('missed_frames', 0)]
                history.append((t, people))
            index += 1
        cap.release()
        for e in packet['evidence']:
            e['people'] = min(history, key=lambda item:abs(item[0]-e['time']))[1] if history else []
        signals = {s for t, people in history if t >= window['start'] for p in people for s in p['signals']}
        packet['rule_assessment'] = 'review' if signals & {'SUSPICIOUS_LOITERING','HIGH_CONFIDENCE_SUSPICIOUS','REPEATED_MOVEMENT'} else 'observe'
        packet['safety_event'] = 'POSSIBLE_FALL' in signals
        cv2.imwrite(str(out/(path.stem+'_sampled.jpg')), sheet)
        report = assess(packet, 'qwen3-vl:4b-instruct')
        report['combined_decision'] = classify_window(report, packet)
        result = dict(window=window, report=report, packet={k:v for k,v in packet.items() if k!='images'}, detector_sampling_fps=5)
        target.write_text(json.dumps(result, indent=2), encoding='utf-8')
        (out/(path.stem+'.txt')).write_text(format_report(report, packet, window['end']), encoding='utf-8')
        print(path.stem, report['combined_decision'], flush=True)

if __name__ == '__main__': main()
