"""Decode the ten supplementary demo clips and sample production person/pose inference.

This is a smoke/coverage review. Without box labels it does not measure precision/recall.
"""
import json, sys, time
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentinel.config import SentinelConfig
from sentinel.detection import load_person_model, detect_persons, iou_deduplicate
from sentinel.tracking import match_entities
from sentinel.posture import load_pose_model, estimate_postures


def main():
    SentinelConfig.DEBUG_DETECTION = False
    torch.set_num_threads(2)
    person_model, status = load_person_model(str(ROOT))
    pose_model = load_pose_model(str(ROOT))
    if isinstance(pose_model, tuple):
        pose_model = pose_model[0]
    if person_model is None or pose_model is None:
        raise RuntimeError(status)
    out = ROOT / 'evaluation/context_review/additional_pixabay_qwen3vl4b_scaled/detector'
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted((ROOT / 'demo_videos/additional').glob('*.mp4')):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or expected <= 0:
            raise RuntimeError(f'Unreadable input: {path}')
        stride = max(1, round(fps * 2))
        frame_index = 0
        tracks = {}
        samples = []
        started = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if frame_index % stride == 0 or frame_index == expected - 1:
                detections = detect_persons(frame, person_model, device='cpu')
                tracks = match_entities(iou_deduplicate(detections), tracks, current_time=timestamp)
                alerts = estimate_postures(frame, tracks, pose_model, current_time=timestamp)
                samples.append({
                    'time': round(timestamp, 3),
                    'detections': detections,
                    'tracks': [{'id': t['track_id'], 'bbox': t['bbox'], 'posture': t.get('posture'),
                                'missed': t.get('missed_frames')} for t in tracks.values()],
                    'posture_alerts': alerts,
                })
            frame_index += 1
        cap.release()
        item = {
            'file': path.name, 'decoded_frames': frame_index, 'expected_frames': expected,
            'sampled_frames': len(samples),
            'samples_with_person_detections': sum(bool(s['detections']) for s in samples),
            'peak_detections': max((len(s['detections']) for s in samples), default=0),
            'seconds': round(time.monotonic() - started, 2), 'samples': samples,
        }
        (out / f'{path.stem}.json').write_text(json.dumps(item, indent=2), encoding='utf-8')
        summary.append({k: v for k, v in item.items() if k != 'samples'})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary[-1]), flush=True)
    print(f'Detector loaded: {status}')

if __name__ == '__main__':
    main()
