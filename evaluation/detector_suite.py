"""Decode every demo frame; run production person/pose inference at sampled times.

This is a smoke/coverage test, not frame-level detector accuracy measurement.
"""
import json
import sys
import time
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sentinel.config import SentinelConfig
from sentinel.detection import load_person_model, detect_persons, iou_deduplicate
from sentinel.tracking import match_entities
from sentinel.posture import load_pose_model, estimate_postures

def main():
    SentinelConfig.DEBUG_DETECTION = False
    model,status = load_person_model(str(ROOT))
    pose = load_pose_model(str(ROOT))
    if isinstance(pose,tuple): pose=pose[0]
    if model is None or pose is None: raise RuntimeError(status)
    torch.set_num_threads(2)
    output=ROOT/'evaluation/context_review/detector_suite'
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    manifest=json.loads((ROOT/'demo_videos/manifest.json').read_text())
    for item in manifest['files']:
        path=ROOT/'demo_videos'/item['file']
        cap=cv2.VideoCapture(str(path)); fps=cap.get(5); expected=int(cap.get(7))
        count=0; samples=[]; tracks={}; started=time.monotonic()
        while True:
            ok,frame=cap.read()
            if not ok: break
            timestamp=count/fps
            if count % max(1,round(fps*2)) == 0 or count == expected-1:
                detections=detect_persons(frame,model,device='cpu')
                tracks=match_entities(iou_deduplicate(detections),tracks,current_time=timestamp)
                alerts=estimate_postures(frame,tracks,pose,current_time=timestamp)
                torch.set_num_threads(2)
                samples.append(dict(time=timestamp,detections=detections,
                    tracks=[dict(id=t['track_id'],bbox=t['bbox'],posture=t.get('posture'),
                    missed=t.get('missed_frames')) for t in tracks.values()],alerts=alerts))
            count+=1
        cap.release()
        result=dict(file=item['file'],decoded=count,expected=expected,sampled_frames=len(samples),
                    samples_with_people=sum(bool(s['detections']) for s in samples),
                    seconds=round(time.monotonic()-started,2),samples=samples)
        (output/f'{path.stem}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        results.append({k:v for k,v in result.items() if k!='samples'})
        (output/'summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results[-1]),flush=True)

if __name__=='__main__': main()
