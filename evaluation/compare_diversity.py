"""Resumable full-duration review; independent annotations remain pending until inspected."""
import json
import sys
import time
import hashlib
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.review_demo_context import make_packet
from evaluation.review_suite import inventory
from sentinel.context import assess
from sentinel.context_report import format_report
from sentinel.voting import classify_window
from sentinel.detection import load_person_model, detect_persons, iou_deduplicate
from sentinel.tracking import match_entities
from sentinel.posture import load_pose_model, estimate_postures
from sentinel.risk import analyze_behaviors
from sentinel.config import SentinelConfig

OUT = ROOT/'evaluation/context_review/diversity_50_v1'

def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'demo_videos/diversity_50/MANIFEST.json').read_text())
    config = {'model': 'qwen3-vl:4b-instruct', 'detector_fps': 5,
        'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('evaluation/compare_diversity.py', 'sentinel/context.py', 'sentinel/context_report.py', 'sentinel/voting.py', 'sentinel/behavior.py', 'sentinel/tracking.py')}}
    conf = OUT/'configuration.json'
    if conf.exists() and json.loads(conf.read_text()) != config:
        raise RuntimeError('Sources changed; use a fresh output directory.')
    save(conf, config)
    windows = []
    for item in manifest['files']:
        path = ROOT/item['file']
        cap = cv2.VideoCapture(str(path)); fps=cap.get(5); frames=int(cap.get(7)); cap.release()
        if fps <= 0 or frames <= 0: raise RuntimeError(str(path))
        end = (frames-1)/fps
        start = 0
        while start < end:
            windows.append({'file': str(path.relative_to(ROOT/'demo_videos')), 'start': start, 'end': min(end, start+29)})
            if start+29 >= end: break
            start += 25
    save(OUT/'windows.json', windows)
    inventory(OUT, windows)
    SentinelConfig.DEBUG_DETECTION=False
    torch.set_num_threads(2)
    model,status=load_person_model(str(ROOT)); pose=load_pose_model(str(ROOT))
    if isinstance(pose,tuple): pose=pose[0]
    if model is None or pose is None: raise RuntimeError(status)
    done=failed=0
    for item in manifest['files']:
        path=ROOT/item['file']; folder=OUT/path.stem; folder.mkdir(exist_ok=True)
        cache=folder/'telemetry.json'
        if cache.exists(): history=json.loads(cache.read_text())
        else:
            cap=cv2.VideoCapture(str(path)); fps=cap.get(5); stride=max(1,round(fps/5))
            tracks={}; history=[]; index=0
            while True:
                ok,frame=cap.read()
                if not ok: break
                if index%stride==0:
                    t=index/fps
                    boxes=detect_persons(frame,model,device='cpu')
                    tracks=match_entities(iou_deduplicate(boxes),tracks,current_time=t)
                    estimate_postures(frame,tracks,pose,current_time=t)
                    analyze_behaviors(tracks,frame.shape[1],frame.shape[0],current_time=t)
                    people=[dict(track_id=p['track_id'],signals=sorted(p.get('active_threats',[])),behavior=p.get('behavior_state','NORMAL'),posture=p.get('posture','UNKNOWN')) for p in tracks.values() if not p.get('missed_frames',0)]
                    history.append({'time':t,'people':people})
                index+=1
            cap.release(); save(cache,history)
        for w in [w for w in windows if Path(w['file']).stem==path.stem]:
            name=str(w['start']); target=folder/(name+'.json')
            if target.exists() and 'report' in json.loads(target.read_text()): done+=1; continue
            try:
                packet,sheet=make_packet(path,[w['start'],w['end']],adaptive=True)
                for e in packet['evidence']:
                    e['people']=min(history,key=lambda h:abs(h['time']-e['time']))['people'] if history else []
                boundary=packet.get('scene_boundary_seconds')
                signals={s for h in history if max(w['start'],boundary or 0)<=h['time']<=w['end'] for p in h['people'] for s in p['signals']}
                packet['rule_assessment']='review' if signals & {'SUSPICIOUS_LOITERING','HIGH_CONFIDENCE_SUSPICIOUS','REPEATED_MOVEMENT'} else 'observe'
                packet['safety_event']='POSSIBLE_FALL' in signals
                cv2.imwrite(str(folder/(name+'_evidence.jpg')),sheet)
                report=assess(packet,config['model']); report['combined_decision']=classify_window(report,packet)
                save(target,dict(window=w,report=report,packet={k:v for k,v in packet.items() if k!='images'},independent_review='pending'))
                (folder/(name+'.txt')).write_text(format_report(report,packet,w['end']),encoding='utf-8')
                done+=1
            except Exception as exc:
                failed+=1; save(target,dict(window=w,error=str(exc),independent_review='pending'))
            save(OUT/'progress.json',dict(completed_windows=done,failed_windows=failed,total_windows=len(windows),latest_video=path.name,latest_window=w,independent_review='pending'))
            print(path.name,name,done,'/',len(windows),'errors',failed,flush=True)
    save(OUT/'finished.json',dict(completed_windows=done,failed_windows=failed,independent_review='pending'))

if __name__=='__main__': main()
