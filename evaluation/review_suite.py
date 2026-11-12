"""Reproducible local demo review. Dataset names never enter model prompts."""
import argparse
import json
import hashlib
import base64
import sys
import time
import urllib.error
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.review_demo_context import make_packet
from sentinel.context import assess, MODEL
from sentinel.context_report import format_report


def add_person_views(packet):
    from sentinel.context_views import person_detail
    from sentinel.detection import load_person_model, detect_persons
    from sentinel.config import SentinelConfig
    SentinelConfig.DEBUG_DETECTION=False
    if not hasattr(add_person_views,'model'):
        add_person_views.model,status=load_person_model(str(ROOT))
        if add_person_views.model is None: raise RuntimeError(status)
    frames=[cv2.imdecode(np.frombuffer(base64.b64decode(encoded),dtype=np.uint8),cv2.IMREAD_COLOR)
            for encoded in packet['images']]
    boxes=[box for frame in frames for box in detect_persons(frame,add_person_views.model,device='cpu')]
    # A fixed union keeps framing stable across the window and survives individual
    # detector misses. Preserve the first full scene for environmental context.
    for index,frame in enumerate(frames[1:],1):
        detail=person_detail(frame,boxes)
        if detail is None: continue
        crop,rectangle=detail
        packet['images'][index]=base64.b64encode(cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,90])[1]).decode()
        packet['visual_frames'][index].update(view='person_detail',source_crop_xyxy=rectangle)


def inventory(output, windows=None):
    if windows:
        paths = sorted({ROOT / 'demo_videos' / item['file'] for item in windows})
    else:
        manifest = json.loads((ROOT / 'demo_videos/manifest.json').read_text())
        paths = [ROOT / 'demo_videos' / item['file'] for item in manifest['files']]
    rows = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise RuntimeError(f'Unreadable video: {path}')
        end = (count - 1) / fps
        tiles = []
        for timestamp in np.linspace(0, end, 20):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Decode failure: {path} at {timestamp}')
            tile = np.zeros((268, 320, 3), dtype=np.uint8)
            tile[28:] = cv2.resize(frame, (320, 240))
            cv2.putText(tile, f'{timestamp:.2f}s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
            tiles.append(tile)
        cap.release()
        cv2.imwrite(str(output / f'{path.stem}_overview.jpg'), np.vstack([
            np.hstack(tiles[i:i+4]) for i in range(0,20,4)]))
        rows.append(dict(file=str(path.relative_to(ROOT / 'demo_videos')), frames=count, fps=fps, end=end))
    (output / 'inventory.json').write_text(json.dumps(rows, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Unique local artifact directory name')
    parser.add_argument('--windows', type=Path, help='JSON list of file/start/end review intervals')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--short-window', action='store_true', help='Experimental uniform sampling of the latest eight seconds')
    parser.add_argument('--person-views', action='store_true', help='Experimental padded person detail views after the first full frame')
    parser.add_argument('--framewise', action='store_true', help='Experimental separate observations for each image before report synthesis')
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        parser.error('--run must be a directory name')
    output = ROOT / 'evaluation/context_review' / args.run
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'configuration.json'
    sources = ['sentinel/context.py', 'sentinel/context_report.py', 'sentinel/context_views.py',
               'evaluation/review_demo_context.py', 'evaluation/review_suite.py']
    configuration = dict(model=args.model, short_window=args.short_window, person_views=args.person_views, framewise=args.framewise,
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        windows_sha256=hashlib.sha256(args.windows.read_bytes()).hexdigest() if args.windows else None)
    if metadata.exists() and json.loads(metadata.read_text()) != configuration:
        raise RuntimeError('Run configuration changed. Use a new --run name to avoid mixing experiments.')
    metadata.write_text(json.dumps(configuration, indent=2))
    windows = json.loads(args.windows.read_text()) if args.windows else None
    inventory(output, windows)
    if windows is None:
        return
    (output / 'review_windows.json').write_text(json.dumps(windows, indent=2))
    results = []
    for window in windows:
        path = ROOT / 'demo_videos' / window['file']
        name = f'{path.stem}_{window["start"]}_{window["end"]}'
        target = output / f'{name}.json'
        if target.exists():
            result = json.loads(target.read_text())
            if result.get('model') == args.model and 'report' in result:
                results.append(result)
                continue
        start = time.monotonic()
        result = dict(window=window, model=args.model)
        try:
            packet, sheet = make_packet(path, [window['start'], window['end']], adaptive=True, short_window=args.short_window)
            if args.framewise: packet['framewise_review']=True
            if args.person_views:
                from sentinel.context import chronological_sheet
                add_person_views(packet)
                sheet=cv2.imdecode(np.frombuffer(base64.b64decode(chronological_sheet(packet)[0]),dtype=np.uint8),cv2.IMREAD_COLOR)
            cv2.imwrite(str(output / f'{name}.jpg'), sheet)
            report = assess(packet, args.model)
            result.update(report=report, packet={k:v for k,v in packet.items() if k != 'images'})
            (output / f'{name}.txt').write_text(format_report(report, packet, window['end']), encoding='utf-8')
        except Exception as exc:
            detail = exc.read().decode('utf-8', errors='replace').strip() if isinstance(exc, urllib.error.HTTPError) else ''
            result['error'] = f'{type(exc).__name__}: {detail or exc}'
        result['seconds'] = round(time.monotonic() - start, 2)
        target.write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(dict(case=name, classification=result.get('report',{}).get('classification'),
                             error=result.get('error'), seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
