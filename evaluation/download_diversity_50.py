import requests,json,hashlib,concurrent.futures
from pathlib import Path
import cv2
root=Path('demo_videos'); out=root/'diversity_50'; out.mkdir(exist_ok=True)
existing={p.name.lower() for p in root.rglob('*.mp4') if out not in p.parents}
counts={'Testing_Normal':17,'Burglary':4,'Shoplifting':4,'Stealing':4,'Vandalism':3,'Robbery':2,'RoadAccidents':6,'Arrest':3,'Fighting':3,'Arson':2,'Assault':2}
base='https://huggingface.co/datasets/mteb/ucf-crime/resolve/main/'
jobs=[]
for category,count in counts.items():
    r=requests.get('https://huggingface.co/api/datasets/mteb/ucf-crime/tree/main/videos/'+category,timeout=60); r.raise_for_status()
    candidates=[x for x in r.json() if x['path'].endswith('.mp4') and Path(x['path']).name.lower() not in existing]
    candidates.sort(key=lambda x:x['path'])
    selected=[candidates[round(i*(len(candidates)-1)/max(1,count-1))] for i in range(count)]
    jobs.extend((category,x) for x in selected)
def download(job):
    category,item=job; path=out/category/Path(item['path']).name; path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with requests.get(base+item['path'],stream=True,timeout=120) as r:
            r.raise_for_status()
            with path.with_suffix('.part').open('wb') as f:
                for chunk in r.iter_content(1048576): f.write(chunk)
        path.with_suffix('.part').replace(path)
    if path.stat().st_size!=item['size']: raise RuntimeError('Size mismatch '+str(path))
    cap=cv2.VideoCapture(str(path)); fps=cap.get(5); frames=int(cap.get(7)); ok,frame=cap.read(); cap.release()
    if not ok or fps<=0: raise RuntimeError('Unreadable '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return dict(file=str(path),category=category,source=base+item['path'],bytes=path.stat().st_size,sha256=h.hexdigest(),duration_seconds=frames/fps,width=frame.shape[1],height=frame.shape[0])
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for future in concurrent.futures.as_completed([pool.submit(download,j) for j in jobs]):
        results.append(future.result()); (out/'MANIFEST.json').write_text(json.dumps({'source':'UCF-Crime via Hugging Face mteb mirror','note':'Dataset categories are not per-frame verified classifications. Download and opening-frame validation only.','files':results},indent=2))
        print(len(results),results[-1]['file'],flush=True)
print('COMPLETE',len(results),sum(x['bytes'] for x in results),flush=True)
