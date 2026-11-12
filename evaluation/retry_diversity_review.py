import json,sys,cv2,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from evaluation.review_demo_context import make_packet
from sentinel.context import assess
from sentinel.context_report import format_report
from sentinel.voting import classify_window
src=Path('evaluation/context_review/diversity_50_v1'); out=Path('evaluation/context_review/diversity_50_review_r1');out.mkdir(exist_ok=True)
cases=[('RoadAccidents151_x264',0),('Arson053_x264',150),('Arson053_x264',200),('Normal_Videos_745_x264',0)]
for name,start in cases:
    old=json.loads((src/name/(str(start)+'.json')).read_text());w=old['window']
    packet,sheet=make_packet(Path('demo_videos')/w['file'],[w['start'],w['end']],adaptive=True)
    history=json.loads((src/name/'telemetry.json').read_text())
    for e in packet['evidence']: e['people']=min(history,key=lambda h:abs(h['time']-e['time']))['people'] if history else []
    packet['safety_event']=old.get('packet',{}).get('safety_event',False)
    p=out/(name+'_'+str(start));cv2.imwrite(str(p.with_suffix('.jpg')),sheet)
    try:
        r=assess(packet,'qwen3-vl:4b-instruct');r['combined_decision']=classify_window(r,packet)
        p.with_suffix('.json').write_text(json.dumps(dict(window=w,report=r,packet={k:v for k,v in packet.items() if k!='images'}),indent=2))
        p.with_suffix('.txt').write_text(format_report(r,packet,w['end']),encoding='utf-8')
        print(name,start,r['classification'],flush=True)
    except Exception as e: print(name,'ERROR',str(e),flush=True)
