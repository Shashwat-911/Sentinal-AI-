import json,sys,cv2
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from evaluation.review_demo_context import make_packet
from sentinel.context import assess
from sentinel.context_report import format_report
from sentinel.voting import classify_window
src=Path('evaluation/context_review/diversity_50_v1');out=Path('evaluation/context_review/diversity_50_review_r2');out.mkdir(exist_ok=True)
for name,start,end in [('Arson053_x264',150,179),('Vandalism026_x264',10,22)]:
 old=json.loads((src/name/('150.json' if start==150 else '0.json')).read_text());w=old['window']; path=Path('demo_videos')/w['file']
 packet,sheet=make_packet(path,[start+(end-start)*i/4 for i in range(5)],adaptive=False);packet['framewise_review']=True
 r=assess(packet,'qwen3-vl:4b-instruct');r['combined_decision']=classify_window(r,packet)
 (out/(name+'.json')).write_text(json.dumps(dict(window=dict(file=w['file'],start=start,end=end),report=r,packet={k:v for k,v in packet.items() if k!='images'},method='targeted visual-only framewise rerun; rule vote abstains'),indent=2))
 (out/(name+'.txt')).write_text(format_report(r,packet,end),encoding='utf-8');cv2.imwrite(str(out/(name+'.jpg')),sheet)
 print(name,r['classification'],flush=True)
