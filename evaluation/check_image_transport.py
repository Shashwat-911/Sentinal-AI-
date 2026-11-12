"""Compare adjacent vs text-separated image markers on the installed local runner."""
import base64,json,sys,urllib.request
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
images=[]
for shape,color in [('circle',(0,0,255)),('square',(255,0,0)),('triangle',(0,180,0))]:
    frame=np.full((240,320,3),255,dtype=np.uint8)
    if shape=='circle': cv2.circle(frame,(160,120),65,color,-1)
    if shape=='square': cv2.rectangle(frame,(95,55),(225,185),color,-1)
    if shape=='triangle': cv2.fillPoly(frame,[np.array([[160,45],[85,190],[235,190]])],color)
    images.append(base64.b64encode(cv2.imencode('.png',frame)[1]).decode())
rows=[]
for separated in [False,True]:
    content='List the color and shape in each supplied image in order. Do not combine images. Be brief.'
    if separated: content='Image 1: [img]\nImage 2: [img]\nImage 3: [img]\n'+content
    body=dict(model='qwen3-vl:4b-instruct',stream=False,think=False,
              options=dict(temperature=0,num_ctx=8192,num_predict=200),
              messages=[dict(role='user',content=content,images=images)])
    request=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response: result=json.load(response)
    rows.append(dict(separated=separated,response=result['message']['content'],prompt_tokens=result.get('prompt_eval_count')))
    print(json.dumps(rows[-1]),flush=True)
output=ROOT/'evaluation/context_review/image_transport.json'
output.write_text(json.dumps(rows,indent=2))
