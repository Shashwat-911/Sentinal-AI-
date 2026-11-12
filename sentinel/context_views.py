"""Optional person detail views; never classify risk or change detector output."""
import math


def person_detail(frame, boxes):
    """Return a padded union of visible person boxes, or None for broad/tiny views."""
    height, width = frame.shape[:2]
    valid=[]
    for box in boxes:
        if len(box)<4 or not all(math.isfinite(float(v)) for v in box[:4]):
            continue
        x1,y1,x2,y2=box[:4]
        x1,y1=max(0,int(x1)),max(0,int(y1))
        x2,y2=min(width,int(x2)),min(height,int(y2))
        if x2>x1 and y2>y1 and (x2-x1)*(y2-y1)>=100:
            valid.append((x1,y1,x2,y2))
    if not valid:
        return None
    x1,y1=min(b[0] for b in valid),min(b[1] for b in valid)
    x2,y2=max(b[2] for b in valid),max(b[3] for b in valid)
    # Padding keeps nearby bodies and contacted surfaces, not just the detector box.
    pad_x=max(20,round((x2-x1)*.45));pad_y=max(15,round((y2-y1)*.30))
    x1,y1=max(0,x1-pad_x),max(0,y1-pad_y)
    x2,y2=min(width,x2+pad_x),min(height,y2+pad_y)
    if (x2-x1)*(y2-y1) > .80*width*height:
        return None
    return frame[y1:y2,x1:x2].copy(), [x1,y1,x2,y2]
