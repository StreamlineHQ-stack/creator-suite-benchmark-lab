"""V4 experimental OCR detector. Drop-in replacement interface for V3.
Conservative proposals; manual rectangles remain supported.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _ocr(rgb, psm, scale=1.0, min_conf=25):
    import pytesseract
    from pytesseract import Output
    h,w=rgb.shape[:2]
    work=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale!=1 else rgb
    data=pytesseract.image_to_data(Image.fromarray(work),config=f'--psm {psm}',output_type=Output.DICT)
    result=[]
    for i,t in enumerate(data['text']):
        t=str(t).strip()
        if not t:continue
        try:
            conf=float(data['conf'][i]); x=int(data['left'][i]/scale);y=int(data['top'][i]/scale)
            bw=int(np.ceil(data['width'][i]/scale));bh=int(np.ceil(data['height'][i]/scale))
        except (ValueError,TypeError,KeyError):continue
        if conf<min_conf or bw<2 or bh<2:continue
        box=(max(0,x),max(0,y),min(w,x+bw),min(h,y+bh))
        if box[2]<=box[0] or box[3]<=box[1]:continue
        result.append({'text':t,'confidence':round(conf,1),'bbox':box,'psm':psm})
    return result


def _overlap(a,b):
    x1,y1,x2,y2=a;u1,v1,u2,v2=b
    inter=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    area1=(x2-x1)*(y2-y1);area2=(u2-u1)*(v2-v1)
    return inter/max(1,min(area1,area2))


def detect_boxes(image:Image.Image,min_confidence:int=25):
    src=np.asarray(image.convert('RGB'));h,w=src.shape[:2]
    primary=_ocr(src,11,1.0,min_confidence)
    secondary=_ocr(src,6,1.0,min_confidence)
    # Only run costly upscale when neither native pass detected any text.
    fallback=_ocr(src,11,1.7,min_confidence) if not(primary or secondary) and max(h,w)<1400 else []
    candidates=primary+secondary+fallback
    candidates.sort(key=lambda b:-b['confidence'])
    result=[]
    for item in candidates:
        if any(_overlap(item['bbox'],other['bbox'])>.82 for other in result):continue
        # Low-confidence isolated proposals can be background textures mistaken for text.
        # This is deliberately conservative and can miss faint lettering.
        if item['confidence']<40 and not any(_overlap(item['bbox'],other['bbox'])>.45 and item['psm']!=other['psm'] for other in candidates if other is not item):
            continue
        result.append(item)
    for b in result:b.pop('psm',None)
    return result


def _mask_crop_with_context(rgb,box):
    h,w=rgb.shape[:2];x1,y1,x2,y2=box
    bw=x2-x1;bh=y2-y1
    if min(bw,bh)<4:return np.zeros((bh,bw),np.uint8)
    margin=max(3,min(18,int(round(min(bw,bh)*.35))))
    xa=max(0,x1-margin);ya=max(0,y1-margin);xb=min(w,x2+margin);yb=min(h,y2+margin)
    roi=rgb[ya:yb,xa:xb]
    lab=cv2.cvtColor(roi,cv2.COLOR_RGB2LAB).astype(np.float32)
    # Estimate background from a ring *outside* the word box, rather than from letter pixels.
    ring=np.ones(lab.shape[:2],np.uint8)
    ring[y1-ya:y2-ya,x1-xa:x2-xa]=0
    samples=lab[ring>0]
    if len(samples)<8:samples=np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])
    bg=np.median(samples,axis=0)
    local=lab[y1-ya:y2-ya,x1-xa:x2-xa]
    distance=np.linalg.norm(local-bg,axis=2)
    gray=cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
    threshold,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dark=gray<threshold;light=gray>threshold
    def score(c):
        fraction=float(c.mean());edge=float(np.concatenate((c[0],c[-1],c[:,0],c[:,-1])).mean())
        return abs(fraction-.28)+.8*edge
    candidate=min((dark,light),key=score)
    cutoff=max(9.0,float(np.percentile(distance,50)))
    proposed=(distance>cutoff)&candidate
    # Avoid filling entire rectangles when segmentation is uncertain.
    coverage=float(proposed.mean())
    if coverage<.015 or coverage>.62:
        proposed=candidate & (distance>max(6.0,float(np.percentile(distance,35))))
    if proposed.mean()>.68:return np.zeros((bh,bw),np.uint8)
    return proposed.astype(np.uint8)*255


def build_mask(image:Image.Image,boxes:list[dict],padding:int=5,extra_rectangles=None):
    src=np.asarray(image.convert('RGB'));h,w=src.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    for item in boxes:
        x1,y1,x2,y2=map(int,item['bbox']);x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
        if x2<=x1 or y2<=y1:continue
        part=_mask_crop_with_context(src,(x1,y1,x2,y2))
        grow=min(max(1,int(padding)//3),max(1,min(x2-x1,y2-y1)//5),4)
        part=cv2.dilate(part,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grow+1,2*grow+1)))
        mask[y1:y2,x1:x2]=np.maximum(mask[y1:y2,x1:x2],part)
    for rect in extra_rectangles or []:
        a,b,c,d=map(int,rect);x1,x2=max(0,min(a,c)),min(w,max(a,c));y1,y2=max(0,min(b,d)),min(h,max(b,d))
        if x2>x1 and y2>y1:mask[y1:y2,x1:x2]=255
    return Image.fromarray(mask,'L')
