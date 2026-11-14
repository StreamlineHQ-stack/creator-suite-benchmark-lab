"""Experimental multi-pass OCR text detector; no network or AI calls."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _ocr(image: np.ndarray, psm: int, scale: float = 1.0, min_conf: float = 25):
    import pytesseract
    from pytesseract import Output
    h,w=image.shape[:2]
    work=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale != 1 else image
    data=pytesseract.image_to_data(Image.fromarray(work),config=f'--psm {psm}',output_type=Output.DICT)
    found=[]
    for i,t in enumerate(data['text']):
        if not str(t).strip(): continue
        try: conf=float(data['conf'][i]);x=int(data['left'][i]/scale);y=int(data['top'][i]/scale);bw=int(np.ceil(data['width'][i]/scale));bh=int(np.ceil(data['height'][i]/scale))
        except (ValueError,KeyError,TypeError):continue
        if conf<min_conf or bw<2 or bh<2:continue
        x1=max(0,x);y1=max(0,y);x2=min(w,x+bw);y2=min(h,y+bh)
        if x2>x1 and y2>y1:found.append({'text':str(t).strip(),'confidence':round(conf,1),'bbox':(x1,y1,x2,y2)})
    return found


def detect_boxes(image: Image.Image, min_confidence: int=25):
    src=np.asarray(image.convert('RGB'))
    h,w=src.shape[:2]
    passes=[(src,11,1.0),(src,6,1.0)]
    # Additional upscaled OCR pass for small slide previews; useful for missed headline text.
    if max(h,w)<1400:passes.append((src,11,1.7))
    boxes=[]
    for arr,psm,scale in passes:
        boxes.extend(_ocr(arr,psm,scale,min_confidence))
    # Deduplicate highly overlapping word rectangles.
    result=[]
    for box in sorted(boxes,key=lambda b:-b['confidence']):
        a=box['bbox']; area=(a[2]-a[0])*(a[3]-a[1]); duplicate=False
        for old in result:
            b=old['bbox']; inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])); other=(b[2]-b[0])*(b[3]-b[1]);
            if inter/max(1,min(area,other))>.82:duplicate=True;break
        if not duplicate:result.append(box)
    return result


def _mask_crop(crop: np.ndarray):
    h,w=crop.shape[:2]
    if min(h,w)<4:return np.full((h,w),255,np.uint8)
    # V2 border-color glyph extraction, retained because it outperformed rectangular masks.
    lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)
    edge=np.concatenate((lab[0],lab[-1],lab[:,0],lab[:,-1]),axis=0)
    bg=np.median(edge.astype(np.float32),axis=0)
    delta=np.linalg.norm(lab.astype(np.float32)-bg,axis=2)
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    threshold,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    candidates=[(gray<threshold).astype(np.uint8),(gray>threshold).astype(np.uint8)]
    def score(c):
        border=float(np.concatenate((c[0],c[-1],c[:,0],c[:,-1])).mean())
        return abs(float(c.mean())-.28)+1.5*border
    cand=min(candidates,key=score)
    cutoff=max(10.,float(np.percentile(delta,55)))
    mask=((delta>cutoff)&(cand>0)).astype(np.uint8)*255
    coverage=np.count_nonzero(mask)/mask.size
    if coverage<.025 or coverage>.70:mask=cand*255
    return mask


def build_mask(image: Image.Image, boxes: list[dict], padding: int=5, extra_rectangles=None):
    src=np.asarray(image.convert('RGB'));h,w=src.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    for item in boxes:
        x1,y1,x2,y2=item['bbox'];x1=max(0,x1);x2=min(w,x2);y1=max(0,y1);y2=min(h,y2)
        if x2<=x1 or y2<=y1:continue
        part=_mask_crop(src[y1:y2,x1:x2])
        grow=min(max(1,int(padding)//3),max(1,min(x2-x1,y2-y1)//5),4)
        part=cv2.dilate(part,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grow+1,2*grow+1)))
        mask[y1:y2,x1:x2]=np.maximum(mask[y1:y2,x1:x2],part)
    for rect in extra_rectangles or []:
        a,b,c,d=map(int,rect);x1,x2=max(0,min(a,c)),min(w,max(a,c));y1,y2=max(0,min(b,d)),min(h,max(b,d));mask[y1:y2,x1:x2]=255
    return Image.fromarray(mask,'L')
