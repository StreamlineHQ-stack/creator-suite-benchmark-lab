"""Independent PaddleOCR polygon detector; no paid API calls."""
from functools import lru_cache
import threading
import cv2
import numpy as np
from PIL import Image
from extraction_engine import inpaint

_LOCK = threading.RLock()
@lru_cache(maxsize=1)
def _model():
    from paddleocr import TextDetection
    return TextDetection(model_name='PP-OCRv5_mobile_det', device='cpu', enable_mkldnn=False, cpu_threads=2)

def extract_v81(image, padding=5, radius=4):
    image = image.convert('RGB')
    rgb = np.asarray(image)
    h,w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h,w), np.uint8)
    boxes=[]
    with _LOCK:
        results=list(_model().predict(bgr, batch_size=1))
    for result in results:
        for poly, score in zip(result['dt_polys'], result['dt_scores']):
            pts=np.rint(poly).astype(np.int32).reshape(-1,2)
            if len(pts)<3 or not np.isfinite(score): continue
            pts[:,0]=np.clip(pts[:,0],0,w-1); pts[:,1]=np.clip(pts[:,1],0,h-1)
            if cv2.contourArea(pts)<4: continue
            cv2.fillPoly(mask,[pts],255)
            x,y,bw,bh=cv2.boundingRect(pts)
            boxes.append({'text':'','confidence':float(score)*100,'bbox':(x,y,x+bw,y+bh),'polygon':pts.tolist()})
    p=max(0,min(12,int(padding)))
    if p: mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*p+1,2*p+1)))
    m=Image.fromarray(mask)
    return inpaint(image,m,radius),m,boxes,float(np.count_nonzero(mask)/mask.size)
