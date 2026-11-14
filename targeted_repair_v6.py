"""V6 experimental original-supported residual repair. Local OCR only.
No model calls; candidate validation is heuristic, not proof of text removal.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from text_detector_v3 import detect_boxes as detect_v3, build_mask as mask_v3


def _intersection_ratio(a,b):
    x1,y1,x2,y2=a; u1,v1,u2,v2=b
    area=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return area/max(1,(x2-x1)*(y2-y1))


def repair(original:Image.Image, initial:Image.Image, base_mask:Image.Image,
           radius:int=4, padding:int=5, extra_rectangles=None):
    """Return (image, mask, diagnostics), preserving pixels outside combined mask."""
    original=original.convert('RGB'); initial=initial.convert('RGB')
    if original.size!=initial.size or original.size!=base_mask.size:
        raise ValueError('Original, result, and mask must have identical dimensions')
    base=np.asarray(base_mask.convert('L'))>0
    source_boxes=detect_v3(original,min_confidence=35)
    residual_boxes=detect_v3(initial,min_confidence=50)
    # Residual OCR alone is unsafe: require spatial evidence on the original.
    confirmed=[]
    for r in residual_boxes:
        if any(_intersection_ratio(r['bbox'],b['bbox'])>=.45 for b in source_boxes):
            confirmed.append(r)
    # Use V3's tighter glyph proposal only in confirmed residual boxes.
    proposed=np.asarray(mask_v3(original,confirmed,padding=max(1,padding//2)))>0 if confirmed else np.zeros_like(base)
    # Protect against large additions: per-region and whole-image limits.
    additions=np.zeros_like(base)
    for r in confirmed:
        x1,y1,x2,y2=map(int,r['bbox']); h,w=base.shape
        x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
        if x2<=x1 or y2<=y1:continue
        region=proposed[y1:y2,x1:x2]&~base[y1:y2,x1:x2]
        if region.sum()>0.48*region.size:continue
        additions[y1:y2,x1:x2] |= region
    # Extra rectangles are explicitly user-provided; never discard them.
    new_pixels=int(additions.sum())
    cap=max(100,int(base.size*.025))
    if new_pixels>cap:
        return initial,base_mask,{'status':'rejected_large_residual_mask','new_pixels':new_pixels,'residual_boxes':len(residual_boxes),'confirmed_boxes':len(confirmed)}
    if not new_pixels:
        return initial,base_mask,{'status':'no_confirmed_additions','new_pixels':0,'residual_boxes':len(residual_boxes),'confirmed_boxes':len(confirmed)}
    combined=(base|additions).astype(np.uint8)*255
    src=cv2.cvtColor(np.asarray(original),cv2.COLOR_RGB2BGR)
    out=cv2.inpaint(src,combined,float(max(1,radius)),cv2.INPAINT_TELEA)
    out[combined==0]=src[combined==0]
    return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)),Image.fromarray(combined,'L'),{
        'status':'repaired','new_pixels':new_pixels,'residual_boxes':len(residual_boxes),'confirmed_boxes':len(confirmed)}
