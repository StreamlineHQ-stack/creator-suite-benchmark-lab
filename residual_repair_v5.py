"""V5 experimental conservative residual OCR pass. No API calls.
Only repairs residual OCR boxes supported by ORIGINAL-image OCR, so new
hallucinated text detections in scenery are less likely to be painted over.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from text_detector_v3 import detect_boxes as v3_detect, build_mask as v3_mask


def _overlap(a,b):
    x1,y1,x2,y2=a;u1,v1,u2,v2=b
    intersection=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return intersection/max(1,(x2-x1)*(y2-y1))


def residual_repair(original:Image.Image, cleaned:Image.Image, base_mask:Image.Image,
                    radius:int=4, padding:int=5):
    """Returns (new_cleaned, combined_mask, diagnostics). Never alters outside combined mask."""
    original=original.convert('RGB');cleaned=cleaned.convert('RGB')
    if original.size!=cleaned.size:raise ValueError('Image dimensions must match')
    original_boxes=v3_detect(original,min_confidence=35)
    residual_boxes=v3_detect(cleaned,min_confidence=45)
    accepted=[]
    for item in residual_boxes:
        if any(_overlap(item['bbox'], b['bbox'])>=.55 for b in original_boxes):
            accepted.append(item)
    old=np.asarray(base_mask.convert('L'))>0
    # Build glyph candidates on ORIGINAL pixels, not reconstructed artifacts.
    additions=np.asarray(v3_mask(original,accepted,padding=padding))>0 if accepted else np.zeros_like(old)
    additions &= ~old
    # Limit how much any single residual pass may change. If too broad, refuse repair.
    maximum=max(100,int(old.size*.035))
    count=int(additions.sum())
    if count>maximum:
        return cleaned,base_mask,{'status':'skipped_excessive_mask','residual_ocr':len(residual_boxes),
                                  'confirmed_boxes':len(accepted),'new_pixels':count}
    if count==0:
        return cleaned,base_mask,{'status':'no_safe_additions','residual_ocr':len(residual_boxes),
                                  'confirmed_boxes':len(accepted),'new_pixels':0}
    # Inpaint ORIGINAL once using combined mask to avoid accumulating blur.
    combined=(old|additions).astype(np.uint8)*255
    src=cv2.cvtColor(np.asarray(original),cv2.COLOR_RGB2BGR)
    out=cv2.inpaint(src,combined,float(max(1,radius)),cv2.INPAINT_TELEA)
    out[combined==0]=src[combined==0]
    return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)),Image.fromarray(combined,'L'),{
        'status':'repaired','residual_ocr':len(residual_boxes),'confirmed_boxes':len(accepted),
        'new_pixels':count}
