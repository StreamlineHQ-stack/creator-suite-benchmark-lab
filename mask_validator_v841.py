"""Experimental extension of V8.4; preserve ambiguous candidates by default."""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
from mask_validator_v84 import inspect_region


def inspect_advanced(rgb, box):
    reason=inspect_region(rgb,box)
    if reason!='retain_text_evidence': return reason
    x,y,x2,y2=box['bbox']; crop=rgb[y:y2,x:x2]
    # V8.4 has already found a central circle. Require independent color
    # and smooth-shading evidence before challenging a weak OCR token.
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    sat=hsv[:,:,1]>65
    if sat.mean()<.18: return reason
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    small=cv2.resize(gray,(128,128)); smooth=cv2.GaussianBlur(small,(9,9),2)
    if np.mean(np.abs(small.astype(float)-smooth))>9: return reason
    if np.unique((gray//8)).size<14: return reason
    g=cv2.resize(gray,None,fx=max(1,160/min(gray.shape)),fy=max(1,160/min(gray.shape)))
    variants=[(g,6),(g,11),(cv2.createCLAHE(2,(4,4)).apply(g),11),(g,7)]
    weak=False
    clean_short=set()
    for index,(a,psm) in enumerate(variants):
        try: d=pytesseract.image_to_data(a,config=f'--psm {psm}',output_type=Output.DICT,timeout=5)
        except (RuntimeError,pytesseract.TesseractError): return 'retain_advanced_ocr_error'
        for token,conf in zip(d['text'],d['conf']):
            t=str(token).strip(); n=sum(c.isalnum() for c in t)
            if float(conf)<25 or not n: continue
            if n>=3 or (t.isalnum() and n==1): return 'retain_advanced_text_evidence'
            if n==2 and t.isalnum():
                clean_short.add(index==2)
                if len(clean_short)>1: return 'retain_advanced_text_evidence'
            else: weak=True
    return 'reject_shaded_round_weak_ocr' if weak else reason


def validate(image,boxes,padding=5):
    rgb=np.asarray(image.convert('RGB')); mask=np.zeros(rgb.shape[:2],np.uint8); kept=[]; decisions=[]
    for box in boxes:
        reason=inspect_advanced(rgb,box); decisions.append(dict(box,decision=reason))
        if reason.startswith('reject_'): continue
        cv2.fillPoly(mask,[np.asarray(box['polygon'],np.int32)],255); kept.append(box)
    p=max(0,min(12,int(padding)))
    if p: mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*p+1,2*p+1)))
    return Image.fromarray(mask),kept,decisions


def extract_v841(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from lama_v82 import reconstruct
    _,_,boxes,_=extract_v81(image,padding,radius)
    mask,kept,decisions=validate(image,boxes,padding)
    extract_v841.last_decisions=decisions
    return reconstruct(image,mask),mask,kept,float(np.count_nonzero(np.asarray(mask))/(image.width*image.height))
