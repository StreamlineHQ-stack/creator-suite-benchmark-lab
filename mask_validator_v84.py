"""Conservative experimental guard for isolated round decorative objects."""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


def inspect_region(rgb, box):
    x,y,x2,y2=box['bbox']; w=x2-x; h=y2-y
    if min(w,h)<24 or not .72<=w/h<=1.39:
        return 'retain_shape'
    crop=rgb[y:y2,x:x2]
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    # Normalize candidate scale for circle detection; require a near-central large circle.
    g=cv2.resize(gray,(128,128)); g=cv2.GaussianBlur(g,(5,5),1)
    circles=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1,60,param1=70,param2=25,minRadius=38,maxRadius=65)
    if circles is None or not any(abs(cx-64)<17 and abs(cy-64)<17 for cx,cy,r in circles[0]):
        return 'retain_no_circle'
    # Independent positive text evidence always overrides circle suspicion.
    for psm in (6,11):
        try:
            data=pytesseract.image_to_data(cv2.resize(gray,None,fx=max(1,160/min(w,h)),fy=max(1,160/min(w,h))),config=f'--psm {psm}',output_type=Output.DICT,timeout=5)
        except (RuntimeError,pytesseract.TesseractError):
            return 'retain_ocr_error'
        for text,conf in zip(data['text'],data['conf']):
            if any(c.isalnum() for c in str(text)) and float(conf)>=25:
                return 'retain_text_evidence'
    return 'reject_round_no_text'


def validate(image,boxes,padding=5):
    rgb=np.asarray(image.convert('RGB')); mask=np.zeros(rgb.shape[:2],np.uint8); kept=[]; decisions=[]
    for box in boxes:
        reason=inspect_region(rgb,box)
        decisions.append(dict(box,decision=reason))
        if reason.startswith('reject_'): continue
        cv2.fillPoly(mask,[np.asarray(box['polygon'],np.int32)],255); kept.append(box)
    p=max(0,min(12,int(padding)))
    if p: mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*p+1,2*p+1)))
    return Image.fromarray(mask),kept,decisions


def extract_v84(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from lama_v82 import reconstruct
    _,_,boxes,_=extract_v81(image,padding,radius)
    mask,kept,decisions=validate(image,boxes,padding)
    extract_v84.last_decisions=decisions
    return reconstruct(image,mask),mask,kept,float(np.count_nonzero(np.asarray(mask))/ (image.width*image.height))
