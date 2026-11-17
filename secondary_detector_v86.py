"""Bounded independent OCR recovery; adds candidates without changing V8.5.1 masks."""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
from hybrid_mask_v851 import refine
from mask_validator_v841 import validate

def recover(image, primary, baseline, padding=5):
    rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
    scale=min(2.,1600/max(h,w)); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    g=cv2.resize(gray,None,fx=scale,fy=scale)
    variants=[('gray',g,11),('contrast',cv2.createCLAHE(clipLimit=3,tileGridSize=(8,8)).apply(g),11),('binary',cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],6)]
    occupied=np.zeros((h,w),np.uint8)
    for b in primary: cv2.fillPoly(occupied,[np.asarray(b['polygon'],np.int32)],255)
    base=np.asarray(baseline.convert('L')).copy(); added=np.zeros_like(base); accepted=[]; log=[]
    for name,arr,psm in variants:
        try: d=pytesseract.image_to_data(arr,config=f'--psm {psm}',output_type=Output.DICT,timeout=8)
        except (RuntimeError,pytesseract.TesseractError) as e:
            log.append({'variant':name,'decision':'ocr_error','error':str(e)}); continue
        for i,t in enumerate(d['text']):
            t=str(t).strip(); conf=float(d['conf'][i]); n=sum(c.isalnum() for c in t)
            if n<3 or conf<70 or n/max(1,len(t))<.7: continue
            x,y,bw,bh=[int(d[k][i]) for k in ('left','top','width','height')]
            x1=max(0,int(x/scale)); y1=max(0,int(y/scale)); x2=min(w,int(np.ceil((x+bw)/scale))); y2=min(h,int(np.ceil((y+bh)/scale)))
            if x2-x1<12 or y2-y1<12: continue
            box=dict(text=t,confidence=conf,bbox=[x1,y1,x2,y2],polygon=[[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]])
            entry=dict(box,variant=name)
            if np.mean(occupied[y1:y2,x1:x2]>0)>.25 or np.mean(added[y1:y2,x1:x2]>0)>.25:
                log.append(dict(entry,decision='overlap')); continue
            _,kept,_=validate(image,[box],padding)
            if not kept: log.append(dict(entry,decision='object_rejected')); continue
            part,dec,_=refine(image,kept,padding)
            # Never let uncertain secondary rectangles erase overlapping subjects wholesale.
            if dec[0]['decision'] not in ('pixel_expanded','regional_simple'):
                # A partial large OCR token on a dark background can still support
                # neutral bright glyph pixels, without masking colorful occluders.
                crop=rgb[y1:y2,x1:x2]; low=crop.min(axis=2); high=crop.max(axis=2)
                core=(low>=225)&((high-low)<=18)
                if y2-y1<.055*h or x2-x1<.10*w or not .08<=core.mean()<=.65 or np.median(low)>145:
                    log.append(dict(entry,decision='uncertain_mask')); continue
                local=np.zeros_like(base); local[y1:y2,x1:x2]=core.astype(np.uint8)*255
                p=max(0,min(4,int(padding)))
                if p: local=cv2.dilate(local,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*p+1,2*p+1)))
                # Constrain growth to near-neutral pixels; colored subjects remain outside.
                neutral=(rgb.max(axis=2)-rgb.min(axis=2)<=35)
                part=Image.fromarray(local*neutral.astype(np.uint8))
                entry['mask_method']='neutral_glyph_core' 
            new=np.asarray(part)&~base&~added
            if not np.any(new) or (np.count_nonzero(added)+np.count_nonzero(new))> .18*h*w:
                log.append(dict(entry,decision='empty_or_budget')); continue
            added|=new; accepted.append(box); log.append(dict(entry,decision='accepted'))
    return Image.fromarray(base|added),accepted,log

def extract_v86(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from lama_v82 import reconstruct
    _,_,boxes,_=extract_v81(image,padding,radius)
    _,kept,imv=validate(image,boxes,padding)
    baseline,_,_=refine(image,kept,padding)
    mask,extra,log=recover(image,boxes,baseline,padding)
    extract_v86.last_secondary=log
    extract_v86.baseline_coverage=float(np.count_nonzero(np.asarray(baseline))/ (image.width*image.height))
    return reconstruct(image,mask),mask,kept+extra,float(np.count_nonzero(np.asarray(mask))/(image.width*image.height))
