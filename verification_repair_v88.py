"""Bounded reference-free selection; never certifies visual quality."""
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
from difflib import SequenceMatcher
from mask_validator_v841 import validate
from hybrid_mask_v851 import refine
from secondary_detector_v86 import recover
from contextual_protection_v87 import protect


def words(image):
    try:
        d=pytesseract.image_to_data(image.convert('L'),config='--psm 11',output_type=Output.DICT,timeout=8)
    except (RuntimeError,pytesseract.TesseractError):
        return None
    out=[]
    for i,t in enumerate(d['text']):
        t=''.join(c.lower() for c in t if c.isalnum())
        if len(t)<3 or float(d['conf'][i])<65: continue
        x,y,w,h=[int(d[k][i]) for k in ('left','top','width','height')]
        if w*h>0: out.append((t,(x,y,x+w,y+h)))
    return out


def residual(source, output, target):
    if source is None or output is None: return None
    hits=[]
    for t,(x,y,X,Y) in source:
        roi=target[y:Y,x:X]
        if not roi.size or roi.mean()<.12: continue
        for u,(a,b,A,B) in output:
            overlap=max(0,min(X,A)-max(x,a))*max(0,min(Y,B)-max(y,b))
            if overlap/max(1,min((X-x)*(Y-y),(A-a)*(B-b)))>=.45 and SequenceMatcher(None,t,u).ratio()>=.8:
                hits.append((t,(x,y,X,Y))); break
    return hits


def select(image, baseline, protected, boxes, log, reconstruct, padding=5):
    src=np.asarray(image.convert('RGB')); base=np.asarray(baseline)>127; safe=np.asarray(protected)>127
    if np.any(safe & ~base): raise ValueError('Protection expanded mask')
    # Deliberately preserve small-print regions explicitly classified as scene print.
    target=base.copy()
    for e in log:
        if e['decision']=='preserve_possible_scene_print':
            x,y,X,Y=e['bbox']; target[y:Y,x:X]=False
    delta=base & ~safe
    reduction=float(delta.mean())
    diag={'mask_reduction':reduction,'candidates':[],'status':'REVIEW','selected':'v86','repair':'not_attempted'}
    def run(m,name):
        mask=Image.fromarray(m.astype('uint8')*255)
        result=reconstruct(image,mask)
        arr=np.asarray(result.convert('RGB'))
        if arr.shape!=src.shape or np.any(arr[~m]!=src[~m]): raise ValueError('Reconstruction invariant violation')
        hits=residual(original_words,words(result),target) if original_words is not None else None
        diag['candidates'].append({'name':name,'coverage':float(m.mean()),'residual_matches':None if hits is None else len(hits)})
        return result,mask,hits
    original_words=words(image)
    diag['source_ocr_words']=None if original_words is None else len(original_words)
    diag['target_ocr_words']=None if original_words is None else len(residual(original_words,original_words,target))
    chosen=run(base,'v86')
    # Avoid a second inference for small differences or unavailable OCR.
    if reduction>=.025 and reduction/max(float(base.mean()),1e-9)>=.20 and original_words is not None and chosen[2] is not None:
        try:
            alt=run(safe,'v87')
            if alt[2] is not None and len(alt[2])<=len(chosen[2]):
                chosen=alt; diag['selected']='v87'
        except Exception as exc: diag['alternative_error']=str(exc)
    # Only restore pixels from the previously validated baseline near matched residuals.
    if diag['selected']=='v87' and chosen[2]:
        current=np.asarray(chosen[1])>127; patch=np.zeros_like(current)
        for _,(x,y,X,Y) in chosen[2]:
            p=min(4,max(0,padding)); patch[max(0,y-p):Y+p,max(0,x-p):X+p]=True
        addition=patch & base & ~current & target
        if 0<addition.mean()<=.025:
            try:
                repaired=run(current|addition,'selective_repair')
                if repaired[2] is not None and len(repaired[2])<len(chosen[2]):
                    chosen=repaired; diag['repair']='accepted'; diag['selected']='v87_repaired'
                else: diag['repair']='rejected'
            except Exception as exc: diag['repair']='error'; diag['repair_error']=str(exc)
    diag['residual_matches']=None if chosen[2] is None else len(chosen[2])
    return chosen[0],chosen[1],boxes,float(np.mean(np.asarray(chosen[1])>127)),diag


def extract_v88(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from lama_v82 import reconstruct
    image=image.convert('RGB')
    _,_,boxes,_=extract_v81(image,padding,radius)
    _,kept,_=validate(image,boxes,padding)
    baseline,_,_=refine(image,kept,padding)
    combined,extra,_=recover(image,boxes,baseline,padding)
    secondary=Image.fromarray(np.asarray(combined)&~np.asarray(baseline))
    protected,log=protect(image,kept,secondary,padding)
    protected=Image.fromarray(np.asarray(protected)&np.asarray(combined))
    result,mask,boxes,coverage,diag=select(image,combined,protected,kept+extra,log,reconstruct,padding)
    extract_v88.last_diagnostics=diag
    return result,mask,boxes,coverage
