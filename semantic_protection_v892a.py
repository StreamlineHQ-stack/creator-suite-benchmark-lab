"""Optional ONNX person segmentation; conservative mask reduction, not quality certification."""
import os
from semantic_protection_v892 import session
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def person_probability(image):
    path=os.environ.get('CREATOR_PERSON_MODEL','models/person_lraspp.onnx')
    if not Path(path).is_file(): raise FileNotFoundError('Person segmentation model missing: '+path)
    rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
    scale=320/max(h,w); nh=max(1,round(h*scale)); nw=max(1,round(w*scale))
    small=cv2.resize(rgb,(nw,nh)); top=(320-nh)//2; left=(320-nw)//2
    canvas=cv2.copyMakeBorder(small,top,320-nh-top,left,320-nw-left,cv2.BORDER_REPLICATE)
    x=(canvas.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    logits=session(path).run(['logits'],{'image':np.ascontiguousarray(x.transpose(2,0,1)[None])})[0]
    if logits.shape!=(1,21,320,320) or not np.isfinite(logits).all(): raise ValueError('Unexpected segmentation output')
    logits=logits[0]; ex=np.exp(logits-logits.max(axis=0)); prob=ex[15]/ex.sum(axis=0)
    prob=cv2.resize(prob[top:top+nh,left:left+nw],(w,h))
    return prob

def reduce_mask(image,mask,boxes,prob,padding=5):
    from pixel_refiner_v85 import refine_region
    rgb=np.asarray(image.convert('RGB')); old=np.asarray(mask.convert('L'))>127; h,w=old.shape
    if prob.shape!=old.shape: raise ValueError('Subject mask dimensions')
    # Moderate-confidence pixels must connect to a high-confidence seed.
    n, labels=cv2.connectedComponents((prob>=.55).astype(np.uint8))
    seeds=np.unique(labels[prob>=.80]); seeds=seeds[seeds!=0]
    person=np.isin(labels,seeds)
    remove=np.zeros_like(old); keep=np.zeros_like(old); covered=np.zeros_like(old)
    refined=0
    for b in boxes:
        poly=np.asarray(b['polygon'],np.int32).copy(); poly[:,0]=np.clip(poly[:,0],0,w-1); poly[:,1]=np.clip(poly[:,1],0,h-1)
        region=np.zeros((h,w),np.uint8); cv2.fillPoly(region,[poly],255)
        p=max(1,min(8,int(padding))); kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*p+1,2*p+1)); full=cv2.dilate(region,kernel)>0
        covered|=full
        if not np.any(full & old & person): keep|=full; continue
        local,reason=refine_region(rgb,poly,p)
        if reason!='refined_color':
            # Only outer padding, never uncertain text polygon or its 2px halo.
            core=cv2.dilate(region,np.ones((5,5),np.uint8))>0
            keep|=core; remove|=full & ~core & (prob>=.85); continue
        refined+=1
        x,y,bw,bh=cv2.boundingRect(poly); glyph=np.zeros_like(region); glyph[y:y+bh,x:x+bw]=local
        glyph=cv2.dilate(glyph,kernel)>0
        keep|=glyph; remove|=full & ~glyph & person
    delta=remove & old & ~keep & covered
    fraction=float(delta.mean())
    if fraction>.10 or delta.sum()<max(12,old.size*.0001): delta[:]=False
    return Image.fromarray((old & ~delta).astype(np.uint8)*255),float(delta.mean()),dict(person_fraction=float(person.mean()),refined_regions=refined,proposed_fraction=fraction)

def extract_v892a(image,padding=5,radius=4):
    from glyph_detector_v891 import extract_v891
    from lama_v82 import reconstruct
    image=image.convert('RGB'); baseline=extract_v891(image,padding,radius); result,mask,boxes,coverage=baseline
    diag={'baseline':extract_v891.last_diagnostics,'status':'REVIEW','semantic':'not_applied','restored_fraction':0.0}
    try:
        prob=person_probability(image); diag['high_confidence_fraction']=float((prob>=.85).mean())
        candidate,delta,extra=reduce_mask(image,mask,boxes,prob,padding); diag.update(extra)
        diag['candidate_restored_fraction']=delta
        if delta:
            output=reconstruct(image,candidate); m=np.asarray(candidate)>127
            if output.size!=image.size or np.any(np.asarray(output.convert('RGB'))[~m]!=np.asarray(image)[~m]): raise ValueError('Outside-mask invariant failed')
            baseline=(output,candidate,boxes,float(m.mean())); diag.update(semantic='applied',restored_fraction=delta)
        else: diag['semantic']='no_safe_reduction'
    except Exception as exc: diag.update(semantic='fallback',error=str(exc))
    extract_v892a.last_diagnostics=diag
    return baseline
