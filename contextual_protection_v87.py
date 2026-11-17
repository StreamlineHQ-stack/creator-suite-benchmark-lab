"""Experimental color-context protection; not semantic segmentation."""
import cv2
import numpy as np
from PIL import Image
from hybrid_mask_v851 import refine
from mask_validator_v841 import validate
from secondary_detector_v86 import recover

def protect(image, boxes, secondary_mask=None, padding=5):
    rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.int16)
    parts=[]; log=[]; large=[]
    for b in boxes:
        p=np.asarray(b['polygon'],np.int32); region=np.zeros((h,w),np.uint8); cv2.fillPoly(region,[p],255)
        part,dec,_=refine(image,[b],padding); base=np.array(part); x,y,bw,bh=cv2.boundingRect(p)
        entry={'bbox':b['bbox'],'decision':'baseline'}; chosen=base
        if bw>=.34*w and bh>=.07*h and region.any():
            vals=lab[region>0]; bins=(vals//12).astype(np.int32); keys=bins[:,0]*484+bins[:,1]*22+bins[:,2]; counts=np.bincount(keys)
            candidates=[]
            for k in np.argsort(counts)[-8:]:
                if counts[k]<.035*len(vals): continue
                center=np.median(vals[keys==k],axis=0); dist=np.max(np.abs(lab-center),axis=2); core=(dist<=15)&(region>0)
                frac=core.sum()/len(vals)
                if not .055<=frac<=.68: continue
                n,_,stats,_=cv2.connectedComponentsWithStats(core.astype('uint8'),8)
                a=stats[1:,4]; useful=a>=max(5,.001*len(vals))
                if useful.sum()<2 or useful.sum()>45 or a.max()>.82*core.sum(): continue
                if a[useful].sum()<.75*core.sum(): continue
                # Prefer strongly chromatic or very bright ink; never infer dark ink on complex images.
                strength=max(abs(center[1]-128),abs(center[2]-128))
                if center[0]<205 and strength<36: continue
                q=max(0,min(3,int(padding))); expanded=cv2.dilate(core.astype('uint8')*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*q+1,2*q+1))) & base
                ratio=np.count_nonzero(expanded)/max(1,np.count_nonzero(base))
                if not .12<=ratio<=.90: continue
                score=counts[k]/len(vals)+.004*strength+.001*center[0]; candidates.append((score,expanded,center))
            if candidates and dec[0]['background_spread'] is not None and dec[0]['background_spread']>24:
                candidates.sort(key=lambda z:z[0],reverse=True); _,chosen,center=candidates[0]; large.append(center); entry['decision']='color_glyph_protection'
        parts.append((b,base,chosen,entry))
    # Dense small-print fields are ambiguous. Preserve them only when several strong
    # large chromatic overlay regions agree on ink color and small regions lack it.
    chrom=[c for c in large if max(abs(c[1]-128),abs(c[2]-128))>=36]
    small=[b for b,_,_,_ in parts if b['bbox'][3]-b['bbox'][1]<.055*h]
    coherent=[]
    for c in chrom:
        group=[d for d in chrom if np.max(np.abs(c-d))<=24]
        if len(group)>len(coherent): coherent=group
    out=np.zeros((h,w),np.uint8); removed=np.zeros_like(out); retained=np.zeros_like(out)
    for b,base,chosen,entry in parts:
        x1,y1,x2,y2=b['bbox']; x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
        if len(coherent)>=3 and len(small)>=12 and y2-y1<.055*h and x2>x1 and y2>y1:
            center=np.median(coherent,axis=0); crop=lab[y1:y2,x1:x2]; match=(np.max(np.abs(crop-center),axis=2)<=24).mean()
            if match<.035: chosen=np.zeros_like(base); entry['decision']='preserve_possible_scene_print'
        removed |= base & ~chosen; retained |= chosen; out|=chosen; entry['before']=int(np.count_nonzero(base)); entry['after']=int(np.count_nonzero(chosen)); log.append(entry)
    baseline,_,_=refine(image,boxes,padding)
    out=np.asarray(baseline) & ~(removed & ~retained)
    if secondary_mask is not None: out|=np.asarray(secondary_mask.convert('L'))
    return Image.fromarray(out),log

def extract_v87(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from lama_v82 import reconstruct
    _,_,boxes,_=extract_v81(image,padding,radius)
    _,kept,_=validate(image,boxes,padding)
    baseline,_,_=refine(image,kept,padding)
    combined,extra,_=recover(image,boxes,baseline,padding)
    secondary=Image.fromarray(np.asarray(combined)&~np.asarray(baseline))
    mask,log=protect(image,kept,secondary,padding)
    # Single-region replay can differ from context-aware baseline. Never expand it.
    mask=Image.fromarray(np.asarray(mask)&np.asarray(combined))
    extract_v87.last_protection=log
    return reconstruct(image,mask),mask,kept+extra,float(np.count_nonzero(np.asarray(mask))/ (image.width*image.height))
