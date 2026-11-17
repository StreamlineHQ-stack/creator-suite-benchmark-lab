"""Experimental conservative color-based glyph segmentation within IMV polygons."""
import cv2
import numpy as np
from PIL import Image


def refine_region(rgb, polygon, padding):
    x,y,w,h=cv2.boundingRect(polygon)
    crop=rgb[y:y+h,x:x+w]; region=np.zeros((h,w),np.uint8)
    cv2.fillPoly(region,[polygon-[x,y]],255); valid=region>0
    if min(w,h)<12 or valid.sum()<100: return region,'fallback_small'
    # Compare both polarities of luminance and chromatic channels, only inside polygon.
    lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)
    inner=cv2.erode(region,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    rim=valid & ~inner
    candidates=[]
    for channel in cv2.split(lab):
        values=channel[valid]; low,high=np.percentile(values,[10,90])
        if high-low<22: continue
        threshold,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for fg in (channel>threshold,channel<=threshold):
            fg=fg & valid; fraction=fg.sum()/valid.sum()
            if not .10<=fraction<=.58: continue
            rim_fraction=fg[rim].mean() if rim.any() else 1
            if rim_fraction>.30 or rim_fraction>fraction*.85: continue
            # Reject noisy masks and dominant solid blobs; require multiple glyph-like components.
            n,labels,stats,_=cv2.connectedComponentsWithStats(fg.astype(np.uint8),8)
            areas=stats[1:,cv2.CC_STAT_AREA]; useful=areas>=max(3,valid.sum()*.001)
            if useful.sum()<2 or useful.sum()>40: continue
            if areas.max()/max(1,fg.sum())>.79 or areas[useful].sum()/fg.sum()<.90: continue
            # Require coherent color and strong separation from remaining pixels.
            delta=abs(float(channel[fg].mean())-float(channel[valid & ~fg].mean()))
            if delta<32: continue
            score=delta/255 + .35*(1-rim_fraction) - .20*abs(fraction-.30)
            candidates.append((score,fg))
    if not candidates: return region,'fallback_uncertain'
    candidates.sort(key=lambda c:c[0],reverse=True)
    best=candidates[0][1]
    # If similarly plausible alternatives disagree strongly, preserve original mask.
    for score,other in candidates[1:]:
        if score<candidates[0][0]-.045: break
        if (best & other).sum()/max(1,(best|other).sum())<.70: return region,'fallback_ambiguous'
    return best.astype(np.uint8)*255,'refined_color'


def refine(image,kept,padding=5):
    rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]; mask=np.zeros((h,w),np.uint8); baseline=mask.copy(); decisions=[]
    p=max(0,min(12,int(padding)))
    for box in kept:
        poly=np.asarray(box['polygon'],np.int32).copy(); poly[:,0]=np.clip(poly[:,0],0,w-1); poly[:,1]=np.clip(poly[:,1],0,h-1)
        cv2.fillPoly(baseline,[poly],255)
        x,y,bw,bh=cv2.boundingRect(poly); local,reason=refine_region(rgb,poly,p)
        # Dilate independently, so uncertainty keeps exactly the old regional padding.
        q=min(p, max(1,min(3,round(min(bw,bh)*.045)))) if reason=='refined_color' else p
        part=np.zeros_like(mask); part[y:y+bh,x:x+bw]=local
        if q: part=cv2.dilate(part,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*q+1,2*q+1)))
        mask|=part; decisions.append(dict(bbox=box['bbox'],decision=reason,masked_pixels=int(np.count_nonzero(part))))
    if p: baseline=cv2.dilate(baseline,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*p+1,2*p+1)))
    mask &= baseline
    return Image.fromarray(mask),decisions, float(np.count_nonzero(baseline)/baseline.size)


def extract_v85(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from mask_validator_v841 import validate
    from lama_v82 import reconstruct
    _,_,boxes,_=extract_v81(image,padding,radius)
    _,kept,imv=validate(image,boxes,padding)
    mask,decisions,baseline=refine(image,kept,padding)
    extract_v85.last_decisions=imv
    extract_v85.last_refinement=decisions
    extract_v85.baseline_coverage=baseline
    return reconstruct(image,mask),mask,kept,float(np.count_nonzero(np.asarray(mask))/(image.width*image.height))
