"""Adaptive region/pixel masking. Conservative heuristic, not a quality verifier."""
import cv2
import numpy as np
from PIL import Image
from pixel_refiner_v85 import refine_region


def refine(image, kept, padding=5):
    rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
    mask=np.zeros((h,w),np.uint8); baseline=mask.copy(); decisions=[]
    p=max(0,min(12,int(padding)))
    kernel=lambda q: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*q+1,2*q+1))
    for box in kept:
        poly=np.asarray(box['polygon'],np.int32).copy()
        poly[:,0]=np.clip(poly[:,0],0,w-1); poly[:,1]=np.clip(poly[:,1],0,h-1)
        region=np.zeros_like(mask); cv2.fillPoly(region,[poly],255)
        regional=cv2.dilate(region,kernel(p)) if p else region
        baseline|=regional
        x,y,bw,bh=cv2.boundingRect(poly)
        # Sample outside the OCR region, excluding other accepted text regions.
        q=max(4,min(18,round(min(bw,bh)*.22)))
        surround=cv2.dilate(region,kernel(q))>0
        surround &= region==0
        for other in kept:
            if other is box: continue
            exclude=np.zeros_like(region); cv2.fillPoly(exclude,[np.asarray(other['polygon'],np.int32)],255)
            surround &= exclude==0
        sample=rgb[surround].astype(np.float32)
        spread=float(np.max(np.percentile(sample,90,axis=0)-np.percentile(sample,10,axis=0))) if len(sample)>=50 else None
        local,reason=refine_region(rgb,poly,p)
        decision='regional_uncertain'; part=regional
        if spread is not None and spread<=24:
            decision='regional_simple'
        elif spread is not None and reason=='refined_color' and p>0:
            # Recover antialiasing and thin shadows with full permitted dilation.
            core=np.zeros_like(mask); core[y:y+bh,x:x+bw]=local
            expanded=cv2.dilate(core,kernel(p)) & regional
            ratio=np.count_nonzero(expanded)/max(1,np.count_nonzero(regional))
            if ratio<.88:
                part=expanded; decision='pixel_expanded'
            else: decision='regional_small_saving'
        mask|=part
        decisions.append(dict(bbox=box['bbox'],decision=decision,background_spread=spread,pixel_evidence=reason,masked_pixels=int(np.count_nonzero(part))))
    return Image.fromarray(mask),decisions,float(np.count_nonzero(baseline)/baseline.size)


def extract_v851(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    from mask_validator_v841 import validate
    from lama_v82 import reconstruct
    _,_,boxes,_=extract_v81(image,padding,radius)
    _,kept,imv=validate(image,boxes,padding)
    mask,decisions,baseline=refine(image,kept,padding)
    extract_v851.last_decisions=imv
    extract_v851.last_refinement=decisions
    extract_v851.baseline_coverage=baseline
    return reconstruct(image,mask),mask,kept,float(np.count_nonzero(np.asarray(mask))/(image.width*image.height))
