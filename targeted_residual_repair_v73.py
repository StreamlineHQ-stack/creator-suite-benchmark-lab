"""V7.3 conservative source-region repair candidate. No benchmark truth or API calls.

A proposal is not evidence of success: returned status always requires review.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from extraction_verifier_v72 import _unique_boxes


def propose(original: Image.Image, extracted: Image.Image, mask: Image.Image,
            radius: int = 4, min_uncovered: float = .35,
            max_new_fraction: float = .025):
    original=original.convert('RGB'); extracted=extracted.convert('RGB'); mask=mask.convert('L')
    if original.size != extracted.size or mask.size != original.size:
        return extracted, mask, {'status':'rejected_dimensions','attempted':False,'accepted':False}
    src=np.asarray(original,dtype=np.uint8)
    old=np.asarray(extracted,dtype=np.uint8)
    base=np.asarray(mask)>127
    h,w=base.shape
    proposal=np.zeros((h,w),dtype=np.uint8)
    candidates=0
    # Use source detections only, not detections of the already reconstructed output.
    for x1,y1,x2,y2 in _unique_boxes(original):
        x1=max(0,min(w,x1)); x2=max(0,min(w,x2))
        y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
        if x2-x1<3 or y2-y1<3: continue
        roi=base[y1:y2,x1:x2]
        if float((~roi).mean()) < min_uncovered: continue
        candidates+=1
        # Local proposal: only the unmasked pixels within the source OCR box.
        # Bounding-box proposals can include foreground; caps and review are mandatory.
        proposal[y1:y2,x1:x2]=255
    if not candidates:
        return extracted,mask,{'status':'no_source_regions_eligible','attempted':False,'accepted':False,'regions':0,'new_pixels':0}
    # Dilate candidate slightly to avoid leaving edge fragments.
    proposal=cv2.dilate(proposal,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    additions=proposal & ~base
    new_pixels=int(additions.sum())
    if new_pixels==0:
        return extracted,mask,{'status':'no_new_pixels','attempted':False,'accepted':False,'regions':candidates,'new_pixels':0}
    if new_pixels > int(base.size*max_new_fraction):
        return extracted,mask,{'status':'rejected_mask_budget','attempted':True,'accepted':False,'regions':candidates,'new_pixels':new_pixels}
    combined=base|additions
    # Preserve existing extracted pixels outside the newly proposed mask. Inpaint
    # only the additions using existing extraction as input, not the original text.
    bgr=cv2.cvtColor(old,cv2.COLOR_RGB2BGR)
    result=cv2.inpaint(bgr,(additions.astype(np.uint8)*255),max(1,int(radius)),cv2.INPAINT_TELEA)
    result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    result[~additions]=old[~additions]
    # Reject degenerate repairs; don't claim successful text removal on heuristics.
    if not np.array_equal(result[~additions],old[~additions]):
        return extracted,mask,{'status':'rejected_outside_mask_changes','attempted':True,'accepted':False}
    return Image.fromarray(result,'RGB'),Image.fromarray(combined.astype(np.uint8)*255,'L'),{
        'status':'candidate_for_review','attempted':True,'accepted':False,
        'regions':candidates,'new_pixels':new_pixels,'new_mask_fraction':round(new_pixels/base.size,6)}
