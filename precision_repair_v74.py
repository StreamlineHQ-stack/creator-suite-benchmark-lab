"""Experimental, ground-truth-free precision repair proposals.

Produces small localized candidates; never automatically certifies success.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from extraction_verifier_v72 import _unique_boxes


def propose(original: Image.Image, extracted: Image.Image, mask: Image.Image,
            radius: int = 3, max_new_fraction: float = .012,
            max_region_fraction: float = .003, max_regions: int = 8):
    original, extracted, mask = original.convert('RGB'), extracted.convert('RGB'), mask.convert('L')
    if original.size != extracted.size or mask.size != original.size:
        return extracted, mask, {'status':'rejected_dimensions','accepted':False,'regions':0,'new_pixels':0}
    src=np.asarray(original); old=np.asarray(extracted); base=np.asarray(mask)>127
    h,w=base.shape; image_area=h*w
    gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    # Thin, high-contrast connected strokes only; avoid filling entire OCR boxes.
    smooth=cv2.GaussianBlur(gray,(0,0),2.2)
    contrast=cv2.absdiff(gray,smooth)
    strokes=(contrast>=24).astype(np.uint8)
    # Protect pixels that have already been repaired; add only where baseline mask is absent.
    strokes[base]=0
    boxes=_unique_boxes(original)
    components=[]
    for x1,y1,x2,y2 in boxes:
        x1=max(0,min(w,int(x1)));x2=max(0,min(w,int(x2)))
        y1=max(0,min(h,int(y1)));y2=max(0,min(h,int(y2)))
        if x2-x1<4 or y2-y1<4: continue
        local=strokes[y1:y2,x1:x2]
        n, labels, stats, _=cv2.connectedComponentsWithStats(local,8)
        for i in range(1,n):
            xx,yy,ww,hh,area=map(int,stats[i])
            if area<4 or area>max(16,int(image_area*max_region_fraction)): continue
            if ww>max(16,int(w*.12)) or hh>max(16,int(h*.12)): continue
            # Candidate stroke must be near an existing text mask; avoid unrelated distant objects.
            cx=x1+xx+ww//2;cy=y1+yy+hh//2
            xlo=max(0,cx-14);xhi=min(w,cx+15);ylo=max(0,cy-14);yhi=min(h,cy+15)
            if not base[ylo:yhi,xlo:xhi].any(): continue
            components.append((area,x1+xx,y1+yy,ww,hh,labels[yy:yy+hh,xx:xx+ww]==i))
    # Deterministic, smallest first. Limit proposals by region and global budgets.
    components.sort(key=lambda x:(x[0],x[2],x[1]))
    addition=np.zeros((h,w),np.uint8)
    chosen=0
    per_limit=max(8,int(image_area*max_region_fraction))
    total_limit=max(1,int(image_area*max_new_fraction))
    for area,x,y,ww,hh,comp in components:
        if chosen>=max_regions:break
        region=np.zeros((hh,ww),np.uint8);region[comp]=255
        region=cv2.dilate(region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        region[base[y:y+hh,x:x+ww]]=0
        count=int((region>0).sum())
        if count<3 or count>per_limit or int((addition>0).sum())+count>total_limit:continue
        addition[y:y+hh,x:x+ww]|=region
        chosen+=1
    added=(addition>0)&~base
    new_pixels=int(added.sum())
    if not new_pixels:
        return extracted,mask,{'status':'no_precise_candidates','accepted':False,'regions':0,'new_pixels':0,'source_boxes':len(boxes)}
    if new_pixels>total_limit:
        return extracted,mask,{'status':'rejected_mask_budget','accepted':False,'regions':chosen,'new_pixels':new_pixels}
    # Inpaint from the baseline output; preserve everything outside the selected pixels.
    repaired=cv2.inpaint(cv2.cvtColor(old,cv2.COLOR_RGB2BGR),addition,max(1,int(radius)),cv2.INPAINT_TELEA)
    repaired=cv2.cvtColor(repaired,cv2.COLOR_BGR2RGB)
    repaired[~added]=old[~added]
    if not np.array_equal(repaired[~added],old[~added]):
        return extracted,mask,{'status':'rejected_outside_changes','accepted':False,'regions':chosen,'new_pixels':new_pixels}
    return Image.fromarray(repaired),Image.fromarray(((base|added)*255).astype(np.uint8)),{
        'status':'candidate_for_review','accepted':False,'regions':chosen,'new_pixels':new_pixels,
        'new_mask_fraction':round(new_pixels/image_area,6),'source_boxes':len(boxes)}
