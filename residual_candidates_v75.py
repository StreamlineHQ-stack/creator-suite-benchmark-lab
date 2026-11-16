"""V7.5 experimental multi-scale residual stroke proposals (no ground truth).
Candidate-only; benchmarking selects using reference, never production.
"""
import cv2
import numpy as np
from PIL import Image
from extraction_verifier_v72 import _unique_boxes


def propose_candidates(original, extracted, mask, scales=(1, 2, 3), max_fraction=0.006, max_components=48, max_candidates=3):
    original=original.convert('RGB'); extracted=extracted.convert('RGB'); mask=mask.convert('L')
    if original.size!=extracted.size or original.size!=mask.size:
        return [], {'status':'rejected_dimensions'}
    src=np.asarray(original); dst=np.asarray(extracted); base=np.asarray(mask)>127
    h,w=base.shape; boxes=_unique_boxes(original)
    if not boxes: return [], {'status':'no_source_text_boxes'}
    gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    local=cv2.GaussianBlur(gray,(0,0),2.0)
    contrast=cv2.absdiff(gray,local)
    # Only within OCR source boxes, padded modestly; no global contrast search.
    roi=np.zeros((h,w),np.uint8)
    for x1,y1,x2,y2 in boxes:
        x1=max(0,int(x1)-3); y1=max(0,int(y1)-3)
        x2=min(w,int(x2)+3); y2=min(h,int(y2)+3)
        if x2>x1 and y2>y1: roi[y1:y2,x1:x2]=1
    near=cv2.dilate(base.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))>0
    candidates=[]; max_pixels=max(8,int(h*w*max_fraction))
    for threshold in (18,26,36):
        strokes=((contrast>=threshold)&(roi>0)&near&~base).astype('uint8')
        n,labels,stats,_=cv2.connectedComponentsWithStats(strokes,8)
        parts=[]
        for i in range(1,n):
            x,y,ww,hh,area=map(int,stats[i]);
            if area<4 or area>max(200,int(h*w*.001)) or ww>max(40,int(w*.18)) or hh>max(40,int(h*.18)): continue
            parts.append((area,i))
        parts.sort(reverse=True)
        core=np.zeros((h,w),np.uint8)
        for _,i in parts[:max_components]: core[labels==i]=255
        for dilation in scales:
            k=2*int(dilation)+1
            add=cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))>0
            add &= (roi>0)&~base
            count=int(add.sum())
            if count<5 or count>max_pixels: continue
            # Strict pixel preservation outside candidate mask.
            repaired=cv2.inpaint(cv2.cvtColor(dst,cv2.COLOR_RGB2BGR),(add*255).astype('uint8'),3,cv2.INPAINT_TELEA)
            repaired=cv2.cvtColor(repaired,cv2.COLOR_BGR2RGB)
            repaired[~add]=dst[~add]
            new_mask=(base|add).astype('uint8')*255
            name=f't{threshold}_d{dilation}'
            candidates.append({'name':name,'image':Image.fromarray(repaired),'mask':Image.fromarray(new_mask),
                               'new_pixels':count,'components':min(len(parts),max_components)})
            if len(candidates) >= max_candidates:
                return candidates, {'status':'candidates_generated_capped', 'count':len(candidates),
                                    'source_boxes':len(boxes),'max_pixels':max_pixels}
    return candidates, {'status':'candidates_generated' if candidates else 'no_safe_candidates',
                        'count':len(candidates),'source_boxes':len(boxes),'max_pixels':max_pixels}
