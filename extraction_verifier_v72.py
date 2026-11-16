"""V7.2 conservative, ground-truth-free verification.

A REVIEW is not a pass. OCR omissions and reconstruction fidelity remain
unresolved without human inspection or benchmark ground truth.
"""
import cv2
import numpy as np
from extraction_verifier import verify as verify_v71
import text_detector_v3 as v3
import text_detector_v4 as v4


def _unique_boxes(image):
    boxes=[]
    for detector in (v3,v4):
        try:
            boxes.extend(detector.detect_boxes(image))
        except Exception:
            pass
    seen=set(); output=[]
    for b in boxes:
        rect=b.get('bbox')
        if not rect or len(rect)!=4: continue
        rect=tuple(map(int,rect))
        if rect in seen: continue
        seen.add(rect);output.append(rect)
    return output


def verify(original, mask, extracted):
    base=verify_v71(original,mask,extracted)
    result=dict(base)
    result['verification_version']='v7.2'
    result['source_regions_inspected']=0
    result['source_regions_uncovered']=0
    result['source_regions_partially_uncovered']=0
    result['unchanged_source_regions']=0
    result['uncovered_source_boxes']=[]
    if original.size != extracted.size or original.size != mask.size:
        return result
    src=np.asarray(original.convert('RGB'),dtype=np.uint8)
    dst=np.asarray(extracted.convert('RGB'),dtype=np.uint8)
    m=np.asarray(mask.convert('L'))>127
    changed=np.any(src!=dst,axis=2)
    h,w=m.shape
    boxes=_unique_boxes(original)
    suspicious=[]
    for x1,y1,x2,y2 in boxes:
        x1=max(0,min(w,x1));x2=max(0,min(w,x2));y1=max(0,min(h,y1));y2=max(0,min(h,y2))
        if x2<=x1 or y2<=y1:continue
        area=(x2-x1)*(y2-y1)
        if area<9:continue
        roi_mask=m[y1:y2,x1:x2]
        roi_changed=changed[y1:y2,x1:x2]
        coverage=float(roi_mask.mean())
        altered=float(roi_changed.mean())
        result['source_regions_inspected']+=1
        if coverage<0.08:
            result['source_regions_uncovered']+=1
            suspicious.append((x1,y1,x2,y2))
        elif coverage<0.35:
            result['source_regions_partially_uncovered']+=1
            suspicious.append((x1,y1,x2,y2))
        if altered<0.025:
            result['unchanged_source_regions']+=1
    result['uncovered_source_boxes']=suspicious
    reasons=list(result.get('reasons',[]))
    if result['source_regions_uncovered']:
        reasons.append('source_text_regions_not_masked')
    if result['source_regions_partially_uncovered']:
        reasons.append('source_text_regions_insufficiently_masked')
    if result['unchanged_source_regions']:
        reasons.append('source_text_regions_essentially_unchanged')
    if not boxes:
        reasons.append('no_source_ocr_boxes_unverifiable')
    # Existing FAIL invariants remain authoritative. Never certify PASS.
    result['status']='FAIL' if base['status']=='FAIL' else 'REVIEW'
    result['reasons']=list(dict.fromkeys(reasons))
    return result
