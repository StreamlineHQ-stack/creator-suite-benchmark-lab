"""V7 experimental, ground-truth-free candidate selector.

Scores are heuristic rankings, NOT verified extraction quality. Keeps the
original image unchanged and never relies on benchmark ground truth.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import text_detector_v3 as v3
import text_detector_v4 as v4
from extraction_engine import inpaint
from targeted_repair_v6 import repair as repair_v6
from residual_repair_v5 import residual_repair

CANDIDATES = ('v3_local', 'v4_local', 'v3_v5', 'v4_v6')

def _boxes(img, detector):
    return detector.detect_boxes(img)

def _run(img, name, padding, radius, boxes3, boxes4):
    detector = v3 if name.startswith('v3') else v4
    boxes = boxes3 if name.startswith('v3') else boxes4
    mask = detector.build_mask(img, boxes, padding=padding)
    initial = inpaint(img, mask, radius=radius)
    if name == 'v3_v5':
        result, mask, _ = residual_repair(img, initial, mask, radius=radius, padding=padding)
    elif name == 'v4_v6':
        result, mask, _ = repair_v6(img, initial, mask, radius=radius, padding=padding)
    else:
        result = initial
    return result, mask, boxes

def _score(original, result, mask, boxes, original_count):
    if result.size != original.size or mask.size != original.size:
        return -1e9, {'error': 'dimension_mismatch'}
    a=np.asarray(original.convert('RGB'), dtype=np.int16)
    b=np.asarray(result.convert('RGB'), dtype=np.int16)
    m=np.asarray(mask.convert('L')) > 0
    coverage=float(m.mean())
    # Strict invariant: candidate must preserve every unmasked pixel exactly.
    changed_outside=int(np.count_nonzero(np.any(a[~m]!=b[~m], axis=1))) if (~m).any() else 0
    if changed_outside: return -1e9, {'error':'outside_mask_modified', 'changed_pixels':changed_outside}
    try:
        residual=len(v3.detect_boxes(result,min_confidence=50))
    except Exception:
        residual=original_count
    # Overly broad masks destroy content. Tiny masks with remaining text also fail.
    coverage_penalty=max(0.0,coverage-.12)*140 + max(0.0,coverage-.30)*250
    score=-(residual*5.0) - coverage_penalty - (coverage*4.0)
    if original_count and coverage < .0002: score -= 20
    return score, {'residual_ocr_boxes':residual,'mask_coverage':round(coverage,5),
                   'changed_outside_mask':changed_outside}

def extract_auto(image:Image.Image, padding:int=5, radius:int=4):
    image=image.convert('RGB')
    boxes3=_boxes(image,v3); boxes4=_boxes(image,v4)
    original_count=max(len(boxes3),len(boxes4))
    outcomes=[]
    for name in CANDIDATES:
        try:
            result,mask,boxes=_run(image,name,padding,radius,boxes3,boxes4)
            score,details=_score(image,result,mask,boxes,original_count)
            outcomes.append((score,name,result,mask,boxes,details))
        except Exception as e:
            outcomes.append((-1e9,name,None,None,[],{'error':f'{type(e).__name__}: {e}'}))
    valid=[x for x in outcomes if x[0]>-1e8]
    if not valid: raise RuntimeError('All V7 candidates failed safety checks')
    # Deterministic tie break in CANDIDATES order.
    winner=max(valid,key=lambda x:x[0])
    score,name,result,mask,boxes,details=winner
    coverage=details['mask_coverage']
    # This is a conservative REVIEW flag, not a verified pass/fail classifier.
    review=bool(details['residual_ocr_boxes'] or coverage>.15 or not original_count)
    diagnostics={'selected_strategy':name,'manual_review_required':review,
                 'verification_status':'unverified_candidate',
                 'candidate_scores':{n:round(float(s),3) if s>-1e8 else None for s,n,*_ in outcomes},
                 'candidate_diagnostics':{n:d for _,n,_,_,_,d in outcomes}}
    return result,mask,boxes,coverage,diagnostics
