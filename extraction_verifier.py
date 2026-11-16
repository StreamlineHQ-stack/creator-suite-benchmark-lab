"""V7.1 independent, ground-truth-free three-way verification. Heuristics are not certification."""
import cv2
import numpy as np
from PIL import Image
import text_detector_v3 as v3
import text_detector_v4 as v4

def verify(original, mask, extracted):
    if original.size != extracted.size or original.size != mask.size:
        return {'status':'FAIL','reasons':['dimension_mismatch'],'residual_boxes':[]}
    src=np.asarray(original.convert('RGB'),dtype=np.int16)
    dst=np.asarray(extracted.convert('RGB'),dtype=np.int16)
    m=np.asarray(mask.convert('L'))>127
    changed=np.any(src!=dst,axis=2)
    outside=int(np.count_nonzero(changed & ~m))
    coverage=float(m.mean())
    # OCR is imperfect: use both detectors; never treat zero OCR boxes as proof of no text.
    detected=[]
    for detector in (v3,v4):
        try: detected.extend(detector.detect_boxes(extracted))
        except Exception: pass
    # Residual candidates only where original also had text-like detections.
    source=[]
    for detector in (v3,v4):
        try: source.extend(detector.detect_boxes(original))
        except Exception: pass
    def overlap(a,b):
        x1,y1,x2,y2=a;u1,v1,u2,v2=b
        return max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    residual=[]
    for box in detected:
        rect=box.get('bbox')
        if not rect or not any(overlap(rect,s.get('bbox',(0,0,0,0)))>0 for s in source): continue
        if rect not in residual: residual.append(rect)
    # Boundary discontinuity proxy: average gradient along the mask boundary.
    boundary=cv2.morphologyEx(m.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    gray=cv2.cvtColor(dst.astype('uint8'),cv2.COLOR_RGB2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    edge=float(np.hypot(gx,gy)[boundary].mean()) if boundary.any() else 0.0
    reasons=[]
    if outside: reasons.append('pixels_changed_outside_mask')
    if coverage>.30: reasons.append('excessive_mask_coverage')
    if residual: reasons.append('possible_remaining_text')
    if edge>110: reasons.append('high_mask_boundary_gradient')
    if not source: reasons.append('no_source_text_detected_cannot_verify')
    status='FAIL' if outside else 'REVIEW' # never certify PASS using these weak heuristics
    return {'status':status,'reasons':reasons,'mask_coverage':round(coverage,5),
            'changed_outside_mask':outside,'boundary_gradient':round(edge,2),
            'source_ocr_boxes':len(source),'residual_ocr_boxes':len(residual),
            'residual_boxes':residual}
