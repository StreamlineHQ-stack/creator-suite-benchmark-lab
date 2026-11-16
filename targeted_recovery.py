"""V7.1 conservative targeted retry, only if independently verified as improved."""
import cv2
import numpy as np
from PIL import Image
from extraction_verifier import verify
from extraction_engine import inpaint

def attempt(original, extracted, mask, assessment, radius=4):
    if assessment['status']=='FAIL' or not assessment.get('residual_boxes'):
        return extracted,mask,{'attempted':False,'reason':'no_safe_residual_regions'}
    old=np.asarray(mask.convert('L'))>127
    additions=np.zeros_like(old)
    h,w=old.shape
    for rect in assessment['residual_boxes']:
        x1,y1,x2,y2=map(int,rect)
        x1=max(0,x1-2);y1=max(0,y1-2);x2=min(w,x2+2);y2=min(h,y2+2)
        if x2>x1 and y2>y1: additions[y1:y2,x1:x2]=True
    additions &= ~old
    added=int(additions.sum())
    if added==0 or added>int(old.size*.015):
        return extracted,mask,{'attempted':False,'reason':'empty_or_oversized_addition','added_pixels':added}
    combined=Image.fromarray(((old|additions)*255).astype('uint8'),'L')
    candidate=inpaint(original,combined,radius=radius)
    checked=verify(original,combined,candidate)
    before=assessment.get('residual_ocr_boxes',999)
    after=checked.get('residual_ocr_boxes',999)
    if checked['status']!='FAIL' and after<before and checked['mask_coverage']<=.30:
        return candidate,combined,{'attempted':True,'accepted':True,'before':before,'after':after,'added_pixels':added}
    return extracted,mask,{'attempted':True,'accepted':False,'before':before,'after':after,'added_pixels':added}
