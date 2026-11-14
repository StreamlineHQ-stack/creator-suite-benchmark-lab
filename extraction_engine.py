"""V3 adapter. Public API unchanged; text detection is isolated in text_detector_v3.py."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from text_detector_v3 import detect_boxes,build_mask

def _rgb(image):return image.convert('RGB')

def ocr_boxes(image,min_confidence=25):return detect_boxes(image,min_confidence)

def make_mask(image,boxes,padding=5,extra_rectangles=None):return build_mask(image,boxes,padding,extra_rectangles)

def inpaint(image,mask,radius=4):
    src=cv2.cvtColor(np.asarray(_rgb(image)),cv2.COLOR_RGB2BGR)
    binary=(np.asarray(mask.convert('L'))>0).astype(np.uint8)*255
    if not np.any(binary):return _rgb(image).copy()
    output=cv2.inpaint(src,binary,float(max(1,radius)),cv2.INPAINT_TELEA)
    output[binary==0]=src[binary==0]
    return Image.fromarray(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))

def detect_and_remove(image,padding=5,radius=4,rectangles=None):
    boxes=ocr_boxes(image)
    mask=make_mask(image,boxes,padding,rectangles)
    cleaned=inpaint(image,mask,radius)
    coverage=float(np.count_nonzero(np.asarray(mask)))/(image.width*image.height)
    return cleaned,mask,boxes,coverage
