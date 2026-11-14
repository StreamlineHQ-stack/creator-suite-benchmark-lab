"""V6 benchmark adapter. Same public API as V3-V5; separate V6 repair module."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from text_detector_v4 import detect_boxes, build_mask
from targeted_repair_v6 import repair


def ocr_boxes(image,min_confidence=25):
    return detect_boxes(image,min_confidence)


def make_mask(image,boxes,padding=5,extra_rectangles=None):
    return build_mask(image,boxes,padding,extra_rectangles)


def inpaint(image,mask,radius=4):
    original=image.convert('RGB')
    src=cv2.cvtColor(np.asarray(original),cv2.COLOR_RGB2BGR)
    binary=(np.asarray(mask.convert('L'))>0).astype(np.uint8)*255
    if not binary.any():return original.copy()
    output=cv2.inpaint(src,binary,float(max(1,radius)),cv2.INPAINT_TELEA)
    output[binary==0]=src[binary==0]
    return Image.fromarray(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))


def detect_and_remove(image,padding=5,radius=4,rectangles=None):
    original=image.convert('RGB')
    boxes=ocr_boxes(original)
    mask=make_mask(original,boxes,padding,rectangles)
    initial=inpaint(original,mask,radius)
    cleaned,mask,diagnostics=repair(original,initial,mask,radius,padding,rectangles)
    coverage=float(np.count_nonzero(np.asarray(mask)))/(original.width*original.height)
    return cleaned,mask,boxes,coverage
