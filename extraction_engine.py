"""Local text masking and conservative OpenCV reconstruction."""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw


def _rgb(image: Image.Image) -> Image.Image:
    return image.convert('RGB')


def ocr_boxes(image: Image.Image, min_confidence: int = 25) -> list[dict]:
    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(_rgb(image), config='--psm 11', output_type=Output.DICT)
    except Exception as exc:
        raise RuntimeError(f'OCR unavailable: {exc}. Check packages.txt / Tesseract installation.') from exc
    boxes = []
    for i, txt in enumerate(data.get('text', [])):
        if not (txt or '').strip():
            continue
        try:
            confidence = float(data['conf'][i])
        except (ValueError, KeyError, TypeError):
            continue
        if confidence < min_confidence:
            continue
        x, y = int(data['left'][i]), int(data['top'][i])
        w, h = int(data['width'][i]), int(data['height'][i])
        if w > 0 and h > 0:
            boxes.append({'text': txt.strip(), 'confidence': round(confidence, 1), 'bbox': (x, y, x+w, y+h)})
    return boxes


def make_mask(image: Image.Image, boxes: list[dict], padding: int = 5, extra_rectangles: list[tuple] | None = None) -> Image.Image:
    mask = Image.new('L', image.size, 0)
    d = ImageDraw.Draw(mask)
    for item in boxes:
        x1, y1, x2, y2 = item['bbox']
        d.rectangle((max(0,x1-padding),max(0,y1-padding),min(image.width,x2+padding),min(image.height,y2+padding)),fill=255)
    for rect in extra_rectangles or []:
        x1,y1,x2,y2=rect
        d.rectangle((max(0,x1),max(0,y1),min(image.width,x2),min(image.height,y2)),fill=255)
    return mask


def inpaint(image: Image.Image, mask: Image.Image, radius: int = 4) -> Image.Image:
    src = cv2.cvtColor(np.array(_rgb(image)), cv2.COLOR_RGB2BGR)
    binary = np.array(mask.convert('L'))
    if not np.any(binary):
        return _rgb(image).copy()
    result = cv2.inpaint(src, binary, float(radius), cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def detect_and_remove(image: Image.Image, padding: int = 5, radius: int = 4, rectangles: list[tuple] | None = None):
    boxes = ocr_boxes(image)
    mask = make_mask(image, boxes, padding, rectangles)
    cleaned = inpaint(image, mask, radius)
    coverage = float(np.count_nonzero(np.array(mask))) / (image.width*image.height)
    return cleaned, mask, boxes, coverage
