"""Experimental, backward-compatible OCR text extraction engine.

Public API matches extraction_engine.py: ocr_boxes, make_mask, inpaint,
detect_and_remove. No AI/network calls. Keep the original module as a backup.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


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
            x, y = int(data['left'][i]), int(data['top'][i])
            w, h = int(data['width'][i]), int(data['height'][i])
        except (ValueError, KeyError, TypeError):
            continue
        if confidence >= min_confidence and w > 0 and h > 0:
            boxes.append({'text': txt.strip(), 'confidence': round(confidence, 1), 'bbox': (x, y, x+w, y+h)})
    return boxes


def _glyph_mask(crop: np.ndarray) -> np.ndarray:
    """Choose plausible letter pixels using border colors and local contrast.

    Returns a binary mask inside an OCR word box. Not a semantic detector.
    """
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return np.ones((h, w), dtype=np.uint8) * 255
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    # Estimate background from the outermost pixels of the OCR word crop.
    edge = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
    background = np.median(edge.astype(np.float32), axis=0)
    delta = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Two hypotheses: dark or light text. Select the one less like the border.
    threshold, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = [(gray < threshold).astype(np.uint8), (gray > threshold).astype(np.uint8)]
    scores = []
    for candidate in candidates:
        area = float(candidate.mean())
        border_ratio = float(np.concatenate((candidate[0], candidate[-1], candidate[:,0], candidate[:,-1])).mean())
        scores.append((abs(area - .28) + border_ratio * 1.5, candidate))
    candidate = min(scores, key=lambda x: x[0])[1]
    # Combine foreground-color contrast with Otsu hypothesis.
    cut = max(10.0, float(np.percentile(delta, 55)))
    mask = ((delta > cut) & (candidate > 0)).astype(np.uint8) * 255
    # Don't accept masks that are implausibly empty or almost the entire box.
    coverage = np.count_nonzero(mask) / mask.size
    if coverage < .025 or coverage > .70:
        # Safer fallback: local threshold only, still bounded to the word.
        mask = candidate * 255
    return mask.astype(np.uint8)


def make_mask(image: Image.Image, boxes: list[dict], padding: int = 5,
              extra_rectangles: list[tuple] | None = None) -> Image.Image:
    """Generate tighter per-pixel text masks inside OCR boxes.

    Explicit user rectangles remain fully masked (manual SOS behavior).
    """
    src = np.asarray(_rgb(image))
    height, width = src.shape[:2]
    output = np.zeros((height, width), dtype=np.uint8)
    padding = max(0, int(padding))
    for item in boxes:
        x1, y1, x2, y2 = item['bbox']
        x1, x2 = max(0, int(x1)), min(width, int(x2))
        y1, y2 = max(0, int(y1)), min(height, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        glyph = _glyph_mask(src[y1:y2, x1:x2])
        # Apply small, size-aware expansion to cover anti-aliasing, not entire boxes.
        grow = min(max(1, padding // 3), max(1, min(x2-x1, y2-y1) // 5), 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow+1, 2*grow+1))
        glyph = cv2.dilate(glyph, kernel, iterations=1)
        output[y1:y2, x1:x2] = np.maximum(output[y1:y2, x1:x2], glyph)
    for rect in extra_rectangles or []:
        x1, y1, x2, y2 = (int(v) for v in rect)
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        output[y1:y2, x1:x2] = 255
    return Image.fromarray(output, mode='L')


def inpaint(image: Image.Image, mask: Image.Image, radius: int = 4) -> Image.Image:
    src = cv2.cvtColor(np.asarray(_rgb(image)), cv2.COLOR_RGB2BGR)
    binary = (np.asarray(mask.convert('L')) > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return _rgb(image).copy()
    result = cv2.inpaint(src, binary, float(max(1, radius)), cv2.INPAINT_TELEA)
    # Copy original unmasked pixels exactly; avoid accidental full-image changes.
    result[binary == 0] = src[binary == 0]
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def detect_and_remove(image: Image.Image, padding: int = 5, radius: int = 4,
                      rectangles: list[tuple] | None = None):
    boxes = ocr_boxes(image)
    mask = make_mask(image, boxes, padding, rectangles)
    cleaned = inpaint(image, mask, radius)
    coverage = float(np.count_nonzero(np.asarray(mask))) / (image.width * image.height)
    return cleaned, mask, boxes, coverage
