"""Independent, explicitly selected extraction strategies. No AI calls.
No automatic selection until benchmark evidence supports it.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import text_detector_v3 as v3
import text_detector_v4 as v4
from residual_repair_v5 import residual_repair
from targeted_repair_v6 import repair as repair_v6
from extraction_engine import inpaint

STRATEGIES = {
    "v82_local": "V8.2 PaddleOCR + LaMa ONNX (experimental)",
    "v81_local": "V8.1 PaddleOCR polygon detection (experimental)",
    "v77_local": "V7.7 multi-orientation text detection (experimental)",
    "auto_v7": "V7 automatic (experimental; unverified)",
    "auto_v7_1": "V7.1 automatic + three-way verification (experimental)",
    "auto_v7_2": "V7.2 automatic + source-region verification (experimental)",
    "auto_v7_5": "V7.5 automatic + multi-scale residual candidates (experimental)",
    "auto_v7_4": "V7.4 automatic + precision repair candidate (experimental)",
    "auto_v7_3": "V7.3 automatic + targeted repair candidate (experimental)",
    "baseline_v6": "V4 detection + V6 repair (existing baseline)",
    "v3_local": "V3 detection + OpenCV inpainting",
    "v4_local": "V4 detection + OpenCV inpainting",
    "v3_v5": "V3 detection + V5 repair",
    "v4_v6": "V4 detection + V6 repair",
}

def extract(image: Image.Image, strategy: str = "baseline_v6", padding: int = 5, radius: int = 4):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    image = image.convert("RGB")
    if strategy in ("auto_v7","auto_v7_1","auto_v7_2","auto_v7_3","auto_v7_4","auto_v7_5"):
        from intelligent_router_v7 import extract_auto
        cleaned, mask, boxes, coverage, diagnostics = extract_auto(image, padding, radius)
        extract.last_diagnostics = diagnostics
        return cleaned, mask, boxes, coverage
    extract.last_diagnostics = None
    if strategy == "v82_local":
        from lama_v82 import extract_v82
        return extract_v82(image, padding, radius)
    if strategy == "v81_local":
        from text_detector_v81 import extract_v81
        return extract_v81(image, padding, radius)
    if strategy == "v77_local":
        from text_detector_v77 import extract_v77
        return extract_v77(image, padding, radius)
    if strategy == "baseline_v6":
        from extraction_engine import detect_and_remove
        return detect_and_remove(image, padding=padding, radius=radius)
    detector = v3 if strategy in ("v3_local", "v3_v5") else v4
    boxes = detector.detect_boxes(image)
    mask = detector.build_mask(image, boxes, padding=padding)
    initial = inpaint(image, mask, radius=radius)
    if strategy == "v3_v5":
        result, mask, _ = residual_repair(image, initial, mask, radius=radius, padding=padding)
    elif strategy == "v4_v6":
        result, mask, _ = repair_v6(image, initial, mask, radius=radius, padding=padding)
    else:
        result = initial
    if result.size != image.size or mask.size != image.size:
        raise ValueError("Strategy changed image dimensions")
    coverage = float(np.count_nonzero(np.asarray(mask.convert("L")))) / (image.width * image.height)
    return result, mask, boxes, coverage
