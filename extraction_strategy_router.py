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
    "v893_local": "V8.9.3 Adaptive Reconstruction (experimental)",
    "v892a_local": "V8.9.2a Adaptive Semantic Protection (experimental)",
    "v892_local": "V8.9.2 Semantic Subject Protection (experimental)",
    "v891_local": "V8.9.1 Recognition-free glyph detection (experimental)",
    "v88_local": "V8.8 Automatic Verification & Repair (experimental)",
    "v87_local": "V8.7 Contextual Subject Protection + LaMa (experimental)",
    "v851_local": "V8.5.1 Adaptive Hybrid Masking + LaMa (experimental)",
    "v86_local": "V8.6 Secondary Text Detection + LaMa (experimental)",
    "v85_local": "V8.5 Pixel-level Mask Refinement + LaMa (experimental)",
    "v841_local": "V8.4.1 Advanced Object Validation + LaMa (experimental)",
    "v84_local": "V8.4 Intelligent Mask Validator + LaMa (experimental)",
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
    if strategy == "v893_local":
        from adaptive_reconstruction_v893 import extract_v893
        return extract_v893(image, padding, radius)
    if strategy == "v892a_local":
        from semantic_protection_v892a import extract_v892a
        return extract_v892a(image, padding, radius)
    if strategy == "v892_local":
        from semantic_protection_v892 import extract_v892
        return extract_v892(image, padding, radius)
    if strategy == "v891_local":
        from glyph_detector_v891 import extract_v891
        return extract_v891(image, padding, radius)
    if strategy == "v88_local":
        from verification_repair_v88 import extract_v88
        return extract_v88(image, padding, radius)
    if strategy == "v87_local":
        from contextual_protection_v87 import extract_v87
        return extract_v87(image, padding, radius)
    if strategy == "v86_local":
        from secondary_detector_v86 import extract_v86
        return extract_v86(image, padding, radius)
    if strategy == "v851_local":
        from hybrid_mask_v851 import extract_v851
        return extract_v851(image, padding, radius)
    if strategy == "v85_local":
        from pixel_refiner_v85 import extract_v85
        return extract_v85(image, padding, radius)
    if strategy == "v841_local":
        from mask_validator_v841 import extract_v841
        return extract_v841(image, padding, radius)
    if strategy == "v84_local":
        from mask_validator_v84 import extract_v84
        return extract_v84(image, padding, radius)
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
