"""Deterministic metrics for paired synthetic text-removal ground truth."""
import numpy as np
from PIL import Image

def measure(pred_mask: Image.Image, true_mask: Image.Image, output: Image.Image, clean: Image.Image, original: Image.Image) -> dict:
    pm=np.asarray(pred_mask.convert('L'))>127
    tm=np.asarray(true_mask.convert('L'))>127
    tp=int((pm & tm).sum()); fp=int((pm & ~tm).sum()); fn=int((~pm & tm).sum())
    precision=tp/(tp+fp) if tp+fp else (1.0 if not tm.any() else 0.0)
    recall=tp/(tp+fn) if tp+fn else 1.0
    iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0
    out=np.asarray(output.convert('RGB'),dtype=np.float32)
    target=np.asarray(clean.convert('RGB'),dtype=np.float32)
    src=np.asarray(original.convert('RGB'),dtype=np.float32)
    if out.shape!=target.shape: raise ValueError('Dimensions differ from reference')
    err=np.abs(out-target).mean(axis=2)
    before=np.abs(src-target).mean(axis=2)
    inside=float(err[tm].mean()) if tm.any() else 0.0
    outside=float(err[~tm].mean()) if (~tm).any() else 0.0
    baseline_inside=float(before[tm].mean()) if tm.any() else 0.0
    return dict(mask_precision=round(precision,4),mask_recall=round(recall,4),mask_iou=round(iou,4),
        missed_text_pixels=fn, false_positive_pixels=fp,mask_coverage=round(float(pm.mean()),4),
        reference_mae=round(float(err.mean()),3),text_region_mae=round(inside,3),
        nontext_region_mae=round(outside,3),unprocessed_text_region_mae=round(baseline_inside,3),
        reconstruction_improvement=round((baseline_inside-inside)/baseline_inside,4) if baseline_inside>0 else None,
        output_size_correct=True)
