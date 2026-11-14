CREATOR SUITE BENCHMARK LAB - EXPERIMENTAL V3

WHAT'S NEW
- A separate text_detector_v3.py module runs multiple Tesseract OCR passes.
- Passes: sparse text (PSM 11), text blocks (PSM 6), and upscaled sparse text for images below 1400px.
- Overlapping OCR word boxes are deduplicated.
- V2's tighter glyph masks are retained; no large OCR rectangles are erased automatically.
- Original pixels outside the removal mask remain unchanged.
- No OpenRouter calls, extra dependencies, or paid API usage.

INSTALL (BENCHMARK LAB ONLY)
1. Download this ZIP and extract it on your computer.
2. Open the GitHub repository for your Creator Suite Benchmark Lab (NOT your original Background Extraction Lab or main Creator Suite).
3. Back up its existing extraction_engine.py.
4. Upload BOTH extraction_engine.py and text_detector_v3.py into the SAME repository directory as the existing extraction_engine.py (usually repository root). Choose overwrite/commit for extraction_engine.py.
5. Do NOT upload README_V3.txt unless you want to keep it as documentation.
6. Wait for Streamlit to redeploy. If it fails with ModuleNotFoundError: text_detector_v3, confirm the two .py files are next to each other.
7. Upload the same 150-slide dataset, prepare it, select 3 slides, check Save extracted images and detected masks, then Run local benchmark.
8. Download the results CSV, report, and image previews ZIP. Send all three back for analysis.

IMPORTANT
This is an experimental detector. It is NOT production-ready. The first three synthetic slides improved, but there are still false positives and inpainting errors. Do not assume this generalizes to the other 147 slides or to customer uploads. Test on the same three images first.

LOCAL PRE-DEPLOYMENT SMOKE TEST (same three synthetic slides, not Streamlit)
- Slide 001 mask IoU: 0.2224
- Slide 002 mask IoU: 0.7991
- Slide 003 mask IoU: 0.8567
These are measured with mask > 0 and ground truth mask > 0. Streamlit's reported figures should be close if it uses the same evaluation convention.
