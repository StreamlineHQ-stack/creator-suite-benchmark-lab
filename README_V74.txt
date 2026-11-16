Creator Suite Benchmark Lab — V7.4 precision repair experiment

This patch is based on the V7.3 patch plus the user's earlier V7.2 repository.
The current deployed V7.3 repository has NOT been fetched directly.

GitHub changes:
ADD precision_repair_v74.py at repository root.
REPLACE extraction_strategy_router.py at repository root.
REPLACE app.py at repository root.
REPLACE benchmark/benchmark_runner.py inside benchmark/.
ADD README_V74.txt at repository root.
KEEP every other file, including both verifiers and V7.3 repair module.
Do not update the separate Background Extraction Lab or main Creator Suite.

In Streamlit select 3 slides, strategy 'V7.4 automatic + precision repair candidate (experimental)',
and check 'Save extracted images and detected masks'. Run and download CSV and previews ZIP.

V7.4 detects small high-contrast components within OCR boxes, requires proximity to existing
text masks, caps the number and area of candidate regions, and inpaints only those pixels.
This is an experimental conservative proposal, NOT a production text segmentation system.
The detector can mistake edges of products or decorative elements for letters; inspect
candidate previews and benchmark reference MAE. In particular, a very small candidate
may not meaningfully fix Slide 002. It is NOT auto accepted.

Negative v74_reference_mae_delta = closer to synthetic clean reference (good).
Positive delta = worse. Neither certifies visual quality on real slides.

The smoke test used marked-original preview images (which include a colored mask overlay),
not the pristine source images. Real benchmark run is required.
