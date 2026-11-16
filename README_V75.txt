V7.5: MULTI-SCALE RESIDUAL CANDIDATES — BENCHMARK ONLY

Upload these files to the EXISTING creator-suite-benchmark-lab GitHub repository:
ADD root/residual_candidates_v75.py
REPLACE root/extraction_strategy_router.py
REPLACE root/app.py
REPLACE root/benchmark/benchmark_runner.py
ADD root/README_V75.txt

Keep all existing modules including V7.4 and both verifier files.
Do not modify Background Extraction Lab or production Creator Suite.

After Streamlit redeploys, upload the benchmark dataset if necessary, choose
3 slides, select V7.5 automatic + multi-scale residual candidates (experimental),
check Save extracted images and detected masks, and click Run local benchmark.
Download Results CSV and Image Previews ZIP and share them for inspection.

V7.5 uses OCR source boxes, local contrast, and three dilation scales with
conservative budgets. It compares multiple candidates against synthetic clean
ground truth ONLY in the benchmark. No candidate is auto-applied.
The candidate with the lowest benchmark reference MAE is shown for analysis,
even if none improves the baseline; inspect v75_best_reference_delta:
negative = improvement; zero = no change; positive = worse.

This is an experimental diagnostic, NOT proof of reliable production text removal.
