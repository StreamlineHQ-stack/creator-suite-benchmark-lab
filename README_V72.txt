V7.2 BENCHMARK PATCH — INSTALLATION

This ZIP contains only files to add or replace in your existing GitHub benchmark repository.

1. Open your creator-suite-benchmark-lab GitHub repository.
2. Add extraction_verifier_v72.py at repository root (NEW).
3. Replace extraction_strategy_router.py at repository root.
4. Replace benchmark/benchmark_runner.py INSIDE the benchmark folder.
5. Replace app.py at repository root.
6. Wait for Streamlit to redeploy. Do not change Background Extraction Lab.
7. Upload the 150-slide dataset as before. Choose 3 slides and strategy:
   V7.2 automatic + source-region verification (experimental).
8. Keep Save extracted images and detected masks checked. Run benchmark.
9. Download results CSV and image previews ZIP. Share both for review.

IMPORTANT: V7.2 is a verification improvement, not a new reconstruction model.
The verification cannot prove the output is text-free or visually accurate.
A source text region that is incompletely masked is flagged for review even
when output OCR sees nothing. These are conservative heuristic warnings.
False positives and false negatives remain possible.
The automatic selector remains V7; this patch does not change its engines.
V7.2 retains V7.1 targeted recovery behavior.
