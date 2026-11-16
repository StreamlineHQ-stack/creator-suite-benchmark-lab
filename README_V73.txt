Creator Suite Benchmark Lab — V7.3 experimental patch

Upload these four files to the existing GitHub benchmark repository:
  root/targeted_residual_repair_v73.py   (NEW)
  root/extraction_strategy_router.py      (REPLACE)
  root/app.py                             (REPLACE)
  benchmark/benchmark_runner.py           (REPLACE)

Keep extraction_verifier.py and extraction_verifier_v72.py unchanged.
Keep the existing engines and the main Creator Suite untouched.

After Streamlit redeploys, upload your 150-slide dataset ZIP, choose 3 slides,
choose V7.3 automatic + targeted repair candidate (experimental), check
Save extracted images and detected masks, then Run local benchmark.

Download results.csv and image previews ZIP. Upload them to ChatGPT for review.

IMPORTANT: V7.3 is a PROPOSAL-ONLY test. It DOES NOT automatically apply
candidate repairs to the extracted result. Existing V7.2 output is retained.
If a candidate is created, the ZIP contains <id>_v73_candidate.png and
<id>_v73_candidate_mask.png. CSV contains paired baseline/candidate metrics.
Synthetic clean references are used for benchmarking only; they are never
passed to the production repair function. Candidate quality is unproven.

Safety limits: added mask <=2.5% of total image, image dimensions unchanged,
no changes outside new candidate mask, no API calls.

A rejected candidate is an expected safe outcome. OCR misses and foreground
preservation remain open limitations. Do not ship this as a customer feature.
