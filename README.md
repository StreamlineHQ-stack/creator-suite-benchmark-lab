# Creator Suite Phase 1 Benchmark Lab

Standalone Streamlit application for measuring the **existing** local extraction engine against the 150-slide synthetic benchmark. No OpenRouter charges. Does not alter the existing Background Extraction Lab.

## Deployment (no local Python needed)

1. Create a **new private GitHub repository** called `creator-suite-benchmark-lab`.
2. Upload all the contents of this folder to the root of the new repository. Preserve `benchmark/` and `.streamlit/` directories.
3. **Do not upload** `creator_suite_150_slide_benchmark.zip` to GitHub: it is ~128 MB, over GitHub's normal 100 MB single-file limit.
4. At https://share.streamlit.io choose **Create app**, connect the new repository, select branch `main`, and main file `app.py`.
5. After deployment, open the app. Upload `creator_suite_150_slide_benchmark.zip` through its uploader. Click **Prepare benchmark dataset**.
6. Start with **3 slides**, then 10, 25, 50, and finally 150 if resources allow. Click **Run local benchmark** each time.
7. Download the results CSV and report. Send them back for analysis.

## Notes

- Streamlit Community Cloud may restart apps or limit RAM/time. Uploaded data and results are **temporary**; download reports before leaving the app. If a run stops, use a smaller batch.
- The current runner processes the **first N** manifest entries; it does not yet stratify by difficulty. This is a baseline smoke/quality run, not a difficulty-balanced assessment.
- Tesseract is installed by `packages.txt` on Streamlit Community Cloud.
- The synthetic benchmark measures against known clean images and masks. Real-world validation is still required.
- Never upload OpenRouter secrets; this benchmark does not use them.
