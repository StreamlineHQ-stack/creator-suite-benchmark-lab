"""Streamlit UI for the independent Phase 1 local-engine benchmark."""
import csv
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from benchmark.benchmark_runner import run
from benchmark.real_world_runner import run_real
from extraction_strategy_router import STRATEGIES

st.set_page_config(page_title="Creator Suite · Benchmark Lab", layout="wide")
st.title("Creator Suite · Phase 1 Benchmark Lab")
st.caption("Independent benchmark of the current local OCR/OpenCV extractor. No OpenRouter calls, no Creator Suite changes.")
st.info("Upload the synthetic 150-slide ZIP or the private 30-slide real-world ZIP. Never commit benchmark assets to GitHub. Use a private deployment for internal materials.")

def workspace():
    if "private_workspace" not in st.session_state:
        st.session_state["private_workspace"]=tempfile.mkdtemp(prefix="cs_benchmark_")
    return Path(st.session_state["private_workspace"])

work=workspace()
mode=st.radio("Benchmark mode",["Synthetic (paired)","Real-world (internal visual review)"])
real=mode.startswith("Real-world")
upload=st.file_uploader("Upload benchmark ZIP",type="zip",key=mode)
if upload is not None:
    import hashlib
    signature=f"{mode}:{hashlib.sha256(upload.getbuffer()).hexdigest()}"
    if st.session_state.get("dataset_signature") != signature:
        st.session_state["dataset_signature"]=signature
        st.session_state.pop("dataset_path",None)
        st.session_state.pop("results_path",None)
    if "dataset_path" not in st.session_state:
        if st.button("1 · Prepare benchmark dataset",type="primary"):
            try:
                root=work/"dataset"
                if root.exists(): shutil.rmtree(root)
                root.mkdir(parents=True)
                zip_file=work/"uploaded_dataset.zip"
                upload.seek(0)
                with open(zip_file,"wb") as f:
                    while True:
                        chunk=upload.read(1024*1024)
                        if not chunk: break
                        f.write(chunk)
                with zipfile.ZipFile(zip_file) as z:
                    members=z.infolist()
                    if sum(x.file_size for x in members)>900*1024*1024:
                        raise ValueError("Archive uncompressed size exceeds the 900 MB safety limit")
                    for member in members:
                        p=Path(member.filename)
                        if p.is_absolute() or ".." in p.parts:
                            raise ValueError("Archive contains an unsafe path")
                    z.extractall(root)
                candidates=[p.parent for p in root.rglob("manifest.csv") if (p.parent/"originals").is_dir() and (p.parent/"text_masks").is_dir() and (p.parent/"clean_ground_truth").is_dir()]
                if real:
                    candidates=[p.parent for p in root.rglob("manifest.csv") if (p.parent/"originals").is_dir() and "path" in p.read_text(encoding="utf-8-sig").splitlines()[0].split(",")]
                if not candidates: raise ValueError("Could not find a dataset matching the selected benchmark mode")
                st.session_state["dataset_path"]=str(candidates[0])
                st.success("Dataset ready. You can now run the benchmark.")
            except Exception as e:
                st.error(f"Dataset preparation failed: {e}")
    if st.session_state.get("dataset_path"):
        dataset=Path(st.session_state["dataset_path"])
        with (dataset/"manifest.csv").open(encoding="utf-8-sig",newline="") as f: count=sum(1 for _ in csv.DictReader(f))
        st.success(f"Dataset loaded: {count} slides")
        category=st.selectbox("Category",["All","A","B","C","D","E"]) if real else "All"
        n=st.selectbox("2 · How many slides should we test?",[3,10,30] if real else [3,10,25,50,150],index=0)
        strategy=st.selectbox("Extraction strategy",[k for k in STRATEGIES if not real or not k.startswith("auto_")],format_func=lambda key: STRATEGIES[key],index=0)
        previews=st.checkbox("Save extracted images and detected masks",value=real,help="Produces larger downloads and uses more storage")
        st.warning("Run 3 slides first. Then 10 or 25. A full 150-slide run may exceed Streamlit Community Cloud's resources or time limits. If it stops, try a smaller run.")
        if st.button("3 · Run local benchmark",type="primary"):
            st.session_state.pop("results_path",None)
            target=work/f"run_{mode.split()[0]}_{category}_{n}_{int(previews)}_{strategy}"
            if target.exists(): shutil.rmtree(target)
            with st.spinner(f"Processing {n} slides. Please keep this tab open..."):
                try:
                    rows=run_real(dataset,target,limit=n,save_previews=previews,strategy=strategy,category=category) if real else run(dataset,target,limit=n,save_previews=previews,strategy=strategy)
                    st.session_state["results_path"]=str(target)
                    st.success(f"Completed: {sum(r.get('status')=='ok' for r in rows)}/{len(rows)} processed without runtime errors. This is NOT a quality pass rate.")
                except Exception as e: st.error(f"Benchmark stopped: {type(e).__name__}: {e}")
        result=st.session_state.get("results_path")
        if result and Path(result,"results.csv").exists():
            target=Path(result)
            st.subheader("Results")
            df=pd.read_csv(target/"results.csv",dtype={"id":str})
            st.dataframe(df,use_container_width=True)
            if real and (target/"previews").exists():
                st.caption("Manual visual review only; no ground-truth accuracy metrics.")
                for ident in df["id"]:
                    cols=st.columns(3)
                    for col,suffix in zip(cols,["original","marked_original","extracted"]):
                        path=target/"previews"/f"{ident}_{suffix}.png"
                        if path.exists(): col.image(str(path),caption=f"{ident} · {suffix}",use_container_width=True)
            if str(df['strategy'].iloc[0]) in ('auto_v7_1','auto_v7_2','auto_v7_3','auto_v7_4','auto_v7_5') and (target/'previews').exists():
                st.subheader('Three-way verification previews')
                st.caption('REVIEW does not mean a quality pass. These are experimental checks.')
                for ident in df['id'].astype(str).tolist():
                    original_file=next((dataset/p for p in df.loc[df['id'].astype(str)==ident,'original'].tolist()),None)
                    marked=target/'previews'/f'{ident}_marked_original.png'
                    extracted=target/'previews'/f'{ident}_extracted.png'
                    if original_file and marked.exists() and extracted.exists():
                        st.write(f'Slide {ident}')
                        a,b,c=st.columns(3)
                        a.image(str(original_file),caption='Original',use_container_width=True)
                        b.image(str(marked),caption='Marked for deletion',use_container_width=True)
                        c.image(str(extracted),caption='Extracted',use_container_width=True)
            if str(df['strategy'].iloc[0])=='auto_v7_3':
                st.caption('V7.3 repair candidates are experimental and are not automatically accepted. Compare baseline and candidate metrics in the CSV.')
            if str(df['strategy'].iloc[0])=='auto_v7_4':
                st.caption('V7.4 candidates are NOT automatically accepted. Negative MAE delta means improvement on the synthetic reference. Inspect candidate images.')
            if str(df['strategy'].iloc[0])=='auto_v7_5':
                st.caption('V7.5 ranks candidate repairs using synthetic ground truth for benchmarking only. No candidate is automatically applied.')
            st.download_button("Download results CSV",(target/"results.csv").read_bytes(),"results.csv","text/csv")
            st.download_button("Download report",(target/"REPORT.md").read_bytes(),"REPORT.md","text/markdown")
            if (target/"previews").exists():
                with io.BytesIO() as b:
                    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
                        for p in (target/"previews").glob("*.png"): z.write(p,arcname=p.name)
                    st.download_button("Download image previews ZIP",b.getvalue(),"benchmark_previews.zip","application/zip")
else:
    st.write("Upload the selected benchmark ZIP to begin. Your current Background Extraction Lab is unaffected.")
