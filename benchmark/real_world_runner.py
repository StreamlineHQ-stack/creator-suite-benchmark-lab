"""Reference-free internal evaluation. No synthetic accuracy metrics."""
import csv,time,json,re
from pathlib import Path
import numpy as np
from PIL import Image
from extraction_strategy_router import extract,STRATEGIES
from benchmark_ocr_cache import install,clear

def run_real(dataset,output,limit=None,save_previews=True,strategy='baseline_v6',category='All',slide_ids=None):
    dataset=Path(dataset).resolve(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (dataset/'manifest.csv').open(encoding='utf-8-sig',newline='') as f: entries=list(csv.DictReader(f))
    entries=[e for e in entries if category=='All' or e['category']==category]
    if slide_ids is not None:
        requested=set(slide_ids)
        if not requested or requested-{e['id'] for e in entries}: raise ValueError('Empty or unknown slide selection')
        entries=[e for e in entries if e['id'] in requested]
    if limit is not None: entries=entries[:limit]
    if not entries: raise ValueError('No matching slides')
    install(); rows=[]
    for e in entries:
        clear(); start=time.monotonic(); ident=e['id']; row={k:e.get(k,'') for k in ('id','category','name','artifact_flags')}; row.update(original=e['path'],strategy=strategy,ground_truth='unavailable',review_status='pending')
        try:
            if not re.fullmatch(r'[A-Za-z0-9_-]+',ident): raise ValueError('Invalid slide ID')
            p=(dataset/e['path']).resolve()
            if not p.is_relative_to(dataset): raise ValueError('Unsafe image path')
            with Image.open(p) as im: original=im.convert('RGB')
            t=time.monotonic(); result,mask,boxes,_=extract(original,strategy=strategy); row['time_extract_seconds']=round(time.monotonic()-t,3)
            if result.size!=original.size or mask.size!=original.size: raise ValueError('Dimensions changed')
            a=np.asarray(original); b=np.asarray(result.convert('RGB')); m=np.asarray(mask.convert('L'))>127; changed=np.any(a!=b,axis=2)
            row.update(mask_coverage=float(m.mean()),changed_fraction=float(changed.mean()),outside_mask_changed_fraction=float((changed & ~m).sum()/max(1,(~m).sum())),ocr_box_count=len(boxes),status='ok',error='')
            if strategy == 'v86_local':
                from secondary_detector_v86 import extract_v86
                log=extract_v86.last_secondary
                row.update(secondary_accepted=sum(d['decision']=='accepted' for d in log),secondary_decisions=json.dumps(log),secondary_baseline_coverage=extract_v86.baseline_coverage)
            if strategy == 'v851_local':
                from hybrid_mask_v851 import extract_v851
                decisions=extract_v851.last_decisions; refinement=extract_v851.last_refinement
                row.update(imv_candidates=len(decisions),imv_rejected=sum(d['decision'].startswith('reject_') for d in decisions),imv_decisions=json.dumps(decisions),pmr_baseline_coverage=extract_v851.baseline_coverage,pmr_refined=sum(d['decision']=='pixel_expanded' for d in refinement),pmr_decisions=json.dumps(refinement))
            if strategy == 'v85_local':
                from pixel_refiner_v85 import extract_v85
                decisions=extract_v85.last_decisions; refinement=extract_v85.last_refinement
                row.update(imv_candidates=len(decisions),imv_rejected=sum(d['decision'].startswith('reject_') for d in decisions),imv_decisions=json.dumps(decisions),pmr_baseline_coverage=extract_v85.baseline_coverage,pmr_refined=sum(d['decision']=='refined_color' for d in refinement),pmr_decisions=json.dumps(refinement))
            if strategy in ('v84_local','v841_local'):
                if strategy == 'v841_local':
                    from mask_validator_v841 import extract_v841 as validator
                else:
                    from mask_validator_v84 import extract_v84 as validator
                decisions=validator.last_decisions
                row.update(imv_candidates=len(decisions),imv_rejected=sum(d['decision'].startswith('reject_') for d in decisions),imv_decisions=json.dumps(decisions))
            if save_previews:
                d=output/'previews'; d.mkdir(exist_ok=True)
                original.save(d/f'{ident}_original.png'); result.save(d/f'{ident}_extracted.png'); mask.save(d/f'{ident}_detected_mask.png')
                tint=np.zeros((*m.shape,4),np.uint8); tint[m]=[255,40,40,105]
                Image.alpha_composite(original.convert('RGBA'),Image.fromarray(tint)).convert('RGB').save(d/f'{ident}_marked_original.png')
        except Exception as ex: row.update(status='error',error=f'{type(ex).__name__}: {ex}',review_status='not_reviewable')
        row['seconds']=round(time.monotonic()-start,3); rows.append(row)
        _checkpoint(output,rows,strategy,len(entries))
    clear()
    return rows

def _checkpoint(output,rows,strategy,total):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'results.csv.tmp').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (output/'results.csv.tmp').replace(output/'results.csv')
    (output/'results.json').write_text(json.dumps(rows,indent=2))
    (output/'REPORT.md').write_text(f'# Internal real-world benchmark\n\nStrategy: {strategy}\n\nSlides recorded: {len(rows)}/{total}.\n\nRuntime success: {sum(r["status"]=="ok" for r in rows)}/{len(rows)} (not quality).\n\nNo ground truth: precision, recall, IoU and reference MAE are unavailable. Mask coverage and changed pixels are diagnostics only. Manual review required: residual text, background damage, subject preservation, inherent scene text.\n\nRouter extraction only; synthetic-only candidate ranking and post-extraction verification/repair are not run.\n')

