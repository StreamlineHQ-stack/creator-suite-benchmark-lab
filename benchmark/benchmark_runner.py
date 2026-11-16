"""Run the unmodified local extraction_engine.py against the 150-slide benchmark.
Usage: python -m benchmark.benchmark_runner --dataset /path/to/creator_suite_150_benchmark --output ./benchmark_results
"""
import argparse, csv, json, sys, time, traceback
from pathlib import Path
from PIL import Image
import numpy as np
from extraction_strategy_router import extract, STRATEGIES
from benchmark.quality_metrics import measure
from extraction_verifier import verify as verify_v71
from extraction_verifier_v72 import verify as verify_v72
from targeted_recovery import attempt
from verification_diagnostics import flatten

def run(dataset, output, limit=None, save_previews=False, strategy="baseline_v6"):
    if strategy not in STRATEGIES: raise ValueError(f"Unknown strategy: {strategy}")
    dataset=Path(dataset); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (dataset/'manifest.csv').open(newline='',encoding='utf-8-sig') as f: entries=list(csv.DictReader(f))
    if limit: entries=entries[:limit]
    rows=[]
    for index, item in enumerate(entries,1):
        start=time.monotonic(); row={k:item[k] for k in ('id','difficulty','category','original')}; row['strategy']=strategy
        try:
            with Image.open(dataset/item['original']) as x: original=x.convert('RGB')
            with Image.open(dataset/item['clean_ground_truth']) as x: clean=x.convert('RGB')
            with Image.open(dataset/item['text_mask']) as x: truth=x.convert('L')
            extracted,mask,boxes,coverage=extract(original,strategy=strategy)
            if strategy in ('auto_v7_1','auto_v7_2'):
                verify=verify_v72 if strategy=='auto_v7_2' else verify_v71
                first=verify(original,mask,extracted)
                extracted,mask,recovery=attempt(original,extracted,mask,first)
                verification=verify(original,mask,extracted)
                row.update(flatten(verification,recovery))
                if strategy=='auto_v7_2':
                    row.update(source_regions_inspected=verification.get('source_regions_inspected',0),
                               source_regions_uncovered=verification.get('source_regions_uncovered',0),
                               source_regions_partially_uncovered=verification.get('source_regions_partially_uncovered',0),
                               unchanged_source_regions=verification.get('unchanged_source_regions',0),
                               uncovered_source_boxes=json.dumps(verification.get('uncovered_source_boxes',[])))
            row.update(measure(mask,truth,extracted,clean,original))
            if strategy in ('auto_v7','auto_v7_1','auto_v7_2'):
                from extraction_strategy_router import extract as selected_extract
                diagnostic = getattr(selected_extract, 'last_diagnostics', None) or {}
                row['selected_strategy'] = diagnostic.get('selected_strategy', '')
                row['manual_review_required'] = diagnostic.get('manual_review_required', True)
                row.setdefault('verification_status', diagnostic.get('verification_status', 'unverified_candidate'))
                row['candidate_scores'] = json.dumps(diagnostic.get('candidate_scores', {}))
            row.update(status='ok',ocr_box_count=len(boxes),error='')
            if save_previews:
                preview=output/'previews';preview.mkdir(exist_ok=True)
                extracted.save(preview/f"{item['id']}_extracted.png")
                mask.save(preview/f"{item['id']}_detected_mask.png")
                if strategy in ('auto_v7_1','auto_v7_2'):
                    from PIL import ImageDraw
                    marked=original.copy(); overlay=Image.new('RGBA',marked.size,(0,0,0,0))
                    arr=np.asarray(mask.convert('L'))>127
                    tint=np.zeros((original.height,original.width,4),dtype=np.uint8)
                    tint[arr]=[255,40,40,105]
                    overlay=Image.fromarray(tint,'RGBA')
                    marked=Image.alpha_composite(marked.convert('RGBA'),overlay).convert('RGB')
                    marked.save(preview/f"{item['id']}_marked_original.png")
        except Exception as e:
            row.update(status='error',error=f'{type(e).__name__}: {e}')
        row['seconds']=round(time.monotonic()-start,2)
        rows.append(row)
        print(f"{index}/{len(entries)} {item['id']} {row['status']} {row['seconds']}s",flush=True)
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as f:
        columns=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    with (output/'results.json').open('w',encoding='utf-8') as f: json.dump(rows,f,indent=2)
    from benchmark.benchmark_report import write_report
    write_report(rows,output/'REPORT.md')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',default='benchmark_results')
    parser.add_argument('--limit',type=int,default=None);parser.add_argument('--save-previews',action='store_true')
    parser.add_argument('--strategy',choices=list(STRATEGIES),default='baseline_v6')
    a=parser.parse_args();run(a.dataset,a.output,a.limit,a.save_previews,a.strategy)
