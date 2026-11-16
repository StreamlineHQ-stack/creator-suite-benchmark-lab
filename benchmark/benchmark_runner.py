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
from targeted_residual_repair_v73 import propose as propose_v73
from precision_repair_v74 import propose as propose_v74
from residual_candidates_v75 import propose_candidates as propose_v75
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
            t_extract=time.monotonic()
            extracted,mask,boxes,coverage=extract(original,strategy=strategy)
            row['time_extract_seconds']=round(time.monotonic()-t_extract,3)
            if strategy in ('auto_v7_1','auto_v7_2','auto_v7_3','auto_v7_4','auto_v7_5'):
                verify=verify_v72 if strategy in ('auto_v7_2','auto_v7_3','auto_v7_4','auto_v7_5') else verify_v71
                t_verify=time.monotonic()
                first=verify(original,mask,extracted)
                row['time_initial_verify_seconds']=round(time.monotonic()-t_verify,3)
                t_recovery=time.monotonic()
                extracted,mask,recovery=attempt(original,extracted,mask,first)
                row['time_recovery_seconds']=round(time.monotonic()-t_recovery,3)
                if strategy=='auto_v7_5':
                    baseline=measure(mask,truth,extracted,clean,original)
                    t_proposal=time.monotonic()
                    candidates,info=propose_v75(original,extracted,mask, max_candidates=3)
                    row['time_v75_proposal_seconds']=round(time.monotonic()-t_proposal,3)
                    row.update(v75_status=info['status'],v75_candidates=len(candidates),
                               v75_baseline_reference_mae=baseline['reference_mae'],
                               v75_baseline_text_mae=baseline['text_region_mae'])
                    evaluations=[]
                    t_candidate_measure=time.monotonic()
                    for cand in candidates:
                        metrics=measure(cand['mask'],truth,cand['image'],clean,original)
                        evaluations.append((metrics['reference_mae']-baseline['reference_mae'],
                                            metrics['text_region_mae']-baseline['text_region_mae'],cand,metrics))
                    row['time_v75_evaluation_seconds']=round(time.monotonic()-t_candidate_measure,3)
                    # Benchmark-only ranking with known clean reference; never deploy this selector to customers.
                    if evaluations:
                        evaluations.sort(key=lambda x:(x[0],x[1],x[2]['new_pixels']))
                        delta,text_delta,best,metrics=evaluations[0]
                        row.update(v75_best_candidate=best['name'],v75_best_new_pixels=best['new_pixels'],
                                   v75_best_reference_delta=round(delta,6),
                                   v75_best_text_delta=round(text_delta,6),
                                   v75_best_reference_mae=metrics['reference_mae'],
                                   v75_best_text_mae=metrics['text_region_mae'],
                                   v75_candidate_metrics=json.dumps([{'name':c['name'],'pixels':c['new_pixels'],
                                   'reference_delta':round(d,6),'text_delta':round(t,6)} for d,t,c,_ in evaluations]))
                        if save_previews:
                            preview=output/'previews';preview.mkdir(exist_ok=True)
                            best['image'].save(preview/f"{item['id']}_v75_best_candidate.png")
                            best['mask'].save(preview/f"{item['id']}_v75_best_mask.png")
                    # Candidate never replaces baseline extraction.
                if strategy=='auto_v7_4':
                    baseline_metrics=measure(mask,truth,extracted,clean,original)
                    candidate,candidate_mask,proposal=propose_v74(original,extracted,mask)
                    row.update(v74_proposal_status=proposal['status'],v74_regions=proposal.get('regions',0),
                               v74_new_pixels=proposal.get('new_pixels',0),
                               v74_baseline_reference_mae=baseline_metrics['reference_mae'],
                               v74_baseline_text_region_mae=baseline_metrics['text_region_mae'])
                    if proposal['status']=='candidate_for_review':
                        metrics=measure(candidate_mask,truth,candidate,clean,original)
                        row.update(v74_candidate_reference_mae=metrics['reference_mae'],
                                   v74_candidate_text_region_mae=metrics['text_region_mae'],
                                   v74_reference_mae_delta=round(metrics['reference_mae']-baseline_metrics['reference_mae'],6),
                                   v74_text_mae_delta=round(metrics['text_region_mae']-baseline_metrics['text_region_mae'],6))
                        if save_previews:
                            preview=output/'previews';preview.mkdir(exist_ok=True)
                            candidate.save(preview/f"{item['id']}_v74_candidate.png")
                            candidate_mask.save(preview/f"{item['id']}_v74_candidate_mask.png")
                    # Never auto-accept a benchmark candidate.
                if strategy=='auto_v7_3':
                    baseline_metrics=measure(mask,truth,extracted,clean,original)
                    candidate, candidate_mask, proposal=propose_v73(original,extracted,mask)
                    row.update(v73_proposal_status=proposal['status'],
                               v73_regions=proposal.get('regions',0),
                               v73_new_pixels=proposal.get('new_pixels',0),
                               v72_reference_mae=baseline_metrics['reference_mae'],
                               v72_text_region_mae=baseline_metrics['text_region_mae'],
                               v72_mask_iou=baseline_metrics['mask_iou'])
                    # The reference is used only by this benchmark to assess the
                    # proposal. Production code never receives clean/truth images.
                    if proposal['status']=='candidate_for_review':
                        proposed_metrics=measure(candidate_mask,truth,candidate,clean,original)
                        row.update(v73_candidate_reference_mae=proposed_metrics['reference_mae'],
                                   v73_candidate_text_region_mae=proposed_metrics['text_region_mae'],
                                   v73_candidate_mask_iou=proposed_metrics['mask_iou'])
                        # Keep candidate as a benchmark preview, not a certified fix.
                        if save_previews:
                            preview=output/'previews';preview.mkdir(exist_ok=True)
                            candidate.save(preview/f"{item['id']}_v73_candidate.png")
                            candidate_mask.save(preview/f"{item['id']}_v73_candidate_mask.png")
                    # Baseline remains the reported extraction until verified.
                t_final_verify=time.monotonic()
                verification=verify(original,mask,extracted)
                row['time_final_verify_seconds']=round(time.monotonic()-t_final_verify,3)
                row.update(flatten(verification,recovery))
                if strategy in ('auto_v7_2','auto_v7_3','auto_v7_4','auto_v7_5'):
                    row.update(source_regions_inspected=verification.get('source_regions_inspected',0),
                               source_regions_uncovered=verification.get('source_regions_uncovered',0),
                               source_regions_partially_uncovered=verification.get('source_regions_partially_uncovered',0),
                               unchanged_source_regions=verification.get('unchanged_source_regions',0),
                               uncovered_source_boxes=json.dumps(verification.get('uncovered_source_boxes',[])))
            t_measure=time.monotonic()
            row.update(measure(mask,truth,extracted,clean,original))
            row['time_final_measure_seconds']=round(time.monotonic()-t_measure,3)
            if strategy in ('auto_v7','auto_v7_1','auto_v7_2','auto_v7_3','auto_v7_4','auto_v7_5'):
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
                if strategy in ('auto_v7_1','auto_v7_2','auto_v7_3','auto_v7_4','auto_v7_5'):
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
        print(f"{index}/{len(entries)} {item['id']} {row['status']} {row['seconds']}s | timings: " + str({k:v for k,v in row.items() if k.startswith('time_')}),flush=True)
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
