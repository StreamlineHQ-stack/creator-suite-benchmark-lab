"""Summarize benchmark results without claiming visual acceptance from proxy metrics."""
from collections import defaultdict
from statistics import mean

def write_report(rows,path):
    valid=[r for r in rows if r.get('status')=='ok']
    def avg(rs,key): return round(mean(float(r[key]) for r in rs),4) if rs else None
    lines=['# Background Extraction Lab — Phase 1 Local Baseline','',
      'Automated metrics on synthetic paired images. **These scores do not certify human-visible quality.**',
      f'Images processed: {len(rows)}; completed: {len(valid)}; errors: {len(rows)-len(valid)}','',
      '## Overall',f"- Mean text-mask IoU: {avg(valid,'mask_iou')}",
      f"- Mean text-mask recall: {avg(valid,'mask_recall')}",
      f"- Mean non-text-region MAE (0–255): {avg(valid,'nontext_region_mae')}",
      f"- Mean reconstructed text-region MAE (0–255): {avg(valid,'text_region_mae')}",
      f"- Mean time per slide (s): {avg(valid,'seconds')}",'',
      '## By difficulty','| Difficulty | N | Mean mask IoU | Mean recall | Text-region MAE |','|---|---:|---:|---:|---:|']
    for label in ('Very Hard','Hard','Medium','Easy','Very Easy'):
        group=[r for r in valid if r['difficulty']==label]
        lines.append(f"| {label} | {len(group)} | {avg(group,'mask_iou')} | {avg(group,'mask_recall')} | {avg(group,'text_region_mae')} |")
    lines+=['','## By category','| Category | N | Mean mask IoU | Mean recall | Text-region MAE |','|---|---:|---:|---:|---:|']
    for label in sorted({r['category'] for r in valid}):
        group=[r for r in valid if r['category']==label]
        lines.append(f"| {label} | {len(group)} | {avg(group,'mask_iou')} | {avg(group,'mask_recall')} | {avg(group,'text_region_mae')} |")
    lines+=['','## Lowest mask IoU (review first)','| Slide | Difficulty | Category | IoU | Recall |','|---|---|---|---:|---:|']
    for r in sorted(valid,key=lambda r:float(r['mask_iou']))[:20]:
        lines.append(f"| {r['id']} | {r['difficulty']} | {r['category']} | {r['mask_iou']} | {r['mask_recall']} |")
    if len(valid)!=len(rows):
        lines+=['','## Errors']+[f"- {r['id']}: {r['error']}" for r in rows if r['status']!='ok']
    lines+=['','## Interpretation','- Mask IoU compares detected text-region pixels with synthetic ground-truth text masks.','- Text-region MAE compares the result against the clean reference; lower is better.','- Non-text MAE reveals collateral changes outside true text pixels.','- These metrics do not establish suitability for real photographic or human-subject slides.','- The existing engine uses rectangular OCR masks; a high mask area can still create visible inpainting artifacts.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
