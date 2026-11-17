"""Reference-free diagnostics and bounded candidate selection; not a quality certificate."""
import numpy as np
import cv2


def assess(image, candidate, source_words, read_words, match_words, target):
    result, mask, boxes, _ = candidate
    a=np.asarray(image.convert('RGB')); b=np.asarray(result.convert('RGB')); m=np.asarray(mask.convert('L'))>127
    if a.shape!=b.shape or m.shape!=a.shape[:2]: raise ValueError('dimension mismatch')
    if np.any(a[~m]!=b[~m]): raise ValueError('outside mask changed')
    output_words=read_words(result)
    hits=match_words(source_words,output_words,target)
    all_hits=match_words(source_words,output_words,np.ones_like(target))
    edges=[]
    for axis in (0,1):
        z=np.moveaxis(b.astype('float32'),axis,0); q=np.moveaxis(m,axis,0)
        edge=q[1:]!=q[:-1]
        if edge.any(): edges.extend(np.abs(z[1:]-z[:-1])[edge].mean(axis=1).tolist())
    seam=float(np.mean(edges)) if edges else 0.
    coverage=float(m.mean()); reasons=[]
    if hits is None: reasons.append('ocr_unavailable')
    elif hits: reasons.append('matched_residual_text')
    if all_hits and (hits is not None and len(all_hits)>len(hits)): reasons.append('possible_unmasked_or_scene_text')
    if coverage>.35: reasons.append('large_mask')
    if seam>18: reasons.append('boundary_discontinuity')
    if not m.any(): reasons.append('empty_mask')
    return {'coverage':coverage,'residual_matches':None if hits is None else len(hits),'boundary_mean':seam,'all_residual_matches':None if all_hits is None else len(all_hits),'flags':reasons}


def choose(primary, alternative, p, q):
    """Only choose alternate for strictly fewer matched words, no mask expansion,
    no substantial boundary regression. OCR misses and semantic damage remain possible."""
    if p['residual_matches'] is None or q['residual_matches'] is None: return False
    m=np.asarray(primary[1].convert('L'))>127; n=np.asarray(alternative[1].convert('L'))>127
    return bool(q['residual_matches']<p['residual_matches'] and q.get('all_residual_matches',0)<=p.get('all_residual_matches',0) and not np.any(n & ~m) and q['boundary_mean']<=p['boundary_mean']+2)


def extract_v894(image,padding=5,radius=4):
    from adaptive_reconstruction_v893 import extract_v893
    from mask_validator_v841 import extract_v841
    from verification_repair_v88 import words,residual
    primary=extract_v893(image,padding,radius); chosen=primary
    target=np.asarray(primary[1].convert('L'))>127
    diag={'status':'REVIEW','selected':'v893','alternative_attempted':False}
    try:
        source=words(image)
        p=assess(image,primary,source,words,residual,target); diag['primary']=p
        if p['residual_matches'] is not None and p['flags']:
            diag['alternative_attempted']=True
            alt=extract_v841(image,padding,radius)
            q=assess(image,alt,source,words,residual,target); diag['alternative']=q
            if choose(primary,alt,p,q): chosen=alt; diag['selected']='v841'
        diag['selected_flags']=diag.get('alternative',p)['flags'] if diag['selected']=='v841' else p['flags']
    except Exception as exc:
        diag['verification_error']=f'{type(exc).__name__}: {exc}'
    extract_v894.last_diagnostics=diag
    return chosen
