"""Experimental recognition-free, aligned glyph component detector."""
import cv2
import numpy as np
from PIL import Image


def detect(image, mask):
    rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
    old=np.asarray(mask.convert('L'))>127
    if old.shape!=(h,w): raise ValueError('mask size mismatch')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    added=np.zeros((h,w),np.uint8); groups=[]
    # Fixed luminance bands; no recognized words or slide identifiers.
    for binary in (gray>=235, gray<=25):
        n,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype('uint8'),8)
        candidates=[]
        for k,(x,y,a,b,area) in enumerate(stats[1:],1):
            if not (.035*h<=b<=.40*h and .009*w<=a<=.30*w and .12<=area/(a*b)<=.85 and .10<=a/b<=1.35): continue
            if x==0 or y==0 or x+a==w or y+b==h: continue
            component=labels[y:y+b,x:x+a]==k
            if old[y:y+b,x:x+a][component].mean()>.08: continue
            pixels=rgb[y:y+b,x:x+a][component].astype(float)
            if np.median(np.ptp(pixels,axis=1))>22: continue
            candidates.append((k,int(x),int(y),int(a),int(b)))
        if len(candidates)>250: continue
        for k,x,y,a,b in candidates:
            row=sorted([v for v in candidates if abs(v[2]-y)<=.13*b and .75*b<=v[4]<=1.25*b],key=lambda v:v[1])
            # Split lines on unusually large gaps.
            runs=[]
            for v in row:
                if not runs or v[1]-(runs[-1][-1][1]+runs[-1][-1][3])>.65*b: runs.append([v])
                else: runs[-1].append(v)
            for run in runs:
                if len(run)<4 or k not in [v[0] for v in run]: continue
                if run[-1][1]+run[-1][3]-run[0][1]<.19*w: continue
                if any(v[1]<u[1]+u[3] for u,v in zip(run,run[1:])): continue
                ids=[v[0] for v in run]
                if any(ids==g for g in groups): continue
                groups.append(ids)
                for j,X,Y,A,B in run:
                    roi=added[Y:Y+B,X:X+A]; roi[labels[Y:Y+B,X:X+A]==j]=255
    added=cv2.dilate(added,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    added[old]=0
    # Conservative total area budget; reject rather than truncate.
    if np.count_nonzero(added)>.10*h*w: added[:]=0
    return Image.fromarray(added), {'groups':len(groups),'added_coverage':float(np.mean(added>0))}


def extract_v891(image,padding=5,radius=4):
    from verification_repair_v88 import extract_v88
    from lama_v82 import reconstruct
    result,mask,boxes,coverage=extract_v88(image,padding,radius)
    extra,diag=detect(result,mask)
    diag['baseline']=extract_v88.last_diagnostics
    diag['status']='REVIEW'; diag['applied']=False
    if diag['added_coverage']:
        try:
            repaired=reconstruct(result,extra)
            a=np.asarray(extra)>0
            if not np.array_equal(np.asarray(repaired)[~a],np.asarray(result)[~a]): raise ValueError('outside mask changed')
            result=repaired; mask=Image.fromarray(np.maximum(np.asarray(mask),np.asarray(extra)))
            coverage=float(np.mean(np.asarray(mask)>0)); diag['applied']=True
        except Exception as exc: diag['error']=str(exc)
    extract_v891.last_diagnostics=diag
    return result,mask,boxes,coverage
