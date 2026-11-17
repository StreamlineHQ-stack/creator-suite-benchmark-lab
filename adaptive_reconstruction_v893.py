"""Conservative region reconstruction. Never alters the detector mask."""
import cv2
import numpy as np
from PIL import Image


def reconstruct_regions(image, baseline, mask):
    src=np.asarray(image.convert('RGB')); out=np.array(baseline.convert('RGB')); m=np.asarray(mask.convert('L'))>127
    if src.shape!=out.shape or m.shape!=src.shape[:2]: raise ValueError('Shape mismatch')
    if not np.array_equal(src[~m],out[~m]): raise ValueError('Baseline changed unmasked pixels')
    h,w=m.shape; n,labels,stats,_=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
    diag={'status':'REVIEW','components':n-1,'accepted':{},'fallback':0,'changed_fraction':0.0}
    if n>501: diag['fallback']=n-1; return baseline,diag
    for k,(x,y,a,b,area) in enumerate(stats[1:],1):
        if area<12 or area>.08*h*w or min(a,b)<2 or max(a,b)>.65*max(h,w): diag['fallback']+=1; continue
        p=max(8,min(24,int(min(a,b)*.25))); X=max(0,x-p); Y=max(0,y-p); R=min(w,x+a+p); B=min(h,y+b+p)
        region=labels[Y:B,X:R]==k; allmask=m[Y:B,X:R]; patch=src[Y:B,X:R].astype(np.float32)
        ring=(cv2.dilate(region.astype('uint8'),np.ones((p*2+1,p*2+1),np.uint8))>0)&~allmask
        # Exclude immediate mask boundary, where antialiasing may survive.
        ring &= ~(cv2.dilate(allmask.astype('uint8'),np.ones((5,5),np.uint8))>0)
        yy,xx=np.indices(region.shape); ry,rx=np.where(ring)
        if len(rx)<max(80,area*.20) or any(v<8 for v in [(rx<x-X).sum(),(rx>=x-X+a).sum(),(ry<y-Y).sum(),(ry>=y-Y+b).sum()]): diag['fallback']+=1; continue
        if len(rx)>12000: ix=np.linspace(0,len(rx)-1,12000,dtype=int); ry,rx=ry[ix],rx[ix]
        design=np.stack([np.ones_like(xx),xx/max(1,R-X-1),yy/max(1,B-Y-1)],axis=-1)
        A=design[ry,rx]; values=patch[ry,rx]; train=(np.arange(len(rx))%5)!=0
        coef=np.linalg.lstsq(A[train],values[train],rcond=None)[0]; pred=design@coef
        err=np.abs(A[~train]@coef-values[~train]); mae=float(err.mean()); q=float(np.quantile(err,.95))
        flat=np.median(values[train],axis=0); flaterr=np.abs(values[~train]-flat)
        if flaterr.mean()<=1.5 and np.quantile(flaterr,.95)<=4: candidate=np.broadcast_to(flat,patch.shape); mode='solid'
        elif mae<=2.5 and q<=6: candidate=pred; mode='gradient'
        else:
            # Only low-amplitude stationary fine grain; no semantic texture synthesis.
            residual=patch-pred; std=float(np.std((values-A@coef)))
            if not (mae<=5 and q<=12 and 1.5<=std<=7): diag['fallback']+=1; continue
            # A contiguous fully known 8x8 tile, validated for comparable residual variance.
            valid=cv2.boxFilter(ring.astype('float32'),-1,(8,8),normalize=False,borderType=cv2.BORDER_CONSTANT)>=64
            cy,cx=np.where(valid); tile=None
            for j in range(0,len(cx),max(1,len(cx)//80)):
                t=residual[cy[j]-4:cy[j]+4,cx[j]-4:cx[j]+4]
                if t.shape==(8,8,3) and abs(float(t.std())-std)<=2 and abs(float(t.mean()))<=2: tile=t-t.mean(axis=(0,1)); break
            if tile is None: diag['fallback']+=1; continue
            candidate=pred+tile[yy%8,xx%8]; mode='fine_grain'
        if np.any(candidate[region]<-3) or np.any(candidate[region]>258): diag['fallback']+=1; continue
        # Require boundary continuity at least comparable to baseline.
        edge=(cv2.dilate(region.astype('uint8'),np.ones((3,3),np.uint8))>0)&~allmask
        if not edge.any(): diag['fallback']+=1; continue
        old=out[Y:B,X:R].astype(np.float32); new=old.copy(); new[region]=np.clip(candidate[region],0,255)
        def seam(v):
            scores=[]
            for axis in (0,1):
                r=np.moveaxis(region,axis,0); e=np.moveaxis(edge,axis,0); z=np.moveaxis(v,axis,0)
                t=(r[1:]&e[:-1])|(r[:-1]&e[1:]); scores.extend(np.abs(z[1:]-z[:-1])[t].mean(axis=-1).tolist())
            return np.mean(scores) if scores else float('inf')
        if seam(new)>seam(old)+1: diag['fallback']+=1; continue
        out[Y:B,X:R][region]=np.rint(new[region]).astype('uint8'); diag['accepted'][mode]=diag['accepted'].get(mode,0)+1
    assert np.array_equal(out[~m],src[~m])
    diag['changed_fraction']=float(np.any(out!=np.asarray(baseline.convert('RGB')),axis=2).mean())
    return Image.fromarray(out),diag


def extract_v893(image,padding=5,radius=4):
    from glyph_detector_v891 import extract_v891
    result,mask,boxes,coverage=extract_v891(image,padding,radius)
    try: result,diag=reconstruct_regions(image,result,mask)
    except Exception as exc: diag={'status':'REVIEW','error':str(exc),'fallback':'baseline'}
    extract_v893.last_diagnostics=diag
    return result,mask,boxes,coverage
