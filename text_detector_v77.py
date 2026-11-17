"""Experimental multi-orientation OCR detector; no reference-image access."""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
from extraction_engine import inpaint


def extract_v77(image, padding=5, radius=4):
    image=image.convert('RGB'); rgb=np.asarray(image); h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    mask=np.zeros((h,w),np.uint8); boxes=[]
    # Suppress broad graphic shapes while retaining narrow strokes.
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dark=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
    light=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)
    stroke=255-cv2.normalize(cv2.max(dark,light),None,0,255,cv2.NORM_MINMAX)
    # Rotate the complete canvas, then map glyph proposals back to source pixels.
    for angle in (0,-15,15,-30,30,-55,55,-75,75):
        mat=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        nw=int(np.ceil(w*abs(mat[0,0])+h*abs(mat[0,1]))); nh=int(np.ceil(w*abs(mat[0,1])+h*abs(mat[0,0])))
        mat[:,2]+=[(nw-w)/2,(nh-h)/2]
        work=cv2.warpAffine(stroke,mat,(nw,nh),borderValue=255)
        try:
            data=pytesseract.image_to_data(work,config='--psm 11',output_type=Output.DICT,timeout=8)
        except RuntimeError:
            continue
        inv=cv2.invertAffineTransform(mat)
        for i,t in enumerate(data['text']):
            t=str(t).strip(); conf=float(data['conf'][i])
            if conf<35 or sum(c.isalnum() for c in t)<2: continue
            x,y,bw,bh=[int(data[k][i]) for k in ('left','top','width','height')]
            if bw<5 or bh<5 or bw*bh>.08*w*h: continue
            crop=work[y:y+bh,x:x+bw]
            if crop.size==0:continue
            _,bright=cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            def cost(m):
                v=m>0; return abs(v.mean()-.28)+.65*np.concatenate([v[0],v[-1],v[:,0],v[:,-1]]).mean()
            part=min((bright,255-bright),key=cost)
            if not .025<(part>0).mean()<.65:continue
            local=np.zeros_like(work); local[y:y+bh,x:x+bw]=part
            mapped=cv2.warpAffine(local,inv,(w,h),flags=cv2.INTER_NEAREST)
            mapped=cv2.dilate(mapped,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
            new=(mapped>0)&(mask==0)
            if new.sum()<max(5,.12*np.count_nonzero(mapped)):continue
            if np.count_nonzero(mask|mapped)>.18*w*h:continue
            mask|=mapped
            ys,xs=np.where(mapped>0)
            boxes.append({'text':t,'confidence':conf,'bbox':(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)),'angle':angle})
    m=Image.fromarray(mask); return inpaint(image,m,radius),m,boxes,float((mask>0).mean())
