"""CPU LaMa ONNX reconstruction, preserving all pixels outside the supplied mask."""
import os, hashlib, tempfile, threading, urllib.request
from pathlib import Path
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image

URL='https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e46893/lama_fp32.onnx'
SHA='1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6'
LOCK=threading.RLock()

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

@lru_cache(maxsize=1)
def session():
    import onnxruntime as ort
    p=Path(os.environ.get('LAMA_MODEL_PATH',str(Path.home()/'.cache/creator_suite/lama_fp32.onnx')))
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        fd,tmp=tempfile.mkstemp(dir=p.parent); os.close(fd)
        try:
            with urllib.request.urlopen(URL,timeout=120) as src, open(tmp,'wb') as dst:
                import shutil
                shutil.copyfileobj(src,dst)
            if digest(tmp)!=SHA: raise RuntimeError('LaMa model checksum mismatch')
            os.replace(tmp,p)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    if digest(p)!=SHA: raise RuntimeError('LaMa model checksum mismatch')
    opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
    opts.enable_cpu_mem_arena=False; opts.enable_mem_pattern=False
    return ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider'])

def reconstruct(image,mask):
    rgb=np.asarray(image.convert('RGB')); m=np.asarray(mask.convert('L'))>127
    if m.shape!=rgb.shape[:2]: raise ValueError('Mask dimensions mismatch')
    if not m.any(): return image.convert('RGB').copy()
    h,w=m.shape; scale=512/max(h,w); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
    small=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    # Preserve small masked pixels during downsampling.
    sm=(cv2.resize(m.astype(np.float32),(nw,nh),interpolation=cv2.INTER_AREA)>0).astype(np.float32)
    top=(512-nh)//2; left=(512-nw)//2
    pads=(top,512-nh-top,left,512-nw-left)
    canvas=cv2.copyMakeBorder(small,*pads,cv2.BORDER_REPLICATE)
    sm=cv2.copyMakeBorder(sm,*pads,cv2.BORDER_REPLICATE)
    x=np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)/255
    with LOCK:
        pred=session().run(['output'],{'image':x,'mask':sm[None,None]})[0]
    if pred.shape!=(1,3,512,512) or not np.isfinite(pred).all(): raise RuntimeError('Invalid LaMa output')
    patch=pred[0].transpose(1,2,0)[top:top+nh,left:left+nw]
    patch=cv2.resize(patch,(w,h),interpolation=cv2.INTER_LINEAR)
    out=rgb.copy(); out[m]=np.rint(np.clip(patch[m],0,255)).astype(np.uint8)
    return Image.fromarray(out)

def extract_v82(image,padding=5,radius=4):
    from text_detector_v81 import extract_v81
    _,mask,boxes,coverage=extract_v81(image,padding,radius)
    return reconstruct(image,mask),mask,boxes,coverage
