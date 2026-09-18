"""In-process oracle simulator to measure the CEILING of different architectures.

No ML. We feed ground truth through the real camera dynamics, but a detector can
only "see" an object if its on-screen size at the current resolution level is
large enough (models the downsample). We compare:

  A: stay at L0, no memory              (naive full-frame)
  B: zoom sweep, current-view only      (zoom but stateless)
  C: zoom sweep + motion-model memory   (the proposed approach)

Score with the SAME faster-coco-eval as the competition.
"""
import json, glob, math, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, '..')
from dtos import (ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS,
                  SOURCE_REGION_SIZES, OBJECT_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT)

ANNDIR = '../src/helsinki/annotations'

def load():
    gt = {}
    for f in sorted(glob.glob(f'{ANNDIR}/frame_*.json')):
        d = json.load(open(f))
        gt[d['frame']] = d['annotations']
    return gt

DOWNSCALE = {0: 4, 1: 2, 2: 1}          # source px per transmitted px
MIN_ONSCREEN = 9                         # min transmitted px on shorter side to detect

def source_region(level, cx, cy):
    w, h = SOURCE_REGION_SIZES[level]
    return (cx - w//2, cy - h//2, cx + w//2, cy + h//2)

def visible_and_detectable(bbox, region, level):
    x1,y1,x2,y2 = bbox; rx1,ry1,rx2,ry2 = region
    # center must be inside region
    cx,cy = (x1+x2)/2, (y1+y2)/2
    if not (rx1<=cx<=rx2 and ry1<=cy<=ry2): return False
    w = (x2-x1)/DOWNSCALE[level]; h = (y2-y1)/DOWNSCALE[level]
    return min(w,h) >= MIN_ONSCREEN

def center_bounds(level):
    w,h = SOURCE_REGION_SIZES[level]
    return (w//2, IMAGE_WIDTH-w//2, h//2, IMAGE_HEIGHT-h//2)

class Camera:
    def __init__(s): s.level=0; s.cx=IMAGE_WIDTH//2; s.cy=IMAGE_HEIGHT//2
    def apply(s, level, cx, cy):
        if level not in ALLOWED_RESOLUTION_LEVELS[s.level]: return False
        mnx,mxx,mny,mxy = center_bounds(level)
        if not (mnx<=cx<=mxx and mny<=cy<=mxy): return False
        if level==0:
            if (cx,cy)!=(IMAGE_WIDTH//2,IMAGE_HEIGHT//2): return False
        else:
            if math.hypot(cx-s.cx, cy-s.cy) > MAXIMUM_CENTER_DELTA_PIXELS[s.level]: return False
        s.level,s.cx,s.cy = level,cx,cy; return True

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0

# ----- camera policies -----
def policy_stay_l0(cam, frame_idx): return None

def policy_zoom_sweep(cam, frame_idx):
    """Go to L1 and raster-sweep quadrant centres to cover the whole frame."""
    if cam.level == 0:
        return (1, 960, 540)
    # L1 grid of centres covering frame: x in {960,2880}, y in {540,1620}
    xs=[960,2880]; ys=[540,1620]
    grid=[(x,y) for y in ys for x in xs]
    tgt = grid[frame_idx % len(grid)]
    # move toward target within L1 delta limit
    step=MAXIMUM_CENTER_DELTA_PIXELS[cam.level]
    dx,dy=tgt[0]-cam.cx, tgt[1]-cam.cy
    d=math.hypot(dx,dy)
    if d>step and d>0:
        dx,dy=dx/d*step, dy/d*step
    nx,ny=int(cam.cx+dx),int(cam.cy+dy)
    mnx,mxx,mny,mxy=center_bounds(1)
    nx=min(max(nx,mnx),mxx); ny=min(max(ny,mny),mxy)
    return (1,nx,ny)

# ----- run a config -----
def run(gt, policy, use_memory):
    frames = sorted(gt)
    cam = Camera()
    # memory: class -> list of (frame, bbox) observations
    hist = defaultdict(list)
    preds = defaultdict(list)  # frame -> list of dicts
    for fi, frame in enumerate(frames):
        region = source_region(cam.level, cam.cx, cam.cy)
        # detect: GT objects visible+detectable this frame
        seen = {}
        for a in gt[frame]:
            if visible_and_detectable(a['bbox'], region, cam.level):
                seen[a['object_id']] = a['bbox']
                hist[a['object_id']].append((frame, a['bbox']))
        if use_memory:
            # report every class ever seen, projected to this frame via const velocity
            out=[]
            for cls, obs in hist.items():
                if cls in seen:
                    out.append((cls, seen[cls], 1.0)); continue
                # predict from last two observations
                if len(obs)>=2:
                    (f0,b0),(f1,b1)=obs[-2],obs[-1]
                    if f1!=f0:
                        vel=[(b1[j]-b0[j])/(f1-f0) for j in range(4)]
                        k=frame-f1
                        pred=[b1[j]+vel[j]*k for j in range(4)]
                    else: pred=b1
                else:
                    f1,b1=obs[-1]; pred=b1
                # clip
                px1=max(0,min(IMAGE_WIDTH-1,pred[0])); py1=max(0,min(IMAGE_HEIGHT-1,pred[1]))
                px2=max(px1+1,min(IMAGE_WIDTH,pred[2])); py2=max(py1+1,min(IMAGE_HEIGHT,pred[3]))
                conf=max(0.1, 1.0 - 0.1*abs(frame-f1))
                out.append((cls,(px1,py1,px2,py2),conf))
            preds[frame]=[{'object_id':c,'bbox':b,'confidence':cf} for c,b,cf in out]
        else:
            preds[frame]=[{'object_id':c,'bbox':b,'confidence':1.0} for c,b in seen.items()]
        # move camera
        cmd = policy(cam, fi)
        if cmd is not None: cam.apply(*cmd)
    return preds

# ----- scoring (same as local_evaluator) -----
def score(gt, preds):
    from faster_coco_eval import COCO, COCOeval_faster
    frames=sorted(gt)
    present={a['object_id'] for anns in gt.values() for a in anns}
    ev=[c for c in OBJECT_CLASSES if c in present]
    fid={f:i for i,f in enumerate(frames,1)}
    cid={n:i for i,n in enumerate(OBJECT_CLASSES,1)}
    anns=[]; aid=1
    for f in frames:
        for a in gt[f]:
            x1,y1,x2,y2=a['bbox']
            anns.append({'id':aid,'image_id':fid[f],'category_id':cid[a['object_id']],
                         'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0}); aid+=1
    G={'info':{},'licenses':[],'images':[{'id':fid[f],'width':IMAGE_WIDTH,'height':IMAGE_HEIGHT} for f in frames],
       'categories':[{'id':cid[n],'name':n,'supercategory':'o'} for n in OBJECT_CLASSES],'annotations':anns}
    dt=[]
    for f in frames:
        for p in preds.get(f,[]):
            x1,y1,x2,y2=p['bbox']
            if x2-x1<=0 or y2-y1<=0: continue
            dt.append({'image_id':fid[f],'category_id':cid[p['object_id']],'bbox':[x1,y1,x2-x1,y2-y1],'score':p['confidence']})
    if not dt: return 0.0,{}
    cg=COCO(G); cd=cg.loadRes(dt)
    E=COCOeval_faster(cg,cd,'bbox'); E.params.imgIds=list(fid.values())
    E.params.catIds=[cid[n] for n in ev]; E.params.iouThrs=np.array([0.5])
    E.evaluate(); E.accumulate()
    prec=E.eval['precision']; ap={}
    for i,n in enumerate(ev):
        cp=prec[0,:,i,0,-1]; v=cp[cp>-1]; ap[n]=float(np.mean(v)) if v.size else 0.0
    return sum(ap.values())/len(ap), ap

if __name__=='__main__':
    gt=load()
    configs=[('A: stay L0, no memory', policy_stay_l0, False),
             ('B: zoom sweep, no memory', policy_zoom_sweep, False),
             ('C: zoom sweep + motion memory', policy_zoom_sweep, True)]
    for name,pol,mem in configs:
        preds=run(gt,pol,mem)
        m,ap=score(gt,preds)
        print(f'{name:35s} mAP@0.5 = {m:.3f}')
        if '--verbose' in sys.argv:
            for n,v in sorted(ap.items(),key=lambda x:-x[1]): print(f'      {n:16s} {v:.3f}')
