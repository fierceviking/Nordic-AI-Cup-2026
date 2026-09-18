"""Generate a YOLO detection dataset that mimics the wire views (L0/L1/L2 crops).

We render 960x540 tiles exactly like local_evaluator.render_view (crop the 4K
frame to a level's source region, INTER_AREA resize to 960x540), remap GT boxes,
and keep boxes whose center is inside the tile. This trains the detector on the
object on-screen sizes it will actually meet at inference.
"""
import json, glob, cv2, os, random, sys
import numpy as np
sys.path.insert(0, '..')
from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, IMAGE_WIDTH, IMAGE_HEIGHT

random.seed(0)
OUT = 'dataset'
VIEW_W, VIEW_H = 960, 540
CID = {n:i for i,n in enumerate(OBJECT_CLASSES)}

def load():
    anns={}
    for f in sorted(glob.glob('../src/helsinki/annotations/frame_*.json')):
        d=json.load(open(f)); anns[d['frame']]=d['annotations']
    return anns

def make_tile(img, region, gts):
    x1,y1,x2,y2=region
    crop=img[y1:y2, x1:x2]
    sh,sw=crop.shape[:2]
    if (sw,sh)!=(VIEW_W,VIEW_H):
        crop=cv2.resize(crop,(VIEW_W,VIEW_H),interpolation=cv2.INTER_AREA)
    sx=VIEW_W/(x2-x1); sy=VIEW_H/(y2-y1)
    labels=[]
    for a in gts:
        gx1,gy1,gx2,gy2=a['bbox']
        cx,cy=(gx1+gx2)/2,(gy1+gy2)/2
        if not (x1<=cx<x2 and y1<=cy<y2): continue
        # map to tile px, clip
        tx1=max(0,(gx1-x1)*sx); ty1=max(0,(gy1-y1)*sy)
        tx2=min(VIEW_W,(gx2-x1)*sx); ty2=min(VIEW_H,(gy2-y1)*sy)
        if tx2-tx1<2 or ty2-ty1<2: continue
        # yolo normalized cxcywh
        bw=(tx2-tx1)/VIEW_W; bh=(ty2-ty1)/VIEW_H
        bcx=(tx1+tx2)/2/VIEW_W; bcy=(ty1+ty2)/2/VIEW_H
        labels.append((CID[a['object_id']],bcx,bcy,bw,bh))
    return crop, labels

def main():
    anns=load()
    for split in ['train','val']:
        os.makedirs(f'{OUT}/images/{split}',exist_ok=True)
        os.makedirs(f'{OUT}/labels/{split}',exist_ok=True)
    frames=sorted(anns)
    n=0
    for fr in frames:
        img=cv2.imread(f'../src/helsinki/images/frame_{fr:06d}.png')
        gts=anns[fr]
        # val: hold out ~15% of tiles randomly for a detector sanity check
        samples=[]
        # L0: whole frame
        samples.append((0, (0,0,IMAGE_WIDTH,IMAGE_HEIGHT)))
        # L1: 1920x1080 windows — grid + jitter, plus windows centered on each object
        w1,h1=SOURCE_REGION_SIZES[1]
        for a in gts:
            cx,cy=(a['bbox'][0]+a['bbox'][2])/2,(a['bbox'][1]+a['bbox'][3])/2
            for _ in range(3):
                jx=random.randint(-400,400); jy=random.randint(-250,250)
                rx=int(min(max(cx+jx-w1/2,0),IMAGE_WIDTH-w1))
                ry=int(min(max(cy+jy-h1/2,0),IMAGE_HEIGHT-h1))
                samples.append((1,(rx,ry,rx+w1,ry+h1)))
        # L2: 960x540 windows centered on each object with jitter
        w2,h2=SOURCE_REGION_SIZES[2]
        for a in gts:
            cx,cy=(a['bbox'][0]+a['bbox'][2])/2,(a['bbox'][1]+a['bbox'][3])/2
            for _ in range(4):
                jx=random.randint(-300,300); jy=random.randint(-180,180)
                rx=int(min(max(cx+jx-w2/2,0),IMAGE_WIDTH-w2))
                ry=int(min(max(cy+jy-h2/2,0),IMAGE_HEIGHT-h2))
                samples.append((2,(rx,ry,rx+w2,ry+h2)))
        for level,region in samples:
            crop,labels=make_tile(img,region,gts)
            if not labels: continue
            split='val' if random.random()<0.12 else 'train'
            stem=f'f{fr:03d}_L{level}_{n:05d}'
            cv2.imwrite(f'{OUT}/images/{split}/{stem}.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
            with open(f'{OUT}/labels/{split}/{stem}.txt','w') as fh:
                for c,bx,by,bw,bh in labels:
                    fh.write(f'{c} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n')
            n+=1
    # data.yaml
    with open(f'{OUT}/data.yaml','w') as fh:
        fh.write(f'path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\n')
        fh.write(f'nc: {len(OBJECT_CLASSES)}\nnames: {list(OBJECT_CLASSES)}\n')
    print(f'wrote {n} tiles')
    # count per split
    for s in ['train','val']:
        print(s, len(glob.glob(f'{OUT}/images/{s}/*.jpg')))

if __name__=='__main__': main()
