"""Hand-label real recorded competition views, then break down detector error.

Helsinki is saturated: true L1 recovers no object that an upsampled L0 view
already finds (exp11), so it cannot separate the hypotheses behind a 0.15
competition score. The only real imagery we hold is what the camera actually
transmitted, under ``lab/recordings/<sequence>/views``. This labels a strategic
subset of those views by hand and reports where the detector actually fails.

Each view is labelled independently in its own 960x540 pixel space, so nothing
depends on propagating boxes between frames with a homography. That is what
made the earlier provisional labels untrustworthy.

Scope: this measures the detector on the views we received. It is not full
frame AP, because we cannot label ground the camera never looked at.

    python lab/label_tool.py select --count 50
    python lab/label_tool.py serve
    python lab/label_tool.py report --weights model/v4s1.pt

Selection is half evenly spaced and half chosen where the current system looks
unsure. Class disagreement is deliberately not used as an uncertainty signal:
the reporter emits damped alternate classes on the same geometry by design, so
that signal is manufactured rather than observed.
"""

import argparse
import json
import re
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES  # noqa: E402

RECORDINGS = LAB / 'recordings'
LABELS = LAB / 'labels'
REFERENCE = LAB / 'out' / 'reference'
VIEW_WIDTH, VIEW_HEIGHT = 960, 540
CELL = 116
# Older recordings name frames 000001; newer ones append the level, 000001_L0.
STEM = re.compile(r'^[0-9]{6}(_L[0-2])?$')


def sequence_directory(name):
    if name:
        directory = RECORDINGS / name
        if not (directory / 'meta').is_dir():
            raise SystemExit(f'no recorded meta under {directory}')
        return directory
    candidates = [(len(list((p / 'views').glob('*.png'))), p)
                  for p in RECORDINGS.iterdir() if (p / 'views').is_dir()]
    if not candidates:
        raise SystemExit(f'no recordings under {RECORDINGS}')
    return max(candidates)[1]


def read_meta(directory):
    """Recorded requests plus our own response, ordered by frame index."""
    records = []
    for path in sorted((directory / 'meta').glob('*.json')):
        if not STEM.match(path.stem):
            continue
        meta = json.loads(path.read_text(encoding='utf-8'))
        if not (directory / 'views' / f'{path.stem}.png').is_file():
            continue
        meta['stem'] = path.stem
        records.append(meta)
    records.sort(key=lambda m: m['frame_index'])
    return records


def to_view(bbox, region):
    """Global normalised box -> pixel box in the transmitted view."""
    left, top, right, bottom = region
    x1, y1, x2, y2 = bbox
    return [(x1 * 3840 - left) / max(1, right - left) * VIEW_WIDTH,
            (y1 * 2160 - top) / max(1, bottom - top) * VIEW_HEIGHT,
            (x2 * 3840 - left) / max(1, right - left) * VIEW_WIDTH,
            (y2 * 2160 - top) / max(1, bottom - top) * VIEW_HEIGHT]


def inside_view(box, margin=2):
    return (box[2] > margin and box[3] > margin and
            box[0] < VIEW_WIDTH - margin and box[1] < VIEW_HEIGHT - margin)


# --------------------------------------------------------------------------- #
# Class reference, cut from the only ground truth we have
# --------------------------------------------------------------------------- #

def build_reference(force=False):
    """One plate per class: real examples at source, L1 and L0 sampling.

    The competition scene is a different city, so these show the assets rather
    than the backgrounds they will appear against.
    """
    if (not force and REFERENCE.is_dir()
            and len(list(REFERENCE.glob('*.png'))) >= len(OBJECT_CLASSES)):
        return
    import cv2
    from utils import frame_numbers, load_annotations, load_frame

    sightings = {}
    for frame in frame_numbers('helsinki'):
        for annotation in load_annotations(frame, 'helsinki'):
            x1, y1, x2, y2 = (int(v) for v in annotation['bbox'])
            if x1 <= 0 or y1 <= 0 or x2 >= 3840 or y2 >= 2160:
                continue
            sightings.setdefault(annotation['object_id'], []).append(
                ((x2 - x1) * (y2 - y1), frame, (x1, y1, x2, y2)))

    wanted = {}
    for name, items in sightings.items():
        for _, frame, box in sorted(items, key=lambda item: -item[0])[:3]:
            wanted.setdefault(frame, []).append((name, box))

    crops = {}
    for frame in sorted(wanted):
        image = load_frame(frame, 'helsinki')
        for name, (x1, y1, x2, y2) in wanted[frame]:
            pad = max(6, (x2 - x1) // 5)
            crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            if crop.size:
                crops.setdefault(name, []).append((crop, x2 - x1, y2 - y1))
        del image

    REFERENCE.mkdir(parents=True, exist_ok=True)
    for name in OBJECT_CLASSES:
        columns = []
        for crop, width, height in crops.get(name, []):
            scales = []
            for divisor in (1, 2, 4):
                small = cv2.resize(crop, (max(1, crop.shape[1] // divisor),
                                          max(1, crop.shape[0] // divisor)),
                                   interpolation=cv2.INTER_AREA)
                scales.append(cv2.resize(small, (CELL, CELL),
                                         interpolation=cv2.INTER_NEAREST))
            caption = np.zeros((16, CELL, 3), np.uint8)
            cv2.putText(caption, f'{width}x{height}px', (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
            columns.append(cv2.vconcat([caption] + scales))
        if not columns:
            columns = [np.zeros((16 + 3 * CELL, CELL, 3), np.uint8)]
        plate = cv2.hconcat(columns)
        legend = np.zeros((16, plate.shape[1], 3), np.uint8)
        cv2.putText(legend, 'source | L1 | L0', (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.imwrite(str(REFERENCE / f'{name}.png'), cv2.vconcat([plate, legend]))
    print(f'reference plates -> {REFERENCE}')


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def frame_features(records):
    features = []
    previous = 0
    for meta in records:
        region = meta['source_region_xyxy']
        boxes = [(to_view(p['bbox'], region), p['confidence']) for p in meta['predictions']]
        visible = [(b, c) for b, c in boxes if inside_view(b)]
        confidences = [c for _, c in visible]
        strong = sum(1 for c in confidences if c >= 0.30)
        edge = sum(1 for b, _ in visible
                   if min(b[0], b[1]) < 6 or b[2] > VIEW_WIDTH - 6 or b[3] > VIEW_HEIGHT - 6)
        features.append({
            'stem': meta['stem'], 'frame': meta['frame'], 'frame_index': meta['frame_index'],
            'level': meta['resolution_level'], 'region': region,
            'center': [meta['center_x'], meta['center_y']],
            'detections': len(visible), 'strong': strong, 'edge': edge,
            'max_conf': max(confidences, default=0.0),
            'mid_band': sum(1 for c in confidences if 0.05 <= c < 0.30),
            'churn': abs(strong - previous),
        })
        previous = strong
    return features


def choose(features, count, random_share, separation, seed):
    """Half evenly spaced, half where the current system looks least certain."""
    generator = np.random.default_rng(seed)
    chosen, taken = [], []

    def free(candidate):
        return all(abs(candidate['frame_index'] - other['frame_index']) >= separation
                   for other in taken)

    stratified = max(0, min(count, round(count * random_share)))
    by_level = {}
    for item in features:
        by_level.setdefault(item['level'], []).append(item)
    quota = {level: max(1, round(stratified * len(items) / len(features)))
             for level, items in by_level.items()}
    for level, items in sorted(by_level.items()):
        positions = np.linspace(0, len(items) - 1, quota[level] * 3).astype(int)
        for position in dict.fromkeys(positions.tolist()):
            if len([c for c in chosen if c['level'] == level]) >= quota[level]:
                break
            candidate = items[position]
            if free(candidate):
                candidate = dict(candidate, reason='stratified')
                chosen.append(candidate)
                taken.append(candidate)

    counts = np.array([f['detections'] for f in features], dtype=float)
    median = float(np.median(counts)) if len(counts) else 0.0
    scored = []
    for item in features:
        # Frames carrying confident predictions decide AP: a wrong box that
        # outranks a real one is what actually costs score. Labelling these
        # answers whether our confident output is real, which mid-band
        # uncertainty sampling does not.
        informative = (3.0 * item['strong']
                       + 1.0 * item['mid_band'] / max(1.0, median + 1)
                       + 0.8 * item['churn'] / max(1.0, median + 1)
                       + 0.5 * item['edge'] / max(1.0, median + 1))
        scored.append((informative + generator.normal(0, 1e-6), item))
    for _, item in sorted(scored, key=lambda pair: -pair[0]):
        if len(chosen) >= count:
            break
        if free(item):
            item = dict(item, reason='confident')
            chosen.append(item)
            taken.append(item)

    chosen.sort(key=lambda item: item['frame_index'])
    return chosen


def command_select(arguments):
    directory = sequence_directory(arguments.sequence)
    records = read_meta(directory)
    if arguments.levels:
        allowed = {int(value) for value in arguments.levels.split(',')}
        records = [m for m in records if m['resolution_level'] in allowed]
    if not records:
        raise SystemExit('no recorded views matched the filter')
    features = frame_features(records)
    chosen = choose(features, arguments.count, arguments.random_share,
                    arguments.separation, arguments.seed)
    LABELS.mkdir(parents=True, exist_ok=True)
    path = LABELS / f'{directory.name}.manifest.json'
    if path.exists() and not arguments.force:
        raise SystemExit(f'{path} exists; pass --force to reselect (labels are kept separately)')
    path.write_text(json.dumps({
        'sequence': directory.name, 'created': datetime.now(timezone.utc).isoformat(),
        'count': len(chosen), 'separation': arguments.separation, 'seed': arguments.seed,
        'items': chosen}, indent=2), encoding='utf-8')
    levels = {}
    for item in chosen:
        levels[item['level']] = levels.get(item['level'], 0) + 1
    reasons = {}
    for item in chosen:
        reasons[item['reason']] = reasons.get(item['reason'], 0) + 1
    print(f'sequence   {directory.name} ({len(records)} views considered)')
    print(f'selected   {len(chosen)}  by level {dict(sorted(levels.items()))}  {reasons}')
    print(f'manifest   {path}')
    print(f'next       python lab/label_tool.py serve --sequence {directory.name}')


# --------------------------------------------------------------------------- #
# Annotation server
# --------------------------------------------------------------------------- #

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>label</title>
<style>
body{margin:0;font:13px system-ui;background:#121212;color:#ddd;display:flex;height:100vh}
#side{width:250px;padding:10px;overflow-y:auto;background:#1b1b1b;flex:none}
#main{flex:1;position:relative;overflow:hidden}
canvas{position:absolute;top:0;left:0;cursor:crosshair}
#bar{position:absolute;bottom:0;left:0;right:0;padding:6px 10px;background:#000c;font-size:12px}
.cls{padding:3px 5px;margin:1px 0;border-radius:3px;cursor:pointer;display:flex;justify-content:space-between}
.cls.on{outline:2px solid #fff}
kbd{background:#333;padding:0 4px;border-radius:3px}
button{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:4px 8px;cursor:pointer;border-radius:3px}
#list div{padding:2px 4px;cursor:pointer;border-bottom:1px solid #2a2a2a}
#list div.sel{background:#333}
</style></head><body>
<div id="side">
<div><button onclick="go(-1)">&lt; prev</button> <button onclick="go(1)">next &gt;</button>
<button onclick="save(1)">save</button></div>
<div style="margin-top:4px"><button onclick="undo()">undo</button>
<button onclick="redo()">redo</button> <span id="undoinfo" style="color:#888"></span></div>
<div id="progress" style="margin:8px 0"></div>
<div style="margin:6px 0"><label>contrast <input type="range" id="gain" min="100" max="400" value="100"></label></div>
<div style="margin:6px 0"><label><input type="checkbox" id="showpred"> show model output (<kbd>q</kbd>)</label></div>
<img id="ref" style="width:100%;image-rendering:pixelated;background:#000;border:1px solid #333">
<div id="refname" style="color:#ffe600;margin:2px 0 6px"></div>
<div id="classes"></div>
<div style="margin-top:8px"><b>boxes</b><div id="list"></div></div>
<div style="margin-top:8px;color:#999;line-height:1.6">
drag = new box<br>wheel = zoom, right-drag = pan<br><kbd>0</kbd> fit view, <kbd>del</kbd> remove
<br><kbd>ctrl</kbd>+<kbd>z</kbd> undo, <kbd>ctrl</kbd>+<kbd>y</kbd> redo
<br><kbd>&larr;</kbd><kbd>&rarr;</kbd> change frame<br><kbd>enter</kbd> mark frame finished
<br><kbd>x</kbd> mark object difficult<br><kbd>a</kbd>..<kbd>p</kbd> pick class
<br>shift-click a box to select</div>
</div>
<div id="main"><canvas id="cv"></canvas><div id="bar"></div></div>
<script>
const CLASSES=__CLASSES__; let items=[], labels={}, at=0, cls=0, sel=-1;
let undoStack=[], redoStack=[];
let img=new Image(), zoom=2, ox=0, oy=0, drag=null, pan=null;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
function key(i){return 'abcdefghijklmnop'[i];}
function cur(){return items[at];}
function rec(){const k=cur().stem; if(!labels[k]) labels[k]={boxes:[],done:false}; return labels[k];}
function resize(){cv.width=document.getElementById('main').clientWidth;
  cv.height=document.getElementById('main').clientHeight; draw();}
function toImg(e){const r=cv.getBoundingClientRect();
  return [(e.clientX-r.left-ox)/zoom,(e.clientY-r.top-oy)/zoom];}
function draw(){
  ctx.setTransform(1,0,0,1,0,0); ctx.fillStyle='#000'; ctx.fillRect(0,0,cv.width,cv.height);
  if(!items.length){ctx.fillStyle='#888';ctx.font='14px system-ui';
    ctx.fillText('loading manifest...',20,30);return;}
  ctx.setTransform(zoom,0,0,zoom,ox,oy); ctx.imageSmoothingEnabled=false;
  const g=document.getElementById('gain').value/100;
  ctx.filter = g>1 ? 'contrast('+g+') brightness('+(1+(g-1)*0.35)+')' : 'none';
  if(img.complete) ctx.drawImage(img,0,0);
  ctx.filter='none';
  ctx.lineWidth=1/zoom;
  if(document.getElementById('showpred').checked){
    ctx.strokeStyle='#00a6ff';
    for(const p of cur().predictions||[]){ctx.strokeRect(p.bbox[0],p.bbox[1],p.bbox[2]-p.bbox[0],p.bbox[3]-p.bbox[1]);}
  }
  rec().boxes.forEach((b,i)=>{
    ctx.strokeStyle = i===sel ? '#ffe600' : '#2bd94a';
    ctx.strokeRect(b.bbox[0],b.bbox[1],b.bbox[2]-b.bbox[0],b.bbox[3]-b.bbox[1]);
    ctx.fillStyle=ctx.strokeStyle; ctx.font=(11/zoom)+'px system-ui';
    ctx.fillText(b.object_id+(b.difficult?'?':''),b.bbox[0],b.bbox[1]-3/zoom);
  });
  if(drag){ctx.strokeStyle='#fff';ctx.strokeRect(drag[0],drag[1],drag[2]-drag[0],drag[3]-drag[1]);}
  ctx.setTransform(1,0,0,1,0,0);
  const d=items.filter(t=>labels[t.stem]&&labels[t.stem].done).length;
  document.getElementById('bar').textContent=
    `${at+1}/${items.length}  frame ${cur().frame} idx ${cur().frame_index}  L${cur().level}  `+
    `${cur().reason}  boxes ${rec().boxes.length}  ${rec().done?'FINISHED':'unfinished'}  done ${d}/${items.length}`;
  document.getElementById('progress').textContent=`finished ${d} of ${items.length}`;
  const L=document.getElementById('list'); L.innerHTML='';
  rec().boxes.forEach((b,i)=>{const e=document.createElement('div');
    e.textContent=`${i+1}. ${b.object_id}${b.difficult?' (difficult)':''}`;
    if(i===sel)e.className='sel'; e.onclick=()=>{sel=i;draw();}; L.appendChild(e);});
}
function paint(){const c=document.getElementById('classes'); c.innerHTML='';
  document.getElementById('ref').src='reference/'+CLASSES[cls]+'.png';
  document.getElementById('refname').textContent=CLASSES[cls];
  CLASSES.forEach((n,i)=>{const e=document.createElement('div');
    e.className='cls'+(i===cls?' on':''); e.style.background='#2a2a2a';
    e.innerHTML=`<span>${n}</span><kbd>${key(i)}</kbd>`;
    e.onmouseenter=()=>{document.getElementById('ref').src='reference/'+n+'.png';
      document.getElementById('refname').textContent=n;};
    e.onmouseleave=()=>{document.getElementById('ref').src='reference/'+CLASSES[cls]+'.png';
      document.getElementById('refname').textContent=CLASSES[cls];};
    e.onclick=()=>{cls=i; if(sel>=0){push(); rec().boxes[sel].object_id=n;} paint(); draw(); save(0);};
    c.appendChild(e);});}
function fail(what){document.getElementById('bar').textContent='ERROR: '+what;
  console.error(what);}
async function load(){
  try{
    const m=await (await fetch('api/manifest')).json(); items=m.items;
    labels=await (await fetch('api/labels')).json();
    if(!items.length){return fail('manifest is empty; run "select" first');}
    at=Math.max(0,items.findIndex(t=>!(labels[t.stem]&&labels[t.stem].done))); show();
  }catch(e){fail(e.message);}}
function show(){sel=-1; img=new Image();
  img.onload=draw; img.onerror=()=>fail('could not load view '+cur().stem);
  img.src='view/'+cur().stem+'.png';
  fit(); draw();}
function fit(){zoom=Math.min(cv.width/960,cv.height/540);
  ox=(cv.width-960*zoom)/2; oy=(cv.height-540*zoom)/2;}
function go(d){if(!items.length)return; save(0); at=Math.min(items.length-1,Math.max(0,at+d)); show();}
function note(t){document.getElementById('bar').textContent=t;}
function snap(stem){const r=labels[stem]||{boxes:[],done:false};
  return {stem:stem,boxes:JSON.parse(JSON.stringify(r.boxes)),done:!!r.done};}
function push(){if(!items.length)return;
  undoStack.push(snap(cur().stem)); if(undoStack.length>300)undoStack.shift();
  redoStack=[]; status();}
function status(){const u=document.getElementById('undoinfo');
  if(u)u.textContent=`undo ${undoStack.length} / redo ${redoStack.length}`;}
function applySnap(s){const i=items.findIndex(t=>t.stem===s.stem); if(i<0)return;
  labels[s.stem]={boxes:s.boxes,done:s.done};
  if(i!==at){at=i; show();} else {sel=-1; draw();}
  save(0,i); status();}
function undo(){if(!undoStack.length){note('nothing to undo');return;}
  const s=undoStack.pop(); redoStack.push(snap(s.stem)); applySnap(s);}
function redo(){if(!redoStack.length){note('nothing to redo');return;}
  const s=redoStack.pop(); undoStack.push(snap(s.stem)); applySnap(s);}
async function save(flash,index){if(!items.length)return;
  const it=items[index===undefined?at:index];
  const r=labels[it.stem]||{boxes:[],done:false};
  await fetch('api/labels',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stem:it.stem,frame:it.frame,frame_index:it.frame_index,
      level:it.level,region:it.region,record:r})});
  if(flash)note('saved');}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{if(!items.length)return; const p=toImg(e);
  if(e.button===2){pan=[e.clientX,e.clientY,ox,oy];return;}
  const hit=rec().boxes.findIndex(b=>p[0]>=b.bbox[0]&&p[0]<=b.bbox[2]&&p[1]>=b.bbox[1]&&p[1]<=b.bbox[3]);
  if(hit>=0&&e.shiftKey){sel=hit;draw();return;}
  drag=[p[0],p[1],p[0],p[1]];});
window.addEventListener('mousemove',e=>{
  if(pan){ox=pan[2]+e.clientX-pan[0];oy=pan[3]+e.clientY-pan[1];draw();return;}
  if(drag){const p=toImg(e);drag[2]=p[0];drag[3]=p[1];draw();}});
window.addEventListener('mouseup',()=>{pan=null;
  if(!drag)return;
  const b=[Math.min(drag[0],drag[2]),Math.min(drag[1],drag[3]),
           Math.max(drag[0],drag[2]),Math.max(drag[1],drag[3])];
  drag=null;
  if(b[2]-b[0]>=1&&b[3]-b[1]>=1){push(); rec().boxes.push({object_id:CLASSES[cls],
    bbox:b.map(v=>Math.round(v*100)/100),difficult:false}); sel=rec().boxes.length-1; save(0);}
  draw();});
cv.addEventListener('wheel',e=>{e.preventDefault(); const p=toImg(e);
  zoom=Math.min(24,Math.max(0.5,zoom*(e.deltaY<0?1.25:0.8)));
  const r=cv.getBoundingClientRect();
  ox=e.clientX-r.left-p[0]*zoom; oy=e.clientY-r.top-p[1]*zoom; draw();},{passive:false});
window.addEventListener('keydown',e=>{
  if(!items.length)return;
  if(e.ctrlKey||e.metaKey){const k=e.key.toLowerCase();
    if(k==='z'&&!e.shiftKey){e.preventDefault();undo();}
    else if(k==='y'||(k==='z'&&e.shiftKey)){e.preventDefault();redo();}
    return;}
  if(e.key==='ArrowRight')go(1);
  else if(e.key==='ArrowLeft')go(-1);
  else if(e.key==='Delete'||e.key==='Backspace'){if(sel>=0){push();rec().boxes.splice(sel,1);sel=-1;save(0);draw();}}
  else if(e.key==='Enter'){push();rec().done=!rec().done;save(1);draw();}
  else if(e.key==='x'){if(sel>=0){push();rec().boxes[sel].difficult=!rec().boxes[sel].difficult;save(0);draw();}}
  else if(e.key==='q'){const c=document.getElementById('showpred');c.checked=!c.checked;draw();}
  else if(e.key==='0'){fit();draw();}
  else if('abcdefghijklmnop'.includes(e.key)){cls='abcdefghijklmnop'.indexOf(e.key);
    if(sel>=0){push(); rec().boxes[sel].object_id=CLASSES[cls];} paint(); draw(); save(0);}
});
document.getElementById('gain').addEventListener('input',draw);
document.getElementById('showpred').addEventListener('change',draw);
window.addEventListener('resize',resize);
window.addEventListener('error',e=>fail(e.message+' @'+e.lineno));
paint(); resize(); status(); load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state=None, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def _send(self, code, body, kind='application/json'):
        try:
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, OSError):
            # The browser cancels in-flight reference images while hovering.
            pass

    def do_GET(self):
        state = self.state
        path = self.path.split('?')[0].lstrip('/')
        if path in ('', 'index.html'):
            page = PAGE.replace('__CLASSES__', json.dumps(list(OBJECT_CLASSES)))
            return self._send(200, page.encode('utf-8'), 'text/html; charset=utf-8')
        if path == 'api/manifest':
            return self._send(200, json.dumps(state['manifest']).encode('utf-8'))
        if path == 'api/labels':
            return self._send(200, json.dumps(state['labels']['frames']).encode('utf-8'))
        if path.startswith('view/'):
            stem = path[5:-4] if path.endswith('.png') else ''
            # Only stems present in the manifest are readable; no path traversal.
            if stem not in state['stems']:
                return self._send(404, b'{}')
            data = (state['directory'] / 'views' / f'{stem}.png').read_bytes()
            return self._send(200, data, 'image/png')
        if path.startswith('reference/'):
            name = path[10:-4] if path.endswith('.png') else ''
            plate = REFERENCE / f'{name}.png'
            if name not in OBJECT_CLASSES or not plate.is_file():
                return self._send(404, b'{}')
            return self._send(200, plate.read_bytes(), 'image/png')
        return self._send(404, b'{}')

    def do_POST(self):
        state = self.state
        if self.path.split('?')[0] != '/api/labels':
            return self._send(404, b'{}')
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > 4_000_000:
            return self._send(400, b'{}')
        payload = json.loads(self.rfile.read(length))
        stem = str(payload.get('stem', ''))
        if stem not in state['stems']:
            return self._send(400, b'{}')
        record = payload.get('record') or {}
        boxes = []
        for box in record.get('boxes') or []:
            if box.get('object_id') not in OBJECT_CLASSES:
                continue
            values = [float(v) for v in box.get('bbox', [])[:4]]
            if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
                continue
            boxes.append({'object_id': box['object_id'],
                          'bbox': [round(v, 2) for v in values],
                          'difficult': bool(box.get('difficult'))})
        with state['lock']:
            state['labels']['frames'][stem] = {
                'stem': stem, 'frame': payload.get('frame'),
                'frame_index': payload.get('frame_index'), 'level': payload.get('level'),
                'region': payload.get('region'), 'boxes': boxes,
                'done': bool(record.get('done')),
                'updated': datetime.now(timezone.utc).isoformat()}
            state['path'].write_text(json.dumps(state['labels'], indent=1), encoding='utf-8')
        return self._send(200, b'{"ok":true}')


def command_serve(arguments):
    directory = sequence_directory(arguments.sequence)
    build_reference()
    manifest_path = LABELS / f'{directory.name}.manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'{manifest_path} missing; run "select" first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = {m['stem']: m for m in read_meta(directory)}
    for item in manifest['items']:
        meta = records.get(item['stem'], {})
        item['predictions'] = [
            {'bbox': [round(v, 2) for v in to_view(p['bbox'], item['region'])],
             'object_id': p['object_id'], 'confidence': p['confidence']}
            for p in meta.get('predictions', []) if p['confidence'] >= arguments.preview_conf]
    path = LABELS / f'{directory.name}.labels.json'
    labels = (json.loads(path.read_text(encoding='utf-8')) if path.is_file()
              else {'sequence': directory.name, 'frames': {}})
    state = {'manifest': manifest, 'labels': labels, 'path': path, 'directory': directory,
             'stems': {item['stem'] for item in manifest['items']}, 'lock': threading.Lock()}
    server = ThreadingHTTPServer(('127.0.0.1', arguments.port),
                                 partial(Handler, state=state))
    url = f'http://127.0.0.1:{arguments.port}/'
    print(f'sequence {directory.name}: {len(manifest["items"])} views to label')
    print(f'labels   {path}')
    print(f'open     {url}   (ctrl-c to stop)')
    if not arguments.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped; labels saved after every edit')
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# Error breakdown
# --------------------------------------------------------------------------- #

def overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def match_frame(truth, detections, threshold=0.5):
    remaining = sorted(detections, key=lambda d: -d['confidence'])
    rows = []
    for target in truth:
        best, score = None, threshold
        for candidate in remaining:
            value = overlap(candidate['bbox'], target['bbox'])
            if value >= score:
                best, score = candidate, value
        if best is not None:
            remaining.remove(best)
        rows.append({'truth': target, 'match': best, 'iou': score if best else 0.0})
    return rows, remaining


def command_report(arguments):
    directory = sequence_directory(arguments.sequence)
    path = LABELS / f'{directory.name}.labels.json'
    if not path.is_file():
        raise SystemExit(f'{path} missing; label some views first')
    labels = json.loads(path.read_text(encoding='utf-8'))
    frames = [f for f in labels['frames'].values() if f.get('done')]
    if not frames:
        raise SystemExit('no frames marked finished (press f in the tool)')

    import cv2
    from solution import Detector
    detector = Detector(weights=arguments.weights, imgsz=960,
                        confidence=arguments.conf, device='0', drop_view_edge=False)
    rows, spurious, per_class, per_level = [], [], {}, {}
    try:
        for record in sorted(frames, key=lambda f: f['frame_index']):
            image = cv2.imread(str(directory / 'views' / f'{record["stem"]}.png'))
            region = record['region']
            detections = [{'object_id': d['object_id'], 'confidence': d['confidence'],
                           'bbox': to_view([d['bbox'][0] / 3840, d['bbox'][1] / 2160,
                                            d['bbox'][2] / 3840, d['bbox'][3] / 2160], region)}
                          for d in detector(image, region)]
            truth = [b for b in record['boxes'] if arguments.difficult or not b['difficult']]
            matched, unmatched = match_frame(truth, detections, arguments.iou)
            rows.extend(matched)
            spurious.extend(unmatched)
            for item in matched:
                name = item['truth']['object_id']
                bucket = per_class.setdefault(name, [0, 0, 0])
                bucket[0] += 1
                bucket[1] += int(item['match'] is not None)
                bucket[2] += int(item['match'] is not None and
                                 item['match']['object_id'] == name)
                level = per_level.setdefault(record['level'], [0, 0])
                level[0] += 1
                level[1] += int(item['match'] is not None)
    finally:
        detector._pool.shutdown(wait=True)

    total = len(rows)
    located = [r for r in rows if r['match'] is not None]
    named = [r for r in located if r['match']['object_id'] == r['truth']['object_id']]
    print(f'\nlabelled views {len(frames)}   labelled objects {total}   '
          f'detector {Path(arguments.weights).name} @ conf {arguments.conf}')
    print(f'\n{"localisation recall":24s} {len(located) / max(1, total):.3f}  '
          f'({len(located)}/{total} at IoU>={arguments.iou})')
    print(f'{"same-class recall":24s} {len(named) / max(1, total):.3f}  ({len(named)}/{total})')
    print(f'{"naming given located":24s} {len(named) / max(1, len(located)):.3f}')
    predicted = len(located) + len(spurious)
    print(f'{"precision":24s} {len(named) / max(1, predicted):.3f}  '
          f'({len(named)} correct of {predicted} predictions)')
    print(f'{"mean IoU when located":24s} {np.mean([r["iou"] for r in located]) if located else 0:.3f}')

    sides = [min(r['truth']['bbox'][2] - r['truth']['bbox'][0],
                 r['truth']['bbox'][3] - r['truth']['bbox'][1]) for r in rows]
    print('\nrecall by object short side in view pixels')
    for low, high in ((0, 8), (8, 16), (16, 32), (32, 10 ** 6)):
        subset = [r for r, s in zip(rows, sides) if low <= s < high]
        if subset:
            hit = sum(1 for r in subset if r['match'] is not None)
            name = sum(1 for r in subset if r['match'] and
                       r['match']['object_id'] == r['truth']['object_id'])
            print(f'  {low:3d}-{high if high < 10 ** 6 else "inf":>4} px  n={len(subset):4d}  '
                  f'located {hit / len(subset):.3f}  named {name / len(subset):.3f}')

    print('\nrecall by resolution level')
    for level, (count, hit) in sorted(per_level.items()):
        print(f'  L{level}  n={count:4d}  located {hit / max(1, count):.3f}')

    print('\nper class (n, located, named)')
    for name, (count, hit, correct) in sorted(per_class.items(), key=lambda kv: -kv[1][0]):
        print(f'  {name:16s} n={count:4d}  located {hit / count:.3f}  named {correct / count:.3f}')

    if located:
        values = np.array([r['match']['confidence'] for r in located])
        print(f'\nconfidence of located truths: median {np.median(values):.3f}  '
              f'p10 {np.percentile(values, 10):.3f}  p90 {np.percentile(values, 90):.3f}')
    if spurious:
        values = np.array([d['confidence'] for d in spurious])
        print(f'confidence of unmatched predictions: median {np.median(values):.3f}  '
              f'p90 {np.percentile(values, 90):.3f}  count {len(spurious)}')

    summary = {'sequence': directory.name, 'weights': arguments.weights,
               'views': len(frames), 'objects': total, 'iou': arguments.iou,
               'conf': arguments.conf,
               'localisation_recall': len(located) / max(1, total),
               'same_class_recall': len(named) / max(1, total),
               'precision': len(named) / max(1, predicted),
               'per_class': per_class, 'per_level': per_level}
    out = LABELS / f'{directory.name}.report.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')
    print('Scope: detector on transmitted views only. Not full-frame AP, and not '
          'the competition metric.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sequence', default=None, help='default: recording with most views')
    subparsers = parser.add_subparsers(dest='command', required=True)

    select = subparsers.add_parser('select', help='choose which views to label')
    select.add_argument('--count', type=int, default=50)
    select.add_argument('--random-share', type=float, default=0.5)
    select.add_argument('--separation', type=int, default=4,
                        help='minimum frame-index gap; neighbouring views overlap heavily')
    select.add_argument('--levels', default=None, help='restrict to e.g. "1,2"')
    select.add_argument('--seed', type=int, default=0)
    select.add_argument('--force', action='store_true')

    serve = subparsers.add_parser('serve', help='open the annotation tool')
    serve.add_argument('--port', type=int, default=8711)
    serve.add_argument('--preview-conf', type=float, default=0.30)
    serve.add_argument('--no-browser', action='store_true')

    report = subparsers.add_parser('report', help='detector error breakdown on the labels')
    report.add_argument('--weights', default=str(ROOT / 'model' / 'v4s1.pt'))
    report.add_argument('--conf', type=float, default=0.05)
    report.add_argument('--iou', type=float, default=0.5)
    report.add_argument('--difficult', action='store_true', help='include difficult objects')

    reference = subparsers.add_parser('reference', help='rebuild the class reference plates')
    reference.add_argument('--force', action='store_true', default=True)

    arguments = parser.parse_args()
    if arguments.command == 'select':
        if not 0 <= arguments.random_share <= 1 or arguments.count < 1:
            parser.error('invalid --count or --random-share')
        command_select(arguments)
    elif arguments.command == 'serve':
        command_serve(arguments)
    elif arguments.command == 'reference':
        build_reference(force=True)
    else:
        command_report(arguments)


if __name__ == '__main__':
    main()
