"""Approve or disapprove each pre-labelled object, one crop at a time.

``autolabel.py`` proposes objects (persistent detector tracks named by appearance
against the Helsinki reference bank) and dumps a crop + a multi-sighting strip
for each. This shows one object per screen - its strip and crop on the left, the
guessed class's reference plate on the right - and takes a single keystroke:

    y  approve (as the shown class)      n  disapprove (drop it)
    a..p  relabel to another class, then approve
    <- ->  move between objects            s  skip / undecided

A real CGI asset holds a crisp silhouette and ground position across the strip;
a moored boat, a rooftop corner or a tree does not. Approving fans the object's
track out into per-frame, view-pixel boxes in ``<seq>.auto.labels.json`` (the
label_tool schema), so a few dozen approvals become a full labelled set.

    python lab/review_crops.py serve --sequence 14356d0b32754c4f9484a4bbb2d5b25d

Offline only: no API call, and it never touches the running deployment.
"""

import argparse
import json
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import source_bbox_to_view  # noqa: E402

RECORDINGS = LAB / 'recordings'
LABELS = LAB / 'labels'
REFERENCE = LAB / 'out' / 'reference'
VIEW_WIDTH, VIEW_HEIGHT = 960, 540


def proposals_dir(sequence):
    return LAB / 'out' / f'proposals_{sequence[:8]}'


def build_labels(sequence, proposals, review):
    """Fan every approved object's track out into per-view, view-pixel boxes.

    Rebuilt in full from the current approvals each time, so changing a decision
    never leaves a stale box behind. Schema matches ``label_tool``'s labels.json.
    """
    by_id = {p['obj_id']: p for p in proposals}
    frames = {}
    for key, decision in review.get('decisions', {}).items():
        if decision.get('decision') != 'approve':
            continue
        proposal = by_id.get(int(key))
        if proposal is None:
            continue
        object_id = decision.get('object_id') or proposal['guess']
        if object_id not in OBJECT_CLASSES:
            continue
        for entry in proposal['track']:
            if not entry.get('inview', True):
                continue
            nx1, ny1, nx2, ny2 = source_bbox_to_view(entry['bbox_source'], entry['region'])
            box = [nx1 * VIEW_WIDTH, ny1 * VIEW_HEIGHT,
                   nx2 * VIEW_WIDTH, ny2 * VIEW_HEIGHT]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            record = frames.setdefault(entry['stem'], {
                'stem': entry['stem'], 'frame': entry.get('frame'),
                'frame_index': entry['frame_index'], 'level': entry['level'],
                'region': entry['region'], 'boxes': [], 'done': True})
            record['boxes'].append({'object_id': object_id,
                                    'bbox': [round(v, 2) for v in box],
                                    'difficult': False})
    return {'sequence': sequence, 'source': 'review_crops',
            'updated': datetime.now(timezone.utc).isoformat(), 'frames': frames}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>review</title>
<style>
body{margin:0;font:14px system-ui;background:#121212;color:#ddd;display:flex;flex-direction:column;height:100vh}
#top{padding:6px 12px;background:#1b1b1b;display:flex;gap:16px;align-items:center;flex:none}
#top b{color:#ffe600}
#wrap{flex:1;display:flex;min-height:0}
#left{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:12px;overflow:auto}
#right{width:340px;background:#1b1b1b;padding:12px;overflow-y:auto;flex:none}
#strip{max-width:100%;image-rendering:pixelated;background:#000;border:1px solid #333}
#crop{width:300px;height:300px;image-rendering:pixelated;background:#000;border:1px solid #333}
#ref{width:100%;image-rendering:pixelated;background:#000;border:1px solid #333}
.big{font-size:20px;font-weight:bold}
.approve{color:#2bd94a}.disapprove{color:#ff5a5a}.skip{color:#888}
button{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:6px 10px;cursor:pointer;border-radius:3px;font-size:14px}
#yes{border-color:#2bd94a}#no{border-color:#ff5a5a}
kbd{background:#333;padding:0 5px;border-radius:3px}
.cls{display:inline-block;padding:3px 6px;margin:2px;border-radius:3px;cursor:pointer;background:#2a2a2a}
.cls.on{outline:2px solid #ffe600;color:#ffe600}
.cand{padding:3px 6px;margin:2px 0;border-radius:3px;cursor:pointer;background:#242424}
.cand:hover{background:#333}
#votes{color:#bbb;line-height:1.7;margin:6px 0}
</style></head><body>
<div id="top">
<span id="pos"></span><span id="tally"></span>
<span>selected class: <b id="selcls"></b></span>
<span id="state" class="big"></span>
<span style="margin-left:auto;color:#999"><kbd>y</kbd> approve <kbd>n</kbd> disapprove
<kbd>a-p</kbd> relabel+approve <kbd>&larr;</kbd><kbd>&rarr;</kbd> move <kbd>s</kbd> skip</span>
</div>
<div id="wrap">
<div id="left">
<img id="strip" title="sightings across frames">
<img id="crop" title="best sighting">
<div><button id="yes" onclick="decide('approve')">approve (y)</button>
<button id="no" onclick="decide('disapprove')">disapprove (n)</button>
<button onclick="go(1)">skip &gt;</button></div>
</div>
<div id="right">
<div class="big">guess: <span id="guess" style="color:#ffe600"></span></div>
<div id="votes"></div>
<img id="ref">
<div id="refname" style="color:#ffe600;margin:4px 0"></div>
<div><b>candidates</b><div id="cands"></div></div>
<div style="margin-top:8px"><b>relabel</b><div id="classes"></div></div>
</div></div>
<script>
const CLASSES=__CLASSES__; let P=[], R={}, at=0, sel=null;
function cur(){return P[at];}
function key(i){return 'abcdefghijklmnop'[i];}
async function load(){
  const d=await (await fetch('api/proposals')).json(); P=d.proposals;
  R=await (await fetch('api/review')).json(); if(!R.decisions)R.decisions={};
  at=Math.max(0,P.findIndex(p=>!R.decisions[p.obj_id])); if(at<0)at=0; show();
}
function decisionOf(p){const e=R.decisions[p.obj_id]; return e?e.decision:null;}
function show(){
  if(!P.length)return; const p=cur();
  const dec=R.decisions[p.obj_id];
  sel=dec&&dec.object_id?dec.object_id:p.guess;
  document.getElementById('strip').src=p.strip?('crop/'+p.strip):'';
  document.getElementById('crop').src='crop/'+p.crop;
  document.getElementById('guess').textContent=p.guess+'  ('+p.appearance_score.toFixed(3)+')';
  document.getElementById('votes').innerHTML=
    'seen in <b>'+p.hits+'</b> frames, span '+p.frame_span[0]+'-'+p.frame_span[1]+
    ' (L'+p.rep_level+')<br>detector: '+
    p.detector_votes.map(v=>v[0]+' '+v[1].toFixed(1)).join(', ');
  const cands=[]; const seen={};
  for(const [c,s] of p.class_candidates){cands.push([c,'appear '+s.toFixed(2)]);seen[c]=1;}
  for(const [c,s] of p.detector_votes){if(!seen[c])cands.push([c,'detector '+s.toFixed(1)]);}
  const C=document.getElementById('cands');C.innerHTML='';
  cands.forEach(([c,tag])=>{const e=document.createElement('div');e.className='cand';
    e.innerHTML='<b>'+c+'</b> <span style="color:#888">'+tag+'</span>';
    e.onclick=()=>{sel=c;paint();};C.appendChild(e);});
  paint();
}
function paint(){
  document.getElementById('selcls').textContent=sel;
  document.getElementById('ref').src='reference/'+sel+'.png';
  document.getElementById('refname').textContent=sel;
  const dec=decisionOf(cur());
  const st=document.getElementById('state');
  st.textContent=dec?dec.toUpperCase():'undecided';
  st.className='big '+(dec||'');
  const done=P.filter(p=>R.decisions[p.obj_id]).length;
  const app=P.filter(p=>decisionOf(p)==='approve').length;
  const dis=P.filter(p=>decisionOf(p)==='disapprove').length;
  document.getElementById('pos').innerHTML='<b>'+(at+1)+'</b> / '+P.length;
  document.getElementById('tally').innerHTML=
    'decided '+done+' (<span class="approve">'+app+' approved</span> / '+
    '<span class="disapprove">'+dis+' disapproved</span>)';
  const G=document.getElementById('classes');G.innerHTML='';
  CLASSES.forEach((n,i)=>{const e=document.createElement('span');
    e.className='cls'+(n===sel?' on':'');
    e.innerHTML=n+' <kbd>'+key(i)+'</kbd>';
    e.onmouseenter=()=>{document.getElementById('ref').src='reference/'+n+'.png';
      document.getElementById('refname').textContent=n;};
    e.onmouseleave=()=>{document.getElementById('ref').src='reference/'+sel+'.png';
      document.getElementById('refname').textContent=sel;};
    e.onclick=()=>{sel=n;paint();};G.appendChild(e);});
}
async function decide(decision){
  if(!P.length)return; const p=cur();
  R.decisions[p.obj_id]={decision:decision,object_id:sel,
    ts:new Date().toISOString()};
  paint();
  await fetch('api/decision',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({obj_id:p.obj_id,decision:decision,object_id:sel})});
  next();
}
function next(){const i=P.findIndex((p,j)=>j>at&&!R.decisions[p.obj_id]);
  at=i>=0?i:Math.min(P.length-1,at+1); show();}
function go(d){at=Math.min(P.length-1,Math.max(0,at+d));show();}
window.addEventListener('keydown',e=>{
  if(!P.length)return; const k=e.key.toLowerCase();
  if(k==='y'){decide('approve');}
  else if(k==='n'){decide('disapprove');}
  else if(k==='s'){go(1);}
  else if(e.key==='ArrowRight'){go(1);}
  else if(e.key==='ArrowLeft'){go(-1);}
  else if('abcdefghijklmnop'.includes(k)){sel=CLASSES['abcdefghijklmnop'.indexOf(k)];
    paint();decide('approve');}
});
load();
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
            pass

    def do_GET(self):
        state = self.state
        path = self.path.split('?')[0].lstrip('/')
        if path in ('', 'index.html'):
            page = PAGE.replace('__CLASSES__', json.dumps(list(OBJECT_CLASSES)))
            return self._send(200, page.encode('utf-8'), 'text/html; charset=utf-8')
        if path == 'api/proposals':
            return self._send(200, json.dumps(state['payload']).encode('utf-8'))
        if path == 'api/review':
            return self._send(200, json.dumps(state['review']).encode('utf-8'))
        if path.startswith('crop/'):
            name = path[5:]
            if name not in state['crops']:
                return self._send(404, b'{}')
            return self._send(200, (state['crop_dir'] / name).read_bytes(), 'image/png')
        if path.startswith('reference/'):
            name = path[10:-4] if path.endswith('.png') else ''
            plate = REFERENCE / f'{name}.png'
            if name not in OBJECT_CLASSES or not plate.is_file():
                return self._send(404, b'{}')
            return self._send(200, plate.read_bytes(), 'image/png')
        return self._send(404, b'{}')

    def do_POST(self):
        state = self.state
        if self.path.split('?')[0] != '/api/decision':
            return self._send(404, b'{}')
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > 100_000:
            return self._send(400, b'{}')
        payload = json.loads(self.rfile.read(length))
        obj_id = payload.get('obj_id')
        decision = payload.get('decision')
        if obj_id not in state['ids'] or decision not in ('approve', 'disapprove'):
            return self._send(400, b'{}')
        object_id = payload.get('object_id')
        if object_id is not None and object_id not in OBJECT_CLASSES:
            object_id = None
        with state['lock']:
            state['review'].setdefault('decisions', {})[str(obj_id)] = {
                'decision': decision, 'object_id': object_id,
                'ts': datetime.now(timezone.utc).isoformat()}
            state['review_path'].write_text(json.dumps(state['review'], indent=1),
                                            encoding='utf-8')
            labels = build_labels(state['sequence'], state['payload']['proposals'],
                                  state['review'])
            state['labels_path'].write_text(json.dumps(labels, indent=1), encoding='utf-8')
        return self._send(200, b'{"ok":true}')


def command_serve(arguments):
    sequence = arguments.sequence
    proposals_path = LABELS / f'{sequence}.proposals.json'
    if not proposals_path.is_file():
        raise SystemExit(f'{proposals_path} missing; run autolabel.py first')
    payload = json.loads(proposals_path.read_text(encoding='utf-8'))
    crop_dir = proposals_dir(sequence)
    review_path = LABELS / f'{sequence}.review.json'
    review = (json.loads(review_path.read_text(encoding='utf-8'))
              if review_path.is_file() else {'sequence': sequence, 'decisions': {}})

    crops = set()
    for proposal in payload['proposals']:
        crops.add(proposal['crop'])
        if proposal.get('strip'):
            crops.add(proposal['strip'])

    state = {'payload': payload, 'review': review, 'sequence': sequence,
             'crop_dir': crop_dir, 'crops': crops,
             'ids': {p['obj_id'] for p in payload['proposals']},
             'review_path': review_path,
             'labels_path': LABELS / f'{sequence}.auto.labels.json',
             'lock': threading.Lock()}
    server = ThreadingHTTPServer(('127.0.0.1', arguments.port),
                                 partial(Handler, state=state))
    url = f'http://127.0.0.1:{arguments.port}/'
    decided = len(review.get('decisions', {}))
    print(f'sequence {sequence}: {len(payload["proposals"])} objects '
          f'({decided} already decided)')
    print(f'labels   {state["labels_path"]}')
    print(f'open     {url}   (ctrl-c to stop)')
    if not arguments.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped; decisions saved after every keystroke')
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    serve = subparsers.add_parser('serve', help='open the crop-approval tool')
    serve.add_argument('--sequence', required=True)
    serve.add_argument('--port', type=int, default=8712)
    serve.add_argument('--no-browser', action='store_true')
    arguments = parser.parse_args()
    if arguments.command == 'serve':
        command_serve(arguments)


if __name__ == '__main__':
    main()
