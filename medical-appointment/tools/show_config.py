"""Print the configuration the server will actually serve with."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

start = time.time()

import api  # noqa: E402,F401  (imports example.py, which runs pipeline.warmup)
from solution import pipeline  # noqa: E402

approach = pipeline.get_approach()

print('import + warmup   %.1f s' % (time.time() - start))
print('APPROACH          %s' % pipeline.APPROACH)
print('ASR_MODEL         %s' % pipeline.ASR_MODEL)
if pipeline.APPROACH in ('quote', 'quote_judge'):
    print('quote model       %s' % approach.model_name)
    print('quote device      MLX Metal')
    print('calibration       scale %s  start %s  end %s' % approach.calibration)
    print('adjudication      %s' % approach.adjudicate)
    print('training examples %s' % len(approach.examples))
    if approach.baseline is not None:
        print('torch device      %s' % approach.baseline.device)
else:
    print('torch device      %s' % getattr(approach, 'device', 'not used'))
    print('span_mode         %s' % getattr(approach, 'span_mode', 'not used'))
print('mps fallback      %s' % os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))
print('routes            %s' % sorted(
    route.path for route in api.app.routes if hasattr(route, 'path')))
