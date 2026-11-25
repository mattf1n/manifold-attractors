"""Compare FP32 runs against FP64 runs of basins_speed.py for every strength both computed: greedy tokens (first N steps)
and max relative difference of every speed curve (through probe; step-midpoint refusal and between series).
Usage: compare_precision.py "FP32JOB ..." "FP64JOB ..." """
import json, sys
from pathlib import Path
import numpy as np
OUT = Path.home() / 'thesis-slides' / 'out'
def index(jobs):
    found = {}
    for j in jobs.split():
        for p in (OUT / j).glob('*.json'):
            if p.name != 'meta.json': found[p.stem] = p
    return found
a, b = index(sys.argv[1]), index(sys.argv[2])
token_mismatch = []; worst = {}
for stem in sorted(set(a) & set(b)):
    ra, rb = json.loads(a[stem].read_text()), json.loads(b[stem].read_text())
    n = min(len(ra['steps']), len(rb['steps'])); h = ra['steering_position']
    if ra['ids'][:h + 1 + n] != rb['ids'][:h + 1 + n]: token_mismatch.append(stem)
    for suffix in (['_speed', 'b_speed'] if stem.endswith('_mid') else ['_speed']):
        fa, fb = a[stem].with_name(stem + suffix + '.npy'), b[stem].with_name(stem + suffix + '.npy')
        if fa.exists() and fb.exists():
            sa, sb = np.load(fa)[:n], np.load(fb)[:n]
            if len(sa) == len(sb): worst[stem + suffix] = float(np.max(abs(sa / sb - 1)))
kinds = {k: [v for s, v in worst.items() if s.endswith(k)] for k in ['0_speed', '_mid_speed', '_midb_speed']}
print(json.dumps(dict(compared=len(set(a) & set(b)), token_mismatch=token_mismatch,
    max_rel={'through': max(kinds['0_speed'] + [v for s, v in worst.items() if not s.endswith(('_mid_speed', '_midb_speed'))], default=None),
             'midpoint_refusal': max(kinds['_mid_speed'], default=None), 'midpoint_between': max(kinds['_midb_speed'], default=None)}), indent=1))
