"""Export basins_speed.py results (~/thesis-slides/out/<job>/, one or more Slurm jobs) to the slides' data files.
Usage: export_js.py [--midpoint] JOB [JOB ...] > assets/basins-steering.js   (--midpoint: > assets/basins-midpoint.js)
Every strength keeps the same number of steps: the fewest any job computed (runs with more are cut to it)."""
import json, os, sys, subprocess, datetime
from pathlib import Path
import numpy as np
args = sys.argv[1:]; MID = '--midpoint' in args; jobs = [a for a in args if a != '--midpoint']
dirs = [Path.home() / 'thesis-slides' / 'out' / j for j in jobs]
# A job still running has no meta.json yet: its finished strengths are exported with the settings of a finished job.
metas = [json.loads((d / 'meta.json').read_text()) for d in dirs if (d / 'meta.json').exists()]
meta = metas[0] if metas else json.loads((Path.home() / 'thesis-slides' / 'out' / '7107923' / 'meta.json').read_text())
STEPS = min(m['n'] for m in metas) if metas else 8
grid = np.linspace(-4, 4, 4001)[::10]; assert len(grid) == meta['alpha_count']
commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
g = lambda v: float(f'{v:.4g}')
suffix = '_mid' if MID else ''
def load(path):
    row = json.loads(path.read_text()); speed = np.load(path.with_name(path.stem + '_speed.npy'))
    between = np.load(path.with_name(path.stem + 'b_speed.npy')) if MID else speed
    steps = []
    for s, sp, sb in zip(row['steps'][:STEPS], speed[:STEPS], between[:STEPS]):
        step = dict(token=s['token'], tokenId=s['token_id'], position=s['position'], applied=s['applied'],
                    marker=round(s['projection'], 4) if MID else s['applied'],
                    minima=[round(a, 3) for a in s['minima']], maxima=[round(a, 3) for a in s['maxima']], speed=[g(v) for v in sp])
        if MID:
            b = s['between']; step['distance'] = round(s['distance'], 4)
            step['between'] = dict(marker=round(b['projection'], 4), distance=round(b['distance'], 4),
                                   minima=[round(a, 3) for a in b['minima']], maxima=[round(a, 3) for a in b['maxima']], speed=[g(v) for v in sb])
        steps.append(step)
    return row, dict(alpha=row['alpha'], response=row['response'], steps=steps)
strengths = {}; harmful = None; prompt = None; position = None
for d in dirs:
    for p in sorted(d.glob('benign_*.json')):
        if p.stem.endswith('_mid') != MID: continue
        row, out = load(p); strengths[round(out['alpha'], 6)] = out; prompt = row['prompt']; position = row['steering_position']
    if not MID:
        for p in sorted(d.glob('harmful_*.json')):
            if p.stem.endswith('_mid'): continue
            row, out = load(p); harmful = dict(prompt=row['prompt'], response=out['response'], steps=out['steps'])
data = dict(model=meta['model'], revision=meta['revision'], precision=meta['precision'], directionLayer=meta['direction_layer'],
            blocks=[meta['input_block'], meta['output_after_block']], prompt=prompt, steeringPosition=position,
            alpha=dict(min=float(grid[0]), max=float(grid[-1]), count=len(grid)),
            strengths=[strengths[a] for a in sorted(strengths)])
if MID: data['referenceAlphas'] = json.loads(next(p for d in dirs for p in d.glob('benign_*_mid.json')).read_text())['reference_alphas']
else: data['harmful'] = harmful
probe_text = ("""// Probe (step midpoint): for step t, the line m_t + b d, m_t = the midpoint of step t's actual activation in the unsteered response and
// in the response steered by 3.3699. speed[i] = || d/db f(m_t + b d) ||_2 at b = grid[i], f under the current response's own prefix.
// marker: the current response's step-t activation x projected onto the line, b = (x - m_t).d / d.d (distance: its distance from the line).
// between: the same along the step's between-activations direction e_t (from the unsteered to the steered activation, scaled to |d|).""" if MID else
"""// Probe (the paper's): the line h + a d through each step's own activation h. speed[i] = || d/da f(h + a d) ||_2 at a = grid[i].
// marker: where the actual activation sits on that line (the steering applied at that step).""")
validation = os.environ.get("VALIDATION", "")
header = f"""// Computed for real on the USC carc cluster (Slurm job(s) {', '.join(jobs)}, {datetime.date.today()}) with the released code of
// "Language Model Activations Inhabit Privileged Error-Correcting Basins" (Finlayson et al., arXiv:2610.04183):
// ~/manifold-attractors at commit {commit}, driver ~/thesis-slides/basins_speed.py (reuses manifold_attractors.maps.refusal),
// exported by ~/thesis-slides/export_js.py. Model {meta['model']} @ {meta['revision'][:12]}, {meta['precision']}; the paper's fresh refusal direction d (layer {meta['direction_layer']}).
// strengths[]: the benign request ("{prompt}") steered by alpha along d at the assistant-header "\\n\\n" only (token position {position};
// the paper's one-position intervention, alpha 0 = unsteered; the paper uses 3.3699), decoded greedily.{'' if MID else ' harmful: the unsteered harmful request.'}
// Steps: first the "\\n\\n" token predicting the first response token (as in the paper's figure); each response keeps its first {STEPS} steps (or up to EOS).
// f = blocks {meta['input_block']}-{meta['output_after_block']} under the token's full causal prefix, applied to activations entering block {meta['input_block']}; centered difference
// (epsilon = {meta['epsilon']}); grid = every 10th point of the paper's linspace(-4, 4, 4001) ({meta['alpha_count']} points); 4 significant digits.
{probe_text}
// minima/maxima: speed extrema on the grid. applied: the steering added at that step.
{validation}"""
name = 'BASINS_MIDPOINT' if MID else 'BASINS_STEERING'
sys.stdout.write(header + f'window.{name} = ' + json.dumps(data, separators=(',', ':'), ensure_ascii=False) + ';\n')
