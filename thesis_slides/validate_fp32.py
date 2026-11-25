"""Validate the FP32 variant (precision.py) against FP64: step-0 speed vs the paper's cached FP64 profile
(data/refusal_steering_heatmaps/steering_profile.json, every 10th point), and the 8-step greedy responses plus every
step's speed curve vs our FP64 run (out/7107923) for alpha 0 and 3.3699."""
import os, json, numpy as np, torch
import split, precision  # noqa: F401
from manifold_attractors.maps.refusal import Backend, BENIGN, TEMPLATE
ROOT = '/home1/mfinlays/manifold-attractors'; GRID = np.linspace(-4, 4, 4001)[::10]
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
d = torch.load(f'{ROOT}/data/refusal_direction/direction.pt', map_location='cpu', weights_only=True).double().numpy()
b = Backend(d); print('dtype', b.model.dtype, flush=True)
def speed(evaluate, h, alpha, epsilon=.0005):
    out = []
    for i in range(0, len(alpha), 128):
        p = h + alpha[i:i+128, None] * d
        out.extend(np.linalg.norm((evaluate(p + epsilon * d) - evaluate(p - epsilon * d)) / (2 * epsilon), axis=1))
    return np.array(out)
paper = np.array(json.load(open(f'{ROOT}/data/refusal_steering_heatmaps/steering_profile.json'))['speed'])[::10]
result = {}
for alpha, tag in [(0., '0.0000'), (3.369866762438435, '3.3699')]:
    ref = json.load(open(f'/home1/mfinlays/thesis-slides/out/7107923/benign_{tag}.json')); ref_speed = np.load(f'/home1/mfinlays/thesis-slides/out/7107923/benign_{tag}_speed.npy')
    ids = b.tokenizer(TEMPLATE.format(instruction=BENIGN))['input_ids']; coef = {len(ids) - 1: alpha} if alpha else {}
    errs = []; toks = []
    for t in range(8):
        pos = len(ids) - 1; p = b.probe(ids, {q: a for q, a in coef.items() if q < pos})
        s = speed(p.forward, p.point, GRID)
        if t == 0 and alpha == 0: result['step0_vs_paper_max_rel'] = float(np.max(abs(s / paper - 1)))
        errs.append(float(np.max(abs(s / ref_speed[t] - 1))))
        tok, _ = b.call(ids, coef); ids.append(tok); toks.append(tok)
    result[tag] = dict(tokens_match=toks == ref['ids'][len(ref['ids']) - len(ref['steps']) - 16 + 16:][:0] or toks == [s['token_id'] for s in ref['steps'][1:]] + [ref['ids'][ref['steps'][-1]['position'] + 1]],
                       per_step_max_rel=errs)
    print(tag, result[tag], flush=True)
print(json.dumps(result, indent=1))
