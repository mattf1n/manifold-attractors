"""Per-output-step refusal-direction speed profiles for the thesis slide "Steering between basins".

Reuses the released code of "Language Model Activations Inhabit Privileged Error-Correcting Basins"
(~/manifold-attractors, manifold_attractors.maps.refusal): same model (Llama 3 8B Instruct, FP64), same fresh refusal
direction (data/refusal_direction), prompts, template, probe (blocks 13-16 under each token's full causal prefix) and the
speed of `next_minimum`:  speed(alpha) = || d/dalpha f(h + alpha d) ||_2 (centered difference, epsilon = 0.0005),
evaluated at every 10th point of its grid linspace(-4, 4, 4001).
For each steering strength alpha, the benign request is steered by alpha at the assistant-header newline only (the paper's
one-position intervention; alpha = 0 is unsteered) and decoded greedily for N steps, recording one profile per step.
Usage: basins_speed.py N KEY ALPHA [ALPHA ...]   (KEY: benign | harmful; harmful ignores alphas and runs unsteered)
Environment PROBE (default both):
  through   the paper's probe: the line h + a d through each step's own activation h (files <tag>.json, <tag>_speed.npy);
  midpoint  per step t, the line m_t + b d with m_t = the midpoint of step t's actual activation in the unsteered response and in
            the response steered by the paper's 3.3699 (reference responses decoded first); the current response's step-t
            activation x is projected onto it: b = (x - m_t).d / d.d (files <tag>_mid.json, <tag>_mid_speed.npy; the speed is
            evaluated under the current response's own prefix). Also along the step's between-activations direction
            e_t = (x1_t - x0_t) scaled to |d| (x0_t, x1_t: the two reference activations), stored under "between" and in
            <tag>_midb_speed.npy;
  both      both of the above.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
import split  # noqa: F401  (device placement only; see split.py)
if os.environ.get('DTYPE') == 'float32': import precision  # noqa: F401  (FP32 variant; see precision.py)
from manifold_attractors.maps.refusal import Backend, BENIGN, HARMFUL, TEMPLATE

N, KEY, ALPHAS = int(sys.argv[1]), sys.argv[2], [float(a) for a in sys.argv[3:]] or [0.]
ROOT = Path.home() / 'manifold-attractors'
OUT = Path.home() / 'thesis-slides' / 'out' / os.environ.get('SLURM_JOB_ID', 'local'); OUT.mkdir(parents=True, exist_ok=True)
GRID = np.linspace(-4, 4, 4001)[::10]
PROBE = os.environ.get('PROBE', 'both'); THROUGH, MIDPOINT = PROBE in ('through', 'both'), PROBE in ('midpoint', 'both') and sys.argv[2] == 'benign'
PAPER_ALPHA = 3.369866762438435


def speed(evaluate, h, d, alpha, epsilon=.0005):
    # Verbatim from manifold_attractors.maps.refusal.next_minimum.speed
    values = []; alpha = np.atleast_1d(alpha)
    for i in range(0, len(alpha), 128):
        points = h + alpha[i:i+128, None] * d
        derivative = (evaluate(points + epsilon * d) - evaluate(points - epsilon * d)) / (2 * epsilon)
        values.extend(np.linalg.norm(derivative, axis=1))
    return np.asarray(values)


torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
meta = json.loads((ROOT / 'data/refusal_direction/direction.json').read_text())
d = torch.load(ROOT / 'data/refusal_direction/direction.pt', map_location='cpu', weights_only=True).double().numpy()
backend = Backend(d)
stops = backend.model.generation_config.eos_token_id; stops = set(stops if isinstance(stops, list) else [stops])
prompt = BENIGN if KEY == 'benign' else HARMFUL
runs = [(a, a != 0) for a in ALPHAS] if KEY == 'benign' else [(0., False)]
def actual_activations(alpha):
    # Step-by-step actual activations (entering block 13, steering included) of the benign response steered by alpha.
    ids = backend.tokenizer(TEMPLATE.format(instruction=BENIGN))['input_ids']; coefficients = {len(ids) - 1: alpha} if alpha else {}
    xs = []
    for index in range(N):
        position = len(ids) - 1
        probe = backend.probe(ids, {p: a for p, a in coefficients.items() if p < position})
        xs.append(probe.point + coefficients.get(position, 0.) * d)
        token, _ = backend.call(ids, coefficients); ids.append(token)
        if token in stops: break
    return xs
if MIDPOINT and KEY == 'benign':
    ref0, ref1 = actual_activations(0.), actual_activations(PAPER_ALPHA)
    mids = [(a + b) / 2 for a, b in zip(ref0, ref1)]
    betweens = [(b - a) / np.linalg.norm(b - a) * np.linalg.norm(d) for a, b in zip(ref0, ref1)]   # no step has identical reference activations
rows = []
for alpha, steered in runs:
    ids = backend.tokenizer(TEMPLATE.format(instruction=prompt))['input_ids']
    header = len(ids) - 1
    coefficients = {header: alpha} if steered else {}
    steps = []; curves = []; mid_steps = []; mid_curves = []; midb_curves = []
    for index in range(N):
        position = len(ids) - 1
        prior = {p: a for p, a in coefficients.items() if p < position}
        probe = backend.probe(ids, prior)
        applied = coefficients.get(position, 0.)
        token_info = dict(token=backend.tokenizer.decode([ids[-1]]), token_id=ids[-1], position=position, applied=applied)
        if THROUGH:
            values = speed(probe.forward, probe.point, d, GRID)
            prom = float(values.max()) * 1e-6
            steps.append(dict(token_info, minima=GRID[find_peaks(-values, prominence=prom)[0]].tolist(),
                              maxima=GRID[find_peaks(values, prominence=prom)[0]].tolist()))
            curves.append(values)
        if MIDPOINT and index < len(mids):
            m = mids[index]
            actual = probe.point + applied * d   # the activation the model actually computes with at this step
            beta = float((actual - m) @ d / (d @ d)); offset = float(np.linalg.norm(actual - m - beta * d))
            values = speed(probe.forward, m, d, GRID)
            prom = float(values.max()) * 1e-6
            e = betweens[index]
            beta_e = float((actual - m) @ e / (e @ e)); offset_e = float(np.linalg.norm(actual - m - beta_e * e))
            values_e = speed(probe.forward, m, e, GRID); prom_e = float(values_e.max()) * 1e-6
            mid_steps.append(dict(token_info, projection=beta, distance=offset,
                                  minima=GRID[find_peaks(-values, prominence=prom)[0]].tolist(),
                                  maxima=GRID[find_peaks(values, prominence=prom)[0]].tolist(),
                                  between=dict(projection=beta_e, distance=offset_e,
                                               minima=GRID[find_peaks(-values_e, prominence=prom_e)[0]].tolist(),
                                               maxima=GRID[find_peaks(values_e, prominence=prom_e)[0]].tolist())))
            mid_curves.append(values); midb_curves.append(values_e)
        token, _ = backend.call(ids, coefficients)
        print(KEY, alpha, index, repr(token_info['token']), '->', repr(backend.tokenizer.decode([token])), flush=True)
        ids.append(token)
        if token in stops: break
    tag = f'{KEY}_{alpha:.4f}'
    common = dict(key=KEY, alpha=alpha, prompt=prompt, steering_position=header, ids=ids, response=backend.tokenizer.decode(ids[header + 1:]))
    if THROUGH:
        np.save(OUT / f'{tag}_speed.npy', np.array(curves))
        (OUT / f'{tag}.json').write_text(json.dumps(dict(common, steps=steps), indent=1))
    if MIDPOINT:
        np.save(OUT / f'{tag}_mid_speed.npy', np.array(mid_curves)); np.save(OUT / f'{tag}_midb_speed.npy', np.array(midb_curves))
        (OUT / f'{tag}_mid.json').write_text(json.dumps(dict(common, probe='step midpoint', reference_alphas=[0., PAPER_ALPHA], steps=mid_steps), indent=1))
(OUT / 'meta.json').write_text(json.dumps(dict(model=meta['model'], revision=meta['revision'], precision=os.environ.get('DTYPE', 'float64'),
    direction_layer=meta['layer'], input_block=13, output_after_block=16, alpha_count=len(GRID),
    alpha_grid='linspace(-4, 4, 4001)[::10]', epsilon=.0005, n=N, probe=PROBE), indent=1))
