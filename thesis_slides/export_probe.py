"""Export out/probe_layers.{npz,json} (probe_layers.py) to assets/probe-layers.js for the slides (4 significant digits)."""
import json
from pathlib import Path
import numpy as np
OUT = Path.home() / 'thesis-slides/out'
z = np.load(OUT / 'probe_layers.npz'); info = json.loads((OUT / 'probe_layers.json').read_text())
g = lambda v: float(f'{v:.4g}')
rec = info['record']
data = dict(model=info['model'], dtype=info['dtype'], token=rec['token'], sentence=rec['sentence'], position=rec['position'],
            inject=info['inject'], layers=[int(l) for l in z['layers']], points=int(len(z['theta'])), radius=g(info['R']),
            norms=[g(v) for v in z['norms']], act=[[g(v) for v in a] for a in z['act']],
            xy=[[g(v) for v in c.reshape(-1)] for c in z['xy']])
snap = info['snapshot'].rstrip('/').split('/')[-1]
head = f"""// Computed for real on USC CARC (Slurm job {info['slurm_job']}, host {info['host']}) with the code of "Language Model Activations
// Inhabit Privileged Error-Correcting Basins" (Finlayson et al.): ~/manifold-attractors at commit {info['commit']}, driver
// ~/thesis-slides/probe_layers.py (reuses manifold_attractors.maps.bubble_coords / bubble_forward / bubble_geometry), exported by
// ~/thesis-slides/export_probe.py. Model {info['model']} @ {snap[:12]}, {info['dtype']} (torch {info['torch']}, transformers {info['transformers']}).
// Activation x: the paper's sphere-probe (Fig. "Projections of 2D spheres") single sample 0, hidden_states[{info['inject']}] of the token
// "{rec['token']}" (position {rec['position']}) in "{rec['sentence']}" (data/bubble_sources.json records[0]); |x| = {info['R']:.4g}.
// Probe: the great circle gamma(theta) = |x| (cos theta xhat + sin theta e1), theta = 2 pi k / {info['points']}, k = 0..{info['points'] - 1} (k = 0 is x),
// e1 = the bubble's first tangent (torch.manual_seed(4242) randn, Gram-Schmidt against x), i.e. the bubble sphere's meridian through x.
// Forward: the paper's fast KV-cache probe (bubble_forward.make_forward): context [BOS, BOS], the circle injected as the last position's
// residual at layer {info['inject']}; layers[] = residual index (layer {info['inject']} = the injected circle, 32 = the last block's raw output,
// no final norm). Fast vs full re-forward max relative difference {info['fast_full_audit']:.1e}.
// Projection at each layer l: the paper's avg_image_frame with the single frame point x: a1 = F_l(x)/|F_l(x)|, a2 = J_l e1 (central
// difference, epsilon = {info['ep']:.3g}) made orthogonal to a1 and normalized. xy[l] = [F_l(gamma_k).a1, F_l(gamma_k).a2, ...] measured from
// the origin; act[l] = F_l(x) in the same frame; norms[l] = |F_l(x)|; radius = |x|. 4 significant digits.
"""
(OUT / 'probe-layers.js').write_text(head + 'window.PROBE_LAYERS = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('wrote', OUT / 'probe-layers.js', (OUT / 'probe-layers.js').stat().st_size)
