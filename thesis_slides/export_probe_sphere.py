"""Export out/probe_sphere/probe_sphere.{npz,json} (probe_sphere.py --merge) to assets/probe-sphere.js for the slides.

Only what the probe slide draws from this run: the mesh, the per-layer activation norms and expansion (log10 area), for layers up to
LAST (the slide's last layer). The run itself also holds layers to 32 and the activation-frame projections (out/probe_sphere/).
"""
import base64, json
from pathlib import Path
import numpy as np
OUT = Path.home() / 'thesis-slides/out/probe_sphere'
LAST = 20
z = np.load(OUT / 'probe_sphere.npz'); info = json.loads((OUT / 'probe_sphere.json').read_text())
AREA = 1000
b64 = lambda a, dtype: base64.b64encode(np.ascontiguousarray(a).astype(dtype).tobytes()).decode()
keep = np.asarray(z['layers']) <= LAST
layers = [int(l) for l in np.asarray(z['layers'])[keep]]
norms = z['norms'][keep]
log_area = np.log10(z['area'][keep])
assert np.abs(log_area).max() * AREA < 32767
faces = z['faces'].astype(np.uint16); assert z['faces'].max() < 65536
fib = z['fib']; k = np.arange(len(fib)); zz = 1 - 2 * (k + 0.5) / len(fib)   # the JS rebuilds the lattice with this formula
assert np.allclose(fib[:, 2], zz, atol=1e-6)
rec = info['record']
data = dict(model=info['model'], dtype=info['dtype'], token=rec['token'], inject=layers[0], layers=layers,
            points=info['points'], radius=round(info['R'], 5), norms=[round(float(v), 5) for v in norms],
            areaScale=AREA, faces=b64(faces.reshape(-1), '<u2'), logArea=[b64(np.rint(a * AREA), '<i2') for a in log_area])
snap = info['snapshot'].rstrip('/').split('/')[-1]
head = f"""// Computed for real on USC CARC (Slurm jobs {', '.join(info['jobs'])} on the discovery cluster) with the code of "Language Model
// Activations Inhabit Privileged Error-Correcting Basins" (Finlayson et al.): ~/manifold-attractors at commit {info['commit']},
// driver ~/thesis-slides/probe_sphere.py (reuses manifold_attractors.maps.bubble_coords / bubble_forward / bubble_geometry and the
// area definition of maps/geometry.surface_area_map), exported by ~/thesis-slides/export_probe_sphere.py.
// Model {info['model']} @ {snap[:12]}, {info['dtype']} (torch {info['torch']}, transformers {info['transformers']}).
// Activation x: the paper's sphere probe (Fig. "Projections of 2D spheres") single sample 0, hidden_states[{data['inject']}] of the token
// "{rec['token']}" (position {rec['position']}) in "{rec['sentence']}" (data/bubble_sources.json records[0]); radius = |x|.
// Sphere: the paper's {info['points']}-point Fibonacci lattice n_k (bubble_geometry.fib_lattice: z = 1 - 2 (k + 0.5) / N, r = sqrt(1 - z^2),
// phi = pi (3 - sqrt 5) k, n = (r cos phi, r sin phi, z)), mapped to X_k = |x| (n_k1 e1 + n_k2 e2 + n_k3 xhat): the activation is the pole
// n = (0, 0, 1); e1, e2 = the bubble's tangents (torch.manual_seed(4242) randn, Gram-Schmidt against x). faces = the lattice's convex hull
// (as the paper's figure triangulates the source sphere), vertex indices.
// Forward: the paper's fast KV-cache probe (bubble_forward.make_forward): context [BOS, BOS], X injected as the last position's residual
// at layer {data['inject']}; layers[] = residual index ({data['inject']} = the injected sphere, ..., {LAST}; raw block outputs, no final norm). The run
// covers layers {info['layers'][0]}-{info['layers'][-1]}; this file keeps those the slide shows. norms[l] = |F_l(x)|, the activation's norm at layer l.
// Expansion: the paper's intrinsic area ratio (maps/geometry.surface_area_map): at each lattice point, orthonormal sphere tangents t1, t2,
// central perturbations kept on the sphere |x| (cos d n +- sin d t_i), d = {info['step_degrees']} degrees, J_i = (F_l(+) - F_l(-)) / (2 |x| d),
// area = sqrt(det J^T J) = the output/input area ratio of the map from layer {data['inject']} to layer l. logArea[l] = log10(area) per point.
// Finite-difference check against d = {info['check_step_degrees']} degrees on {info['check_points']} points x all {len(info['layers'])} layers of the run: relative
// difference median {info['check_rel_median']:.1e}, 99th percentile {info['check_rel_p99']:.1e}, max {info['check_rel_max']:.1e}.
// Encoding: faces = base64 little-endian uint16; logArea[l] = base64 little-endian int16 of round(log10(area) * areaScale).
"""
dest = OUT / 'probe-sphere.js'
dest.write_text(head + 'window.PROBE_SPHERE = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('wrote', dest, dest.stat().st_size, layers)
