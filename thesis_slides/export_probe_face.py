"""Export out/probe_face/probe_face.{npz,json} (probe_face.py) to assets/probe-face.js; checks it against out/probe_sphere (FP32 asset data).

Rank per layer: the stored basis (residual <= 0.5% of |F_l(x)| per point) is truncated to the fewest leading components for which the
front-face projection the slide draws (all points onto the span of a face's three output points, axes: centroid direction and the two
face edges made orthonormal) differs from the untruncated one by <= TOL of the panel extent (max over points, 95th percentile over
FACES random faces). Coordinates are int16 with one scale per component (8-bit per component was tested: ~20% errors, rejected).
"""
import base64, json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from manifold_attractors.maps.bubble_geometry import fib_lattice
OUT = Path.home() / 'thesis-slides/out/probe_face'
TOL, FACES, SEED, LAST = 0.01, 200, 1, 20   # LAST: the slide's last layer (the run covers 15-25; later layers are not exported)
z = np.load(OUT / 'probe_face.npz'); info = json.loads((OUT / 'probe_face.json').read_text())
ref = np.load(Path.home() / 'thesis-slides/out/probe_sphere/probe_sphere.npz')
faces = ConvexHull(fib_lattice(info['points']).astype(np.float64)).simplices
sel = np.random.default_rng(SEED).choice(len(faces), FACES, replace=False)
b64 = (lambda a, t: base64.b64encode(np.ascontiguousarray(a).astype(t).tobytes()).decode())


def frame(C, f):
    A, B, Cc = C[faces[f]]; cen = (A + B + Cc) / 3; u1 = cen / np.linalg.norm(cen)
    b = B - A; b -= b @ u1 * u1; u2 = b / np.linalg.norm(b)
    c = Cc - A; c -= c @ u1 * u1 + c @ u2 * u2; u3 = c / np.linalg.norm(c)
    return np.stack([u1, u2, u3], 1)


def face_error(Ck, C, ext):   # 95th percentile over faces of the max point error, in panel extents (sign of each axis ignored)
    return float(np.percentile([np.abs(np.abs(Ck @ frame(Ck, f)) - np.abs(C @ frame(C, f))).max() / ext for f in sel], 95))


layers, rows = [], []
for L in [L for L in info['layers'] if L['layer'] <= LAST]:
    l = L['layer']; C = z[f'C{l}'] / L['act_norm']                    # normalized by the layer's activation norm |F_l(x)|
    ext = np.linalg.norm(C, axis=1).max(); r = C.shape[1]
    k = r
    if l != info['layers'][0]['layer']:
        for k in range(4, r + 1):
            if face_error(C[:, :k], C, ext) <= TOL: break
    Ck = C[:, :k]
    scale = np.abs(Ck).max(0) / 32767; scale[scale == 0] = 1
    q = np.rint(Ck / scale)
    qerr = face_error(q * scale, Ck, ext) if l != info['layers'][0]['layer'] else 0.0
    i = list(ref['layers']).index(l)                                   # FP64 vs the FP32 run behind assets/probe-sphere.js
    d32 = float(np.abs(z[f'proj{l}'] / L['act_norm'] - ref['proj'][i] / ref['norms'][i]).max())
    layers.append(dict(layer=l, rank=k, scale=[float(s) for s in scale], coords=b64(q.reshape(-1), '<i2')))
    rows.append(dict(layer=l, rank=k, stored_rank=r, truncation_error=face_error(Ck, C, ext) if k < r else 0.0, int16_error=qerr,
                     fp32_vs_fp64=d32, residual=L['max_residual_over_act_norm']))
    print(rows[-1], flush=True)
data = dict(model=info['model'], dtype=info['dtype'], points=info['points'], layers=layers)
table = '\n'.join(f"//   {r['layer']}: rank {r['rank']} (of {r['stored_rank']}), truncation {r['truncation_error']:.4f}, int16 {r['int16_error']:.4f}, "
                  f"FP32 vs FP64 {r['fp32_vs_fp64']:.1e}" for r in rows)
snap = info['snapshot'].rstrip('/').split('/')[-1]
head = f"""// Computed for real on USC CARC (Slurm job {info['slurm_job']}, discovery cluster, host {info['host']}) with the code of "Language Model
// Activations Inhabit Privileged Error-Correcting Basins" (Finlayson et al.): ~/manifold-attractors at commit {info['commit']}, driver
// ~/thesis-slides/probe_face.py (reuses manifold_attractors.maps.bubble_coords / bubble_forward / bubble_geometry), exported by
// ~/thesis-slides/export_probe_face.py. Model {info['model']} @ {snap[:12]}, {info['dtype']} (torch {info['torch']}, transformers {info['transformers']}).
// The same sphere probe as assets/probe-sphere.js (the paper's bubble single sample 0: x = hidden_states[15] of "{info['record']['token']}", the paper's
// {info['points']}-point Fibonacci lattice on the sphere of radius |x| through x spanned by x and the bubble's tangents e1, e2; injected at layer 15
// with the paper's fast KV-cache forward), here in float64 and kept in full D = 4096. Point order = the lattice order of probe-sphere.js.
// Per layer l (15 = the injected sphere, ..., {LAST}; the run covers 15-{info['layers'][-1]['layer']}) the raw residual outputs Y (points x D) ~= C B, B orthonormal (r x D): Gram-Schmidt of
// [mean direction, top centered principal directions], with enough components that every point's residual |Y_k - C_k B| <= 0.5% of
// |F_l(x)| (the activation's norm at that layer); then truncated to the leading `rank` components for which the slide's front-face
// projection (all points onto the span of one face's three output points) moves by <= {100 * TOL:.0f}% of the panel extent (max over points,
// 95th percentile over {FACES} random faces). Per layer: rank, that truncation error, the int16 rounding error (same measure), and the max
// abs difference of the paper's activation-tangent-frame projection between this FP64 run and the FP32 run of probe-sphere.js (normalized):
{table}
// coords = C / |F_l(x)|, row-major (point, component), base64 little-endian int16 q with C = q * scale[component].
// Any orthogonal projection onto the span of output points is computed from C (B is orthonormal, so inner products are preserved).
"""
dest = OUT / 'probe-face.js'
dest.write_text(head + 'window.PROBE_FACE = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('wrote', dest, dest.stat().st_size)
