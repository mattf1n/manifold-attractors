"""Layer-by-layer evolution of the paper's single-activation sphere probe (2-D sphere), with the paper's expansion (area) measure.

Sphere: the paper's bubble protocol (src/manifold_attractors/maps/bubble_run.py + bubble_coords.py, Fig. "Projections of 2D spheres"),
single sample 0: Llama-3.1-8B float32, x = hidden_states[15] of "University" (data/bubble_sources.json records[0]), tangents
e1, e2 from torch.manual_seed(4242) + bubble_coords.tangent_axes_np, sphere points X_k = |x| (FIB_k @ [e1, e2, xhat]) on the
paper's 4000-point Fibonacci lattice (bubble_geometry.fib_lattice), injected as the last position of a [BOS, BOS] context with
the paper's fast KV-cache forward (bubble_forward.make_forward, inject_layer = 15). Raw residual outputs of every later block
(layers 16..32, no final norm) are captured with forward hooks; layer 15 is the injected sphere itself.
Expansion: the paper's intrinsic area ratio (maps/geometry.surface_area_map): at each lattice point n, orthonormal sphere tangents
t1, t2; central perturbations kept on the sphere, |x| (cos d n +- sin d t_i); J_i = (F_l(+) - F_l(-)) / (2 |x| d) (divided by the
INPUT arc length); area = sqrt(g11) |J2 - (g12/g11) J1| = sqrt(det J^T J). d = 0.1 degrees (FP32); a subset is re-evaluated
at 0.05 degrees to report the finite-difference agreement.
Projection: the paper's avg_image_frame with the single frame point x (stencils +-ep e1, +-ep e2): a1 = unit F_l(x),
a2 = unit J_l e1, a3 = unit J_l e2 (Gram-Schmidt). Coordinates from the origin.
Sharded over lattice points: --shard i --shards k writes out/probe_sphere/shard-i.npz; --merge combines them.
"""
import argparse, json, os, platform, subprocess
from pathlib import Path
import numpy as np

OUT = Path.home() / 'thesis-slides/out/probe_sphere'
STEP, CHECK_STEP, CHECK_EVERY = 0.1, 0.05, 20


def tangents(n):
    """Orthonormal tangents at unit vectors n (3-D, lattice coordinates)."""
    a = np.zeros_like(n); a[np.arange(len(n)), np.argmin(np.abs(n), 1)] = 1
    t1 = a - (a * n).sum(1, keepdims=True) * n; t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(n, t1)


def compute(shard, shards):
    import torch, transformers
    from manifold_attractors.core.inputs import model_snapshot
    from manifold_attractors.maps import bubble_coords as b
    from manifold_attractors.maps.bubble_forward import make_forward
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
    src = model_snapshot('llama3')
    tok = transformers.AutoTokenizer.from_pretrained(src)
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, device_map='cpu',
                                                              attn_implementation='eager').eval()
    model.requires_grad_(False)
    rec = json.loads((Path.home() / 'manifold-attractors/data/bubble_sources.json').read_text())['records'][0]
    assert rec['token'] == 'University'
    with torch.inference_mode():
        x = model(torch.tensor([rec['input_ids']]), output_hidden_states=True,
                  use_cache=False).hidden_states[b.INJECT][0, rec['position']].float().numpy()
    D = x.shape[0]
    torch.manual_seed(4242 + 0); raw = torch.randn(D, 2).numpy()          # bubble_run.compute, single sample i = 0
    e1, e2 = b.tangent_axes_np(x, raw[:, 0], raw[:, 1])
    R = float(np.linalg.norm(x)); basis = np.stack([e1, e2, x / R])        # lattice coordinates -> R^D (as bubble_coords)
    fib = b.FIB.astype(np.float64)
    mine = np.arange(len(fib))[shard::shards]
    n = fib[mine]; t1, t2 = tangents(n)
    def ring(d, idx):
        d = np.deg2rad(d); m = n[idx]; s1, s2 = t1[idx], t2[idx]
        return [np.cos(d) * m + sgn * np.sin(d) * t for t in (s1, s2) for sgn in (1, -1)]   # +t1, -t1, +t2, -t2
    check = np.arange(len(mine))[::CHECK_EVERY]
    pts3 = [n] + ring(STEP, slice(None)) + ring(CHECK_STEP, check)
    pts = np.concatenate([R * (p @ basis) for p in pts3])
    ep = 1e-3 * R + 1e-6
    frame = np.stack([x, x + ep * e1, x - ep * e1, x + ep * e2, x - ep * e2])
    H = torch.tensor(np.concatenate([frame, pts]), dtype=torch.float32)
    bos = tok.bos_token_id or tok.eos_token_id
    layers = list(range(b.INJECT, len(model.model.layers) + 1))
    F = make_forward(model, torch.tensor([[bos, bos]]), b.INJECT, b.INJECT, fast=True)
    outs = {l: [] for l in layers}
    bs = 256
    for s in range(0, len(H), bs):
        cap = {}
        hooks = [model.model.layers[l - 1].register_forward_hook(
                    lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1].float().clone()))
                 for l in layers[1:]]
        try:
            with torch.inference_mode():
                F(H[s:s + bs])
        finally:
            for h in hooks: h.remove()
        cap[b.INJECT] = H[s:s + bs].float().clone()   # the injected points (see probe_layers.py on hidden_states[15])
        for l in layers: outs[l].append(cap[l].numpy())
        print(f'shard {shard}: batch {s // bs + 1}/{-(-len(H) // bs)}', flush=True)
    N, C = len(mine), len(check)
    proj, area, area_check, frames = [], [], [], []
    for l in layers:
        Y = np.concatenate(outs[l]).astype(np.float64); outs[l] = None
        FP, FD, Y = Y[:1], Y[1:5][None], Y[5:]
        a1, a2, a3 = b.avg_image_frame(FP, FD, ep)
        A = np.stack([a1, a2, a3], 1)
        def areas(block, d):
            j1 = (block[0] - block[1]) / (2 * R * np.deg2rad(d)); j2 = (block[2] - block[3]) / (2 * R * np.deg2rad(d))
            g11, g12 = (j1 * j1).sum(1), (j1 * j2).sum(1)
            return np.sqrt(g11) * np.linalg.norm(j2 - (g12 / g11)[:, None] * j1, axis=1)
        st = Y[N:5 * N].reshape(4, N, D); ck = Y[5 * N:].reshape(4, C, D)
        proj.append(Y[:N] @ A); area.append(areas(st, STEP)); area_check.append(areas(ck, CHECK_STEP))
        frames.append(dict(act=(FP[0] @ A).tolist(), norm=float(np.linalg.norm(FP[0]))))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'shard-{shard}.npz', index=mine, check=mine[check], proj=np.array(proj), area=np.array(area),
                        area_check=np.array(area_check), layers=np.array(layers))
    (OUT / f'shard-{shard}.json').write_text(json.dumps(dict(frames=frames, R=R, ep=ep, record=rec, snapshot=src,
        torch=torch.__version__, transformers=transformers.__version__, host=platform.node(),
        slurm_job=os.getenv('SLURM_JOB_ID')), indent=1) + '\n')
    print('saved shard', shard, flush=True)


def merge(shards):
    zs = [np.load(OUT / f'shard-{i}.npz') for i in range(shards)]
    js = [json.loads((OUT / f'shard-{i}.json').read_text()) for i in range(shards)]
    for j in js[1:]:
        assert np.allclose([f['norm'] for f in j['frames']], [f['norm'] for f in js[0]['frames']], rtol=1e-5)
    idx = np.concatenate([z['index'] for z in zs]); order = np.argsort(idx); assert (idx[order] == np.arange(len(idx))).all()
    proj = np.concatenate([z['proj'] for z in zs], 1)[:, order]; area = np.concatenate([z['area'] for z in zs], 1)[:, order]
    check = np.concatenate([z['check'] for z in zs]); area_check = np.concatenate([z['area_check'] for z in zs], 1)
    rel = np.abs(area[:, check] - area_check) / area_check
    layers = zs[0]['layers']; R = js[0]['R']
    assert np.allclose(area[0], 1, atol=1e-3), 'identity map must have area ratio 1 at the injection layer'
    assert np.allclose(np.linalg.norm(proj[0], axis=1), R, rtol=1e-3)
    from scipy.spatial import ConvexHull
    from manifold_attractors.maps.bubble_geometry import fib_lattice
    fib = fib_lattice(len(idx)).astype(np.float64)
    faces = ConvexHull(fib).simplices
    commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    np.savez_compressed(OUT / 'probe_sphere.npz', layers=layers, proj=proj, area=area, faces=faces, fib=fib,
                        act=np.array([f['act'] for f in js[0]['frames']]), norms=np.array([f['norm'] for f in js[0]['frames']]))
    info = dict(model='meta-llama/Llama-3.1-8B', snapshot=js[0]['snapshot'], dtype='float32', commit=commit, record=js[0]['record'],
                R=R, ep=js[0]['ep'], step_degrees=STEP, check_step_degrees=CHECK_STEP, check_points=int(len(check)),
                check_rel_median=float(np.median(rel)), check_rel_p99=float(np.quantile(rel, .99)), check_rel_max=float(rel.max()),
                points=int(len(idx)), faces=int(len(faces)), layers=layers.tolist(), torch=js[0]['torch'],
                transformers=js[0]['transformers'], jobs=[j['slurm_job'] for j in js], hosts=sorted({j['host'] for j in js}))
    (OUT / 'probe_sphere.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k.startswith('check')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int); p.add_argument('--shards', type=int, required=True); p.add_argument('--merge', action='store_true')
    a = p.parse_args()
    merge(a.shards) if a.merge else compute(a.shard, a.shards)
