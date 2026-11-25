"""FP32, sharded version of `trisphere.py --index 2 --end 32` (blocks 16-32), for when the FP64 run is too slow: the same sphere, map and
area metric, evaluated in float32 on CPU shared nodes, on the slide's own grid (every 2nd latitude/longitude of the paper's 1.25-degree
grid) instead of the paper's full grid.

Probe: FixedProbe's construction (maps/fixed_probe.py) with float32 weights and math instead of float64 (no enable_float64_math):
[EOS, EOS] context run once through the model with forward pre-hooks capturing every block's input and kwargs; a point is injected as
position 1 of block 16's input and blocks 16..L are applied to both positions. The full 32-block model is loaded.
Area: the paper's intrinsic area ratio exactly as in maps/geometry.surface_area_map / extend_centroid_surface_map (orthonormal sphere
tangents t1 = d/dlon, t2 = d/dlat on the paper's frozen frame, on-sphere central differences r (cos d n +- sin d t), divided by the INPUT
arc length 2 r d, area = sqrt(g11) |J2 - (g12/g11) J1|), at d = .05 and .025 degrees (the paper's convergence check).
FP32 check: the same code at depth 6 reproduces the paper's frozen FP64 areas at 5 cells; at depth 17 the FP32 speeds |dF/dtheta| at a
few points of the segments are compared with the FP64 ones of trisphere_circle.py.
Usage: --shard i --shards k (compute) | --merge --shards k
"""
import argparse, json, os, platform, subprocess
from pathlib import Path
import numpy as np

DATA = Path.home() / 'manifold-attractors/data/triple_area_heatmaps_6blocks'
CIRCLE = Path.home() / 'thesis-slides/out/trisphere_circle'
OUT = Path.home() / 'thesis-slides/out/trisphere_fp32'
INDEX, END, STRIDE, STEP = 2, 32, 2, .05


def chart(basis, lat, lon):
    pole, first, second = basis
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    meridian = np.cos(lon)[:, None] * pole + np.sin(lon)[:, None] * first
    normal = np.cos(lat)[:, None] * meridian + np.sin(lat)[:, None] * second
    t1 = -np.sin(lon)[:, None] * pole + np.cos(lon)[:, None] * first
    t2 = -np.sin(lat)[:, None] * meridian + np.cos(lat)[:, None] * second
    return normal, t1, t2


class Probe32:
    """FixedProbe (maps/fixed_probe.py) in float32."""
    def __init__(self, snapshot, start=15, batch=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.start, self.batch = torch, start, batch
        tok = AutoTokenizer.from_pretrained(snapshot)
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32, device_map='cpu', attn_implementation='eager').eval().requires_grad_(False)
        self.layers = self.model.model.layers; self.captured = {}
        prefix = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        ids = torch.tensor([[prefix, prefix]])
        def capture(index):
            def hook(module, args, kwargs): self.captured[index] = (args[0].clone(), kwargs.copy())
            return hook
        hooks = [l.register_forward_pre_hook(capture(i), with_kwargs=True) for i, l in enumerate(self.layers) if i >= start]
        try:
            with torch.inference_mode(): self.model(ids, use_cache=False)
        finally:
            for h in hooks: h.remove()

    def forward(self, points, depth):
        torch, outs = self.torch, []
        with torch.inference_mode():
            for off in range(0, len(points), self.batch):
                b = torch.as_tensor(points[off:off + self.batch], dtype=torch.float32)
                hidden = self.captured[self.start][0].expand(len(b), -1, -1).clone(); hidden[:, 1] = b
                for index in range(self.start, self.start + depth):
                    v = self.layers[index](hidden, **self.captured[index][1]); hidden = v[0] if isinstance(v, tuple) else v
                outs.append(hidden[:, 1].double().numpy())
        return np.concatenate(outs)


def area(evaluate, radius, normal, t1, t2, step):
    d = np.deg2rad(step); cols = []
    for t in (t1, t2):
        cols.append((evaluate(radius * (np.cos(d) * normal + np.sin(d) * t)) - evaluate(radius * (np.cos(d) * normal - np.sin(d) * t))) / (2 * radius * d))
    j1, j2 = cols; g11 = (j1 * j1).sum(1); g12 = (j1 * j2).sum(1)
    return np.sqrt(g11) * np.linalg.norm(j2 - (g12 / g11)[:, None] * j1, axis=1)


def compute(shard, shards):
    import torch
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
    meta = json.loads((DATA / 'inputs.json').read_text())
    probe = Probe32(meta['snapshot'])
    with np.load(DATA / f'map-{INDEX:02d}.npz') as f: z = {k: f[k] for k in f.files}
    basis, radius = z['basis'], float(z['sphere_radius'])
    with np.load(DATA / 'inputs.npz') as f: states = f['states'][3 * INDEX:3 * INDEX + 3].astype(np.float64)
    info = {}
    if shard == 0:   # FP32 vs the paper's FP64: frozen areas (blocks 16-21) and the FP64 segment speeds (blocks 16-32)
        cells = [(72, 144), (36, 100), (108, 200), (10, 30), (140, 260)]
        n, t1, t2 = chart(basis, z['latitude'][[c[0] for c in cells]], z['longitude'][[c[1] for c in cells]])
        fresh = area(lambda p: probe.forward(p, 6), radius, n, t1, t2, STEP); frozen = np.array([z['area_ratio'][c] for c in cells])
        c = np.load(CIRCLE / 'circle.npz'); U = states / np.linalg.norm(states, axis=1, keepdims=True); sp32, sp64 = [], []
        for s, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
            ui, uj = U[i], U[j]; w = uj - (uj @ ui) * ui; w /= np.linalg.norm(w)
            x = c[f'x_{s}']; pick = np.searchsorted(x, [-0.5, 0, 0.5, 1, 1.5]); th = x[pick] * c['totals'][s]; dd = np.deg2rad(STEP)
            circ = lambda a: radius * (np.cos(a)[:, None] * ui + np.sin(a)[:, None] * w)
            sp32 += list(np.linalg.norm(probe.forward(circ(th + dd), 17) - probe.forward(circ(th - dd), 17), axis=1) / (2 * dd)); sp64 += list(c[f'arc_32_{s}'][pick])
        sp32, sp64 = np.array(sp32), np.array(sp64)
        info.update(fp32_vs_frozen_fp64_area_16_21=dict(frozen=frozen.tolist(), fp32=fresh.tolist(), max_relative=float(np.max(abs(fresh - frozen) / frozen))),
                    fp32_vs_fp64_speed_16_32=dict(fp64=sp64.tolist(), fp32=sp32.tolist(), max_relative=float(np.max(abs(sp32 - sp64) / sp64))))
        print(json.dumps(info), flush=True)
    lat, lon = z['latitude'][::STRIDE], z['longitude'][::STRIDE]
    LAT, LON = np.meshgrid(lat, lon, indexing='ij'); idx = np.arange(LAT.size)[shard::shards]
    n, t1, t2 = chart(basis, LAT.ravel()[idx], LON.ravel()[idx])
    ev = lambda p: probe.forward(p, END - 15)
    a1, a2 = area(ev, radius, n, t1, t2, STEP), area(ev, radius, n, t1, t2, STEP / 2)
    FS = ev(radius * n).astype(np.float32)
    FA = ev(radius * states / np.linalg.norm(states, axis=1, keepdims=True)) if shard == 0 else np.zeros((0, states.shape[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'shard-{shard}.npz', index=idx, area=a1, area_half=a2, FS=FS, FA=FA)
    (OUT / f'shard-{shard}.json').write_text(json.dumps(dict(info, host=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'), torch=torch.__version__)) + '\n')
    print('saved shard', shard, flush=True)


def merge(shards):
    zs = [np.load(OUT / f'shard-{i}.npz') for i in range(shards)]; js = [json.loads((OUT / f'shard-{i}.json').read_text()) for i in range(shards)]
    idx = np.concatenate([q['index'] for q in zs]); order = np.argsort(idx); assert (idx[order] == np.arange(len(idx))).all()
    a1 = np.concatenate([q['area'] for q in zs])[order]; a2 = np.concatenate([q['area_half'] for q in zs])[order]
    FS = np.concatenate([q['FS'] for q in zs])[order].astype(np.float64); FA = zs[0]['FA']
    rel = np.abs(a1 - a2) / a2
    assert np.quantile(rel, .99) < .01 and rel.max() < .05, 'area finite differences not converged'
    with np.load(DATA / f'map-{INDEX:02d}.npz') as f: z = {k: f[k] for k in f.files}
    with np.load(DATA / 'inputs.npz') as f: states = f['states'][3 * INDEX:3 * INDEX + 3].astype(np.float64)
    lat, lon = z['latitude'][::STRIDE], z['longitude'][::STRIDE]
    shape = (len(lat), len(lon)); A = a1.reshape(shape)
    seam = float(np.max(np.abs(A[:, 0] - A[:, -1]) / A[:, 0])); poles = max(float(np.ptp(r) / np.mean(r)) for r in (A[0], A[-1]))
    un = lambda v: v / np.linalg.norm(v)
    b1 = un(FA.sum(0)); b2 = un(FA[0] - (FA[0] @ b1) * b1); b3 = un(FA[2] - (FA[2] @ b1) * b1 - (FA[2] @ b2) * b2)
    frame = np.stack([b1, b2, b3]); proj, proj_act = FS @ frame.T, FA @ frame.T
    captured = float(np.mean(np.linalg.norm(proj, axis=1) ** 2 / np.linalg.norm(FS, axis=1) ** 2))
    base = json.loads((Path.home() / 'thesis-slides/out/trisphere/map-02.json').read_text())
    name = f'map-{INDEX:02d}-end{END}-fp32'   # (the FP64 run, trisphere.py --end 32, writes map-02-end32)
    np.savez_compressed(Path.home() / f'thesis-slides/out/trisphere/{name}.npz', latitude=lat, longitude=lon, proj=proj.reshape(shape + (3,)),
                        proj_act=proj_act, area_ratio=A, area_min=A.min(), area_max=A.max(), marker_latitude=z['marker_latitude'],
                        marker_longitude=z['marker_longitude'], act_norms=np.linalg.norm(states, axis=1), radius=float(z['sphere_radius']),
                        out_norms=np.linalg.norm(FA, axis=1))
    commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    info = dict(base, end=END, dtype='float32', commit=commit, frame_variance_captured_mean=captured, fp32=True,
                area_checks=dict(grid='slide grid (every 2nd of the paper\'s 1.25-degree grid)', convergence_p99=float(np.quantile(rel, .99)),
                                 convergence_max=float(rel.max()), pole_relative_variation=poles, seam_relative_max=seam),
                fp32_checks={k: v for k, v in js[0].items() if k.startswith('fp32')}, slurm_job=','.join(j['slurm_job'] for j in js),
                host=','.join(sorted({j['host'] for j in js})), num_hidden_layers_loaded=32)
    (Path.home() / f'thesis-slides/out/trisphere/{name}.json').write_text(json.dumps(info, indent=1, default=float) + '\n')
    print(json.dumps(info['area_checks']), json.dumps(info['fp32_checks']), 'captured', captured, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int); p.add_argument('--shards', type=int, required=True); p.add_argument('--merge', action='store_true')
    a = p.parse_args()
    merge(a.shards) if a.merge else compute(a.shard, a.shards)
