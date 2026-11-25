"""Output-space speed along the trisphere slide's three chord segments, for several block lengths (16 -> 19, 21, 24, 28, 32).

Triple: the paper's fig:tripleheatmap panel 3 (data/triple_area_heatmaps_6blocks, inputs.npz states[6:9] = A, B, C; OLMo-3-1025-7B,
native FP32 input-block-16 activations), the sphere of radius |B| (the paper's frame, map-02.npz). Segments A->B, B->C, C->A between
the activations' sphere points r * unit(X): the slide's 'chord' path, p(t) = r * unit((1 - t) u_i + t u_j) (u = unit directions), i.e.
the great-circle arc, at angle theta(t) from u_i; t runs over [T0, T1] = [-0.5, 1.5]: the segment (t in [0, 1]) and the same straight
line continued half a segment past each activation (projected onto the sphere the same way), every DT = 0.005.
Evaluator: the paper's maps.fixed_probe.FixedProbe (FP64, enable_float64_math, [EOS, EOS] context, injected at position 1 into block 16),
full 32-block model on CPU. Its forward loop is run once with every block's output captured (the same loop as FixedProbe.forward;
checked equal to FixedProbe.forward at depth 6 and 17).
Speed: central differences in theta along the arc (step H degrees, checked against H/2): |dF/dtheta| per radian (the paper's speed-
profile convention: raw output-vector derivative per radian, not radius-normalized), and per unit chord length (what the slide's
constant-chord-speed dot shows): |dF/dtheta| * dtheta/dt / (r |u_j - u_i|).
Output: out/trisphere_speed/speed.npz/.json
"""
import json, os, platform, subprocess
from pathlib import Path
import numpy as np

DATA = Path.home() / 'manifold-attractors/data/triple_area_heatmaps_6blocks'
OUT = Path.home() / 'thesis-slides/out/trisphere_speed'
INDEX, H = 2, 0.05
T0, T1, DT = -0.5, 1.5, 0.005
NS = int(round((T1 - T0) / DT)) + 1
I0, I1 = int(round(-T0 / DT)), int(round((1 - T0) / DT))   # the samples at t = 0 and t = 1 (the activations)
ENDS = [19, 21, 24, 28, 32]


def main():
    import torch
    from manifold_attractors.maps.fixed_probe import FixedProbe
    meta = json.loads((DATA / 'inputs.json').read_text())
    probe = FixedProbe(meta['snapshot'], key='olmo3', depth=17, device='cpu')
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
    assert len(probe.layers) == 32 and probe.start == 15
    with np.load(DATA / 'inputs.npz') as f: states = f['states'][3 * INDEX:3 * INDEX + 3].astype(np.float64)
    with np.load(DATA / f'map-{INDEX:02d}.npz') as f: radius = float(f['sphere_radius']); basis = f['basis']
    assert abs(radius - np.linalg.norm(states[1])) < 1e-9 * radius
    U = states / np.linalg.norm(states, axis=1, keepdims=True)

    def forward_all(points):
        """FixedProbe.forward's loop, keeping each block's output (block index l -> residual after block l, 1-based)."""
        outs = {l: [] for l in ENDS}
        with torch.inference_mode():
            for off in range(0, len(points), probe.batch):
                batch = torch.as_tensor(points[off:off + probe.batch], dtype=torch.float64)
                hidden = probe.captured[probe.start][0].expand(len(batch), -1, -1).clone(); hidden[:, 1] = batch
                for index in range(probe.start, len(probe.layers)):
                    v = probe.layers[index](hidden, **probe.captured[index][1]); hidden = v[0] if isinstance(v, tuple) else v
                    if index + 1 in outs: outs[index + 1].append(hidden[:, 1].numpy().copy())
        return {l: np.concatenate(v) for l, v in outs.items()}

    test = radius * U
    ref = forward_all(test)
    probe.depth = 6; assert np.array_equal(probe.forward(test), ref[21])
    probe.depth = 17; assert np.array_equal(probe.forward(test), ref[32])

    t = np.round(np.linspace(T0, T1, NS), 12)
    assert t[I0] == 0 and t[I1] == 1
    segs = [(0, 1), (1, 2), (2, 0)]
    pts, meta_seg = [], []
    for i, j in segs:
        ui, uj = U[i], U[j]
        theta_ij = np.arccos(np.clip(ui @ uj, -1, 1))
        w = uj - (uj @ ui) * ui; w /= np.linalg.norm(w)          # in-plane unit tangent at u_i toward u_j
        q = (1 - t)[:, None] * ui + t[:, None] * uj
        th = np.arctan2(q @ w, q @ ui)                            # theta(t): angle from u_i along the arc
        circ = lambda a: radius * (np.cos(a)[:, None] * ui + np.sin(a)[:, None] * w)
        for h in (H, H / 2):
            d = np.deg2rad(h); pts += [circ(th + d), circ(th - d)]
        pts.append(circ(th))
        x, y = q @ ui, q @ w; dx, dy = (uj - ui) @ ui, (uj - ui) @ w
        dth_dt = (x * dy - y * dx) / (x * x + y * y)               # d theta / d t of theta = atan2(y, x)
        meta_seg.append(dict(theta=th, dth_dt=dth_dt, chord=float(np.linalg.norm(uj - ui)), theta_total=float(theta_ij)))
    P = np.concatenate(pts)
    Y = forward_all(P)
    res = {}
    for l in ENDS:
        Yl = Y[l].reshape(len(segs), 5, NS, -1)
        sp, sp2, rel = [], [], []
        for s in range(len(segs)):
            v1 = np.linalg.norm(Yl[s, 0] - Yl[s, 1], axis=1) / (2 * np.deg2rad(H))
            v2 = np.linalg.norm(Yl[s, 2] - Yl[s, 3], axis=1) / (2 * np.deg2rad(H / 2))
            sp.append(v1); sp2.append(v2); rel.append(float(np.max(np.abs(v1 - v2) / v2)))
        res[l] = dict(arc=np.array(sp), fd_rel_max=max(rel))
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = dict(t=t, ends=np.array(ENDS), theta=np.array([m['theta'] for m in meta_seg]), dth_dt=np.array([m['dth_dt'] for m in meta_seg]),
                  chord=np.array([m['chord'] for m in meta_seg]), radius=radius)
    table = {}
    for l in ENDS:
        arc = res[l]['arc']; chord = arc * arrays['dth_dt'] / (radius * arrays['chord'][:, None])
        arrays[f'arc_{l}'] = arc; arrays[f'chord_{l}'] = chord
        mid = (I0 + I1) // 2
        def ratios(v): return [float(v[s, mid] / ((v[s, I0] + v[s, I1]) / 2)) for s in range(3)]
        table[l] = dict(blocks=f'16-{l}', arc_mid_over_end=ratios(arc), chord_mid_over_end=ratios(chord),
                        arc_max_over_end=[float(arc[s, I0:I1 + 1].max() / ((arc[s, I0] + arc[s, I1]) / 2)) for s in range(3)],
                        arc_end=[[float(arc[s, I0]), float(arc[s, I1])] for s in range(3)], arc_mid=[float(arc[s, mid]) for s in range(3)],
                        fd_rel_max=res[l]['fd_rel_max'])
    np.savez_compressed(OUT / 'speed.npz', **arrays)
    commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    info = dict(index=INDEX, segments=segs, samples=NS, t_range=[T0, T1], t_step=DT, t_index_of_activations=[I0, I1], step_degrees=H,
                table=table, commit=commit, host=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'),
                torch=torch.__version__)
    (OUT / 'speed.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps(table, indent=1), flush=True)


if __name__ == '__main__': main()
