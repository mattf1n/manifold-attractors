"""Output-space speed along the whole great circle through each pair of the trisphere slide's activations (for the speed plot's
segment and its continuation past both activations), for block lengths 16 -> 21 and 16 -> 32.

Triple: the paper's fig:tripleheatmap panel 3 (data/triple_area_heatmaps_6blocks, inputs.npz states[6:9] = A, B, C; OLMo-3-1025-7B,
native FP32 input-block-16 activations), the sphere of radius r = |B| (the paper's frame, map-02.npz). Segment k runs from u_k to u_k+1
(u = unit activation directions, k + 1 mod 3) on the great circle through both: p(theta) = r (cos theta u_k + sin theta w), w = the unit
tangent at u_k toward u_k+1, theta_k = angle(u_k, u_k+1). Samples at x = theta / theta_k = 0, +-DX, +-2 DX, ... over the whole circle
(theta from theta_k / 2 - 180 to theta_k / 2 + 180 degrees): x = 0 and x = 1 are the two activations; x outside [0, 1] is the same
circle (the slide's chord path, projected onto the sphere, continued) past them.
Evaluator: the paper's maps.fixed_probe.FixedProbe (FP64, enable_float64_math, [EOS, EOS] context, injected at position 1 into block 16),
full 32-block model on CPU, its forward loop run once with the outputs of blocks 21 and 32 kept (checked equal to FixedProbe.forward).
Speed: |dF/dtheta| per radian (the paper's speed-profile convention: raw output-vector derivative, not radius-normalized), central
differences in theta at H degrees, checked against H/2.
Output: out/trisphere_circle/circle.npz/.json
"""
import json, os, platform, subprocess
from pathlib import Path
import numpy as np

DATA = Path.home() / 'manifold-attractors/data/triple_area_heatmaps_6blocks'
OUT = Path.home() / 'thesis-slides/out/trisphere_circle'
INDEX, H, DX = 2, 0.05, 0.005
ENDS = [21, 32]


def main():
    import torch
    from manifold_attractors.maps.fixed_probe import FixedProbe
    meta = json.loads((DATA / 'inputs.json').read_text())
    probe = FixedProbe(meta['snapshot'], key='olmo3', depth=17, device='cpu')
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
    assert len(probe.layers) == 32 and probe.start == 15
    with np.load(DATA / 'inputs.npz') as f: states = f['states'][3 * INDEX:3 * INDEX + 3].astype(np.float64)
    with np.load(DATA / f'map-{INDEX:02d}.npz') as f: radius = float(f['sphere_radius'])
    assert abs(radius - np.linalg.norm(states[1])) < 1e-9 * radius
    U = states / np.linalg.norm(states, axis=1, keepdims=True)

    def forward_all(points):
        """FixedProbe.forward's loop, keeping the outputs of the blocks in ENDS."""
        outs = {l: [] for l in ENDS}
        with torch.inference_mode():
            for off in range(0, len(points), probe.batch):
                batch = torch.as_tensor(points[off:off + probe.batch], dtype=torch.float64)
                hidden = probe.captured[probe.start][0].expand(len(batch), -1, -1).clone(); hidden[:, 1] = batch
                for index in range(probe.start, len(probe.layers)):
                    v = probe.layers[index](hidden, **probe.captured[index][1]); hidden = v[0] if isinstance(v, tuple) else v
                    if index + 1 in outs: outs[index + 1].append(hidden[:, 1].numpy().copy())
                print(f'{off + len(batch)}/{len(points)}', flush=True)
        return {l: np.concatenate(v) for l, v in outs.items()}

    test = radius * U
    ref = forward_all(test)
    probe.depth = 6; assert np.array_equal(probe.forward(test), ref[21])
    probe.depth = 17; assert np.array_equal(probe.forward(test), ref[32])

    segs = [(0, 1), (1, 2), (2, 0)]
    pts, xs, thetas, totals, counts = [], [], [], [], []
    for i, j in segs:
        ui, uj = U[i], U[j]
        total = float(np.arccos(np.clip(ui @ uj, -1, 1)))
        w = uj - (uj @ ui) * ui; w /= np.linalg.norm(w)
        half = np.pi / total                                       # half a circle, in segment lengths
        k = np.arange(int(np.ceil((0.5 - half) / DX)), int(np.floor((0.5 + half) / DX)) + 1)
        x = np.round(k * DX, 12); th = x * total
        assert (x == 0).any() and (x == 1).any()
        circ = lambda a: radius * (np.cos(a)[:, None] * ui + np.sin(a)[:, None] * w)
        np.testing.assert_allclose(circ(np.array([total]))[0], radius * uj, atol=1e-9 * radius)
        for h in (H, H / 2):
            d = np.deg2rad(h); pts += [circ(th + d), circ(th - d)]
        xs.append(x); thetas.append(th); totals.append(total); counts.append(len(x))
    Y = forward_all(np.concatenate(pts))
    arrays, table = dict(radius=radius, ends=np.array(ENDS), totals=np.array(totals)), {}
    for s in range(3): arrays[f'x_{s}'] = xs[s]; arrays[f'theta_{s}'] = thetas[s]
    for l in ENDS:
        off, rel, tab = 0, [], []
        for s in range(3):
            n = counts[s]; B = Y[l][off:off + 4 * n].reshape(4, n, -1); off += 4 * n
            v1 = np.linalg.norm(B[0] - B[1], axis=1) / (2 * np.deg2rad(H)); v2 = np.linalg.norm(B[2] - B[3], axis=1) / (2 * np.deg2rad(H / 2))
            rel.append(float(np.max(np.abs(v1 - v2) / v2))); arrays[f'arc_{l}_{s}'] = v1
            x = xs[s]; i0, i1, im = int(np.argmin(abs(x))), int(np.argmin(abs(x - 1))), int(np.argmin(abs(x - .5)))
            tab.append(dict(at_start=float(v1[i0]), at_end=float(v1[i1]), at_mid=float(v1[im]), x_range=[float(x[0]), float(x[-1])]))
        table[l] = dict(blocks=f'16-{l}', segments=tab, fd_rel_max=max(rel))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'circle.npz', **arrays)
    commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    info = dict(index=INDEX, segments=segs, dx=DX, step_degrees=H, totals_degrees=[float(np.degrees(t)) for t in totals], samples=counts, table=table,
                commit=commit, host=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'), torch=torch.__version__)
    (OUT / 'circle.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps(info, indent=1), flush=True)


if __name__ == '__main__': main()
