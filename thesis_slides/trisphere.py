"""Output-space image of the paper's triple sphere probe (Fig. "Sphere probe expansion maps for six randomly sampled sets of 3
activations", fig:tripleheatmap, data/triple_area_heatmaps_6blocks), for the slides.

Sphere: the paper's frozen frame of map-0k.npz (maps.area.compute('triple') -> geometry.surface_area_map(center_kind='centroid') +
extend_centroid_surface_map): three native FP32 input-block-16 OLMo-3-7B activations A, B, C (inputs.npz states[3k:3k+3]); pole =
normalized raw A+B+C, first = A's direction orthogonal to the pole, second = C's Gram-Schmidt residual; radius = |B|. Chart point
r (cos lat (cos lon pole + sin lon first) + sin lat second) on the paper's 145 x 289 grid (lat -90..90, lon -180..180, 1.25 deg),
sampled here every STRIDE-th latitude/longitude.
Evaluator: the paper's maps.fixed_probe.FixedProbe (OLMo-3-1025-7B at the frozen revision, FP64 with enable_float64_math, [EOS, EOS]
context, point injected at position 1 into block 16, raw output of block 21 -- the paper's six blocks 16..21), on CPU.
Only override (no repo edit): the model is loaded with num_hidden_layers=22 (layer_types truncated to match), so the 10 later blocks
(never used by FixedProbe.forward, which stops at block 21) are not materialized -- the full FP64 model would not fit a 64 GB node.
Validation: the frozen area at a few grid cells is recomputed with the paper's metric and compared (proves the same map).
Output frame (display only; the paper plots this probe as a Mollweide heat map, not in 3-D): the triple rule applied to the images
of the activations' sphere points P = r * unit(A, B, C): a1 = normalized F(P_A) + F(P_B) + F(P_C), a2 = F(P_A) orthogonal to a1,
a3 = F(P_C) Gram-Schmidt residual -- the same construction that gives the input chart (pole, first, second) from A, B, C, so the
three activations' images lie exactly in the frame, as the activations lie in the input chart.
Usage: python trisphere.py --index k [--end L] -> out/trisphere/map-0k[-endL].npz/.json
--end L (default 21, the paper's six blocks): map through blocks 16..L instead (FixedProbe depth L - 15, loading L + 1 blocks). The frozen-cell
check still runs at the paper's depth 6 first (same model, same code path); then the area colors are recomputed for the new map with the
paper's own routine geometry.extend_centroid_surface_map(recompute_all=True) on the paper's frozen frame (all 145 x 289 cells, .05 and
.025 degree central differences, its convergence / pole / seam / batch asserts), since the frozen areas are for blocks 16-21 only.
"""
import argparse, json, os, platform, subprocess
from pathlib import Path
import numpy as np

DATA = Path.home() / 'manifold-attractors/data/triple_area_heatmaps_6blocks'
OUT = Path.home() / 'thesis-slides/out/trisphere'
STRIDE = 2


def chart(basis, lat, lon):
    pole, first, second = basis
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    meridian = np.cos(lon)[:, None] * pole + np.sin(lon)[:, None] * first
    normal = np.cos(lat)[:, None] * meridian + np.sin(lat)[:, None] * second
    t1 = -np.sin(lon)[:, None] * pole + np.cos(lon)[:, None] * first
    t2 = -np.sin(lat)[:, None] * meridian + np.cos(lat)[:, None] * second
    return normal, t1, t2


def recompute_cells(z, evaluate, cells, step=.05):
    """The paper's surface_area_map metric (geometry.py) at selected frozen (lat, lon) grid cells."""
    radius = float(z['sphere_radius'])
    normal, t1, t2 = chart(z['basis'], z['latitude'][[c[0] for c in cells]], z['longitude'][[c[1] for c in cells]])
    d = np.deg2rad(step); cols = []
    for t in (t1, t2):
        cols.append((evaluate(radius * (np.cos(d) * normal + np.sin(d) * t)) - evaluate(radius * (np.cos(d) * normal - np.sin(d) * t))) / (2 * radius * d))
    j1, j2 = cols; g11 = (j1 * j1).sum(1); g12 = (j1 * j2).sum(1)
    return np.sqrt(g11) * np.linalg.norm(j2 - (g12 / g11)[:, None] * j1, axis=1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True)
    p.add_argument('--end', type=int, default=21, help='last block of the map (the paper: 21); other ends recompute the area with the paper routine')
    a = p.parse_args()
    import torch, transformers
    from manifold_attractors.maps.fixed_probe import FixedProbe
    from manifold_attractors.maps.geometry import extend_centroid_surface_map
    loaded = min(32, max(22, a.end + 1))
    _from = transformers.AutoModelForCausalLM.from_pretrained.__func__
    def from_pretrained(cls, path, *args, **kwargs):
        cfg = transformers.AutoConfig.from_pretrained(path)
        kwargs['config'] = transformers.AutoConfig.from_pretrained(path, num_hidden_layers=loaded, layer_types=cfg.layer_types[:loaded])
        return _from(cls, path, *args, **kwargs)
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)
    meta = json.loads((DATA / 'inputs.json').read_text()); conf = json.loads((DATA / 'configuration.json').read_text())
    assert conf['kind'] == 'triple' and conf['blocks'] == [16, 17, 18, 19, 20, 21]
    probe = FixedProbe(meta['snapshot'], key='olmo3', depth=6, device='cpu')
    threads = int(os.getenv('SLURM_CPUS_PER_TASK', '8')); torch.set_num_threads(threads)   # FixedProbe pins 4
    assert len(probe.layers) == loaded and probe.start == 15 and probe.depth == 6
    with np.load(DATA / f'map-{a.index:02d}.npz') as f: z = {k: f[k] for k in f.files}
    with np.load(DATA / 'inputs.npz') as f: states = f['states'][3 * a.index:3 * a.index + 3].astype(np.float64)
    basis, radius = z['basis'], float(z['sphere_radius'])
    # the frame is the paper's: rebuild it from the three states exactly as surface_area_map(center_kind='centroid') does
    pole = states.mean(0); pole /= np.linalg.norm(pole)
    first = states[0] - (states[0] @ pole) * pole; first /= np.linalg.norm(first)
    second = states[2] - (states[2] @ pole) * pole - (states[2] @ first) * first; second /= np.linalg.norm(second)
    np.testing.assert_allclose(np.stack([pole, first, second]), basis, atol=1e-10)
    assert abs(radius - np.linalg.norm(states[1])) < 1e-9 * radius
    cells = [(72, 144), (36, 100), (108, 200), (10, 30), (140, 260)]
    fresh = recompute_cells(z, probe.forward, cells)
    frozen = np.array([z['area_ratio'][c] for c in cells])
    rel = np.abs(fresh - frozen) / frozen
    print('frozen-cell check', frozen, fresh, rel, flush=True)
    assert rel.max() < 1e-6, 'CPU recomputation disagrees with the frozen map'
    probe.depth = a.end - 15                                                       # blocks 16..end
    full_checks = None
    if a.end != 21:
        # The paper's routine (geometry.extend_centroid_surface_map, recompute_all=True: every one of the 145 x 289 cells evaluated with the
        # new map, nothing reused) on the paper's frozen frame; it takes the frame from a +-90 degree map, so it gets the frozen map's
        # middle 145 longitudes (only their frame, grid and array shapes are read).
        half = {k: (v[:, 72:217] if k in ('area_ratio', 'area_ratio_half', 'gram', 'sigma_max', 'sigma_min') else v) for k, v in z.items()}
        half['longitude'] = z['longitude'][72:217]
        arrays, full_checks = extend_centroid_surface_map(half, probe.forward, recompute_all=True)
        np.testing.assert_array_equal(arrays['longitude'], z['longitude']); np.testing.assert_array_equal(arrays['latitude'], z['latitude'])
        z = dict(z, area_ratio=arrays['area_ratio'])
        print('area recomputed', json.dumps(full_checks, default=float), flush=True)
    lat, lon = z['latitude'][::STRIDE], z['longitude'][::STRIDE]
    assert lat[-1] == 90 and lon[0] == -180 and lon[-1] == 180
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    normal, _, _ = chart(basis, LAT.ravel(), LON.ravel())
    acts = radius * states / np.linalg.norm(states, axis=1, keepdims=True)        # the activations' points on the sphere
    Y = probe.forward(np.concatenate([acts, radius * normal]))
    FA, FS = Y[:3], Y[3:]
    un = lambda v: v / np.linalg.norm(v)
    a1 = un(FA.sum(0)); a2 = un(FA[0] - (FA[0] @ a1) * a1); a3 = FA[2] - (FA[2] @ a1) * a1 - (FA[2] @ a2) * a2; a3 = un(a3)
    frame = np.stack([a1, a2, a3])
    proj, proj_act = FS @ frame.T, FA @ frame.T
    captured = float(np.mean(np.linalg.norm(proj, axis=1) ** 2 / np.linalg.norm(FS, axis=1) ** 2))
    tok = probe.tokenizer
    recs = meta['records'][3 * a.index:3 * a.index + 3]
    tokens = [tok.decode([r['input_ids'][r['position']]]) for r in recs]
    commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    name = f'map-{a.index:02d}' + ('' if a.end == 21 else f'-end{a.end}')
    np.savez_compressed(OUT / f'{name}.npz', latitude=lat, longitude=lon, proj=proj.reshape(len(lat), len(lon), 3),
                        proj_act=proj_act, area_ratio=z['area_ratio'][::STRIDE, ::STRIDE], area_min=z['area_ratio'].min(),
                        area_max=z['area_ratio'].max(), marker_latitude=z['marker_latitude'], marker_longitude=z['marker_longitude'],
                        act_norms=np.linalg.norm(states, axis=1), radius=radius, out_norms=np.linalg.norm(FA, axis=1))
    info = dict(index=a.index, model=meta['model'], revision=meta['revision'], commit=commit, tokens=tokens,
                titles=[r['title'] for r in recs], positions=[r['position'] for r in recs], sample_ids=[r['sample_id'] for r in recs],
                frozen_cell_check=dict(cells=cells, frozen=frozen.tolist(), recomputed=fresh.tolist(), max_relative=float(rel.max()), blocks='16-21'),
                end=a.end, area_checks=full_checks,
                stride=STRIDE, grid=[len(lat), len(lon)], frame_variance_captured_mean=captured, dtype='float64', device='cpu', threads=threads,
                num_hidden_layers_loaded=loaded, torch=torch.__version__, transformers=transformers.__version__, host=platform.node(),
                slurm_job=os.getenv('SLURM_JOB_ID'))
    (OUT / f'{name}.json').write_text(json.dumps(info, indent=1, default=float) + '\n')
    print(json.dumps(info, default=float), flush=True)


if __name__ == '__main__': main()
