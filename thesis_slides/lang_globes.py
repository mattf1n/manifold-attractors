"""Full-sphere extension of the paper's six language expansion maps (Fig. language_area_heatmaps), for the slide globes.

The paper's frozen maps (~/manifold-attractors/data/language_area_heatmaps/map-0k.npz, maps.area.compute('area')) cover only
the hemisphere around each anchor (longitude, latitude in -90..90). This script fills the far hemisphere with the paper's own
extension routine, maps.geometry.extend_centroid_surface_map(recompute_all=False), which keeps the frozen frame (basis,
radius) and the frozen hemisphere cells bit-for-bit and evaluates only the missing longitudes (-180..-91.25, 91.25..180) with
the same intrinsic area metric (central on-sphere finite differences at .05 and .025 degrees, convergence/seam/pole checks).
Evaluator: the paper's maps.fixed_probe.FixedProbe (Llama-3.1-8B at the frozen revision, FP64 with enable_float64_math,
[BOS, BOS] context, point injected at position 1 into block 16, raw output of block 19), on CPU.
Only override (no repo edit): the model is loaded with num_hidden_layers=20, so the 12 later blocks (never used by
FixedProbe.forward, which stops at block 19) are not materialized -- the full FP64 model would not fit a 64 GB node.
Blocks 0..19 are the checkpoint's, so the captured layer-16..19 inputs are identical to the full model's.
Validation: the frozen-hemisphere area at a few grid cells is recomputed with the same metric and compared.
Usage: python lang_globes.py --index k   (k = 0..5) -> out/lang_globes/map-0k.npz/.json
"""
import argparse, json, os
from pathlib import Path
import numpy as np

DATA = Path.home() / 'manifold-attractors/data/language_area_heatmaps'
OUT = Path.home() / 'thesis-slides/out/lang_globes'


def recompute_cells(original, evaluate, cells, step=.05):
    """The paper's surface_area_map metric (geometry.py) at selected frozen (lat, lon) grid cells."""
    pole, first, second = original['basis']; radius = float(original['sphere_radius'])
    lat = np.deg2rad(original['latitude'][[c[0] for c in cells]]); lon = np.deg2rad(original['longitude'][[c[1] for c in cells]])
    meridian = np.cos(lon)[:, None] * pole + np.sin(lon)[:, None] * first
    normal = np.cos(lat)[:, None] * meridian + np.sin(lat)[:, None] * second
    t1 = -np.sin(lon)[:, None] * pole + np.cos(lon)[:, None] * first
    t2 = -np.sin(lat)[:, None] * meridian + np.cos(lat)[:, None] * second
    d = np.deg2rad(step); cols = []
    for t in (t1, t2):
        cols.append((evaluate(radius * (np.cos(d) * normal + np.sin(d) * t)) - evaluate(radius * (np.cos(d) * normal - np.sin(d) * t))) / (2 * radius * d))
    j1, j2 = cols; g11 = (j1 * j1).sum(1); g12 = (j1 * j2).sum(1)
    return np.sqrt(g11) * np.linalg.norm(j2 - (g12 / g11)[:, None] * j1, axis=1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True); a = p.parse_args()
    import torch, transformers
    from manifold_attractors.maps.fixed_probe import FixedProbe
    from manifold_attractors.maps.geometry import extend_centroid_surface_map
    _from = transformers.AutoModelForCausalLM.from_pretrained.__func__
    def from_pretrained(cls, path, *args, **kwargs):
        kwargs['config'] = transformers.AutoConfig.from_pretrained(path, num_hidden_layers=20)
        return _from(cls, path, *args, **kwargs)
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)
    meta = json.loads((DATA / 'configuration.json').read_text())
    probe = FixedProbe(meta['input_metadata']['snapshot'], key='llama3', depth=4, device='cpu')
    threads = int(os.getenv('SLURM_CPUS_PER_TASK', '8')); torch.set_num_threads(threads)   # FixedProbe pins 4
    assert len(probe.layers) == 20 and probe.start == 15 and probe.depth == 4
    with np.load(DATA / f'map-{a.index:02d}.npz') as z: original = {k: z[k] for k in z.files}
    cells = [(72, 72), (36, 36), (108, 100), (10, 130), (140, 5)]
    fresh = recompute_cells(original, probe.forward, cells)
    frozen = np.array([original['area_ratio'][c] for c in cells])
    rel = np.abs(fresh - frozen) / frozen
    print('frozen-cell check', frozen, fresh, rel, flush=True)
    assert rel.max() < 1e-6, 'CPU recomputation disagrees with the frozen hemisphere'
    arrays, checks = extend_centroid_surface_map(original, probe.forward, recompute_all=False)
    checks.update(frozen_cell_check=dict(cells=cells, frozen=frozen.tolist(), recomputed=fresh.tolist(), max_relative=float(rel.max())),
                  source=str(DATA / f'map-{a.index:02d}.npz'), device='cpu', threads=threads, num_hidden_layers_loaded=20)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'map-{a.index:02d}.npz', **arrays)
    (OUT / f'map-{a.index:02d}.json').write_text(json.dumps(checks, indent=1, default=float))
    print(json.dumps(checks, default=float), flush=True)


if __name__ == '__main__': main()
