"""Export out/trisphere/map-0k[-endL].{npz,json} (trisphere.py) and, for the same triple, the segment speeds of
out/trisphere_speed/speed.{npz,json} (trisphere_speed.py) to assets/tri-sphere.js for the slides.
Usage: python export_trisphere.py --index k [--end L]"""
import argparse, base64, json
from pathlib import Path
import numpy as np
OUT = Path.home() / 'thesis-slides/out/trisphere'
SPEED = Path.home() / 'thesis-slides/out/trisphere_speed'     # trisphere_speed.py: midpoint/endpoint table for several block lengths
CIRCLE = Path.home() / 'thesis-slides/out/trisphere_circle'   # trisphere_circle.py: the speeds the slide plots
p = argparse.ArgumentParser(); p.add_argument('--index', type=int, default=0); p.add_argument('--end', type=int, default=21)
p.add_argument('--fp32', action='store_true', help='use the FP32 sharded run (trisphere_fp32.py) instead of trisphere.py --end'); a = p.parse_args()
name = f'map-{a.index:02d}' + ('' if a.end == 21 else f'-end{a.end}') + ('-fp32' if a.fp32 else '')
z = np.load(OUT / f'{name}.npz'); info = json.loads((OUT / f'{name}.json').read_text())
assert info.get('end', 21) == a.end and bool(info.get('fp32')) == a.fp32
COORD, AREA = 30000, 1000
b64 = lambda v, dtype: base64.b64encode(np.ascontiguousarray(v).astype(dtype).tobytes()).decode()
lat, lon, proj, act = z['latitude'], z['longitude'], z['proj'], z['proj_act']
scale = max(np.linalg.norm(proj, axis=2).max(), np.linalg.norm(act, axis=1).max())   # the farthest output point -> 1
assert np.allclose(np.diff(lat), lat[1] - lat[0]) and np.allclose(np.diff(lon), lon[1] - lon[0])
q = np.rint(proj / scale * COORD); assert np.abs(q).max() < 32767
log_area = np.log10(z['area_ratio']); la = np.rint(log_area * AREA); assert np.abs(la).max() < 32767
mlat, mlon = np.deg2rad(z['marker_latitude']), np.deg2rad(z['marker_longitude'])
dirs = np.stack([np.cos(mlat) * np.cos(mlon), np.cos(mlat) * np.sin(mlon), np.sin(mlat)], 1)   # the chart's (pole, first, second) coordinates
# the outputs at the activations' sphere points, vs. the grid interpolation the slide does there (reported below)
def bilinear(la_, lo_):
    i = (la_ - lat[0]) / (lat[1] - lat[0]); j = (lo_ - lon[0]) / (lon[1] - lon[0])
    i0, j0 = min(int(i), len(lat) - 2), min(int(j), len(lon) - 2); fi, fj = i - i0, j - j0
    return (proj[i0, j0] * (1 - fi) * (1 - fj) + proj[i0 + 1, j0] * fi * (1 - fj) + proj[i0, j0 + 1] * (1 - fi) * fj + proj[i0 + 1, j0 + 1] * fi * fj)
interp_err = [float(np.linalg.norm(bilinear(z['marker_latitude'][k], z['marker_longitude'][k]) - act[k]) / np.linalg.norm(act[k])) for k in range(3)]
r6 = lambda v: [round(float(x), 6) for x in v]
sig = lambda v: [float(f'{float(x):.5g}') for x in v]
data = dict(model=info['model'], revision=info['revision'], blocks=[16, a.end], index=a.index, tokens=info['tokens'], titles=info['titles'],
            grid=dict(lat=dict(min=float(lat[0]), max=float(lat[-1]), count=len(lat)), lon=dict(min=float(lon[0]), max=float(lon[-1]), count=len(lon))),
            coordScale=COORD, areaScale=AREA, colorRange=[round(float(np.log10(z['area_min'])), 6), round(float(np.log10(z['area_max'])), 6)],
            activations=[r6(d) for d in dirs], markerLatLon=[[float(z['marker_latitude'][k]), float(z['marker_longitude'][k])] for k in range(3)],
            images=[r6(v / scale) for v in act], proj=b64(q.reshape(-1), '<i2'), logArea=b64(la.reshape(-1), '<i2'))
speed_note = '// (no segment speeds: out/trisphere_circle is for another triple)\n'
cinfo = json.loads((CIRCLE / 'circle.json').read_text())
if cinfo['index'] == a.index:
    c = np.load(CIRCLE / 'circle.npz'); assert a.end in c['ends'].tolist()
    assert [tuple(x) for x in cinfo['segments']] == [(0, 1), (1, 2), (2, 0)]
    xs = [c[f'x_{s}'] for s in range(3)]; DX = cinfo['dx']
    for x in xs: assert np.allclose(np.diff(x), DX) and (x == 0).any() and (x == 1).any()
    # the slide's activations (from the chart) span the same angles as the circle data's segments
    for s, (i, j) in enumerate(cinfo['segments']): assert abs(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)) - c['totals'][s]) < 1e-6
    tab = cinfo['table'][str(a.end)]
    data['segments'] = dict(dx=DX, x0=[float(x[0]) for x in xs], angle=[float(t) for t in c['totals']], arc=[sig(c[f'arc_{a.end}_{s}']) for s in range(3)])
    sinfo = json.loads((SPEED / 'speed.json').read_text()); st = sinfo['table'].get(str(a.end)) if sinfo['index'] == a.index else None
    speed_note = f"""// Segment speeds (Slurm job {cinfo['slurm_job']}, {cinfo['host']}, ~/thesis-slides/trisphere_circle.py, same FixedProbe map, blocks 16-{a.end}):
// segment k runs from activation k (x = 0) to k + 1 mod 3 (x = 1) on the great circle through both (the slide's dot path; x = angle /
// angle[k], angle in radians); samples at x = x0[k] + i dx over the whole circle (x outside 0..1: the same circle continued past the
// activations, computed like the rest, not extrapolated). arc[k] = |dF/dtheta| per radian (the paper's speed-profile convention: raw
// output-vector derivative, not radius-normalized), central differences at {cinfo['step_degrees']} degrees (vs. half that: max relative difference {tab['fd_rel_max']:.1e}).
""" + (f"""// Midpoint over mean-activation speed per segment (trisphere_speed.py, job {sinfo['slurm_job']}): per radian {', '.join(f'{x:.2f}' for x in st['arc_mid_over_end'])};
// per unit chord length (the dot's constant-speed parameter) {', '.join(f'{x:.2f}' for x in st['chord_mid_over_end'])}.
""" if st else '')
if a.end == 21:
    area_note = f"""// Map: the paper's FP64 probe (FixedProbe: [EOS, EOS] context, the point injected at position 1 into block 16, raw output of block 21,
// the six blocks 16-21), on CPU. Recomputing the paper's area at 5 frozen grid cells agrees to {info['frozen_cell_check']['max_relative']:.1e} (relative).
// Color: the paper's frozen log10 area ratio (output/input intrinsic area through blocks 16-21) at the same cells; colorRange = its
// min..max over the paper's full grid (the paper's per-panel Normalize for magma).
"""
elif a.fp32:
    c, f = info['area_checks'], info['fp32_checks']
    area_note = f"""// Map: the paper's probe (maps/fixed_probe.FixedProbe's construction: [EOS, EOS] context, the point injected at position 1 into block 16)
// run through blocks 16-{a.end} (raw output of block {a.end}, no final norm; the paper's figure uses 16-21) in FLOAT32 on CPU
// (~/thesis-slides/trisphere_fp32.py, 4 shards; the paper uses FP64). FP32 check: at the paper's depth (16-21) the FP32 area at 5 frozen
// grid cells matches the paper's FP64 values to {f['fp32_vs_frozen_fp64_area_16_21']['max_relative']:.1e}; at 16-{a.end} the FP32 segment speeds at 15 points match the
// FP64 ones (trisphere_circle.py) to {f['fp32_vs_fp64_speed_16_32']['max_relative']:.1e} (relative).
// Color: log10 area ratio (output/input intrinsic area through blocks 16-{a.end}) computed with the paper's metric (maps.geometry
// surface_area_map: orthonormal sphere tangents on the paper's frozen frame, on-sphere central differences divided by the input arc
// length, sqrt(det J^T J)) at the slide's grid cells, at .05 vs .025 degrees: p99 {c['convergence_p99']:.1e}, max {c['convergence_max']:.1e}; pole
// variation {c['pole_relative_variation']:.1e}, seam {c['seam_relative_max']:.1e}; colorRange = its min..max over these cells (the paper's per-panel Normalize).
"""
else:
    c = info['area_checks']
    area_note = f"""// Map: the paper's FP64 probe (FixedProbe: [EOS, EOS] context, the point injected at position 1 into block 16), here run through blocks
// 16-{a.end} (raw output of block {a.end}, no final norm; the paper's figure uses 16-21) on CPU. At the paper's depth (16-21) the same setup
// reproduces the paper's frozen area at 5 grid cells to {info['frozen_cell_check']['max_relative']:.1e} (relative).
// Color: log10 area ratio (output/input intrinsic area through blocks 16-{a.end}) recomputed with the paper's own routine
// (maps.geometry.extend_centroid_surface_map, recompute_all=True, on the paper's frozen frame: all 145 x 289 cells, central differences
// at .05 vs .025 degrees: p99 {c['convergence_p99']:.1e}, max {c['convergence_max']:.1e}; pole variation {c['pole_relative_variation']:.1e}, seam {c['seam_relative_max']:.1e}),
// at the same cells; colorRange = its min..max over the full grid (the paper's per-panel Normalize for magma).
"""
drv = '~/thesis-slides/trisphere_fp32.py (merge)' if a.fp32 else f'~/thesis-slides/trisphere.py --index {a.index} --end {a.end}'
head = f"""// Computed for real on USC CARC (Slurm job {info['slurm_job']}, CPU, {info['host']}) with the code of "Language Model Activations
// Inhabit Privileged Error-Correcting Basins" (Finlayson et al.): ~/manifold-attractors at commit {info['commit']}, driver
// {drv} (reuses the paper's FixedProbe construction, maps.geometry and the frozen frame of the paper's
// data/triple_area_heatmaps_6blocks/map-{a.index:02d}.npz), exported by ~/thesis-slides/export_trisphere.py --index {a.index} --end {a.end}{' --fp32' if a.fp32 else ''}.
// The paper's Fig. "Sphere probe expansion maps for six randomly sampled sets of 3 activations" (fig:tripleheatmap), panel {a.index + 1}:
// {info['model']} @ {info['revision'][:12]}, three native FP32 input-block-16 activations A, B, C of English Wikipedia tokens
// {', '.join(repr(t) for t in info['tokens'])} (articles {'; '.join(info['titles'])}; positions {info['positions']}).
// Sphere: the paper's frame -- pole = normalized A + B + C, first = A orthogonal to the pole, second = C's Gram-Schmidt residual,
// radius |B| -- with chart point r (cos lat (cos lon pole + sin lon first) + sin lat second) on the paper's 1.25-degree grid, every
// {info['stride']}nd latitude/longitude here (grid: lat -90..90, lon -180..180, the last column repeats the first). activations[k] = the
// unit direction of A, B, C in (pole, first, second) (the paper's white crosses; markerLatLon in degrees).
{area_note}// Output frame (display only; the paper shows this probe as a Mollweide heat map): the same rule applied to the images of the
// activations' sphere points F(rA/|A|), F(rB/|B|), F(rC/|C|): a1 = normalized sum, a2 = F(A-point) orthogonal to a1, a3 = F(C-point)'s
// Gram-Schmidt residual. proj = the grid's images in (a1, a2, a3), divided by the farthest one ({scale:.4g} residual units); images[k] = the
// activations' images, same units. The frame holds {100 * info['frame_variance_captured_mean']:.1f}% of the images' squared norm on average.
// Bilinear interpolation of proj at the activations misses their exact images by {max(interp_err):.1e} (relative, worst of three).
{speed_note}// Encoding: proj (lat-major, 3 per vertex) and logArea (lat-major) are base64 little-endian int16 of round(value * coordScale) and
// round(value * areaScale).
"""
dest = OUT / 'tri-sphere.js'
dest.write_text(head + 'window.TRI_SPHERE = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('wrote', dest, dest.stat().st_size, 'interp_err', interp_err, 'captured', info['frame_variance_captured_mean'], 'scale', scale, 'tokens', info['tokens'])
