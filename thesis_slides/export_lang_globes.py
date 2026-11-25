"""Export the six full-sphere language expansion maps (out/lang_globes, from lang_globes.py) to the slide asset assets/lang-globes.js.
Checks that each map's frozen hemisphere is bit-identical to the paper's data/language_area_heatmaps/map-0k.npz.
Values: log10 of the area ratio, quantized to uint16 over each map's own full-sphere [min, max] (base64, little-endian)."""
import base64, json, subprocess
from pathlib import Path
import numpy as np

REPO = Path.home() / 'manifold-attractors'
DATA = REPO / 'data/language_area_heatmaps'
OUT = Path.home() / 'thesis-slides/out/lang_globes'
config = json.loads((DATA / 'configuration.json').read_text()); meta = config['input_metadata']
commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(meta['snapshot'])
maps, checks = [], []
for k, rec in enumerate(meta['records']):
    with np.load(OUT / f'map-{k:02d}.npz') as z, np.load(DATA / f'map-{k:02d}.npz') as f:
        lon, lat, area = z['longitude'], z['latitude'], z['area_ratio']
        np.testing.assert_array_equal(lon, np.linspace(-180, 180, 289)); np.testing.assert_array_equal(lat, np.linspace(-90, 90, 145))
        np.testing.assert_array_equal(area[:, 72:217], f['area_ratio'])           # frozen hemisphere untouched
        np.testing.assert_array_equal(z['basis'], f['basis'])
        crop = f['area_ratio'][36:109, 36:109]                                      # the paper's displayed +-45 deg window
    log = np.log10(area); lo, hi = float(log.min()), float(log.max())
    q = np.round((log - lo) / (hi - lo) * 65535).astype('<u2')
    info = json.loads((OUT / f'map-{k:02d}.json').read_text())
    token = tok.decode([rec['input_ids'][rec['position']]])
    maps.append(dict(language=rec['language'], title=rec['title'], url=rec['url'], document_id=rec['document_id'], position=rec['position'],
                     token=token, log10Min=round(lo, 6), log10Max=round(hi, 6),
                     paperLog10Min=round(float(np.log10(crop.min())), 6), paperLog10Max=round(float(np.log10(crop.max())), 6),
                     values=base64.b64encode(q.tobytes()).decode()))
    checks.append(dict(map=k, convergence_p99=info['convergence_p99'], convergence_max=info['convergence_max'], seam=info['seam_relative_max'],
                       poles=info['pole_relative_variation'], frozen_cell_rel=info['frozen_cell_check']['max_relative']))
    print(k, rec['language'], rec['title'], repr(token), lo, hi, maps[-1]['paperLog10Min'], maps[-1]['paperLog10Max'], checks[-1])
asset = dict(model=meta['model'], revision=meta['revision'], blocks=config['blocks'], hiddenStateIndex=meta['hidden_state_index'],
             grid=dict(lat=dict(min=-90, max=90, count=145), lon=dict(min=-180, max=180, count=289)),
             frame='pole = anchor activation direction; e1 = en-zh centroid difference, e2 = zh-el centroid difference, each projected off the earlier axes',
             maps=maps)
cs = '\n'.join(f'//   map {c["map"]}: FD convergence p99 {c["convergence_p99"]:.1e}, max {c["convergence_max"]:.1e}; seam {c["seam"]:.1e}; poles {c["poles"]:.1e}; frozen-cell recomputation {c["frozen_cell_rel"]:.1e}' for c in checks)
header = f"""// The paper's language expansion maps as full spheres. "Language Model Activations Inhabit Privileged Error-Correcting Basins"
// (Finlayson et al., arXiv 2026), Figure "Activation space expansion along en-zh and en-el tangent directions" (main.tex,
// fig:language_expansion_maps; scripts/fig/language_area_heatmaps.py), ~/manifold-attractors on the USC carc cluster at commit {commit[:12]}.
// Six maps: two anchor activations each from English (en), Chinese (zh) and Greek (el) Wikipedia documents (the paper's records,
// data/language_area_heatmaps/inputs.json; corpus wikimedia/wikipedia 20231101 @ {meta['corpus_revision'][:12]}). Model {meta['model']} @ {meta['revision'][:12]};
// anchor x = residual stream entering block 16 (hidden_states[15]), FP32; probe in FP64 (maps.fixed_probe.FixedProbe, [BOS, BOS] context).
// Sphere: radius |x|, through x; frame (pole, e1, e2) = x/|x|, the en-zh centroid difference and the zh-el centroid difference (layer-16
// Wikipedia centroids), each projected off the earlier axes. Point at (lon, lat): |x| (cos lat (cos lon pole + sin lon e1) + sin lat e2).
// Value: the paper's intrinsic expansion, output/input area ratio sqrt(det J^T J) of blocks 16-19 on the sphere's tangent plane
// (maps.geometry.surface_area_map), stored as log10. The paper displays lon, lat in -45..45 (both axes reflected: en->zh right, en->el up)
// with per-panel min-max magma colors; dark = low expansion (the basins).
// Hemisphere lon in -90..90: the paper's frozen data/language_area_heatmaps/map-0k.npz, bit-identical. Far hemisphere (|lon| > 90): computed
// for these slides with the paper's maps.geometry.extend_centroid_surface_map (same frame, metric and finite-difference checks) on CPU by
// ~/thesis-slides/lang_globes.py (FixedProbe with only blocks 0-19 loaded; FP64 as in the paper); exported by export_lang_globes.py.
// Checks:
{cs}
// values: base64 little-endian uint16, row-major [lat (-90..90, 145 rows)][lon (-180..180, 289 columns)], log10 area =
// log10Min + v / 65535 (log10Max - log10Min). paperLog10Min/Max: the range within the paper's displayed +-45 deg window.
"""
Path(Path.home() / 'thesis-slides/out/lang-globes.js').write_text(header + 'window.LANG_GLOBES = ' + json.dumps(asset, ensure_ascii=False, separators=(',', ':')) + ';\n')
