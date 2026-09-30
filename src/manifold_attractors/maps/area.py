"""Fresh triple (six-block current figure) and language (four-block) heatmaps."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import surface_area_map, extend_centroid_surface_map
from .inputs import prepare, save


def compute(kind, output, blocks=6, device='cuda'):
    from .fixed_probe import FixedProbe
    output = Path(output)
    meta = json.loads((output/'inputs.json').read_text())
    with np.load(output/'inputs.npz') as z:
        states = z['states'].astype(np.float64)
        if kind == 'area':
            d1 = z['centroid_en'] - z['centroid_zh']
            d2 = z['centroid_zh'] - z['centroid_el']
    probe = FixedProbe(meta['snapshot'], key='olmo3' if kind=='triple' else 'llama3',
                       depth=blocks if kind=='triple' else 4, device=device)
    for index in range(6):
        if kind == 'triple':
            defining = states[3*index:3*index+3]
            arrays, checks = surface_area_map(defining, probe.forward, grid_size=145, center_kind='centroid')
            # Recompute every full-sphere point, never seed from a previous map.
            arrays, full = extend_centroid_surface_map(arrays, probe.forward, recompute_all=True)
            checks.update(full)
        else:
            x = states[index]
            arrays, checks = surface_area_map(np.stack([x+d1, x, x+d2]), probe.forward, grid_size=145)
            for key in ['longitude','latitude','offplane']:
                original = key if key=='offplane' else 'marker_'+key
                arrays['frame_construction_'+key] = arrays.pop(original)
            arrays['marker_longitude'] = np.array([0.]); arrays['marker_latitude'] = np.array([0.])
        checks['full_hook_max_abs'] = probe.audit(states[index*3 if kind=='triple' else index])
        np.savez_compressed(output/f'map-{index:02d}.npz', **arrays)
        save(output/f'map-{index:02d}.json', checks)
    save(output/'configuration.json', dict(kind=kind, blocks=list(range(16,16+(blocks if kind=='triple' else 4))),
         grid=[145,289] if kind=='triple' else [145,145], derivative_degrees=.05,
         fresh_recomputation=True, input_metadata=meta))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['triple','area'])
    p.add_argument('stage', choices=['prepare','compute','all'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blocks',type=int,choices=[4,6],default=6,help='triple only; current paper uses six')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.stage in ['prepare','all']: prepare(a.kind,a.output,a.device)
    if a.stage in ['compute','all']: compute(a.kind,a.output,a.blocks,a.device)

if __name__=='__main__': main()
