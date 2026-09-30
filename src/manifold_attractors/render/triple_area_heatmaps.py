# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Six OLMo random-triplet expansion maps through blocks 16–21, in Mollweide projection.

Frozen intrinsic output/input area ratios; projection affects display only.
Each panel has an independent logarithmic color range. No model inference.
"""

import argparse
from pathlib import Path

import numpy as np
from .figure_style import FIGURE_DIR, PAPER, TEXT_WIDTH_INCHES, configure_style
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def cell_edges(centers):
    """Keep boundary samples as half cells rather than extending beyond the sphere."""
    return np.r_[centers[0], (centers[:-1] + centers[1:]) / 2, centers[-1]]


def render(data_dir, output):
    configure_style(PAPER / 'styles/matplotlibrc')
    fig, axes = plt.subplots(2, 3, figsize=((TEXT_WIDTH_INCHES - .1) / 2, 1.2),
                             subplot_kw={'projection': 'mollweide'}, layout='compressed')
    fig.set_constrained_layout_pads(w_pad=.005, h_pad=.005, wspace=.035, hspace=0)
    for index, ax in enumerate(axes.flat):
        with np.load(data_dir / f'map-{index:02d}.npz') as data:
            area = data['area_ratio']
            assert area.shape == (145, 289) and np.isfinite(area).all() and (area > 0).all()
            colors = np.log10(area)
            mesh = ax.pcolormesh(
                np.deg2rad(cell_edges(data['longitude'])),
                np.deg2rad(cell_edges(data['latitude'])), colors,
                shading='flat', cmap='magma', norm=Normalize(colors.min(), colors.max()),
                rasterized=True, edgecolors='none', antialiased=False)
            np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(area.shape), colors)
            assert len(data['marker_longitude']) == len(data['marker_latitude']) == 3
            ax.plot(np.deg2rad(data['marker_longitude']), np.deg2rad(data['marker_latitude']),
                    linestyle='none', marker='+', markersize=3.8, markeredgewidth=.4,
                    color='white', zorder=4)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['geo'].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(output, dpi=600)
        fig.savefig(output.with_suffix('.png'), dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PAPER / 'data/triple_area_heatmaps_6blocks')
    parser.add_argument('--output', '--figure', type=Path, default=FIGURE_DIR / "triple_area_heatmaps.pdf")
    args = parser.parse_args()
    render(args.data, args.output)
