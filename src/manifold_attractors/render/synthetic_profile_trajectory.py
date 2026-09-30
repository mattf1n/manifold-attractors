# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Norm-growth profiles around propagated synthetic states; no model inference."""

import argparse
from pathlib import Path

import numpy as np
from .figure_style import TEXT_WIDTH_INCHES, FIGURE_DIR, PAPER, configure_style, draw_profiles, font_size
import matplotlib.pyplot as plt

BLOCKS = [13, 14, 15, 18, 20, 24, 26, 28]
CLASSES = [
    ('shuffled.npz', 'Shuffled', '#a34b9b'),
    ('llama3.npz', 'Norm-matched', '#31688e'),
]


def render(data_dir, output):
    configure_style(PAPER / 'styles/matplotlibrc')
    profiles = []
    for filename, _, _ in CLASSES:
        with np.load(data_dir / filename) as data:
            indices = [int(np.flatnonzero(data['blocks'] == block)[0]) for block in BLOCKS]
            profiles.append(data['ratios'][indices])
            theta = data['theta']
            np.testing.assert_array_equal(theta, np.arange(-200, 201))
    means = np.stack(profiles).mean(axis=2)
    fig, axes = plt.subplots(2, 8, figsize=(TEXT_WIDTH_INCHES, 1.2), sharex=True, sharey='row',
                             layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, wspace=0)
    for row, ((_, label, color), ratios) in enumerate(zip(CLASSES, profiles)):
        low, high = min(1., float(means[row].min())), max(1., float(means[row].max()))
        pad = max(.07 * (high - low), 1e-5)
        for column, (block, ax) in enumerate(zip(BLOCKS, axes[row])):
            draw_profiles(ax, theta, ratios[column], color)
            ax.set_ylim(low - pad, high + pad)
            ax.set_xlim(-200, 200)
            ax.set_xticks([-180, 0, 180])
            ax.set_xticks([-90, 90], minor=True)
            ax.set_axisbelow(True)
            ax.grid(True, color='white')
            ax.grid(True, axis='x', which='minor', color='white')
            ax.tick_params(axis='x', which='minor', bottom=False)
            ax.tick_params(labelsize=font_size('tiny'), length=0, pad=1,
                           labelleft=column == 0, labelbottom=row == len(CLASSES) - 1)
            ax.locator_params(axis='y', nbins=3)
            ticks = ax.get_yticks()
            ticks = ticks[(ticks >= low - pad) & (ticks <= high + pad) & ~np.isclose(ticks, 1.)]
            ax.set_yticks(np.sort(np.r_[ticks, 1.]))
            if row == 0:
                ax.set_title(f'Block {block}', fontsize=font_size('scriptsize'), pad=3)
        axes[row, -1].yaxis.set_label_position('right')
        axes[row, -1].set_ylabel(label, fontsize=font_size('tiny'), labelpad=3, rotation=0, ha='left', va='center')
    fig.supylabel('Norm\ngrowth', fontsize=font_size('scriptsize'), rotation=0)
    fig.supxlabel(r'Angle $\theta$ (degrees)', fontsize=font_size('scriptsize'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(output)
        fig.savefig(output.with_suffix('.png'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PAPER / 'data/synthetic_profile_trajectory')
    parser.add_argument('--output', '--figure', type=Path, default=FIGURE_DIR / "synthetic_profile_trajectory.pdf")
    args = parser.parse_args()
    render(args.data, args.output)
