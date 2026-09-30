# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Render raw single-block great-circle speed (residual units per radian).

Consumes frozen great_circle_speed_grid data. No model evaluation.
Shares the manuscript's named font sizes and native textwidth with the norm-growth grid.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from .figure_style import (
    FIGURE_DIR,
    PAPER, MODELS, LATEX_SIZES,
    TEXT_WIDTH_TEX_POINTS, TEX_POINTS_PER_INCH, configure_style,
    SAMPLE_CURVES, OUTSIDE_CIRCLE_ALPHA, file_hash, font_size,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator


BLOCK_COUNT = 8
FIGURE_HEIGHT_INCHES = 2.0


def draw_sample(ax, angles, speeds, selected, color):
    """Highlight one randomly chosen curve, retaining the gray sample backdrop."""
    segments = [
        (angles <= -180, OUTSIDE_CIRCLE_ALPHA),
        ((angles >= -180) & (angles <= 180), 1.0),
        (angles >= 180, OUTSIDE_CIRCLE_ALPHA),
    ]
    for mask, opacity in segments:
        for speed in speeds[:SAMPLE_CURVES]:
            ax.plot(angles[mask], speed[mask], color='.45',
                    alpha=.25 * opacity, linewidth=.3)
        ax.plot(angles[mask], speeds[selected, mask], color=color,
                alpha=opacity, linewidth=.65)


def plot(data_dir, figure_path, style_path, seed):
    style_path = style_path.resolve()
    manuscript, use_tex = configure_style(style_path)
    fig, axes = plt.subplots(3, BLOCK_COUNT,
        figsize=(TEXT_WIDTH_TEX_POINTS / TEX_POINTS_PER_INCH, FIGURE_HEIGHT_INCHES), sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, wspace=0)
    rng = np.random.default_rng(seed)
    selected_samples = {}
    for row, (key, label, color) in enumerate(MODELS):
        with np.load(data_dir / f'{key}.npz') as data:
            angles, speeds = data['theta'], data['speeds']
            assert speeds.shape == (8, 20, 401)
            assert np.array_equal(angles, np.arange(-200, 201))
            assert np.isfinite(speeds).all() and (speeds > 0).all()
            selected_samples[key] = rng.integers(speeds.shape[1], size=BLOCK_COUNT).tolist()
            for column, block in enumerate(data['layers']):
                ax = axes[row, column]
                draw_sample(ax, angles, speeds[column], selected_samples[key][column], color)
                ax.set_axisbelow(True)
                ax.set_title(f'Block {block}', pad=2)
                ax.set_xlim(-200, 200)
                ax.set_xticks([-180, 0, 180])
                ax.set_xticks([-90, 90], minor=True)
                ax.grid(True, which='both', color='white', linewidth=.5)
                ax.tick_params(axis='x', which='minor', bottom=False)
                ax.tick_params(labelsize=font_size('tiny'), length=0, pad=1,
                               labelleft=True, labelbottom=(row == 2))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
                ax.yaxis.set_major_formatter(EngFormatter(sep='', places=None))
        axes[row, -1].yaxis.set_label_position('right')
        axes[row, -1].set_ylabel(label, fontsize=font_size('tiny'), labelpad=2, rotation=0, ha='left', va='center')
    fig.supxlabel(r'Angle $\theta$ (degrees)', fontsize=font_size('scriptsize'))
    fig.supylabel('Speed', fontsize=font_size('scriptsize'), rotation=0)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(figure_path)
        fig.savefig(figure_path.with_suffix('.png'), dpi=300)
    plt.close(fig)
    provenance = dict(script_sha256=file_hash(__file__),
        shared_style_script_sha256=file_hash(Path(__file__).with_name('great_circle_norm_grid.py')),
        style=str(style_path), style_sha256=file_hash(style_path), matplotlib=matplotlib.__version__,
        text_usetex=use_tex, manuscript_sha256=file_hash(manuscript), latex_sizes_pt=LATEX_SIZES,
        font_roles=dict(panel_titles='tiny', ticks='tiny', axis_labels='scriptsize'),
        width_tex_pt=TEXT_WIDTH_TEX_POINTS, quantity='raw output-vector speed per radian, not radius-normalized',
        curves='all 20 samples in gray; one uniformly random sample per panel in color; periodic extensions faded',
        selection_seed=seed, selected_samples_zero_based=selected_samples,
        data_sha256={key: file_hash(data_dir / f'{key}.npz') for key, _, _ in MODELS},
        records_sha256={key: file_hash(data_dir / f'{key}.json') for key, _, _ in MODELS})
    figure_path.with_suffix('.render.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PAPER / 'data/great_circle_speed_grid')
    p.add_argument('--figure', type=Path, default=FIGURE_DIR / "great_circle_speed_grid.pdf")
    p.add_argument('--style', type=Path, default=PAPER / 'styles/matplotlibrc')
    p.add_argument('--seed', type=int, default=20260910, help='Random highlighted-sample selection')
    a = p.parse_args()
    plot(a.data, a.figure, a.style, a.seed)
