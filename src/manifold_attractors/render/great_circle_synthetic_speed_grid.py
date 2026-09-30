# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Midpoint natural/synthetic raw speed: means of EXACT norm-grid samples.

Colored lines average all 20 samples; all 20 shown in gray. Shared y range
per model includes every rendered sample and mean (no clipping). Speed is per
radian, not divided by input radius. Input speed itself equals that radius.
Consumes frozen great_circle_synthetic_speed_grid data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from .figure_style import (
    FIGURE_DIR,
    PAPER, MODELS, TEXT_WIDTH_TEX_POINTS, TEX_POINTS_PER_INCH,
    configure_style, draw_profiles, font_size, file_hash, SAMPLE_CURVES,
)
from .great_circle_synthetic_grid import CLASSES
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def plot(data_dir, figure_path, style_path):
    manuscript, use_tex = configure_style(style_path.resolve())
    fig, axes = plt.subplots(3, 4, sharex=True, sharey='row',
        figsize=(TEXT_WIDTH_TEX_POINTS / TEX_POINTS_PER_INCH, 2.8), layout='constrained')
    ranges = {}
    for row, (key, label, _) in enumerate(MODELS):
        with np.load(data_dir / f'{key}.npz') as data:
            angles, speeds, block = data['theta'], data['speeds'], int(data['block'])
            assert speeds.shape == (4, 20, 401)
            assert np.array_equal(angles, np.arange(-200, 201))
            assert np.isfinite(speeds).all() and (speeds > 0).all()
            shown = np.concatenate([speeds[:, :SAMPLE_CURVES].reshape(-1, 401), speeds.mean(axis=1)])
            low, high = float(shown.min()), float(shown.max())
            pad = .06 * (high-low)
            ranges[key] = dict(ylim=[max(0, low-pad), high+pad],
                              all_sample_range=[float(speeds.min()), float(speeds.max())])
            for column, (title, color) in enumerate(CLASSES):
                ax = axes[row, column]
                draw_profiles(ax, angles, speeds[column], color)
                ax.set_xlim(-200, 200)
                ax.set_ylim(*ranges[key]['ylim'])
                ax.set_axisbelow(True)
                ax.set_xticks([-180, 0, 180])
                ax.set_xticks([-90, 90], minor=True)
                ax.grid(True, which='both', color='white', linewidth=.5)
                ax.tick_params(axis='x', which='minor', bottom=False)
                ax.tick_params(labelsize=font_size('tiny'), length=0, pad=1,
                    labelleft=(column == 0), labelbottom=(row == 2))
                ax.locator_params(axis='y', nbins=3)
                ax.yaxis.set_major_formatter(EngFormatter(sep=''))
                if row == 0:
                    ax.set_title(title, fontsize=font_size('scriptsize'), pad=3)
            axes[row, -1].yaxis.set_label_position('right')
            axes[row, -1].set_ylabel(f'{label}\nBlock {block}', fontsize=font_size('tiny'), labelpad=3, rotation=0, ha='left', va='center')
    fig.supxlabel(r'Angle $\theta$ (degrees)', fontsize=font_size('scriptsize'))
    fig.supylabel('Speed', fontsize=font_size('scriptsize'), rotation=0)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(figure_path)
        fig.savefig(figure_path.with_suffix('.png'), dpi=300)
    provenance = dict(script_sha256=file_hash(__file__),
        helper_sha256=file_hash(Path(__file__).with_name('great_circle_norm_grid.py')),
        class_style_sha256=file_hash(Path(__file__).with_name('great_circle_synthetic_grid.py')),
        style_sha256=file_hash(style_path), manuscript_sha256=file_hash(manuscript), text_usetex=use_tex,
        width_tex_pt=TEXT_WIDTH_TEX_POINTS,
        font_roles=dict(titles='scriptsize', axis_labels='scriptsize', ticks_and_rows='tiny'),
        data_sha256={key: file_hash(data_dir / f'{key}.npz') for key, _, _ in MODELS},
        records_sha256={key: file_hash(data_dir / f'{key}.json') for key, _, _ in MODELS},
        primary_curves='arithmetic mean of all 20 samples; all 20 individual profiles in gray',
        quantity='raw output-vector speed per radian; not input-radius normalized',
        ranges=ranges, y_limits='Shared per model; all rendered samples/means, padded 6%, floor 0; no clipping')
    figure_path.with_suffix('.render.json').write_text(json.dumps(provenance, indent=2)+'\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PAPER / 'data/great_circle_synthetic_speed_grid')
    p.add_argument('--figure', type=Path, default=FIGURE_DIR / "great_circle_synthetic_speed_grid.pdf")
    p.add_argument('--style', type=Path, default=PAPER / 'styles/matplotlibrc')
    a = p.parse_args()
    plot(a.data, a.figure, a.style)
