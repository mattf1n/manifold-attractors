# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Frozen refusal-steering heatmaps at native manuscript width; no model inference.

Rows: benign unsteered, harmful unsteered, benign newline-only steering.
The intervention is at the final assistant-header newline input (predicting the
first response token), not at the first generated token. Its strength was selected
using a speed minimum; displayed white dots instead mark sampled AREA minima.
Run: uv run scripts/fig/refusal_steering_heatmaps.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'refusal-paper-mpl-{os.getuid()}'))

import numpy as np
from .figure_style import (FIGURE_DIR, PAPER, TEXT_WIDTH_TEX_POINTS, TEX_POINTS_PER_INCH,
                          configure_style, font_size)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DATA = PAPER / 'data/refusal_steering_heatmaps'


def tex(text):
    escapes = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '%': r'\%',
               '_': r'\_', '&': r'\&', '#': r'\#', '$': r'\$',
               '^': r'\textasciicircum{}', '~': r'\textasciitilde{}'}
    return ''.join(escapes.get(char, char) for char in text)


def cell_edges(values):
    return np.r_[values[0], (values[:-1]+values[1:])/2, values[-1]]


def minimum_indices(values):
    """Strict local minima, with a flat-bottom midpoint; exclude interval ends."""
    result = []
    index = 1
    while index < len(values)-1:
        if values[index] < values[index-1]:
            stop = index+1
            while stop < len(values) and values[stop] == values[index]:
                stop += 1
            if stop < len(values) and values[stop] > values[index]:
                result.append((index+stop-1)//2)
            index = stop
        else:
            index += 1
    return np.asarray(result, dtype=int)


def render(data_dir, output):
    data_dir, output = Path(data_dir), Path(output)
    selection = json.loads((data_dir/'selection.json').read_text())
    rows = selection['rows']
    configure_style(PAPER/'styles/matplotlibrc')
    columns = 8
    # Fix the canvas width and shorten panels vertically, not proportionally.
    fig = plt.figure(figsize=(TEXT_WIDTH_TEX_POINTS/TEX_POINTS_PER_INCH, 2.15),
                     facecolor='white', layout='compressed')
    # Use physical padding, not gaps proportional to the provisional height.
    fig.get_layout_engine().set(hspace=0, h_pad=0.01)
    grid = fig.add_gridspec(len(rows), columns)
    with plt.rc_context({'axes.unicode_minus': False, 'savefig.bbox': None}):
        for row_index, row in enumerate(rows):
            for column, panel in enumerate(row['panels']):
                with np.load(data_dir/panel['file'], allow_pickle=False) as data:
                    steering = data['x'].copy(); random = data['y'].copy()
                    area = data['area_ratio'].copy()
                    marker = [float(data['marker_y'][panel['marker_index']]),
                              float(data['marker_x'][panel['marker_index']])]
                zero = int(np.argmin(abs(random)));assert abs(random[zero]) < 1e-12
                minima = minimum_indices(area[zero])
                colors = np.log10(area)
                ax = fig.add_subplot(grid[row_index, column])
                mesh = ax.pcolormesh(cell_edges(random), cell_edges(steering), colors.T,
                    shading='flat', cmap='magma', norm=Normalize(colors.min(), colors.max()),
                    rasterized=True, edgecolors='none', antialiased=False)
                ax.set_xlim(row['random_range']);ax.set_ylim(-4, 4)
                ax.set_aspect('auto');ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([-4, 0, 4])
                ax.tick_params(axis='y', left=False, labelleft=column == 0,
                               labelsize=font_size('tiny'), length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                label = panel['token'].strip().replace('\n', r'\n')
                ax.set_title(tex(label), fontsize=font_size('tiny'), pad=2)
                if column == columns-1:
                    ax.yaxis.set_label_position('right')
                    row_label = row['label'].replace('newline-only steering', 'steered')
                    ax.set_ylabel(tex(row_label).replace(': ', ':\n').replace(', ', ',\n'),
                                  rotation=0, ha='left', va='center',
                                  fontsize=font_size('scriptsize'), labelpad=6)
                ax.plot(np.zeros(len(minima)), steering[minima], linestyle='none', marker='o',
                        markersize=1.8, markeredgewidth=0, color='white', zorder=4)
                ax.plot(*marker, linestyle='none', marker='+', markersize=3.8,
                        markeredgewidth=.5, color='white', zorder=5)
                if row_index == 0 and column == 0:
                    upper_minimum = steering[minima].max()
                    annotation = ax.annotate('Refusal basin', xy=(0, upper_minimum),
                                xytext=(-0.35, upper_minimum),
                                textcoords=('axes fraction', 'data'),
                                ha='right', va='center', fontsize=font_size('tiny'),
                                color='black',
                                arrowprops={'arrowstyle': '-', 'lw': .5, 'color': 'black'},
                                zorder=6)
                    # Use the existing left margin without changing subplot spacing.
                    annotation.set_in_layout(False)
        xlabel = fig.supxlabel('Random direction $\\leftrightarrow$', fontsize=font_size('scriptsize'))
        ylabel = fig.supylabel('Refusal\ndirection $\\uparrow$', fontsize=font_size('scriptsize'), rotation=0)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=600)
        fig.savefig(output.with_suffix('.png'), dpi=600)
    plt.close(fig)
    print(output)
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', '--figure', type=Path, default=FIGURE_DIR / "refusal_steering_heatmaps.pdf")
    args = parser.parse_args()
    render(args.data, args.output)
