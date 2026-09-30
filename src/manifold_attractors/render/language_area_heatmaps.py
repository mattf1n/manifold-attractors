# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Six frozen Llama 3.1 language expansion maps; no model inference."""

import argparse
from pathlib import Path

import numpy as np
from .figure_style import FIGURE_DIR, PAPER, TEXT_WIDTH_TEX_POINTS, TEX_POINTS_PER_INCH, configure_style, font_size
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def render(data_dir, output):
    configure_style(PAPER / 'styles/matplotlibrc')
    width = TEXT_WIDTH_TEX_POINTS / TEX_POINTS_PER_INCH
    # Leave vertical room for square panels plus the headings and axis labels.
    height = width / 6 + .4
    fig, axes = plt.subplots(1, 6, figsize=(width, height),
                             layout='compressed')
    fig.set_constrained_layout_pads(w_pad=.012, h_pad=.015, wspace=.035, hspace=.03)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - .20 / height))
    for index, ax in enumerate(axes):
        with np.load(data_dir / f'map-{index:02d}.npz') as data:
            longitude, latitude = data['longitude'], data['latitude']
            x = np.flatnonzero((longitude >= -45) & (longitude <= 45))
            y = np.flatnonzero((latitude >= -45) & (latitude <= 45))
            np.testing.assert_array_equal(longitude[x[[0, -1]]], [-45, 45])
            np.testing.assert_array_equal(latitude[y[[0, -1]]], [-45, 45])
            area = data['area_ratio'][np.ix_(y, x)]
            assert np.isfinite(area).all() and (area > 0).all()
            # Frozen tangents follow en-zh and zh-el. Negate both for en->zh
            # and en->el: projecting out the anchor and first tangent removes
            # zh-en, so the remaining el-en component equals that of el-zh.
            colors = np.log10(area)[::-1, ::-1]
            ax.imshow(colors, origin='lower', extent=[-45, 45, -45, 45],
                      cmap='magma', norm=Normalize(colors.min(), colors.max()),
                      interpolation='nearest', aspect='equal', rasterized=True)
            assert len(data['marker_longitude']) == len(data['marker_latitude']) == 1
            ax.plot(-data['marker_longitude'], -data['marker_latitude'],
                    linestyle='none', marker='+', markersize=3.8,
                    markeredgewidth=.4, color='white', zorder=4)
        ax.set(xlim=(-45, 45), ylim=(-45, 45), xticks=[-45, 0, 45], yticks=[-45, 0, 45])
        ax.set_box_aspect(1)
        ax.grid(False)
        ax.tick_params(labelsize=font_size('tiny'), length=0, pad=1,
                       labelleft=index == 0, labelbottom=True)
        if index == 0:
            # Positions in the frozen first map; reflect labels and endpoints
            # along with the data: English left, Chinese right, Greek above.
            basins = [
                ('English', (2.5, 1.25), (20, 22)),
                ('Chinese', (-20, 0), (-22, 30)),
                ('Greek', (-3.75, -32.5), (20, -30)),
            ]
            for label, minimum, text_position in basins:
                ax.annotate(label, xy=np.negative(minimum), xytext=np.negative(text_position),
                            ha='center', va='center', color='white',
                            fontsize=font_size('tiny'),
                            arrowprops={'arrowstyle': '-', 'color': 'white', 'lw': .5},
                            zorder=5)
    axes[0].set_ylabel(r'\texttt{en}\textendash\texttt{el}', fontsize=font_size('scriptsize'),
                       rotation=0, ha='right', va='center')
    fig.supxlabel(r'\texttt{en}\textendash\texttt{zh}', fontsize=font_size('scriptsize'))
    fig.canvas.draw()
    for pair, label in enumerate(['English', 'Chinese', 'Greek']):
        left, right = axes[2 * pair].get_position(), axes[2 * pair + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, max(left.y1, right.y1) + .055 / height,
                 label, ha='center', va='bottom', fontsize=font_size('scriptsize'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(output, dpi=600)
        fig.savefig(output.with_suffix('.png'), dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PAPER / 'data/language_area_heatmaps')
    parser.add_argument('--output', '--figure', type=Path, default=FIGURE_DIR / "language_area_heatmaps.pdf")
    args = parser.parse_args()
    render(args.data, args.output)
