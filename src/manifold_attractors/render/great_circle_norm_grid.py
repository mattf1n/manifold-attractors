# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Plot single-block norm growth from precomputed NPZ files (no GPU needed).

From paper/: uv run scripts/fig/great_circle_norm_grid.py
Each model's NPZ contains theta (angles), layers (block numbers), and
ratios (blocks × samples × angles). Data generation lives in
the great_circle_norm_grid data generator (not imported by this renderer).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("pgf")
import matplotlib.pyplot as plt


from .figure_style import (
    FIGURE_DIR,
    PAPER, MODELS, LATEX_SIZES, TEX_POINTS_PER_INCH, TEXT_WIDTH_TEX_POINTS,
    configure_style, draw_profiles, file_hash, font_size,
)

BLOCK_COUNT = 8
FIGURE_HEIGHT_INCHES = 2.25


def format_panel(ax, block, bottom_row):
    """Use 90° grid spacing, with angle labels only on the bottom row."""
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.axhline(1, color=".8", linewidth=0.3, zorder=1)
    ax.set_title(f"Block {block}", pad=2)
    ax.set_xlim(-200, 200)
    ax.set_xticks([-180, 0, 180])
    ax.set_xticks([-90, 90], minor=True)
    ax.grid(True, axis="x", which="minor")
    ax.tick_params(axis="x", which="minor", bottom=False)
    # Keep numerical y labels: each panel has its own vertical scale.
    ax.tick_params(
        labelsize=font_size("tiny"), length=0, pad=1,
        labelleft=True, labelbottom=bottom_row,
    )
    ax.locator_params(axis="y", nbins=2)


def save_figure(fig, figure_path, data_dir, style_path, manuscript, use_tex):
    """Save PDF, PNG preview, and provenance without altering computation records."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the selected manuscript textwidth. Include this PDF at natural size in LaTeX;
    # tight cropping or inclusion scaling would change the physical font sizes.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(figure_path)
        fig.savefig(figure_path.with_suffix(".png"), dpi=300)

    provenance = {
        "script_sha256": file_hash(__file__),
        "style": str(style_path),
        "style_sha256": file_hash(style_path),
        "matplotlib": matplotlib.__version__,
        "text_usetex": use_tex,
        "manuscript_sha256": file_hash(manuscript),
        "latex_sizes_pt": LATEX_SIZES,
        "font_roles": {"panel_titles": "tiny", "ticks": "tiny", "axis_labels": "scriptsize"},
        "width_tex_pt": TEXT_WIDTH_TEX_POINTS,
        "data_sha256": {
            key: file_hash(data_dir / f"{key}.npz") for key, _, _ in MODELS
        },
    }
    figure_path.with_suffix(".render.json").write_text(json.dumps(provenance, indent=2) + "\n")


def plot(data_dir, figure_path, style_path):
    style_path = style_path.resolve()
    manuscript, use_tex = configure_style(style_path)
    fig, axes = plt.subplots(
        len(MODELS), BLOCK_COUNT,
        figsize=(TEXT_WIDTH_TEX_POINTS / TEX_POINTS_PER_INCH, FIGURE_HEIGHT_INCHES),
        sharex=True, layout='constrained',
    )
    fig.get_layout_engine().set(w_pad=0.01, wspace=0)

    for row, (key, label, color) in enumerate(MODELS):
        with np.load(data_dir / f"{key}.npz") as data:
            angles = data["theta"]
            for column, block in enumerate(data["layers"]):
                ax = axes[row, column]
                draw_profiles(ax, angles, data["ratios"][column], color)
                format_panel(ax, block, bottom_row=(row == len(MODELS) - 1))
        axes[row, -1].yaxis.set_label_position('right')
        axes[row, -1].set_ylabel(label, fontsize=font_size("tiny"), labelpad=2, rotation=0, ha='left', va='center')

    fig.supxlabel(r"Angle $\theta$ (degrees)", fontsize=font_size("scriptsize"))
    fig.supylabel("Norm\ngrowth", fontsize=font_size("scriptsize"), rotation=0)
    save_figure(fig, figure_path, data_dir, style_path, manuscript, use_tex)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PAPER / "data/great_circle_norm_grid")
    parser.add_argument("--figure", type=Path, default=FIGURE_DIR / "great_circle_norm_grid.pdf")
    parser.add_argument("--style", type=Path, default=PAPER / "styles/matplotlibrc")
    args = parser.parse_args()
    plot(args.data, args.figure, args.style)


if __name__ == "__main__":
    main()
