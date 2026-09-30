# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Render selected-block natural/synthetic comparisons from precomputed NPZ data."""
import argparse
import json
from pathlib import Path

import numpy as np

from .figure_style import (
    FIGURE_DIR,
    PAPER, MODELS, TEXT_WIDTH_TEX_POINTS, TEX_POINTS_PER_INCH,
    configure_style, draw_profiles, font_size, file_hash, SAMPLE_CURVES,
)
import matplotlib.pyplot as plt

CLASSES = [
    ("Natural", "#31688e"),
    ("Gaussian", "#238b45"),
    ("Shuffled", "#a34b9b"),
    ("Norm-matched", "#c17c27"),
]


def plot(data_dir, figure_path, style_path, llama_data):
    data_files = {key: data_dir / f"{key}.npz" for key, _, _ in MODELS}
    data_files["llama3"] = llama_data
    manuscript, use_tex = configure_style(style_path.resolve())
    fig, axes = plt.subplots(3, 4, sharex=True, sharey="row",
        figsize=(TEXT_WIDTH_TEX_POINTS / TEX_POINTS_PER_INCH, 1.7), layout='constrained')
    ranges = {}
    blocks = {}
    for row, (key, label, _) in enumerate(MODELS):
        with np.load(data_files[key]) as data:
            angles, ratios, block = data["theta"], data["ratios"], int(data["block"])
            blocks[key] = block
            assert ratios.shape == (4, 20, 401)
            # Include ALL rendered sample extrema and means: no clipping or quantiles.
            shown = np.concatenate([ratios[:, :SAMPLE_CURVES].reshape(-1, 401), ratios.mean(axis=1)])
            low, high = float(shown.min()), float(shown.max())
            pad = 0.06 * (high-low)
            ranges[key] = dict(ylim=[low-pad, high+pad], all_sample_range=[float(ratios.min()), float(ratios.max())])
            for column, (title, color) in enumerate(CLASSES):
                ax = axes[row, column]
                draw_profiles(ax, angles, ratios[column], color)
                ax.set_xlim(-200, 200)
                ax.set_ylim(low-pad, high+pad)
                ax.set_axisbelow(True)
                ax.set_xticks([-180, 0, 180])
                ax.set_xticks([-90, 90], minor=True)
                ax.grid(True)
                ax.grid(True, axis="x", which="minor")
                ax.tick_params(axis="x", which="minor", bottom=False)
                ax.tick_params(labelsize=font_size("tiny"), length=0, pad=1,
                    labelleft=(column == 0), labelbottom=(row == 2))
                ax.locator_params(axis="y", nbins=3)
                if row == 0:
                    ax.set_title(title, fontsize=font_size("scriptsize"), pad=3)
            axes[row, -1].yaxis.set_label_position('right')
            axes[row, -1].set_ylabel(f"{label} Block {block}", fontsize=font_size("tiny"),
                                     labelpad=3, rotation=0, ha='left', va='center')
    fig.supxlabel(r"Angle $\theta$ (degrees)", fontsize=font_size("scriptsize"))
    fig.supylabel("Norm growth", fontsize=font_size("scriptsize"), rotation=0)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(figure_path)
        fig.savefig(figure_path.with_suffix(".png"), dpi=300)
    provenance = dict(script_sha256=file_hash(__file__), helper_sha256=file_hash(Path(__file__).with_name("great_circle_norm_grid.py")),
        style_sha256=file_hash(style_path), manuscript_sha256=file_hash(manuscript), text_usetex=use_tex,
        width_tex_pt=TEXT_WIDTH_TEX_POINTS, font_roles={"titles": "scriptsize", "axis_labels": "scriptsize", "ticks_and_rows": "tiny"},
        blocks_1based=blocks,
        data_files={key: str(path) for key, path in data_files.items()},
        data_sha256={key: file_hash(path) for key, path in data_files.items()},
        ranges=ranges, y_limits="Shared per model; full extrema of all 20 samples and their means, padded 6%; no clipping")
    figure_path.with_suffix(".render.json").write_text(json.dumps(provenance, indent=2)+"\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PAPER / "data/great_circle_synthetic_grid_best_layers",
                        help="Data directory for OLMo and Qwen")
    parser.add_argument("--llama-data", type=Path, default=PAPER / "data/great_circle_synthetic_grid_llama21/llama3.npz",
                        help="Llama probes; defaults to block 21 for comparable depth to OLMo")
    parser.add_argument("--figure", type=Path, default=FIGURE_DIR / "great_circle_synthetic_grid.pdf")
    parser.add_argument("--style", type=Path, default=PAPER / "styles/matplotlibrc")
    args = parser.parse_args()
    plot(args.data, args.figure, args.style, args.llama_data)
