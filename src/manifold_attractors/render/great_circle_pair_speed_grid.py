# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Speed toward same-class partner directions: exact anchor states, new tangents.

Colored lines average all supplied pairs at unshifted anchor-centered theta;
First 20 are gray. Bottom rugs mark partner DIRECTION angles. Circles preserve anchor radius,
so they do not hit the literal partner state unless partner/anchor radii match.
Y ranges follow colored means only; gray examples may extend beyond the panel. Raw
vector speed per radian, no radius normalization. Generator is
the great_circle_pair_speed_grid data generator (not imported here).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .figure_style import (
    TEXT_WIDTH_INCHES, FIGURE_DIR,
    PAPER, MODELS, TEXT_WIDTH_TEX_POINTS, TEX_POINTS_PER_INCH,
    configure_style, draw_profiles, font_size, SAMPLE_CURVES,
)
from .great_circle_synthetic_grid import CLASSES
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def file_hash(path):
    """Stream large retained-vector artifacts rather than loading gigabytes at once."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def plot(data_dir, figure_path, style_path, model_keys=None, independent_y=False, partner_rugs=True, column_title=None,
         random_representative=False, seed=20260920, width=TEXT_WIDTH_INCHES):
    manuscript, use_tex = configure_style(style_path.resolve())
    models=[item for item in MODELS if model_keys is None or item[0] in model_keys]
    with np.load(data_dir/f'{models[0][0]}.npz') as first:
        class_keys=first['classes'].tolist()
    styles={title.lower():(title,color) for title,color in CLASSES}
    classes=[styles[key] for key in class_keys]
    fig, axes = plt.subplots(len(models), len(classes), sharex=True, sharey=False if independent_y else 'row', squeeze=False,
        figsize=(width, .8*len(models)+.4), layout='constrained')
    ranges = {}
    resolutions = {}
    sample_counts = {}
    selections = {}
    rng = np.random.default_rng(seed)
    for row, (key, label, _) in enumerate(models):
        with np.load(data_dir / f'{key}.npz') as data:
            angles, speeds, block = data['theta'], data['speeds'], int(data['block'])
            block_end = int(data['block_end']) if 'block_end' in data else block
            block_label = f'Blocks {block}--{block_end}' if block_end != block else f'Block {block}'
            resolutions[key] = float(data['step_degrees']) if 'step_degrees' in data else 1.
            assert data['classes'].tolist()==class_keys
            assert speeds.ndim==3 and speeds.shape[0]==len(classes) and speeds.shape[2]==401
            sample_counts[key]=speeds.shape[1]
            assert np.array_equal(angles, np.arange(-200, 201))
            assert np.isfinite(speeds).all() and (speeds > 0).all()
            selected = rng.integers(speeds.shape[1], size=len(classes)).tolist() if random_representative else None
            selections[key] = selected
            representative = np.stack([speeds[ci, i] for ci, i in enumerate(selected)]) if selected is not None else speeds.mean(axis=1)
            low, high = float(representative.min()), float(representative.max())
            pad = .06 * (high-low)
            ranges[key] = dict(ylim=[max(0, low-pad), high+pad],
                              all_sample_range=[float(speeds.min()), float(speeds.max())])
            for column, (title, color) in enumerate(classes):
                ax = axes[row, column]
                draw_profiles(ax, angles, speeds[column], color,
                              representative_index=selected[column] if selected is not None else None)
                if partner_rugs:
                    partner_angles = data['partner_angles'][column]
                    assert partner_angles.shape == (speeds.shape[1],) and np.all((partner_angles > 0) & (partner_angles < 180))
                    ax.plot(partner_angles, np.full(len(partner_angles), .025), transform=ax.get_xaxis_transform(),
                            linestyle='none', marker='|', markersize=2.5, markeredgewidth=.5,
                            color=color, alpha=.5, zorder=4)
                ax.set_xlim(-200, 200)
                if independent_y:
                    panel_curve = representative[column]
                    panel_low, panel_high = float(panel_curve.min()), float(panel_curve.max())
                    panel_pad=.06*(panel_high-panel_low)
                    limits=[max(0,panel_low-panel_pad),panel_high+panel_pad]
                    ranges[key].setdefault('panel_ylims',{})[class_keys[column]]=limits
                    ax.set_ylim(*limits)
                else:
                    ax.set_ylim(*ranges[key]['ylim'])
                ax.set_axisbelow(True)
                ax.set_xticks([-180, 0, 180])
                if width < 4:
                    ax.xaxis.get_major_ticks()[0].label1.set_ha('left')
                    ax.xaxis.get_major_ticks()[-1].label1.set_ha('right')
                ax.set_xticks([-90, 90], minor=True)
                ax.grid(True, which='both', color='white', linewidth=.5)
                ax.tick_params(axis='x', which='minor', bottom=False)
                ax.tick_params(labelsize=font_size('tiny'), length=0, pad=1,
                    labelleft=(independent_y or column == 0), labelbottom=(row == len(models)-1))
                ax.locator_params(axis='y', nbins=3)
                ax.yaxis.set_major_formatter(EngFormatter(sep=''))
                if row == 0:
                    ax.set_title(column_title or title, fontsize=font_size('scriptsize'), pad=3)
            if len(models) > 1:
                axes[row, -1].yaxis.set_label_position('right')
                axes[row, -1].set_ylabel(f'{label}\n{block_label}', fontsize=font_size('tiny'), labelpad=3, rotation=0, ha='left', va='center')
            else:
                axes[row, 0].set_ylabel('Speed', fontsize=font_size('scriptsize'), labelpad=3, rotation=0, ha='right', va='center')
    fig.supxlabel(r'Angle $\theta$ (degrees)', fontsize=font_size('scriptsize'))
    if len(models) > 1:
        fig.supylabel('Speed', fontsize=font_size('scriptsize'), rotation=0)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(figure_path)
        fig.savefig(figure_path.with_suffix('.png'), dpi=300)
    provenance = dict(script_sha256=file_hash(__file__),
        helper_sha256=file_hash(Path(__file__).with_name('great_circle_norm_grid.py')),
        class_style_sha256=file_hash(Path(__file__).with_name('great_circle_synthetic_grid.py')),
        style_sha256=file_hash(style_path), manuscript_sha256=file_hash(manuscript), text_usetex=use_tex,
        width_tex_pt=width*TEX_POINTS_PER_INCH,
        font_roles=dict(titles='scriptsize', axis_labels='scriptsize', ticks_and_rows='tiny'),
        models=[key for key,_,_ in models],
        data_sha256={key: file_hash(data_dir / f'{key}.npz') for key, _, _ in models},
        records_sha256={key: file_hash(data_dir / f'{key}.json') for key, _, _ in models},
        primary_curves=(f'uniformly random individual sample; first {SAMPLE_CURVES} individual profiles in gray' if random_representative
                        else f'arithmetic mean of all supplied samples; first {SAMPLE_CURVES} individual profiles in gray'),
        selection_seed=seed if random_representative else None, selected_sample_indices=selections,
        samples_per_class=sample_counts,classes=class_keys,
        quantity='raw output-vector speed per radian; not input-radius normalized',
        column_title=column_title,
        partner_rugs=('all partner-direction angles, unshifted; anchor theta=0; rug at2.5% panel height' if partner_rugs else None),
        radius_caveat=('partner direction reached at anchor radius, not literal partner when norms differ' if partner_rugs
                       else 'circle preserves anchor radius; no assumed natural partner on circle'),
        derivative_step_degrees=resolutions,
        ranges=ranges, independent_y=independent_y,
        y_limits='Colored highlighted curves only, padded 6%, floor 0; gray examples may be clipped; independent panels if requested')
    figure_path.with_suffix('.render.json').write_text(json.dumps(provenance, indent=2)+'\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PAPER / 'data/great_circle_pair_speed_grid')
    p.add_argument('--figure', type=Path, default=FIGURE_DIR / "great_circle_pair_speed_grid.pdf")
    p.add_argument('--style', type=Path, default=PAPER / 'styles/matplotlibrc')
    p.add_argument('--models',nargs='+',choices=[key for key,_,_ in MODELS])
    width_options = p.add_mutually_exclusive_group()
    width_options.add_argument('--width', type=float, default=TEXT_WIDTH_INCHES, help='Figure width in inches')
    width_options.add_argument('--half-width', action='store_true', help='Half the text width after a 0.1-inch gap')
    p.add_argument('--independent-y',action='store_true')
    p.add_argument('--no-partner-rugs',action='store_true')
    p.add_argument('--column-title')
    p.add_argument('--random-representative', action='store_true')
    p.add_argument('--seed', type=int, default=20260920)
    a = p.parse_args()
    plot(a.data, a.figure, a.style, a.models, a.independent_y, not a.no_partner_rugs, a.column_title,
         a.random_representative, a.seed, (TEXT_WIDTH_INCHES - .1) / 2 if a.half_width else a.width)
