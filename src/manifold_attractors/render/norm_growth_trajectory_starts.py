# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Synthetic cohorts at stored input blocks: means +/- sample SD (ddof=1).

Models x two synthetic classes (shuffled, norm-matched); colors encode start.
--starts all displays every cohort, including the final single-point cohort.
The default preserves the old start15..19 display for six-block midpoint data.
All lines are solid; natural uses squares, synthetic uses circles. The same
natural baseline (initialized at the first stored block) appears in both columns.
Native-document states at later layers fit controls but are NOT the plotted
natural baseline. No synthetic resampling within a trajectory. Per-block ratios,
not cumulative growth. No unity anchor. Independent y limits; every panel includes a tick at1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from .figure_style import (FIGURE_DIR, PAPER,MODELS,TEXT_WIDTH_TEX_POINTS,TEX_POINTS_PER_INCH,
    configure_style,font_size,file_hash,LATEX_SIZES)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

CLASSES=[(1,'Shuffled'),(2,'Norm-matched')]
COLORS=['#440154','#414487','#2a788e','#22a884','#7ad151']


def render(a):
    manuscript,use_tex=configure_style(a.style.resolve())
    models=[item for item in MODELS if item[0] in a.models]
    fig,axes=plt.subplots(len(models),2,sharex=False,squeeze=False,
        figsize=(TEXT_WIDTH_TEX_POINTS/TEX_POINTS_PER_INCH,a.height), layout='constrained')
    summaries={};displayed_starts={};baseline_starts={}
    for row,(key,label,_) in enumerate(models):
        with np.load(a.data/f'{key}.npz') as d:
            blocks,starts,ratios,natural=d['blocks'],d['starts'],d['ratios'],d['natural_ratios']
            assert ratios.shape==(3,len(starts),128,len(blocks)) and natural.shape==(128,len(blocks))
            assert np.array_equal(blocks,np.arange(blocks[0],blocks[-1]+1))
            if a.starts is None:
                chosen=starts[:-1] if len(starts)==6 and starts[0]==15 else starts
            elif a.starts==['all']:chosen=starts
            else:chosen=np.array([int(x) for x in a.starts])
            assert len(chosen)>0 and len(set(chosen))==len(chosen) and set(chosen)<=set(starts)
            displayed_starts[key]=[int(x) for x in chosen];baseline_starts[key]=int(blocks[0])
            colors=COLORS if len(chosen)==5 else plt.get_cmap('viridis')(np.linspace(0,.9,len(chosen)))
            nmean,nsd=natural.mean(0),natural.std(0,ddof=1)
            model_summary={}; panel_ranges={}
            for col,(class_index,name) in enumerate(CLASSES):
                low,high=min(1.,float((nmean-nsd).min())),max(1.,float((nmean+nsd).max()))
                ax=axes[row,col]; model_summary[name]={}
                ax.set_axisbelow(True)
                ax.grid(True,color='white',linewidth=.5)
                for ci,start_block in enumerate(chosen):
                    start=int(np.flatnonzero(starts==start_block)[0])
                    mask=d['valid'][start]
                    y=ratios[class_index,start][:,mask]
                    assert np.isfinite(y).all() and (y>0).all()
                    mean,sd=y.mean(0),y.std(0,ddof=1)
                    low=min(low,float((mean-sd).min())); high=max(high,float((mean+sd).max()))
                    model_summary[name][str(int(start_block))]=dict(blocks=blocks[mask].tolist(),mean=mean.tolist(),sample_sd=sd.tolist())
                    x=blocks[mask]
                    ax.plot(x,mean,color=colors[ci],linestyle='-',linewidth=.55 if len(chosen)>6 else .7,
                            marker='o',markersize=1.2 if len(chosen)>6 else 1.7,alpha=.4,zorder=4)
                ax.plot(blocks,nmean,color='.12',linewidth=.9,marker='s',markersize=2,zorder=5)
                ax.set_xlim(float(blocks[0])-.2,float(blocks[-1])+.25)
                ticks=blocks if len(blocks)<=8 else np.unique(np.r_[blocks[0],np.arange(5,blocks[-1]-1,5),blocks[-1]])
                ax.set_xticks(ticks)
                ax.tick_params(labelsize=font_size('tiny'),length=0,pad=2,labelbottom=(len(blocks)>8 or row==len(models)-1),labelleft=True)
                limits = [0.8, 1.5]
                ax.set_ylim(*limits)
                ticks=MaxNLocator(nbins=3).tick_values(*limits)
                ticks=ticks[(ticks>=limits[0]) & (ticks<=limits[1])]
                ticks=ticks[np.abs(ticks-1.)>max(1e-8,.06*(limits[1]-limits[0]))]
                ax.set_yticks(np.sort(np.append(ticks,1.)))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
                panel_ranges[name]=limits
                if row==0:ax.set_title(name,fontsize=font_size('scriptsize'),pad=3)
            if len(models) > 1:
                axes[row,-1].yaxis.set_label_position('right')
                axes[row,-1].set_ylabel(label,fontsize=font_size('tiny'),labelpad=3, rotation=0, ha='left', va='center')
            else:
                axes[row,0].set_ylabel('Per-block\nnorm growth',fontsize=font_size('scriptsize'),labelpad=3, rotation=0, ha='right', va='center')
            summaries[key]=dict(natural_mean=nmean.tolist(),natural_sample_sd=nsd.tolist(),
                                cohorts=model_summary,panel_y_limits=panel_ranges)
    fig.supxlabel('Block',fontsize=font_size('scriptsize'))
    if len(models) > 1:
        fig.supylabel('Per-block\nnorm growth',fontsize=font_size('scriptsize'), rotation=0)
    handles=[Line2D([0],[0],color='.12',linewidth=.9,marker='s',markersize=2,label='Natural'),
             Line2D([0],[0],color=COLORS[0],linewidth=.7,marker='o',markersize=2,alpha=.4,label='Synthetic')]
    axes[0,0].legend(handles=handles,loc='upper center',ncol=2,frameon=False,
                     fontsize=font_size('tiny'),handlelength=2)
    a.figure.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'savefig.bbox':None}):
        fig.savefig(a.figure);fig.savefig(a.figure.with_suffix('.png'),dpi=300)
    plt.close(fig)
    provenance=dict(script_sha256=file_hash(__file__),style_sha256=file_hash(a.style),
        helper_sha256=file_hash(Path(__file__).with_name('figure_style.py')),
        manuscript_sha256=file_hash(manuscript),text_usetex=use_tex,width_tex_pt=TEXT_WIDTH_TEX_POINTS,latex_sizes=LATEX_SIZES,
        font_roles=dict(column_titles='scriptsize',axis_labels='scriptsize',ticks_rows_legend='tiny'),
        quantity='per-block output/input norm ratio; initialized once at own one-based input block',
        bands=None,
        displayed_classes=['shuffled','norm-matched'],displayed_start_blocks=displayed_starts,
        last_start_single_point='included when selected; all computed starts retained in data',
        markers=dict(natural='square',synthetic='circle'),line_styles='all solid',
        legend='Natural / Synthetic; upper center of top-left panel, two columns',
        natural='same single fixed BOS/EOS-probe baseline in both columns, NOT native-document trajectory',
        natural_start_blocks=baseline_starts,
        shared_y=False,required_y_tick=1,summaries=summaries,
        models=[key for key,_,_ in models],
        data_sha256={key:file_hash(a.data/f'{key}.npz') for key,_,_ in models},
        records_sha256={key:file_hash(a.data/f'{key}.json') for key,_,_ in models})
    a.figure.with_suffix('.render.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=PAPER/'data/norm_growth_trajectory_starts')
    p.add_argument('--figure',type=Path,default=FIGURE_DIR / "norm_growth_trajectory_starts.pdf")
    p.add_argument('--models',nargs='+',choices=[key for key,_,_ in MODELS],default=[key for key,_,_ in MODELS])
    p.add_argument('--style',type=Path,default=PAPER/'styles/matplotlibrc')
    p.add_argument('--starts',nargs='+',help='all or explicit one-based start blocks; default preserves legacy midpoint display and shows all starts otherwise')
    p.add_argument('--height',type=float,default=2.,help='Figure height in inches; use 3 for the all-layer draft')
    render(p.parse_args())
