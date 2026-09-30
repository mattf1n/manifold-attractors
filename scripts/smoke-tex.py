#!/usr/bin/env python3
"""Model-free media, inline TikZ and scientific-font smoke; reject font fallback.
Run from repo root: uv run --locked --isolated --extra render python scripts/smoke-tex.py
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def run(args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def compile_tex(tex):
    with tex.with_suffix('.stdout').open('w') as log:
        run(['lualatex', '-interaction=nonstopmode', '-halt-on-error', tex.name],
            cwd=tex.parent, stdout=log, stderr=subprocess.STDOUT)
    log = tex.with_suffix('.log').read_text()
    issues = re.findall(r'.*(?:LaTeX Font Warning|Missing character|defaults substituted).*', log)
    if issues:
        raise RuntimeError(f'Font fallback in {tex}:\n' + '\n'.join(issues))


def profile(out):
    import matplotlib.pyplot as plt
    from manifold_attractors.render.figure_style import STYLE, PAPER
    from manifold_attractors.render.inline_tikz import document
    from manifold_attractors.render.language_steering_speed import setup_style

    for kind in ('intro', 'sphere_probe'):
        tex = out / f'{STYLE}-{kind}.tex'
        tex.write_text(document(kind, out / 'two_basin_field.dat', out / 'dead_fish_sequence.pdf'))
        compile_tex(tex)
    setup_style(PAPER / 'styles/matplotlibrc')
    family = r'\sffamily' if STYLE == 'goodfire' else r'\rmfamily'
    body = (family + r''' Cabin or Termes $\theta\,\pi\,\alpha\,\Delta\,x^2\,\|x\|\,\sum_{i=1}^n i$;
\texttt{en--el--zh}; {\bfseries Bold}; \textsc{nom.acc.sg};
{\cjkSCfont 中文代}{\cjkTCfont 中文體}
{\cyrillicfont русский}{\greekfont ελληνικά}
''')
    tex = out / f'{STYLE}-fonts.tex'
    tex.write_text(r'\documentclass{standalone}' + '\n' + plt.rcParams['pgf.preamble']
                   + '\n' + r'\begin{document}\fontsize{5}{7}\selectfont' + body
                   + r'\end{document}' + '\n')
    compile_tex(tex)
    # Exercise the real Matplotlib PGF path, including PNG conversion.
    fig, ax = plt.subplots(figsize=(2, 1.2))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel(r'Angle $\theta$ (degrees)')
    ax.set_ylabel(r'\texttt{en}')
    fig.savefig(out / f'{STYLE}-pgf.pdf')
    fig.savefig(out / f'{STYLE}-pgf.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/tex-smoke'))
    parser.add_argument('--profile', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.profile:
        profile(out)
        return
    version = subprocess.check_output(['lualatex', '--version'], text=True)
    if 'Version 1.21.0 (TeX Live 2025)' not in version:
        raise RuntimeError('Expected LuaHBTeX 1.21.0 / TeX Live 2025; got ' + version)
    run([sys.executable, '-m', 'manifold_attractors.render.dead_fish_sequence',
         '--data', 'assets/A-Dead-Fish-Swims.gif', '--figure', out / 'dead_fish_sequence.pdf'])
    run([sys.executable, '-m', 'manifold_attractors.render.two_basin_field',
         '--output', out / 'two_basin_field.dat'])
    for style in ('iclr', 'goodfire'):
        run([sys.executable, Path(__file__).resolve(), '--profile', '--output', out],
            env=dict(os.environ, FIGURE_STYLE=style))
    from pypdf import PdfReader
    evidence = {'engine': version, 'reconstructed_styles': True, 'pdfs': {}}
    for name in ('dead_fish_sequence', 'iclr-intro', 'iclr-sphere_probe', 'iclr-fonts',
                 'iclr-pgf', 'goodfire-intro', 'goodfire-sphere_probe', 'goodfire-fonts',
                 'goodfire-pgf'):
        pdf = out / f'{name}.pdf'
        reader = PdfReader(pdf)
        fonts = sorted({str(font.get_object().get('/BaseFont'))
                        for page in reader.pages
                        for font in page.get('/Resources', {}).get('/Font', {}).values()})
        evidence['pdfs'][pdf.name] = {'producer': reader.metadata.get('/Producer'), 'fonts': fonts}
        run(['pdftocairo', '-png', '-singlefile', '-r', '150', pdf, out / name])
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'PASS: media, four inline artworks, two font probes, two PGF plots: {out}')
    print('No font-substitution warnings or missing glyphs in retained TeX logs.')
    print('RECONSTRUCTION ONLY: this is not a historical pixel-equality check.')


if __name__ == '__main__':
    main()
