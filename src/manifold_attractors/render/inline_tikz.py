"""Compile the two inline manuscript artworks, without captions or page furniture."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from .figure_style import FIGURE_DIR, PAPER, STYLE, TEXT_WIDTH_INCHES, font_size


def tex_path(path):
    path = str(Path(path).resolve())
    if any(char in path for char in '{}\n\r%'):
        raise ValueError(f'Unsupported TeX path: {path!r}')
    return r'\detokenize{' + path + '}'


def document(kind, data, fish):
    preamble = (PAPER / 'styles' / f'{STYLE}-fonts.tex').read_text()
    # Inline TikZ inherits manuscript body math, not the PGF figure-caption font.
    if STYLE == 'goodfire':
        preamble += (r'\usepackage[subdued,defaultmathsizes,italic,LGRgreek+]{mathastext}'
                     r'\MTgreekfont{\notosansfamily}')
    preamble += r'''
\usepackage{bm,graphicx}
\usepackage[export]{adjustbox}
\usepackage{siunitx}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,colormaps}
\usetikzlibrary{hobby,arrows.meta}
\pgfplotsset{compat=newest,width=0.9\textwidth,height=4cm}
\newcommand{\vx}{\bm{x}}
\newcommand{\vu}{\bm{u}}
\pagestyle{empty}
'''
    size = font_size('footnotesize')
    preamble += rf'\tikzset{{font=\fontsize{{{size}}}{{{size + 2}}}\selectfont}}' + '\n'
    preamble += rf'\setlength{{\textwidth}}{{{TEXT_WIDTH_INCHES}in}}' + '\n'
    artwork = (PAPER / 'assets' / f'{kind}.tex').read_text()
    if kind == 'intro':
        for path in (data, fish):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        preamble += r'\edef\FishSequence{' + tex_path(fish) + '}\n'
        preamble += r'\edef\BasinField{' + tex_path(data) + '}\n'
        artwork = (rf'\begin{{minipage}}{{{TEXT_WIDTH_INCHES}in}}\noindent' + '\n'
                   + artwork + '\\end{minipage}\n')
    return (r'\documentclass[border=0pt]{standalone}' + '\n' + preamble
            + '\\begin{document}\n' + artwork + '\\end{document}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['intro', 'sphere_probe'])
    parser.add_argument('--data', type=Path, default=PAPER / 'data/two_basin_field.dat')
    parser.add_argument('--fish', type=Path, default=PAPER / 'fig/shared/dead_fish_sequence.pdf')
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--tex-only', action='store_true', help='Write source without invoking LuaLaTeX')
    args = parser.parse_args()
    output = args.figure or FIGURE_DIR / f'{args.kind}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    source = document(args.kind, args.data, args.fish)
    output.with_suffix('.tex').write_text(source)
    if args.tex_only:
        return
    if not shutil.which('lualatex'):
        raise RuntimeError('LuaLaTeX is required; install TeX Live and see RENDER.md')
    with tempfile.TemporaryDirectory(prefix='manifold-tikz-') as directory:
        tex = Path(directory) / 'artwork.tex'
        tex.write_text(source)
        process = subprocess.run(['lualatex', '-interaction=nonstopmode', '-halt-on-error', tex.name],
                                 cwd=directory, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
        shutil.copyfile(tex.with_suffix('.pdf'), output)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
