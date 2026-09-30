"""Shared manuscript figure styling and profile-drawing helpers.

Keep figure-specific dimensions and layouts in the individual renderers.
"""
import hashlib
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

# Source/editable-install root; no manuscript checkout is needed.
PAPER = Path(__file__).resolve().parents[3]

# Rows: data filename stem, display label, mean-curve color.
MODELS = [
    ("llama3", "Llama 3.1 8B", "#31688e"),
    ("olmo3", "OLMo 3 7B", "#238b45"),
    ("qwen3", "Qwen 3 8B", "#a34b9b"),
]
SAMPLE_CURVES = 20
OUTSIDE_CIRCLE_ALPHA = 0.28

# Select once per process so imported size constants and output paths agree.
STYLE = os.environ.get("FIGURE_STYLE", "iclr")
if STYLE not in {"iclr", "goodfire"}:
    raise ValueError(f"Unknown FIGURE_STYLE: {STYLE!r}; expected iclr or goodfire")
FIGURE_DIR = PAPER / "fig" / STYLE
FONT_PREAMBLE = PAPER / "styles" / f"{STYLE}-fonts.tex"
TEX_POINTS_PER_INCH = 72.27
# ICLR overrides article's tiny/footnotesize; Goodfire uses 10pt article sizes.
LATEX_SIZES = {"tiny": 6, "scriptsize": 7, "footnotesize": 9, "small": 9, "normalsize": 10}
TEXT_WIDTH_INCHES = 5.5
if STYLE == "goodfire":
    LATEX_SIZES = dict(LATEX_SIZES, tiny=5, footnotesize=8)
    # Default letter paper, with the class's 2.5 cm left and right margins.
    TEXT_WIDTH_INCHES = 8.5 - 5 / 2.54
TEXT_WIDTH_TEX_POINTS = TEXT_WIDTH_INCHES * TEX_POINTS_PER_INCH


def font_size(name):
    r"""Named TeX size for the LaTeX/PGF renderers used by the paper.

    These renderers pass fontsize directly to TeX's \\fontsize command. Do not
    preconvert to PDF points: that would apply the 72/72.27 conversion twice.
    """
    return LATEX_SIZES[name]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_style(style_path):
    """Render with the profile's figure typography and LuaLaTeX font metrics."""
    settings = dict(matplotlib.rc_params_from_file(style_path, use_default_template=False))
    settings.pop("backend", None)
    plt.rcParams.update(settings)
    if shutil.which("lualatex") is None:
        raise RuntimeError("LuaLaTeX is required to match the document fonts")
    plt.switch_backend("pgf")
    preamble = FONT_PREAMBLE.read_text()
    if STYLE == "goodfire":
        preamble += (PAPER / "styles/figure-fonts.tex").read_text()
        preamble += (PAPER / "styles/figure-math.tex").read_text()
        # Cabin has no small caps; render grammatical tags as uppercase instead.
        preamble += (r"\let\sffamily\figurefont"
                     r"\renewcommand{\textsc}[1]{\MakeUppercase{#1}}")
    preamble += r"\pagestyle{empty}\usepackage{siunitx}\tracinglostchars=3"
    plt.rcParams.update({
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False,
        "pgf.preamble": preamble,
        # PGF still uses LuaLaTeX; avoid forcing Goodfire's numeric ticks into math mode.
        "text.usetex": STYLE == "iclr",
        "font.family": "sans-serif" if STYLE == "goodfire" else "serif",
        "font.size": font_size("tiny"),
        "text.latex.preamble": preamble,
        "axes.formatter.use_mathtext": STYLE == "iclr",
        "axes.titlesize": font_size("tiny"),
        "figure.figsize": (TEXT_WIDTH_INCHES, 3.0),
        "savefig.bbox": None,
    })
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FONT_PREAMBLE, plt.rcParams["text.usetex"]


def draw_profiles(ax, angles, curves, color, representative_index=None):
    """Draw samples and a mean or selected sample, fading periodic extensions."""
    mean_curve = curves.mean(axis=0) if representative_index is None else curves[representative_index]
    # Include each ±180° endpoint in both adjacent segments to avoid gaps.
    segments = [
        (angles <= -180, OUTSIDE_CIRCLE_ALPHA),
        ((angles >= -180) & (angles <= 180), 1.0),
        (angles >= 180, OUTSIDE_CIRCLE_ALPHA),
    ]
    for mask, opacity in segments:
        for curve in curves[:SAMPLE_CURVES]:
            ax.plot(
                angles[mask], curve[mask],
                color=".45", alpha=0.25 * opacity, linewidth=0.3,
            )
        ax.plot(
            angles[mask], mean_curve[mask],
            color=color, alpha=opacity, linewidth=0.65,
        )
