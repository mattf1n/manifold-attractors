"""Native-size manuscript styling for the historical data-only figure renderers."""
from pathlib import Path
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from .figure_style import FIGURE_DIR, TEXT_WIDTH_INCHES, configure_style, font_size, PAPER


ROOT = PAPER


def setup():
    configure_style(ROOT / "styles/matplotlibrc")
    plt.rcParams.update({
        "axes.titlesize": font_size("scriptsize"),
        "axes.labelsize": font_size("scriptsize"),
        "xtick.labelsize": font_size("tiny"),
        "ytick.labelsize": font_size("tiny"),
        "legend.fontsize": font_size("tiny"),
        "savefig.bbox": None,
    })


def save(path, **kwargs):
    """Keep the native canvas (never tight-crop or scale) and a review PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches=None, **kwargs)
    plt.savefig(Path(path).with_suffix(".png"), dpi=180, bbox_inches=None)
