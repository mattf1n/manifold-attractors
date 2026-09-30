# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Render language-steering curves and their decoded-token regions.

Run from paper/: uv run scripts/fig/language_steering_speed.py

Edit EXAMPLES and the color constants to change the presentation.
Edit data/language_steering_speed/glosses.json to change token labels.
The default view shows four English contexts above four Chinese contexts,
over ±45° from frozen ±90° curves. The English URL-/blog example is omitted.
No model inference runs here.

Goodfire uses Cabin with Noto Sans/CJK fallbacks; ICLR retains NewTX text
and serif fallbacks. Sizes, math fonts, and width follow the selected profile.
"""

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from itertools import combinations, groupby, pairwise
from pathlib import Path

# Backend/cache setup must precede plotting imports.
# LaTeX temporary-file cleanup is unreliable on the mounted home filesystem.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"language-steering-mpl-{os.getuid()}"),
)

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
from .figure_style import (
    PAPER,
    FIGURE_DIR,
    STYLE,
    TEX_POINTS_PER_INCH,
    TEXT_WIDTH_TEX_POINTS,
    configure_style,
    font_size,
)
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

DATA = PAPER / "data/language_steering_speed"

# Zero-based rows in the frozen five-example dataset.
EXAMPLES = (0, 1, 3, 4)
ANGLE_LIMIT = 45
CURVE_COLOR = "#30343b"
REGION_ALPHA = 0.32

SCRIPT_RUNS = re.compile(r"([\u3000-\u9fff\uff00-\uffef]+|[\u0400-\u052f]+|[\u0370-\u03ff\u1f00-\u1fff]+)")
TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
    "→": r"$\rightarrow$",
    "␣": r"{\cjkSCfont\textvisiblespace}" if STYLE == "goodfire" else r"\textvisiblespace{}",
    "−": r"$-$",
}


@dataclass
class Inputs:
    """Frozen numerical arrays and their human-readable metadata."""

    arrays: dict[str, np.ndarray]
    regions: list[list[dict]]
    glosses: dict[str, dict]
    selection: dict
    context_glosses: dict = field(default_factory=dict)


@dataclass
class Panel:
    """One cropped example with script-colored generation regions."""

    title: tuple[str, str]
    angles: np.ndarray
    speed: np.ndarray
    regions: list[dict]
    context_gloss: dict | None = None
    title_alternatives: tuple[tuple[str, str], ...] = ()
    colors: list[tuple] = field(default_factory=list)


# Input loading and scientific consistency checks.


def read_json(path: Path):
    return json.loads(path.read_text())


def load_inputs(folder: Path, suffix: str = "") -> Inputs:
    with np.load(folder / f"llama3{suffix}.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    gloss_data = read_json(folder / f"glosses{suffix}.json")
    data = Inputs(
        arrays=arrays,
        regions=read_json(folder / f"regions{suffix}.json"),
        glosses=gloss_data["tokens"],
        selection=read_json(folder / f"selection{suffix}.json"),
        context_glosses=gloss_data.get("interlinear_contexts", {}),
    )
    validate_inputs(data)
    return data


def validate_inputs(data: Inputs) -> None:
    """Check token-region coverage, source selection, and projected tangents."""
    arrays = data.arrays
    angles = arrays["theta"]
    assert arrays["speeds"].shape == (len(data.regions), len(angles))
    assert (angles[0], angles[-1]) == (-90, 90)

    for regions, token_ids in zip(data.regions, arrays["token_ids"]):
        assert regions[0]["start"] == -90 and regions[-1]["end"] == 90
        assert all(a["end"] == b["start"] for a, b in pairwise(regions))
        for region in regions:
            assert region["text"] == data.glosses[str(region["token_id"])]["text"]
        for angle, token_id in zip(angles, token_ids):
            region = next(
                region
                for region in regions
                if region["start"] <= angle < region["end"]
                or (angle == 90 and region["end"] == 90)
            )
            assert region["token_id"] == int(token_id)

    np.testing.assert_array_equal(
        arrays["source_indices"], data.selection["selected_indices"]
    )
    if data.context_glosses:
        contexts = {str(item["index"]): item for item in data.selection["contexts"]}
        assert set(data.context_glosses) == {str(i) for i in arrays["source_indices"]}
        for key, annotation in data.context_glosses.items():
            source = contexts[key]
            marked = [
                unit["highlight"]
                for unit in annotation["units"]
                if unit.get("highlight")
            ]
            assert marked == [source["target_token"]]
            shown = "".join(unit["source"] for unit in annotation["units"])
            normalized_source = re.sub(r"[\W_]+", "", source["prefix_through_target"])
            normalized_shown = re.sub(r"[\W_]+", "", shown)
            assert normalized_source.endswith(normalized_shown)
    for state, tangent in zip(arrays["states"], arrays["tangents"]):
        pole = state.astype(float) / np.linalg.norm(state.astype(float))
        projected = arrays["direction"] - (arrays["direction"] @ pole) * pole
        projected /= np.linalg.norm(projected)
        np.testing.assert_allclose(projected, tangent, rtol=0, atol=1e-15)


def context_title(
    context: dict, words: int = 3, chinese: bool = False
) -> tuple[str, str]:
    """Return separate source-prefix and target text for highlighting."""
    prefix = context["prefix_through_target"]
    token = context["target_token"]
    assert token and prefix.endswith(token)
    before = " ".join(prefix[: -len(token)].split())
    # The Wikipedia extraction itself contains empty parentheticals; omit only
    # those display artifacts, without changing the frozen source/model input.
    before = re.sub(r"（[，,\s]*）", "", before)
    before = before[-8:] if chinese else " ".join(before.split()[-words:])
    separator = " " if token.startswith(" ") else ""
    shown = token.lstrip(" ") or "␣"
    shown = shown.replace("\n", r"\n").replace("\t", r"\t")
    return f"... {before}{separator}", shown


def make_panels(
    data: Inputs, limit: float, examples: tuple = EXAMPLES, chinese: bool = False
) -> list[Panel]:
    """Crop only the display; preserve every region intersecting the window."""
    if not 0 < limit <= 90:
        raise ValueError("The angle limit must be in (0, 90].")
    if not examples or len(set(examples)) != len(examples):
        raise ValueError("EXAMPLES must contain distinct source rows.")
    if any(row < 0 or row >= len(data.regions) for row in examples):
        raise ValueError("EXAMPLES contains a row outside the frozen dataset.")

    angles = data.arrays["theta"]
    mask = (angles >= -limit) & (angles <= limit)
    if mask.sum() < 2:
        raise ValueError("The requested window is narrower than the cached angle grid.")
    panels = []
    contexts = {context["index"]: context for context in data.selection["contexts"]}
    for row in examples:
        regions = [
            dict(
                region,
                start=max(-limit, region["start"]),
                end=min(limit, region["end"]),
            )
            for region in data.regions[row]
            if min(limit, region["end"]) > max(-limit, region["start"])
        ]
        assert regions[0]["start"] == -limit and regions[-1]["end"] == limit
        panels.append(
            Panel(
                title=context_title(
                    contexts[data.selection["selected_indices"][row]], chinese=chinese
                ),
                angles=angles[mask],
                speed=data.arrays["speeds"][row, mask],
                regions=regions,
                context_gloss=data.context_glosses.get(
                    str(data.selection["selected_indices"][row])
                ),
                title_alternatives=tuple(
                    context_title(
                        contexts[data.selection["selected_indices"][row]], words=words
                    )
                    for words in (2, 1)
                )
                if not chinese
                else (),
            )
        )
    return panels


# Fonts and reusable annotation helpers.


def setup_style(path: Path) -> None:
    configure_style(path.resolve())
    if STYLE == "goodfire":
        plt.rcParams["pgf.preamble"] += (
            r"\defaultfontfeatures{Renderer=HarfBuzz}"
            r"\newfontfamily\cyrillicfont{Noto Sans}"
            r"\let\greekfont\cyrillicfont"
            r"\newfontfamily\cjkSCfont{Noto Sans CJK SC}"
            r"\newfontfamily\cjkTCfont{Noto Sans CJK TC}"
        )
    else:
        plt.rcParams["pgf.preamble"] += (
            r"\newfontfamily\cyrillicfont{Tempora-Regular.otf}[BoldFont=Tempora-Bold.otf]"
            r"\newfontfamily\greekfont{cmunrm.otf}[BoldFont=cmunbx.otf]"
            r"\newfontfamily\cjkSCfont{gbsn00lp}[Extension=.ttf,AutoFakeBold=2]"
            r"\newfontfamily\cjkTCfont{bsmi00lp}[Extension=.ttf,AutoFakeBold=2]"
        )


def cjk_font(char: str) -> str:
    """Choose simplified or traditional Chinese glyph forms."""
    try:
        char.encode("gb2312")
        return "cjkSC"
    except UnicodeEncodeError:
        char.encode("big5")  # Fail visibly for unsupported glyphs, never silently omit.
        return "cjkTC"


def latex_label(text: str, bold: bool = False) -> str:
    """Escape literal tokens and route each script to the requested TeX font."""
    parts = []
    for run in SCRIPT_RUNS.split(text):
        if not run:
            continue
        first = run[0]
        if "\u0400" <= first <= "\u052f":
            parts.append(r"{\cyrillicfont " + run + "}")
        elif "\u0370" <= first <= "\u03ff" or "\u1f00" <= first <= "\u1fff":
            parts.append(r"{\greekfont " + run + "}")
        elif "\u3000" <= first <= "\u9fff" or "\uff00" <= first <= "\uffef":
            series = r"\bfseries " if bold else ""
            for family, characters in groupby(run, key=cjk_font):
                parts.append(
                    "{\\" + family + "font " + series + "".join(characters) + "}"
                )
        else:
            parts.append("".join(TEX_ESCAPES.get(char, char) for char in run))
    return "".join(parts)


def assign_colors(panels: list[Panel], glosses: dict) -> None:
    """Alternate light/dark bands left to right, retaining script-specific hues."""
    palettes = {"Latin": plt.cm.Reds, "Chinese": plt.cm.Blues}
    shades = (0.48, 0.75)
    for panel in panels:
        panel.colors = [
            palettes.get(
                glosses[str(region["token_id"])]["script_class"], plt.cm.Greys
            )(shades[index % 2])
            for index, region in enumerate(panel.regions)
        ]


# Drawing: profiles, direct minimum-region labels, and shared figure labels.


def draw_regions(ax, panel: Panel) -> None:
    for region, color in zip(panel.regions, panel.colors):
        ax.axvspan(
            region["start"],
            region["end"],
            facecolor=color,
            alpha=REGION_ALPHA,
            linewidth=0,
            zorder=0,
        )


def draw_profile(ax, panel: Panel, limit: float, y_limits: tuple[float, float]):
    draw_regions(ax, panel)
    ax.plot(panel.angles, panel.speed, color=CURVE_COLOR, linewidth=0.65, zorder=3)
    ax.set_xlim(-limit, limit)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xticks([-limit, 0, limit])
    # Keep endpoint labels inside each panel with the compact column spacing.
    ax.xaxis.get_major_ticks()[0].label1.set_ha('left')
    ax.xaxis.get_major_ticks()[-1].label1.set_ha('right')
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color="white", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.locator_params(axis="y", nbins=4)
    ax.tick_params(axis="both", labelsize=font_size("tiny"), length=0, pad=1.5,
                   labelleft=True, labelbottom=True)
    ax.label_outer()
    title = AnchoredOffsetbox(
        loc="lower left",
        child=context_box(panel),
        frameon=False,
        pad=0,
        bbox_to_anchor=(0, 1),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(title)


def minimum_regions(panel: Panel) -> dict[int, int]:
    """Every displayed region containing a local minimum, not only the steering pair."""
    speed = panel.speed
    minima = np.flatnonzero((speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])) + 1
    selected = {}
    for index in minima:
        for key, region in enumerate(panel.regions):
            if region["start"] <= panel.angles[index] < region["end"] and (
                key not in selected or speed[index] < speed[selected[key]]
            ):
                selected[key] = index
    return selected


def text_area(text: str, *, highlight=False, grammatical=False) -> TextArea:
    label = r"\textsc{" + text.lower() + "}" if grammatical else latex_label(text)
    props = {"fontsize": font_size("tiny"), "fontweight": "normal"}
    if highlight:
        props["bbox"] = {
            "facecolor": "#fff0af",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.04",
        }
    return TextArea(label, textprops=props)


def highlighted_source(source: str, token: str):
    before, after = source.split(token, 1)
    children = []
    if before:
        children.append(text_area(before))
    children.append(text_area(token, highlight=True))
    if after:
        children.append(text_area(after))
    return HPacker(children=children, align="baseline", pad=0, sep=0)


def context_box(panel: Panel, title=None):
    """Native offset-box layout aligns source/gloss units without manual y offsets."""
    if panel.context_gloss:
        columns = []
        for unit in panel.context_gloss["units"]:
            marked = unit.get("highlight")
            source = (
                highlighted_source(unit["source"], marked)
                if marked
                else text_area(unit["source"])
            )
            gloss = text_area(
                unit["gloss"],
                highlight=bool(marked),
                grammatical=unit.get("grammatical", False),
            )
            columns.append(
                VPacker(children=[source, gloss], align="left", pad=0, sep=0)
            )
        aligned = HPacker(
            children=columns, align="top", pad=0, sep=font_size("tiny") / 4
        )
        translation = text_area("‘" + panel.context_gloss["translation"] + "’")
        return VPacker(children=[aligned, translation], align="left", pad=0, sep=0)
    prefix, token = title or panel.title
    gap = font_size("tiny") / 4 if prefix.endswith(" ") else 0
    return HPacker(
        children=[text_area(prefix.rstrip()), text_area(token, highlight=True)],
        align="baseline",
        pad=0,
        sep=gap,
    )


def draw_minimum_labels(ax, panel: Panel, glosses: dict):
    """Place generation labels inside their regions, with no pins or leaders."""
    labels = []
    for number, key in enumerate(minimum_regions(panel)):
        region = panel.regions[key]
        meta = glosses[str(region["token_id"])]
        if meta.get("grammatical_gloss"):
            text = (
                latex_label(meta["display"].split(" → ")[0])
                + "\n"
                + r"\textsc{"
                + meta["grammatical_gloss"]
                + "}"
            )
        else:
            text = latex_label(meta["display"].replace(" → ", "\n"))
        midpoint = (region["start"] + region["end"]) / 2
        x = midpoint
        y = 0.94 if number % 2 == 0 else 0.74
        color = {"Latin": "#591719", "Chinese": "#0b304e"}.get(
            meta["script_class"], ".12"
        )
        label = ax.text(
            x,
            y,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=font_size("tiny"),
            clip_on=False,
            color=color,
            linespacing=1.05,
            zorder=5,
        )
        label.set_in_layout(False)
        renderer = ax.figure.canvas.get_renderer()
        box = label.get_window_extent(renderer)
        bounds = ax.get_window_extent(renderer)
        shift = max(bounds.x0 + 2 - box.x0, 0) - max(box.x1 - bounds.x1 + 2, 0)
        x += shift / bounds.width * (ax.get_xlim()[1] - ax.get_xlim()[0])
        label.set_x(x)
        labels.append(label)
    return labels


def draw_shared_labels(fig) -> None:
    fig.supylabel("Speed", rotation=0, fontsize=font_size("tiny"))
    fig.supxlabel(r"Angle $\theta$ (degrees)", fontsize=font_size("tiny"))


def draw_color_key(fig) -> None:
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(
                facecolor=plt.cm.Reds(0.65), alpha=REGION_ALPHA, label="English /\nLatin"
            ),
            Patch(facecolor=plt.cm.Blues(0.65), alpha=REGION_ALPHA, label="Chinese"),
            Patch(
                facecolor=plt.cm.Greys(0.65),
                alpha=REGION_ALPHA,
                label="Neutral /\nother",
            ),
        ],
        loc="outside right upper",
        borderaxespad=0,
        ncol=1,
        frameon=False,
        fontsize=font_size("tiny"),
        handlelength=0.8,
        handletextpad=0.5,
        handleheight=0.65,
        columnspacing=1.5,
    )


def validate_layout(fig, groups) -> None:
    """Reject overlapping labels and visible text outside the figure bounds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for artist in fig.findobj(AnchoredOffsetbox):
        if artist.get_visible():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -0.2 and box.y0 >= -0.2
            assert box.x1 <= fig.bbox.x1 + 0.2 and box.y1 <= fig.bbox.y1 + 0.2
    for group in groups:
        boxes = [text.get_window_extent(renderer) for text in group]
        for a, b in combinations(boxes, 2):
            overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
            overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            assert overlap_x <= 0.05 or overlap_y <= 0.05, "Annotation overlap"
        texts.extend(group)
    for ax in fig.axes:
        texts.append(ax.yaxis.label)
        for ticks, labels, limits in (
            (ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
            (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim()),
        ):
            texts.extend(
                text
                for pos, text in zip(ticks, labels)
                if limits[0] <= pos <= limits[1]
            )
    for text in texts:
        box = text.get_window_extent(renderer)
        assert (
            box.x0 >= -0.2
            and box.x1 <= fig.bbox.x1 + 0.2
            and box.y0 >= -0.2
            and box.y1 <= fig.bbox.y1 + 0.2
        ), (text.get_text(), box.bounds)


# Rendering entry point.


def render(
    folder: Path,
    figure: Path,
    style: Path,
    limit: float = ANGLE_LIMIT,
    english_only: bool = False,
) -> None:
    setup_style(style)
    english = load_inputs(folder)
    row_data = [("English contexts", english, make_panels(english, limit))]
    if not english_only:
        chinese = load_inputs(folder, "_chinese")
        np.testing.assert_array_equal(english.arrays["theta"], chinese.arrays["theta"])
        np.testing.assert_allclose(
            english.arrays["direction"], chinese.arrays["direction"], atol=1e-14, rtol=0
        )
        row_data.append(
            (
                "Chinese contexts",
                chinese,
                make_panels(chinese, limit, (0, 1, 2, 3), chinese=True),
            )
        )
    panels = [panel for _, _, row_panels in row_data for panel in row_panels]
    glosses = {
        token: meta for _, data, _ in row_data for token, meta in data.glosses.items()
    }
    assign_colors(panels, glosses)
    fig, axes = plt.subplots(
        len(row_data), 4, sharex=True, sharey=True, squeeze=False,
        figsize=(TEXT_WIDTH_TEX_POINTS / TEX_POINTS_PER_INCH,
                 (1.0225 if STYLE == "iclr" else 1.2) * len(row_data)),
        layout="constrained",
    )
    if STYLE == "goodfire":
        fig.get_layout_engine().set(w_pad=0.01, wspace=0)
    draw_shared_labels(fig)
    draw_color_key(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, panel in zip(axes.flat, panels):
        available = ax.get_window_extent(renderer).width
        for candidate in (panel.title, *panel.title_alternatives):
            artist = context_box(panel, candidate)
            artist.set_figure(fig)
            box = artist.get_bbox(renderer)
            if box.width <= available:
                panel.title = candidate
                break
        else:
            raise ValueError(
                f"Context {panel.title!r} needs shortening: {box.width:.2f} > {available:.2f} pixels at tiny size"
            )
    low = min(panel.speed.min() for panel in panels)
    high = max(panel.speed.max() for panel in panels)
    pad = 0.06 * (high - low)
    y_limits = (max(0, low - pad), high + pad)
    groups = []
    try:
        for ax, panel in zip(axes.flat, panels):
            draw_profile(ax, panel, limit, y_limits)
        fig.canvas.draw()
        for ax, panel in zip(axes.flat, panels):
            groups.append(draw_minimum_labels(ax, panel, glosses))
        validate_layout(fig, groups)
        figure.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(figure)
            fig.savefig(figure.with_suffix(".png"), dpi=300)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--figure", type=Path, default=FIGURE_DIR / "language_steering_speed.pdf"
    )
    parser.add_argument("--style", type=Path, default=PAPER / "styles/matplotlibrc")
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Render only the original English row.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=ANGLE_LIMIT,
        help="Symmetric displayed angle limit in degrees; source data retain ±90°.",
    )
    args = parser.parse_args()
    render(args.data, args.figure, args.style, args.limit, args.english_only)


if __name__ == "__main__":
    main()
