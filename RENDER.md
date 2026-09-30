# Standalone manuscript rendering

Scope: every figure artwork referenced by `main.tex` and `appendix.tex` at paper
commit **ed8b89bc8344db678eb5e360a1893937d3b94a1f**: 16 PDF inclusions plus the two
inline TikZ diagrams. Tables and the manuscript page layout are not render targets.
No paper checkout, model code, GPU, or original absolute data path is needed.
All numerical inputs must be generated separately into this repository's `data/`;
none are distributed here. The only binary asset is the literal source fish GIF.

## Reproduction caveat: missing original style files

The tracked renderers require four `styles/*.tex` files absent from both the
tracked paper tree/history and the supplied `paper-archive-20260930-214925`
archive. The included replacements are **reconstructions**, not recovered
originals: ICLR typography comes from `iclr.tex`; Goodfire's Cabin/Noto Sans
and caption math come from `goodfire.cls` and `goodfire.tex`.
Scene geometry, samples, colors, layout constants, and drawing operations are
retained. **Exact font metrics / pixel-identical scientific figures cannot be
certified until the original styles are recovered and compared.** No silent
Matplotlib font fallback is used: plotting requires LuaLaTeX.

The pipeline blocks scientific rendering by default while this provenance gap
remains. Pass `--allow-reconstructed-styles` only when deliberately accepting the
reconstructed typography. The pipeline preflights `lualatex` before running any
GPU dependencies, so missing TeX fails before expensive computation. Standalone
module commands below bypass that pipeline acknowledgment gate; they still
require LuaLaTeX and use the explicitly disclosed reconstructed styles.

## Environment

Run commands from this repository root. `uv` creates isolated environments below;
no editable install is required. In particular, the source scripts have two
incompatible sets of exact plotting pins. Running all of them in one environment
would lose that original distinction. The PEP 723 headers remain as provenance;
`python -m` does not itself install those dependencies.

Install a full TeX Live distribution with `lualatex` on `PATH`, including NewTX,
fontspec, Cabin, Noto Sans, Inconsolata, mathastext, siunitx, standalone, adjustbox,
PGFPlots and the TikZ hobby library. Language-token labels also require:

- ICLR: `Tempora-Regular.otf`, `Tempora-Bold.otf`, `cmunrm.otf`, `cmunbx.otf`,
  `gbsn00lp.ttf`, `bsmi00lp.ttf` (TeX Live fonts).
- Goodfire: fontconfig-visible `Noto Sans`, `Noto Sans CJK SC`, `Noto Sans CJK TC`.

Define these shell helpers once:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export FIGURE_STYLE=iclr                 # repeat below with goodfire
mkdir -p "fig/$FIGURE_STYLE" fig/shared data

render() {
  local name="$1"; shift
  uv run --no-project --python 3.13 \
    --with numpy==2.2.6 --with matplotlib==3.10.9 \
    python -m "manifold_attractors.render.$name" "$@"
}
render_bubbles() {
  uv run --no-project --python 3.13 \
    --with numpy==2.5.3 --with matplotlib==3.11.2 \
    --with pandas==3.0.6 --with pyarrow==25.0.1 --with scipy==1.18.1 \
    python -m manifold_attractors.render.bubble_coords "$@"
}
render_fish() {
  uv run --no-project --python 3.13 --with 'pillow>=12.0' \
    python -m manifold_attractors.render.dead_fish_sequence "$@"
}
```

`FIGURE_STYLE` is read once on module import; use a fresh process for each style.
ICLR width is 5.5 inches; Goodfire width is `8.5 - 5/2.54` inches. Per-panel
widths/heights and half-width pair panels retain the manuscript settings.
Default inputs resolve inside this checkout's `data/`; defaults for scientific
PDFs resolve to `fig/{iclr,goodfire}/`. Explicit relative CLI paths resolve from
the current directory. `--style` (where present) means a Matplotlib rc file,
not the ICLR/Goodfire selector. Every PDF renderer accepts `--figure`; historical
`--output` and fish `--source` options remain accepted aliases. Fish additionally
accepts `--data` for its GIF. Two-basin `--data` names its **output** table.

## Exact manuscript recipes

`configs/render.json` contains the same recipes as 19 pipeline tasks (16 PDF
inclusions, the vector-table generator, and two standalone TikZ artworks).
Task arguments/output paths use `{style}`; the pipeline sets `FIGURE_STYLE`.
Scientific/TikZ tasks select the isolated `render` environment; bubbles select
`render-bubble`; fish and the pure-Python field generator use the current interpreter.

The directory/filenames below are part of the figure specification. In particular,
do not substitute the older default datasets for the selected trajectory, pair,
or synthetic-grid runs.

### Main figures

```bash
render great_circle_norm_grid \
  --data data/great_circle_norm_grid \
  --figure "fig/$FIGURE_STYLE/great_circle_norm_grid.pdf"

render great_circle_synthetic_grid \
  --data data/great_circle_synthetic_grid_best_layers \
  --llama-data data/great_circle_synthetic_grid_llama21/llama3.npz \
  --figure "fig/$FIGURE_STYLE/great_circle_synthetic_grid.pdf"

render norm_growth_trajectory_starts \
  --data data/norm_growth_trajectory_starts_all_layers \
  --starts all --height 1.25 --models llama3 \
  --figure "fig/$FIGURE_STYLE/norm_growth_trajectory_starts.pdf"

render synthetic_profile_trajectory \
  --data data/synthetic_profile_trajectory \
  --figure "fig/$FIGURE_STYLE/synthetic_profile_trajectory.pdf"

render great_circle_pair_speed_grid \
  --data data/pair_barrier_validation --models llama3 --half-width \
  --figure "fig/$FIGURE_STYLE/great_circle_pair_speed_grid.pdf"

render triple_area_heatmaps \
  --data data/triple_area_heatmaps_6blocks \
  --figure "fig/$FIGURE_STYLE/triple_area_heatmaps.pdf"

render_bubbles \
  --data data/bubble_coords.parquet --extra-data data/bubble_coords_extra.parquet \
  --figure "fig/$FIGURE_STYLE/bubble_coords.pdf"

render refusal_steering_heatmaps \
  --data data/refusal_steering_heatmaps \
  --figure "fig/$FIGURE_STYLE/refusal_steering_heatmaps.pdf"

render language_area_heatmaps \
  --data data/language_area_heatmaps \
  --figure "fig/$FIGURE_STYLE/language_area_heatmaps.pdf"

render language_steering_speed \
  --data data/language_steering_speed \
  --figure "fig/$FIGURE_STYLE/language_steering_speed.pdf"
```

### Appendix figures

```bash
render great_circle_norm_grid \
  --data data/great_circle_norm_grid_context \
  --figure "fig/$FIGURE_STYLE/great_circle_norm_grid_context.pdf"

render norm_growth_trajectory_starts \
  --data data/norm_growth_trajectory_starts_all_layers \
  --starts all --height 2.2 --models olmo3 qwen3 \
  --figure "fig/$FIGURE_STYLE/norm_growth_trajectory_starts_appendix.pdf"

render great_circle_speed_grid \
  --data data/great_circle_speed_grid \
  --figure "fig/$FIGURE_STYLE/great_circle_speed_grid.pdf"

render great_circle_synthetic_speed_grid \
  --data data/great_circle_synthetic_speed_grid \
  --figure "fig/$FIGURE_STYLE/great_circle_synthetic_speed_grid.pdf"

render great_circle_pair_speed_grid \
  --data data/pair_barrier_validation --independent-y --models olmo3 qwen3 \
  --figure "fig/$FIGURE_STYLE/great_circle_pair_speed_grid_appendix.pdf"
```

Keep the NPZ files **and JSON companions** emitted by the generators. Curve
renderers read/hash `{llama3,olmo3,qwen3}.{npz,json}` for the selected models;
synthetic-grid uses the extra Llama NPZ above. Synthetic-profile needs `llama3.npz`
and `shuffled.npz`. Triple and language heatmaps consume `map-*.npz`; refusal
heatmaps additionally need `selection.json` and its referenced NPZ files.
Language steering consumes `llama3.npz`, `regions.json`, `selection.json`, and
`glosses.json`, plus matching `_chinese` filenames for its second row. Preserve
the complete generated directory. Bubble rendering requires both parquet files,
including supplemental natural sample 5.

Language gloss metadata has schema `{"tokens": {"TOKEN_ID": {"text": "exact decoded token",
"display": "display label", "script_class": "Latin"}}}`. `script_class` selects
Latin/red, Chinese/blue, or other/gray region colors. `grammatical_gloss` per token
and top-level `interlinear_contexts` are optional in the renderer schema. The
packaged generator preserves the original hand-authored `glosses.json` and
`glosses_chinese.json`, including grammar and interlinear annotations, as source
assets. It strictly matches freshly generated token IDs/text/pieces and the
selected context quartet; changed or unknown labels/contexts fail rather than
falling back to literal labels. Source annotation byte hashes match the originals.
Missing gloss files also deliberately fail in the renderer. Custom literal-token
labels are structurally accepted, but would not reproduce the original annotations.

### Introduction: fish media and actual TikZ field

The introduction is **not** a Matplotlib basin plot. It juxtaposes a shared
fish montage with a PGFPlots quiver diagram embedded directly in `main.tex`.
The historically named `two_basin_field` generator actually makes **three**
rounded basins; its name is retained. Its deterministic table is generated, not
bundled. Run once for the common inputs, then compile for each profile:

```bash
render_fish --data assets/A-Dead-Fish-Swims.gif \
  --figure fig/shared/dead_fish_sequence.pdf
uv run --no-project --python 3.13 \
  python -m manifold_attractors.render.two_basin_field \
  --output data/two_basin_field.dat
render inline_tikz intro \
  --data data/two_basin_field.dat --fish fig/shared/dead_fish_sequence.pdf \
  --figure "fig/$FIGURE_STYLE/intro.pdf"
render inline_tikz sphere_probe \
  --figure "fig/$FIGURE_STYLE/sphere_probe.pdf"
```

`assets/intro.tex` retains the original `.49\\linewidth` panels, 0.177 arrow
scale, gray background and fixed bounds. `assets/sphere_probe.tex` preserves the
second inline TikZ drawing. `inline_tikz` adds only a standalone document wrapper,
profile width/font setup, and input path substitutions; output excludes manuscript
captions. It also writes `.tex`; add `--tex-only` to inspect without LuaLaTeX.

The montage selects frames 54, 56, 58, 60, 62, 64; crops `(45,65,345,175)`;
rotates clockwise; maps white to RGB 240; and adds the original gray padding.
Its native 5.5-inch PDF is shared by both styles and scaled by the intro TikZ
wrapper exactly as in the manuscript. Most renderers also emit PNG previews and
`.render.json` provenance. Legacy JSON `manuscript`/`manuscript_sha256` fields now
refer to the local profile font preamble, not an external manuscript.

## Provenance audit

`assets/render-sources.sha256` records original source hashes at the commit above.
All 16 copied Python files originate in tracked `scripts/fig/` with the same
basenames, now under `src/manifold_attractors/render/`. Only necessary helpers
`figure_style.py` and `legacy_style.py` were retained; unused exploratory figures,
model/experiment modules, logos, generated PDFs and numerical data were omitted.
`styles/matplotlibrc` and the GIF are byte-identical copies of tracked source.
The two TikZ snippets originate in `main.tex` lines 11–42 and 153–168;
`inline_tikz.py` and the package initializer are new glue.

Adaptations: relative package imports, repository-local paths, output aliases,
no cwd/sys.path manipulation, bubble's top-level code moved under `main()` for
safe import, and a CLI on the unchanged two-basin generator. Scientific draw
logic is unchanged. Font preambles were reconstructed as disclosed above.
The source bubble renderer itself already reconstructs missing historical
surface shading; this package retains that ed8b89b reconstruction rather than
claiming to recover the earlier missing shader.

The GIF's scientific reference in the source is Beal et al. (2006), *Passive
propulsion in vortex wakes*, Figure 4 (printed page 391), not apparatus Figure 8.
The checked-in GIF contains no independently verified redistribution license or
source URL; it is carried forward as user-supplied literal source media, not
relicensed by this package. The selected GIF times are not asserted to match
published experimental timestamps.

## Verification performed

- All 18 render/helper modules parsed and imported in an isolated environment;
  historical-pin environment imports also passed (excluding the separate bubble environment).
- The pipeline loaded all 19 rendering tasks; all 16 reference-manifest figures
  were covered, style paths expanded correctly, and intro dependency order passed.
- All 15 executable module CLIs passed `--help` under both `FIGURE_STYLE` values.
- Deterministic field generation produced 1,304 vectors; table bytes exactly
  matched the unchanged ed8b89b algorithm.
- Fish rendering produced PDF/PNG; PNG bytes exactly matched ed8b89b from the
  original GIF in the same Pillow environment. Integration tests additionally
  verified fresh PDF pixels, page box and content stream against the old shared PDF.
- Integration tests verified the field SHA256 against the tracked `.dat` reference;
  media regression tests are in `tests/test_media.py`.
- Locked isolated `render` and `render-bubble` environments installed their exact
  pins and passed CLI `--help` checks.
- Standalone TeX source generation succeeded for both diagrams and both styles.
- No paper/project absolute paths or imports remain in render code.

Test outputs were written outside the repository. LuaLaTeX was unavailable on the
migration host, so scientific PDF and TikZ compilation/font comparisons remain
**unverified**. No generated data or PDFs are included as a substitute for those
tests. Basic checks that do not require TeX can be repeated with the commands
above for the field, fish, `--help`, and `inline_tikz --tex-only`.
