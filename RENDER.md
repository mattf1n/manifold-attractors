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
originals: ICLR typography comes from `iclr.tex`; Goodfire body typography
comes from `goodfire.cls`/`goodfire.tex`, while scientific math and mono font
routing now follow the original PDF resources documented in `AUDIT.md`.
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

### Rootless pinned TeX and PDF rasterizer

On Linux x86_64, with `curl`, `perl`, `tar`, `xz`, `unzip`, `sha256sum`, and
`fc-cache` available:

```bash
bash scripts/setup-tex.sh
source "$HOME/.local/texlive/2025/env.sh"
# micromamba is a user-space conda installer; no sudo is used.
micromamba create -y -p "$HOME/.local/share/manifold-rasterizer" \
  --file scripts/tex-rasterizer-linux-64.lock
export PATH="$HOME/.local/share/manifold-rasterizer/bin:$PATH"
uv run --locked --isolated --extra render python scripts/smoke-tex.py
```

The setup script accepts an optional installation prefix. It installs only into
that prefix and `~/.local/share/fonts/manifold-attractors`, never system paths or
shell startup files. Source `env.sh` explicitly in batch jobs too. It records
`packages.tsv` (TeX package revisions) and `font-engine.sha256` under the prefix.
The rasterizer lock contains exact conda URLs/checksums for **Poppler 25.07.0**
and its dependencies. This pins a **new comparison rasterizer**, not the unknown
historical PNG converter. Matplotlib's PGF PNG export needs `pdftocairo` on PATH.

TeX packages come from the frozen **TeX Live 2025 tlnet-final** repository; its
installer and package-database SHA256s are checked. Its signing key has expired;
the pinned database digest is intentional. TL2025-final ships **LuaHBTeX 1.22**,
so the script overlays the SHA256-checked **1.21.0** binary from the 2025-03-09
archive, then rebuilds `lualatex.fmt`. **Do not run `tlmgr update --all`**: that
would undo the engine pin. Re-run setup to restore it. This reproducible hybrid
matches the observed engine version, but the original TeX package-update snapshot
is unknown; it is not claimed to recover that workstation exactly.

Installed font constraints from original scientific PDF resources:

| Font | Installed source/version |
|---|---|
| Cabin | TL2025 `Cabin-Regular.otf`, **3.001** |
| Goodfire math | upstream `NotoSansMath-v3.000`, googlefonts hinted TTF, **3.000; ttfautohint(v1.8.4.7-5d5b)** |
| Mono labels | TL2025 `lmmono8-regular.otf`, **2.004** |
| ICLR text/math | TL2025 NewTX / TeXGyreTermesX **2.004**, NewTXMI, txsys |
| ICLR Chinese | `arphic-ttf`: Bousung **2.11**, ShanHeiSun **2.10** |
| Goodfire Chinese | Noto CJK Sans **2.004**, SC/TC regular, upstream commit `523d033d6cb47f4a80c58a35753646f5c3608a78` |
| Other token scripts | TL2025 Tempora, CM Unicode, and Noto Sans **2.005** |

The Noto CJK version is a pinned candidate, **not verified as the original**:
its original subsets retain copyright but not the complete version. External
font downloads have SHA256 checks in setup. Noto Sans Math TTF SHA256 is
`fe380412e05c2cca4b712644c58ed38ed7b1e3153bc483a8b38725fa487b8830`.
Fonts retain their upstream licenses; none are copied into this repository.

### Reconstruction changes and strict smoke

The missing four originals remain unrecovered; targeted archive/style-path and
paper-history checks found no new originals. `figure-math.tex` now uses
`unicode-math`/Noto Sans Math instead of the previously guessed Cabin/mathastext
routing. `figure-fonts.tex` selects LMMono8 for scientific mono labels instead
of Inconsolata. These changes follow **embedded-font evidence**, not guessed
pixel equality. Inline Goodfire diagrams still use manuscript-body mathastext,
not the scientific Noto math setup. Goodfire's reconstructed body preamble
restores article's `bx` bold series: fontspec's `b` otherwise caused explicit
`LGR/lmr/b/{n,it}` substitution warnings in the sphere diagram.

Both profile preambles set `\\tracinglostchars=3`: missing glyphs are fatal.
The smoke script retains `.tex`, `.stdout`, and `.log` files and **fails on
font-substitution warnings**, rather than accepting a PDF merely because TeX
returned zero. It builds fish/field media, both inline artworks in both styles,
Latin/math/mono/Greek/Cyrillic/SC/TC probes, and real Matplotlib PGF PDF/PNG plots.
Its `evidence.json` lists embedded PDF fonts and producer versions. Default
outputs: `runs/tex-smoke/`; override with `--output PATH`. It runs no models and
uses no archived numerical results. A passing smoke is not an equality test.

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

## Render verified completed results and publish the README

After numerical stages have produced their execution records, the following
renders ready figures without restarting expensive computation. Missing stages
are reported as pending; `--require-all` turns pending figures into an error.
The script checks every available numerical output against its recorded hash.

```bash
source "$HOME/.local/texlive/2025/env.sh"
export PATH="$HOME/.local/share/manifold-rasterizer/bin:$PATH"
# Run inside the generation checkout, once per profile.
uv run python scripts/render_results.py --style iclr --allow-reconstructed-styles
uv run python scripts/render_results.py --style goodfire --allow-reconstructed-styles
# Run inside the publication checkout; only recorded, hash-matching outputs copy.
uv run python scripts/publish_results.py --run /path/to/generation-checkout
```

`publish_results.py` embeds actual new PNGs and PDF links in a marked README
section and writes `docs/figures/validation.json`. It preserves per-artwork
strict pixel comparison status and explicitly labels missing inline references.
It never substitutes the historical paper artwork for a failed reproduction.
The `--require-all` option is available here too. Source media and old reference
hashes are inputs; the published galleries are **outputs**, not runtime caches.

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

The bullets above describe the initial migration checks. On 2026-09-30 the
rootless setup and strict smoke above additionally passed, both in the default
installation and a **fresh second TeX prefix**, using the locked render extra:

- LuaHBTeX **1.21.0 / TeX Live 2025** confirmed; PDF producers report LuaTeX-1.21.0.
- Fish PDF/PNG and all **four inline TikZ PDFs** compiled and rasterized.
- Both multilingual font probes and both real Matplotlib PGF PDF/PNG smoke plots
  passed; retained TeX logs contain no missing glyphs or font-substitution warnings.
- Goodfire probe embeds Cabin, NotoSansMath, LMMono8, NotoSans and both Noto CJK
  families. ICLR probe embeds TeXGyreTermesX, NewTXMI/txsys, LMMono8, Tempora,
  CMUSerif, Bousung and ShanHeiSun. Bold probe text also exercises the bold faces.
- Nine decoded PNG outputs matched pixel-for-pixel between the two TeX installs
  in the initial smoke; this checks installation repeatability, not history.
- The package suite after these changes passed **40 tests**.
- Field table SHA256:
  `990906b548647b1a8bfba8be80c9ff36fc6afa463df332673030d48979ecab90` (1,304 vectors).

Local evidence is in Git-ignored `runs/tex-smoke/` and `runs/tex-smoke-clean/`;
setup logs are under `runs/tex-smoke/`. No generated PDFs/data are committed.
Full numerical scientific figures and historical pixel equality were **not**
validated by this typography task. Original preambles and inline reference
artworks are still missing, so the exactness gate must remain in place.
