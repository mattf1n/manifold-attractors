# Manifold attractors: figure reproduction

Standalone source package for paper commit
`ed8b89bc8344db678eb5e360a1893937d3b94a1f`: all **16 included PDFs, two inline
TikZ artworks, and three active tables**, main text and appendix. Figures support
ICLR and Goodfire styles; table values are shared.
No dependency on the original project/paper checkout, archived experiment data,
or precomputed activations. Only required numerical kernels and renderers are
retained. Public weights/corpora are downloaded; numerical intermediates are
computed locally. Literal prompts, human-authored labels, and the source fish
GIF are source inputs, not precomputed experiment results.

**Target: scientific reproduction, not mandatory pixel identity.** Inconsequential
hardware-rounding and typographic differences are acceptable; changing sampled
tokens or numerical conclusions is not. **All figures are reproduced in both
styles and scientifically validated, and all three tables are complete.** The
norm-growth and pair-barrier tables match exactly. Adaptive-steering differences
are at most **0.01712 percentage points**, traced to two Unicode-version-dependent
Han-prefix token labels; all greedy counts, tuned parameters and method rankings
match. Results and validation are below. Missing original TeX
preambles are replaced by explicitly documented reconstructions; source and
corpus-selection provenance limitations remain in [AUDIT.md](AUDIT.md).

## Setup and commands

Install [uv](https://docs.astral.sh/uv/) and use this source checkout (`uv sync`
creates an editable installation). Generation requires substantial CPU RAM and
a CUDA GPU suitable for float32/float64 8B models. Accept the gated Meta model
licenses and authenticate with Hugging Face before downloading weights.
Rendering needs LuaLaTeX and the TeX/font packages listed in [RENDER.md](RENDER.md).
That document now includes a rootless, hash-pinned TeX Live 2025 / LuaHBTeX 1.21
setup and a locked PDF rasterizer.

```sh
uv sync --locked --extra compute --extra render
uv run reproduce list
uv run reproduce plan                     # full dependency graph, no computation
uv run --extra compute reproduce run data --resume  # numerical generation only
make smoke                                # source GIF + deterministic field; no models/TeX
make test                                 # CPU/tiny-model integration and unit tests
```

After activating the TeX and rasterizer environment from `RENDER.md`:

```sh
uv run --extra compute reproduce run paper --resume   # figures + tables
uv run --extra compute reproduce run figures --style goodfire --resume
uv run --extra compute reproduce run tables --resume  # tables alone
uv run reproduce check                    # optional STRICT pixel-reference comparison
```

Documented reconstructed typography is allowed by default; numerical checks
remain mandatory. `--require-original-styles` enables strict provenance mode
and fails until the four original preambles are recovered. The older explicit
`--allow-reconstructed-styles` flag remains supported. A single task ID may replace `data`, e.g.
`reproduce run data-great_circle_norm_grid-llama3 --resume` inside the compute
environment. Optional SLURM submission, from this checkout:

```sh
sbatch scripts/reproduce.sbatch iclr data
```

The template requests one GPU for up to three days, using cluster-default
partition/CPU/RAM settings. Adjust the wall-time to your resources. No jobs are
submitted automatically.

## Reproducibility contract

- `configs/{core,maps,render,classifier_table,steering_table,render_tables}.json` is the explicit task DAG.
  `plan` prints actual commands; no hidden call to the old project is needed.
- Models and available corpora are revision-pinned; input hashes and frozen
  token records guard against silently changing examples. Human-reviewed labels
  remain source labels rather than invented translations of regenerated tokens.
- `uv.lock` locks modern computation and two **incompatible** plotting stacks.
  The runner automatically uses isolated `render` / `render-bubble` extras;
  invoking a script as a module alone would not honor its PEP 723 pins.
- Refusal-vector extraction downloads only pinned public author code and input
  splits, **not** published directions/activations/completions. Its historical
  Python 3.10 / Torch 2.3 / Transformers 4.41 environment is separately pinned.
  All other computation uses the modern compute environment.
- `--resume` skips a task only after checking source/config fingerprints and
  declared output hashes. Scientific data are shared between styles. Generator
  checkpoints retain their own numerical-contract checks. Use an empty `data/`
  for a genuinely fresh full run; no generated data are shipped in Git.
- `reproduce stage TASK` executes only a named new stage after verifying every
  completed prerequisite's recorded output hashes. This supports adding table
  scoring to freshly computed native pools without overwriting figure inputs;
  it does not certify that upstream code matches a subsequently edited version.
- Task logs/timing/provenance live in `runs/`; generated outputs in `data/`,
  `fig/`, and `tab/`; downloaded author source in `external/`. These are Git-ignored.
  Deliberately published figure/table outputs live under `docs/`.

The optional strict `check` compares canonical decoded PNG pixels for the 15 scientific figures and
embedded image pixels/page geometry for the simple fish PDF. Original file
hashes are also retained as provenance. PDF timestamps and PNG encoding are not
scientific differences. No original standalone reference was supplied for the
two inline artworks, so they are reported **UNVERIFIED**, never silently passed.
Exit codes: 1 for missing/different artwork, 2 for incomplete reference coverage,
0 only for complete matching coverage. A nonzero strict pixel result is not
itself a scientific failure: see the separate scientific-validation evidence.
Pixel agreement alone is not proof of a from-scratch execution; keep run provenance too.

## What was actually checked

**All74 tests pass in a fresh Git clone** with isolated locked environments and
no `data/`, `fig/`, `tab/`, `runs/`, or `external/` directories. The four table
formatter tests also pass separately under their exact locked environment.
Regression coverage includes download scope, scientific sampling, source-input
contracts and safe figure/table publication. CPU mathematical and
tiny-random-model integration tests exercise the core and advanced generators
without pretrained weights or old result arrays. The source
GIF regenerates historical fish PDF pixels/placement exactly; the schematic's
1,304-vector table matches its historical SHA256. A public Pile input was
independently downloaded into an empty cache and matched its pinned hash; the
pinned author source was also fetched into a fresh temporary checkout without
published run artifacts. Both isolated plotting environments resolve and their
CLIs work. Two independent rootless TeX installations produced identical smoke
images with no missing-glyph or font-substitution warnings. These checks do not
substitute for the full-size numerical and per-artwork comparisons below.

Protocol details and limitations: [CORE.md](CORE.md), [MAPS.md](MAPS.md),
[RENDER.md](RENDER.md), [TABLES.md](TABLES.md), [AUDIT.md](AUDIT.md).
Unused/commented diagnostic tables and plots, and full-manuscript pagination,
remain out of scope. All three active tables below come from completed fresh
experiments, including all20,000 adaptive-steering test examples and the full
finite-search oracle.

<!-- reproduced-figures:start -->
## Reproduced figures

These are **newly generated results**, not copies of the paper figures. Numerical experiments use the standalone package; typography uses explicitly reconstructed styles.

Published **36/36** style-specific artworks; **10** pass the optional exact image check. **Pixel identity is not required**: small hardware-rounding and typographic differences are acceptable when scientifically inconsequential. Token selection, numerical results and conclusions must remain faithful. The per-artwork report distinguishes scientific validation from pixel identity; inline artworks lack historical image references.

[Per-artwork hashes and validation](docs/figures/validation.json). [Protocols and limitations](AUDIT.md).

<details open>
<summary>ICLR figures</summary>

### Intro

Scientific validation: **source reproduced**. No historical reference — unverified. [PDF](docs/figures/iclr/intro.pdf)

![Intro](docs/figures/iclr/intro.png)

### Sphere probe

Scientific validation: **source reproduced**. No historical reference — unverified. [PDF](docs/figures/iclr/sphere_probe.pdf)

![Sphere probe](docs/figures/iclr/sphere_probe.png)

### Great circle norm grid

Scientific validation: **exact numerics**. Exact reference pixels. [PDF](docs/figures/iclr/great_circle_norm_grid.pdf)

![Great circle norm grid](docs/figures/iclr/great_circle_norm_grid.png)

### Great circle synthetic grid

Scientific validation: **exact numerics**. Exact reference pixels. [PDF](docs/figures/iclr/great_circle_synthetic_grid.pdf)

![Great circle synthetic grid](docs/figures/iclr/great_circle_synthetic_grid.png)

### Norm growth trajectory starts

Scientific validation: **accepted**. Exact reference pixels. [PDF](docs/figures/iclr/norm_growth_trajectory_starts.pdf)

![Norm growth trajectory starts](docs/figures/iclr/norm_growth_trajectory_starts.png)

### Synthetic profile trajectory

Scientific validation: **exact numerics**. Exact reference pixels. [PDF](docs/figures/iclr/synthetic_profile_trajectory.pdf)

![Synthetic profile trajectory](docs/figures/iclr/synthetic_profile_trajectory.png)

### Great circle pair speed grid

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/iclr/great_circle_pair_speed_grid.pdf)

![Great circle pair speed grid](docs/figures/iclr/great_circle_pair_speed_grid.png)

### Triple area heatmaps

Scientific validation: **inconsequential numerical differences**. Reference pixels differ. [PDF](docs/figures/iclr/triple_area_heatmaps.pdf)

![Triple area heatmaps](docs/figures/iclr/triple_area_heatmaps.png)

### Bubble coords

Scientific validation: **exact numerics**. Exact reference pixels. [PDF](docs/figures/iclr/bubble_coords.pdf)

![Bubble coords](docs/figures/iclr/bubble_coords.png)

### Refusal steering heatmaps

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/iclr/refusal_steering_heatmaps.pdf)

![Refusal steering heatmaps](docs/figures/iclr/refusal_steering_heatmaps.png)

### Language area heatmaps

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/iclr/language_area_heatmaps.pdf)

![Language area heatmaps](docs/figures/iclr/language_area_heatmaps.png)

### Language steering speed

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/iclr/language_steering_speed.pdf)

![Language steering speed](docs/figures/iclr/language_steering_speed.png)

### Great circle norm grid context

Scientific validation: **exact numerics**. Exact reference pixels. [PDF](docs/figures/iclr/great_circle_norm_grid_context.pdf)

![Great circle norm grid context](docs/figures/iclr/great_circle_norm_grid_context.png)

### Norm growth trajectory starts appendix

Scientific validation: **accepted**. Exact reference pixels. [PDF](docs/figures/iclr/norm_growth_trajectory_starts_appendix.pdf)

![Norm growth trajectory starts appendix](docs/figures/iclr/norm_growth_trajectory_starts_appendix.png)

### Great circle speed grid

Scientific validation: **exact numerics**. Exact reference pixels. [PDF](docs/figures/iclr/great_circle_speed_grid.pdf)

![Great circle speed grid](docs/figures/iclr/great_circle_speed_grid.png)

### Great circle synthetic speed grid

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/iclr/great_circle_synthetic_speed_grid.pdf)

![Great circle synthetic speed grid](docs/figures/iclr/great_circle_synthetic_speed_grid.png)

### Great circle pair speed grid appendix

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/iclr/great_circle_pair_speed_grid_appendix.pdf)

![Great circle pair speed grid appendix](docs/figures/iclr/great_circle_pair_speed_grid_appendix.png)

Fish component: [Exact embedded image and placement](docs/figures/iclr/dead_fish_sequence.pdf).

</details>

<details >
<summary>Goodfire figures</summary>

### Intro

Scientific validation: **source reproduced**. No historical reference — unverified. [PDF](docs/figures/goodfire/intro.pdf)

![Intro](docs/figures/goodfire/intro.png)

### Sphere probe

Scientific validation: **source reproduced**. No historical reference — unverified. [PDF](docs/figures/goodfire/sphere_probe.pdf)

![Sphere probe](docs/figures/goodfire/sphere_probe.png)

### Great circle norm grid

Scientific validation: **exact numerics**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_norm_grid.pdf)

![Great circle norm grid](docs/figures/goodfire/great_circle_norm_grid.png)

### Great circle synthetic grid

Scientific validation: **exact numerics**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_synthetic_grid.pdf)

![Great circle synthetic grid](docs/figures/goodfire/great_circle_synthetic_grid.png)

### Norm growth trajectory starts

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/norm_growth_trajectory_starts.pdf)

![Norm growth trajectory starts](docs/figures/goodfire/norm_growth_trajectory_starts.png)

### Synthetic profile trajectory

Scientific validation: **exact numerics**. Reference pixels differ. [PDF](docs/figures/goodfire/synthetic_profile_trajectory.pdf)

![Synthetic profile trajectory](docs/figures/goodfire/synthetic_profile_trajectory.png)

### Great circle pair speed grid

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_pair_speed_grid.pdf)

![Great circle pair speed grid](docs/figures/goodfire/great_circle_pair_speed_grid.png)

### Triple area heatmaps

Scientific validation: **inconsequential numerical differences**. Reference pixels differ. [PDF](docs/figures/goodfire/triple_area_heatmaps.pdf)

![Triple area heatmaps](docs/figures/goodfire/triple_area_heatmaps.png)

### Bubble coords

Scientific validation: **exact numerics**. Reference pixels differ. [PDF](docs/figures/goodfire/bubble_coords.pdf)

![Bubble coords](docs/figures/goodfire/bubble_coords.png)

### Refusal steering heatmaps

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/refusal_steering_heatmaps.pdf)

![Refusal steering heatmaps](docs/figures/goodfire/refusal_steering_heatmaps.png)

### Language area heatmaps

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/language_area_heatmaps.pdf)

![Language area heatmaps](docs/figures/goodfire/language_area_heatmaps.png)

### Language steering speed

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/language_steering_speed.pdf)

![Language steering speed](docs/figures/goodfire/language_steering_speed.png)

### Great circle norm grid context

Scientific validation: **exact numerics**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_norm_grid_context.pdf)

![Great circle norm grid context](docs/figures/goodfire/great_circle_norm_grid_context.png)

### Norm growth trajectory starts appendix

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/norm_growth_trajectory_starts_appendix.pdf)

![Norm growth trajectory starts appendix](docs/figures/goodfire/norm_growth_trajectory_starts_appendix.png)

### Great circle speed grid

Scientific validation: **exact numerics**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_speed_grid.pdf)

![Great circle speed grid](docs/figures/goodfire/great_circle_speed_grid.png)

### Great circle synthetic speed grid

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_synthetic_speed_grid.pdf)

![Great circle synthetic speed grid](docs/figures/goodfire/great_circle_synthetic_speed_grid.png)

### Great circle pair speed grid appendix

Scientific validation: **accepted**. Reference pixels differ. [PDF](docs/figures/goodfire/great_circle_pair_speed_grid_appendix.pdf)

![Great circle pair speed grid appendix](docs/figures/goodfire/great_circle_pair_speed_grid_appendix.png)

Fish component: [Exact embedded image and placement](docs/figures/goodfire/dead_fish_sequence.pdf).

</details>

<!-- reproduced-figures:end -->

<!-- reproduced-tables:start -->
## Reproduced tables

These values come from fresh experiments, not the old paper tables. [Protocols](TABLES.md). [Validation](docs/tables/validation.json).

### Norm growth classifier

Matches the historical formatted table exactly. [LaTeX](docs/tables/norm_growth_classifier.tex) · [Numeric JSON](docs/tables/norm_growth_classifier.json)

| Class | Model | 1 | 4 | 6 | 8 | 11 | 14 | 16 | 18 | 21 | 24 | 26 | 28 | 31 | 34 | 36 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian | Llama 3.1 8B | 0.851 | 0.963 | 0.915 | 0.897 | 0.789 | 0.818 | 0.963 | 0.983 | 0.981 | 0.899 | 0.869 | 0.759 | 0.555 | — | — |
| Gaussian | OLMo 3 7B | 0.618 | 0.808 | 0.887 | 0.890 | 0.893 | 0.933 | 0.947 | 0.954 | 0.997 | 0.973 | 0.935 | 0.928 | 0.742 | — | — |
| Gaussian | Qwen 3 8B | 0.765 | 0.621 | 0.824 | 0.865 | 0.923 | 0.833 | 0.690 | 0.740 | 0.830 | 0.923 | 0.965 | 0.955 | 0.836 | 0.669 | 0.478 |
| Shuffled | Llama 3.1 8B | 0.927 | 0.999 | 0.991 | 0.956 | 0.980 | 0.999 | 1.000 | 0.998 | 1.000 | 0.980 | 0.950 | 0.928 | 0.835 | — | — |
| Shuffled | OLMo 3 7B | 0.525 | 0.887 | 0.954 | 0.913 | 0.927 | 0.956 | 0.974 | 0.986 | 0.990 | 0.975 | 0.903 | 0.902 | 0.866 | — | — |
| Shuffled | Qwen 3 8B | 0.997 | 0.906 | 0.969 | 0.908 | 0.943 | 0.953 | 0.871 | 0.957 | 0.995 | 0.998 | 0.995 | 0.981 | 0.944 | 0.786 | 0.503 |

### Pair barrier validation

Matches the historical formatted table exactly. [LaTeX](docs/tables/pair_barrier_validation.tex) · [Numeric JSON](docs/tables/pair_barrier_validation.json)

| Synthetic class | Llama 3.1 8B | OLMo 3 7B | Qwen 3 8B |
| --- | --- | --- | --- |
| Gaussian | 0.963 | 0.984 | 0.938 |
| Shuffled | 0.980 | 0.996 | 0.986 |

### Adaptive steering

Scientifically reproduced; documented differences are inconsequential. Maximum cell difference: 0.0171 percentage points. [LaTeX](docs/tables/adaptive_steering.tex) · [Numeric JSON](docs/tables/adaptive_steering.json)

| Method | Non-Chinese ↓ / Mass (%) / ID | Non-Chinese ↓ / Mass (%) / OOD | Non-Chinese ↓ / Greedy (%) / ID | Non-Chinese ↓ / Greedy (%) / OOD | Chinese ↑ / Mass (%) / ID | Chinese ↑ / Mass (%) / OOD | Chinese ↑ / Greedy (%) / ID | Chinese ↑ / Greedy (%) / OOD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Untuned | 20.10 | 21.48 | 17.16 | 18.74 | 61.65 | 63.84 | 59.23 | 58.61 |
| Tuned | 13.82 | 10.43 | 11.18 | 8.18 | 69.51 | 77.64 | 68.07 | 74.81 |
| Adaptive | 12.81 | 9.17 | 10.55 | 6.61 | 70.71 | 79.51 | 69.10 | 77.94 |
| Oracle | 9.39 | 7.66 | 4.43 | 3.92 | 73.20 | 80.33 | 79.13 | 82.36 |

<!-- reproduced-tables:end -->
