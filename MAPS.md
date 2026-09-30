# Advanced-figure generators

`src/manifold_attractors/maps/` computes new activations, directions, maps and
curves. It never imports the old project or reads its generated arrays. Bundled
`maps/inputs/*.json` contains **input texts, token IDs, selected positions,
configuration, corpus hashes and hand-authored figure-label annotations**,
transcribed from source metadata. Those are frozen experimental/editorial inputs,
not cached numerical model outputs. Source-file and input
hashes are in `maps/SOURCE_PROVENANCE.json`.

No GPU experiment was run while building this package. CPU mathematical tests,
a tiny randomly initialized Llama replay, import/CLI checks and compilation were
run. Numerical agreement of fresh full-size GPU outputs remains **unverified**.
Commands below are explicit computations, not submission scripts; run GPU stages
inside an appropriate allocation. Nothing submits a job or chooses a partition.

## Orchestration and dependencies

`configs/maps.json` defines ten tasks with explicit modules, arguments, outputs
and dependencies. Final targets are `data-triple_area_heatmaps`,
`data-language_area_heatmaps`, `data-language_steering_speed`,
`data-synthetic_profile_trajectory`, `data-bubble_coords`, and
`data-refusal_steering_heatmaps`. Triple/language tasks depend on their own fresh
`prepare-*` stages; trajectory depends on core `data-native_layers-llama3`,
which in turn depends on a fresh synthetic-reference pool. Refusal depends on
`data-refusal_direction`, whose wrapper prepares its own isolated historical
Python environment; no special pipeline environment handler is required.

Compute dependencies are NumPy, PyTorch, Transformers, Accelerate,
Huggingface Hub, datasets, SciPy and PyArrow (bubble tables). They are supplied
by the repository's compute extra. Refusal extraction instead uses the pinned
historical requirements file and external author checkout below. The task
manifest records external prerequisites and exactness limitations separately;
it does not pretend that missing corpus provenance is a downloadable cache.
Core helpers used are `core.inputs.model_snapshot` and
`experiments.great_circle_synthetic_grid.synthetic_states`, and
`experiments.pair_barrier_validation.enable_float64_math`. Minimal original
bubble forward helpers remain local in `maps/bubble_forward.py`.

## Commands

From the repository root, install the compute environment as described in
`README.md` (`uv sync --extra compute`). Commands use the normal HF cache and
`HF_TOKEN` if needed. Output directories are arbitrary and must be new; do not
mix prepared inputs from different experiments.

```sh
# Current triple-area figure: six blocks, not the older four-block figure.
uv run --extra compute python -m manifold_attractors.maps.area triple all \
  --blocks 6 --output data/triple_area_heatmaps_6blocks
# Optional older four-block experiment: same inputs/frame, different map.
uv run --extra compute python -m manifold_attractors.maps.area triple all \
  --blocks 4 --output data/triple_area_heatmaps

uv run --extra compute python -m manifold_attractors.maps.area area all \
  --output data/language_area_heatmaps
uv run --extra compute python -m manifold_attractors.maps.language_speed all \
  --output data/language_steering_speed

# First regenerate reference inputs using the core pipeline (CORE.md):
# great_circle_synthetic_grid/llama3.{npz,json} must be FRESH local outputs.
uv run --extra compute python -m manifold_attractors.experiments.prepare_native \
  --model llama3 --data-root data
uv run --extra compute python -m manifold_attractors.maps.synthetic_profile_trajectory \
  --reference data/norm_growth_classifier_layers/llama3.npz \
  --output data/synthetic_profile_trajectory

uv run --extra compute python -m manifold_attractors.maps.bubble_run --output data
```

Area/speed `all` can be split into `prepare` followed by `compute`. Preparation
loads public weights, extracts native FP32 anchors and recomputes centroid
vectors; compute loads a separate FP64 probe. Each map writes its numerical NPZ
and validation JSON. The trajectory writes the renderer's flat `gaussian.npz`,
`shuffled.npz`, `llama3.npz`, plus detailed class subdirectories. Bubbles write
both `bubble_coords.parquet` and `bubble_coords_extra.parquet`. Rendering is
separate; see `RENDER.md`.

### Refusal: two deliberately separate numerical environments

Direction extraction must retain the authors' 2024 BF16 environment. Do **not**
extract that vector in the modern FP64 map environment and call it equivalent.
The wrapper automatically clones a sparse source/splits-only author checkout at
commit `0b9c8bc56a73b42f595fd173156d4aefe9a9143c` and invokes
`uv run --isolated --no-project --python 3.10 --with-requirements ...` with the
pinned historical requirements. It refuses modified or wrong-revision existing
checkouts instead of changing them. Git, uv, network access and HF authorization
must be available. No published run artifacts are checked out or read.

```sh
uv run --extra compute python -m manifold_attractors.maps.refusal_extract \
  --author-repo external/refusal_direction --output data/refusal_direction
# Return to modern dependencies for full-prefix FP64 geometry and generation.
uv run --extra compute python -m manifold_attractors.maps.refusal \
  --direction-dir data/refusal_direction --output data/refusal_steering_heatmaps
```

The extraction runner invokes the pinned authors' sampling, filtering,
`generate_directions` and `select_direction`; it does not read the published
vector/candidates/completions. It records its own sources and selected vector and
fails if selection does not return zero-based layer12 (input block13). It does
not run harmful completion generation or external API judges. It requires the
external repository's dataset splits and dependencies, not its experiment
artifacts. The historical package pins include platform-specific CUDA wheels;
installation on other platforms is not verified.

## Exact scientific configurations and source lineage

### Triple area

Sources: `exploration/language_steering_report_figure.py` (`prepare_olmo`,
`prepare_centroid`, `extend_centroid_surface_map`),
`exploration/language_steering_mollweide.py` (`six_block_probe`), and
`paper_scripts/concept_manifold_speed_profiles.py` (`surface_area_map`).

* Model `allenai/OLMo-3-1025-7B`, revision
  `a81bae42db3975be1671e27b9c9a56da1a9f980f`.
* Native FP32 input-block16 anchors, FP64 map through blocks16–21 by default.
  Native fixed two-position EOS probe; both positions evolve. No final norm.
* Six triples are `[4654,3707,6984]`, `[7473,6917,3872]`,
  `[3293,7563,4167]`, `[3026,1571,7925]`, `[8130,5826,3660]`,
  `[7413,4729,5465]`. Originally `default_rng(20260930).choice(10000,18,
  replace=False)` on a Wikipedia English test manifest.
* The original Llama source-character spans were mapped to the earliest native
  nonspecial OLMo token covering the span start, otherwise earliest overlapping
  token. The exact resulting texts, native IDs, positions and mappings are
  frozen in `inputs/triple.json`; no fresh tokenizer resampling replaces them.
* Pole is normalized **raw** A+B+C, radius is B's native norm; first axis is A's
  projected direction, second is projected C Gram–Schmidt orthogonalized.
* Full sphere 145 latitudes × 289 longitudes, spacing1.25°, lat±90°, lon±180°.
  Central intrinsic orthonormal-tangent great-circle FD at .05° and .025°,
  divided by input arc length; area is Gram–Schmidt output parallelogram area.
  It is not a projection Jacobian. Half-step convergence, seam, poles and
  scalar/batch checks are retained. All full-sphere points are recomputed;
  nothing is seeded from the former hemisphere results.

### Language area

Sources: `language_steering_report_figure.py` (`prepare_llama_language_row`,
`compute_language_row`), `llama_olmo_language_order.py` (`corpus`, `extract`),
`language_steering_greek_extension.py`, and the corrected-v2 anchor manifests.

* Llama3.1-8B revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, native FP32
  input16 anchors; FP64 four-block map16–19, evolving two BOS positions.
* Current six panels: original English0/1, **corrected-v2 Chinese3**, original
  Chinese4, Greek-extension0/1. These exact source records are bundled, not the
  earlier byte-fragment or superseded Chinese anchors.
* EN/ZH/EL direction corpora are rebuilt from `wikimedia/wikipedia`,
  `20231101.<lang>`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
  Take first10,000 eligible unique ID/normalized-prefix records, at most100,000
  scanned rows, stripped text≥200 chars truncated to1200. Permute with
  seed20261001 + language index in the frozen30-language list. Reject pairs
  with normalized five-character-shingle Jaccard≥.8; choose200 direction docs
  and four disjoint anchor docs (distinct digit-normalized anchor title families).
  Full algorithm is in `inputs.public_corpus`; only the200 direction texts enter
  the recomputed centroids. Existing exact panel anchors stay fixed.
* Centroids: tokenizer max128, batch4, all attention-mask-valid tokens including
  special tokens, native FP32 activations accumulated and pooled in FP64.
  First tangent is EN−ZH projected off the anchor; second is ZH−EL projected
  off anchor/first. Same145² ±90° sphere chart and .05° FD as original.
  The renderer crops±45° and reflects both axes; generator does neither.

### Language steering speed

Sources: `exploration/concept_manifold_barriers/language_regions.py`,
`paper_scripts/export_language_steering_speed.py`, `language_chinese_figure.py`,
`language_strength_grid.py`, and original `steering_common.wiki_passages` /
`_mean_hidden`. The latter input-preparation provenance was recovered from the
historical `cache_lang_text.py` job script; it is not a runtime dependency.

* Same pinned Llama3.1, native FP32 states, FP64 blocks16–19 and full remaining
  model for argmax tokens. Both BOS positions evolve; **not source-context
  decoding**. Positive direction is unit(ZH−EN), tangent-projected at each anchor;
  radius is the anchor's own norm.
* This centroid corpus is **different from language-area**: first200 eligible
  Wikipedia passages per language, stripped length≥200 and truncated1200.
  Public preparation rebuilds it and checks every passage hash against the
  original. Original fetch did not pin a dataset revision; the later audited
  Wikipedia revision is used, with hash mismatch a hard error, not substitution.
* English selected source indices104,61,8,14,124 (source-only English-prose
  eligibility, seed20260920). Current renderer omits index8. Chinese final
  indices409,397,357,449 (seed20261026, with later source-only replacement of338
  by357). Exact native input IDs/positions are retained, along with eligibility
  text. This is illustrative development data, not a held-out confirmatory test.
* English computes the original EN-directed periodic circle then sign-reverses
  the sampled arrays exactly at export. Start1° grid, interleave half-step
  evaluations; refine up to3 times if relative-L2≥.05 or relative-p99≥.1.
  Central vector difference, speed per radian, no final normalization. Chinese
  uses the published1°/.5°/.25° comparison, validates shared argmax IDs and
  exports .5°. ±90° outputs retain all Voronoi-cell token regions, including
  narrow regions. Boundaries are half-grid estimates, not exact transitions.
* Original hand-reviewed translations, grammatical glosses, script display
  classes and interlinear context annotations are preserved as **source label
  inputs** in `inputs/language_speed_labels{,_chinese}.json`. They contain no
  numerical curve/region/minimum statistics. After fresh region computation,
  labels are matched by token ID and exact decoded text/token piece. Unknown
  tokens, changed target highlights or mismatched contexts hard-fail: no literal
  label fallback, invented translation, or silently substituted editorial text.
  The Chinese source annotations cover the current409/397/357/449 quartet.

### Refusal maps

Sources: `refusal_paper_replication.py`, `refusal_live_demo.py`,
`context_heatmap_series.py`, `context_heatmaps.py`, `refusal_instruct_context.py`.

* `meta-llama/Meta-Llama-3-8B-Instruct`, revision
  `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` (original authors did not state a
  revision; this is the pinned snapshot used by the later replication).
* Author extraction: Python random42,128 train and32 validation per class;
  first-token-I log-odds filter; original BF16 hidden states and FP64 means,
  five post-instruction positions; original constrained direction selection.
* Preserve original literal leading quote, explicit BOS plus tokenizer BOS,
  and assistant-header double newline. Requests are exactly the sky-explanation
  and poisoning prompts in `refusal.py`. Native FP64 greedy generation.
* Three rows, eight input tokens each: benign unsteered, harmful unsteered,
  benign header-newline-only steering. Only the final header input is steered;
  later tokens receive no further addition. The coefficient is **recomputed**
  by the original next-basin rule, not hardcoded to the historical
  3.369865844775658. Speed grid[-4,4],4001 points, FD .0005/.00025 raw-vector
  coefficient; maxima/minima bounded refinement `xatol=1e-8` and stability gate.
* Full-prefix conditional FP64 blocks13–16. Plane `h + alpha*d + beta*||d||*e2`,
  where e2 is seed20261003 Gaussian projected off pole/refusal tangent.
  Alpha[-4,8] (benign) or[-8,4] (harmful),217 points; beta[-3,3],109 points.
  Grid dimensions were verified from NPZ **headers only**, never array values.
  Orthogonal-unit FD epsilon=||h||·radians(.05); retain epsilon/2 area, require
  max relative half-step error<.01. The area is neither speed nor refusal score.

### Synthetic trajectory

Exact `exploration/synthetic_profile_trajectory.py::compute` port, importing the
core `synthetic_states` implementation. Use freshly extracted256 reference input
block13 states, sampled positive nonspecial positions, from `prepare_native`.
Seed20260909; model_seed=seed+1000; synthetic seed+1 and tangent seed+2.
Twenty states/class: full-covariance empirical Gaussian, independently shuffled
coordinates, and isotropic Gaussian-reference-norm-matched control. Float32
Llama3.1 eager attention; fixed `[128000,128000]`; blocks13 through28 inclusive.
Only central anchors propagate. At every block project the same ambient tangent
draw off the current anchor and re-probe401 angles−200…200°; discard diagnostic
circle outputs. Raw output/input norm ratio, no final normalization. Original
full-model injection, scalar/batch, center, tangent, radius and periodic checks
are retained. The renderer chooses13,14,15,18,20,24,26,28.

### Bubble coordinates: recovered v3, not current old-repository file

The correct producer was recovered from
`cc2a8ac:paper_scripts/gen_bubble_coords.py`; the later working-tree producer
has a different frame/schema and must **not** replace it. Supplemental pair
selection follows `paper_scripts/bubble_coords_extra.py` without its old-cache
reconstruction gate (fresh runs have no such cache).

FP32 Llama3.1 eager, hidden-state index15→25 (blocks16–25), fixed two BOS tokens;
original `pullback_solve.make_forward(..., fast=True)` and batch2048 preserved in
`bubble_forward.py`. Five single, five natural-pair and five shuffled-pair
bubbles,4000 Fibonacci points each, plus exact negative-index endpoint rows:
60,025 rows. Supplemental natural pair sample5 adds4,002 rows. Natural Python
random seed20240706,100 synthetic-source states with seed+100, per-coordinate
`default_rng(seed).random(shape).argsort(axis=0)` shuffling. CPU axis seeds4242+i,
9000+i,11000+i; supplemental selection20261004 and axis9005.

Pair pole is the unit-direction bisector, radius mean endpoint magnitude, true
per-endpoint great-circle tangents. Finite difference `1e-3*radius+1e-6`.
Output frame: mean unit images first, then mean unit first/second tangent
velocities Gram–Schmidt; coordinates measured from **output origin**, not F(center).

**Missing provenance:** original FineWeb document IDs, corpus revision and
sentence-list selection procedure were never stored. The small original raw
sentence list is included, so fresh model computation and historical random
selection are runnable, but independent reconstruction of that list from
FineWeb is not claimed. Original bubble model revision also was unpinned; default
is the paper's other pinned Llama3.1 snapshot and override is recorded. Bitwise
identity to old cached forward images is not promised.

## Access, provenance limits, tests

Meta Llama checkpoints require HF authentication and acceptance of their model
licenses; OLMo and Wikipedia are public downloads. Wikipedia excerpts retain
Wikipedia's applicable attribution/share-alike terms. FineWeb sentence source
rights depend on underlying pages, and original URLs are unavailable. The
refusal repository/dataset splits and its own licensing must be obtained
separately. Network/storage/GPU resources are not bundled. No private model
endpoint or API judge is required.

Frozen source-record replay avoids changing displayed examples, but it is not
an independent re-run of all upstream cohort formation, manual anchor repairs
or source-only example review. Those distinctions are intentional. Changing
software/BLAS/CUDA/model versions can change values; output audit JSON and the
explicit convergence assertions are evidence, not a bitwise-reproduction claim.

```sh
uv run --extra compute python -m unittest manifold_attractors.maps.test_maps -v
```

14 CPU tests cover intrinsic identity/scaling, full-sphere seams/poles, affine
area, random frames, periodic region cells and speed units, Fibonacci/frame
geometry, exact frozen-input counts, fail-closed public-corpus hashes, next-basin
selection, tiny-Llama fixed/full/conditional replay, and the recovered bubble
cached-prefix/full-forward agreement, strict editorial-label matching, and the
isolated historical wrapper command contract. The wrapper's actual dependency
installation/full extraction was not launched. CLI `--help` for all six
entry points and Python compilation were also checked. No old result arrays
were read to supply any computation or to make these tests pass.
