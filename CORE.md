# Core figure data generators

All commands run from this repository after `uv sync --extra compute` (or an
editable installation with the compute extra). Use `uv run python` in place of
`python` below. No `~/project`, archived NPZ, existing paper checkout, or generated
input data is needed. Intermediate files below are **regenerated**, not shipped.
No GPU jobs are launched by these modules; allocate hardware separately.

## Inputs and dependencies

Runtime: NumPy **2.2.6**, PyTorch **2.11.0** (paper: `2.11.0+cu130`),
Transformers **5.8.0**, PyArrow, huggingface-hub, accelerate. The original
FP32 eager-attention arithmetic, disabled TF32, seeds, low-rank covariance fits,
coordinate shuffles, finite differences and verification tolerances are retained.
CPU/GPU arithmetic is not bit-identical; use the devices below for paper fidelity.

`core/inputs.py` downloads these immutable public Hugging Face inputs into the
standard HF cache; `HF_HOME`/`HF_HUB_CACHE` may relocate it. Llama is gated: accept
Meta's model license and authenticate with `HF_TOKEN` before downloading.

| Key | Repository | Revision |
|---|---|---|
| llama3 | meta-llama/Llama-3.1-8B | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b |
| olmo3 | allenai/OLMo-3-1025-7B | a81bae42db3975be1671e27b9c9a56da1a9f980f |
| qwen3 | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |

Dataset: `NeelNanda/pile-10k`, revision
`127bfedcd5047750df5ccf3a12979a47bfa0bafa`, file
`data/train-00000-of-00001-4746b8785c874cc7.parquet`, SHA256
`a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31`.
The hash is checked even with `--dataset PATH`; arbitrary replacements are not a
paper replication. Model revision overrides are explicit `--revision` options.

## Exact paper commands, dependency order

Defaults below are part of the protocol, not tuning suggestions. Run all three
models. Generated paths match the current paper Makefile. `--threads 4` is the
historical default except natural speed (`8`). All probe batches are `128`.

### 1. Natural norm circles and their matched document-context comparison

```sh
for m in llama3 olmo3 qwen3; do
  python -m manifold_attractors.experiments.great_circle_norm_grid \
    --model "$m" --device cuda --seed 20260908 --samples 20 --max-tokens 256 --step 1 --batch 128
  python -m manifold_attractors.experiments.great_circle_norm_grid_context \
    --model "$m" --device cuda --batch 128
 done
```

Outputs: `data/great_circle_norm_grid/{model}.{npz,json}` and
`data/great_circle_norm_grid_context/{model}.{npz,json}`. Eight equally spaced
one-based blocks (rounded from a linspace), 20 seeded documents/tokens, 401 angles
−200…200°. Context mode reuses **exact** norm-circle anchors/tangents; fresh target
K/V and original causal-prefix K/V, with direct/full-model injection checks.
The alias is equivalent to `great_circle_norm_grid --context natural`.

### 2. Original midpoint reference fits, native-layer pools, selected norm plots

```sh
for m in llama3 olmo3 qwen3; do
  python -m manifold_attractors.experiments.great_circle_synthetic_grid \
    --model "$m" --device cpu --seed 20260909 --reference 256 --samples 20 --max-tokens 256 --batch 128
  python -m manifold_attractors.experiments.prepare_native --model "$m" --device cuda
 done
python -m manifold_attractors.experiments.great_circle_synthetic_grid \
  --model llama3 --block 21 --reference-cache data/norm_growth_classifier_layers \
  --output data/great_circle_synthetic_grid_llama21 --device cuda
python -m manifold_attractors.experiments.great_circle_synthetic_grid \
  --model olmo3 --block 21 --reference-cache data/norm_growth_classifier_layers \
  --output data/great_circle_synthetic_grid_best_layers --device cuda
python -m manifold_attractors.experiments.great_circle_synthetic_grid \
  --model qwen3 --block 27 --reference-cache data/norm_growth_classifier_layers \
  --output data/great_circle_synthetic_grid_best_layers --device cuda
```

Midpoint blocks are **16/16/18** (Llama/OLMo/Qwen); main-figure norm blocks are
**21/21/27**. These are the already selected blocks in the Makefile/archive, not a
new layer selection on regenerated evaluation scores. Native preparation rebuilds
256 fit + 128 disjoint heldout document states at every block. It writes the
historically named `norm_growth_classifier_layers` pool and
`norm_growth_classifier/{model}.json` sampling metadata, **not classifier scores**.
No classifier experiment is required by these figures. `--records-only` skips
native-state extraction when only heldout sampling metadata is needed.

Synthetic classes: natural first 20 fit rows; full low-rank Gaussian with
float64 ddof=1 moments; independent coordinate permutations of all 256 fit rows;
isotropic direction with positive Gaussian-distributed norm (resample nonpositive
radii). All four use shared preprojection Gaussian tangent draws. Model seed is
`20260909 + 1000*(model_index+1)`; synthetic seed is +1, tangent seed +2.

### 3. Natural and synthetic output-vector speed

```sh
for m in llama3 olmo3 qwen3; do
  python -m manifold_attractors.experiments.great_circle_speed_grid \
    --model "$m" --device cpu --seed 20260910 --samples 20 --max-tokens 256 --threads 8 --batch 128
  python -m manifold_attractors.experiments.great_circle_synthetic_speed_grid \
    --model "$m" --device cuda --source data/great_circle_synthetic_grid --batch 128
 done
```

Outputs: `data/great_circle_speed_grid` and
`data/great_circle_synthetic_speed_grid`. The latter intentionally uses the
**midpoint**, not the main-figure selected blocks. Periodic central differences of
raw output **vectors**, 1° divided by radians, no radius normalization. First two
samples/block or class also use 0.5°; relative-L2 < .05 and p99 < .1 must pass.
Synthetic speed reuses the freshly generated midpoint anchors/tangents exactly.

### 4. Native-input trajectory starts, all layers (main + appendix)

```sh
for m in llama3 olmo3 qwen3; do
  python -m manifold_attractors.experiments.norm_growth_trajectory_starts \
    --model "$m" --device cpu --seed 20260912 --batch 128
  python -m manifold_attractors.experiments.norm_growth_trajectory_starts \
    --model "$m" --all-layers --device cuda --seed 20260912 --batch 128
 done
```

Requires step 2. Output used by the figures:
`data/norm_growth_trajectory_starts_all_layers/{model}.{npz,json}`. The first
command regenerates the historical CPU starts15…20, including full-document
start15 extraction (the original start15-only artifact is no longer needed).
The all-layer run preserves these exact input pools/cohorts, verifies overlap,
and propagates every start to the final block. Fit 256, 128 samples/cohort,
three synthetic classes; seed per start is
`20260912 + 1000*(model_index+1) + 10000*(start_block-15)`.
Both probe positions advance; no final normalization, no refitting downstream.
Natural baseline is the continuous block1-start fixed-probe trajectory, not the
native-document diagnostic. Ratios are per-step, not cumulative; pre-start NaNs
are intentional. This mirrors the archived CPU/GPU midpoint provenance rather
than silently extracting all cohorts with one device.

### 5. Heldout four-block pair barriers (plots and table)

```sh
for m in llama3 olmo3 qwen3; do
  python -m manifold_attractors.experiments.pair_barrier_validation \
    --validate-model "$m" --precision float32 --device cuda --pairs 128 --seed 20260915 \
    --output data/pair_barrier_validation_gpu
  python -m manifold_attractors.experiments.pair_barrier_validation \
    --validate-model "$m" --precision float64 --device cuda --pairs 128 --seed 20260915 \
    --audit-validation data/pair_barrier_validation_gpu --output data/pair_barrier_validation_float64
 done
python -m manifold_attractors.experiments.pair_barrier_validation \
  --collect-validation --output data/pair_barrier_validation_float64 \
  --paper-export data/pair_barrier_validation
```

Only midpoint fits (step 2's first command) are prerequisites. The exclusion
union of original norm/speed/reference/classifier documents is regenerated from
its exact seeds, **not** conditional on which result files happen to exist.
Pair seed 20260915; 128 disjoint pairs/class, 256 new documents/model. Four blocks
starting 16/16/18, natural/Gaussian/shuffled classes. Periodic initial steps
1°/1°/0.25°, half-step vector convergence (.05 relative L2/.1 p99), at most three
refinements. Downhill minima without smoothing/cutoff; score
`log(peak/higher minimum)` or zero. Conditional pair bootstrap: 2000 resamples.
The published figures/table use **float64 audited forwards of the original
float32 prepared inputs**. RMSNorm and softmax are explicitly upgraded to
float64 for this audit; running only float32 does not reproduce the final figure.
Export creates renderer-ready profiles and `summary.json`.

## Validation and scope

CPU analytic tests: `python -m unittest manifold_attractors.core.test_core`.
All modules support `--help` without loading weights or downloading inputs.
Generator outputs contain token records, hashes, seeds, versions, execution
provenance and numerical checks. Runtime assertions are intentional publication
gates; do not use `python -O`. Checkpoints require matching inputs/contracts.
Delete or use a new output directory after changing configuration.

Ported only the six requested generators, their activation-cache/downhill helpers,
and the native extraction/sampling portions required by those figures. Original
workstation paths and classifier-score/cache prerequisites are removed; numerical
kernels are otherwise unchanged. Full-size pretrained-model regeneration requires
substantial CPU RAM/GPU time and gated Llama access; CPU tests do not establish
bitwise agreement of complete regenerated figures across different hardware.

Source SHA256 provenance is recorded in `configs/core.json:source_provenance`.
Some current generator/helper source hashes differ from those recorded in the
archive; historical byte-identical source has not been recovered. Independent
validation nevertheless reproduced all six relevant synthetic state/construction
array sets exactly from archived fit pools, and all 1152 coarse plus 1152
half-step pair scores exactly from archived profiles. These were **audit-only**
checks: archived arrays are neither bundled nor read by any runtime generator.
This does not establish full-size forward/output or final-image equality.
