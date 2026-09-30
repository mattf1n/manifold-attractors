# Fresh norm-growth classifier table data

`experiments.norm_growth_classifier` generates **all-layer scores for all three
models**, not rendered tables. It reuses only the fresh fit/heldout activation
pools produced by `experiments.prepare_native`. There is no runtime dependency
on `~/project`, a paper checkout, archived arrays, or historical experiments.

## DAG and invocation

`configs/classifier_table.json` exports:

| Task | Dependency | Outputs |
| --- | --- | --- |
| `data-norm_growth_classifier-llama3` | `data-native_layers-llama3` | `data/norm_growth_classifier_scores/llama3.{npz,json}` |
| `data-norm_growth_classifier-olmo3` | `data-native_layers-olmo3` | `data/norm_growth_classifier_scores/olmo3.{npz,json}` |
| `data-norm_growth_classifier-qwen3` | `data-native_layers-qwen3` | `data/norm_growth_classifier_scores/qwen3.{npz,json}` |

The native tasks are defined in `configs/core.json`. **Do not replace or modify
`data/norm_growth_classifier_layers`**: those files are already hashed inputs to
other experiments. The score generator verifies source hashes before/after
computation and rejects output aliases of the native input directory/files.
It does not invoke extraction again. Existing native tasks in the fresh full
run can supply these dependencies without any change to their output contract.

From a checkout whose fresh native tasks have completed:

```sh
uv run --extra compute python -m manifold_attractors.experiments.norm_growth_classifier \
  --model llama3 --data-root data \
  --native-layers data/norm_growth_classifier_layers \
  --output data/norm_growth_classifier_scores \
  --device cuda --threads 4 --batch 128
```

Repeat for `olmo3` and `qwen3`, or use the three DAG tasks. All paths are relative
to the run checkout; there is no hardcoded run location. For the in-progress
`~/manifold-reproduction-20260930-222930` run, use that run's freshly generated
native pools and upstream JSON metadata. Do not copy archived NPZ inputs into
it. `--device cpu` is supported; full-size model scoring is expensive. This
implementation task submitted **no GPU jobs**.

## Exact scientific protocol

- Models: pinned Llama 3.1 8B, OLMo 3 7B, Qwen 3 8B snapshots inherited from
  the fresh core pipeline; 32, 32, and 36 one-based blocks, respectively.
- Upstream reference pool: 256 Pile-10k documents, seed `20260909`, maximum
  token length 256, one uniformly sampled nonspecial token at position > 0
  per document. The existing core sampler defines the exact examples.
- Heldout natural pool: 128 documents disjoint from reference rows, document
  ordering seed `20260910`. Model token RNG seeds are `20261910`, `20262910`,
  `20263910`, in model order above. Native extraction supplies each block's
  **input** states and its causal-prefix/full-sequence consistency diagnostics.
- Classes, in order: `natural`, `gaussian`, `shuffled`. No norm-matched class.
- At **each layer**, fit mean and full low-rank sample covariance (`ddof=1`)
  to the 256 reference vectors using float64. With centered factor
  `F = (reference - mean) / sqrt(255)`, draw 128-by-256 standard normals `Z`
  and form `mean + Z @ F`. Independently permute all 256 rows for each
  coordinate, then take the first 128 shuffled rows. Cast states to float32.
- Reset the synthetic RNG to the **heldout model seed + 1** at every layer:
  `20261911`, `20262911`, `20263911`. Do not increment by layer or use the
  reference-pool seed. Gaussian draws and permutations are coupled across
  layers. The repository's existing `synthetic_states` helper is reused;
  its later, unused norm-matched draws do not alter either classifier class.
- Capture each block's baseline and keyword arguments by forwarding
  `[BOS/EOS, BOS/EOS]` once. Replace position 1 with each test vector, keeping
  that block's prefix state fixed. Score is the **raw single-block output
  L2 norm / injected input L2 norm** at position 1. No final model RMSNorm,
  angle perturbation, context replay, threshold fitting, or sign selection.
- Float32, eager attention, TF32 disabled, batch 128 by default. Natural is
  always positive; higher scores always predict natural. AUC compares every
  natural/synthetic pair and assigns half credit to exact ties.

## Output / renderer contract

NPZ arrays retain the original all-layer layout:

- `layers`: one-based block IDs; `classes`: the three names above.
- `scores`: float32 `(layers, 3, 128)`.
- `states`: float32 `(layers, 3, 128, hidden_size)`.
- `reference`, `evaluation`: fresh native pools, unchanged.
- `reference_mean`, `covariance_factor`: per-layer float64 fitted quantities.
- `gaussian_coefficients`, `shuffled_permutations`: shared across layers,
  stored once, as in the original generator.

JSON records source hashes, exact fit/test records, seeds, fitted statistics,
score/context definitions, per-layer AUC, runtime versions and validations.
It contains no claim of comparison with old score arrays.

The current paper renderer consumes `layers`, `classes`, `scores` and shows
15 default columns: **1, 4, 6, 8, 11, 14, 16, 18, 21, 24, 26, 28, 31, 34, 36**,
computed by `rint(linspace(1, 36, 15))`. Gaussian/Shuffled rows each contain the
three models; unavailable Llama/OLMo layers 34/36 are dashes. Scores must still
be generated for **every layer**, not only these columns. Rendering uses
three-decimal AUC, 15 columns per panel, no confidence intervals. The renderer
integration is separately owned; point it at `data/norm_growth_classifier_scores`,
not the native pool directory.

## Validation and provenance limits

The implementation extracts the original all-layer scoring logic only, reusing
local synthetic/atomic-write helpers. It does not copy the legacy midpoint
experiment or frozen-old-array comparison paths. `configs/classifier_table.json`
records inspected source/renderer hashes and all three archived metadata hashes.
Archive JSON was inspected for parameters, not supplied as a runtime input.
Llama/OLMo's archived generator hash differs from current source; Qwen's matches.
Scientific reproduction permits inconsequential floating-point variation, not
changed sampling, examples, fitting, scores, or evaluation methods. Full-size
fresh score/AUC agreement is **not yet verified** by this CPU implementation task.

Checks include pool counts/shape/dtype, seeds, snapshot/dataset consistency,
upstream fit hash, exact metadata record agreement, document disjointness,
positive nonspecial positions, direct-block replay error < `1e-5`, and finite
positive scores. Each batch's first vector is independently replayed; relative
output L2 and output-norm error must both be < `2e-5`. Legacy elementwise
`rtol=atol=2e-4` remains a diagnostic. If that diagnostic fails (as in the
archived unnormalized Qwen final block), **all 384 vectors** are individually
replayed and the same relative bounds plus AUC difference <= `2 / 128²` must
hold. Batched scores remain the reported scores.

CPU-only tests (no model downloads or archived data):

```sh
uv run --extra compute --group dev python -m pytest tests/test_classifier_table.py -q
```

Tests cover AUC ties/direction, original synthetic RNG math, all-three-model
DAG wiring, protected inputs, seed validation, partial batches, the cancellation
fallback, and real tiny random Hugging Face Llama/OLMo3/Qwen3 models with their
full 32/32/36 block counts. Integration tests assert source hashes unchanged,
all-layer state construction, renderer-compatible AUC, and independently replayed
late-block scores. These tiny tests validate APIs and protocol, not the actual
8B/7B-model numerical results.
