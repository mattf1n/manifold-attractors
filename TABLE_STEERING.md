# Fresh adaptive-steering table

`configs/steering_table.json` defines the standalone data DAG:

```
data-adaptive_steering-prepare
    -> data-adaptive_steering-compute
        -> data-adaptive_steering-oracle
            -> data-adaptive_steering
```

The final task explicitly depends on all three producer stages. It writes
`data/language_steering_confirmatory/data_complete.json`. The renderer-facing
files are:

- `data/language_steering_confirmatory/summary.json`
- `data/language_steering_confirmatory/linear_oracle/summary.json`

Their dataset/method dictionaries match the paper table's input schema. Table
rows are Untuned=`raw_centroid`, Tuned=`additive`, Adaptive=`adaptive_straight`,
Oracle=`oracles[track]`; tracks are `mass` and `greedy`; datasets are `wiki_test`
and `fineweb_test`. Non-Chinese letters means Latin + Other alphabet, **not**
1 minus Chinese. Percentages/rounding and rendering belong to the renderer.

## Run

Install the project's compute extra and accept the Llama model license/download
terms. Allocate hardware separately; none of these modules submits GPU jobs.
Run from the repository root using the compute Python environment:

```sh
python -m manifold_attractors.steering.prepare --root data/language_steering_confirmatory
python -m manifold_attractors.steering.compute --root data/language_steering_confirmatory
python -m manifold_attractors.steering.oracle all --root data/language_steering_confirmatory
python -m manifold_attractors.steering.verify --root data/language_steering_confirmatory
```

The same four tasks are discoverable by the project's DAG runner. No dependency
on figure-produced activations or on an existing reproduction checkout exists.
Use a new empty root for a genuinely new experiment. A nonempty root without
`run.json` is rejected. Resumption is permitted only for this implementation,
source-input fingerprint, model pin, and recorded library versions. Prepared
artifacts and oracle dependencies are checksum-verified. Do not edit source
code mid-run: changed scientific code requires a new root.

For separately scheduled parallel work, after prepare:

```sh
# Calibration controls their own repeated endpoint expansions.
python -m manifold_attractors.steering.compute --stage calibrate --root ROOT
# For each DATASET in wiki_test/fineweb_test and SHARD in 0..19:
python -m manifold_attractors.steering.geometry --root ROOT --dataset DATASET --shard SHARD
python -m manifold_attractors.steering.fixed test --root ROOT --split DATASET --shard SHARD --adaptive .
# After every geometry and test shard completes:
python -m manifold_attractors.steering.compute --stage collect --root ROOT
python -m manifold_attractors.steering.oracle prepare --root ROOT
# For each DATASET/SHARD, after all compute dependencies complete:
python -m manifold_attractors.steering.oracle worker --root ROOT --dataset DATASET --shard SHARD
python -m manifold_attractors.steering.oracle collect --root ROOT
python -m manifold_attractors.steering.verify --root ROOT
```

Do not run concurrent producers for the same shard/root. `compute` and `oracle
all` run workers sequentially in subprocesses so native FP32 and FP64 models do
not coexist in GPU memory. Intermediate per-example profiles, eight-document
calibration/oracle chunks and per-shard predictions are retained. Endpoint
expansion is automatic; a safety cap of 16 expansions raises a blocker, never
substitutes a held-out-selected parameter. Increase it explicitly if necessary.

## Inputs, pins and held-out split

- Model: `meta-llama/Llama-3.1-8B`, revision
  `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- Wikipedia: `wikimedia/wikipedia`, `20231101.en` and `20231101.zh`, revision
  `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- FineWeb: `HuggingFaceFW/fineweb`, `sample-10BT`, revision
  `9bb295ddab0e05d785b879661af7260fed5140fc`.

`src/manifold_attractors/steering/inputs/*.json.gz` are **source inputs only**:
21,000 selected source excerpts, document IDs/URLs, corpus coordinates, recorded
identity groups, input token sequences and sampled positions. They contain no
activations, centroids, vocabulary labels, selected steering parameters,
predictions, profiles or results. `inputs/sources.json` records checksums of
these packages, the original source-selection manifests and relevant producer
source files. Compression is lossless; records and positions retain the exact
original order. This avoids an irreproducible live Wikimedia/Wikidata identity
API dependency without changing the held-out cohort.

Prepare independently retokenizes every excerpt with the pinned tokenizer,
checks source-text hashes and all 256 special IDs, checks cross-split document,
context and known identity-group uniqueness, and exactly replays the
`20261022` position RNG. Wikipedia calibration is 1,000; the two tests are
10,000 each. There is **no FineWeb calibration** and no after-freeze resampling.
The historical input-only selection used Wikipedia shuffle seed `20261021`,
FineWeb row-group seed `20261031`, and FineWeb final shuffle seed `20261023`.
It excluded the old development prefix, frozen IDs/titles/URLs, known translated
counterparts, exact contexts, and word-5-shingle Jaccard >=0.80 duplicates;
FineWeb also excluded Wikimedia/mirror domains and markers. The fixed source
selections, not a newly queried identity service, are the standalone contract.

This does not newly certify historical translation-family independence. The
original practical contract's limitations remain explicit: 52 unresolved English
QIDs still had frozen-document bans; canonical ZH397 had no verified English
counterpart; current identity mappings can differ from 2023; lexical filters
cannot rule out paraphrased/translated web mirrors. The original development
selection history is not rerun or falsely presented as a new API audit.

Wikipedia excerpts retain their source URLs/titles for attribution under the
source corpus's Wikipedia licensing. FineWeb records retain source URLs and
corpus coordinates; underlying web-document rights remain those of their
respective sources. The compressed files are not new model-generated text.

## Exact scientific protocol

1. **Direction:** the speed/confirmatory corpus, **not** the area's deduplicated
   corpus. Download the first 200 qualifying stripped >=200-character English
   and Chinese Wikipedia passages, crop at 1,200 characters, and verify every
   text against the speed corpus's source hashes. Recompute native FP32 model
   states with eager attention. Batch **4**, max 128 tokens, token-weighted
   pooling over every attention-mask-valid position including special tokens,
   FP64 accumulation. Preserve the FP64 centroid difference without a FP32
   round-trip. Untuned adds `mean_zh - mean_en`; every unit-direction method uses
   its normalized version. The shared `maps.inputs` helpers are used only for
   this verified **speed** text/centroid protocol, never area activations.
2. **States:** one sampled nonspecial position >0 within the first 128 tokens;
   fresh native FP32 hidden-state index 15 (input of human block 16). Causal
   truncation and early stopping are independently checked against full-document
   extraction on the first two examples of every 500-record shard. The cache is
   newly populated inside this run; no external activation cache is accepted.
3. **Probe:** replace position 1 of two evolving BOS positions, not a source
   document continuation. Promote native states to FP64; use FP64 weights,
   RMSNorm and attention softmax. Geometry uses raw outputs of blocks 16–19,
   without final norm. Decoding runs every remaining block, final norm and
   vocabulary head. Full 128,256-way temperature-one softmax; no API inference.
4. **Calibration:** fit additive and norm-preserving controls on 0..8 in steps
   of .05, and constant projection on -4..8 in steps of .05. Each method/track
   independently minimizes bad mass or bad argmax rate, then maximizes Chinese,
   then chooses the smaller scalar. If **either track** selects an endpoint,
   expand that endpoint by 8 (additive/norm) or 12 (projection) and repeat on
   calibration only until all selections are interior. No saved fitted strengths
   are included or used. Controls not displayed in the four-row table are still
   computed because the oracle's exact comparator candidate set needs them.
5. **Adaptive:** preserve the original plateau inequalities and unsmoothed
   speed extrema. Straight-line grid initially [-16,32] step .1; expand to
   [-32,64] only for the frozen bracketing/reference failure. The basin containing
   zero lies between sampled maxima; a sampled maximum at zero selects the
   **left** basin. Select the lowest-speed eligible minimum after its unique
   reference with 0<s<10, otherwise an eligible other left minimum -10<s<0 gates
   no steering, otherwise use that track's freshly calibrated additive fallback.
   Angular control uses the full circle and the original 10..55-degree gate.
   Central finite differences, half-step relative L2<.05 and p99<.1, at most
   three additional refinements; every dataset-local index divisible by 100
   gets an extra quarter-step diagnostic. A disagreement never replaces primary
   selection. Invalid primary cases remain explicit and block full-cohort
   completion rather than being dropped or assigned a convenient fallback.
6. **Oracle:** output-aware finite additive-family search, **not** a calibrated
   baseline or continuous optimality certificate. Strict 0<s<10; multiples of
   .025 plus 1e-6 and 10-1e-6; exact newly computed comparator strengths for both
   additive, both constant-projection, both adaptive-straight tracks and raw
   centroid, only when in the strict interval. Norm-preserving/angular points
   are outside this family. Separate mass/greedy lexicographic choices and a
   diagnostic Chinese ceiling. Retain .05 coarse-grid comparisons and add all
   400 odd .0125 multiples on prespecified 1% global-audit examples. Exact
   comparator predictions are independently replayed, **within this fresh run**.
7. **Collection:** all examples; five category and eleven subtype probabilities,
   argmax counts, modes, strength/norm/projection diagnostics, geometry replays,
   full-model injection checks, and 10,000 paired bootstrap draws. Common indices
   across contrasts within a dataset. Baseline seeds `20261025 + dataset_index`;
   oracle seeds `20261101 + dataset_index`. Intervals are conditional on fitted
   calibration/search, without multiple-comparison adjustment.

The numerical/policy/calibration/collection/oracle implementations are narrowly
ported from the named confirmatory producers and their transitive helpers.
Non-table Gutenberg evaluation and historical development preflight/copy jobs
are intentionally absent. All six confirmatory methods and the oracle diagnostics
for the two table datasets are retained.

## Completed full-size reproduction

The complete fresh run finished before the requested 6pm Pacific deadline:
1,000 calibration examples, both10,000-example test cohorts, all40 geometry/test
shards, all40 dense-oracle shards, prespecified refinement audits and final
collection. No old activations, fitted strengths or outcomes were used as inputs.

The table is **scientifically reproduced**, not byte-identical. All calibrated
parameters, baseline/oracle argmax-category counts and displayed method rankings
match the historical results. Twenty-four of32 displayed cells match at printed
precision. The eight Chinese-mass cells differ by at most **0.01712 percentage
points** (at most0.02 at two-decimal printing); non-Chinese mass differences are
below0.00001 percentage points.

This small difference is diagnosed, not attributed vaguely to floating point:
Python3.13 uses Unicode15.1, whereas the historical Python3.10 classifier used
Unicode13. Under15.1, token IDs165 and18630 (incomplete UTF-8 Han prefixes) move
from Other nonalphabetic to Chinese because U+9FFD..U+9FFF are now assigned.
The **bad-letter mask is bitwise unchanged**. Recomputing the partition from the
pinned tokenizer under Python3.10/Unicode13 reproduces every historical category
exactly; old labels were used only afterward to validate that diagnostic.
The completed inference run retains its fresh15.1 partition consistently rather
than patching results. This version-dependent effect is inconsequential to the
paper's steering comparison and is disclosed in the README/table validation.

See `docs/tables/validation.json` for numerical comparisons and provenance.

## Implementation validation and resource planning

CPU tests:

```sh
python -m pytest tests/test_steering_table.py tests/test_portability.py -q
python -m manifold_attractors.steering.fixed dev-self-test --root /tmp/steering-dev
```

Tests include input-only source checks, DAG dependencies, objective tie ordering,
endpoint expansion, the zero-maximum left-basin rule, delayed track-specific
fallback, strict oracle interval/grid, UTF-8 Han-prefix classification, full
calibration/test filesystem assembly, missing outcomes and tamper rejection.
A **real locally initialized tiny 20-layer Llama** runs native FP32 extraction,
FP64 four-block propagation, full vocabulary probability/argmax, hook/scalar
agreement, calibration selection and oracle selection entirely on CPU. This is
an executable model-path test, not a claimed scientific table reproduction.
All 21,000 bundled texts were also retokenized using the cached pinned real
Llama tokenizer and every position RNG draw replayed successfully, without
reading old outputs or running GPU inference.

The initial extraction task performed no full-size inference; the subsequent
orchestrated run and its measured results are documented above. Implementation
unit tests alone are not the basis for the full-size reproduction claim.

**Observed H100 run:** 3.843 allocated GPU-hours for the complete adaptive table,
across82 GPU allocations (prepare, calibration,40 test shards,40 oracle shards)
and two additional CPU collection allocations. Every allocation completed.
The independent500-example shards were externally scheduled; the oracle cap was
raised to40 to meet the deadline. This supersedes the initial unbenchmarked
20–100 GPU-hour planning estimate. Runtime on other devices or with more
refinement can differ. FP64 weights occupy roughly64GB;40GB GPUs cannot run
this implementation as written.

Work accounting before grid expansion/refinement:

- 400 centroid source documents + 21,000 native extraction records;
- 563,000 full-decoder calibration candidate evaluations;
- at least 20,000 × (720 angular + 961 straight) = 33.62 million four-block
  geometry points, plus quarter audits/expanded ranges/refinements;
- 20,000 × (401 grid + up to 7 exact extras) = up to 8.16 million oracle
  full-decoder candidates, plus 80,000 .0125 global-audit points;
- selected-method predictions, full-hook/scalar audits and CPU paired bootstrap.

The dominant cost is genuine FP64 model work, not downloading a result archive.
Budget several GB of intermediate data plus the pinned model download (~16 GB),
HF cache, and headroom; no full logits or archived activations are bundled.
External blockers are gated-weight authorization/download availability, adequate
CUDA/FP64 memory, and the unperformed full run. No commercial inference API,
translation API or live Wikipedia/Wikidata API is required.
