# Exact-reproduction audit

Target: `~/paper` commit `ed8b89bc8344db678eb5e360a1893937d3b94a1f`;
standalone extraction: `~/manifold-attractors` (under active development).
Audit date: 2026-09-30. This audit owns only this file.

## Verdict

**Exact from-scratch reproduction of all current figure artworks is not yet
established.** The principal source-provenance blocker is missing original TeX
style files. Reconstructed styles are useful but are not evidence of identical
font metrics. The language-speed editorial-label mismatch identified during
audit is now resolved: original annotations are preserved as verified source
inputs with fail-closed token/context matching (section 6).
The original per-renderer environment mismatch was corrected during this
audit: final inspection verified isolated task routing. Experiment-to-render
DAG integration remains unfinished at the latest test checkpoint. Full numerical regeneration and TeX rendering have not been verified
by this audit; CPU unit tests or existence of renderer code do not establish them.

Scope is **16 external PDF inclusions plus two inline TikZ artworks**, in both
ICLR and Goodfire profiles. The intro includes both the fish montage and the
quiver field; the sphere-probe cartoon is a separate inline artwork. Tables and
full manuscript pagination are outside this figure-only audit. I read all of
`main.tex` and `appendix.tex`, the active Makefile recipes, relevant renderers,
selected original JSON provenance, and standalone source. No archived
experimental arrays, parquet results, or activations were used as runtime inputs.

## Findings and smallest fixes (including fixes landed during audit)

### 1. Recover four original font/style sources, not a visually similar substitute

Original `scripts/fig/figure_style.py` reads:

- `styles/iclr-fonts.tex`
- `styles/goodfire-fonts.tex`
- `styles/figure-fonts.tex`
- `styles/figure-math.tex`

They are absent from the supplied paper tree and `git ls-tree HEAD styles`.
The standalone files explicitly identify themselves as reconstructions. The
renderer agent also checked the supplied archive/history without recovering
originals (see `RENDER.md`). The tracked `matplotlibrc` *is* available; its SHA256
is `3608116a2bc4edf1d33a385254b15dc31b50f9b3de5778cbc91088220c33ff01`.

**Required recovery:** ask the original rendering workstation owner for those
four files, plus the TeX/font environment used to create `fig/iclr` and
`fig/goodfire`. Commit the recovered bytes and their hashes. If unavailable,
retain a machine-readable exactness blocker for typography instead of claiming
that reconstruction is exact. This is not a request for experimental results.

The language-speed renderer additionally names actual fonts: ICLR uses
`Tempora-{Regular,Bold}.otf`, `cmun{rm,bx}.otf`, `gbsn00lp.ttf`, and `bsmi00lp.ttf`;
Goodfire uses `Noto Sans`, `Noto Sans CJK SC`, and `Noto Sans CJK TC`. Installing
only the main Cabin/NewTX fonts is insufficient. Record the resolved font-file
hashes, TeX Live/LuaHBTeX version and rasterizer versions, not only family names.
The old `paper-data/language_steering_speed/token-labels.otf` is not a runtime
requirement of the current tracked renderer; copying it would not fix the
missing current preambles.

### 2. Make the actual DAG select the original per-renderer environments

At initial inspection `pyproject.toml` pins Matplotlib **3.11.0**, and
`pipeline.command()` invokes every task with the same `sys.executable -m`.
Original norm/context/speed `.render.json` files explicitly record **3.10.9**
for both styles. Most current scripts' PEP 723 headers specify
`numpy==2.2.6, matplotlib==3.10.9`. The current bubble renderer instead specifies
`numpy==2.5.3, matplotlib==3.11.2, pandas==3.0.6, pyarrow==25.0.1,
scipy==1.18.1`. A module invocation does **not** honor its PEP 723 header.

**Small fix:** extend task commands with a render-environment/profile selector,
using two separately locked environments, and make `reproduce plan/run` use it.
`RENDER.md` already documents separate `uv run --no-project --with ...` recipes;
those recipes must be connected to the advertised top-level command. If a
specified version cannot be installed, fail explicitly rather than replacing
it silently. Preserve a separate compute stack (NumPy 2.2.6, Torch 2.11.0,
Transformers 5.8.0); the bubble renderer's NumPy is not a reason to change
experiment RNG/numerical dependencies.

**Resolved and verified:** the parent added mutually exclusive `render` and
`render-bubble` extras with these pins. Current task records route all 14
non-bubble scientific renderers and both `inline_tikz` tasks to `render`, and
bubble rendering to `render-bubble`. I inspected the expanded commands:
`uv run --locked --isolated --no-default-groups --extra ENV python -m ...`.
The GIF/field correctly use neither plotting extra. No `uv sync` was invoked
for this final inspection, avoiding concurrent environment changes.

### 3. Close reference-check coverage and distinguish pixels from encoded bytes

At initial inspection `check_reference()` compared PNG **file** hashes, skipped
entries without a PNG hash, and had no reference entries for the inline artworks.
The fish entry had only a shared PDF hash, so a successful check did not certify
it. PDF timestamps preclude treating arbitrary PDF byte hashes as pixel tests.
PNG compression/metadata can likewise change bytes without changing pixels.

**Small fix:** record coverage explicitly for all 18 artworks; report missing
reference evidence as `UNVERIFIED`, not success. For ordinary PNGs add hashes of
canonical decoded dimensions/mode/pixel bytes. For PDFs use a pinned rasterizer,
DPI, background and crop rule. Include the shared fish output and both inline
artworks; exact inline reference crops require the original compiled manuscript
or independently verified artwork export. Keep original file hashes as a
separate provenance check. Do not manufacture a reference by treating a new
reconstruction as the old figure.

**Update:** parent implemented canonical RGBA pixel hashes, a fish check over
embedded image pixels/page geometry/content, explicit `UNVERIFIED` inline
coverage, and a nonzero exit for incomplete coverage. This addresses the silent
pass and PNG-encoding issues. Original inline references remain unavailable;
that is correctly reported as unverified rather than silently accepted.
Final manifest inspection found canonical pixel references for **all 15**
scientific artworks, fish embedded-image evidence, both inline artworks marked
unverified, and an explicit `exactness_blockers` record naming the four missing
original style files.

### 4. Preserve the *selected* protocols, including manuscript discrepancies

The following are requirements, not interchangeable choices or suggested reruns.
They are recoverable from source/metadata and should be asserted in small tests:

| Artwork | Exact selection/protocol evidence |
|---|---|
| Natural norm + context | Norm seed `20260908`; context must reuse its anchors/tangents rather than resample. |
| Synthetic norm | Main panels use blocks **21/21/27**, with Llama from `great_circle_synthetic_grid_llama21` and OLMo/Qwen from `great_circle_synthetic_grid_best_layers`. Original figure sidecar records both filenames and data hashes. |
| Synthetic speed | Uses original midpoint **16/16/18**, not the AUC-selected norm blocks. |
| Natural speed | Distinct sampling seed `20260910`; renderer selection seed `20260910`, not mean curves. Highlighted sample IDs below. |
| Trajectory main/appendix | `norm_growth_trajectory_starts_all_layers`; main Llama height 1.25, appendix OLMo/Qwen height 2.2; `--starts all`. |
| Pair speed main/appendix | Final `pair_barrier_validation` export, 128 pairs/class, four-block float64 audit of float32 prepared inputs; not old exploratory `great_circle_pair_speed_grid*` files. `CORE.md` documents the GPU audit provenance. |
| Triple maps | `triple_area_heatmaps_6blocks`, OLMo blocks 16–21, not the older similarly named output directory. |
| Bubble panels | Current renderer uses supplemental natural pair sample **5** first (`[5,1,2,3,4]`), singles projected `(x,z)`, pairs `(x,y)`. Supplemental selection seed `20261004`, tangent seed `9005`, source producer `cc2a8ac:paper_scripts/gen_bubble_coords.py`. |
| Language speed | English IDs `[104,61,8,14,124]`, seed `20260920`; current Chinese IDs **`[409,397,357,449]`**, seed `20261026`. Do not restore replaced ID `338`. |
| Language area vs speed | Separate centroid corpora/protocols; sharing a convenient centroid cache would change the experiment. |
| Refusal maps | Blocks 13–16, first displayed token is the assistant-header newline, one-time steering at that position, displayed vertical crop `[-4,4]` from larger source maps; do not steer every decoded token. |
| Synthetic profile trajectory | Current renderer reads `llama3.npz` and `shuffled.npz`, not the Gaussian file merely present alongside them. |

Natural-speed highlight IDs from the original sidecar:

- Llama: `[5,0,14,18,9,19,2,15]`
- OLMo: `[16,0,2,6,8,9,3,18]`
- Qwen: `[9,7,9,8,16,10,12,13]`

The Chinese replacement metadata specifies replaying the original
`choice(eligible,4,replace=False)` and continuing the same RNG to the first unused
candidate, not a fresh five-sample draw. Fixed source records in the standalone
JSON already retain ID 357. Keep these records as **source inputs** and assert
IDs/token positions; there is no need to distribute old activation arrays.

**Manuscript discrepancy to disclose, not silently repair:** the bubble caption
says layers 16–20, but the recovered producer and
`paper-data/bubble_coords_extra.json` identify zero-based injection/readout
`[15,25]`. The historical producer documents forwarding L15 to L25 and a
symmetric pair radius/frame, which is also more specific than the appendix's
informal construction. Follow the recovered producer for exact current pixels;
log this discrepancy instead of changing the experiment to match the caption.

### 5. Freeze execution identity, not only RNG seeds

Core code now pins model revisions and the Pile source revision **and file
hash**. Maps pin the Wikipedia revision and include actual token source records;
the speed corpus additionally checks all passage hashes. These are concrete
provenance protections, not outstanding missing-seed warnings.

Remaining exactness evidence needed after implementation: record device model,
CUDA/cuBLAS build, Torch/Transformers versions, batch size, thread count,
attention backend, TF32 settings, and native-state/probe dtypes per task.
`CORE.md` correctly preserves mixed CPU/GPU preparation and separate float64
pair auditing. A blanket switch to CPU or a larger batch is not an exactness fix.
Store generated-array hashes and compare final artwork against the old reference;
existing numerical tolerance assertions establish consistency, not bit identity.
For Wikipedia area centroids, also add expected selected passage hashes/IDs to
the corpus protocol if source metadata supplies them, as already done for speed.

**Core-agent provenance verification:** all original core metadata records
Torch **2.11.0+cu130**, Transformers **5.8.0**, NumPy **2.2.6**. Final pair
validation JSON explicitly says `precision=float64, device=cuda`; an earlier
working assumption that this final audit was CPU is incorrect. Retain CUDA
float32 preparation followed by CUDA float64 auditing of those exact inputs,
including the HF RMSNorm/softmax float64 patch. Synthetic-speed midpoint
reference pools were CPU-extracted but speed evaluation was CUDA. Trajectory
midpoint blocks 15–20 were CPU-prepared (block15 full-document arithmetic,
blocks16–20 causal-truncated), then replaced into CUDA all-layer inputs.
The core agent independently regenerated **all 256 reference + 128 heldout
record dictionaries for each of three models** exactly from pinned public
inputs/tokenizers, verified **424 exclusion rows/text hashes**, and checked
AST equality of the synthetic, derivative, downhill and cache-extraction
kernels against the available producer sources. These are positive provenance
checks, but not proof that every producer is the historical rendering-era version.

**Historical producer-hash caveat:** the core agent's final
`configs/core.json.source_provenance` records ten source hashes. Available
producer hashes differ from archived provenance for natural norm, midpoint
synthetic norm/speed, midpoint trajectory, and pair validation; selected
synthetic norm and natural speed match. The pair helper's current hash starts
`5547d9...`, versus archived `b3d4ff...`; its original source was untracked and
not recoverable from git. Preserve this distinction in the manifest rather than
identifying the current files as byte-identical historical sources. Smallest
source recovery request: original versions corresponding to these recorded
hashes, particularly the pair helper, from the original producer workstation.

The core agent separately reports diagnostic agreement of the current helper
with all 1,152 coarse and 1,152 half-step archived pair scores, plus identical
synthetic arrays for three midpoint and three selected pools. This audit did
**not** run those archived-data diagnostics; they are attributed agent evidence,
not an approved runtime dependency or proof that full forwards are unchanged.
Do not ship/load their archived numerical inputs in the standalone pipeline.
Full-size GPU outputs have still not been regenerated, so they do not prove
bitwise equality. `configs/core.json` has now landed; renderer dependency edges
remain to be connected at this audit checkpoint.

**Maps-agent source limitations (not missing RNG seeds):**

- Bubble's historical `fineweb_sents_triples` raw sentence list survives, but
  original FineWeb document IDs, dataset revision and sentence-selection
  procedure do not. Keep this literal list as an immutable public-text input,
  hash it and state that corpus acquisition starts there. It permits fresh
  model computation; it does **not** establish independent regeneration of
  that list from FineWeb. If upstream-corpus regeneration is required, recover
  its document IDs/revision/selection script from the originating workstation.
- The original bubble model revision was unpinned. Using Llama snapshot
  `d04e592...` is supported by the supplemental sample metadata but does not
  independently establish the older base artifact's checkpoint. Identify this
  as an inferred historical pin unless an original model/cache manifest is
  recovered; matching regenerated original-sample outputs would provide
  additional evidence, but no such full-scale test has been run here.
- Language-speed's original Wikipedia fetch was unpinned. The later audited
  `b04c8d...` revision is acceptable only together with the implemented
  **every-passage hash check**; do not remove this gate or claim the original
  fetch itself was revision-pinned.
- Frozen token source records replay the selected/manual-corrected anchors,
  not the entire upstream cohort-selection and human-review history. Keep
  them as explicit immutable inputs with selection explanations. English
  records `[104,61,8,14,124]` are retained, but the current renderer **omits 8**;
  Chinese display uses `[409,397,357,449]`. Neither a fresh random resample nor
  showing all five English anchors reproduces the current figure.

The maps agent found no missing experiment seeds and reports 12 CPU tests
passing; full-scale GPU output equivalence remains unverified. These details
are distinguished from the editorial-label mismatch resolved in section 6.

### 6. Preserve human-authored glosses as source, not a numerical approximation

The initial `maps/language_speed.py` wrote `gloss=r['text']` literal-token
replacements. These were schema-compatible but **not the current figure's
annotations**. This was reported and has now been fixed as described below.
Original `paper-data/language_steering_speed/glosses.json` contains 48
human-reviewed token entries; `glosses_chinese.json` has 47 token entries,
four record-keyed `interlinear_contexts`, and grammatical abbreviations.
For example token 31640 labels `代 → era`, while record 397 decomposes the
context into Gauss/Germany/mathematics. Numerical model execution cannot
recreate those editorial judgments. Byte-fragment notation and script-class
labels also differ from naive Unicode/literal replacements.

**Smallest fix:** preserve these annotation-only JSON fields verbatim under
source `assets/` or `maps/inputs/`, with original hashes, and apply them after
freshly computing token regions. Assert regenerated token IDs and selected
record IDs are covered; fail on unknown IDs instead of silently inventing
new labels. They contain source labels/translations, not activation tensors,
probabilities or speed curves, so bundling them does not require bundling old
experimental results. If the source-only policy excludes even human-authored
labels, list this figure as blocked for exact reproduction, not reproduced
using visibly different fallback text.

**Resolved and independently inspected:** source annotations now live in
`maps/inputs/language_speed_labels{,_chinese}.json`. Byte hashes exactly match
the originals:

- English: `0ce3a75e07896093209a262657f19d3304cb17813e62536d70da60ae4768861e`.
- Chinese: `53ded4659ce279264b3f46e4e0a1ad1288131cb6e3f895d15d5f6efbdacd6fe1`.

Current `source_labels()` applies them only after fresh region generation,
requires every token ID/text/token-piece to match, checks the selected
interlinear record set/target/context, and fails instead of inventing fallback
labels. No numerical curve/region results are copied into source inputs.
This removes the editorial exactness blocker; it does not establish full model
output equality or resolve typography.

Also correct the `MAPS.md` explanatory typo: the replacement is **338 → 357**,
not 338 → 449. The implemented final ID list already contains the correct
`[409,397,357,449]`.

## Additional PDF/font forensics (requested by parent)

Inspected PDF dictionaries/font resources with the standalone `.venv`'s
`pypdf`; read embedded font naming tables in memory with `fontTools`. No PDF
rasterization or reuse as a generated output was performed.

**All 30 scientific PDFs** (15 artworks × two styles) record:

```
/PTEX.FullBanner: This is LuaHBTeX, Version 1.21.0 (TeX Live 2025)
```

Their `/Creator` and `/Producer` agree on Matplotlib **3.10.9**, except both
bubble PDFs, which explicitly identify **3.11.2**. They were rendered on
2026-09-29. The shared fish PDF has no TeX/font metadata, as expected.
This establishes an actual historical TeX target, not a guess from the source.
It does not identify the exact TeX Live package-update snapshot or PNG converter.

Embedded family names below omit PDF subset prefixes:

| Profile/artwork | Families actually embedded |
|---|---|
| ICLR ordinary profiles | `TeXGyreTermesX-Regular`, `NewTXMI7`, `txsys` (subset depends on labels) |
| ICLR bubble/trajectory | `TeXGyreTermesX-Regular` |
| ICLR language-area | Above text/symbol fonts plus **`LMMono8-Regular`** |
| ICLR language-speed | Above text/math fonts plus **`BousungEG-Light-GB`**, **`ShanHeiSun-Light`** |
| Goodfire ordinary profiles | **`Cabin-Regular`**, **`NotoSansMath-Regular`** |
| Goodfire bubble/trajectory | `Cabin-Regular` |
| Goodfire language-area | `Cabin-Regular`, `NotoSansMath-Regular`, **`LMMono8-Regular`** |
| Goodfire language-speed | `Cabin-Regular`, `NotoSansMath-Regular`, **`NotoSansCJKsc-Regular`**, **`NotoSansCJKtc-Regular`** |
| Triple heatmaps (both) | No font resources (unlabelled panels) |

Versions retained in the embedded subsets:

- Cabin: **3.001**.
- Noto Sans Math: **3.000; ttfautohint (v1.8.4.7-5d5b)**.
- TeX Gyre Termes X: CFF **2.004**, Michael Sharpe 2015–2023 modifications.
- Latin Modern Mono 8: CFF **2.004**.
- Bousung: **2.11** (`AR PL SungtiL GB`).
- ShanHeiSun: **2.10** (`AR PL Mingti2L Big5`).
- Noto CJK CFF subsets retain Adobe 2014–2021 copyright but no version field;
  do not infer a precise release from copyright alone.

**Two concrete implications for the reconstructed styles:**

1. Goodfire requires **Noto Sans Math**, not simply Noto Sans text plus
   `mathastext`. The currently reconstructed files do not name this mathematical
   font. Add the actual family to setup requirements and restore the original
   glyph routing when its preamble can be recovered; guessing a different
   `mathastext` configuration cannot certify the same glyphs/metrics.
2. Goodfire language-map typewriter labels used **Latin Modern Mono 8**, not the
   manuscript's Inconsolata. Do not blindly copy the manuscript `\ttdefault`
   configuration into the figure preamble. Use font-resource comparisons as a
   cheap preflight test before claiming pixel equivalence.

### Narrow source-recovery search

No original style files recovered from:

- `git log --all -- styles` or `-- '*fonts.tex' '*math.tex'` in the paper,
  covering local branches and the origin references.
- The equivalent style/font path-history queries in `~/project`.
- Filename searches excluding `.git`/`.venv` under `~/paper`, `~/project`
  (including its worktree), `~/paper-archive-20260930-214925`, and
  `~/project-archive-20260616`.
- The supplied paper archive manifest/patch; no `styles/` entries.
- Targeted `.cache/matplotlib/tex.cache` search for Cabin, Noto Sans Math,
  `figurefont`, `figurecaption`, or `Renderer=HarfBuzz`: no hits. Older cached
  TeX snippets do contain `newtxtext`/`newtxmath`, but are not the current LuaTeX
  profile preambles and must not be presented as recovered originals.

The exact remaining recovery request is therefore the four style files from
**the original 2026-09-29 LuaHBTeX rendering workstation**, ideally its TeX Live
package/version manifest and font-file hashes. Embedded PDF subsets provide
families/versions and metrics for used glyphs, but not the original complete
preamble, full font files or all layout decisions.

## Tests and inspection performed

- Confirmed paper HEAD and read complete main/appendix, including commented
  methodological details and active versus commented-out table inclusions.
- Checked original renderer SHA256 against ICLR `.render.json` for natural norm,
  natural speed, synthetic norm, pair speed and trajectory: all five match.
  Thus those sidecars really describe the current tracked renderer sources.
- Initial standalone test run:
  `PYTHONDONTWRITEBYTECODE=1 uv run --locked --group dev pytest -q -p no:cacheprovider`:
  **6 passed, 1 failed**. Failure was `test_manifest_is_complete`: at that instant
  only `configs/reference.json` existed and `load_tasks()` was empty. This was
  an in-progress integration observation, not proof of an irrecoverable blocker.
- Intermediate package test run after render task integration: **8 passed**.
  At that instant scientific render tasks existed but had empty `deps`; the
  passing manifest test did not yet verify from-scratch experiment edges.
- **Latest test checkpoint:** direct `.venv/bin/python -m pytest -q
  -p no:cacheprovider`, with bytecode disabled: **11 passed, 1 failed**.
  The parent strengthened `test_manifest_is_complete` to require generation
  dependencies. It now correctly fails on `render-great_circle_norm_grid`
  because scientific tasks still have `deps: []`; no experiment task configs
  were present yet. This is an actionable, transient integration gate: add
  the core/maps task configs and connect renderer deps, then rerun. It is not
  a rendering-version failure.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest
  manifold_attractors.core.test_core -v`: **10 passed in 46.652 s**, including
  the tiny randomly initialized CPU Llama end-to-end generator test. All its
  inputs/results were generated in a temporary directory, with no pretrained
  weights/downloads/archived experimental data. This validates core APIs and
  contracts, not final 8B numerical equality.
- Fresh fish rendering from the bundled **source GIF**, into a temporary directory,
  succeeded: PDF plus RGB PNG of **700×350**. Source GIF SHA256:
  `ff511d95f85275e673afeddd752d4c963883e3e197756ecf8e6dd1b47a2dff91`.
  No old figure/data was used to generate it; this audit did not claim old/new
  pixel equality from that smoke test alone.
- `lualatex`, `latex`, `pdftocairo`, `pdftoppm`, and `gs` were absent from PATH.
  Scientific PDF and inline-TikZ compilation therefore remain untested here.
- Identified that the initial resume fingerprint omitted `styles/`, `assets/`
  and `maps/inputs/*.json`; reported it to the parent. **Parent corrected this
  during the audit.** A subsequent isolated temporary-root test confirmed that
  changing a style or adding a frozen-record JSON now changes `code_digest()`.
  Current digest also includes Python version. This resolved issue should not
  remain an outstanding blocker.

No full GPU/CPU model experiment was launched by this audit. No old experimental
results were copied, loaded for rendering, or used as caches. Recheck the final
DAG/tests after all three implementation agents finish; the package is changing
concurrently. The missing-original-font caveat remains regardless of how many
structural tests pass.
