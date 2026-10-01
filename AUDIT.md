# Reproduction audit

## Full-run update and acceptance criterion

The current user requirement accepts **scientifically inconsequential** rounding
and typography differences; pixel identity is optional. The strict historical
source assessment below is retained as provenance, not a requirement to recover
unavailable fonts before producing useful figures. Rendering permits documented
reconstructed typography by default; `--require-original-styles` opts into the
old strict gate.

Full-size independent comparisons identified and distinguished three cases:

- **Substantive sampling regression (corrected and verified):** natural norm figures
  historically sampled all nonspecial positions. A later source revision
  excluded position zero, changing 14/20 OLMo and 13/20 Qwen anchors even though
  no historical selected token was at zero. Replaying seed20260908 with the
  historical candidate rule recovers all60 original selections. The norm CLI
  and recipe now explicitly restore that rule; other speed/synthetic/pair
  protocols retain their separate positive-position policy. Fresh norm/context
  forwards were rerun rather than substituting archived activations. Every
  numerical array now matches the historical reference exactly for all three
  models, for both fixed-probe and matched-document-context norm figures.
- **Selected synthetic norm grids:** every array also matches exactly for the
  independently regenerated Llama21/OLMo21/Qwen27 panels.
- **Inconsequential arithmetic drift:** six triple-area maps agree within
  relative1e-5 (observed maximum <3.514e-6). Historical native extraction used
  A100; the fresh run used H100. Only10 of251430 quantized source colors and
  41/48 final ICLR/Goodfire pixels differ. No rounding or output manipulation
  is applied to force a match. The user accepted this degree of difference;
  an A100 replay is unnecessary.
- **Exact bubble science, harmless typography:** all13 columns of both bubble
  tables (60025 +4002 rows) are exact. ICLR pixels are exact. Goodfire surface
  images/geometry are exact; almost all changed pixels are Cabin font outlines
  and hinting. Matching a font family/version did not identify the same binary:
  historical Cabin is TrueType, the initial reconstruction used CFF/OpenType.

Historical arrays were read **only as external validation references** after
fresh computation, never as generator inputs or result replacements. Published
per-artwork evidence is in `docs/figures/validation.json`; pending scientific
comparisons must not be interpreted as passing.

### Completed figure run

All18 artwork components have now been rendered in both ICLR/Goodfire styles
(36 published style-specific outputs). Every model-data family was independently
generated and scientifically validated. Additional full-run findings:

- Natural-speed grids and synthetic-profile trajectories match all compared
  arrays exactly.
- Trajectory natural baselines and validity masks are exact; the maximum error
  in plotted synthetic-cohort means is3.802e-6. The worst individual Qwen norm
  ratio relative difference is0.01868%, an inconsequential Float32 effect.
- Synthetic-speed displayed curves differ by at most0.001344% relative.
- Pair-speed Qwen curves differ by at most0.01308% relative. All three models'
  AUCs, conditional bootstrap intervals and barrier counts are exactly unchanged.
- Language-area primary fields agree to relative6.358e-6, with all31 displayed
  local minima preserved. Language-speed primary curves agree to relative1.500e-6;
  all11889 exported/refinement token IDs, selections, regions and glosses match.
- Refusal areas differ by up to0.14010% after independent BF16 direction
  extraction on different GPU hardware. The selected position/layer, displayed
  tokens, basin counts and compliance/refusal continuations are unchanged. This
  is accepted scientific drift, **not** claimed to be1e-5 equivalence.

Ten of36 artifacts also pass the optional strict image check; remaining pixel
identity is not a requirement. Detailed per-model metrics/exceptions and hashes
are published in `docs/figures/validation.json`.

The added norm-growth classifier table reproduces every score exactly and its
formatted LaTeX is exact. The pair-barrier table likewise matches exactly.
Adaptive-steering table computation is separate and still pending; these figure
and table results do not constitute a claim that the expanded full scope is done.

## Original source assessment

Target: `~/paper` commit `ed8b89bc8344db678eb5e360a1893937d3b94a1f`.
Standalone repository: `~/manifold-attractors`. Audit date: 2026-09-30.
This audit edited only `AUDIT.md`.

## Verdict

**Exact from-scratch reproduction of every current artwork is not established.**
The repository now explicitly distinguishes executable reconstruction from exact
reproduction. Missing original TeX preambles remain a source blocker. Historical
producer-version and upstream bubble-corpus provenance have additional gaps.
Full-size model regeneration, scientific PDF rendering and final pixel equality
have not been demonstrated by this audit.

Scope: **16 external PDF inclusions plus two inline TikZ artworks**, for ICLR
and Goodfire. Includes the introductory fish/field composition and sphere-probe
cartoon; excludes tables and manuscript pagination. I read complete `main.tex`
and `appendix.tex`, active Makefile recipes, relevant source, selected original
JSON provenance and original PDF metadata/font resources. No archived numerical
arrays, result parquet files or activations were used as runtime inputs by this
audit. Public source text, source media and editorial annotations are inputs,
not precomputed experimental results.

## Completed integration fixes

These are **resolved**, not outstanding requests:

- Per-task locked environments: standard renderers use NumPy **2.2.6** /
  Matplotlib **3.10.9**; bubbles use NumPy **2.5.3**, Matplotlib **3.11.2**,
  Pandas **3.0.6**, PyArrow **25.0.1**, SciPy **1.18.1**. Mutually exclusive
  extras and `task.environment` select isolated execution. Module invocation
  no longer silently disregards the differing source PEP 723 pins.
- Core/maps experiment tasks and scientific renderer dependency edges are
  integrated. A late stale refusal-wrapper dependency detected during audit
  was corrected; the final independent test rerun passes.
- Resume fingerprints include source/configuration, `styles/`, `assets/`,
  source-record JSON and Python identity. I verified style/record mutations
  invalidate the digest.
- Reference checks use canonical decoded RGBA hashes for all **15 scientific
  PNGs**; the shared fish PDF is checked by embedded decoded image, page content
  and media box. Both inline artworks explicitly remain `UNVERIFIED`; incomplete
  coverage cannot falsely pass. Original file hashes remain provenance evidence.
- Missing original styles are machine-readable exactness blockers. Preflight
  stops before expensive work; `--allow-reconstructed-styles` explicitly permits
  only an **unverified reconstruction**, and still requires LuaLaTeX.
- Original language-speed editorial labels are preserved byte-for-byte as
  source inputs. Freshly computed regions must match token IDs/text/pieces;
  interlinear record sets, target tokens and contexts are checked. Unknown or
  changed inputs hard-fail; there is no literal-label fallback.
- Parent reports fresh sparse-source downloading for the isolated refusal
  extraction wrapper was verified, without retaining experimental outputs.

## Remaining source blockers and smallest recovery actions

### 1. Four original style files and exact TeX/font environment

The original renderer requires:

```
styles/iclr-fonts.tex
styles/goodfire-fonts.tex
styles/figure-fonts.tex
styles/figure-math.tex
```

These are absent from the supplied paper checkout/history/archive. Included
replacements identify themselves as reconstructions. **Recover these four files
from the original 2026-09-29 rendering workstation**, plus its TeX package
manifest/font-file hashes. No experimental tensors are needed for this recovery.
Until then, keep the preflight exactness blocker. The available `matplotlibrc`
is verified source, SHA256
`3608116a2bc4edf1d33a385254b15dc31b50f9b3de5778cbc91088220c33ff01`.

PDF forensics below identifies actual font versions and two specific mismatches
with a naive manuscript-derived preamble: Goodfire **Noto Sans Math** and
**Latin Modern Mono 8**. Font-family similarity is not proof of matching metrics.

### 2. Historical producer versions

`configs/core.json.source_provenance` records ten source hashes. The core agent
reports available producer hashes differ from archived provenance for natural
norm, midpoint synthetic norm/speed, midpoint trajectory and pair validation;
selected synthetic norm and natural speed match. In particular, the pair helper
is currently `5547d9...`, versus archived `b3d4ff...`; the old helper was untracked
and unavailable from git.

**Smallest recovery:** obtain original files corresponding to recorded historical
hashes, especially the pair helper. Otherwise preserve this caveat, rather than
calling available sources byte-identical historical producers. AST equality
proves faithful extraction from available code, not historical source identity.

The core agent separately reports exact diagnostic agreement for 1,152 coarse
and 1,152 half-step archived pair scores and synthetic arrays from three midpoint
plus three selected pools. This audit did **not** run those archived-data
checks. They are attributed supporting evidence, not an approved runtime cache
or proof of unchanged full-size forward outputs.

### 3. Bubble upstream corpus and model pin

The literal historical `fineweb_sents_triples` sentence list survives and permits
fresh model computation. Its original FineWeb document IDs, dataset revision and
sentence-selection procedure do not. Treat the hashed raw sentence list as the
explicit acquisition boundary. **If independent upstream-list reconstruction is
required, recover those IDs/revision/selection code** from the originating host.
Do not replace the list with fresh FineWeb draws.

The original base bubble model revision was unpinned. Snapshot `d04e592...` is
supported by supplemental sample provenance, but not independent proof of the
older base artifact's model revision. Recover the original model/cache manifest
or retain an inferred-pin caveat. No missing experiment RNG seeds were identified.
The current bubble renderer itself reconstructs earlier missing surface shading;
the reproduction target is its **ed8b89b** logic, not that older lost shader.

### 4. Reference and execution evidence still needed

Original compiled references for the two inline artworks were not supplied.
Recover original manuscript/artwork exports and define pinned rasterization,
DPI/background/crop rules; do not elevate a newly reconstructed image into its
own historical reference. Existing checks correctly report incomplete coverage.

Full-size execution should retain device model, CUDA/cuBLAS/Torch build, thread
count, batch sizes, attention backend, TF32 and arithmetic dtypes. Preserve the
mixed CPU/CUDA protocol below. Tolerance assertions and passing CPU tests do not
establish bit identity. No TeX/rasterizer executables were available on this host.

The literal fish GIF is preserved, but its original URL/redistribution license
was not independently established; the Beal et al. Figure 4 citation is not a
license. Resolve source-media rights separately before redistribution.

## Exact selected protocols retained

| Artwork | Required selection/protocol |
|---|---|
| Natural norm + context | Seed `20260908`; context reuses norm anchors/tangents rather than resampling. |
| Synthetic norm | Blocks **21/21/27**; Llama from `great_circle_synthetic_grid_llama21`, OLMo/Qwen from `great_circle_synthetic_grid_best_layers`. |
| Synthetic speed | Original midpoint blocks **16/16/18**, CPU-extracted reference pool followed by CUDA speed, not selected norm blocks. |
| Natural speed | Sampling and renderer selection seed `20260910`; selected curves, not means. |
| Trajectory | `norm_growth_trajectory_starts_all_layers`; `--starts all`; Llama height1.25, appendix OLMo/Qwen height2.2. |
| Pair speed | Final `pair_barrier_validation` export:128 pairs/class, four blocks; **CUDA float32 preparation → CUDA float64 audit** of exact inputs; HF RMSNorm/softmax float64 patch. |
| Triple maps | `triple_area_heatmaps_6blocks`, OLMo blocks16–21, not older four-block output. |
| Bubbles | Recovered `cc2a8ac` v3 producer, not later working-tree producer; supplemental natural sample5 first (`[5,1,2,3,4]`); singles `(x,z)`, pairs `(x,y)`; supplemental seed20261004, tangent seed9005. |
| Language speed | English source IDs `[104,61,8,14,124]`, seed20260920; renderer **omits8**. Chinese `[409,397,357,449]`, seed20261026; replacement **338→357**. |
| Language centroids | Area and speed use distinct corpus protocols; no interchangeable centroid caches. |
| Refusal maps | Blocks13–16; assistant-header newline is first displayed state; one-time steering there; display vertical crop `[-4,4]`; no repeated token steering. |
| Synthetic profile | Current renderer uses `llama3.npz` and `shuffled.npz`, not the neighboring Gaussian artifact. |

Natural-speed highlighted samples from original sidecar:
Llama `[5,0,14,18,9,19,2,15]`, OLMo `[16,0,2,6,8,9,3,18]`,
Qwen `[9,7,9,8,16,10,12,13]`.

Chinese replacement continues the same RNG after original four-item selection,
rejecting used IDs; it is not a new five-item draw. Frozen token records preserve
selected/manual-corrected inputs, not independent reconstruction of all upstream
selection/review decisions.

**Manuscript discrepancy:** bubble caption says layers16–20, but recovered
producer/supplement identify zero-based injection/readout `[15,25]` and a symmetric
pair radius/frame. Preserve recovered producer behavior for exact current pixels;
do not silently modify it to agree with the informal caption/appendix.

Core original stack: Torch **2.11.0+cu130**, Transformers **5.8.0**, NumPy **2.2.6**.
Trajectory midpoint blocks15–20 were CPU-prepared: block15 full-document
arithmetic, blocks16–20 causal-truncated. These inputs replace corresponding
cohorts in CUDA all-layer preparation. Final pair validation is **not CPU**.
The core agent exactly regenerated all 256 reference+128 heldout record dictionaries
for each model and 424 exclusion rows/text hashes from pinned public inputs.

Speed Wikipedia acquisition was originally unpinned; later audited revision
`b04c8d...` is guarded by **every-passage hashes**. Keep that gate. Original
model/corpus identities and frozen source records are documented in `CORE.md`,
`MAPS.md` and configs; arbitrary snapshot overrides are not exact reproduction.

## Editorial labels — resolved and verified

`maps/inputs/language_speed_labels{,_chinese}.json` preserve original annotations,
not model outputs. English contains48 token entries; Chinese47 plus four
interlinear contexts and grammatical abbreviations. I independently verified
byte hashes against original source annotations:

- English: `0ce3a75e07896093209a262657f19d3304cb17813e62536d70da60ae4768861e`.
- Chinese: `53ded4659ce279264b3f46e4e0a1ad1288131cb6e3f895d15d5f6efbdacd6fe1`.

Examples such as `代 → era` are human editorial choices, not reproducible by
numerical inference. Current `source_labels()` validates token/record/context
compatibility after fresh generation and fails instead of substituting labels.
No curve/region statistics were bundled to resolve this issue.

## Original PDF/font forensics

Using `pypdf` and in-memory `fontTools` inspection (no rasterization), all **30
scientific PDFs** identify **LuaHBTeX1.21.0 / TeX Live2025**, rendered2026-09-29.
Matplotlib metadata is **3.10.9**, except both bubble PDFs **3.11.2**. This does
not identify the exact TeX Live package-update snapshot or PNG converter.

| Profile/artwork | Actual embedded families (subset prefixes omitted) |
|---|---|
| ICLR ordinary profiles | `TeXGyreTermesX-Regular`, `NewTXMI7`, `txsys` as required by labels |
| ICLR bubble/trajectory | `TeXGyreTermesX-Regular` |
| ICLR language-area | Text/symbol fonts plus `LMMono8-Regular` |
| ICLR language-speed | Text/math fonts plus `BousungEG-Light-GB`, `ShanHeiSun-Light` |
| Goodfire ordinary profiles | `Cabin-Regular`, **`NotoSansMath-Regular`** |
| Goodfire bubble/trajectory | `Cabin-Regular` |
| Goodfire language-area | Cabin, Noto Sans Math, **`LMMono8-Regular`** |
| Goodfire language-speed | Cabin, Noto Sans Math, `NotoSansCJKsc-Regular`, `NotoSansCJKtc-Regular` |
| Triple heatmaps (both) | No fonts; unlabelled panels |

Versions retained in subsets: Cabin **3.001**; Noto Sans Math **3.000;
ttfautohint(v1.8.4.7-5d5b)**; TeXGyreTermesX and LMMono8 CFF **2.004**;
Bousung **2.11**; ShanHeiSun **2.10**. Noto CJK subsets retain Adobe2014–2021
copyright, not an exact version.

Goodfire mathematical glyphs therefore require **Noto Sans Math**, not merely
Noto Sans text plus `mathastext`. Goodfire language-map mono labels used **Latin
Modern Mono8**, not the manuscript's Inconsolata. These are concrete recovery
constraints; guessed font routing cannot certify identical metrics. Current
language-label source also names Tempora, Computer Modern Unicode and AR PL
CJK files; the old `token-labels.otf` is not a current runtime dependency.

No original style files recovered from paper/project `git log --all` path
queries, targeted filename searches under paper/project and supplied archives,
archive manifest/patch, or relevant Matplotlib TeX cache. Cache snippets with
`newtxtext/newtxmath` are older and are not recovered current LuaTeX preambles.
Embedded subsets reveal used fonts/metrics, not full fonts or original preambles.

## Tests and evidence

- Original current renderer hashes match five checked ICLR sidecars: natural
  norm, natural speed, synthetic norm, pair speed, trajectory.
- Core audit run: **10 tests passed in46.652s**, including tiny randomly
  initialized CPU Llama execution through all generators, both trajectory
  stages and float32→64 pair auditing/export. All inputs/results were freshly
  generated in temporary directories; no pretrained weights were downloaded.
- Fresh source-GIF render succeeded: RGB **700×350** PNG and PDF. Source SHA256:
  `ff511d95f85275e673afeddd752d4c963883e3e197756ecf8e6dd1b47a2dff91`.
  This smoke test alone did not establish equality to the historical PDF.
- Font/source-record digest invalidation was independently tested.
- `lualatex`, `latex`, `pdftocairo`, `pdftoppm`, `gs` absent; scientific TeX
  rendering, inline composition and final pixel comparisons untested here.
- **Final independent package run:38 passed in28.52s**, using
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`.
  Parent's earlier suite reported47 passing tests. An intervening rerun caught
  a stale refusal-wrapper task dependency; parent corrected it and the final
  current suite above passes. No outstanding DAG test failure is claimed.
  No agent-owned config was edited here.

Only the tests above and attributed agent evidence are claimed. Full-scale
numerical and typography equivalence remain unverified regardless of structural
suite success. No old experimental outputs were copied or used as runtime
caches by this audit.
