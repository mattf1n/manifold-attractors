# Paper-table reproduction

The scope is the **three active table inclusions** in paper `ed8b89b`:

| Table | Source | Fresh computation |
|---|---|---|
| Norm-growth classifier | Appendix, `tab/norm_growth_classifier` | All-layer heldout natural/Gaussian/shuffled norm-growth scores for three models |
| Pair-barrier validation | Appendix, `tab/pair_barrier_validation` | The same fresh float32-prepared/float64-audited pair experiment used by the pair-speed figures |
| Adaptive steering | Main text, `tab/adaptive_steering` | Frozen tuning + ID/OOD test protocol and linear oracle; separate mass and greedy objectives |

The commented-out velocity-classifier table and old unused exploratory tables
are not included in the current paper. No original result values are supplied
to the generators or copied into the reproduced tables.

`render.tables` preserves the original numerical selection and LaTeX formatting
from the paper's three `scripts/tab/*.py` files. Norm and pair AUCs are recomputed
from the freshly generated scores, with half credit for ties. The norm table
shows the same 15 rounded linspace layers from 1 through 36; unavailable layers
for 32-block models are em dashes. Both AUC tables print three decimals.
Adaptive steering prints two-decimal percentages for Untuned, Tuned, Adaptive,
and Oracle across mass/greedy and ID/OOD tracks. Non-Chinese means Latin plus
Other alphabet, **excluding neutral**, rather than one minus Chinese.

Table formatting runs in its own locked `table` extra: NumPy2.2.6,
Pandas3.0.6, Jinja2 3.1.6. This avoids replacing the compute stack's Pandas pin.
It produces `.tex`, `.md`, and numeric `.json` outputs; formatting itself needs
neither a GPU nor TeX. The `.tex` snippets compile inside the manuscript using
booktabs/siunitx. The same generated values can be embedded directly in the README.

```sh
uv run --locked --isolated --extra table python -m manifold_attractors.render.tables \
  norm_growth_classifier --data data/norm_growth_classifier_scores
uv run --locked --isolated --extra table python -m manifold_attractors.render.tables \
  pair_barrier_validation --data data/pair_barrier_validation/summary.json
uv run --locked --isolated --extra table python -m manifold_attractors.render.tables \
  adaptive_steering --data data/language_steering_confirmatory
```

Classifier scores are written separately from the native activation pools used
by figures, so generating the table cannot overwrite an existing figure input.
The pair table shares the figure experiment; no redundant pair forwards are
necessary. Adaptive steering must preserve the published tuning/test separation
and corpus protocol, not substitute the illustrative language-figure examples.

All three tables have now completed fresh computation, formatting and validation.
Norm-growth and pair-barrier scores/printed tables match exactly. Adaptive
steering matches every greedy count, fitted parameter and method ranking; only
Chinese-mass cells differ, by at most0.01712 percentage points, from an explicitly
diagnosed Unicode-version effect. See [TABLE_CLASSIFIER.md](TABLE_CLASSIFIER.md),
[TABLE_STEERING.md](TABLE_STEERING.md), and `docs/tables/validation.json`.
Formatter-only tests are not the basis of these full-size validation claims.
