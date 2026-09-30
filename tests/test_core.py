"""Pure core regressions, runnable without Torch, model weights or downloads.

Covers source-derived token eligibility and raw vector-speed tests, synthetic
moment preservation, downhill pair barriers, AUC, pinned-input rejection and
seeded record/exclusion reconstruction. The heavier tiny-model integration
suite remains exposed separately by tests/test_experiments.py.
"""
from manifold_attractors.core.test_core import CoreTests
