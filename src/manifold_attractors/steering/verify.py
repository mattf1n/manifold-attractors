"""Fail-closed final data target for both adaptive-steering table summaries."""
import argparse
from pathlib import Path
import numpy as np
from .common import DEFAULT_ROOT, atomic_json, digest, read
from .compute import require_computed
from .vocabulary import CLASSES


def verify(root):
    require_computed(root)
    completion = read(root / "linear_oracle/complete.json")
    assert completion["complete"] and completion["fresh_outputs"]
    assert completion["request"]["computed_sha256"] == digest(root / "computed.json")
    assert completion["summary_sha256"] == digest(root / "linear_oracle/summary.json")
    baseline = read(root / "summary.json")
    oracle = read(root / "linear_oracle/summary.json")
    assert baseline["n_per_test"] == 10000 and baseline["calibration_n"] == 1000
    assert set(baseline["datasets"]) == set(oracle["datasets"]) == {"wiki_test", "fineweb_test"}
    for dataset in baseline["datasets"]:
        for track in ("mass", "greedy"):
            rows = [baseline["datasets"][dataset]["methods"][f"{track}/{method}"]
                    for method in ("raw_centroid", "additive", "adaptive_straight")]
            rows.append(oracle["datasets"][dataset]["oracles"][track])
            for row in rows:
                mass = np.array([row["mean_masses"][key] for key in CLASSES])
                counts = np.array([row["argmax_counts"][key] for key in CLASSES])
                assert np.isfinite(mass).all() and (mass >= 0).all()
                np.testing.assert_allclose(mass.sum(), 1, rtol=0, atol=2e-12)
                assert (counts >= 0).all() and counts.sum() == 10000
    atomic_json(root / "data_complete.json", {
        "complete": True, "task": "data-adaptive_steering", "fresh": True,
        "outputs_sha256": {name: digest(root / name) for name in
                           ("summary.json", "linear_oracle/summary.json")},
        "oracle_certified_continuous_bound": False,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    verify(p.parse_args().root)
