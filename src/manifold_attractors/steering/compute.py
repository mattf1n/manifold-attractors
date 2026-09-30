"""Run frozen calibration, geometry, decoding and full-cohort summaries.

Sequential local orchestration only: this module never allocates cluster jobs.
Use fixed/geometry modules for externally scheduled per-shard parallelism.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from .common import DEFAULT_ROOT, atomic_json, digest, read, require_prepared


def invoke(module, root, *arguments, check=True):
    return subprocess.run([sys.executable, "-m", "manifold_attractors.steering." + module,
                           "--root", str(root), *map(str, arguments)], check=check)


def calibrate(root, max_expansions=16):
    from .fixed import grid_definition
    for _ in range(max_expansions + 1):
        version = grid_definition(root)["version"]
        for shard in (0, 1):
            invoke("fixed", root, "calibrate", "--shard", shard)
        result = invoke("fixed", root, "select", check=False)
        if result.returncode == 0:
            return
        if grid_definition(root)["version"] == version:
            raise RuntimeError("Calibration selection failed without an endpoint expansion")
    raise RuntimeError("Calibration expansion safety limit reached; no held-out evaluation performed")


def finalize(root):
    from .collect import collect, audit
    audit(root)
    collect(root)
    # Bind all oracle inputs, including exact comparator points and predictions.
    files = ["summary.json", "completion.json", "final_validation.json",
             "calibration/selection.json", "calibration/grid_definition.json"]
    for directory in ("test", "geometry"):
        files.extend(str(p.relative_to(root)) for p in sorted((root / directory).rglob("*.npz")))
    atomic_json(root / "computed.json", {
        "complete": True, "prepared_sha256": digest(root / "prepared.json"),
        "artifacts_sha256": {relative: digest(root / relative) for relative in files},
    })


def require_computed(root):
    require_prepared(root)
    stamp = read(root / "computed.json")
    if stamp["prepared_sha256"] != digest(root / "prepared.json"):
        raise ValueError("Compute/prepare provenance mismatch")
    for relative, expected in stamp["artifacts_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Oracle dependency changed: {relative}")
    return stamp


def compute(root, stage="all", max_expansions=16):
    root = Path(root)
    require_prepared(root)
    if stage in ("all", "calibrate"):
        calibrate(root, max_expansions)
    if stage in ("all", "test"):
        selected = read(root / "calibration/selection.json")
        if not selected["all_interior"]:
            raise ValueError("Held-out decoding blocked by boundary calibration selection")
        for dataset in ("wiki_test", "fineweb_test"):
            for shard in range(20):
                marker = root / f"geometry/{dataset}/complete_size500_{shard:03d}.json"
                if not marker.exists():
                    invoke("geometry", root, "--dataset", dataset, "--shard", shard)
                invoke("fixed", root, "test", "--split", dataset, "--shard", shard, "--adaptive", ".")
    if stage in ("all", "collect"):
        finalize(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--stage", choices=["all", "calibrate", "test", "collect"], default="all")
    p.add_argument("--max-expansions", type=int, default=16)
    a = p.parse_args()
    compute(a.root, a.stage, a.max_expansions)


if __name__ == "__main__":
    main()
