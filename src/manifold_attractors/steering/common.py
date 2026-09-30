"""Atomic run-local artifacts and fresh native-state extraction."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

DEFAULT_ROOT = Path("runs/adaptive_steering")
MODEL = "meta-llama/Llama-3.1-8B"
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
SIZES = {"wiki_cal": 1000, "wiki_test": 10000, "fineweb_test": 10000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def implementation_hash():
    """Bind resumable outputs to implementation, not just parameter names."""
    here = Path(__file__).parent
    package = here.parent
    files = sorted(here.glob("*.py")) + [
        package / "core/trajectory_activation_cache.py",
        package / "core/pair_barrier_geometry.py",
        package / "maps/inputs.py",
        package / "experiments/pair_barrier_validation.py",
    ]
    return hashlib.sha256("".join(digest(p) for p in files).encode()).hexdigest()


def freeze_run(root):
    """Do not accept an unprovenanced existing output directory."""
    root = Path(root)
    from importlib.metadata import version
    request = {
        "schema": 1, "model": MODEL, "revision": REVISION,
        "source_inputs_sha256": digest(Path(__file__).parent / "inputs/sources.json"),
        "implementation_sha256": implementation_hash(),
        "libraries": {name: version(name) for name in ("torch", "transformers", "numpy")},
        "fresh_only": True,
    }
    path = root / "run.json"
    if path.exists():
        if read(path) != request:
            raise ValueError("Run provenance changed; use a new empty --root")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing nonempty unprovenanced --root")
        atomic_json(path, request)
    return digest(path)


def require_prepared(root):
    freeze_run(root)
    stamp = read(Path(root) / "prepared.json")
    if stamp["run_sha256"] != digest(Path(root) / "run.json"):
        raise ValueError("Prepared provenance mismatch")
    for relative, expected in stamp["artifacts_sha256"].items():
        if digest(Path(root) / relative) != expected:
            raise ValueError(f"Prepared input changed: {relative}")
    return stamp


def load_states(args, records, manifest, folder):
    """Exact native FP32 causal extraction, with fresh full-document audits."""
    import gc
    import torch
    from transformers import AutoModel
    from ..core.trajectory_activation_cache import extract_cached
    path = folder / "states.npz"
    sample_ids = [r["sample_id"] for r in records]
    if path.exists():
        with np.load(path) as data:
            np.testing.assert_array_equal(data["sample_ids"], sample_ids)
            return data["states"].copy()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(
        manifest["snapshot"], dtype=torch.float32, device_map="cuda",
        attn_implementation="eager",
    ).eval().requires_grad_(False)
    with torch.inference_mode():
        values, checks = extract_cached(
            model, records, [manifest["input_layer_index"]], "cuda",
            manifest["snapshot"], args.output / "activation_cache",
            f"{args.large_stage} {args.shard}",
        )
    values = values[0]
    atomic_npz(path, states=values, sample_ids=sample_ids)
    atomic_json(folder / "extraction.json", checks)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return values
