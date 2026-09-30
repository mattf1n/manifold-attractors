"""No workstation checkout or bundled experimental results required."""
import ast
from pathlib import Path

from manifold_attractors.pipeline import ROOT


def test_no_precomputed_experiments_bundled():
    forbidden = {".npz", ".npy", ".parquet", ".pt", ".safetensors", ".pdf", ".png"}
    for directory in ("src", "assets", "configs"):
        for path in (ROOT / directory).rglob("*"):
            assert not path.is_symlink(), path
            assert path.suffix not in forbidden, path


def test_no_workstation_dependencies():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Documentation may name source provenance; actual absolute input
                # paths are forbidden in executable string expressions.
                if "\n" not in node.value:
                    assert not node.value.startswith(("/mnt/home/", "/mnt/data/", "../paper-data", "~/project")), (path, node.value)
