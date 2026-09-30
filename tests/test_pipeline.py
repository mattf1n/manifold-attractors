import json
from pathlib import Path
import sys

import pytest

from manifold_attractors.pipeline import (
    ROOT, load_tasks, order_tasks, command, outputs, run_tasks,
)


def test_dependency_order():
    tasks = {"a": {}, "b": {"deps": ["a"]}, "c": {"deps": ["a", "b"]}}
    assert order_tasks(tasks, ["c", "a"]) == ["a", "b", "c"]


def test_render_environments_are_locked():
    task = {"module": "test.module", "environment": "render-bubble"}
    cmd = command(task, "iclr")
    assert cmd[:7] == ["uv", "run", "--locked", "--isolated", "--no-default-groups", "--extra", "render-bubble"]


def test_bad_dependency():
    with pytest.raises(ValueError, match="Unknown task"):
        order_tasks({"a": {"deps": ["missing"]}}, ["a"])
    with pytest.raises(ValueError, match="cycle"):
        order_tasks({"a": {"deps": ["b"]}, "b": {"deps": ["a"]}}, ["a"])


def test_style_expansion():
    task = {"module": "test.module", "args": ["--figure", "fig/{style}/a.pdf"],
            "outputs": ["fig/{style}/a.pdf"]}
    assert command(task, "goodfire") == [sys.executable, "-m", "test.module", "--figure", "fig/goodfire/a.pdf"]
    assert outputs(task, "iclr", Path("/tmp")) == [Path("/tmp/fig/iclr/a.pdf")]


def test_reject_escaping_outputs(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/test.json").write_text(json.dumps({"tasks": [{"id": "a", "outputs": ["../bad"]}]}))
    with pytest.raises(ValueError, match="repository-relative"):
        load_tasks(tmp_path)


def test_blocked_run_fails_before_work(tmp_path):
    tasks = {"a": {"module": "nonexistent"}, "b": {"blocked": "missing protocol", "deps": ["a"]}}
    with pytest.raises(RuntimeError, match="missing protocol"):
        run_tasks(tasks, ["a", "b"], "iclr", root=tmp_path)
    assert not (tmp_path / "runs").exists()


def test_resume_checks_source_and_output_hashes(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    tasks = {"a": {"module": "fake", "outputs": ["data/a.txt"]}}
    (tmp_path / "data").mkdir()
    (tmp_path / "assets").mkdir()
    asset = tmp_path / "assets/input.txt"
    asset.write_text("input")
    calls = []
    def fake_run(*args, **kwargs):
        calls.append(args)
        (tmp_path / "data/a.txt").write_text("generated")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    run_tasks(tasks, ["a"], "iclr", resume=True, root=tmp_path)
    run_tasks(tasks, ["a"], "iclr", resume=True, root=tmp_path)
    run_tasks(tasks, ["a"], "goodfire", resume=True, root=tmp_path)
    assert len(calls) == 1  # numerical data are shared by both rendering profiles
    (tmp_path / "data/a.txt").write_text("corrupted")
    run_tasks(tasks, ["a"], "iclr", resume=True, root=tmp_path)
    assert len(calls) == 2
    asset.write_text("changed input")
    run_tasks(tasks, ["a"], "iclr", resume=True, root=tmp_path)
    assert len(calls) == 3


def test_ready_stage_verifies_dependencies_without_rerunning(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from manifold_attractors.pipeline import run_ready_stage, sha256
    (tmp_path/'data').mkdir(); (tmp_path/'runs/shared').mkdir(parents=True)
    source = tmp_path/'data/input.txt'; source.write_text('fresh upstream result')
    (tmp_path/'runs/shared/upstream.json').write_text(json.dumps({'outputs': {'data/input.txt': sha256(source)}}))
    tasks = {'upstream': {'module': 'never.run', 'outputs': ['data/input.txt']},
             'new': {'module': 'new.stage', 'deps': ['upstream'], 'outputs': ['data/result.txt']}}
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd); (tmp_path/'data/result.txt').write_text('scored')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    run_ready_stage(tasks, 'new', root=tmp_path)
    assert len(calls) == 1 and 'new.stage' in calls[0]
    source.write_text('tampered')
    with pytest.raises(RuntimeError, match='changed/missing'):
        run_ready_stage(tasks, 'new', root=tmp_path)
    assert len(calls) == 1


def test_manifest_is_complete():
    tasks = load_tasks()
    assert tasks
    assert order_tasks(tasks, list(tasks))
    reference = json.loads((ROOT / "configs/reference.json").read_text())
    for entry in reference["figures"]:
        assert "render-" + entry["name"] in tasks, entry["name"]
    assert "render-two_basin_field" in tasks
    for name, task in tasks.items():
        if name.startswith("render-") and name not in {
            "render-dead_fish_sequence", "render-two_basin_field", "render-sphere_probe"
        }:
            assert task.get("deps"), f"Missing generation dependency: {name}"


def test_module_files_exist():
    for name, task in load_tasks().items():
        module = task["module"]
        file = ROOT / "src" / Path(*module.split("."))
        assert file.with_suffix(".py").exists() or (file / "__main__.py").exists(), (name, module)
