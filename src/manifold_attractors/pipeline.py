"""Small, explicit DAG runner. No implicit use of old experiment results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if ROOT.name == "src":
    ROOT = ROOT.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_tasks(root: Path = ROOT) -> dict:
    tasks = {}
    for file in sorted((root / "configs").glob("*.json")):
        payload = json.loads(file.read_text())
        for task in payload.get("tasks", []):
            name = task["id"]
            if name in tasks:
                raise ValueError(f"Duplicate task: {name}")
            tasks[name] = task
    for name, task in tasks.items():
        for dep in task.get("deps", []):
            if dep not in tasks:
                raise ValueError(f"Unknown dependency {dep!r} of {name!r}")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError(f"Invalid task id: {name!r}")
        for output in task.get("outputs", []):
            p = Path(output)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Output must be repository-relative: {output}")
    return tasks


def order_tasks(tasks: dict, targets: list[str]) -> list[str]:
    done, active, order = set(), set(), []
    def visit(name):
        if name not in tasks:
            raise ValueError(f"Unknown task: {name}")
        if name in active:
            raise ValueError(f"Dependency cycle at {name}")
        if name in done:
            return
        active.add(name)
        for dep in tasks[name].get("deps", []):
            visit(dep)
        active.remove(name)
        done.add(name)
        order.append(name)
    for target in targets:
        visit(target)
    return order


def command(task: dict, style: str) -> list[str]:
    environment = task.get("environment")
    if environment:
        if environment not in {"render", "render-bubble"}:
            raise ValueError(f"Unknown isolated environment: {environment}")
        prefix = ["uv", "run", "--locked", "--isolated", "--no-default-groups",
                  "--extra", environment, "python"]
    else:
        prefix = [sys.executable]
    return [*prefix, "-m", task["module"], *[
        str(arg).replace("{style}", style) for arg in task.get("args", [])]]


def outputs(task: dict, style: str, root: Path = ROOT) -> list[Path]:
    return [root / value.replace("{style}", style) for value in task.get("outputs", [])]


def code_digest(root: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for directory in ("src", "configs", "assets", "styles")
                   for p in (root / directory).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()]
    h.update(sys.version.encode())
    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def fingerprint(task, style, digest, dep_records):
    return hashlib.sha256(json.dumps({"task": task, "style": style, "code": digest,
        "dependencies": dep_records}, sort_keys=True).encode()).hexdigest()


def run_tasks(tasks, names, style, resume=False, root=ROOT, allow_reconstructed_styles=True):
    blocked = [(name, tasks[name]["blocked"]) for name in names if tasks[name].get("blocked")]
    if blocked:
        raise RuntimeError("Cannot regenerate selected targets:\n" + "\n".join(f"  {n}: {why}" for n, why in blocked))
    scientific_rendering = any(tasks[name].get("environment") in {"render", "render-bubble"} for name in names)
    if scientific_rendering:
        import shutil
        reference = root / "configs/reference.json"
        caveats = json.loads(reference.read_text()).get("exactness_blockers", []) if reference.exists() else []
        if any(not item.get("resolved", False) for item in caveats) and not allow_reconstructed_styles:
            raise RuntimeError("Exact rendering is blocked: original TeX preambles are missing. See RENDER.md. "
                               "Recover them and mark the reference blocker resolved, or explicitly use "
                               "--allow-reconstructed-styles for an unverified reconstruction (not exact reproduction).")
        if shutil.which("lualatex") is None:
            raise RuntimeError("LuaLaTeX is required for scientific rendering; install the TeX/fonts listed in RENDER.md first.")
    env = os.environ.copy()
    env.update(FIGURE_STYLE=style, MPLBACKEND="Agg", PYTHONHASHSEED="0",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(root / "src"))
    digest = code_digest(root)
    records = {}
    for name in names:
        task = tasks[name]
        expected = outputs(task, style, root)
        profile = style if any("{style}" in value for value in task.get("outputs", [])) else "shared"
        record_dir = root / "runs" / profile
        record_dir.mkdir(parents=True, exist_ok=True)
        stamp = record_dir / f"{name}.json"
        key = fingerprint(task, profile, digest, {d: records[d] for d in task.get("deps", [])})
        if resume and task.get("cacheable", True) and stamp.exists():
            old = json.loads(stamp.read_text())
            if old.get("fingerprint") == key and expected and all(
                p.is_file() and old.get("outputs", {}).get(str(p.relative_to(root))) == sha256(p) for p in expected
            ):
                print(f"verified cache: {name}", flush=True)
                records[name] = old["outputs"]
                continue
        cmd = command(task, style)
        print(f"run: {name}", flush=True)
        start = time.time()
        with (record_dir / f"{name}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {record_dir / (name + '.log')}")
        missing = [str(p) for p in expected if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{name} did not produce nonempty outputs: {missing}")
        records[name] = {str(p.relative_to(root)): sha256(p) for p in expected}
        data = dict(task=name, command=cmd, fingerprint=key, outputs=records[name],
                    seconds=time.time() - start, python=sys.version, finished=time.time(),
                    allowed_reconstructed_styles=allow_reconstructed_styles)
        temporary = stamp.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(stamp)


def pixel_digest(image):
    """Canonical decoded pixels, independent of PNG metadata/compression."""
    rgba = image.convert("RGBA")
    return hashlib.sha256(f"{rgba.width}x{rgba.height}:RGBA:".encode() + rgba.tobytes()).hexdigest()


def check_reference(root=ROOT, style="iclr"):
    """Check available historical evidence; incomplete coverage never passes."""
    from PIL import Image
    manifest = json.loads((root / "configs" / "reference.json").read_text())
    problems, unverified, matched = [], [], 0
    for entry in manifest["figures"]:
        fish = entry.get("shared_pdf_image")
        if fish:
            from pypdf import PdfReader
            relative = f"fig/shared/{entry['name']}.pdf"
            path = root / relative
            if not path.exists():
                problems.append(f"missing: {relative}")
                continue
            reader = PdfReader(path)
            page = reader.pages[0]
            same = (len(reader.pages) == 1 and len(page.images) == 1
                    and [float(x) for x in page.mediabox] == fish["mediabox"]
                    and hashlib.sha256(page.get_contents().get_data()).hexdigest() == fish["content_sha256"]
                    and pixel_digest(page.images[0].image) == fish["pixel_sha256"])
            if not same:
                problems.append(f"DIFFERS (embedded pixels/page geometry): {relative}")
            else:
                matched += 1
            continue
        relative = f"fig/{style}/{entry['name']}.png"
        expected = entry.get("pixel_sha256", {}).get(style)
        if expected is None:
            unverified.append(f"UNVERIFIED (no canonical reference): {relative}")
            continue
        path = root / relative
        if not path.exists():
            problems.append(f"missing: {relative}")
        else:
            with Image.open(path) as image:
                same = pixel_digest(image) == expected
            if not same:
                problems.append(f"DIFFERS (pixels): {relative}")
            else:
                matched += 1
    for artwork in manifest.get("inline_artworks", []):
        unverified.append(f"UNVERIFIED: {artwork['name']}: {artwork['reason']}")
    for message in problems + unverified:
        print(message)
    print(f"Historical artwork checks matched: {matched}/{len(manifest['figures'])}; "
          f"unverified: {len(unverified)}. This is not a from-scratch execution certificate.")
    return 1 if problems else 2 if unverified else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "plan", "run", "check"])
    parser.add_argument("targets", nargs="*", help="Task ids or 'data'/'figures'; default: all published figures")
    parser.add_argument("--style", choices=["iclr", "goodfire"], default="iclr")
    parser.add_argument("--resume", action="store_true", help="Reuse only fingerprinted, hash-verified task outputs")
    typography = parser.add_mutually_exclusive_group()
    typography.add_argument("--allow-reconstructed-styles", action="store_true", default=True,
                            help="Allow documented reconstructed typography (default); LuaLaTeX is still required")
    typography.add_argument("--require-original-styles", action="store_false", dest="allow_reconstructed_styles",
                            help="Strict provenance mode: require the unavailable original TeX preambles")
    args = parser.parse_args(argv)
    try:
        if args.action == "check":
            return check_reference(style=args.style)
        tasks = load_tasks()
        if not tasks:
            raise ValueError("No task configurations found; run from an editable source installation")
        if args.action == "list":
            for name, task in tasks.items():
                print(name + (" [BLOCKED: " + task["blocked"] + "]" if task.get("blocked") else ""))
            return 0
        targets = []
        for target in args.targets or ["figures"]:
            if target in {"data", "figures"}:
                prefix = "data-" if target == "data" else "render-"
                targets.extend(name for name in tasks if name.startswith(prefix))
            else:
                targets.append(target)
        names = order_tasks(tasks, targets)
        if args.action == "plan":
            import shlex
            for name in names:
                print(f"# {name}" + (f" BLOCKED: {tasks[name]['blocked']}" if tasks[name].get("blocked") else ""))
                print(shlex.join(command(tasks[name], args.style)))
            return 0
        run_tasks(tasks, names, args.style, args.resume,
                  allow_reconstructed_styles=args.allow_reconstructed_styles)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
