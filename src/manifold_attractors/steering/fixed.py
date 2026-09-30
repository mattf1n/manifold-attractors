"""Frozen confirmatory fixed-baseline calibration and distribution evaluation.

No source selection is performed here. Inference requires an explicit approval
JSON, supplied by the experiment coordinator. Grid expansion uses calibration
only; test requires a current, entirely interior selection. All outputs stay
under --root. Geometry inputs contain F4 policy points, never tuned fallbacks.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from .common import atomic_json, atomic_npz
from .vocabulary import (
    COARSE_FROM_SUBTYPE,
    HAN_PREFIX_METRIC,
    SUBTYPES,
    Probe,
    aggregate_subtypes,
    audit_distribution,
    checked_distribution,
    load_partition,
)
from .common import load_states

ROOT = Path("runs/adaptive_steering")
BASELINES = ("additive", "norm_preserving", "constant_projection")
TRACKS = ("mass", "greedy")
SPLITS = ("wiki_cal", "wiki_test", "fineweb_test")
STEP = 0.05


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(root):
    meta = read(root / "direction/llama3.json")
    assert meta["hidden_state_index"] == 15
    with np.load(root / "direction/llama3.npz") as data:
        unit = -data["direction"].astype(np.float64)
        raw = -data["raw_direction"].astype(np.float64)
    assert unit.shape == raw.shape == (4096,)
    unit /= np.linalg.norm(unit)
    assert np.isfinite(unit).all() and np.isfinite(raw).all()
    return meta, unit, raw


def manifest_path(args):
    split = "wiki_cal" if args.stage in ("calibrate", "select") else args.split
    path = args.root / f"manifests/{split}.json"
    if args.manifest is not None:
        assert args.manifest.resolve() == path.resolve(), (
            "Only canonical frozen manifests allowed"
        )
    return split, path


def approval(args):
    """Read-only gate; only the coordinator may create this approval file."""
    path = args.root / "inference_approval.json"
    if args.approval is not None:
        assert args.approval.resolve() == path.resolve(), (
            "Approval must be root/inference_approval.json"
        )
    value = read(path)
    for flag in (
        "approved",
        "split_audit_passed",
        "protocol_frozen",
        "direction_provenance_approved",
    ):
        assert value.get(flag) is True, f"Inference blocked: {flag}"
    hashes = value["manifest_sha256"]
    assert {"wiki_cal", "wiki_test", "fineweb_test"} <= set(hashes) <= set(SPLITS)
    split, manifest = manifest_path(args)
    assert digest(manifest) == hashes[split], "Frozen manifest hash mismatch"
    return digest(path)


def grid_definition(root):
    path = root / "calibration/grid_definition.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            path,
            {
                "version": 0,
                "step": STEP,
                "bounds": {
                    "additive": [0, 8],
                    "norm_preserving": [0, 8],
                    "constant_projection": [-4, 8],
                },
                "expansion_span": {
                    "additive": 8,
                    "norm_preserving": 8,
                    "constant_projection": 12,
                },
                "endpoint_rule": "Expand any selected endpoint in its direction; calibration only",
                "tracks": list(TRACKS),
                "metric": HAN_PREFIX_METRIC,
            },
        )
    return read(path)


def grids(definition):
    return {
        name: np.arange(round(lo / STEP), round(hi / STEP) + 1) * STEP
        for name, (lo, hi) in definition["bounds"].items()
    }


def state_file(args):
    return args.root / f"states/{args.split}/{args.shard:03d}.npz"


def states_stage(args):
    stamp = approval(args)
    meta, _, _ = source(args.root)
    split, path = manifest_path(args)
    manifest = read(path)
    assert manifest["split"] == split and manifest["input_layer_index"] == 15
    records = manifest["records"]
    # Support a per-split manifest or a combined manifest, preserving frozen order.
    if any("split" in row for row in records):
        records = [row for row in records if row.get("split") == args.split]
    assert [row["sample_id"] for row in records] == list(range(len(records)))
    if args.split == "wiki_cal":
        assert len(records) == 1000
    assert 0 <= args.shard * 500 < len(records)
    records = records[args.shard * 500 : (args.shard + 1) * 500]
    assert manifest.get("snapshot", meta["snapshot"]) == meta["snapshot"]
    adapter = dict(manifest, snapshot=meta["snapshot"], input_layer_index=15)
    folder = args.root / f"states/{args.split}/extraction/{args.shard:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    request = {
        "manifest_sha256": digest(path),
        "direction_sha256": digest(args.root / "direction/llama3.npz"),
    }
    request_path = folder / "request.json"
    if request_path.exists():
        assert read(request_path) == request, "Extraction cache provenance mismatch"
    else:
        assert not (folder / "states.npz").exists(), "Unprovenanced extraction cache"
        atomic_json(request_path, request)
    native = load_states(
        SimpleNamespace(output=args.root, large_stage=args.split, shard=args.shard),
        records,
        adapter,
        folder,
    )
    assert native.dtype == np.float32 and native.shape == (len(records), 4096)
    atomic_npz(
        state_file(args),
        states=native,
        sample_ids=np.array([r["sample_id"] for r in records]),
    )
    atomic_json(
        state_file(args).with_suffix(".json"),
        {
            "approval_sha256": stamp,
            "manifest": str(path),
            "manifest_sha256": digest(path),
            "native_dtype": "float32",
            "hidden_state_index": 15,
            "count": len(records),
        },
    )


def get_states(args):
    approval(args)
    _, manifest = manifest_path(args)
    provenance = read(state_file(args).with_suffix(".json"))
    assert provenance["manifest_sha256"] == digest(manifest)
    with np.load(state_file(args)) as data:
        states, ids = data["states"].copy(), data["sample_ids"].copy()
    assert states.dtype == np.float32 and states.shape == (len(ids), 4096)
    np.testing.assert_array_equal(
        ids, np.arange(args.shard * 500, args.shard * 500 + len(ids))
    )
    assert np.isfinite(states).all()
    return states.astype(np.float64), ids


def point(state, unit, method, parameter):
    norm = np.linalg.norm(state)
    if method == "constant_projection":
        strength = parameter - state @ unit
    else:
        strength = parameter
    result = state + strength * unit
    if method == "norm_preserving":
        length = np.linalg.norm(result)
        assert length > 0
        result *= norm / length
    if method == "constant_projection":
        assert abs(result @ unit - parameter) < 1e-12
    return result


def setup(args):
    stamp = approval(args)
    meta, unit, raw = source(args.root)
    probe = Probe(meta["snapshot"], 15)
    labels, matrix = load_partition(
        SimpleNamespace(output=args.root / "vocabulary"), probe, subtypes=True
    )
    assert len(labels) == 128256
    with np.load(args.root / "vocabulary/vocabulary.npz") as data:
        np.testing.assert_array_equal(COARSE_FROM_SUBTYPE[labels], data["categories"])
        assert len(data["special_ids"]) == 256
        assert np.all(data["categories"][data["special_ids"]] == 4)
    return stamp, unit, raw, probe, labels, matrix


def evaluate(probe, points, matrix, batch_size, audit=True):
    values, tokens, checks = [], [], []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        mass, ids = checked_distribution(probe, batch, matrix)
        if audit and start == 0:
            checks.append(audit_distribution(probe, batch[0], matrix, mass[0], ids[0]))
        values.append(mass)
        tokens.append(ids)
    return np.concatenate(values), np.concatenate(tokens), checks


def calibrate_stage(args):
    assert args.shard in (0, 1)
    args.split = "wiki_cal"
    states, ids = get_states(args)
    assert len(ids) == 500
    definition = grid_definition(args.root)
    stamp, unit, _, probe, labels, matrix = setup(args)
    for name, grid in grids(definition).items():
        path = (
            args.root
            / f"calibration/v{definition['version']:03d}/{args.shard:03d}/{name}.npz"
        )
        if path.exists() and path.with_suffix(".json").exists():
            previous = read(path.with_suffix(".json"))
            assert previous["states_sha256"] == digest(state_file(args))
            assert previous["approval_sha256"] == stamp
            with np.load(path) as data:
                np.testing.assert_array_equal(data["sample_ids"], ids)
                np.testing.assert_array_equal(data["parameters"], grid)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        masses = np.empty((len(ids), len(grid), len(SUBTYPES)), dtype=np.float64)
        tokens = np.empty((len(ids), len(grid)), dtype=np.int64)
        strengths = np.empty((len(ids), len(grid)), dtype=np.float64)
        norms = np.empty_like(strengths)
        projections = np.empty_like(strengths)
        audits = []
        # Fill the already validated 128-point GPU batches and checkpoint every
        # eight documents rather than losing an entire baseline to preemption.
        chunks = path.parent / f"{name}_chunks"
        chunks.mkdir(exist_ok=True)
        for start in range(0, len(states), 8):
            stop = min(start + 8, len(states))
            chunk = chunks / f"{start:04d}.npz"
            if chunk.exists() and chunk.with_suffix(".json").exists():
                info = read(chunk.with_suffix(".json"))
                assert info["approval_sha256"] == stamp
                assert info["states_sha256"] == digest(state_file(args))
                with np.load(chunk) as cached:
                    np.testing.assert_array_equal(cached["sample_ids"], ids[start:stop])
                    np.testing.assert_array_equal(cached["parameters"], grid)
                    masses[start:stop] = cached["subtype_masses"]
                    tokens[start:stop] = cached["token_ids"]
                    strengths[start:stop] = cached["effective_strengths"]
                    norms[start:stop] = cached["norms_after"]
                    projections[start:stop] = cached["projections_after"]
                audits.extend(info["audits"])
                continue
            points = np.array(
                [
                    [point(state, unit, name, parameter) for parameter in grid]
                    for state in states[start:stop]
                ]
            )
            values, predicted, checks = evaluate(
                probe,
                points.reshape(-1, states.shape[1]),
                matrix,
                args.batch_size,
                audit=start in (0, 248, 496),
            )
            masses[start:stop] = values.reshape(stop - start, len(grid), len(SUBTYPES))
            tokens[start:stop] = predicted.reshape(stop - start, len(grid))
            strengths[start:stop] = (points - states[start:stop, None]) @ unit
            norms[start:stop] = np.linalg.norm(points, axis=-1)
            projections[start:stop] = points @ unit
            if name == "constant_projection":
                assert np.max(np.abs(projections[start:stop] - grid[None])) < 1e-12
            atomic_npz(
                chunk,
                sample_ids=ids[start:stop],
                parameters=grid,
                subtype_masses=masses[start:stop],
                token_ids=tokens[start:stop],
                effective_strengths=strengths[start:stop],
                norms_after=norms[start:stop],
                projections_after=projections[start:stop],
            )
            atomic_json(
                chunk.with_suffix(".json"),
                {
                    "approval_sha256": stamp,
                    "states_sha256": digest(state_file(args)),
                    "audits": checks,
                },
            )
            audits.extend(checks)
            print("calibration", args.shard, name, stop, len(states), flush=True)
        atomic_npz(
            path,
            sample_ids=ids,
            parameters=grid,
            subtype_masses=masses,
            effective_strengths=strengths,
            norms_after=norms,
            norms_before=np.linalg.norm(states, axis=1),
            projections_before=states @ unit,
            projections_after=projections,
            class_masses=aggregate_subtypes(masses),
            token_ids=tokens,
            argmax_subtypes=labels[tokens],
            argmax_classes=COARSE_FROM_SUBTYPE[labels[tokens]],
        )
        atomic_json(
            path.with_suffix(".json"),
            {
                "audits": audits,
                "approval_sha256": stamp,
                "grid_version": definition["version"],
                "states_sha256": digest(state_file(args)),
            },
        )


def choose(parameters, masses, categories, track):
    if track == "mass":
        bad = masses[..., [0, 3]].sum(-1).mean(0)
        chinese = masses[..., 2].mean(0)
    else:
        bad = np.isin(categories, [0, 3]).mean(0)
        chinese = (categories == 2).mean(0)
    index = int(np.lexsort((parameters, -chinese, bad))[0])
    return index, {
        "parameter": float(parameters[index]),
        "bad": float(bad[index]),
        "chinese": float(chinese[index]),
    }


def select_stage(args):
    stamp = approval(args)
    definition = grid_definition(args.root)
    selections = {track: {} for track in TRACKS}
    expansions, inputs = {}, {}
    for name, grid in grids(definition).items():
        masses, categories, ids = [], [], []
        for shard in (0, 1):
            path = (
                args.root
                / f"calibration/v{definition['version']:03d}/{shard:03d}/{name}.npz"
            )
            inputs[str(path.relative_to(args.root))] = digest(path)
            with np.load(path) as data:
                np.testing.assert_array_equal(data["parameters"], grid)
                masses.append(data["class_masses"])
                categories.append(data["argmax_classes"])
                ids.append(data["sample_ids"])
        np.testing.assert_array_equal(np.concatenate(ids), np.arange(1000))
        for track in TRACKS:
            index, selection = choose(
                grid, np.concatenate(masses), np.concatenate(categories), track
            )
            selections[track][name] = selection
            if index in (0, len(grid) - 1):
                expansions.setdefault(name, set()).add(0 if index == 0 else 1)
    if expansions:
        atomic_json(
            args.root / f"calibration/expansion_v{definition['version']:03d}.json",
            {
                "selections": selections,
                "endpoints": {k: sorted(v) for k, v in expansions.items()},
            },
        )
        for name, ends in expansions.items():
            for end in ends:
                definition["bounds"][name][end] += (-1 if end == 0 else 1) * definition[
                    "expansion_span"
                ][name]
        definition["version"] += 1
        atomic_json(args.root / "calibration/grid_definition.json", definition)
        raise SystemExit(
            "CALIBRATION EXPANDED: rerun both calibration shards, then select; test blocked"
        )
    atomic_json(
        args.root / "calibration/selection.json",
        {
            "grid_version": definition["version"],
            "all_interior": True,
            "tracks": selections,
            "inputs_sha256": inputs,
            "approval_sha256": stamp,
            "grid_definition_sha256": digest(
                args.root / "calibration/grid_definition.json"
            ),
        },
    )


def geometry_points(folder, ids, states, unit, additive):
    """Construct F4 points via the geometry owner's shared implementation.

    Invalid primary selections remain explicit missing outcomes, never fallbacks.
    Quarter-step selections are independent diagnostic points only.
    """
    from .geometry import point_constructor

    result = {}
    for method in ("angular", "straight"):
        points = np.full_like(states, np.nan)
        quarter = np.full_like(states, np.nan)
        modes, flags, fingerprints, hashes = [], [], [], []
        valid = np.zeros(len(ids), dtype=bool)
        quarter_valid = np.zeros(len(ids), dtype=bool)
        for i, sample_id in enumerate(ids):
            path = folder / f"{sample_id:05d}.npz"
            with np.load(path) as data:
                assert int(data["sample_id"]) == sample_id
                prefix = method + "_"
                mode = str(data[prefix + "mode"])
                valid[i] = bool(data[prefix + "valid"])
                modes.append(mode)
                flags.append(json.dumps(data[prefix + "flags"].tolist()))
                fingerprints.append(str(data["protocol_sha256"]))
                hashes.append(digest(path))
                if valid[i]:
                    assert mode in ("right_minimum", "no_steer", "fixed")
                    points[i] = point_constructor(
                        states[i],
                        unit,
                        method,
                        mode,
                        scalar=float(data[prefix + "scalar"]),
                        fixed_strength=additive,
                    )
                    assert np.isfinite(points[i]).all()
                if (prefix + "quarter_mode") in data and bool(
                    data.get(prefix + "quarter_valid", True)
                ):
                    qmode = str(data[prefix + "quarter_mode"])
                    if qmode in ("right_minimum", "no_steer", "fixed"):
                        quarter[i] = point_constructor(
                            states[i],
                            unit,
                            method,
                            qmode,
                            scalar=float(data[prefix + "quarter_scalar"]),
                            fixed_strength=additive,
                        )
                        quarter_valid[i] = np.isfinite(quarter[i]).all()
        result[method] = (
            points,
            np.array(modes),
            valid,
            {
                "geometry_flags": np.array(flags),
                "geometry_protocol_sha256": np.array(fingerprints),
                "geometry_file_sha256": np.array(hashes),
                "quarter_points": quarter,
                "quarter_valid": quarter_valid,
            },
        )
    return result


def test_stage(args):
    assert args.split in ("wiki_test", "fineweb_test")
    selected = read(args.root / "calibration/selection.json")
    assert selected["all_interior"] is True
    assert selected["grid_definition_sha256"] == digest(
        args.root / "calibration/grid_definition.json"
    )
    assert selected["grid_version"] == grid_definition(args.root)["version"]
    for rel, expected in selected["inputs_sha256"].items():
        assert digest(args.root / rel) == expected
    states, ids = get_states(args)
    stamp, unit, raw, probe, labels, matrix = setup(args)
    folder = args.root / f"test/{args.split}/{args.shard:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    selection_hash = digest(args.root / "calibration/selection.json")
    cache_path = folder / "point_cache.npz"
    cache = {}
    if cache_path.exists():
        with np.load(cache_path) as data:
            assert str(data["selection_sha256"]) == selection_hash
            for key, mass, token in zip(
                data["keys"], data["subtype_masses"], data["token_ids"]
            ):
                cache[str(key)] = (mass.copy(), int(token))
    methods = {}
    for track in TRACKS:
        choices = selected["tracks"][track]
        for name in BASELINES:
            parameter = choices[name]["parameter"]
            methods[f"{track}/{name}"] = (
                np.stack([point(s, unit, name, parameter) for s in states]),
                np.full(len(ids), "fixed"),
            )
        methods[f"{track}/raw_centroid"] = (states + raw, np.full(len(ids), "untuned"))
        if args.adaptive is not None:
            geometry_root = (
                args.root / "geometry" if args.adaptive == Path(".") else args.adaptive
            )
            for name, values in geometry_points(
                geometry_root / args.split,
                ids,
                states,
                unit,
                choices["additive"]["parameter"],
            ).items():
                methods[f"{track}/adaptive_{name}"] = values

    def decode(points, valid):
        masses = np.full((len(ids), len(SUBTYPES)), np.nan)
        tokens = np.full(len(ids), -1, dtype=np.int64)
        indices = np.flatnonzero(valid)
        if not len(indices):
            return masses, tokens, []
        keys = {i: hashlib.sha256(points[i].tobytes()).hexdigest() for i in indices}
        missing = {key: i for i, key in keys.items() if key not in cache}
        if missing:
            values, argmax, audits = evaluate(
                probe,
                np.stack([points[i] for i in missing.values()]),
                matrix,
                args.batch_size,
            )
            cache.update(
                {
                    key: (mass, int(token))
                    for key, mass, token in zip(missing, values, argmax)
                }
            )
        else:
            first = indices[0]
            audits = [
                audit_distribution(probe, points[first], matrix, *cache[keys[first]])
            ]
        for i, key in keys.items():
            masses[i], tokens[i] = cache[key]
        return masses, tokens, audits

    for name, values in methods.items():
        points, modes = values[:2]
        valid, extras = (
            values[2:] if len(values) == 4 else (np.ones(len(ids), dtype=bool), {})
        )
        masses, tokens, audits = decode(points, valid)
        if "quarter_points" in extras:
            quarter = extras.pop("quarter_points")
            qmasses, qtokens, qaudits = decode(quarter, extras["quarter_valid"])
            comparable = valid & extras["quarter_valid"]
            extras.update(
                quarter_subtype_masses=qmasses,
                quarter_class_masses=aggregate_subtypes(qmasses),
                quarter_token_ids=qtokens,
                quarter_token_comparable=comparable,
                quarter_token_agreement=comparable & (tokens == qtokens),
            )
            audits.extend(qaudits)
        subtypes = np.full(len(ids), -1, dtype=np.int8)
        classes = np.full(len(ids), -1, dtype=np.int8)
        subtypes[valid] = labels[tokens[valid]]
        classes[valid] = COARSE_FROM_SUBTYPE[subtypes[valid]]
        delta = points - states
        strengths = delta @ unit
        before, after = states @ unit, points @ unit
        if name.endswith("/constant_projection"):
            target = selected["tracks"][name.split("/")[0]]["constant_projection"][
                "parameter"
            ]
            assert np.max(np.abs(after - target)) < 1e-12
        path = folder / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(
            path,
            sample_ids=ids,
            subtype_masses=masses,
            class_masses=aggregate_subtypes(masses),
            token_ids=tokens,
            argmax_subtypes=subtypes,
            argmax_classes=classes,
            valid=valid,
            modes=modes,
            effective_strengths=strengths,
            displacement_norms=np.linalg.norm(delta, axis=1),
            norms_before=np.linalg.norm(states, axis=1),
            norms_after=np.linalg.norm(points, axis=1),
            projections_before=before,
            projections_after=after,
            **extras,
        )
        atomic_json(
            path.with_suffix(".json"),
            {
                "audits": audits,
                "approval_sha256": stamp,
                "selection_sha256": selection_hash,
                "temperature": 1,
                "precision": "float64",
                "invalid_primary_count": int((~valid).sum()),
                "geometry_directory": str(geometry_root / args.split)
                if args.adaptive is not None
                else None,
            },
        )
        atomic_npz(
            cache_path,
            keys=np.array(list(cache)),
            subtype_masses=np.stack([v[0] for v in cache.values()]),
            token_ids=np.array([v[1] for v in cache.values()]),
            selection_sha256=np.array(selection_hash),
        )


def dev_self_test(root):
    """CPU-only fake-probe filesystem integration; no model or source outcomes."""
    import tempfile
    from unittest.mock import patch

    root = root / "development"
    root.mkdir(parents=True, exist_ok=True)
    import sys
    module = sys.modules[__name__]

    class FakeProbe:
        def probability_mass(self, points, matrix):
            mass = np.zeros((len(points), len(SUBTYPES)))
            mass[:, 0], mass[:, 2] = 0.25, 0.75
            return mass, np.full(len(points), 2, dtype=np.int64)

    with tempfile.TemporaryDirectory(prefix="fixed-fake-", dir=root) as temporary:
        work = Path(temporary)
        args = SimpleNamespace(
            root=work,
            stage="states",
            approval=None,
            manifest=None,
            split="wiki_cal",
            shard=0,
            adaptive=None,
            batch_size=8,
        )
        try:
            approval(args)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing approval did not block")
        (work / "manifests").mkdir()
        for split in SPLITS:
            atomic_json(
                work / f"manifests/{split}.json", {"split": split, "records": []}
            )
        approved = {
            key: True
            for key in (
                "approved",
                "split_audit_passed",
                "protocol_frozen",
                "direction_provenance_approved",
            )
        }
        approved["manifest_sha256"] = {
            s: digest(work / f"manifests/{s}.json") for s in SPLITS
        }
        atomic_json(work / "inference_approval.json", approved)
        stamp = approval(args)
        unit, raw = np.zeros(4096), np.zeros(4096)
        unit[0], raw[0] = 1.0, 2.0
        probe = FakeProbe()
        labels = np.arange(len(SUBTYPES), dtype=np.int8)
        matrix = np.eye(len(SUBTYPES))

        def fake_setup(unused):
            return stamp, unit, raw, probe, labels, matrix

        for split, shard, count in [
            ("wiki_cal", 0, 500),
            ("wiki_cal", 1, 500),
            ("wiki_test", 0, 2),
        ]:
            args.split, args.shard = split, shard
            path = state_file(args)
            path.parent.mkdir(parents=True, exist_ok=True)
            native = np.zeros((count, 4096), dtype=np.float32)
            native[:, 1] = 3
            atomic_npz(
                path,
                states=native,
                sample_ids=np.arange(shard * 500, shard * 500 + count),
            )
            atomic_json(
                path.with_suffix(".json"),
                {"manifest_sha256": approved["manifest_sha256"][split]},
            )
        tiny = {name: np.array([-1.0, 0.0, 1.0]) for name in BASELINES}
        fake_audit = lambda *unused: {"fake_probe_only": True}
        with (
            patch.object(module, "setup", fake_setup),
            patch.object(module, "grids", lambda unused: tiny),
            patch.object(module, "audit_distribution", fake_audit),
        ):
            args.stage, args.split = "calibrate", "wiki_cal"
            for shard in (0, 1):
                args.shard = shard
                calibrate_stage(args)
            # Isolate filesystem assembly from objective math (tested below).
            with patch.object(
                module,
                "choose",
                lambda *unused: (1, {"parameter": 0.0, "bad": 0.25, "chinese": 0.75}),
            ):
                args.stage = "select"
                select_stage(args)
            args.stage, args.split, args.shard = "test", "wiki_test", 0
            test_stage(args)
            geometry = work / "geometry/wiki_test"
            geometry.mkdir(parents=True)
            for sample_id in (0, 1):
                payload = {"sample_id": sample_id, "protocol_sha256": "fake-only"}
                for method in ("angular", "straight"):
                    payload.update(
                        {
                            method + "_mode": "fixed",
                            method + "_scalar": np.nan,
                            method + "_valid": sample_id == 0,
                            method + "_flags": np.array(["dev_fake"]),
                            method + "_quarter_mode": "fixed",
                            method + "_quarter_scalar": np.nan,
                            method + "_quarter_valid": True,
                        }
                    )
                atomic_npz(geometry / f"{sample_id:05d}.npz", **payload)
            args.adaptive = Path(".")
            test_stage(args)
            for track in TRACKS:
                for method in (
                    *BASELINES,
                    "raw_centroid",
                    "adaptive_angular",
                    "adaptive_straight",
                ):
                    with np.load(
                        work / f"test/wiki_test/000/{track}/{method}.npz"
                    ) as data:
                        assert data["class_masses"].shape == (2, 5)
                        assert data["subtype_masses"].shape == (2, 11)
                        if method.startswith("adaptive"):
                            assert data["valid"].tolist() == [True, False]
                            assert data["token_ids"].tolist() == [2, -1]
                            assert np.isnan(data["class_masses"][1]).all()
                            assert data["quarter_token_agreement"].tolist() == [
                                True,
                                False,
                            ]
        masses = np.array(
            [[[0.2, 0, 0.7, 0, 0.1], [0.1, 0, 0.8, 0, 0.1], [0.1, 0, 0.7, 0, 0.2]]]
        )
        cats = np.array([[0, 2, 2]])
        for track in TRACKS:
            assert choose(np.array([0.0, 1.0, 2.0]), masses, cats, track)[0] == 1
        s = np.zeros(4096)
        s[1] = 3
        for q in (-4.0, 0.0, 8.0):
            assert abs(point(s, unit, "constant_projection", q) @ unit - q) < 1e-12
        # Tampering must block even if the approval booleans remain true.
        atomic_json(work / "manifests/wiki_test.json", {"tampered": True})
        try:
            approval(args)
        except AssertionError:
            pass
        else:
            raise AssertionError("Manifest tampering did not block")
    print(
        "PASS: CPU fake-probe filesystem/selection/invalid/quarter/cache/approval checks; no inference"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["states", "calibrate", "select", "test", "dev-self-test"]
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=SPLITS, default="wiki_cal")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--adaptive",
        type=Path,
        nargs="?",
        const=Path("."),
        help="Add F4 methods from geometry directory (default root/geometry)",
    )
    args = parser.parse_args()
    assert args.batch_size > 0 and args.shard >= 0
    if args.stage == "dev-self-test":
        dev_self_test(args.root)
        return
    from .common import require_prepared
    require_prepared(args.root)
    approval(args)  # Fail before any output write, cache read, or model load.
    globals()[args.stage + "_stage"](args)


if __name__ == "__main__":
    main()
