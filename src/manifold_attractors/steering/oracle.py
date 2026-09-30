"""Output-aware finite-search additive-family oracles on frozen test states."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from .fixed import point, source
from .vocabulary import (
    BAD,
    CHINESE,
    CLASSES,
    SUBTYPES,
    Probe,
    aggregate_subtypes,
    atomic_json,
    atomic_npz,
    audit_distribution,
    bootstrap_means,
    checked_distribution,
    load_partition,
)

BASE = Path("runs/adaptive_steering")
OUT = BASE / "linear_oracle"
DATASETS = ("wiki_test", "fineweb_test")
EXTRAS = (
    "mass/additive",
    "greedy/additive",
    "mass/constant_projection",
    "greedy/constant_projection",
    "mass/adaptive_straight",
    "greedy/adaptive_straight",
    "mass/raw_centroid",
)
GRID = np.r_[1e-6, np.arange(1, 400) / 40, 10 - 1e-6]
COARSE = np.r_[True, np.arange(1, 400) % 2 == 0, True, np.ones(len(EXTRAS), dtype=bool)]


def read(path):
    return json.loads(path.read_text())


def prepare():
    OUT.mkdir(exist_ok=True)
    definition = {
        "family": "x(s)=x+s*unit(zh-en), no norm restoration, strict0<s<10",
        "status": "Output-aware expensive finite-search oracle, NOT calibrated baseline or certified continuous bound",
        "datasets": DATASETS,
        "n_each": 10000,
        "coarse_step": 0.05,
        "fine_step": 0.025,
        "regular_grid": "integer multiples strictly inside(0,10)",
        "boundary_adjacent_probes": [1e-6, 10 - 1e-6],
        "exact_baseline_candidates": EXTRAS,
        "excluded": "zero/no-steer or any comparator strength outside(0,10)",
        "mass_selection": "minimumPbad, then maximumPChinese, then smallerstrength",
        "greedy_selection": "minimumargmaxbad, then Chinese output, then smallerstrength",
        "separate_chinese_ceiling": "maximumPChinese, then minimumPbad, then smallerstrength; a separate output row",
        "global_audit": "Every datasetlocal index divisible by100: add all odd multiples of.0125; reuse allfine/base candidates",
        "local_refinement": False,
        "temperature": 1,
        "full_logits_saved": False,
        "saved": "allcandidate5/11masses, strengths, argmax, masks, selected indices and gridconvergence",
        "scope": "Only additivefamily; angular/norm-preserving outsidefamily; no certified global optimality",
    }
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == json.loads(json.dumps(definition))
    else:
        atomic_json(path, definition)
    atomic_npz(
        OUT / "candidate_grid.npz",
        strengths=GRID,
        coarse_mask=COARSE,
        extra_names=EXTRAS,
    )


def choices(strengths, masses, ids, valid, categories, subset=None):
    selected = {name: [] for name in ("mass", "greedy", "chinese_ceiling")}
    for row in range(len(strengths)):
        mask = valid[row].copy()
        if subset is not None:
            mask &= subset
        indices = np.flatnonzero(mask)
        s = strengths[row, indices]
        p = masses[row, indices]
        bad = p[:, list(BAD)].sum(1)
        cn = p[:, CHINESE]
        labels = categories[ids[row, indices]]
        keys = {
            "mass": (s, -cn, bad),
            "greedy": (
                s,
                -(labels == CHINESE).astype(int),
                np.isin(labels, list(BAD)).astype(int),
            ),
            "chinese_ceiling": (s, bad, -cn),
        }
        for name, key in keys.items():
            selected[name].append(indices[np.lexsort(key)[0]])
    return {name: np.array(value, dtype=np.int64) for name, value in selected.items()}


def worker(dataset, shard):
    assert dataset in DATASETS and 0 <= shard < 20
    root = BASE
    folder = OUT / dataset / f"{shard:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(root / f"states/{dataset}/{shard:03d}.npz") as z:
        states = z["states"].astype(np.float64)
        sample_ids = z["sample_ids"].copy()
    np.testing.assert_array_equal(sample_ids, np.arange(shard * 500, (shard + 1) * 500))
    meta, d, raw = source(root)
    selection = read(root / "calibration/selection.json")["tracks"]
    # "old"/"old_*" retain the original output schema: these are the preceding
    # COMPUTE stage's newly generated comparator predictions, never an archive.
    old = []
    for name in EXTRAS:
        with np.load(root / f"test/{dataset}/{shard:03d}/{name}.npz") as z:
            np.testing.assert_array_equal(z["sample_ids"], sample_ids)
            assert z["valid"].all()
            old.append(
                {
                    k: z[k].copy()
                    for k in ("subtype_masses", "class_masses", "token_ids", "modes")
                }
            )
    selected_s = []
    for i in sample_ids:
        with np.load(root / f"geometry/{dataset}/{i:05d}.npz") as z:
            selected_s.append(float(z["straight_scalar"]))
    strengths = np.broadcast_to(GRID, (500, len(GRID))).copy()
    extra = np.empty((500, len(EXTRAS)))
    extra_points = np.empty((500, len(EXTRAS), states.shape[1]))
    for j, name in enumerate(EXTRAS):
        track, method = name.split("/")
        for i, x in enumerate(states):
            if method == "raw_centroid":
                s = np.linalg.norm(raw)
                p = x + raw
            elif method == "adaptive_straight":
                mode = old[j]["modes"][i]
                s = (
                    selected_s[i]
                    if mode == "right_minimum"
                    else (
                        0.0
                        if mode == "no_steer"
                        else selection[track]["additive"]["parameter"]
                    )
                )
                p = x + s * d
            else:
                parameter = selection[track][method]["parameter"]
                s = parameter if method == "additive" else parameter - x @ d
                p = point(x, d, method, parameter)
            extra[i, j] = s
            extra_points[i, j] = p
    strengths = np.concatenate([strengths, extra], axis=1)
    valid = np.isfinite(strengths) & (strengths > 0) & (strengths < 10)
    assert valid[:, : len(GRID)].all()
    probe = Probe(meta["snapshot"], 15)
    _, matrix = load_partition(
        SimpleNamespace(output=root / "vocabulary"), probe, subtypes=True
    )
    with np.load(root / "vocabulary/vocabulary.npz") as z:
        categories = z["categories"].copy()
    for start in range(0, 500, 8):
        end = min(500, start + 8)
        ids = sample_ids[start:end]
        path = folder / f"{start:03d}.npz"
        if path.exists() and path.with_suffix(".json").exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z["sample_ids"], ids)
                np.testing.assert_array_equal(z["strengths"], strengths[start:end])
            continue
        s = strengths[start:end]
        mask = valid[start:end]
        points = states[start:end, None] + s[:, :, None] * d
        points[:, len(GRID) :] = extra_points[start:end]
        fine = np.full((*s.shape, len(SUBTYPES)), np.nan)
        tokens = np.full(s.shape, -1, dtype=np.int64)
        cache = {}
        audits = []
        for phase in (COARSE, ~COARSE):
            places = np.argwhere(mask & phase[None])
            unique = {}
            for i, j in places:
                key = points[i, j].tobytes()
                if key not in cache:
                    unique.setdefault(key, (i, j))
            if unique:
                values, pred = checked_distribution(
                    probe, np.stack([points[i, j] for i, j in unique.values()]), matrix
                )
                for key, value, token in zip(unique, values, pred):
                    cache[key] = (value, int(token))
                if start in (0, 248, 496):
                    i, j = next(iter(unique.values()))
                    audits.append(
                        audit_distribution(
                            probe, points[i, j], matrix, values[0], pred[0]
                        )
                    )
            for i, j in places:
                fine[i, j], tokens[i, j] = cache[points[i, j].tobytes()]
        regression_count = 0
        max_error = 0.0
        for j, reference in enumerate(old):
            keep = mask[:, len(GRID) + j]
            actual = fine[keep, len(GRID) + j]
            expected = reference["subtype_masses"][start:end][keep]
            np.testing.assert_array_equal(
                tokens[keep, len(GRID) + j], reference["token_ids"][start:end][keep]
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
            if keep.any():
                max_error = max(max_error, float(np.max(abs(actual - expected))))
            # Retain exact frozen cached values after independent replay verification.
            fine[keep, len(GRID) + j] = expected
            regression_count += int(keep.sum())
        masses = aggregate_subtypes(fine)
        coarse_best = choices(s, masses, tokens, mask, categories, COARSE)
        best = choices(s, masses, tokens, mask, categories)
        atomic_npz(
            path,
            sample_ids=ids,
            strengths=s,
            valid=mask,
            subtype_masses=fine,
            masses=masses,
            token_ids=tokens,
            **{f"coarse_{k}": v for k, v in coarse_best.items()},
            **{f"fine_{k}": v for k, v in best.items()},
        )
        atomic_json(
            path.with_suffix(".json"),
            {
                "old_argmax_replays_exact": regression_count,
                "old_mass_max_abs_error": max_error,
                "audits": audits,
                "max_mass_sum_error": float(np.max(abs(masses[mask].sum(1) - 1))),
            },
        )
        print(dataset, shard, end, flush=True)
    # Prespecified global .0125 audit: only the 400 new odd-index points.
    audit_s = np.arange(1, 800, 2) / 80
    for local in range(0, 500, 100):
        path = folder / f"audit-{local:03d}.npz"
        if path.exists():
            continue
        points = states[local] + audit_s[:, None] * d
        fine, tokens = checked_distribution(probe, points, matrix)
        atomic_npz(
            path,
            sample_id=sample_ids[local],
            strengths=audit_s,
            subtype_masses=fine,
            masses=aggregate_subtypes(fine),
            token_ids=tokens,
        )
    atomic_json(
        folder / "complete.json",
        {"n": 500, "global_audits": 5, "dataset": dataset, "shard": shard},
    )


def collect():
    result = {
        "protocol": read(OUT / "protocol.json"),
        "datasets": {},
        "confidence": "10000 paired bootstrap draws, common indices across contrasts within dataset; conditional on frozen output-aware search",
    }
    for dataset_index, dataset in enumerate(DATASETS):
        root = BASE
        with np.load(root / "vocabulary/vocabulary.npz") as z:
            categories = z["categories"].copy()
        names = ("mass", "greedy", "chinese_ceiling")
        fields = ("strengths", "masses", "subtype_masses", "token_ids")
        selected = {
            level: {name: {field: [] for field in fields} for name in names}
            for level in ("coarse", "fine")
        }
        reach = {
            level: {name: [] for name in ("chinese", "accepted")}
            for level in ("coarse", "fine")
        }
        audit_records = []
        regressions = 0
        max_old_error = 0.0
        max_sum_error = 0.0
        hooks = []
        for shard in range(20):
            folder = OUT / dataset / f"{shard:03d}"
            assert read(folder / "complete.json")["n"] == 500
            for start in range(0, 500, 8):
                path = folder / f"{start:03d}.npz"
                info = read(path.with_suffix(".json"))
                regressions += info["old_argmax_replays_exact"]
                max_old_error = max(max_old_error, info["old_mass_max_abs_error"])
                max_sum_error = max(max_sum_error, info["max_mass_sum_error"])
                hooks.extend(info["audits"])
                with np.load(path) as z:
                    data = {k: z[k].copy() for k in z.files}
                n = len(data["sample_ids"])
                rows = np.arange(n)
                np.testing.assert_array_equal(
                    data["sample_ids"],
                    np.arange(shard * 500 + start, shard * 500 + start + n),
                )
                np.testing.assert_allclose(
                    aggregate_subtypes(data["subtype_masses"]),
                    data["masses"],
                    rtol=0,
                    atol=2e-12,
                    equal_nan=True,
                )
                for level in ("coarse", "fine"):
                    mask = data["valid"].copy()
                    if level == "coarse":
                        mask &= COARSE[None]
                    labels = categories[np.maximum(data["token_ids"], 0)]
                    reach[level]["chinese"].extend(
                        np.any(mask & (labels == CHINESE), axis=1).tolist()
                    )
                    reach[level]["accepted"].extend(
                        np.any(mask & ~np.isin(labels, list(BAD)), axis=1).tolist()
                    )
                    recomputed = choices(
                        data["strengths"],
                        data["masses"],
                        data["token_ids"],
                        data["valid"],
                        categories,
                        COARSE if level == "coarse" else None,
                    )
                    for name in names:
                        indices = data[f"{level}_{name}"]
                        np.testing.assert_array_equal(indices, recomputed[name])
                        for field in fields:
                            selected[level][name][field].append(
                                data[field][rows, indices]
                            )
                for row, sample in enumerate(data["sample_ids"]):
                    if sample % 100:
                        continue
                    with np.load(folder / f"audit-{int(sample % 500):03d}.npz") as z:
                        assert int(z["sample_id"]) == sample
                        merged = {
                            field: np.concatenate([data[field][row], z[field]])
                            for field in fields
                        }
                    valid = (
                        np.isfinite(merged["strengths"])
                        & (merged["strengths"] > 0)
                        & (merged["strengths"] < 10)
                    )
                    index = choices(
                        merged["strengths"][None],
                        merged["masses"][None],
                        merged["token_ids"][None],
                        valid[None],
                        categories,
                    )
                    record = {"sample_id": int(sample), "best": {}}
                    for name in names:
                        i = int(index[name][0])
                        old_i = int(data[f"fine_{name}"][row])
                        old = data["masses"][row, old_i]
                        new = merged["masses"][i]
                        old_class = int(categories[data["token_ids"][row, old_i]])
                        new_class = int(categories[merged["token_ids"][i]])
                        record["best"][name] = {
                            "strength": float(merged["strengths"][i]),
                            "masses": new.tolist(),
                            "subtype_masses": merged["subtype_masses"][i].tolist(),
                            "token_id": int(merged["token_ids"][i]),
                            "bad_mass_reduction": float(
                                old[list(BAD)].sum() - new[list(BAD)].sum()
                            ),
                            "chinese_mass_gain": float(new[CHINESE] - old[CHINESE]),
                            "greedy_bad_reduction": int(old_class in BAD)
                            - int(new_class in BAD),
                            "greedy_chinese_gain": int(new_class == CHINESE)
                            - int(old_class == CHINESE),
                        }
                    audit_records.append(record)
        output = {
            level: {
                name: {field: np.concatenate(parts) for field, parts in record.items()}
                for name, record in methods.items()
            }
            for level, methods in selected.items()
        }
        summary = {
            "n": 10000,
            "oracles": {},
            "comparisons": {},
            "coarse_fine": {},
            "global_audit": audit_records,
            "old_argmax_regressions_exact": regressions,
            "max_old_mass_error": max_old_error,
            "max_mass_sum_error": max_sum_error,
            "max_full_hook_mass_error": max(
                a["full_injection_mass_max_abs"] for a in hooks
            ),
            "max_scalar_batch_mass_error": max(
                a["scalar_batch_mass_max_abs"] for a in hooks
            ),
        }
        for name in names:
            a = output["coarse"][name]
            b = output["fine"][name]
            bad = b["masses"][:, list(BAD)].sum(1)
            labels = categories[b["token_ids"]]
            summary["oracles"][name] = {
                "mean_bad_mass": float(bad.mean()),
                "mean_masses": dict(zip(CLASSES, b["masses"].mean(0).tolist())),
                "mean_subtype_masses": dict(
                    zip(SUBTYPES, b["subtype_masses"].mean(0).tolist())
                ),
                "argmax_counts": {
                    label: int((labels == i).sum()) for i, label in enumerate(CLASSES)
                },
                "strength_quantiles": np.quantile(
                    b["strengths"], [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
                ).tolist(),
                "boundary_adjacent_lower": int((b["strengths"] == 1e-6).sum()),
                "boundary_adjacent_upper": int((b["strengths"] == 10 - 1e-6).sum()),
            }
            gains = a["masses"][:, list(BAD)].sum(1) - bad
            cgains = b["masses"][:, CHINESE] - a["masses"][:, CHINESE]
            summary["coarse_fine"][name] = {
                "bad_mass_reduction_mean": float(gains.mean()),
                "bad_mass_reduction_max": float(gains.max()),
                "chinese_mass_gain_mean": float(cgains.mean()),
                "chinese_mass_gain_max": float(cgains.max()),
                "changed_argmax_count": int((a["token_ids"] != b["token_ids"]).sum()),
            }
        for label in ("chinese", "accepted"):
            coarse = np.array(reach["coarse"][label])
            fine = np.array(reach["fine"][label])
            assert np.all(~coarse | fine)
            summary["coarse_fine"][f"{label}_reachability"] = {
                "coarse_count": int(coarse.sum()),
                "fine_count": int(fine.sum()),
                "newly_reachable": int((fine & ~coarse).sum()),
            }
        baselines = {}
        for track in ("mass", "greedy"):
            for method in (
                "raw_centroid",
                "additive",
                "norm_preserving",
                "constant_projection",
                "adaptive_angular",
                "adaptive_straight",
            ):
                key = f"{track}/{method}"
                parts = []
                ids = []
                for shard in range(20):
                    with np.load(root / f"test/{dataset}/{shard:03d}/{key}.npz") as z:
                        parts.append(z["class_masses"].copy())
                        ids.append(z["token_ids"].copy())
                baselines[key] = (
                    np.concatenate(parts),
                    categories[np.concatenate(ids)],
                )
        summary["baselines"] = {
            key: {
                "mean_bad_mass": float(m[:, list(BAD)].sum(1).mean()),
                "mean_masses": dict(zip(CLASSES, m.mean(0).tolist())),
                "argmax_counts": {
                    label: int((lab == j).sum()) for j, label in enumerate(CLASSES)
                },
            }
            for key, (m, lab) in baselines.items()
        }
        summary["global_audit_summary"] = {}
        for name in names:
            records = [r["best"][name] for r in audit_records]
            summary["global_audit_summary"][name] = {}
            for metric in (
                "bad_mass_reduction",
                "chinese_mass_gain",
                "greedy_bad_reduction",
                "greedy_chinese_gain",
            ):
                gains = np.array([r[metric] for r in records])
                summary["global_audit_summary"][name][metric] = {
                    "mean": float(gains.mean()),
                    "max": float(gains.max()),
                    "positive_count": int((gains > 1e-12).sum()),
                    "above_1e_6": int((gains > 1e-6).sum()),
                    "above_1e_4": int((gains > 1e-4).sum()),
                    "above_1e_3": int((gains > 1e-3).sum()),
                }
        keys = []
        deltas = []
        for oracle in ("mass", "greedy"):
            b = output["fine"][oracle]
            lab = categories[b["token_ids"]]
            for ref, (mass, labels) in baselines.items():
                metrics = {
                    "bad_mass_reduction": mass[:, list(BAD)].sum(1)
                    - b["masses"][:, list(BAD)].sum(1),
                    "chinese_mass_gain": b["masses"][:, CHINESE] - mass[:, CHINESE],
                    "greedy_bad_reduction": np.isin(labels, list(BAD)).astype(float)
                    - np.isin(lab, list(BAD)).astype(float),
                    "greedy_chinese_gain": (lab == CHINESE).astype(float)
                    - (labels == CHINESE).astype(float),
                }
                for metric, delta in metrics.items():
                    keys.append((oracle, ref, metric))
                    deltas.append(delta)
        values = np.column_stack(deltas)
        intervals = bootstrap_means(values, seed=20261101 + dataset_index)
        for column, (oracle, ref, metric) in enumerate(keys):
            delta = values[:, column]
            record = {
                "mean_difference": float(delta.mean()),
                "paired_bootstrap_95": intervals[column].tolist(),
                "fraction_improved": float((delta > 1e-12).mean()),
                "fraction_worse": float((delta < -1e-12).mean()),
                "fraction_tied_at_1e-12": float((abs(delta) <= 1e-12).mean()),
                "gap_quantiles": np.quantile(
                    delta, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
                ).tolist(),
            }
            summary["comparisons"].setdefault(oracle, {}).setdefault(ref, {})[
                metric
            ] = record
        for name, record in output["fine"].items():
            atomic_npz(
                OUT / f"{dataset}_{name}_selected.npz",
                sample_ids=np.arange(10000),
                **record,
            )
        atomic_npz(
            OUT / f"{dataset}_grid_convergence.npz",
            **{
                f"{level}_{name}_{field}": array
                for level, methods in output.items()
                for name, record in methods.items()
                for field, array in record.items()
            },
        )
        atomic_json(OUT / f"{dataset}_summary.json", summary)
        result["datasets"][dataset] = summary
    atomic_json(OUT / "summary.json", result)
    atomic_json(
        OUT / "completion.json",
        {
            "complete": True,
            "certified_continuous_bound": False,
            "datasets": DATASETS,
            "n_each": 10000,
        },
    )



def main():
    global BASE, OUT
    from .common import DEFAULT_ROOT, digest
    from .compute import require_computed, invoke
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "prepare", "worker", "collect"], nargs="?", default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    BASE = args.root
    OUT = BASE / "linear_oracle"
    require_computed(BASE)
    request = {"computed_sha256": digest(BASE / "computed.json"),
               "prepared_sha256": digest(BASE / "prepared.json")}
    gate = OUT / "request.json"
    if gate.exists():
        if read(gate) != request:
            raise ValueError("Oracle dependency fingerprint changed; use a fresh run")
    else:
        atomic_json(gate, request)
    if args.stage in ("all", "prepare"):
        prepare()
    if args.stage == "all":
        for dataset in DATASETS:
            for shard in range(20):
                invoke("oracle", BASE, "worker", "--dataset", dataset, "--shard", shard)
        collect()
    elif args.stage == "worker":
        if args.dataset is None:
            parser.error("worker requires --dataset")
        prepare()
        worker(args.dataset, args.shard)
    elif args.stage == "collect":
        collect()
    if args.stage in ("all", "collect"):
        atomic_json(OUT / "complete.json", {
            "complete": True, "request": request,
            "summary_sha256": digest(OUT / "summary.json"),
            "fresh_outputs": True, "certified_continuous_bound": False,
        })


if __name__ == "__main__":
    main()
