"""Full-cohort paired summaries and independent final audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from .vocabulary import (BAD, CHINESE, CLASSES, SUBTYPES, aggregate_subtypes, bootstrap_means)
from .common import atomic_json, atomic_npz
def collect(root):
    """Paired, full-cohort summaries; unresolved primary outcomes block completion."""
    from itertools import combinations

    tracks = ("mass", "greedy")
    methods = (
        "raw_centroid",
        "additive",
        "norm_preserving",
        "constant_projection",
        "adaptive_angular",
        "adaptive_straight",
    )
    selection = json.loads((root / "calibration/selection.json").read_text())
    assert selection["all_interior"]
    approval = json.loads((root / "inference_approval.json").read_text())
    assert approval["approved"] and approval["split_audit_passed"]
    result = {
        "n_per_test": 10000,
        "calibration_n": 1000,
        "tracks": tracks,
        "methods": methods,
        "calibration": selection,
        "datasets": {},
        "confidence": "10000 paired bootstrap draws; common indices across all methods, both tracks and metrics within each dataset; conditional on calibration",
        "identity_limitations": json.loads((root / "split_audit.json").read_text())[
            "limitations"
        ],
    }
    unresolved = []
    with np.load(root / "vocabulary/vocabulary.npz") as vocab:
        categories = vocab["categories"].copy()
    datasets = approval.get("evaluation_datasets", ["wiki_test", "fineweb_test"])
    for dataset_index, dataset in enumerate(datasets):
        masses, fine, tokens, records, audits, quarter_checks = {}, {}, {}, {}, [], {}
        for track in tracks:
            for method in methods:
                key = f"{track}/{method}"
                pieces, subs, ids, samples, valid, stats = [], [], [], [], [], []
                checks = []
                for shard in range(20):
                    path = root / f"test/{dataset}/{shard:03d}/{key}.npz"
                    audits.extend(
                        json.loads(path.with_suffix(".json").read_text())["audits"]
                    )
                    with np.load(path) as data:
                        pieces.append(data["class_masses"].copy())
                        subs.append(data["subtype_masses"].copy())
                        ids.append(data["token_ids"].copy())
                        samples.extend(data["sample_ids"].tolist())
                        valid.append(data["valid"].copy())
                        stats.append(
                            {
                                name: data[name].copy()
                                for name in (
                                    "modes",
                                    "effective_strengths",
                                    "norms_before",
                                    "norms_after",
                                    "projections_before",
                                    "projections_after",
                                )
                            }
                        )
                        if "quarter_token_comparable" in data:
                            mask = data["quarter_token_comparable"]
                            checks.append(
                                {
                                    "n": int(mask.sum()),
                                    "exact": int(data["quarter_token_agreement"].sum()),
                                    "disagreement_ids": data["sample_ids"][
                                        mask & ~data["quarter_token_agreement"]
                                    ].tolist(),
                                }
                            )
                assert samples == list(range(10000))
                availability = np.concatenate(valid)
                if not availability.all():
                    unresolved.append(
                        {
                            "dataset": dataset,
                            "method": key,
                            "indices": np.flatnonzero(~availability).tolist(),
                        }
                    )
                masses[key] = np.concatenate(pieces)
                fine[key] = np.concatenate(subs)
                tokens[key] = np.concatenate(ids)
                records[key] = {
                    name: np.concatenate([part[name] for part in stats])
                    for name in stats[0]
                }
                if checks:
                    quarter_checks[key] = checks
        if unresolved:
            atomic_json(
                root / "unresolved_primary_outcomes.json",
                {"complete": False, "cases": unresolved, "no_silent_drops": True},
            )
            raise RuntimeError(
                "Unresolved primary cases retained; parent numerical review required before a full-cohort summary"
            )
        for key, mass in masses.items():
            assert mass.shape == (10000, 5) and fine[key].shape == (10000, 11)
            assert np.isfinite(mass).all() and (mass >= 0).all()
            np.testing.assert_allclose(mass.sum(1), 1, rtol=0, atol=2e-12)
            np.testing.assert_allclose(
                aggregate_subtypes(fine[key]), mass, rtol=0, atol=2e-12
            )
            assert np.all((tokens[key] >= 0) & (tokens[key] < len(categories)))
        np.testing.assert_array_equal(
            masses["mass/raw_centroid"], masses["greedy/raw_centroid"]
        )
        bad = {key: value[:, list(BAD)].sum(1) for key, value in masses.items()}
        labels = {key: categories[value] for key, value in tokens.items()}
        summary = {
            "methods": {},
            "comparisons": {},
            "quarter_token_audits": quarter_checks,
            "max_mass_sum_error": max(
                float(np.max(abs(v.sum(1) - 1))) for v in masses.values()
            ),
            "max_full_hook_mass_error": max(
                a["full_injection_mass_max_abs"] for a in audits
            ),
            "max_scalar_batch_mass_error": max(
                a["scalar_batch_mass_max_abs"] for a in audits
            ),
            "audits": audits,
        }
        for key in masses:
            stats = records[key]
            counts = {
                label: int((labels[key] == i).sum()) for i, label in enumerate(CLASSES)
            }
            modes, numbers = np.unique(stats["modes"], return_counts=True)
            summary["methods"][key] = {
                "mean_bad_mass": float(bad[key].mean()),
                "greedy_bad_rate": float(np.isin(labels[key], list(BAD)).mean()),
                "mean_masses": {
                    label: float(masses[key][:, i].mean())
                    for i, label in enumerate(CLASSES)
                },
                "mean_subtype_masses": {
                    label: float(fine[key][:, i].mean())
                    for i, label in enumerate(SUBTYPES)
                },
                "argmax_counts": counts,
                "mode_counts": dict(zip(modes.tolist(), numbers.tolist())),
                "negative_effective_strength_fraction": float(
                    (stats["effective_strengths"] < 0).mean()
                ),
                "effective_strength_quantiles": np.quantile(
                    stats["effective_strengths"], [0, 0.01, 0.5, 0.99, 1]
                ).tolist(),
                "norm_relative_change_quantiles": np.quantile(
                    stats["norms_after"] / stats["norms_before"] - 1,
                    [0, 0.01, 0.5, 0.99, 1],
                ).tolist(),
                "projection_before_mean_std": [
                    float(stats["projections_before"].mean()),
                    float(stats["projections_before"].std()),
                ],
                "projection_after_mean_std": [
                    float(stats["projections_after"].mean()),
                    float(stats["projections_after"].std()),
                ],
            }
        keys, changes = [], []
        for track in tracks:
            for reference, method in combinations(methods, 2):
                ref, candidate = f"{track}/{reference}", f"{track}/{method}"
                metrics = {
                    "bad_mass_reduction": bad[ref] - bad[candidate],
                    "chinese_mass_gain": masses[candidate][:, CHINESE]
                    - masses[ref][:, CHINESE],
                    "greedy_bad_reduction": np.isin(labels[ref], list(BAD)).astype(
                        float
                    )
                    - np.isin(labels[candidate], list(BAD)).astype(float),
                    "greedy_chinese_gain": (labels[candidate] == CHINESE).astype(float)
                    - (labels[ref] == CHINESE).astype(float),
                }
                for metric, delta in metrics.items():
                    keys.append((ref, candidate, metric))
                    changes.append(delta)
        values = np.column_stack(changes)
        intervals = bootstrap_means(values, seed=20261025 + dataset_index)
        for i, (reference, method, metric) in enumerate(keys):
            delta = values[:, i]
            record = {
                "mean_improvement": float(delta.mean()),
                "paired_bootstrap_95": intervals[i].tolist(),
                "fraction_improved": float((delta > 1e-12).mean()),
                "fraction_worse": float((delta < -1e-12).mean()),
                "fraction_tied_at_1e-12": float((abs(delta) <= 1e-12).mean()),
            }
            if metric == "bad_mass_reduction":
                record["relative_reduction"] = float(
                    delta.mean() / bad[reference].mean()
                )
            summary["comparisons"].setdefault(reference, {}).setdefault(method, {})[
                metric
            ] = record
        result["datasets"][dataset] = summary
        atomic_npz(
            root / f"{dataset}_outputs.npz",
            method_names=list(masses),
            class_names=CLASSES,
            subtype_names=SUBTYPES,
            masses=np.array(list(masses.values())),
            subtype_masses=np.array(list(fine.values())),
            token_ids=np.array(list(tokens.values())),
            sample_ids=np.arange(10000),
        )
    atomic_json(root / "summary.json", result)
    atomic_json(
        root / "completion.json",
        {
            "complete": True,
            "test_datasets": datasets,
            "n_per_test": 10000,
            "tracks": tracks,
            "methods": methods,
            "fresh_outcomes": True,
        },
    )


def audit(root):
    """Replay frozen selections and validate every retained prediction/partition."""
    import hashlib
    from collections import Counter

    from .geometry import select_profile

    approval = json.loads((root / "inference_approval.json").read_text())
    result = {"complete": False, "datasets": {}}
    with np.load(root / "vocabulary/vocabulary.npz") as vocabulary:
        categories = vocabulary["categories"].copy()
    for dataset in approval.get("evaluation_datasets", ["wiki_test", "fineweb_test"]):
        manifest = root / f"manifests/{dataset}.json"
        assert (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            == approval["manifest_sha256"][dataset]
        )
        report = {
            "geometry": {},
            "prediction_rows": 0,
            "quarter_token_comparisons": 0,
            "quarter_token_exact": 0,
            "quarter_category_exact": 0,
            "quarter_disagreements": [],
        }
        for curve in ("angular", "straight"):
            counts = Counter()
            max_l2 = max_p99 = 0.0
            quarter_n = 0
            mode_equal = scalar_equal = 0
            for i in range(10000):
                with np.load(root / f"geometry/{dataset}/{i:05d}.npz") as data:
                    assert int(data["sample_id"]) == i and bool(data[f"{curve}_valid"])
                    selected = select_profile(
                        curve, data[f"{curve}_grid"], data[f"{curve}_speed"]
                    )
                    assert selected["mode"] == str(data[f"{curve}_mode"])
                    np.testing.assert_equal(selected["scalar"], data[f"{curve}_scalar"])
                    l2 = float(data[f"{curve}_relative_l2"])
                    p99 = float(data[f"{curve}_relative_p99"])
                    assert l2 < 0.05 and p99 < 0.1
                    max_l2 = max(max_l2, l2)
                    max_p99 = max(max_p99, p99)
                    counts[selected["mode"]] += 1
                    if i % 100 == 0:
                        quarter_n += 1
                        assert bool(data[f"{curve}_quarter_valid"])
                        mode_equal += int(data[f"{curve}_quarter_mode_agreement"])
                        scalar_equal += int(data[f"{curve}_quarter_scalar_agreement"])
            report["geometry"][curve] = {
                "n": 10000,
                "selection_replays_exact": 10000,
                "valid": 10000,
                "mode_counts": dict(counts),
                "max_relative_l2": max_l2,
                "max_relative_p99": max_p99,
                "quarter_n": quarter_n,
                "quarter_mode_exact": mode_equal,
                "quarter_scalar_exact": scalar_equal,
            }
        for shard in range(20):
            for track in ("mass", "greedy"):
                for method in (
                    "raw_centroid",
                    "additive",
                    "norm_preserving",
                    "constant_projection",
                    "adaptive_angular",
                    "adaptive_straight",
                ):
                    with np.load(
                        root / f"test/{dataset}/{shard:03d}/{track}/{method}.npz"
                    ) as data:
                        assert data["valid"].all()
                        np.testing.assert_array_equal(
                            data["argmax_classes"], categories[data["token_ids"]]
                        )
                        np.testing.assert_allclose(
                            data["class_masses"],
                            aggregate_subtypes(data["subtype_masses"]),
                            rtol=0,
                            atol=2e-12,
                        )
                        np.testing.assert_allclose(
                            data["subtype_masses"].sum(1), 1, rtol=0, atol=2e-12
                        )
                        report["prediction_rows"] += len(data["token_ids"])
                        if "quarter_token_comparable" in data:
                            mask = data["quarter_token_comparable"]
                            first = data["token_ids"][mask]
                            second = data["quarter_token_ids"][mask]
                            report["quarter_token_comparisons"] += len(first)
                            report["quarter_token_exact"] += int(
                                (first == second).sum()
                            )
                            report["quarter_category_exact"] += int(
                                (categories[first] == categories[second]).sum()
                            )
                            for sample, left, right in zip(
                                data["sample_ids"][mask], first, second
                            ):
                                if left != right:
                                    report["quarter_disagreements"].append(
                                        {
                                            "sample_id": int(sample),
                                            "track": track,
                                            "method": method,
                                            "primary_token": int(left),
                                            "quarter_token": int(right),
                                            "category_equal": bool(
                                                categories[left] == categories[right]
                                            ),
                                        }
                                    )
        result["datasets"][dataset] = report
    result["complete"] = True
    atomic_json(root / "final_validation.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    audit(a.root)
    collect(a.root)
