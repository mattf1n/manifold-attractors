"""Frozen four-block geometry, independent of calibrated fallback and decoding.

No fresh outcomes are needed to define this protocol. ``evaluate`` accepts an
existing Probe so geometry and downstream probability decoding can share it.
All strengths supplied to point_constructor are unit-direction coefficients;
angular scalars are degrees. Fixed fallback is ordinary additive in both tracks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .policies import basin_selection
from .probe import Probe
from .common import atomic_json, atomic_npz
from .policies import minimum_indices, steering_target
from ..core.pair_barrier_geometry import speed_from_vectors

DEFAULT_ROOT = Path(
    "runs/adaptive_steering"
)
PROTOCOL = {
    "version": "four-block-confirmatory-v1",
    "hidden_state_index": 15,
    "block_indices": [15, 16, 17, 18],
    "human_blocks": [16, 17, 18, 19],
    "output": "raw residual before final norm",
    "precision": "native FP32 states promoted to FP64; FP64 Probe",
    "direction": "negative canonical en-minus-zh direction, unit normalized",
    "angular": {
        "initial_step": 1.0,
        "range": [0, 360],
        "periodic": True,
        "selection": "existing steering_target; lowest right 10<=angle<=55 after nearest reference; left gate else fixed",
    },
    "straight": {
        "initial_step": 0.1,
        "ranges": [[-16, 32], [-32, 64]],
        "selection": "existing basin_selection; zero sampled maximum chooses LEFT basin; lowest after reference at 0<s<10; other left -10<s<0 gates no_steer",
    },
    "convergence": {
        "relative_l2_below": 0.05,
        "relative_p99_below": 0.1,
        "additional_refinements": 3,
        "comparisons": 4,
        "relative_denominator_floor": 1e-12,
        "straight_endpoint_exclusion": 2,
    },
    "audit": "dataset-local sample_id % 100 == 0; extra halving; record mode/scalar agreements; token agreement deferred to shared decoder",
    "fallback": "fixed mode, scalar NaN; calibrate later, never hardcode strength",
    "failures": "retain all cases; withhold genuine reference/numerical failures; no smoothing or extrema cutoffs",
    "ties": "equal-speed eligible minima choose smaller scalar; equidistant angular reference preserves ordered-grid argmin; preserve helper plateau inequalities; ties/plateaux diagnostic only",
    "audit_policy": "mode/scalar/token disagreements diagnostic only; never replace primary selection",
}


def point_constructor(state, direction, track, mode, scalar=None, fixed_strength=None):
    """Construct selected points, including delayed additive fixed fallback."""
    state, direction = (
        np.asarray(state, dtype=float),
        np.asarray(direction, dtype=float),
    )
    if mode == "fixed":
        if fixed_strength is None or not np.isfinite(fixed_strength):
            raise ValueError("Fixed selection requires separately calibrated strength")
        return state + fixed_strength * direction
    if mode == "no_steer":
        return state.copy()
    if mode != "right_minimum":
        raise ValueError(f"Unresolved selection: {mode}")
    scalar = np.asarray(scalar, dtype=float)
    if not np.all(np.isfinite(scalar)):
        raise ValueError("Nonfinite selection")
    if track == "straight":
        return state + scalar[..., None] * direction
    if track != "angular":
        raise ValueError(track)
    radius = np.linalg.norm(state)
    if not np.isfinite(radius) or radius == 0:
        raise ValueError("Degenerate state radius")
    pole = state / radius
    tangent = direction - (direction @ pole) * pole
    norm = np.linalg.norm(tangent)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Degenerate projected tangent")
    tangent /= norm
    theta = np.deg2rad(scalar)
    return radius * (
        np.cos(theta)[..., None] * pole + np.sin(theta)[..., None] * tangent
    )


def _extrema(grid, speed, periodic):
    if periodic:
        lo = np.flatnonzero((speed < np.roll(speed, 1)) & (speed <= np.roll(speed, -1)))
        hi = np.flatnonzero((speed > np.roll(speed, 1)) & (speed >= np.roll(speed, -1)))
    else:
        lo = np.flatnonzero((speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])) + 1
        hi = np.flatnonzero((speed[1:-1] > speed[:-2]) & (speed[1:-1] >= speed[2:])) + 1
        lo = lo[(lo >= 2) & (lo < len(grid) - 2)]
    return lo, hi


def select_profile(track, grid, speed):
    """Reuse legacy gates exactly; expose ambiguities rather than substituting."""
    flags = []
    result = {"mode": "reference_ambiguity", "scalar": np.nan, "reference": np.nan}
    if not np.all(np.isfinite(speed)):
        return dict(result, flags=["nonfinite_speed"])
    lo, hi = _extrema(grid, speed, track == "angular")
    if np.any(np.diff(speed) == 0) or (track == "angular" and speed[0] == speed[-1]):
        flags.append("sampled_plateau")
    if track == "straight":
        if not np.any(grid[hi] < 0) or not np.any(grid[hi] >= 0):
            flags.append("reference_not_bracketed_by_sampled_maxima")
        chosen = basin_selection(grid, speed)
        mode = chosen["mode"]
        if chosen["strengths"] is not None:
            result["reference"] = float(chosen["strengths"][0])
            if mode != "fixed_additive":
                result["scalar"] = float(chosen["strengths"][1])
    else:
        if not len(lo):
            return dict(result, flags=flags + ["no_angular_minima"])
        signed = (grid[lo] + 180) % 360 - 180
        if np.sum(abs(signed) == np.min(abs(signed))) > 1:
            flags.append("equidistant_angular_reference")
        result["reference"] = float(minimum_indices(speed, float(grid[1] - grid[0]))[0])
        mode, scalar = steering_target(speed, float(grid[1] - grid[0]))
        if scalar is not None:
            result["scalar"] = float(scalar)
    result["mode"] = "fixed" if mode == "fixed_additive" else mode
    if mode == "reference_ambiguity":
        flags.append("reference_ambiguity")
    if mode == "right_minimum":
        eligible = lo[
            (grid[lo] > result["reference"])
            & (
                (grid[lo] >= 10) & (grid[lo] <= 55)
                if track == "angular"
                else (grid[lo] > 0) & (grid[lo] < 10)
            )
        ]
        if len(eligible) and np.sum(speed[eligible] == np.min(speed[eligible])) > 1:
            flags.append("tied_lowest_eligible_speed")
    return dict(result, flags=flags)


def _speed(values, step, track):
    if track == "angular":
        return speed_from_vectors(values, step, 1)
    return np.linalg.norm(np.gradient(values, step, axis=0, edge_order=2), axis=1)


def _halve(probe, points, grid, values, step, track):
    periodic = track == "angular"
    fine_grid = (
        np.arange(2 * len(grid)) * (step / 2)
        if periodic
        else np.linspace(grid[0], grid[-1], 2 * len(grid) - 1)
    )
    fine = np.empty((len(fine_grid), values.shape[1]), dtype=np.float64)
    fine[::2] = values
    fine[1::2] = probe.forward(points(fine_grid[1::2]), full=False)
    return fine_grid, fine


def _profile(probe, state, direction, track, extent, audit):
    step = 1.0 if track == "angular" else 0.1
    grid = (
        np.arange(360, dtype=float)
        if track == "angular"
        else np.linspace(-extent / 2, extent, round(1.5 * extent / step) + 1)
    )
    points = lambda scalars: point_constructor(
        state, direction, track, "right_minimum", scalars
    )
    values = probe.forward(points(grid), full=False)
    converged = False
    for refinement in range(4):
        fine_grid, fine = _halve(probe, points, grid, values, step, track)
        coarse_speed, fine_speed = (
            _speed(values, step, track),
            _speed(fine, step / 2, track),
        )
        a, b = coarse_speed, fine_speed[::2]
        if track == "straight":
            a, b = a[2:-2], b[2:-2]
        denominator = np.linalg.norm(b)
        l2 = float(np.linalg.norm(a - b) / denominator) if denominator else np.inf
        p99 = float(np.quantile(abs(a - b) / np.maximum(b, 1e-12), 0.99))
        converged = bool(
            np.isfinite(l2) and np.isfinite(p99) and l2 < 0.05 and p99 < 0.1
        )
        if converged or refinement == 3:
            break
        grid, values, step = fine_grid, fine, step / 2
    selected = select_profile(track, fine_grid, fine_speed)
    result = dict(
        selected,
        grid=fine_grid,
        speed=fine_speed,
        step=step / 2,
        relative_l2=l2,
        relative_p99=p99,
        refinements=refinement,
        converged=converged,
        audit_requested=bool(audit),
    )
    if not converged:
        result["flags"].append("derivative_nonconvergence")
    fatal_flags = {
        "nonfinite_speed",
        "no_angular_minima",
        "reference_not_bracketed_by_sampled_maxima",
        "reference_ambiguity",
        "derivative_nonconvergence",
    }
    result["valid"] = bool(converged and not fatal_flags.intersection(result["flags"]))
    if audit:
        qgrid, quarter = _halve(probe, points, fine_grid, fine, step / 2, track)
        qspeed = _speed(quarter, step / 4, track)
        q = select_profile(track, qgrid, qspeed)
        for key, value in q.items():
            result["quarter_" + key] = value
        result.update(
            quarter_valid=not bool(
                (fatal_flags - {"derivative_nonconvergence"}).intersection(q["flags"])
            ),
            quarter_grid=qgrid,
            quarter_speed=qspeed,
            quarter_step=step / 4,
            quarter_mode_agreement=selected["mode"] == q["mode"],
            quarter_scalar_agreement=bool(selected["scalar"] == q["scalar"])
            if selected["mode"] != "fixed" or q["mode"] != "fixed"
            else True,
            quarter_scalar_delta=float(q["scalar"] - selected["scalar"]),
            quarter_token_agreement_status="pending_shared_decoder",
        )
        if not result["quarter_mode_agreement"]:
            result["flags"].append("quarter_mode_disagreement")
        if not result["quarter_scalar_agreement"]:
            result["flags"].append("quarter_scalar_disagreement")
        # Audit flags do not invalidate or replace the primary selection.
    return result


def evaluate(probe, state, direction, audit=False, layer=15):
    """Return independent angular and straight records; never decode tokens here."""
    if probe.start != layer:
        raise ValueError(
            f"Four-block confirmatory probe must start at hidden-state index {layer}"
        )
    state, direction = (
        np.asarray(state, dtype=float),
        np.asarray(direction, dtype=float),
    )
    if (
        state.shape != direction.shape
        or state.ndim != 1
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("Invalid state/direction")
    if not np.isclose(np.linalg.norm(direction), 1, rtol=0, atol=1e-12):
        raise ValueError("Direction must be unit target-minus-en")
    results = {}
    for track in ("angular", "straight"):
        try:
            result = _profile(probe, state, direction, track, 32, audit)
            if track == "straight" and (
                "reference_not_bracketed_by_sampled_maxima" in result["flags"]
                or result["mode"] == "reference_ambiguity"
            ):
                initial = result
                result = _profile(probe, state, direction, track, 64, audit)
                result["extent_expanded"] = True
                result["initial_flags"] = initial["flags"]
                result["initial_mode"] = initial["mode"]
            results[track] = result
        except (ValueError, AssertionError, FloatingPointError) as error:
            results[track] = {
                "mode": "numerical_failure",
                "scalar": np.nan,
                "valid": False,
                "flags": [type(error).__name__ + ": " + str(error)],
                "audit_requested": bool(audit),
            }
    return results


def check_inference_approval(root, dataset):
    """Fail closed before loading states, direction, or a model."""
    root = Path(root)
    approval = json.loads((root / "inference_approval.json").read_text())
    for key in (
        "approved",
        "split_audit_passed",
        "protocol_frozen",
        "direction_provenance_approved",
    ):
        if approval.get(key) is not True:
            raise ValueError(f"Inference approval missing: {key}")
    manifest = root / "manifests" / f"{dataset}.json"
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected = approval.get("manifest_sha256", {})
    if not isinstance(expected, dict) or expected.get(dataset) != actual:
        raise ValueError(f"Inference manifest hash mismatch: {dataset}")
    return approval


def run(root, dataset, shard, shard_size):
    root = Path(root)
    check_inference_approval(root, dataset)
    metadata = json.loads((root / "direction/llama3.json").read_text())
    from .common import MODELS, run_model
    spec = MODELS[run_model(root)]
    layer = spec["layer"]
    if metadata["hidden_state_index"] != layer:
        raise ValueError("Direction layer mismatch")
    with np.load(root / "direction/llama3.npz") as data:
        direction = -data["direction"].astype(float)
    direction /= np.linalg.norm(direction)
    with np.load(root / f"states/{dataset}/{shard:03d}.npz") as data:
        if data["states"].dtype != np.float32:
            raise ValueError("Expected native FP32 states")
        states, ids = data["states"].astype(float), data["sample_ids"].copy()
    if states.shape != (len(ids), spec["width"]) or not 0 < len(ids) <= shard_size:
        raise ValueError("State shard shape mismatch")
    np.testing.assert_array_equal(
        ids, np.arange(shard * shard_size, shard * shard_size + len(ids))
    )
    output = root / "geometry" / dataset
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    target = metadata.get("target", "zh")
    if target != "zh":
        protocol["direction"] = protocol["direction"].replace("zh", target)
    if layer != PROTOCOL["hidden_state_index"]:
        protocol.update(hidden_state_index=layer, model=spec["model"],
                        block_indices=list(range(layer, layer + 4)),
                        human_blocks=list(range(layer + 1, layer + 5)))
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    atomic_json(
        output / f"protocol_size{shard_size}_{shard:03d}.json",
        dict(
            protocol,
            protocol_sha256=fingerprint,
            shard_size=shard_size,
            shard=shard,
            sample_ids=ids.tolist(),
        ),
    )
    probe = Probe(metadata["snapshot"], layer)
    for state, sample_id in zip(states, ids, strict=True):
        result = evaluate(probe, state, direction, int(sample_id) % 100 == 0, layer)
        flat = {
            f"{track}_{key}": np.asarray(value)
            for track, record in result.items()
            for key, value in record.items()
        }
        atomic_npz(
            output / f"{int(sample_id):05d}.npz",
            sample_id=sample_id,
            protocol_sha256=fingerprint,
            **flat,
        )
        print(
            dataset,
            int(sample_id),
            {k: (r["mode"], r["valid"]) for k, r in result.items()},
            flush=True,
        )
    atomic_json(
        output / f"complete_size{shard_size}_{shard:03d}.json",
        {
            "n": len(ids),
            "sample_ids": ids.tolist(),
            "protocol_sha256": fingerprint,
            "shard_size": shard_size,
            "shard": shard,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--dataset", choices=["wiki_test", "fineweb_test"], required=True
    )
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    args = parser.parse_args()
    if args.shard < 0 or args.shard_size <= 0:
        parser.error("shard must be nonnegative and shard-size positive")
    from .common import require_prepared
    require_prepared(args.root)
    run(args.root, args.dataset, args.shard, args.shard_size)
