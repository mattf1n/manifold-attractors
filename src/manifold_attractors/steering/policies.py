"""Frozen extrema and fallback policies (no outcome-derived inputs)."""
import numpy as np

def basin_selection(grid, speed):
    """Right minimum, else left-basin-gated no-steer, else fixed additive."""
    lo = np.flatnonzero((speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])) + 1
    hi = np.flatnonzero((speed[1:-1] > speed[:-2]) & (speed[1:-1] >= speed[2:])) + 1
    lo = lo[(lo >= 2) & (lo < len(grid) - 2)]
    left = max(grid[hi][grid[hi] < 0], default=grid[0])
    right = min(grid[hi][grid[hi] >= 0], default=grid[-1])
    anchors = lo[(grid[lo] > left) & (grid[lo] < right)]
    if len(anchors) != 1:
        return {"mode": "reference_ambiguity", "strengths": None}
    near = anchors[0]
    eligible = lo[(grid[lo] > max(0.0, grid[near])) & (grid[lo] < 10)]
    if len(eligible):
        chosen = eligible[np.argmin(speed[eligible])]
        return {"mode": "right_minimum", "strengths": grid[[near, chosen]]}
    other_left = lo[(grid[lo] < grid[near]) & (grid[lo] > -10) & (grid[lo] < 0)]
    if len(other_left):
        return {"mode": "no_steer", "strengths": np.array([grid[near], 0.0])}
    if grid[0] >= -10:
        raise ValueError(
            "Left search interval is incomplete; extend below -10 before asserting no left minimum"
        )
    # Confirmatory geometry consumes only the reference in fixed mode. Never
    # carry the obsolete development-tuned fallback: calibrate it in this run.
    return {"mode": "fixed_additive", "strengths": np.array([grid[near], np.nan])}

def minimum_indices(speed, step):
    size = len(speed)
    grid = np.arange(-size // 2, size // 2)
    values = speed[grid % size]
    minima = (
        np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])) + 1
    )
    assert len(minima)
    near = minima[np.argmin(np.abs(grid[minima]))]
    right = minima[minima > near]
    if not len(right):
        # Full-circle continuation instead of dropping an unusual example.
        indices = np.flatnonzero(
            (speed < np.roll(speed, 1)) & (speed <= np.roll(speed, -1))
        )
        delta = (indices - grid[near]) % size
        ahead = indices[delta > 0][np.argmin(delta[delta > 0])]
        right_angle = float((grid[near] + (ahead - grid[near]) % size) * step)
        return np.array([grid[near] * step, right_angle])
    return grid[[near, right[0]]] * step


def bounded_minimum(speed, step, low=10.0, high=55.0, choice="lowest_speed"):
    """Select a local minimum in the allowed range, or return zero for no steering."""
    near = minimum_indices(speed, step)[0]
    angles = (
        np.flatnonzero((speed < np.roll(speed, 1)) & (speed <= np.roll(speed, -1)))
        * step
    )
    candidates = angles[(angles >= low) & (angles <= high) & (angles > near)]
    if not len(candidates):
        return 0.0
    if choice == "leftmost":
        return float(candidates[0])
    if choice == "lowest_speed":
        indices = np.rint(candidates / step).astype(int)
        return float(candidates[np.argmin(speed[indices])])
    raise ValueError(f"Unknown minimum choice: {choice}")


def steering_target(speed, step, low=10.0, high=55.0):
    """Steer right if possible; a left minimum gates no steering; otherwise use fixed."""
    right = bounded_minimum(speed, step, low, high, choice="lowest_speed")
    if right:
        return "right_minimum", right
    near = minimum_indices(speed, step)[0]
    indices = np.flatnonzero(
        (speed < np.roll(speed, 1)) & (speed <= np.roll(speed, -1))
    )
    angles = (indices * step + 180) % 360 - 180
    keep = (angles >= -high) & (angles <= -low) & (angles < near)
    indices, angles = indices[keep], angles[keep]
    if len(indices):
        return "no_steer", 0.0
    return "fixed_additive", None
