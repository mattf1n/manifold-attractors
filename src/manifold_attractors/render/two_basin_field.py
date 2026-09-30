"""Export three rounded basins arranged in an oval for PGFPlots."""
from math import cos, exp, floor, pi, sin, sqrt
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data/two_basin_field.dat"
INNER_DECAY_WIDTH = 0.20
BAND_WIDTH = INNER_DECAY_WIDTH
OUTER_SPEED_FLOOR = 0.25
BOUNDARY_SPEED = 0.75
# Recalibrate if the plot size, sampling, or arrow scale changes.
MIN_DISPLAY_SPEED = 0.014
REGULARIZATION = 0.25
TANGENT_SHEAR = 0.35
HORIZONTAL_STRETCH = 1.5
DIRECTION_NOISE_RADIANS = 0.80  # At most about 46 degrees; preserves speed.
MIN_SAMPLE_DISTANCE = 0.085
SAMPLE_SEED = 42
BOUNDS = (-2.8, 2.8, -1.4, 1.4)


# Round only the intersections of internal borders with the outer oval.
CELL_ROUNDING = 0.50
SITES = [(-0.72, 0.30), (0.70, 0.35), (0.0, -0.48)]


def cell_walls(x, y, index):
    """Round outer/inner wall joins, leaving the central junction unchanged."""
    s = (x - TANGENT_SHEAR * y) / HORIZONTAL_STRETCH
    radius = 1.15
    gx = s / (HORIZONTAL_STRETCH * radius)
    gy = y / radius - TANGENT_SHEAR * gx
    walls = [((s**2 + y**2 - radius**2) / (2 * radius), gx, gy)]
    ax, ay = SITES[index]
    for j, (bx, by) in enumerate(SITES):
        if j == index:
            continue
        length = sqrt((ax - bx)**2 + (ay - by)**2)
        level = ((x - ax)**2 + (y - ay)**2 - (x - bx)**2 - (y - by)**2) / (2 * length)
        walls.append((level, (bx - ax) / length, (by - ay) / length))
    outer = walls[0]
    rounded = []
    for inner in walls[1:]:
        delta = outer[0] - inner[0]
        if delta >= CELL_ROUNDING:
            rounded.append(outer)
        elif delta <= -CELL_ROUNDING:
            rounded.append(inner)
        else:
            # Compact C1 smooth maximum: exactly the original wall away
            # from the outer intersection, unlike log-sum-exp rounding.
            weight = 0.5 * (1 + delta / CELL_ROUNDING)
            level = (outer[0] + inner[0]) / 2 + delta**2 / (4 * CELL_ROUNDING) + CELL_ROUNDING / 4
            rounded.append((level,
                            weight * outer[1] + (1 - weight) * inner[1],
                            weight * outer[2] + (1 - weight) * inner[2]))
    return rounded


def base_velocity(x, y):
    """Point into the cells on both sides, with smoothly blended wall normals."""
    fields, distances = [], []
    for i in range(3):
        walls = cell_walls(x, y, i)
        level = max(wall[0] for wall in walls)
        if abs(level) < 1e-14:
            return 0.0, 0.0
        inside = level < 0
        # Flow points inward on both sides. Outside, use only violated
        # constraints, smoothly weighting them instead of selecting one.
        selected = walls if inside else [wall for wall in walls if wall[0] > 0]
        vectors, ds, weights = [], [], []
        for value, gx, gy in selected:
            length = sqrt(gx**2 + gy**2)
            d = abs(value) / sqrt(length**2 + REGULARIZATION**2)
            width = INNER_DECAY_WIDTH if inside else BAND_WIDTH
            floor_speed = 0.0 if inside else OUTER_SPEED_FLOOR
            speed = floor_speed + (BOUNDARY_SPEED - floor_speed) * exp(-0.5 * (d / width)**2)
            speed = -speed
            vectors.append((speed * gx / length, speed * gy / length) if length else (0.0, 0.0))
            ds.append(d)
            weights.append(1 / d**2 if inside else d**2)
        total = sum(weights)
        field = tuple(sum(w * v[k] for w, v in zip(weights, vectors)) / total for k in range(2))
        if inside:
            return field
        fields.append(field)
        distances.append(sum(w * d for w, d in zip(weights, ds)) / total)
    weights = [1 / d**2 for d in distances]
    total = sum(weights)
    return tuple(sum(w * field[k] for w, field in zip(weights, fields)) / total
                 for k in range(2))


def velocity(x, y):
    """Rotate the field by a small, deterministic, smoothly varying angle."""
    u, v = base_velocity(x, y)
    noise = (sin(2.3 * x + 1.7 * y + 0.4)
             + 0.5 * sin(-3.1 * x + 2.6 * y + 1.8)
             + 0.25 * cos(4.7 * x + 0.9 * y - 0.6)) / 1.75
    angle = DIRECTION_NOISE_RADIANS * noise
    return cos(angle) * u - sin(angle) * v, sin(angle) * u + cos(angle) * v


def sample_points():
    """Bridson Poisson-disc sampling with a spatial index and fixed seed."""
    rng = Random(SAMPLE_SEED)
    xmin, xmax, ymin, ymax = BOUNDS
    radius = MIN_SAMPLE_DISTANCE
    cell_size = radius / sqrt(2)
    cells = {}
    points = []
    active = []

    def cell(x, y):
        return floor((x - xmin) / cell_size), floor((y - ymin) / cell_size)

    def add(x, y):
        cells[cell(x, y)] = (x, y)
        points.append((x, y))
        active.append((x, y))

    add(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
    while active:
        index = rng.randrange(len(active))
        x, y = active[index]
        for _ in range(30):
            angle = rng.uniform(0, 2 * pi)
            distance = radius * sqrt(rng.uniform(1, 4))
            px, py = x + distance * cos(angle), y + distance * sin(angle)
            if not (xmin <= px <= xmax and ymin <= py <= ymax):
                continue
            cx, cy = cell(px, py)
            if any((px - qx)**2 + (py - qy)**2 < radius**2
                   for i in range(cx - 2, cx + 3)
                   for j in range(cy - 2, cy + 3)
                   if (i, j) in cells
                   for qx, qy in [cells[i, j]]):
                continue
            add(px, py)
            break
        else:
            active[index] = active[-1]
            active.pop()
    return points


def generate(output=OUTPUT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as stream:
        stream.write("x y u v speed\n")
        for x, y in sample_points():
            u, v = velocity(x, y)
            speed = sqrt(u**2 + v**2)
            if speed < MIN_DISPLAY_SPEED:
                continue
            stream.write(f"{x:.10g} {y:.10g} {u:.10g} {v:.10g} {speed:.10g}\n")
            count += 1
    print(f"Wrote {output} ({count} visible vectors)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--data', type=Path, default=OUTPUT,
                        help='Generated PGFPlots vector table (not an input)')
    generate(parser.parse_args().output)


if __name__ == "__main__":
    main()
