# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2", "pandas==3.0.6", "pyarrow==25.0.1", "scipy==1.18.1"]
# ///
"""Reconstruct the historical bubble panels from frozen output coordinates.

Show the orthographic (x,z) single-embedding view and (x,y) paired views.
The first natural pair uses supplemental sample 5; all other examples retain
their original positions. In particular,
all five single-bubble raster bounds match the parquet z extrema (not y).
No forward pass, fitted projection, PCA, or changed scientific coordinates.
The missing original surface-shading code is replaced by a translucent gray
triangle mesh connecting neighboring *input* Fibonacci lattice points. Mesh
connectivity is display-only; all displayed vertices are frozen output data.
"""
import argparse
from pathlib import Path
from .legacy_style import TEXT_WIDTH_INCHES, FIGURE_DIR, setup, save, font_size, ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/bubble_coords.parquet')
    parser.add_argument('--extra-data', type=Path, default=ROOT / 'data/bubble_coords_extra.parquet')
    parser.add_argument('--output', '--figure', default=str(FIGURE_DIR / "bubble_coords.pdf"))
    args = parser.parse_args()
    setup()
    df = pd.concat([pd.read_parquet(args.data), pd.read_parquet(args.extra_data)], ignore_index=True)
    fig, axes = plt.subplots(3, 5, figsize=(TEXT_WIDTH_INCHES, 2.6))
    rows = [('single', 'z', '1 embedding'), ('pair', 'y', '2 embeddings'),
            ('synth_pair', 'y', '2 synthetic')]
    pair_order = [5, 1, 2, 3, 4]
    for row, (kind, vertical, title) in enumerate(rows):
        samples = pair_order if kind == 'pair' else range(5)
        for column, sample in enumerate(samples):
            ax = axes[row, column]
            sub = df[(df.kind == kind) & (df['sample'] == sample)]
            surface = sub[sub.i >= 0].sort_values('i')
            points = sub[sub.i < 0]
            # Triangulate the source sphere, never the folded output projection.
            faces = ConvexHull(surface[['fib_x', 'fib_y', 'fib_z']].to_numpy()).simplices
            xyz = surface[['x', 'y', 'z']].to_numpy()
            depth_axis = 1 if vertical == 'z' else 2
            faces = faces[np.argsort(xyz[faces, depth_axis].mean(axis=1))]
            projected = surface[['x', vertical]].to_numpy()
            # Smooth display-only surface lighting; no vertex positions are changed.
            # Orient triangle normals outward using the known source sphere.
            source = surface[['fib_x', 'fib_y', 'fib_z']].to_numpy()
            source_tri = source[faces]
            orientation = np.sign(np.einsum('ij,ij->i',
                np.cross(source_tri[:, 1] - source_tri[:, 0], source_tri[:, 2] - source_tri[:, 0]),
                source_tri.mean(axis=1)))
            tri_xyz = xyz[faces]
            face_normals = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
            face_normals *= orientation[:, None]
            normals = np.zeros_like(xyz)
            for corner in range(3):
                np.add.at(normals, faces[:, corner], face_normals)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            shade = 0.58 + 0.22 * np.abs(normals[:, depth_axis])
            mesh = Triangulation(projected[:, 0], projected[:, 1], triangles=faces)
            ax.tripcolor(mesh, shade, shading='gouraud', cmap='gray', norm=Normalize(0, 1),
                         rasterized=True, alpha=0.65)
            ax.scatter(points.x, points[vertical], marker='x', s=12, color='black',
                       linewidths=0.5, zorder=3)
            ax.scatter([0], [0], marker='+', s=12, color='black', linewidths=0.5, zorder=3)
            ax.set_aspect('equal', adjustable='datalim')
            ax.margins(0.06)
            ax.set_xticks([])
            ax.set_yticks([])
            if title and column == 0:
                ax.set_title(title, loc='left', fontsize=font_size('scriptsize'), pad=4)
    # Each remaining row is a section; leave room for its heading.
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.018, top=0.91,
                        wspace=0.05, hspace=0.25)
    save(args.output)


if __name__ == '__main__':
    main()
