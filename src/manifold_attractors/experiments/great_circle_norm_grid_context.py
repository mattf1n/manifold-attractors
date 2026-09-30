"""Alias for great_circle_norm_grid --context natural (same matched circles)."""
import runpy
import sys

if __name__ == '__main__':
    sys.argv[1:1] = ['--context', 'natural']
    runpy.run_module('manifold_attractors.experiments.great_circle_norm_grid', run_name='__main__')
