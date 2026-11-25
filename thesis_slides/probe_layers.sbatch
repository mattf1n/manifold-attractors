#!/bin/bash
#SBATCH --job-name=probe-layers
#SBATCH --account=swabhas_1625
#SBATCH --partition=shared
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/probe_layers_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/probe_layers.py
