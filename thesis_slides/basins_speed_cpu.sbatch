#!/bin/bash
#SBATCH --job-name=basins-speed
#SBATCH --account=swabhas_1625
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/basins_speed_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=$HOME/thesis-slides .venv/bin/python ~/thesis-slides/basins_speed.py 24
