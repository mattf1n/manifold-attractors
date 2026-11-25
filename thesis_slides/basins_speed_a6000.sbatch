#!/bin/bash
#SBATCH --job-name=basins-speed
#SBATCH --account=swabhas_1625
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/basins_speed_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=$HOME/thesis-slides .venv/bin/python ~/thesis-slides/basins_speed.py 32
