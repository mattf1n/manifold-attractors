#!/bin/bash
#SBATCH --job-name=trisphere
#SBATCH --account=swabhas_1625
#SBATCH --partition=shared
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/trisphere_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/trisphere.py --index $1 --end $2
