#!/bin/bash
#SBATCH --job-name=trisphere-speed
#SBATCH --account=swabhas_1625
#SBATCH --partition=shared
#SBATCH --cpus-per-task=20
#SBATCH --mem=110G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/trisphere_speed_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/trisphere_speed.py
