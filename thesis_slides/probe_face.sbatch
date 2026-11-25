#!/bin/bash
#SBATCH --job-name=probe-face
#SBATCH --account=swabhas_1625
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/probe_face_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/probe_face.py
