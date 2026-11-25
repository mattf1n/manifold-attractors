#!/bin/bash
#SBATCH --job-name=probe-sphere
#SBATCH --account=swabhas_1625
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/probe_sphere_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/probe_sphere.py --shard $1 --shards 4
