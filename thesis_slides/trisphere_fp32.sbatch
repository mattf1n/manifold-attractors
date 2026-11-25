#!/bin/bash
#SBATCH --job-name=trisphere-fp32
#SBATCH --account=swabhas_1625
#SBATCH --partition=shared
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/trisphere_fp32_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/trisphere_fp32.py --shard $1 --shards 4
