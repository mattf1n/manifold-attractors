#!/bin/bash
#SBATCH --job-name=lang-globes
#SBATCH --account=swabhas_1625
#SBATCH --partition=shared
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/mfinlays/thesis-slides/lang_globes_%j.log
set -euo pipefail
cd ~/manifold-attractors
export PYTHONHASHSEED=0
.venv/bin/python ~/thesis-slides/lang_globes.py --index $1
