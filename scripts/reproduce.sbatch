#!/usr/bin/env bash
#SBATCH --job-name=paper-figures
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=reproduce-%j.log
set -euo pipefail
# Submit from repository root: sbatch scripts/reproduce.sbatch [iclr|goodfire] [figures|data]
# No partition or CPU/RAM override: use cluster GPU allocation defaults.
# Download permissions and LuaLaTeX/fonts must be configured before submission.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run --locked --extra compute reproduce run "${2:-figures}" --style "${1:-iclr}" --resume
