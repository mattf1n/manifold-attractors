#!/usr/bin/env bash
#SBATCH --account=swabhas_1625
#SBATCH --partition=nlp,nlp_hiprio
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --open-mode=append
# One stage of the adaptive-steering table, selected by STAGE (see submit_steering_chain.sh).
# Submit from the repo root. Large files live on /scratch1; `data` is a symlink there.
#   STAGE=prepare|calibrate|shard|collect|oracle|final  -> sharded chain (owner "chain")
#   STAGE=mono                                          -> whole pipeline in one job (owner "mono")
# Only one owner may drive the root: a stage exits 3 if the other strategy holds it.
# TARGET=zh (default) or el selects the target language; MODEL=llama3.1-8b (default)
# or llama3.2-3b selects the model. Each combination has its own root/lock.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
module purge

export HF_HOME=/scratch1/mfinlays/hf
export TMPDIR=/scratch1/mfinlays/tmp
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="uv run --locked --extra compute python"
M=manifold_attractors.steering
TARGET=${TARGET:-zh}
case "$TARGET" in zh) SFX= ;; el) SFX=_el ;; *) echo "unknown TARGET"; exit 2 ;; esac
MODEL=${MODEL:-llama3.1-8b}
case "$MODEL" in llama3.1-8b) ;; llama3.2-3b) SFX=$SFX-$MODEL ;; *) echo "unknown MODEL"; exit 2 ;; esac
R=data/language_steering_confirmatory$SFX
LOCK=/scratch1/mfinlays/steering/.owner$SFX
mkdir -p "$TMPDIR"

claim() {  # claim <chain|mono>; chain wins ties only by arriving first
  exec 9>"$LOCK.lk"; flock 9
  if [ -s "$LOCK" ]; then
    read -r owner jobid < "$LOCK" || true
    if [ "$owner" = "$1" ] && [ "$1" = chain ]; then return 0; fi
    if [ "$owner" = chain ] || { [ "$owner" = mono ] && [ -n "${jobid:-}" ] &&
         squeue -h -j "$jobid" 2>/dev/null | grep -q .; }; then
      echo "root owned by $owner ${jobid:-}; exiting"; exit 3
    fi
  fi
  echo "$1 ${SLURM_JOB_ID}" > "$LOCK"
}
require_chain() { read -r owner _ < "$LOCK" 2>/dev/null || owner=none
  [ "$owner" = chain ] || { echo "root not owned by chain ($owner); exiting"; exit 3; }; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
case "${STAGE:?}" in
  prepare)
    claim chain; $PY -m $M.prepare --root $R --target $TARGET --model $MODEL ;;
  calibrate)  # CPU driver: each round runs the 6 (shard, method) grids as a GPU array
    claim chain; [ -f $R/prepared.json ] || { echo "prepare not done"; exit 4; }
    version() { $PY -c "import sys; from pathlib import Path
from manifold_attractors.steering.fixed import grid_definition
print(grid_definition(Path(sys.argv[1]))['version'])" $R; }
    for round in $(seq 0 16); do
      v=$(version)
      sbatch --wait --job-name=$SLURM_JOB_NAME-grid --export=ALL,STAGE=calgrid \
        ${GPU_ARGS:?} --time=${GRID_TIME:-3:00:00} --array=0-5 \
        --output="${LOG:?}/%x_%A_%a.log" \
        scripts/steering_stage.sbatch
      # Lustre can show this node a stale view of files other nodes just wrote.
      for i in $(seq 30); do
        n=$(ls $R/calibration/v$(printf %03d $v)/00[01]/*.npz 2>/dev/null | wc -l)
        [ "$n" -ge 6 ] && break; sleep 10
      done
      $PY -m $M.fixed select --root $R && break
      [ "$(version)" != "$v" ] || { echo "selection failed without expansion"; exit 5; }
      [ "$round" -lt 16 ] || { echo "expansion limit reached"; exit 5; }
    done ;;
  calgrid)  # array task 0..5: shard = T%2, method = additive|norm_preserving|constant_projection
    require_chain
    T=${SLURM_ARRAY_TASK_ID:?}; METHODS=(additive norm_preserving constant_projection)
    $PY -m $M.fixed calibrate --root $R --shard $((T % 2)) --method ${METHODS[$((T / 2))]} ;;
  shard)  # array task 0..39: 0-19 wiki_test, 20-39 fineweb_test
    require_chain
    T=${SLURM_ARRAY_TASK_ID:?}; D=wiki_test; [ "$T" -ge 20 ] && D=fineweb_test; S=$((T % 20))
    [ -f "$R/geometry/$D/complete_size500_$(printf %03d $S).json" ] ||
      $PY -m $M.geometry --root $R --dataset $D --shard $S
    $PY -m $M.fixed test --root $R --split $D --shard $S --adaptive . ;;
  collect)
    require_chain
    $PY -m $M.compute --stage collect --root $R
    $PY -m $M.oracle prepare --root $R ;;
  oracle)  # array task 0..39
    require_chain
    T=${SLURM_ARRAY_TASK_ID:?}; D=wiki_test; [ "$T" -ge 20 ] && D=fineweb_test
    $PY -m $M.oracle worker --root $R --dataset $D --shard $((T % 20)) ;;
  final)
    require_chain
    $PY -m $M.oracle collect --root $R
    $PY -m $M.verify --root $R ;;
  mono)
    claim mono
    $PY -m $M.prepare --root $R --target $TARGET --model $MODEL; $PY -m $M.compute --root $R
    $PY -m $M.oracle all --root $R; $PY -m $M.verify --root $R ;;
  *) echo "unknown STAGE"; exit 2 ;;
esac

if [ "$STAGE" = final ] || [ "$STAGE" = mono ]; then
  # Small final outputs only, plain cp (no -a/-p) into the group project dir.
  D=/project2/swabhas_1625/$USER/adaptive_steering_results$SFX
  mkdir -p "$D/linear_oracle"
  cp "$R"/summary.json "$R"/data_complete.json "$R"/final_validation.json "$D"/
  cp "$R"/linear_oracle/summary.json "$D"/linear_oracle/
fi
