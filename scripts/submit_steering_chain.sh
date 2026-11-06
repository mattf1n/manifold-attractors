#!/usr/bin/env bash
# Submit the sharded adaptive-steering chain (run from the repo root).
# prepare -> calibrate -> 40 geometry/test shards -> collect -> 40 oracle shards -> final.
# nlp_hiprio caps a user at 8 GPUs; arrays are unthrottled so overflow runs on nlp.
# TARGET=el ./scripts/submit_steering_chain.sh runs the en-el experiment (default zh).
# MODEL=llama3.2-3b runs the smaller model; its FP64 probe fits any nlp GPU (A6000/A100).
# Peak RSS is <20GB, so small CPU/mem requests let jobs land beside other users' work.
set -euo pipefail
S=scripts/steering_stage.sbatch
export LOG=/scratch1/mfinlays/steering/logs
export TARGET=${TARGET:-zh} MODEL=${MODEL:-llama3.1-8b}
GPU="--gres=gpu:a100:1"  # 8B FP64 probe needs ~64GB
# c06-03 stalls its GPU intermittently (slower, but finishes); it is often the only free node.
[ "$MODEL" = llama3.1-8b ] || GPU="--gres=gpu:1"
export GPU_ARGS="$GPU --cpus-per-task=4 --mem=32G"
P=st; [ "$TARGET" = zh ] || P=st-$TARGET; [ "$MODEL" = llama3.1-8b ] || P=$P-3b
sub() { sbatch --parsable "$@"; }

p=$(sub --job-name=$P-prepare   --export=ALL,STAGE=prepare   $GPU --time=2:00:00  --output=$LOG/%x_%j.log $S)
c=$(sub --job-name=$P-calibrate --export=ALL,STAGE=calibrate --cpus-per-task=1 --mem=4G --time=12:00:00 --output=$LOG/%x_%j.log -d afterok:$p $S)
s=$(sub --job-name=$P-shard     --export=ALL,STAGE=shard     $GPU --time=4:00:00  --array=0-39 --output=$LOG/%x_%A_%a.log -d afterok:$c $S)
k=$(sub --job-name=$P-collect   --export=ALL,STAGE=collect   --mem=48G --time=2:00:00 --output=$LOG/%x_%j.log -d afterok:$s $S)
o=$(sub --job-name=$P-oracle    --export=ALL,STAGE=oracle    $GPU --time=4:00:00  --array=0-39 --output=$LOG/%x_%A_%a.log -d afterok:$k $S)
f=$(sub --job-name=$P-final     --export=ALL,STAGE=final     --mem=48G --time=2:00:00 --output=$LOG/%x_%j.log -d afterok:$o $S)
echo "prepare=$p calibrate=$c shard=$s collect=$k oracle=$o final=$f"
