#!/bin/bash
#SBATCH --job-name=greekmmlu_generate
#SBATCH --account=michalis
#SBATCH --qos=guaranteed-michalis
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:3
#SBATCH --array=0-11%8
#SBATCH --output=greekmmlu_generate_%A_%a.out

set -euo pipefail

# 0..5: 0-shot; 6..11: 5-shot. Order: K2 7B, K2 3.7B,
# Qwen 2B, Qwen 2B Base, Qwen 4B, Qwen 4B Base.
# Prepare a named run with generative_eval.py before submitting this array.
PROJECT_DIR="/shared/home/mersin.konomi/greekmmlu generate"
PYTHON="/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python"
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/mersin.konomi/cache/huggingface/modules
export HF_DATASETS_CACHE=/shared/home/mersin.konomi/cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/lm-evaluation-harness${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

: "${GEN_RUN_ID:?Set GEN_RUN_ID to the prepared run identifier}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-${ARRAY_ID:-}}"
: "${ARRAY_INDEX:?Submit as a Slurm array, or set ARRAY_ID inside a GPU allocation}"

exec "$PYTHON" "${PROJECT_DIR}/generative_eval.py" run \
    --run-id "$GEN_RUN_ID" --index "$ARRAY_INDEX" \
    --batch-size "${BATCH_SIZE:-4}"


