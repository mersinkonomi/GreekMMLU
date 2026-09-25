#!/bin/bash
#SBATCH --job-name=greekmmlu_reasoning
#SBATCH --account=michalis
#SBATCH --qos=guaranteed-michalis
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:10
#SBATCH --array=0-11%8
#SBATCH --output=greekmmlu_reasoning_%A_%a.out

set -euo pipefail
PROJECT_DIR="/shared/home/mersin.konomi/greekmmlu generate"
PYTHON="/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python"
export PATH="/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/mersin.konomi/cache/huggingface/modules
export HF_DATASETS_CACHE=/shared/home/mersin.konomi/cache/huggingface/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/lm-evaluation-harness${PYTHONPATH:+:${PYTHONPATH}}"
: "${REASONING_RUN_ID:?Set REASONING_RUN_ID to a prepared reasoning run}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-${ARRAY_ID:-}}"
: "${ARRAY_INDEX:?Submit a Slurm array or set ARRAY_ID in an allocation}"
cd "$PROJECT_DIR"
exec "$PYTHON" "$PROJECT_DIR/reasoning_eval.py" run --run-id "$REASONING_RUN_ID" \
    --index "$ARRAY_INDEX" --max-num-seqs "${MAX_NUM_SEQS:-16}" --chunk-size "${CHUNK_SIZE:-128}"
