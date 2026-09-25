#!/bin/bash
#SBATCH --job-name=greekmmlu_reason_report
#SBATCH --account=michalis
#SBATCH --qos=guaranteed-michalis
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --output=greekmmlu_reasoning_report_%j.out
set -euo pipefail
: "${REASONING_RUN_ID:?Set REASONING_RUN_ID}"
exec /shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python \
    "/shared/home/mersin.konomi/greekmmlu generate/reasoning_eval.py" report --run-id "$REASONING_RUN_ID"
