#!/bin/bash
#SBATCH --job-name=greekmmlu_gen_report
#SBATCH --account=michalis
#SBATCH --qos=guaranteed-michalis
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --output=greekmmlu_generate_report_%j.out

set -euo pipefail
: "${GEN_RUN_ID:?Set GEN_RUN_ID to the prepared run identifier}"
exec /shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python \
    "/shared/home/mersin.konomi/greekmmlu generate/generative_eval.py" report \
    --run-id "$GEN_RUN_ID"
