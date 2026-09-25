#!/bin/bash
#SBATCH --job-name=greek_eval
#SBATCH --account=michalis
#SBATCH --qos=guaranteed-michalis
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:5
#SBATCH --array=0-23%16
#SBATCH --output=greek_eval_%A_%a.out

# Generative boxed evaluation of the Greek task suite (greekmmlu protocol:
# generate_until, \boxed{Γ} extraction, exact_match,boxed-extract).
#
# PARALLELISM: one array index = one (model, task) pair, each with shard:5
# (~48 GB, half a GPU — TITAN requires shards, not whole GPUs):
#     index = model_index * n_tasks + task_index
# Default lists below are 4 models x 6 tasks = indices 0..23. The %16 suffix
# caps concurrent jobs (16 x 5 shards = ~8 full GPUs at once); lower it to be
# polite when others need the GPUs.
#
# Submit everything:      sbatch run_greek_eval.sh
# One model (model 2):    sbatch --array=12-17 run_greek_eval.sh
# One pair directly:      INDEX=12 bash run_greek_eval.sh
# All tasks of a model:   MODEL_INDEX=2 bash run_greek_eval.sh
# Lighter footprint:      sbatch --gres=shard:3 --export=ALL,BATCH_SIZE=4 run_greek_eval.sh
#
# Adapted from /datalake/datastore1/yang/el_llm/run_eval.sh for this cluster:
# shared offline HF cache, shard-based GPU allocation, no accelerate launch.

set -euo pipefail

PROJECT_DIR="/shared/home/mersin.konomi/greekmmlu generate"
HARNESS_DIR="$PROJECT_DIR/lm-evaluation-harness"
PYTHON="/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python"
HUB_DIR="/shared/models/huggingface/hub"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/results/greek_generative}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

# Models must have a complete local snapshot in the shared hub cache
# (checked below). To add an uncached model, download it first, or point
# directly at a local checkpoint directory instead of a repo id.
declare -a MODELS=(
    "ilsp/Meltemi-7B-Instruct-v1.5"
    "ilsp/Llama-Krikri-8B-Instruct"
    "meta-llama/Llama-3.2-3B"
    "Qwen/Qwen3-8B"
    # Not in the shared cache yet (download before enabling):
    # "ilsp/Meltemi-7B-v1"
    # "ilsp/Llama-Krikri-8B-Base"
)

# Format: "task_name:num_fewshot".
# greektruthfulqa_mc1 must stay 0-shot (the task has no few-shot split).
# If you change MODELS or TASKS, also update the #SBATCH --array range above.
declare -a TASKS=(
    "greekarc_easy:0"
    "greekarc_challenge:25"
    "greekhellaswag:10"
    "greekmmlu:5"
    "greektruthfulqa_mc1:0"
    "belebele_ell_Grek:8"
    # Optional non-boxed variants (loglikelihood / free-form BLEU-ROUGE):
    # "greektruthfulqa_mc2:0"
    # "greektruthfulqa_gen:0"
)

N_MODELS=${#MODELS[@]}
N_TASKS=${#TASKS[@]}
TOTAL=$((N_MODELS * N_TASKS))

# Select which (model, task) pairs this invocation runs.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" || -n "${INDEX:-}" ]]; then
    COMBINED="${SLURM_ARRAY_TASK_ID:-${INDEX}}"
    if (( COMBINED < 0 || COMBINED >= TOTAL )); then
        echo "Index $COMBINED out of range 0..$((TOTAL - 1)); nothing to do (edit MODELS/TASKS or the --array line)."
        exit 0
    fi
    INDICES=( "$COMBINED" )
elif [[ -n "${MODEL_INDEX:-}" ]]; then
    if (( MODEL_INDEX < 0 || MODEL_INDEX >= N_MODELS )); then
        echo "Model index $MODEL_INDEX out of range 0..$((N_MODELS - 1)); edit MODELS."
        exit 1
    fi
    INDICES=( )
    for ((t = 0; t < N_TASKS; t++)); do
        INDICES+=( $(( MODEL_INDEX * N_TASKS + t )) )
    done
else
    echo "No selection: submit as a Slurm array, or set INDEX (one model,task pair) or MODEL_INDEX (all tasks of one model)."
    exit 1
fi

# Same offline environment as the greekmmlu generative array jobs.
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE="$HUB_DIR"
export HF_MODULES_CACHE=/shared/home/mersin.konomi/cache/huggingface/modules
export HF_DATASETS_CACHE=/shared/models/huggingface/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_ROOT"

echo "Batch: $BATCH_SIZE | Max length: $MAX_LENGTH | Pairs this job: ${INDICES[*]}"

"$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("No CUDA device visible; run inside a GPU allocation (see README.md)")
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}; free {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
if free < 18 * 2**30:
    sys.exit("Less than 18 GiB free on the GPU; request more shards")
PY

cd "$HARNESS_DIR"

ANY_FAILED=0
for idx in "${INDICES[@]}"; do
    model_idx=$(( idx / N_TASKS ))
    task_idx=$(( idx % N_TASKS ))
    MODEL_NAME="${MODELS[$model_idx]}"
    task_config="${TASKS[$task_idx]}"
    task_name="${task_config%%:*}"
    few_shot="${task_config##*:}"

    # Resolve the model to its local snapshot so the run works fully offline.
    MODEL_TAG="${MODEL_NAME//\//--}"
    REPO_DIR="$HUB_DIR/models--$MODEL_TAG"
    if [[ ! -f "$REPO_DIR/refs/main" || ! -d "$REPO_DIR/snapshots" ]]; then
        echo "FAILED: model $MODEL_NAME is not in the shared hub cache ($REPO_DIR); download it first or use a local path."
        ANY_FAILED=1
        continue
    fi
    SNAPSHOT="$(cat "$REPO_DIR/refs/main")"
    MODEL_PATH="$REPO_DIR/snapshots/$SNAPSHOT"
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "FAILED: incomplete snapshot for $MODEL_NAME at $MODEL_PATH (no config.json)."
        ANY_FAILED=1
        continue
    fi

    MODEL_DIR="$OUT_ROOT/$MODEL_TAG"
    LOG_DIR="$MODEL_DIR/logs"
    TASK_DIR="$MODEL_DIR/$task_name"
    mkdir -p "$LOG_DIR" "$TASK_DIR/response_cache"

    echo
    echo "=== [$idx] $MODEL_NAME :: $task_name ($few_shot-shot) ==="
    batch="$BATCH_SIZE"
    while true; do
        log="$LOG_DIR/${task_name}_${few_shot}shot_batch${batch}.log"
        if "$PYTHON" -m lm_eval \
            --model hf \
            --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation=sdpa,max_length=$MAX_LENGTH" \
            --tasks "$task_name" \
            --num_fewshot "$few_shot" \
            --batch_size "$batch" \
            --device cuda:0 \
            --trust_remote_code \
            --seed 0,1234,1234,1234 \
            --use_cache "$TASK_DIR/response_cache" \
            --output_path "$TASK_DIR" \
            --log_samples 2>&1 | tee "$log"
        then
            echo "Completed $task_name for $MODEL_NAME"
            break
        fi
        if grep -qi "out of memory" "$log" && [[ "$batch" -gt 1 ]]; then
            batch=$((batch / 2))
            echo "CUDA OOM; retrying $task_name with batch_size=$batch (cached responses are kept)"
            continue
        fi
        echo "FAILED: $task_name on $MODEL_NAME; see $log"
        ANY_FAILED=1
        break
    done
done

if [[ "$ANY_FAILED" -ne 0 ]]; then
    echo "Some runs failed; check the logs under $OUT_ROOT"
    exit 1
fi
echo "All requested runs completed"
