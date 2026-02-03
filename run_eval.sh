#!/bin/bash
# chmod +x run_eval.sh

# Configuration

OUTPUT_PATH="results/"

# Define models to test
declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen3-0.6B"
)

# Define tasks with their few-shot settings
# Format: "task_name:few_shot_count"
declare -a TASKS=(
    "greekmmlu:0"
    "greekmmlu:5"
)

echo "Starting evaluation for ${#MODELS[@]} model(s)"
# echo "Cache directory: $CACHE_DIR"
echo "Output path: $OUTPUT_PATH"
echo "================================================"

# Run evaluation for each model
for MODEL_NAME in "${MODELS[@]}"; do
    echo "Starting evaluation for model: $MODEL_NAME"
    echo "================================================"
    MODEL_SHORT_NAME="${MODEL_NAME##*/}"
    # Run evaluation for each task with its specific few-shot setting
    for task_config in "${TASKS[@]}"; do
        # Split task name and few-shot count
        IFS=':' read -r task_name few_shot <<< "$task_config"
        
        echo "Running $task_name with $few_shot few-shot examples..."
        TASK_OUTPUT_PATH="${OUTPUT_PATH}/${task_name}_${few_shot}shot/"
        mkdir -p "$TASK_OUTPUT_PATH"
        
        # Build and run the command
        cd "$WORK_DIR"
        accelerate launch --mixed_precision bf16 --num_processes 1 -m lm_eval \
          --model hf \
          --model_args "pretrained=$MODEL_NAME,parallelize=True" \
          --tasks "$task_name" \
          --batch_size 1 \
          --trust_remote_code \
          --num_fewshot "$few_shot" \
          --output_path "$TASK_OUTPUT_PATH" \
          --log_samples

        
        if [ $? -ne 0 ]; then
            echo "Error: Evaluation failed for task $task_name on model $MODEL_NAME"
            exit 1
        fi
        
        echo "Completed $task_name for $MODEL_NAME"
        echo "--------------------------------"
    done

    # rm -rf ~/.cache/huggingface/hub
    echo "Completed all tasks for model: $MODEL_NAME"
    echo "================================================"
done

echo "All evaluations completed successfully for all models!"



