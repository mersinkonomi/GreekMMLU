# GreekMMLU: boxed-answer generation

The active comparison is `boxed_20260908_v1`, evaluation array **27606**.
The final HTML build is job **27612**, with an `afterany:27606` dependency.
Each evaluation also refreshes the report when it exits.

Report: `greekmmlu_generative_report.html` (English, self-contained).
Results: `results/generative/boxed_20260908_v1/`.

## Models and array indices

| Model | 0-shot | 5-shot |
|---|---:|---:|
| IFM/K2-Horizon-7B | 0 | 6 |
| IFM/K2-Horizon-3.7B | 1 | 7 |
| Qwen/Qwen3.5-2B | 2 | 8 |
| Qwen/Qwen3.5-2B-Base | 3 | 9 |
| Qwen/Qwen3.5-4B | 4 | 10 |
| Qwen/Qwen3.5-4B-Base | 5 | 11 |

Slurm permits up to eight concurrent array jobs; each requests three GPU memory
shards (about 28.8 GB). Shard jobs can share a physical GPU. Actual concurrency
depends on free resources. The account/QoS is `michalis/guaranteed-michalis`.

## Protocol

- All 45 subjects and 16,632 test questions per run; no sample limit.
- `generate_until`, BF16, greedy decoding, at most 64 generated tokens.
- The same raw completion prompt protocol for all six checkpoints; no chat
  template or explicit reasoning mode. Five-shot runs use the five dev examples
  with boxed demonstration answers.
- A final valid `\boxed{Α}` / `\boxed{Β}` / `\boxed{Γ}` / `\boxed{Δ}` is extracted;
  lowercase and Latin equivalents are normalized. No valid box scores incorrect.
- `exact_match,boxed-extract`, aggregated by sample count.
- `max_length=8192`; a CPU audit of every prompt found maxima of 7,710 input
  tokens for K2 and 5,048 for Qwen in 5-shot, leaving room for all 64 output tokens.
- Batch size 4, with automatic OOM fallback to 2 then 1. Completed responses are
  retained in a per-run cache across retries.

This is a direct-answer generation benchmark, not a model-native reasoning-mode
evaluation. A model that emits reasoning or ignores the output format may fail
to produce a scorable answer. The report separates extraction failures and
reasoning markers from answer accuracy; it does not infer exact token-budget
exhaustion from decoded text. Historical likelihood results use a different
scoring method and are labelled separately.

## Saved responses and provenance

Final `samples_*.jsonl` files under each model's result directory contain the
question, target, prompt arguments, generated text before regex extraction
(`resps`), extracted response (`filtered_resps`) and score. These are decoded
harness responses: special tokens and stop markers may already have been removed.
They are not a token-ID-level generation trace.

While evaluation is running, generated responses are durably cached in
`response_cache/<array-index>_rank0.db`. The final per-question JSONL files are
written when that run completes. Every model/shot combination has a separate
cache and output directory.

`manifest.json` records local checkpoints, task-file SHA-256 hashes, source HEAD,
generation settings, output paths and job IDs. `task_changes.patch` preserves the
task changes relative to Git HEAD. Per-job state is saved under `status/`; stdout
is saved in both the Slurm log and `logs/` inside the run directory.

## Check progress and rebuild the report

```bash
cd "/shared/home/mersin.konomi/greekmmlu generate"
squeue -j 27606,27612

/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python \
  generative_eval.py report --run-id boxed_20260908_v1
```

The report command also reconciles Slurm terminal failures, including a process
killed before it could write its own final status. Missing/failed evaluations
remain explicitly incomplete.

## Start a separate future comparison

Use a new identifier; the current array is already submitted.

```bash
PYTHON=/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python
"$PYTHON" generative_eval.py prepare --run-id NEW_RUN_ID
sbatch --export=ALL,GEN_RUN_ID=NEW_RUN_ID run_eval.sh
# With the actual numeric job ID returned by sbatch:
"$PYTHON" generative_eval.py record-job --run-id NEW_RUN_ID --job-id ARRAY_JOB_ID
sbatch --dependency=afterany:ARRAY_JOB_ID \
  --export=ALL,GEN_RUN_ID=NEW_RUN_ID run_generative_report.sh
```
