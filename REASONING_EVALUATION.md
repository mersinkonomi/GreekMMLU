# GreekMMLU reasoning evaluation

This is separate from `boxed_20260908_v1`. The earlier results, prompts, raw answers and HTML report are preserved.

## Submitted run (2026-09-10)

Full run: `reasoning_20260910_v3`, Slurm array **29037**. It is gated on successful completion of all pilot jobs in array **29027** (`reasoning_20260910_pilot_v3`). Final report job **29038** runs after the full array finishes. The full benchmark does not use pilot samples or scores. Earlier pilot revisions are diagnostic failures, not benchmark results.

## Protocol

All six models are evaluated at 0 and 5 shots: IFM K2 Horizon 7B / 3.7B, Qwen3.5 2B / 4B, and their two Base counterparts. Full runs cover all 45 subjects and 16,632 test questions per model/shot setting (199,584 answers total).

The Greek prompt asks for reasoning, followed by a last line such as:

```text
Τελική απάντηση: \boxed{Β}
```

Only a valid choice in that final marked line is scored. Greek and Latin choice letters are normalized. Intermediate boxes, an unclosed native thinking block, and answers to a subsequently generated question do not count. The current question's reference answer is never included in its input.

K2 and post-trained Qwen use native chat templates with thinking enabled; K2 uses `reasoning_effort=high`. Base models receive raw prompts, since they do not have an instruction chat protocol. Five-shot examples are the first five dev examples, packed in one user turn; they demonstrate final answers without invented explanations.

Output budget: **32,768 generated tokens**, with a 49,152-token total context. Overlong prompts raise an error rather than silently truncating the question. Sampling uses temperature 1, top-p .95; Qwen uses top-k 20 and post-trained Qwen presence penalty 1.5. Other presence penalties are zero. BF16 inference uses vLLM, one engine per dedicated GPU. Exact settings and seeds are recorded in the manifest and raw-generation records.

The local `reasoning_vllm_compat.py` worker extension preserves K2's original grouped RMSNorm. Stock vLLM's generic Transformers replacement would normalize over the whole vector and change this model's mathematics. Checkpoints and shared library installations are not modified. The job explicitly adds the evaluation environment and CUDA tools to `PATH`; Qwen's unused image/video inputs are disabled.

This is not a guarantee that every model will reason correctly, follow the final-answer format or finish before its budget. Budget stops and missing answers are reported separately. Scores evaluate answer selection and format compliance, not explanation quality. Comparison with the previous 64-token direct run is descriptive: prompts, chat formatting, sampling, backend and budget also differ.

## Files

- `results/generative/<run-id>/manifest.json`: frozen configuration and source hashes.
- `results/generative/<run-id>/status/<index>.json`: job status.
- `results/generative/<run-id>/<shot>-shot/<model>/raw_generations.jsonl`: checkpointed generations, exact decoded response including control tokens, token IDs, token count, finish reason, prompt, subject and question ID.
- Nested `samples_<subject>_<timestamp>.jsonl`: scored questions with `doc`, `target`, full `resps`, normalized `filtered_resps`, and `exact_match`.
- Nested `results_<timestamp>.json`: aggregate and subject scores.
- `greekmmlu_reasoning_report.html`: English report, separate from the previous `greekmmlu_generative_report.html`.

SQLite response caches preserve the first stochastic completion for each tokenized prompt plus generation configuration. Interrupted runs can resume with the same run ID and unchanged source. Each finished response is committed immediately; requests still generating when a job dies may need regenerating. Raw JSONL contains `request_hash` so interrupted exports can be deduplicated. Full scored JSONLs are written after evaluation completes; raw-generation JSONL is available during generation.

## Running

```bash
cd '/shared/home/mersin.konomi/greekmmlu generate'
PYTHON=/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python
"$PYTHON" reasoning_eval.py prepare --run-id NEW_RUN_ID
sbatch --export=ALL,REASONING_RUN_ID=NEW_RUN_ID run_reasoning_eval.sh
# Record the returned array ID before starting the final report job:
"$PYTHON" reasoning_eval.py record-job --run-id NEW_RUN_ID --job-id ARRAY_JOB_ID
sbatch --dependency=afterany:ARRAY_JOB_ID --export=ALL,REASONING_RUN_ID=NEW_RUN_ID run_reasoning_report.sh
```

Indices 0–5 are 0-shot; 6–11 are 5-shot. Order within each block: K2 7B, K2 3.7B, Qwen 2B, Qwen 2B Base, Qwen 4B, Qwen 4B Base. The default array permits at most eight concurrent dedicated GPUs (`shard:10` each); actual concurrency depends on Slurm availability. Each task has a 72-hour wall-time limit. A separate `--pilot` manifest uses only one Mathematics and one World History example per setting and never enters the full-results report.

Refresh the report without rerunning inference:

```bash
"$PYTHON" reasoning_eval.py report --run-id NEW_RUN_ID
```

CPU regression tests (use the same offline cache environment as the job):

```bash
PYTHONPATH=lm-evaluation-harness "$PYTHON" reasoning_tasks/test_reasoning_tasks.py
PYTHONPATH=lm-evaluation-harness:. "$PYTHON" reasoning_tasks/test_reasoning_backend.py
```
