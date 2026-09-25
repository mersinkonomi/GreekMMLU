<p align="center">
  <img src="greekmmlu.png" alt="GreekMMLU logo" width="520" />
</p>

# GreekMMLU

**GreekMMLU** is a **native-sourced** benchmark for evaluating massive multitask language understanding in **Greek**, built from **authentic Greek exam-style multiple-choice questions** (MCQ) rather than machine-translated English benchmarks.

- **21,805** questions across **45** subjects
- 4 high-level groups: **STEM**, **Humanities**, **Social Sciences**, **Other**
- Difficulty/education levels spanning **Primary → Secondary → University → Professional** (+ an N/A bucket)
- Public vs. private split for contamination-resistant evaluation: **16,857 public**/**4,948 private (leaderboard)**

## Links

- Our paper: https://www.arxiv.org/abs/2602.05150
- Dataset on Hugging Face: https://huggingface.co/datasets/dascim/GreekMMLU
- Private leaderboard: https://huggingface.co/spaces/yangzhang33/GreekMMLU-Leaderboard

## What makes GreekMMLU different?

Most “Greek MMLU” style evaluations rely on **machine translation** from English. Instead, GreekMMLU uses **original Greek content** sourced or authored from real educational/professional assessments, aiming to preserve:

- Greek morphology and punctuation
- Greek-specific cultural/institutional knowledge (e.g., Greek History, Greek Traditions)
- Realistic exam difficulty calibration

## Task format

- Multiple choice, **2–4 options**, **exactly one correct**.
- The harness prompt uses Greek option labels (**Α, Β, Γ, Δ**) for a fully native evaluation setup.

## Using GreekMMLU with LM Evaluation Harness (included)

This repo vendors a copy of **lm-evaluation-harness** under `lm-evaluation-harness/` with GreekMMLU task configs under:

- `lm-evaluation-harness/lm_eval/tasks/greekmmlu/`

The task group name is:

- `greekmmlu` (aggregates across STEM/Humanities/Social Sciences/Other)

### Install

From the repository root:

```bash
cd lm-evaluation-harness
pip install -e .
```

On this cluster, use the existing environment instead of installing anything:
`/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python`.

### Quickstart: run evaluation

One task, directly (run inside a GPU allocation):

```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=<model>,dtype=bfloat16,max_length=8192 \
  --tasks greekarc_easy \
  --num_fewshot 0 \
  --batch_size 4 \
  --device cuda:0 \
  --output_path ../results/quickstart \
  --log_samples
```

## The Greek generative task suite

All Greek tasks in this harness follow the same generative protocol: the model
sees the question with Greek-lettered choices (Α, Β, Γ, …) and must answer with
a boxed letter (`\boxed{Γ}`). The harness extracts the final valid box with a
regex, normalizes lowercase and Latin letters to Greek labels, and scores
`exact_match,boxed-extract`. Greedy decoding, at most 64 generated tokens, raw
completion prompt (no chat template).

| Task | Notes | Default few-shot |
|---|---|---|
| `greekmmlu` | aggregate group over all 45 subjects | 0 or 5 |
| `greekarc_easy` | 3–5 choices, labels Α–Ε | 0 |
| `greekarc_challenge` | 3–5 choices, labels Α–Ε | 25 |
| `greekhellaswag` | 4 endings, labels Α–Δ | 10 |
| `greektruthfulqa_mc1` | 2–13 choices, labels Α–Ν, choices shuffled per question; must stay **0-shot** | 0 |
| `belebele_ell_Grek` | reading comprehension over a FLORES passage, labels Α–Δ | 8 |

Non-generative variants kept alongside: `greektruthfulqa_mc2` (loglikelihood
probability mass) and `greektruthfulqa_gen` (free-form generation scored with
BLEU/ROUGE).

## Running the whole suite: `run_greek_eval.sh`

`run_greek_eval.sh` runs every (model, task) pair of the suite as its own
Slurm array job, each with **5 GPU-memory shards** (`--gres=shard:5`, ≈48 GB —
the TITAN cluster requires shards rather than whole GPUs). Splitting the work
this way lets the suite use as many GPUs as are free. With the default lists
(4 models × 6 tasks) that is 24 array indices:

```
index = model_index × n_tasks + task_index      # 0..23
```

The `%16` suffix in `#SBATCH --array=0-23%16` caps how many run at once
(16 × 5 shards ≈ 8 full GPUs concurrently). If fewer than 16 run, the rest are
just waiting for shards or a job limit — they start as others finish. Check
free capacity first:

```bash
sinfo -o "%N %G %t"                      # per-node GPUs/shards and state
squeue -h -t RUNNING -o "%b" | sort | uniq -c   # what is running right now
```

Submit everything:

```bash
cd "/shared/home/mersin.konomi/greekmmlu generate"
sbatch run_greek_eval.sh                      # 24 pairs, ≤16 at a time
sbatch --array=0-23%8 run_greek_eval.sh       # fewer concurrent GPUs
```

One model only (here: model index 2, i.e. its six tasks `12..17`):

```bash
sbatch --array=12-17 run_greek_eval.sh
```

Directly inside an interactive GPU allocation (one pair, or all tasks of one
model):

```bash
srun --account=michalis --qos=guaranteed-michalis --partition=gpu \
  --gres=shard:5 --mem=32G --cpus-per-task=8 --time=04:00:00 --pty bash
INDEX=12 bash run_greek_eval.sh        # one (model, task) pair
MODEL_INDEX=2 bash run_greek_eval.sh   # all six tasks of model 2, sequential
```

Smaller footprint (fits alongside other people's jobs):

```bash
sbatch --gres=shard:3 --export=ALL,BATCH_SIZE=4 run_greek_eval.sh
```

Models are resolved to their local snapshots under
`/shared/models/huggingface/hub` and everything runs offline, so a model must
already be in that cache (or be given as a local checkpoint path). To change
the model or task lists, edit `MODELS`/`TASKS` and the `#SBATCH --array` range
at the top of the script.

Results are written under `results/greek_generative/<org--model>/<task>/` —
lm_eval creates a timestamped subdirectory per run containing `results_*.json`
and, with `--log_samples`, one `samples_*.jsonl` per task with the raw
generations, extracted answers and scores. Per-task logs are kept in
`results/greek_generative/<org--model>/logs/`.

Behaviour details:

- Default `BATCH_SIZE=16` (the 5-shard allocation has ~48 GB); the batch is
  halved automatically on CUDA OOM down to 1, so an oversized default only
  costs one retry. Use `BATCH_SIZE=4` for 3-shard allocations. Greedy decoding
  makes
  results batch-invariant in exact arithmetic; set `BATCH_SIZE=4` if you want
  to match the batch-4 greekmmlu comparison runs bit-for-bit.
- Generated responses are cached per (model, task) in `response_cache/`;
  re-running a failed job reuses completed responses instead of regenerating
  them. Delete that directory if you change task files and need fresh
  generations.
- A failed pair does not stop the other array jobs; that job exits non-zero so
  Slurm reports it.
- `BATCH_SIZE`, `MAX_LENGTH` and `OUT_ROOT` can be overridden via environment
  variables.
- Every job uses the same offline environment as the greekmmlu generative
  jobs (shared HF cache, BF16, greedy, `max_length=8192`, seed `0,1234,1234,1234`).

Environment note: the datasets cache at `/shared/models/huggingface/datasets`
is read-only. All Greek tasks load from it offline, and the generative tasks
avoid writing to it (no `datasets.map()` in their `process_docs`). If you add
tasks that do use `.map()`, redirect `HF_DATASETS_CACHE` to a writable
directory first.

`run_eval.sh` + `generative_eval.py` are the managed comparison pipeline for
greekmmlu only (run ids, manifests, HTML reports) — see
[GENERATIVE_EVALUATION.md](GENERATIVE_EVALUATION.md).

## Subjects (high-level)

GreekMMLU includes 45 subjects, grouped into:

- **Humanities** (e.g., Art; Greek History; Greek Literature; Greek Mythology; Law; World Religions)
- **STEM** (e.g., Mathematics; Physics; Computer Science; Electrical Engineering; Medicine)
- **Social Sciences** (e.g., Economics; Education; Government & Politics; Modern Greek Language; Accounting)
- **Other** (e.g., Driving Rules; General Knowledge; Maritime Safety and Rescue Operations)

See the paper for the full taxonomy and educational-level breakdown.

## Citation

If you use GreekMMLU in your work, please cite the paper:

```bibtex
@article{zhang2026greekmmlu,
  title={GreekMMLU: A Native-Sourced Multitask Benchmark for Evaluating Language Models in Greek},
  author={Zhang, Yang and Konomi, Mersin and Xypolopoulos, Christos and Divriotis, Konstantinos and Skianis, Konstantinos and Nikolentzos, Giannis and Stamou, Giorgos and Shang, Guokan and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:2602.05150},
  year={2026}
}
```

## Notes on data & evaluation

- The dataset is built from **native Greek sources** and curated with quality control, including expert review.
- The benchmark is split into a **public** subset (released) and a **private** subset for leaderboard evaluation.
- Public data is hosted on Hugging Face (`dascim/GreekMMLU`). Please refer to the dataset card for license/terms.
