#!/usr/bin/env python3
"""Separate, resumable GreekMMLU reasoning evaluation; never overwrite boxed runs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from generative_eval import PROJECT, HARNESS, SPECS, atomic_json, now, run_root, sync_scheduler_status

TASKS = PROJECT / "reasoning_tasks"
DIRECT = PROJECT / "results/generative/boxed_20260908_v1"
MAX_NEW = 32768
MAX_CONTEXT = 49152


def source_hashes():
    paths = list((TASKS / "greekmmlu").iterdir())
    paths += list((HARNESS / "lm_eval/tasks/greekmmlu").iterdir())
    paths += [PROJECT / name for name in ("reasoning_backend.py", "reasoning_eval.py", "reasoning_vllm_compat.py", "run_reasoning_eval.sh")]
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def prepare(args):
    root = run_root(args.run_id)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Already prepared: {root}")
    runs = []
    for index in range(12):
        model, path = SPECS[index % 6]
        if not (Path(path) / "config.json").is_file():
            raise FileNotFoundError(path)
        shot = 0 if index < 6 else 5
        native = not model.endswith("-Base")
        sampling = dict(do_sample=True, temperature=1.0, top_p=0.95,
                        top_k=20 if model.startswith("Qwen/") else -1,
                        presence_penalty=1.5 if model.startswith("Qwen/") and native else 0.0,
                        repetition_penalty=1.0, max_gen_toks=MAX_NEW)
        output = root / f"{shot}-shot" / model.replace("/", "--")
        runs.append(dict(index=index, model=model, model_path=path, num_fewshot=shot,
                         status="prepared", job_id=None, results_dir=str(output),
                         raw_generation_path=str(output / "raw_generations.jsonl"),
                         cache_path=str(root / "response_cache" / f"{index}.sqlite"),
                         native_thinking=native, chat_template=native,
                         chat_template_args=({"reasoning_effort": "high"} if model.startswith("IFM/")
                                             else {"enable_thinking": True}) if native else {},
                         generation_kwargs=sampling))
    manifest = dict(
        run_id=args.run_id, created_at=now(), runs=runs, pilot=args.pilot,
        protocol=dict(mode="reasoning", max_gen_toks=MAX_NEW,
                      prompt_mode="native_chat_for_instruct_raw_for_base",
                      instruction="Explain the reasoning, then finish with Τελική απάντηση: \\boxed{Α/Β/Γ/Δ} using one valid choice.",
                      final_answer_marker="Τελική απάντηση:",
                      scoring="Final marked boxed choice only, outside closed native thinking; invalid/missing finals count as incorrect.",
                      sampling="One sample per question; temperature 1, top_p .95. Qwen top_k20; instruct Qwen presence_penalty1.5. IFM high reasoning effort.",
                      fewshot="Five dev examples in one user turn; demonstrations contain final answers, not fabricated reasoning.",
                      comparison_caveat="Compared with direct64, prompting, native chat, sampling, backend and token budget all change; not an isolated causal test of reasoning.",
                      raw_response="Full decoded completion including special tokens in resps; exact generated token IDs and finish reason in raw_generations.jsonl.",
                      runtime_compatibility="Preserve K2's original grouped RMSNorm in the vLLM Transformers backend; disable unused Qwen image/video inputs.",
                      cache="One stochastic completion per tokenized prompt and effective sampling configuration, durably cached across retries."),
        output_type="generate_until", metric="exact_match,boxed-extract", dtype="bfloat16",
        generation=dict(max_gen_toks=MAX_NEW, do_sample=True, temperature=1.0, top_p=0.95),
        max_length=MAX_CONTEXT, seed="0,1234,1234,1234", backend="reasoning_vllm",
        expected_subjects=2 if args.pilot else 45,
        expected_samples_per_run=2 if args.pilot else 16632,
        tasks=["greekmmlu_mathematics", "greekmmlu_world_history"] if args.pilot else ["greekmmlu"],
        limit=1 if args.pilot else None, direct_results_dir=str(DIRECT),
        likelihood_results_dir="/shared/home/mersin.konomi/lm_eval_results",
        task_sha256=source_hashes(), report_path=str(PROJECT / "greekmmlu_reasoning_report.html"),
        harness_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HARNESS, text=True).strip(),
    )
    atomic_json(root / "manifest.json", manifest)
    print(root)
    if not args.pilot:
        refresh_report(root)


def refresh_report(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("pilot"):
        return
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([sys.executable, str(PROJECT / "generate_greekmmlu_generative_report.py"),
                        "--results-dir", str(root), "--manifest", str(root / "manifest.json"),
                        "--output", manifest["report_path"], "--direct-results-dir", str(DIRECT),
                        "--likelihood-results-dir", manifest["likelihood_results_dir"]], check=True)


def record_job(args):
    root = run_root(args.run_id)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["array_job_id"] = args.job_id
    for item in manifest["runs"]:
        item.update(status="submitted", job_id=f"{args.job_id}_{item['index']}",
                    log_path=str(PROJECT / f"greekmmlu_reasoning_{args.job_id}_{item['index']}.out"))
    atomic_json(root / "manifest.json", manifest)
    refresh_report(root)


def validate(root, item, manifest):
    candidates = sorted(Path(item["results_dir"]).rglob("results_*.json"))
    if not candidates:
        raise RuntimeError("No saved aggregated result")
    path = candidates[-1]
    data = json.loads(path.read_text())
    count = sum(x["effective"] for x in data["n-samples"].values())
    if len(data["configs"]) != manifest["expected_subjects"] or count != manifest["expected_samples_per_run"]:
        raise RuntimeError(f"Incomplete evaluation: {len(data['configs'])} subjects, {count} samples")
    stamp = path.name.removeprefix("results_").removesuffix(".json")
    sample_files = sorted(path.parent.glob(f"samples_*_{stamp}.jsonl"))
    saved = sum(sum(1 for line in p.open(encoding="utf-8") if line.strip()) for p in sample_files)
    if len(sample_files) != manifest["expected_subjects"] or saved != count:
        raise RuntimeError(f"Raw sample artifacts incomplete: {len(sample_files)} files, {saved} rows")
    if not manifest["pilot"] and "exact_match,boxed-extract" not in data["groups"]["greekmmlu"]:
        raise RuntimeError("Expected final-answer group metric missing")
    return path, count


def worker(args):
    root = run_root(args.run_id)
    manifest = json.loads((root / "manifest.json").read_text())
    item = manifest["runs"][args.index]
    native = item["native_thinking"]
    os.environ["GREEKMMLU_REASONING_REQUIRE_THINKING_CLOSE"] = "1" if native else "0"
    sys.path.insert(0, str(HARNESS))
    import reasoning_backend  # register the local resumable vLLM backend
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.loggers import EvaluationTracker

    model_args = dict(pretrained=item["model_path"], dtype="bfloat16", trust_remote_code=True,
                      model_impl="transformers" if item["model"].startswith("IFM/") else "auto",
                      max_model_len=MAX_CONTEXT, max_gen_toks=MAX_NEW,
                      max_num_seqs=args.max_num_seqs, gpu_memory_utilization=0.88,
                      tensor_parallel_size=1, seed=1234, enforce_eager=True, generation_config="vllm",
                      raw_generation_path=item["raw_generation_path"], cache_path=item["cache_path"],
                      native_thinking=native, completion_batch_size=args.chunk_size,
                      chat_template_args=item["chat_template_args"],
                      enable_thinking=native)
    if item["model"].startswith("IFM/"):
        model_args["worker_extension_cls"] = "reasoning_vllm_compat.K2WorkerExtension"
    else:
        model_args["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    tracker = EvaluationTracker(output_path=item["results_dir"])
    results = simple_evaluate(
        model="reasoning_vllm", model_args=model_args, tasks=manifest["tasks"],
        task_manager=TaskManager(include_path=str(TASKS)),
        num_fewshot=item["num_fewshot"], batch_size="auto", device="cuda:0",
        apply_chat_template=native, fewshot_as_multiturn=False,
        gen_kwargs=item["generation_kwargs"], limit=manifest["limit"],
        log_samples=True, evaluation_tracker=tracker, use_cache=None,
        random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
        bootstrap_iters=1000 if manifest["pilot"] else 100000,
        verbosity="INFO")
    if results is None:
        raise RuntimeError("Evaluation returned no results")
    samples = results.pop("samples")
    tracker.save_results_aggregated(results=results, samples=samples)
    for task_name in results["configs"]:
        tracker.save_results_samples(task_name=task_name, samples=samples[task_name])
    path, count = validate(root, item, manifest)
    print(f"Validated {count} samples: {path}", flush=True)


def run(args):
    root = run_root(args.run_id)
    manifest = json.loads((root / "manifest.json").read_text())
    item = dict(manifest["runs"][args.index])
    status_path = root / "status" / f"{args.index}.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        validate(root, item, manifest)
        print("Already completed and validated; refusing duplicate generation.")
        return
    item.update(hostname=socket.gethostname(), started_at=now(),
                job_id=f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')}_{args.index}")
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        item["log_path"] = str(PROJECT / f"greekmmlu_reasoning_{os.environ['SLURM_ARRAY_JOB_ID']}_{args.index}.out")
    def update(state, **extra):
        item.update(status=state, updated_at=now(), **extra)
        atomic_json(status_path, item)
    try:
        update("loading")
        if manifest["task_sha256"] != source_hashes():
            raise RuntimeError("Evaluation source changed after preparation; use a fresh run ID")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Requires a Slurm allocation exposing exactly one GPU")
        free, total = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name(0)}: {free/2**30:.1f}/{total/2**30:.1f} GiB free", flush=True)
        if free < 0.90 * total:
            raise RuntimeError("vLLM reasoning needs a dedicated GPU (shard:10); device is already occupied")
        update("running", gpu=torch.cuda.get_device_name(0))
        print(f"{item['model']} | {item['num_fewshot']}-shot | {MAX_NEW} new tokens | pilot={manifest['pilot']}", flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--run-id", args.run_id,
                                 "--index", str(args.index), "--max-num-seqs", str(args.max_num_seqs),
                                 "--chunk-size", str(args.chunk_size)], cwd=PROJECT)
        if result.returncode:
            raise RuntimeError(f"Evaluation worker exited {result.returncode}; generated chunks remain cached")
        path, count = validate(root, item, manifest)
        update("completed", completed_at=now(), result_path=str(path), samples=count)
    except BaseException as error:
        update("failed", completed_at=now(), error=str(error))
        raise
    finally:
        try:
            refresh_report(root)
        except Exception as error:
            print(f"Report refresh failed (can retry without rerunning evaluation): {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "record-job", "run", "worker", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--index", type=int, default=0, choices=range(12))
    parser.add_argument("--job-id")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "record-job":
        if not args.job_id or not args.job_id.isdigit():
            parser.error("--job-id must be numeric")
        record_job(args)
    elif args.action == "run":
        run(args)
    elif args.action == "worker":
        worker(args)
    else:
        sync_scheduler_status(run_root(args.run_id))
        refresh_report(run_root(args.run_id))


if __name__ == "__main__":
    main()
