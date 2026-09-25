#!/usr/bin/env python3
"""Run and track the six-model GreekMMLU boxed-generation comparison."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parent
HARNESS = PROJECT / "lm-evaluation-harness"
TASKS = HARNESS / "lm_eval/tasks/greekmmlu"
MODEL_HOME = Path("/shared/home/mersin.konomi/models")
HUB = Path("/shared/models/huggingface/hub")
SPECS = [
    ("IFM/K2-Horizon-7B", str(MODEL_HOME / "IFM--K2-Horizon-7B")),
    ("IFM/K2-Horizon-3.7B", str(MODEL_HOME / "IFM--K2-Horizon-3.7B")),
    ("Qwen/Qwen3.5-2B", str(HUB / "models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc")),
    ("Qwen/Qwen3.5-2B-Base", str(HUB / "models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c")),
    ("Qwen/Qwen3.5-4B", str(HUB / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")),
    ("Qwen/Qwen3.5-4B-Base", str(HUB / "models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b")),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_root(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Use only letters, numbers, underscores and hyphens in run-id")
    return PROJECT / "results/generative" / run_id


def task_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(TASKS.iterdir()) if path.is_file()
    }


def prepare(args):
    root = run_root(args.run_id)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Run already prepared: {root}")
    for model, path in SPECS:
        if not (Path(path) / "config.json").is_file():
            raise FileNotFoundError(f"Missing local checkpoint: {model}: {path}")
    runs = []
    for index in range(12):
        model, path = SPECS[index % 6]
        shot = (0, 5)[index // 6]
        runs.append({
            "index": index, "model": model, "model_path": path,
            "num_fewshot": shot, "status": "prepared", "job_id": None,
            "results_dir": str(root / f"{shot}-shot" / model.replace("/", "--")),
        })
    manifest = {
        "run_id": args.run_id, "created_at": now(), "runs": runs,
        "protocol": "Raw completion; no chat template; direct boxed answer; not a reasoning-mode evaluation",
        "output_type": "generate_until", "metric": "exact_match,boxed-extract",
        "generation": {"do_sample": False, "temperature": 0.0, "max_gen_toks": 64},
        "dtype": "bfloat16", "seed": "0,1234,1234,1234",
        "expected_subjects": 45, "expected_samples_per_run": 16632,
        "max_length": 8192, "task_sha256": task_hashes(),
        "harness_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HARNESS, text=True).strip(),
        "likelihood_results_dir": "/shared/home/mersin.konomi/lm_eval_results",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "task_changes.patch").write_text(subprocess.check_output(
        ["git", "diff", "--", "lm_eval/tasks/greekmmlu"], cwd=HARNESS, text=True
    ), encoding="utf-8")
    atomic_json(root / "manifest.json", manifest)
    print(root)


def record_job(args):
    root = run_root(args.run_id)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["array_job_id"] = args.job_id
    for run in manifest["runs"]:
        run.update(status="submitted", job_id=f"{args.job_id}_{run['index']}",
                   log_path=str(PROJECT / f"greekmmlu_generate_{args.job_id}_{run['index']}.out"))
    atomic_json(root / "manifest.json", manifest)


def refresh_report(root):
    generator = PROJECT / "generate_greekmmlu_generative_report.py"
    if not generator.is_file():
        print("Report generator is being prepared; final report job will refresh it.", flush=True)
        return
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([
            sys.executable, str(generator), "--results-dir", str(root),
            "--manifest", str(root / "manifest.json"),
            "--output", str(PROJECT / "greekmmlu_generative_report.html"),
            "--likelihood-results-dir", "/shared/home/mersin.konomi/lm_eval_results",
        ], check=True)


def sync_scheduler_status(root):
    """Record scheduler failures even if Slurm killed the process before cleanup."""
    manifest = json.loads((root / "manifest.json").read_text())
    array_id = manifest.get("array_job_id")
    if not array_id:
        return
    accounting = subprocess.run([
        "sacct", "-n", "-P", "-j", str(array_id),
        "--format=JobID,State,ExitCode,Reason",
    ], capture_output=True, text=True, timeout=30)
    if accounting.returncode:
        print(f"Could not refresh Slurm accounting: {accounting.stderr.strip()}")
        return
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 4:
            continue
        job_id, state, exit_code, reason = fields[:4]
        match = re.fullmatch(re.escape(str(array_id)) + r"_(\d+)", job_id)
        if not match or int(match[1]) >= len(manifest["runs"]):
            continue
        index = int(match[1])
        status_path = root / "status" / f"{index}.json"
        record = json.loads(status_path.read_text()) if status_path.exists() else dict(manifest["runs"][index])
        record.update(slurm_state=state, slurm_exit_code=exit_code, updated_at=now())
        state = state.split()[0].rstrip("+")
        if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
            record.update(status="failed", error=f"Slurm {state}; exit {exit_code}; {reason}")
        elif state == "COMPLETED" and record.get("status") != "completed":
            record.update(status="failed", error="Slurm completed without a validated full result")
        elif state == "PENDING":
            record.update(status="pending", pending_reason=reason)
        atomic_json(status_path, record)


def evaluate(args):
    if not 0 <= args.index < 12:
        raise ValueError("Array index must be 0 through 11")
    root = run_root(args.run_id)
    manifest = json.loads((root / "manifest.json").read_text())
    status = dict(manifest["runs"][args.index])
    status.update(job_id=f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')}_{args.index}",
                  hostname=socket.gethostname(), started_at=now())
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        status["log_path"] = str(PROJECT / f"greekmmlu_generate_{os.environ['SLURM_ARRAY_JOB_ID']}_{args.index}.out")
    status_path = root / "status" / f"{args.index}.json"

    def update(state, **extra):
        status.update(status=state, updated_at=now(), **extra)
        atomic_json(status_path, status)

    try:
        update("loading")
        if task_hashes() != manifest["task_sha256"]:
            raise RuntimeError("Task files changed after run preparation; prepare a new run to preserve comparability")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Run inside a Slurm allocation exposing exactly one GPU")
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}; free {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
        if free < 18 * 2**30:
            raise RuntimeError("Less than 18 GiB GPU memory free in the allocated shards")
        status["gpu"] = torch.cuda.get_device_name(0)
        model_path = status["model_path"]
        output = Path(status["results_dir"])
        cache = root / "response_cache" / f"{args.index}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        batch = args.batch_size
        if batch < 1:
            raise ValueError("Batch size must be positive")
        while True:
            update("running", batch_size=batch)
            command = [
                sys.executable, "-m", "lm_eval", "--model", "hf",
                "--model_args", f"pretrained={model_path},dtype=bfloat16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation=sdpa,max_length={manifest['max_length']}",
                "--tasks", "greekmmlu", "--num_fewshot", str(status["num_fewshot"]),
                "--batch_size", str(batch), "--device", "cuda:0", "--trust_remote_code",
                "--seed", "0,1234,1234,1234", "--use_cache", str(cache),
                "--output_path", str(output), "--log_samples",
            ]
            attempt_log = root / "logs" / f"{args.index}_batch{batch}.log"
            attempt_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"Model: {status['model']}; shots: {status['num_fewshot']}; batch: {batch}", flush=True)
            print(f"Results: {output}", flush=True)
            had_oom = False
            with attempt_log.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=HARNESS, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    if "out of memory" in line.lower():
                        had_oom = True
                code = process.wait()
            if code == 0:
                results = list(output.rglob("results_*.json"))
                if len(results) != 1:
                    raise RuntimeError(f"Expected one result JSON; found {len(results)}")
                data = json.loads(results[0].read_text())
                if len(data.get("configs", {})) != 45:
                    raise RuntimeError("Evaluation did not include all 45 subjects")
                count = sum(item["effective"] for item in data["n-samples"].values())
                if count != manifest["expected_samples_per_run"]:
                    raise RuntimeError(f"Incomplete evaluation: {count} samples")
                if "exact_match,boxed-extract" not in data["groups"]["greekmmlu"]:
                    raise RuntimeError("Missing expected generative group metric")
                update("completed", completed_at=now(), result_path=str(results[0]), samples=count)
                break
            if had_oom and batch > 1:
                batch = max(1, batch // 2)
                print(f"Retrying after CUDA OOM with batch_size={batch}; cached responses are retained.", flush=True)
                continue
            raise RuntimeError(f"lm_eval exited {code}; see {attempt_log}")
    except BaseException as error:
        update("failed", error=str(error), completed_at=now())
        raise
    finally:
        try:
            refresh_report(root)
        except Exception as error:
            print(f"Report refresh failed: {error}; the final report job can retry.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "record-job", "run", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "record-job":
        if not args.job_id or not args.job_id.isdigit():
            parser.error("record-job requires a numeric --job-id")
        record_job(args)
    elif args.action == "run":
        evaluate(args)
    else:
        sync_scheduler_status(run_root(args.run_id))
        refresh_report(run_root(args.run_id))


if __name__ == "__main__":
    main()
