#!/usr/bin/env python3
"""Build a self-contained English GreekMMLU boxed-generation report.

Run repeatedly while an evaluation is ongoing; absent results remain explicitly
missing/pending, and limited evaluations are never labelled complete.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from urllib.parse import quote


MODELS = (
    ("IFM/K2-Horizon-7B", "IFM · K2 Horizon", "#fb923c"),
    ("IFM/K2-Horizon-3.7B", "IFM · K2 Horizon", "#2dd4bf"),
    ("Qwen/Qwen3.5-2B", "Qwen · post-trained", "#60a5fa"),
    ("Qwen/Qwen3.5-2B-Base", "Qwen · base", "#818cf8"),
    ("Qwen/Qwen3.5-4B", "Qwen · post-trained", "#c084fc"),
    ("Qwen/Qwen3.5-4B-Base", "Qwen · base", "#f472b6"),
)
CATEGORIES = {"humanities": "Humanities", "social_sciences": "Social Sciences", "stem": "STEM", "other": "Other"}
METRIC = "exact_match,boxed-extract"
STDERR = "exact_match_stderr,boxed-extract"
BOX_RE = re.compile(r"\\boxed\s*\{\s*([ΑΒΓΔαβγδABCDabcd])\s*[.)]?\s*\}")
REASONING_RE = re.compile(r"<think>|</think>|\bThinking Process\b", re.IGNORECASE)
LABEL_MAP = dict(zip("ΑΒΓΔαβγδABCDabcd", "ΑΒΓΔ" * 4))
ROOT = Path(__file__).resolve().parent


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def delta(a, b):
    return "—" if a is None or b is None else f"{(b - a) * 100:+.2f} pp"


def score(entry):
    value, stderr = entry.get("value"), entry.get("stderr")
    if value is None:
        return '<span class="muted">—</span>'
    error = "SE unavailable" if stderr is None else f"±{stderr * 100:.2f} pp SE"
    count = entry.get("n")
    return f'<b>{pct(value)}</b><small>{error}{" · N=" + format(count, ",") if count is not None else ""}</small>'


def href(path, output):
    return quote(os.path.relpath(Path(path), output.parent))


def read_json(path, warnings):
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, ValueError) as error:
        warnings.append(f"Could not read {path}: {error}")
        return None


def subject_catalog(task_dir):
    """Only read simple leaf fields; avoid importing the model environment/YAML tags."""
    catalog = {}
    for path in sorted(task_dir.glob("greekmmlu_*.yaml")):
        fields = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(task|task_alias|tag):\s*(.+?)\s*$", line)
            if match:
                fields[match[1]] = match[2].strip("\"'")
        task = fields.get("task")
        if task:
            category = fields.get("tag", "").removeprefix("greekmmlu_").removesuffix("_tasks")
            catalog[task] = {"label": fields.get("task_alias", task).replace("_", " "), "category": category}
    return catalog


def identify(path, data):
    config = data.get("config") or {}
    model_args = config.get("model_args", "")
    pretrained = model_args.get("pretrained", "") if isinstance(model_args, dict) else str(model_args)
    identity = " ".join((str(path), str(pretrained), str(config.get("model_name", "")), str(data.get("model_name", ""))))
    # Prefer the longest model name so Base/3.7B cannot collide with other variants.
    model = next((m for m, _, _ in sorted(MODELS, key=lambda x: len(x[0]), reverse=True) if m.split("/")[-1] in identity), None)
    shot = config.get("num_fewshot")
    if shot is None:
        shots = {v for k, v in data.get("n-shot", {}).items() if k.startswith("greekmmlu_")}
        shot = next(iter(shots)) if len(shots) == 1 else None
    if shot is None:
        match = re.search(r"(?:^|[/_])(0|5)[-_]?shot(?:[/_]|$)", str(path))
        shot = int(match[1]) if match else None
    try:
        shot = int(shot)
    except (TypeError, ValueError):
        shot = None
    return model, shot


def discover(results_dir, metric, warnings, catalog, require_full_dataset=False):
    selected = {}
    matches = Counter()
    for path in sorted(results_dir.rglob("results*.json")):
        data = read_json(path, warnings)
        if not data or not isinstance(data.get("results"), dict):
            continue
        entries = {**data["results"], **data.get("groups", {})}
        if not any(task.startswith("greekmmlu") and isinstance(entry, dict) and metric in entry for task, entry in entries.items()):
            continue
        key = identify(path, data)
        if key[0] is None or key[1] not in (0, 5):
            continue
        populated = sum(metric in entries.get(task, {}) for task in catalog)
        counts = data.get("n-samples", {})
        full = (populated == len(catalog) == 45 and metric in entries.get("greekmmlu", {})
                and data.get("config", {}).get("limit") is None
                and all(counts.get(task, {}).get("effective") is not None
                        and counts[task]["effective"] == counts[task].get("original") for task in catalog))
        if (metric == "acc,none" or require_full_dataset) and (
            not full or set(data.get("configs", {})) != set(catalog)
            or sum(counts[task]["effective"] for task in catalog) != 16_632
            or any(data["configs"][task].get("dataset_path") != "dascim/GreekMMLU" for task in catalog)
        ):
            continue
        # Prefer full GreekMMLU evaluations over partial/different subject sets,
        # then use the timestamp embedded in the harness filename.
        order = (full, populated, path.name, path.stat().st_mtime)
        matches[key] += 1
        if key not in selected or order > selected[key][0]:
            selected[key] = (order, path, data)
    for key, count in matches.items():
        if count > 1:
            warnings.append(f"{key[0]} {key[1]}-shot: selected {selected[key][1]} from {count} candidate {metric} files, preferring full 45-subject runs and then the newest timestamp.")
    return {key: (path, data) for key, (_, path, data) in selected.items()}


def manifest_runs(manifest_path, warnings):
    if manifest_path is None or not manifest_path.exists():
        return {}, {}
    manifest = read_json(manifest_path, warnings) or {}
    records = manifest.get("runs", [])
    if isinstance(records, dict):
        records = list(records.values())
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        model = record.get("model", record.get("model_id"))
        shot = record.get("num_fewshot", record.get("shot", record.get("fewshot")))
        if model and shot is not None:
            indexed[(model, int(shot))] = record
    return manifest, indexed


def protocol_metadata(manifest):
    """Keep legacy string manifests compatible; new runs carry structured protocol."""
    recorded = manifest.get("protocol", {})
    protocol = dict(recorded) if isinstance(recorded, dict) else {"description": str(recorded)}
    protocol.setdefault("mode", manifest.get("mode", "direct"))
    generation = manifest.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}
    protocol.setdefault("max_gen_toks", generation.get("max_gen_toks", 8192 if protocol["mode"] == "reasoning" else 64))
    return protocol


def flatten_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        return [text for item in value for text in flatten_strings(item)]
    return []


def analyze_sample(sample, reasoning=False):
    texts = flatten_strings(sample.get("resps", []))
    response = texts[0] if texts else ""
    matches = BOX_RE.findall(response)
    filtered = flatten_strings(sample.get("filtered_resps", []))
    if reasoning:
        # The task enforces a final-answer marker after reasoning. A generic last
        # box may be only a tentative answer inside the reasoning, so never use
        # it to replace a missing/invalid saved filter result.
        selected = filtered[0] if filtered else "[invalid]"
        extracted = LABEL_MAP.get(selected, "[invalid]")
    else:
        extracted = LABEL_MAP[matches[-1]] if matches else "[invalid]"
        selected = filtered[0] if filtered else extracted
    choices = sample.get("doc", {}).get("choices", [])
    invalid = extracted == "[invalid]"
    kind = "unboxed" if invalid and r"\boxed" not in response else "malformed" if invalid else "valid"
    valid_labels = "ΑΒΓΔ"[:len(choices)] if choices else "ΑΒΓΔ"
    outside_choices = not invalid and extracted not in valid_labels
    value = number(sample.get("exact_match", sample.get(METRIC)))
    correct = None if value is None else value == 1
    return {
        "doc_id": sample.get("doc_id"), "question": sample.get("doc", {}).get("question", ""),
        "choices": choices, "target": sample.get("target", ""), "response": response,
        "extracted": extracted, "filtered": selected, "kind": kind,
        "multiple_boxes": len(matches) > 1, "outside_choices": outside_choices,
        "filter_mismatch": not reasoning and selected != extracted, "correct": correct,
        "missing_filter": reasoning and not filtered,
        "reasoning_marker": bool(REASONING_RE.search(response)),
    }


def sample_files(directory, result_path):
    files = list(directory.rglob("samples*.jsonl")) + list(directory.rglob("samples*.json"))
    if result_path:
        stamp = result_path.stem.removeprefix("results_")
        # Pair samples with the selected result, never mix distinct retry attempts.
        files = [path for path in files if path.stem.endswith(stamp)]
    else:
        by_task = {}
        for path in files:
            task = re.split(r"_\d{4}-\d{2}-\d{2}T", path.stem.removeprefix("samples_"))[0]
            if task not in by_task or path.name > by_task[task].name:
                by_task[task] = path
        files = list(by_task.values())
    return sorted(files)


def read_samples(directory, result_path, output, limit, warnings, reasoning=False, max_example_chars=8000):
    counts = Counter()
    per_subject = {}
    examples = {"correct": [], "incorrect": [], "invalid": []}
    files = sample_files(directory, result_path)
    links = []
    seen = set()
    for path in files:
        task = re.split(r"_\d{4}-\d{2}-\d{2}T", path.stem.removeprefix("samples_"))[0]
        if not task.startswith("greekmmlu_"):
            continue
        links.append({"task": task, "href": href(path, output)})
        try:
            with path.open(encoding="utf-8") as handle:
                records = enumerate(handle, 1) if path.suffix == ".jsonl" else enumerate(json.load(handle), 1)
                for lineno, line in records:
                    try:
                        sample = json.loads(line) if isinstance(line, str) else line
                        if not isinstance(sample, dict):
                            raise ValueError("sample is not an object")
                        if sample.get("filter", "boxed-extract") != "boxed-extract":
                            continue
                        identity = (task, sample.get("doc_id", lineno), sample.get("doc_hash", ""))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        row = analyze_sample(sample, reasoning)
                        row.update({"task": task, "source": href(path, output), "line": lineno})
                        bucket = per_subject.setdefault(task, Counter())
                        for counter in (counts, bucket):
                            counter["n"] += 1
                            counter[row["kind"]] += 1
                            for flag in ("multiple_boxes", "outside_choices", "filter_mismatch", "missing_filter", "reasoning_marker"):
                                counter[flag] += int(row[flag])
                            if row["correct"] is not None:
                                counter["scored"] += 1
                                counter["correct"] += int(row["correct"])
                        kind = "invalid" if row["kind"] != "valid" else "correct" if row["correct"] else "incorrect"
                        if len(examples[kind]) < limit:
                            row["response_chars"] = len(row["response"])
                            row["response_truncated"] = len(row["response"]) > max_example_chars
                            if row["response_truncated"]:
                                # Keep both the beginning and the final answer visible.
                                half = max_example_chars // 2
                                row["response"] = row["response"][:half] + "\n\n[… middle omitted in HTML preview; full response in linked JSONL …]\n\n" + row["response"][-half:]
                            examples[kind].append(row)
                    except (ValueError, TypeError, KeyError) as error:
                        counts["unreadable"] += 1
                        warnings.append(f"Unreadable sample at {path}:{lineno}: {error}")
        except (OSError, ValueError) as error:
            warnings.append(f"Could not read sample file {path}: {error}")
    return {"counts": dict(counts), "per_subject": per_subject, "examples": [row for rows in examples.values() for row in rows], "files": links}


def read_telemetry(path, output, warnings):
    """Aggregate explicit generation measurements, without guessing token caps."""
    counts = Counter()
    lengths = []
    seen_requests = set()
    if path is None or not path.exists():
        return {"counts": {}, "mean_tokens": None, "max_tokens": None, "href": None}
    try:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("generation record is not an object")
                except (ValueError, TypeError):
                    counts["unreadable"] += 1
                    continue  # The final line can be mid-write during a live refresh.
                request_hash = record.get("request_hash")
                if isinstance(request_hash, str) and request_hash:
                    if request_hash in seen_requests:
                        counts["duplicate_exports"] += 1
                        continue
                    seen_requests.add(request_hash)
                counts["n"] += 1
                tokens = number(record.get("generated_tokens"))
                if tokens is not None and tokens >= 0:
                    lengths.append(tokens)
                capped = record.get("capped", record.get("hit_token_limit"))
                if not isinstance(capped, bool) and record.get("finish_reason") is not None:
                    capped = record["finish_reason"] == "length"
                if isinstance(capped, bool):
                    counts["cap_known"] += 1
                    counts["capped"] += int(capped)
                final = record.get("has_final_answer")
                if isinstance(final, bool):
                    counts["final_known"] += 1
                    counts["no_final_answer"] += int(not final)
    except OSError as error:
        warnings.append(f"Could not read generation telemetry {path}: {error}")
    counts["token_count_known"] = len(lengths)
    return {"counts": dict(counts), "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
            "max_tokens": max(lengths) if lengths else None, "href": href(path, output)}


def run_entry(data, task, metric=METRIC, stderr=STDERR):
    entry = data.get("groups", {}).get(task, data.get("results", {}).get(task, {}))
    sample = data.get("n-samples", {}).get(task, {})
    n = number(sample.get("effective"))
    if n is None and task in data.get("group_subtasks", {}):
        def leaves(name, visited):
            if name in visited:
                return set()
            children = data.get("group_subtasks", {}).get(name)
            if not children:
                return {name} if name in data.get("n-samples", {}) else set()
            return set().union(*(leaves(child, visited | {name}) for child in children))
        values = [number(data.get("n-samples", {}).get(leaf, {}).get("effective")) for leaf in leaves(task, set())]
        n = sum(values) if values and all(value is not None for value in values) else None
    return {"value": number(entry.get(metric)), "stderr": number(entry.get(stderr)), "n": int(n) if n is not None else None}


def assemble(results_dir, output, manifest_path, likelihood_dir, example_limit, direct_dir=None, max_example_chars=8000):
    warnings = []
    catalog = subject_catalog(ROOT / "lm-evaluation-harness/lm_eval/tasks/greekmmlu")
    if len(catalog) != 45:
        warnings.append(f"Local task catalog contains {len(catalog)} subjects; 45 were expected.")
    discovered = discover(results_dir, METRIC, warnings, catalog)
    old = discover(likelihood_dir, "acc,none", warnings, catalog) if likelihood_dir else {}
    direct = discover(direct_dir, METRIC, warnings, catalog, require_full_dataset=True) if direct_dir else {}
    manifest, statuses = manifest_runs(manifest_path, warnings)
    protocol = protocol_metadata(manifest)
    reasoning = protocol["mode"] == "reasoning"
    for status_path in sorted((results_dir / "status").glob("*.json")):
        record = read_json(status_path, warnings)
        if record and record.get("model") is not None and record.get("num_fewshot") is not None:
            key = (record["model"], int(record["num_fewshot"]))
            statuses.setdefault(key, {}).update(record)
    runs = []
    for model, family, color in MODELS:
        for shot in (0, 5):
            status_record = dict(statuses.get((model, shot), {}))
            run_dir = results_dir / f"{shot}-shot" / model.replace("/", "--")
            if status_record.get("results_dir"):
                run_dir = Path(status_record["results_dir"])
                if not run_dir.is_absolute():
                    run_dir = results_dir / run_dir
            status_path = run_dir / "status.json"
            if status_path.exists():
                status_record.update(read_json(status_path, warnings) or {})
            path, data = discovered.get((model, shot), (None, {}))
            config = data.get("config", {})
            subject_entries = {task: run_entry(data, task) for task in catalog}
            populated = sum(entry["value"] is not None for entry in subject_entries.values())
            overall = run_entry(data, "greekmmlu")
            status = str(status_record.get("status", "missing")).lower()
            notes = []
            if path:
                complete = populated == len(catalog) == 45 and overall["value"] is not None
                if config.get("limit") is not None:
                    complete = False
                    notes.append(f"Limited evaluation: limit={config['limit']}")
                for task in catalog:
                    sample_count = data.get("n-samples", {}).get(task, {})
                    if sample_count.get("effective") is None or sample_count.get("effective") != sample_count.get("original"):
                        complete = False
                        notes.append("Missing sample counts or fewer effective than original examples.")
                        break
                shot_values = {data.get("n-shot", {}).get(task, data.get("configs", {}).get(task, {}).get("num_fewshot")) for task in catalog}
                if shot_values != {shot}:
                    complete = False
                    notes.append(f"Recorded few-shot settings do not all equal {shot}.")
                types = {data.get("configs", {}).get(task, {}).get("output_type") for task in catalog}
                if types != {"generate_until"}:
                    complete = False
                    notes.append("Result task metadata does not consistently record generate_until.")
                status = "complete" if complete else "partial"
                if overall["n"] is None:
                    known = [entry["n"] for entry in subject_entries.values()]
                    if all(value is not None for value in known):
                        overall["n"] = sum(known)
            elif status in {"complete", "completed", "success", "succeeded"}:
                status = "missing"
                notes.append("Job reports success but no matching generative result file was found.")
            diagnostics = read_samples(path.parent if path else run_dir, path, output, example_limit, warnings, reasoning, max_example_chars)
            if reasoning and status == "complete":
                failed_states = {"failed", "cancelled", "timeout", "out_of_memory", "node_fail", "preempted"}
                recorded_status = str(status_record.get("status", "")).lower()
                if recorded_status in failed_states:
                    status = recorded_status
                    notes.append("Aggregate scores exist, but the recorded run did not pass final artifact validation.")
                elif (diagnostics["counts"].get("n", 0) != overall["n"]
                      or len(diagnostics["files"]) != len(catalog)
                      or diagnostics["counts"].get("unreadable", 0)
                      or any(diagnostics["per_subject"].get(task, {}).get("n", 0) != entry["n"]
                             for task, entry in subject_entries.items())):
                    status = "partial"
                    notes.append("Full reasoning runs also require complete, readable per-question response artifacts; sample export is incomplete.")
            if diagnostics["counts"].get("n") and not path and status == "missing":
                status = "partial"
            if path and diagnostics["counts"].get("n", 0) != overall["n"]:
                notes.append("Logged sample coverage differs from the scored sample count; format rates apply only to available logs.")
            if diagnostics["counts"].get("filter_mismatch", 0):
                notes.append("Local regex and logged extraction differ on some examples; reported scores remain the harness scores.")
            if diagnostics["counts"].get("missing_filter", 0):
                notes.append("Some samples lack saved final-answer extraction; no answer is inferred from boxes inside their reasoning.")
            if status_record.get("error"):
                notes.append(str(status_record["error"]))
            old_path, old_data = old.get((model, shot), (None, {}))
            direct_path, direct_data = direct.get((model, shot), (None, {}))
            telemetry_path = status_record.get("raw_generation_path", status_record.get("telemetry_path"))
            if telemetry_path:
                telemetry_path = Path(telemetry_path)
                if not telemetry_path.is_absolute():
                    telemetry_path = results_dir / telemetry_path
            else:
                telemetry_path = next((p for p in (run_dir / "raw_generation.jsonl", run_dir / "raw_generations.jsonl") if p.exists()), None)
            telemetry = read_telemetry(telemetry_path, output, warnings)
            log_path = status_record.get("log_path", status_record.get("log"))
            if log_path:
                log_path = Path(log_path)
                if not log_path.is_absolute():
                    log_path = results_dir / log_path
            run = {
                "id": model.replace("/", "--") + f"-{shot}", "model": model,
                "label": model.split("/")[-1], "family": family, "color": color,
                "shot": shot, "status": status, "notes": notes, "overall": overall,
                "subjects": subject_entries, "categories": {key: run_entry(data, "greekmmlu_" + key) for key in CATEGORIES},
                "subject_count": populated, "diagnostics": diagnostics,
                "telemetry": telemetry,
                "result_href": href(path, output) if path else None,
                "log_href": href(log_path, output) if log_path else None,
                "job_id": status_record.get("job_id", "—"),
                "metadata": {"harness_git": data.get("git_hash"), "model_sha": config.get("model_sha"),
                    "dtype": config.get("model_dtype"), "batch_size": config.get("batch_size"),
                    "backend": config.get("model"), "model_args": config.get("model_args"),
                    "generation_kwargs": data.get("configs", {}).get(next(iter(catalog), ""), {}).get("generation_kwargs"),
                    "chat_template": data.get("chat_template"), "system_instruction": data.get("system_instruction"),
                    "duration_seconds": data.get("total_evaluation_time_seconds"), "timestamp": data.get("date"),
                    "num_parameters": config.get("model_num_parameters"), "limit": config.get("limit"),
                    "planned_protocol": {key: status_record[key] for key in
                        ("generation_kwargs", "generation", "sampling", "prompt_mode", "apply_chat_template", "chat_template_kwargs", "chat_template_args", "native_thinking", "max_gen_toks", "max_length")
                        if key in status_record}},
                "likelihood": {**run_entry(old_data, "greekmmlu", "acc,none", "acc_stderr,none"),
                    "href": href(old_path, output) if old_path else None,
                    "harness_git": old_data.get("git_hash"), "chat_template": old_data.get("chat_template") is not None},
                "direct": {**run_entry(direct_data, "greekmmlu"),
                    "href": href(direct_path, output) if direct_path else None},
            }
            runs.append(run)
    for shot in (0, 5):
        reference = None
        for model, _, _ in MODELS:
            _, data = discovered.get((model, shot), (None, {}))
            if data.get("task_hashes"):
                if reference is None:
                    reference = data["task_hashes"]
                elif reference != data["task_hashes"]:
                    warnings.append(f"Task hashes differ between some {shot}-shot runs; inspect protocol metadata before interpreting model differences.")
                    break
    return {"runs": runs, "catalog": catalog, "warnings": list(dict.fromkeys(warnings)),
            "run_id": manifest.get("run_id", results_dir.name), "results_dir": str(results_dir),
            "likelihood_enabled": likelihood_dir is not None, "direct_enabled": direct_dir is not None,
            "protocol": protocol, "max_example_chars": max_example_chars}


def render(report, output):
    runs, catalog = report["runs"], report["catalog"]
    protocol = report.get("protocol", {"mode": "direct", "max_gen_toks": 64})
    reasoning = protocol["mode"] == "reasoning"
    budget = esc(protocol.get("max_gen_toks", "not recorded"))
    title = "Reasoning and final answers" if reasoning else r"Answers inside \boxed{}"
    completed = sum(run["status"] == "complete" for run in runs)
    cards = []
    chart_rows = []
    for model, family, color in MODELS:
        zero, five = [run for run in runs if run["model"] == model]
        paired = "".join(f'<div><small>{run["shot"]}-shot · {esc(run["status"])}</small><strong>{pct(run["overall"]["value"])}</strong><em>{score(run["overall"]).split("<small>")[-1].removesuffix("</small>") if run["overall"]["value"] is not None else "Awaiting results"}</em></div>' for run in (zero, five))
        cards.append(f'<article class="model-card" style="--model:{color}"><span class="muted">{esc(family)}</span><h3>{esc(model.split("/")[-1])}</h3><div class="paired-score">{paired}</div><div class="delta-line"><span>5-shot − 0-shot</span><b>{delta(zero["overall"]["value"], five["overall"]["value"])}</b></div></article>')
        bars = "".join(f'<div class="track"><i class="{"zero" if run["shot"] == 0 else "five"}" style="width:{(run["overall"]["value"] or 0) * 100:.5f}%;background:{color}"></i></div>' for run in (zero, five))
        chart_rows.append(f'<div class="chart-row"><span>{esc(zero["label"])}</span><div>{bars}</div><b>{pct(zero["overall"]["value"])} → {pct(five["overall"]["value"])}</b></div>')
    statuses = []
    formats = []
    metadata = []
    source_links = []
    telemetry_rows = []
    for run in runs:
        notes = " ".join(run["notes"])
        links = " · ".join(f'<a href="{esc(run[key])}">{label}</a>' for key, label in (("result_href", "JSON"), ("log_href", "Job log")) if run[key]) or "—"
        statuses.append(f'<tr><th>{esc(run["label"])}</th><td>{run["shot"]}</td><td><span class="status {esc(run["status"])}">{esc(run["status"])}</span></td><td>{run["subject_count"]}/45</td><td>{run["overall"]["n"] if run["overall"]["n"] is not None else "—"}</td><td>{esc(run["job_id"])}</td><td>{links}</td><td>{esc(notes)}</td></tr>')
        counts = run["diagnostics"]["counts"]
        n = counts.get("n", 0)
        def rate(key):
            return f'{pct(counts.get(key, 0) / n)}<small>{counts.get(key, 0):,} / {n:,}</small>' if n else "—"
        invalid = counts.get("unboxed", 0) + counts.get("malformed", 0)
        formats.append(f'<tr><th>{esc(run["label"])}</th><td>{run["shot"]}</td><td>{n:,}</td><td>{pct(invalid / n) if n else "—"}</td><td>{rate("unboxed")}</td><td>{rate("malformed")}</td><td>{rate("multiple_boxes")}</td><td>{rate("outside_choices")}</td><td>{rate("reasoning_marker")}</td><td>{counts.get("unreadable", 0)}</td></tr>')
        telemetry = run["telemetry"]
        measured = telemetry["counts"]
        def measured_rate(key, denominator):
            total = measured.get(denominator, 0)
            return f'{pct(measured.get(key, 0) / total)}<small>{measured.get(key, 0):,} / {total:,}</small>' if total else "—"
        token_mean = "—" if telemetry["mean_tokens"] is None else f'{telemetry["mean_tokens"]:,.1f}'
        token_max = "—" if telemetry["max_tokens"] is None else f'{telemetry["max_tokens"]:,.0f}'
        raw_link = f'<a href="{esc(telemetry["href"])}">Full raw generation JSONL</a>' if telemetry["href"] else "—"
        duplicate_note = f'<small>{measured["duplicate_exports"]:,} duplicate exports excluded</small>' if measured.get("duplicate_exports") else ""
        telemetry_rows.append(f'<tr><th>{esc(run["label"])}</th><td>{run["shot"]}</td><td>{measured.get("n", 0):,}{duplicate_note}</td><td>{token_mean}<small>N={measured.get("token_count_known", 0):,}</small></td><td>{token_max}</td><td>{measured_rate("capped", "cap_known")}</td><td>{measured_rate("no_final_answer", "final_known")}</td><td>{measured.get("unreadable", 0):,}</td><td>{raw_link}</td></tr>')
        meta = run["metadata"]
        short = {key: value for key, value in meta.items() if key not in {"chat_template", "model_args"}}
        metadata.append(f'<details><summary>{esc(run["label"])} · {run["shot"]}-shot · {esc(run["status"])} · {esc(meta.get("backend") or "backend pending")}</summary><pre>{esc(json.dumps(short, ensure_ascii=False, indent=2))}</pre><details><summary>Model arguments and chat template</summary><pre>{esc(json.dumps({"model_args": meta["model_args"], "chat_template": meta["chat_template"]}, ensure_ascii=False, indent=2))}</pre></details></details>')
        items = "".join(f'<li><a href="{esc(file["href"])}">{esc(catalog.get(file["task"], {}).get("label", file["task"]))}</a></li>' for file in run["diagnostics"]["files"])
        source_links.append(f'<details><summary>{esc(run["label"])} · {run["shot"]}-shot · {len(run["diagnostics"]["files"])} sample files</summary>{"<p>" + raw_link + "</p>" if telemetry["href"] else ""}<ul class="file-list">{items or "<li>No sample files yet.</li>"}</ul></details>')
    category_panels = []
    for category, label in CATEGORIES.items():
        rows = []
        for model, _, _ in MODELS:
            zero, five = [run for run in runs if run["model"] == model]
            z, f = zero["categories"][category], five["categories"][category]
            rows.append(f'<tr><th>{esc(zero["label"])}</th><td class="score">{score(z)}</td><td class="score">{score(f)}</td><td>{delta(z["value"], f["value"])}</td></tr>')
        count = sum(entry["category"] == category for entry in catalog.values())
        category_panels.append(f'<article class="panel"><h3>{label}</h3><p class="muted">{count} subjects · sample-weighted score</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>0-shot</th><th>5-shot</th><th>Change</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></article>')
    subject_headers = "".join(f'<th colspan="3" style="border-top:3px solid {color}">{esc(model.split("/")[-1])}</th>' for model, _, color in MODELS)
    subject_subheaders = "".join(f'<th data-column="{2+i*3}" data-sort="number">0-shot</th><th data-column="{3+i*3}" data-sort="number">5-shot</th><th data-column="{4+i*3}" data-sort="number">Δ pp</th>' for i in range(len(MODELS)))
    subject_rows = []
    for task, item in sorted(catalog.items(), key=lambda pair: (pair[1]["category"], pair[1]["label"])):
        cells = []
        for model, _, _ in MODELS:
            pair = [run for run in runs if run["model"] == model]
            z, f = [run["subjects"][task] for run in pair]
            for run, entry in zip(pair, (z, f)):
                counts = run["diagnostics"]["per_subject"].get(task, {})
                invalid = counts.get("unboxed", 0) + counts.get("malformed", 0)
                format_info = f'<small>Invalid: {pct(invalid / counts["n"])} ({invalid}/{counts["n"]})</small>' if counts.get("n") else ""
                cells.append(f'<td class="score" data-value="{entry["value"] if entry["value"] is not None else ""}">{score(entry)}{format_info}</td>')
            diff = f["value"] - z["value"] if f["value"] is not None and z["value"] is not None else None
            cells.append(f'<td class="num" data-value="{diff if diff is not None else ""}">{delta(z["value"], f["value"])}</td>')
        subject_rows.append(f'<tr data-name="{esc(item["label"].lower())}" data-category="{esc(item["category"])}"><td>{esc(CATEGORIES.get(item["category"], item["category"]))}</td><th>{esc(item["label"])}</th>{"".join(cells)}</tr>')
    likelihood = ""
    if report["likelihood_enabled"]:
        rows = []
        for run in runs:
            old = run["likelihood"]
            source = f'<a href="{esc(old["href"])}">Previous JSON</a>' if old["href"] else "No matching previous result"
            rows.append(f'<tr><th>{esc(run["label"])}</th><td>{run["shot"]}</td><td class="score">{score(old)}</td><td class="score">{score(run["overall"])}</td><td>{delta(old["value"], run["overall"]["value"])}</td><td>{source}</td></tr>')
        likelihood = f'<section><h2>Previous likelihood evaluation · different protocol</h2><p>The earlier score selects the most likely answer label (<code>acc,none</code>). This run generates text and requires a regex-extractable boxed answer (<code>{METRIC}</code>), so formatting failures affect the score. Prompts, chat formatting, task revisions, and decoding can also differ. Differences below describe the two evaluations; they do not isolate a change in model knowledge or establish statistical significance. Only full likelihood evaluations with the same 45 subjects and 16,632 test questions qualify; the newest matching file is selected. Sample counts and source files are shown so coverage can be checked.</p><div class="table-panel"><table><thead><tr><th>Model</th><th>Shot</th><th>Earlier likelihood</th><th>Boxed generation</th><th>Generation − likelihood</th><th>Source</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
    direct = ""
    if report.get("direct_enabled"):
        rows = []
        for run in runs:
            baseline = run["direct"]
            source = f'<a href="{esc(baseline["href"])}">Direct-answer JSON</a>' if baseline["href"] else "No matching full baseline"
            current = run["overall"] if run["status"] == "complete" else {}
            rows.append(f'<tr><th>{esc(run["label"])}</th><td>{run["shot"]}</td><td class="score">{score(baseline)}</td><td class="score">{score(current)}</td><td>{delta(baseline["value"], current.get("value"))}</td><td>{source}</td></tr>')
        direct = f'<section><h2>Previous direct-answer generation · separate baseline</h2><p>The earlier boxed-answer evaluation used raw prompts, greedy decoding and a 64-token output budget. This evaluation allows explanation before a final answer and uses the protocol recorded below. Differences also include prompt formatting, output budget, and any model-specific decoding settings; they cannot be attributed to reasoning alone. Baselines must cover the same 45 subjects and 16,632 test questions. Current-run comparisons appear only after a full run completes.</p><div class="table-panel"><table><thead><tr><th>Model</th><th>Shot</th><th>Direct answer</th><th>Current full run</th><th>Current − direct</th><th>Source</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
    if reasoning:
        format_description = r'“Unboxed” means an invalid saved extraction with no <code>\boxed</code> marker. “Missing / malformed final answer” means a box marker appears but the saved final-answer extractor did not accept a final label. Their sum is the invalid-extraction rate. Boxes within reasoning are not accepted simply because they are the last box. The saved <code>filtered_resps</code> is authoritative; the report never re-scores a tentative boxed answer from the reasoning.'
        budget_description = f'The output budget is {budget} tokens per question. An unfinished response or missing final answer receives no credit. Explicit generation telemetry below separates output-budget stops from other missing-answer cases; token counts are never inferred from response length in characters.'
        extraction_description = r'Prompts allow an explanation and request a final line containing the answer label inside <code>\boxed{Α}</code>. The task extractor requires the final-answer marker after the reasoning. Greek Α–Δ and Latin A–D, including lowercase, are normalized to uppercase Greek. Saved filtered responses are authoritative. Missing or malformed final answers receive no credit; there is no fallback that guesses an answer from reasoning prose.'
        method_description = f'The planned protocol allows up to {budget} output tokens. K2 Horizon and post-trained Qwen models use their native chat formatting; Qwen Base models use raw completion prompts. Native thinking may be enabled through the model template where supported. Sampling and chat-template options can differ by model and are recorded in run metadata below. The score measures the final answer, not the quality or faithfulness of the explanation.'
        telemetry_section = '<section><h2>Generation length and completion</h2><p>These measurements come from the append-only raw-generation telemetry, including runs still in progress. Counts describe unique recorded generations, not completed scored evaluations. Duplicate exports caused by interrupted writes are excluded using <code>request_hash</code>. “Budget stop” uses explicit cap / finish-reason metadata; “No final answer” uses the recorded final-answer validity. Unknown measurements appear as —. An unreadable final line may be an in-progress write during report refresh.</p><div class="table-panel"><table><thead><tr><th>Model</th><th>Shot</th><th>Recorded N</th><th>Mean generated tokens</th><th>Longest output</th><th>Budget stop</th><th>No final answer</th><th>Unreadable lines</th><th>Source</th></tr></thead><tbody>' + "".join(telemetry_rows) + '</tbody></table></div></section>'
    else:
        format_description = r'“Unboxed” means no <code>\boxed</code> marker; “malformed” means the marker appears but no box matches the configured regex. Their sum is the invalid-extraction rate. Multiple valid boxes are accepted using the last matching label.'
        budget_description = f'A model can begin a reasoning-style response and fail to emit a box within the {budget}-token budget. Such outputs receive no credit in this direct-answer protocol. This is not a full reasoning evaluation, and this report does not infer exact token-limit hits from response length in characters.'
        extraction_description = r'Prompts request one Greek answer label inside <code>\boxed{Α}</code>. Extraction accepts Greek Α–Δ and Latin A–D, including lowercase; it normalizes them to uppercase Greek and chooses the last matching box. Missing or malformed answers receive no credit. There is no fallback that guesses an answer from unboxed prose.'
        method_description = f'The planned protocol uses raw text completion for all six models, without a chat template or reasoning instruction, BF16, greedy decoding, and a {budget}-token output cap. This is a direct-answer generation evaluation. The exact saved generation configuration and run-level model arguments are recorded below; any deviations should be checked there.'
        telemetry_section = ""
    warnings = "".join(f'<li>{esc(message)}</li>' for message in report["warnings"])
    warning_section = f'<details class="warning"><summary>{len(report["warnings"])} data / protocol notes</summary><ul>{warnings}</ul></details>' if warnings else ""
    sample_data = [{"id": run["id"], "label": f'{run["label"]} · {run["shot"]}-shot', "samples": run["diagnostics"]["examples"]} for run in runs]
    safe_json = json.dumps(sample_data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return HTML.substitute(
        generated=esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")), run_id=esc(report["run_id"]), completed=completed,
        cards="".join(cards), chart="".join(chart_rows), statuses="".join(statuses), formats="".join(formats),
        categories="".join(category_panels), subject_headers=subject_headers, subject_subheaders=subject_subheaders,
        subject_rows="".join(subject_rows), subject_count=len(catalog), metadata="".join(metadata), sources="".join(source_links),
        likelihood=likelihood, warnings=warning_section, sample_data=safe_json, results_dir=esc(report["results_dir"]),
        direct=direct, telemetry=telemetry_section, title=esc(title),
        protocol_label="Reasoning and final-answer generation" if reasoning else "Boxed generation",
        format_description=format_description, budget_description=budget_description,
        malformed_label="Missing / malformed final answer" if reasoning else "Malformed",
        extraction_description=extraction_description, method_description=method_description,
        protocol_json=esc(json.dumps(protocol, ensure_ascii=False, indent=2)),
        max_example_chars=report.get("max_example_chars", 8000),
    )


HTML = Template(r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GreekMMLU · $protocol_label · K2 Horizon × Qwen3.5</title>
<style>
:root{--bg:#06101c;--panel:#0d1b2b;--line:#213952;--text:#ecf5ff;--soft:#c4d3e5;--muted:#91a7c0;--green:#5ee0a0}
*{box-sizing:border-box}body{margin:0;color:var(--text);background:radial-gradient(circle at 8% -5%,#173c60 0,transparent 30rem),radial-gradient(circle at 93% 8%,#402653 0,transparent 28rem),var(--bg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:#7dd3fc}a:hover{text-decoration:none}.shell{width:min(1660px,calc(100% - 36px));margin:auto}.hero{padding:60px 0 25px}.eyebrow{color:#7dd3fc;font-size:.78rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}h1{margin:14px 0;font-size:clamp(2.4rem,5vw,4.8rem);line-height:1.02;letter-spacing:-.055em}h2{margin:0 0 8px;font-size:1.6rem;letter-spacing:-.025em}h3{margin:4px 0 10px}p{color:var(--soft)}.hero>p{max-width:920px}.pills{display:flex;flex-wrap:wrap;gap:9px}.pill,.status{padding:5px 10px;border:1px solid var(--line);border-radius:99px;font-size:.76rem;background:#112338}.pill.ok,.status.complete{color:#b7f7cd;background:#102c22;border-color:#276a47}.status.failed,.status.timeout,.status.cancelled{color:#fda4af;border-color:#923a51}.status.running,.status.partial{color:#fde68a;border-color:#867121}section{padding:24px 0}.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.model-card,.panel,.table-panel,details{border:1px solid var(--line);border-radius:16px;background:linear-gradient(145deg,rgba(17,36,58,.97),rgba(9,22,38,.97))}.model-card{border-left:4px solid var(--model);padding:20px}.paired-score{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:20px 0 14px}.paired-score>div{background:#081522;border:1px solid var(--line);border-radius:12px;padding:12px}.paired-score strong{display:block;font-size:2.3rem;letter-spacing:-.04em}.paired-score small,.paired-score em{display:block;font-size:.74rem;color:var(--muted);font-style:normal}.delta-line{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding-top:12px;color:var(--soft)}.panel{padding:20px}.muted,small{color:var(--muted)}small{display:block;font-size:.7rem}.chart-row{display:grid;grid-template-columns:175px 1fr 185px;gap:16px;align-items:center;margin:18px 0;font-size:.86rem}.chart-row>b{text-align:right}.track{height:8px;border-radius:10px;background:#06111e;margin:5px 0;overflow:hidden}.track i{display:block;height:100%;border-radius:inherit}.track .zero{opacity:.43}.category-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}table{width:100%;border-collapse:collapse;font-size:.78rem}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--soft);font-weight:650}thead th{background:#0a1929;white-space:nowrap}.table-panel{overflow:auto}.table-scroll{overflow:auto;max-height:780px}.score,.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.score b{display:block;color:var(--text)}.subjects{min-width:2300px}.subjects thead th{position:sticky;top:0;z-index:2}.subjects thead tr:nth-child(2) th{top:43px}.subjects thead th[data-sort]{cursor:pointer}.subjects tbody tr:hover{background:#142b43}.controls{display:flex;gap:10px;flex-wrap:wrap;padding:14px}input,select,button{padding:9px 12px;color:var(--text);background:#071421;border:1px solid var(--line);border-radius:9px;font:inherit;font-size:.84rem}input{flex:1;min-width:220px}button{cursor:pointer}.tools{margin-left:auto}details{margin:10px 0;padding:13px 17px}summary{cursor:pointer;color:var(--soft);font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;font-size:.79rem;line-height:1.65}code{color:#bae6fd}.file-list{columns:3;font-size:.78rem}.warning{border-color:#7b6632}.sample-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.sample-response{background:#071421;padding:16px;border-radius:10px}.sample-meta{color:var(--muted);font-size:.8rem}footer{border-top:1px solid var(--line);padding:30px 0 45px;margin-top:25px;color:var(--muted);font-size:.78rem}[hidden]{display:none!important}
@media(max-width:1050px){.model-grid{grid-template-columns:1fr 1fr}.chart-row{grid-template-columns:150px 1fr 150px}.category-grid{grid-template-columns:1fr}}@media(max-width:700px){.shell{width:calc(100% - 22px)}.model-grid,.sample-grid{grid-template-columns:1fr}.chart-row{grid-template-columns:1fr}.chart-row>b{text-align:left}.file-list{columns:1}.hero{padding-top:30px}}
@media print{:root{--bg:white;--text:#111827;--soft:#263648;--muted:#566474;--line:#ccd6e0}body{background:white}.shell{width:100%}.model-card,.panel,.table-panel,details,.paired-score>div,.sample-response{background:white}.tools,.controls{display:none}.table-scroll{max-height:none}.subjects thead th{position:static}thead th{background:#eef2f6}section{break-inside:avoid}}
</style></head><body><header class="hero shell"><div class="eyebrow">GreekMMLU · $protocol_label</div><h1>K2 Horizon × Qwen3.5<br>$title</h1>
<p>Six models, 0-shot and 5-shot, scored from generated answers. The report separates answer correctness from failures to produce an extractable boxed label. Missing results are shown as —; no score is estimated from job progress.</p>
<div class="pills"><span class="pill ok">$completed / 12 full runs complete</span><span class="pill">$subject_count subjects</span><span class="pill">exact_match · boxed-extract</span><span class="pill">Run $run_id</span><span class="pill">Updated $generated</span><button class="tools" onclick="window.print()">Print / Save PDF</button></div></header>
<main class="shell">$warnings
<section><h2>Overall performance</h2><p>Sample-weighted exact match. SE is the standard error reported by the evaluation harness, in percentage points. A 5-shot change is shown only when both scores exist; check completion status before comparing partial runs.</p><div class="model-grid">$cards</div></section>
<section class="panel"><h2>0-shot → 5-shot comparison</h2><p class="muted">Common 0–100% scale · faded bar: 0-shot · solid bar: 5-shot</p>$chart</section>
<section><h2>Run status and coverage</h2><p>A run is complete only when all 45 subjects and the aggregate score are present, every effective count matches the original count, and the result records no sample limit. Job status is taken from recorded run state, not inferred from a missing result file.</p><div class="table-panel"><table><thead><tr><th>Model</th><th>Shot</th><th>Status</th><th>Subjects</th><th>Scored N</th><th>Job</th><th>Files</th><th>Notes</th></tr></thead><tbody>$statuses</tbody></table></div></section>
<section><h2>Performance by subject area</h2><div class="category-grid">$categories</div></section>
<section><h2>Answer format diagnostics</h2><p>Rates use the available logged responses as their denominator. $format_description “Outside choices” is a valid letter that is not offered by that question. “Reasoning marker” counts responses containing <code>&lt;think&gt;</code>, <code>&lt;/think&gt;</code>, or “Thinking Process” (case-insensitive); it is a textual diagnostic, not a reasoning-quality assessment. Zero rates are shown only when responses were actually inspected.</p><p>$budget_description</p><div class="table-panel"><table><thead><tr><th>Model</th><th>Shot</th><th>Logged N</th><th>Invalid extraction</th><th>Unboxed</th><th>$malformed_label</th><th>Multiple boxes</th><th>Outside choices</th><th>Reasoning marker</th><th>Unreadable lines</th></tr></thead><tbody>$formats</tbody></table></div></section>
$telemetry
<section><h2>Detailed results · $subject_count subjects</h2><p>Every cell includes the reported score, SE, and effective sample count. Format diagnostics appear when sample logs are available. Click a shot or change column to sort.</p><div class="table-panel"><div class="controls"><input id="search" type="search" placeholder="Search subjects…" aria-label="Search subjects"><select id="category" aria-label="Subject area"><option value="">All areas</option><option value="humanities">Humanities</option><option value="social_sciences">Social Sciences</option><option value="stem">STEM</option><option value="other">Other</option></select><span id="count" class="muted"></span></div><div class="table-scroll"><table id="subjects" class="subjects"><thead><tr><th rowspan="2">Area</th><th rowspan="2">Subject</th>$subject_headers</tr><tr>$subject_subheaders</tr></thead><tbody>$subject_rows</tbody></table></div></div></section>
<section><h2>Inspect generated answers</h2><p>The standalone report embeds a bounded selection of correct, incorrect, and invalid responses from each run, in file order. These examples are for inspection, not a random statistical sample. Response previews keep up to $max_example_chars original characters; longer responses retain their beginning and end and are clearly labelled. Full JSONL files are linked below.</p><div class="panel"><div class="controls"><select id="sample-run" aria-label="Sample model and shot"></select><select id="sample-kind" aria-label="Sample outcome"><option value="all">All outcomes</option><option value="correct">Correct</option><option value="incorrect">Incorrect, valid format</option><option value="invalid">Invalid format</option></select><select id="sample-item" aria-label="Select sample"></select></div><div id="sample-view"></div></div></section>
$direct
$likelihood
<section><h2>Methodology and reproducibility</h2><p>The task uses <code>dascim/GreekMMLU</code>, test questions for scoring, and the first five dev examples for 5-shot prompting. $extraction_description</p><p>$method_description The score includes format following as well as answer selection. SE is not a confidence interval; differences are descriptive, not a paired significance test.</p><details><summary>Recorded planned protocol</summary><pre>$protocol_json</pre></details>$metadata</section>
<section><h2>Full sample files</h2><p>Scores, diagnostics, and the selected examples are embedded in this HTML and work offline. Raw-result and log links are relative paths; copy the corresponding results directory alongside the report if you want those links to work on another computer.</p>$sources</section>
</main><footer><div class="shell">Generated $generated · source: <code>$results_dir</code><br>Self-contained HTML · no external libraries, network requests, or estimated scores.</div></footer>
<script id="sample-data" type="application/json">$sample_data</script>
<script>
(() => {
  const table=document.getElementById('subjects'), body=table.tBodies[0], rows=Array.from(body.rows);
  const search=document.getElementById('search'), category=document.getElementById('category');
  function filter(){let count=0;for(const row of rows){row.hidden=!(row.dataset.name.includes(search.value.trim().toLowerCase())&&(!category.value||row.dataset.category===category.value));if(!row.hidden)count++;}document.getElementById('count').textContent=count+' subjects';}
  search.addEventListener('input',filter);category.addEventListener('change',filter);filter();
  let column=-1,ascending=false;for(const th of table.querySelectorAll('[data-sort]'))th.addEventListener('click',()=>{const index=Number(th.dataset.column);ascending=column===index?!ascending:false;column=index;rows.sort((a,b)=>{const av=a.cells[index].dataset.value,bv=b.cells[index].dataset.value;if(av==='')return bv===''?0:1;if(bv==='')return -1;return (Number(av)-Number(bv))*(ascending?1:-1);}).forEach(row=>body.appendChild(row));});
  const data=JSON.parse(document.getElementById('sample-data').textContent), runSelect=document.getElementById('sample-run'), kindSelect=document.getElementById('sample-kind'), itemSelect=document.getElementById('sample-item'), view=document.getElementById('sample-view');
  function option(value,label){const node=document.createElement('option');node.value=value;node.textContent=label;return node;}
  for(const run of data)runSelect.appendChild(option(run.id,run.label));let current=[];
  function load(){const run=data.find(run=>run.id===runSelect.value),kind=kindSelect.value;current=run.samples.filter(s=>kind==='all'||(kind==='invalid'?s.kind!=='valid':kind==='correct'?s.kind==='valid'&&s.correct===true:s.kind==='valid'&&s.correct!==true));itemSelect.replaceChildren();current.forEach((s,i)=>itemSelect.appendChild(option(i,s.task.replace('greekmmlu_','')+' · #'+s.doc_id)));show();}
  function element(tag,text,cls){const node=document.createElement(tag);node.textContent=text;if(cls)node.className=cls;return node;}
  function show(){view.replaceChildren();const sample=current[Number(itemSelect.value)];if(!sample){view.appendChild(element('p','No matching logged examples available yet.','muted'));return;}const grid=document.createElement('div');grid.className='sample-grid';const question=document.createElement('div');question.appendChild(element('h3','Question'));question.appendChild(element('p',sample.question));const choices=document.createElement('ol');sample.choices.forEach((choice,i)=>choices.appendChild(element('li','ΑΒΓΔ'[i]+'. '+choice)));question.appendChild(choices);const response=document.createElement('div');response.appendChild(element('h3','Generated response'));if(sample.response_truncated)response.appendChild(element('p','Preview shortened from '+sample.response_chars.toLocaleString()+' characters. Open the linked JSONL for the full response.','sample-meta'));response.appendChild(element('pre',sample.response,'sample-response'));response.appendChild(element('p','Gold: '+sample.target+' · Extracted answer: '+sample.extracted+' · Logged filter: '+sample.filtered+' · Format: '+sample.kind+' · Correct: '+(sample.correct===null?'not logged':sample.correct?'yes':'no'),'sample-meta'));const link=element('a','Source JSONL · line '+sample.line);link.href=sample.source;response.appendChild(link);grid.append(question,response);view.appendChild(grid);}
  runSelect.addEventListener('change',load);kindSelect.addEventListener('change',load);itemSelect.addEventListener('change',show);load();
})();
</script></body></html>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="One generation run directory (contains 0-shot and 5-shot).")
    parser.add_argument("--output", type=Path, default=ROOT / "greekmmlu_generative_report.html")
    parser.add_argument("--manifest", type=Path, help="Optional run manifest; defaults to RESULTS_DIR/manifest.json.")
    parser.add_argument("--likelihood-results-dir", type=Path, help="Optional earlier likelihood results, presented as a separate protocol.")
    parser.add_argument("--direct-results-dir", type=Path, help="Optional earlier direct boxed-generation results, shown separately from likelihood.")
    parser.add_argument("--examples-per-outcome", type=int, default=8, help="Embedded examples per correct/incorrect/invalid outcome per run.")
    parser.add_argument("--max-example-chars", type=int, default=8000, help="Maximum original response characters embedded per example; full JSONL remains linked.")
    args = parser.parse_args()
    if args.examples_per_outcome < 0:
        parser.error("--examples-per-outcome must be nonnegative")
    if args.max_example_chars < 2:
        parser.error("--max-example-chars must be at least 2")
    results_dir, output = args.results_dir.expanduser().resolve(), args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve() if args.manifest else results_dir / "manifest.json"
    likelihood = args.likelihood_results_dir.expanduser().resolve() if args.likelihood_results_dir else None
    direct = args.direct_results_dir.expanduser().resolve() if args.direct_results_dir else None
    report = assemble(results_dir, output, manifest, likelihood, args.examples_per_outcome, direct, args.max_example_chars)
    document = render(report, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents readers seeing a truncated report during refresh.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(document)
        temporary = Path(handle.name)
    temporary.replace(output)
    complete = sum(run["status"] == "complete" for run in report["runs"])
    print(f"Wrote {output}: {complete}/12 full runs complete; {len(report['warnings'])} data/protocol notes.")


if __name__ == "__main__":
    main()
