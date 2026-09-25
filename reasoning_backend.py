"""vLLM evaluation backend with lossless responses and a resumable sample cache."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

from tqdm import tqdm
from vllm import SamplingParams, TokensPrompt

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


@register_model("reasoning_vllm")
class ReasoningVLLM(VLLM):
    """Keep the entire generated sequence, including reasoning and control tokens.

    SQLite is the authoritative response cache. JSONL is a durable, human-readable
    export. A request's seed and cache key depend on its actual prompt tokens and
    sampling configuration, never the reference answer. Retrying a stochastic
    evaluation therefore reuses the original samples rather than resampling them.
    """

    def __init__(
        self,
        pretrained,
        raw_generation_path,
        cache_path,
        native_thinking=False,
        completion_batch_size=128,
        **kwargs,
    ):
        if int(kwargs.get("data_parallel_size", 1)) != 1:
            raise ValueError("ReasoningVLLM expects one model engine per Slurm task")
        if kwargs.get("think_end_token") is not None:
            raise ValueError("think_end_token would discard reasoning; leave it unset")
        self.native_thinking = _bool(native_thinking)
        self.completion_batch_size = int(completion_batch_size)
        if self.completion_batch_size < 1:
            raise ValueError("completion_batch_size must be positive")
        self.raw_generation_path = Path(raw_generation_path)
        self.cache_path = Path(cache_path)
        self.raw_generation_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._identity = {
            "model": str(Path(pretrained).resolve()),
            "dtype": str(kwargs.get("dtype", "auto")),
            "model_impl": kwargs.get("model_impl", "auto"),
            "revision": kwargs.get("revision"),
            "tokenizer": kwargs.get("tokenizer"),
            "tokenizer_revision": kwargs.get("tokenizer_revision"),
            "native_thinking": self.native_thinking,
            "seed": int(kwargs.get("seed", 1234)),
            "cache_schema": 1,
        }
        kwargs = copy.deepcopy(kwargs)
        super().__init__(pretrained=pretrained, **kwargs)
        if getattr(self._config, "model_type", "") == "k2_horizon":
            audit = self.model.collective_rpc("greekmmlu_k2_norm_audit", timeout=60)
            if not audit or not all(item.get("original_forward") for item in audit):
                raise RuntimeError("K2 grouped normalization was not verified in every worker")
            print(f"K2 grouped normalization audit: {audit}", flush=True)
            self._identity["k2_grouped_norm_preserved"] = True
        self._cache = sqlite3.connect(str(self.cache_path), timeout=60)
        self._cache.execute("PRAGMA journal_mode=WAL")
        self._cache.execute("PRAGMA synchronous=FULL")
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS generations "
            "(request_hash TEXT PRIMARY KEY, record TEXT NOT NULL, exported INTEGER NOT NULL DEFAULT 0)"
        )
        self._cache.commit()
        self._repair_partial_export()
        self._export_pending()
        helper_path = Path(__file__).resolve().parent / "reasoning_tasks/greekmmlu/reasoning_utils.py"
        spec = importlib.util.spec_from_file_location("greekmmlu_reasoning_backend_utils", helper_path)
        self._answer_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._answer_utils)

    def _repair_partial_export(self):
        """Drop only an interrupted final JSONL record before appending new records."""
        if not self.raw_generation_path.exists():
            self._cache.execute("UPDATE generations SET exported=0")
            self._cache.commit()
            return
        with self.raw_generation_path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if not end:
                return
            handle.seek(end - 1)
            if handle.read(1) == b"\n":
                return
            position = end
            while position:
                start = max(0, position - 65536)
                handle.seek(start)
                tail = handle.read(position - start)
                newline = tail.rfind(b"\n")
                if newline >= 0:
                    handle.truncate(start + newline + 1)
                    break
                position = start
            else:
                handle.truncate(0)
            handle.flush()
            os.fsync(handle.fileno())

    def _export_pending(self):
        # A crash between JSONL flush and this flag update may duplicate a record;
        # request_hash makes that unambiguous, while a response can never be lost.
        rows = self._cache.execute(
            "SELECT request_hash, record FROM generations WHERE exported=0 ORDER BY rowid"
        ).fetchall()
        if not rows:
            return
        with self.raw_generation_path.open("a", encoding="utf-8") as handle:
            for _, record in rows:
                handle.write(record + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._cache.executemany(
            "UPDATE generations SET exported=1 WHERE request_hash=?",
            ((request_hash,) for request_hash, _ in rows),
        )
        self._cache.commit()

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        history = copy.deepcopy(chat_history)
        if getattr(self._config, "model_type", "") == "k2_horizon":
            for message in history:
                if message.get("role") == "assistant" and not any(
                    field in message for field in
                    ("think", "reasoning_content", "reasoning", "think_fast", "think_faster")
                ):
                    message["reasoning_content"] = ""
        return super().apply_chat_template(history, add_generation_prompt)

    def _sampling(self, prompt_token_ids, generation):
        params = copy.deepcopy(generation)
        limit = int(params.pop("max_gen_toks", self.max_gen_toks))
        if limit < 1 or len(prompt_token_ids) + limit > self.max_length:
            raise ValueError(
                f"Prompt has {len(prompt_token_ids)} tokens and requests {limit} new tokens, "
                f"exceeding context {self.max_length}; refusing to truncate the question"
            )
        stops = params.pop("until", []) or []
        if isinstance(stops, str):
            stops = [stops]
        stops = list(dict.fromkeys(stops))
        do_sample = params.pop("do_sample", True)
        if not do_sample:
            params["temperature"] = 0.0
        params.setdefault("temperature", 1.0)
        params.update(
            max_tokens=limit, stop=stops, skip_special_tokens=False,
            spaces_between_special_tokens=False, include_stop_str_in_output=True,
        )
        eos_ids = list(params.get("stop_token_ids") or [])
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        # LLM.generate also applies model defaults. Make its EOS defaults explicit
        # in our per-request configuration and therefore in the cache fingerprint.
        config_path = Path(self._identity["model"]) / "generation_config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            saved_eos = saved.get("eos_token_id", [])
            eos_ids.extend([saved_eos] if isinstance(saved_eos, int) else saved_eos)
        params["stop_token_ids"] = sorted(set(eos_ids))
        if int(params.get("n", 1)) != 1:
            raise ValueError("This evaluation records exactly one completion per question")
        fingerprint = {"model": self._identity, "prompt_token_ids": prompt_token_ids, "generation": params}
        preliminary_hash = hashlib.sha256(_json(fingerprint).encode()).hexdigest()
        params.setdefault("seed", int(preliminary_hash[:8], 16))
        request_hash = hashlib.sha256(_json(fingerprint).encode()).hexdigest()
        return request_hash, params, SamplingParams(**params)

    def _generate_stream(self, chunk):
        """Yield finished requests immediately, while other sequences keep running."""
        engine = self.model.llm_engine
        if engine.has_unfinished_requests():
            raise RuntimeError("Cannot mix an evaluation batch with unfinished engine requests")
        request_ids = self.model.enqueue(
            [TokensPrompt(prompt_token_ids=item["prompt_tokens"]) for item in chunk],
            sampling_params=[item["sampling"] for item in chunk], use_tqdm=False,
        )
        # vLLM 0.21 enqueue returns randomized INTERNAL IDs, while step() emits
        # the original EXTERNAL IDs. Snapshot the explicit mapping before step
        # removes completed request states; never infer identity from ordering.
        external_ids = [engine.output_processor.request_states[request_id].external_req_id
                        for request_id in request_ids]
        if len(set(external_ids)) != len(chunk):
            raise RuntimeError("Non-unique external request IDs in this batch")
        waiting = dict(zip(external_ids, chunk, strict=True))
        while waiting:
            if not engine.has_unfinished_requests():
                raise RuntimeError("Engine stopped before every submitted request completed")
            for output in engine.step():
                if output.finished:
                    if output.request_id not in waiting:
                        raise RuntimeError(f"Unexpected completion: {output.request_id}")
                    yield waiting.pop(output.request_id), output

    def generate_until(self, requests, disable_tqdm=False):
        if not requests:
            return []
        responses = [None] * len(requests)
        pending = {}
        tokenized = self.tok_encode([request.args[0] for request in requests])
        progress = tqdm(total=len(requests), disable=disable_tqdm or self.rank != 0,
                        desc="Running reasoning generation requests")
        for index, (request, prompt_tokens) in enumerate(zip(requests, tokenized, strict=True)):
            request_hash, params, sampling = self._sampling(prompt_tokens, request.args[1])
            cached = self._cache.execute(
                "SELECT record FROM generations WHERE request_hash=?", (request_hash,)
            ).fetchone()
            if cached:
                responses[index] = json.loads(cached[0])["raw_response"]
                progress.update(1)
            elif request_hash in pending:
                pending[request_hash]["indices"].append(index)
            else:
                pending[request_hash] = {
                    "indices": [index], "request": request, "hash": request_hash,
                    "prompt_tokens": prompt_tokens, "params": params, "sampling": sampling,
                }
        work = sorted(pending.values(), key=lambda item: -len(item["prompt_tokens"]))
        for start in range(0, len(work), self.completion_batch_size):
            chunk = work[start:start + self.completion_batch_size]
            for item, output in self._generate_stream(chunk):
                completion = output.outputs[0]
                token_ids = list(completion.token_ids)
                raw = self.tokenizer.decode(token_ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                request = item["request"]
                reasoning_closed = self._answer_utils.reasoning_is_closed(raw)
                answer = self._answer_utils.extract_final_answer(raw, len(request.doc["choices"]))
                if self.native_thinking and not reasoning_closed:
                    answer = self._answer_utils.INVALID
                record = {
                    "request_hash": item["hash"], "created_at": datetime.now(timezone.utc).isoformat(),
                    "task_name": request.task_name, "doc_id": request.doc_id,
                    "model": self._identity["model"], "prompt": request.args[0],
                    "prompt_tokens": len(item["prompt_tokens"]),
                    "raw_response": raw, "engine_text": completion.text,
                    "token_ids": token_ids, "generated_tokens": len(token_ids),
                    "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                    "native_thinking": self.native_thinking,
                    "reasoning_closed": reasoning_closed,
                    "has_final_answer": answer not in {None, "", "[invalid]"},
                    "extracted_answer": answer, "generation": item["params"],
                }
                self._cache.execute(
                    "INSERT INTO generations(request_hash,record) VALUES(?,?)",
                    (item["hash"], _json(record)),
                )
                self._cache.commit()
                self._export_pending()
                for index in item["indices"]:
                    responses[index] = raw
                progress.update(len(item["indices"]))
        progress.close()
        if any(response is None for response in responses):
            raise RuntimeError("Missing responses after generation")
        return responses
