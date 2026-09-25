"""CPU regressions for the durable vLLM reasoning backend.

Run with the evaluation environment:
    python reasoning_tasks/test_reasoning_backend.py

The real backend and SamplingParams are imported, but no model or GPU is loaded.
The engine double deliberately reproduces vLLM 0.21's internal/external ID split.
"""

import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "lm-evaluation-harness"))

from greekmmlu import reasoning_utils as helpers
from lm_eval.api.instance import Instance
from reasoning_backend import ReasoningVLLM


GOOD = "Συλλογισμός. </think>\nΤελική απάντηση: \\boxed{Α}<|im_end|>"
UNCLOSED = "Συλλογισμός χωρίς τέλος.\nΤελική απάντηση: \\boxed{Β}"


class FakeTokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Raw responses must preserve all decoded tokens")
        return "".join(chr(token_id) for token_id in token_ids)


class OutOfOrderEngine:
    def __init__(self, response_by_prompt):
        self.response_by_prompt = response_by_prompt
        self.output_processor = SimpleNamespace(request_states={})
        self.pending = []
        self.enqueue_calls = 0
        self.step_calls = 0
        self.request_counter = 0
        self.before_step = lambda: None

    def enqueue(self, prompts, sampling_params, use_tqdm):
        self.enqueue_calls += 1
        internal_ids = []
        for prompt in prompts:
            external = str(self.request_counter)
            self.request_counter += 1
            internal = f"{external}-feedbeef"
            text = FakeTokenizer.decode(prompt["prompt_token_ids"])
            self.output_processor.request_states[internal] = SimpleNamespace(
                external_req_id=external, prompt=text
            )
            internal_ids.append(internal)
        # The second request finishes first, matching a real continuously batched
        # engine. Returned IDs are internal; completion IDs below are external.
        self.pending.extend(reversed(internal_ids))
        return internal_ids

    def has_unfinished_requests(self):
        return bool(self.output_processor.request_states)

    def step(self):
        self.before_step()
        self.step_calls += 1
        internal = self.pending.pop(0)
        # Real OutputProcessor removes a finished RequestState before returning
        # its output, so looking up the mapping after step is already too late.
        state = self.output_processor.request_states.pop(internal)
        raw = self.response_by_prompt[state.prompt]
        closed = "</think>" in raw
        completion = SimpleNamespace(
            token_ids=[ord(character) for character in raw],
            text=raw.removesuffix("<|im_end|>"),
            finish_reason="stop" if closed else "length",
            stop_reason=0 if closed else None,
        )
        return [SimpleNamespace(request_id=state.external_req_id, finished=True,
                                outputs=[completion])]


class ReasoningBackendTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="greekmmlu-backend-regression-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {helpers.REQUIRE_THINKING_CLOSE_ENV: "1"})
        environment.start()
        self.addCleanup(environment.stop)

    def backend(self):
        # Bypass only GPU/model construction, retaining real backend behavior.
        model = ReasoningVLLM.__new__(ReasoningVLLM)
        model._rank = 0
        model._identity = {"model": str(self.root / "model"), "seed": 1234}
        model._max_length = 1000
        model._max_gen_toks = 64
        model.native_thinking = True
        model.completion_batch_size = 2
        model.raw_generation_path = self.root / "raw.jsonl"
        model.cache_path = self.root / "cache.sqlite"
        model._cache = sqlite3.connect(model.cache_path)
        self.addCleanup(model._cache.close)
        model._cache.execute(
            "CREATE TABLE IF NOT EXISTS generations "
            "(request_hash TEXT PRIMARY KEY,record TEXT NOT NULL,exported INTEGER NOT NULL DEFAULT 0)"
        )
        model._cache.commit()
        model.tokenizer = FakeTokenizer()
        model.tok_encode = lambda values: [[ord(c) for c in value] for value in values]
        model._answer_utils = helpers
        engine = OutOfOrderEngine({"First prompt": GOOD, "Other prompt": UNCLOSED})
        model.model = SimpleNamespace(llm_engine=engine, enqueue=engine.enqueue)
        return model, engine

    @staticmethod
    def requests():
        return [
            Instance(
                request_type="generate_until",
                doc={"choices": ["one", "two"], "answer": index},
                arguments=(prompt, {"max_gen_toks": 64, "temperature": 1.0,
                                    "top_p": 0.95, "do_sample": True}),
                idx=0, metadata=("test_subject", index, 1),
            )
            for index, prompt in enumerate(("First prompt", "Other prompt"))
        ]

    def records(self, model):
        if not model.raw_generation_path.exists():
            return []
        return [json.loads(line) for line in model.raw_generation_path.read_text().splitlines()]

    def test_randomized_ids_out_of_order_and_immediate_durable_results(self):
        model, engine = self.backend()

        def require_previous_completion_durable():
            # A separate connection distinguishes committed rows from writes
            # visible only to the generating connection's transaction.
            with sqlite3.connect(model.cache_path) as reader:
                count = reader.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
            self.assertEqual(count, engine.step_calls)
            self.assertEqual(len(self.records(model)), engine.step_calls)

        engine.before_step = require_previous_completion_durable
        self.assertEqual(model.generate_until(self.requests(), disable_tqdm=True), [GOOD, UNCLOSED])
        self.assertFalse(engine.output_processor.request_states)
        records = self.records(model)
        self.assertEqual([record["doc_id"] for record in records], [1, 0])
        good = records[1]
        self.assertEqual(good["raw_response"], GOOD)
        self.assertEqual(good["token_ids"], [ord(character) for character in GOOD])
        self.assertEqual(good["generated_tokens"], len(good["token_ids"]))
        self.assertTrue(good["reasoning_closed"])
        self.assertTrue(good["has_final_answer"])
        self.assertEqual(good["extracted_answer"], "Α")
        self.assertEqual(good["finish_reason"], "stop")
        self.assertFalse(records[0]["reasoning_closed"])
        self.assertFalse(records[0]["has_final_answer"])
        self.assertEqual(records[0]["finish_reason"], "length")

    def test_stochastic_cache_survives_restart_and_ignores_reference_answer(self):
        model, first_engine = self.backend()
        requests = self.requests()
        expected = model.generate_until(requests, disable_tqdm=True)
        self.assertEqual(first_engine.enqueue_calls, 1)
        # A new backend/connection represents a restarted evaluation process.
        restarted, unused_engine = self.backend()
        requests[0].doc["answer"] = 1
        requests[1].doc["answer"] = 0
        self.assertEqual(restarted.generate_until(requests, disable_tqdm=True), expected)
        self.assertEqual(unused_engine.enqueue_calls, 0)
        self.assertEqual(len(self.records(restarted)), 2)

    def test_seed_and_cache_key_follow_prompt_and_sampling_configuration(self):
        model, _ = self.backend()
        generation = self.requests()[0].args[1]
        first_hash, first_params, _ = model._sampling([1, 2], generation)
        repeat_hash, repeat_params, _ = model._sampling([1, 2], generation)
        self.assertEqual(first_hash, repeat_hash)
        self.assertEqual(first_params["seed"], repeat_params["seed"])
        self.assertNotEqual(model._sampling([1, 3], generation)[0], first_hash)
        self.assertNotEqual(model._sampling([1, 2], {**generation, "top_p": 0.8})[0], first_hash)

    def test_context_overflow_is_rejected_before_engine_submission(self):
        model, engine = self.backend()
        request = self.requests()[0]
        request.arguments = ("x" * 999, request.args[1])
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            model.generate_until([request], disable_tqdm=True)
        self.assertEqual(engine.enqueue_calls, 0)
        self.assertEqual(self.records(model), [])

    def test_interrupted_final_jsonl_line_does_not_damage_completed_records(self):
        model, _ = self.backend()
        model.generate_until(self.requests(), disable_tqdm=True)
        before = self.records(model)
        with model.raw_generation_path.open("ab") as handle:
            handle.write(b'{"interrupted":')
        model._repair_partial_export()
        self.assertEqual(self.records(model), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
