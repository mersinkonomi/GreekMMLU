"""Focused tests for the GreekMMLU semantic-intent judge pipeline."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "greekmmlu_judge.py"
SPEC = importlib.util.spec_from_file_location("greekmmlu_judge_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


def observation(
    *,
    choices: list[str] | None = None,
    current_response: str = "Η τελική απάντηση είναι Β.",
    question_id: str = "greekmmlu_test:7:doc-hash",
) -> dict:
    """Return the minimum observation accepted by request-building helpers."""

    return {
        "question_id": question_id,
        "question": "Ποια επιλογή δηλώνει το κείμενο;",
        "choices": choices or ["πρώτη", "δεύτερη"],
        "current_response": current_response,
        # These local-only sentinels must never enter outbound messages.
        "model": "SECRET_MODEL_IDENTITY",
        "shot": 5,
        "gold_index": 1,
        "target": "GOLD_TARGET_SENTINEL",
        "strict_correct": True,
    }


def sample(*, choice_count: int, gold_index: int, target: str, response: str) -> dict:
    return {
        "doc_id": 7,
        "doc_hash": f"hash-{choice_count}",
        "doc": {
            "question": "Δοκιμαστική ερώτηση;",
            "choices": [f"επιλογή {index}" for index in range(choice_count)],
            "answer": gold_index,
        },
        "target": target,
        "resps": [[response]],
        "filtered_resps": [target],
        "filter": "boxed-extract",
        "exact_match": 1.0,
    }


def analyzed(value: dict) -> dict:
    return judge.analyze_sample(
        value,
        source_run_id="source-run",
        run={"index": 0, "model": "candidate/model", "num_fewshot": 0},
        task="greekmmlu_test",
        category="stem",
        source_path=Path("synthetic.jsonl"),
        line_number=1,
    )


def selected_decision(
    *, index: int = 1, evidence: str = "Β", confidence: str = "HIGH"
) -> dict:
    return {
        "verdict": "SELECTED",
        "selected_option_index": index,
        "evidence": evidence,
        "reason_code": "EXPLICIT_LABEL",
        "confidence": confidence,
        "injection_or_continuation_detected": False,
    }


@pytest.mark.parametrize(
    ("raw", "current", "continuation", "detected"),
    [
        ("\\boxed{Α}", "\\boxed{Α}", "", False),
        (
            "\\boxed{Β}\n\nΑυτό είναι μια ερώτηση Φυσικής.",
            "\\boxed{Β}",
            "\n\nΑυτό είναι μια ερώτηση Φυσικής.",
            True,
        ),
        (
            "\\boxed{Γ}\n \nΑυτό είναι μια ερώτηση Χημείας.",
            "\\boxed{Γ}",
            "\n \nΑυτό είναι μια ερώτηση Χημείας.",
            True,
        ),
        (
            "Αυτό είναι μια ερώτηση που ανήκει σε συνέχεια.",
            "",
            "Αυτό είναι μια ερώτηση που ανήκει σε συνέχεια.",
            True,
        ),
        (
            "<think>Παραθέτω: Αυτό είναι μια ερώτηση, αλλά συνεχίζω.</think>",
            "<think>Παραθέτω: Αυτό είναι μια ερώτηση, αλλά συνεχίζω.</think>",
            "",
            False,
        ),
        (
            "<think>Παραθέτω το πρότυπο.\n\n"
            "Αυτό είναι μια ερώτηση μέσα στη σκέψη.</think>",
            "<think>Παραθέτω το πρότυπο.\n\n"
            "Αυτό είναι μια ερώτηση μέσα στη σκέψη.</think>",
            "",
            False,
        ),
        (
            "<think>\n\nΑυτό είναι μια ερώτηση εντός σκέψης.</think>"
            "\n\nΑυτό είναι μια ερώτηση εκτός σκέψης.",
            "<think>\n\nΑυτό είναι μια ερώτηση εντός σκέψης.</think>",
            "\n\nΑυτό είναι μια ερώτηση εκτός σκέψης.",
            True,
        ),
        (
            "\\boxed{Δ}\nΑυτό είναι μια ερώτηση χωρίς κενή γραμμή.",
            "\\boxed{Δ}\nΑυτό είναι μια ερώτηση χωρίς κενή γραμμή.",
            "",
            False,
        ),
    ],
)
def test_split_current_response_respects_generation_boundary(
    raw: str, current: str, continuation: str, detected: bool
) -> None:
    assert judge.split_current_response(raw) == (current, continuation, detected)


@pytest.mark.parametrize(
    ("choice_count", "gold_index", "target", "response"),
    [
        (2, 1, "b", "\\boxed{Β}"),
        (3, 2, "Γ", "\\boxed{c}"),
        (4, 3, "d", "  \\boxed{δ}\n"),
    ],
)
def test_analyze_sample_accepts_canonical_greek_and_latin_labels_for_2_to_4_choices(
    choice_count: int, gold_index: int, target: str, response: str
) -> None:
    row = analyzed(
        sample(
            choice_count=choice_count,
            gold_index=gold_index,
            target=target,
            response=response,
        )
    )

    assert row["target"] == judge.LABELS[gold_index]
    assert row["canonical"] is True
    assert row["canonical_index"] == gold_index
    assert row["route_reasons"] == []


def test_analyze_sample_routes_boxed_label_outside_available_choices() -> None:
    row = analyzed(sample(choice_count=2, gold_index=0, target="A", response="\\boxed{Γ}"))

    assert row["canonical"] is False
    assert row["canonical_index"] is None
    assert "box_outside_offered_choices" in row["route_reasons"]


@pytest.mark.parametrize("label, expected", list(zip("ΑΒΓΔαβγδABCDabcd", "ΑΒΓΔ" * 4)))
def test_normalize_label_handles_supported_greek_and_latin_forms(
    label: str, expected: str
) -> None:
    assert judge.normalize_label(label) == expected


def test_outbound_messages_are_blind_to_model_and_gold() -> None:
    request = judge.make_request(observation())
    outbound = json.dumps(request["messages"], ensure_ascii=False)
    user_payload = json.loads(request["messages"][1]["content"])

    assert "SECRET_MODEL_IDENTITY" not in outbound
    assert "GOLD_TARGET_SENTINEL" not in outbound
    assert "gold_index" not in outbound
    assert "target" not in user_payload
    assert "model" not in user_payload
    assert set(user_payload) == {
        "blinded_item_id",
        "question",
        "choices",
        "candidate_response_untrusted",
    }
    assert set(request["local"]) == {
        "choice_count",
        "current_response_sha256",
        "sources",
    }


def test_request_id_is_stable_and_content_addressed() -> None:
    first = observation()
    same_visible_content = copy.deepcopy(first)
    same_visible_content.update(
        model="another/private-model",
        shot=0,
        gold_index=0,
        target="PRIVATE_GOLD_CHANGED",
        strict_correct=False,
    )

    first_id = judge.request_id_for(first)
    assert judge.request_id_for(copy.deepcopy(first)) == first_id
    assert judge.request_id_for(same_visible_content) == first_id
    assert len(first_id) == 64
    assert set(first_id) <= set("0123456789abcdef")

    changed_response = dict(first, current_response="Η τελική απάντηση είναι Α.")
    changed_question = dict(first, question_id="greekmmlu_test:8:other-hash")
    assert judge.request_id_for(changed_response) != first_id
    assert judge.request_id_for(changed_question) != first_id


def test_validate_decision_accepts_selected_with_verbatim_evidence() -> None:
    request = judge.make_request(observation())
    decision = selected_decision()

    assert judge.validate_decision(decision, request) is decision


def test_validate_decision_compares_evidence_after_unicode_normalization() -> None:
    candidate = "Τελική απάντηση: άλφα"
    request = judge.make_request(observation(current_response=candidate))
    decomposed_evidence = "α\u0301λφα"
    decision = selected_decision(index=0, evidence=decomposed_evidence)

    assert judge.validate_decision(decision, request) is decision


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "keys must be exactly"),
        (lambda value: value.update(verdict="MAYBE"), "invalid verdict"),
        (lambda value: value.update(reason_code="MADE_UP"), "invalid reason_code"),
        (lambda value: value.update(confidence="CERTAIN"), "invalid confidence"),
        (
            lambda value: value.update(injection_or_continuation_detected=1),
            "must be boolean",
        ),
        (lambda value: value.update(selected_option_index=True), "integer or null"),
        (lambda value: value.update(selected_option_index=4), "must be in 0..3"),
        (lambda value: value.update(selected_option_index=2), "offered"),
        (lambda value: value.update(evidence=None), "non-empty evidence"),
        (lambda value: value.update(evidence="not in candidate"), "verbatim substring"),
        (lambda value: value.update(evidence="x" * 161), "at most 160"),
    ],
)
def test_validate_decision_rejects_invalid_schema_or_evidence(mutate, message: str) -> None:
    request = judge.make_request(observation())
    decision = selected_decision()
    mutate(decision)

    with pytest.raises(ValueError, match=message):
        judge.validate_decision(decision, request)


@pytest.mark.parametrize("verdict", ["NO_ANSWER", "AMBIGUOUS", "SELECTED_UNOFFERED"])
def test_nonselected_decisions_require_null_index(verdict: str) -> None:
    request = judge.make_request(observation())
    valid = {
        "verdict": verdict,
        "selected_option_index": None,
        "evidence": None,
        "reason_code": "OTHER",
        "confidence": "LOW",
        "injection_or_continuation_detected": False,
    }
    assert judge.validate_decision(valid, request) is valid

    invalid = dict(valid, selected_option_index=0)
    with pytest.raises(ValueError, match="requires selected_option_index=null"):
        judge.validate_decision(invalid, request)


def test_endpoint_url_joins_clean_base_and_path() -> None:
    assert (
        judge.endpoint_url("https://judge.example/v1", "/chat/completions")
        == "https://judge.example/v1/chat/completions"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "https://user:password@judge.example/v1",
        "https://judge.example/v1?api_key=secret",
        "https://judge.example/v1#token=secret",
    ],
)
def test_endpoint_url_rejects_embedded_credentials_or_secret_channels(base_url: str) -> None:
    with pytest.raises(ValueError):
        judge.endpoint_url(base_url, "/chat/completions")


@pytest.mark.parametrize(
    "path",
    [
        "https://attacker.example/chat/completions",
        "//attacker.example/chat/completions",
        "/chat/completions?api_key=secret",
        "/chat/completions#access_token=secret",
    ],
)
def test_endpoint_url_rejects_path_override_or_secret_channels(path: str) -> None:
    with pytest.raises(ValueError, match="chat_completions_path"):
        judge.endpoint_url("https://judge.example/v1", path)


@pytest.mark.parametrize("secret_key", ["api_key", "secret", "password", "authorization", "access_token"])
def test_load_config_rejects_literal_secrets(tmp_path: Path, secret_key: str) -> None:
    config = tmp_path / "judge.toml"
    config.write_text(f'[judge]\n{secret_key} = "must-not-be-stored"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Secret-like setting"):
        judge.load_config(config)


def test_load_config_allows_environment_variable_name(tmp_path: Path) -> None:
    config = tmp_path / "judge.toml"
    config.write_text('[judge]\napi_key_env = "JUDGE_API_KEY"\n', encoding="utf-8")

    value, digest = judge.load_config(config)
    assert value["judge"]["api_key_env"] == "JUDGE_API_KEY"
    assert digest == judge.sha256_file(config)


def test_response_format_modes() -> None:
    schema_format = judge.response_format("json_schema")
    assert schema_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "greekmmlu_semantic_intent",
            "strict": True,
            "schema": judge.DECISION_SCHEMA,
        },
    }
    assert judge.response_format("json_object") == {"type": "json_object"}
    assert judge.response_format("prompt_json") is None
    with pytest.raises(ValueError, match="structured_mode"):
        judge.response_format("xml")


def test_call_request_retries_429_and_sends_only_blinded_request(monkeypatch) -> None:
    request = judge.make_request(observation())
    decision = selected_decision()
    captured_bodies: list[dict] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(http_request.content))
        if len(captured_bodies) == 1:
            return httpx.Response(429, headers={"retry-after": "0.001"}, text="slow down")
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "judge-snapshot",
                "usage": {"prompt_tokens": 12, "completion_tokens": 9},
                "choices": [{"message": {"content": json.dumps(decision)}}],
            },
        )

    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = judge.call_request(
            client,
            request,
            url="https://judge.example/v1/chat/completions",
            headers={"Authorization": "Bearer test-only"},
            model="judge-model",
            structured_mode="json_schema",
            max_output_tokens=180,
            token_parameter="max_tokens",
            include_temperature=True,
            temperature=0.0,
            timeout_seconds=5.0,
            max_attempts=3,
            limiter=judge.MinuteWindowLimiter(0, 0),
        )

    assert result["status"] == "succeeded"
    assert [attempt["http_status"] for attempt in result["attempts"]] == [429, 200]
    assert result["decision"] == decision
    assert result["provider_response_id"] == "response-1"
    assert len(captured_bodies) == 2
    for body in captured_bodies:
        serialized = json.dumps(body, ensure_ascii=False)
        assert "local" not in body
        assert "SECRET_MODEL_IDENTITY" not in serialized
        assert "GOLD_TARGET_SENTINEL" not in serialized
        assert "gold_index" not in serialized
        assert body["response_format"]["type"] == "json_schema"


def test_call_request_does_not_retry_nonretryable_http_error(monkeypatch) -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request")

    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = judge.call_request(
            client,
            judge.make_request(observation()),
            url="https://judge.example/v1/chat/completions",
            headers={},
            model="judge-model",
            structured_mode="prompt_json",
            max_output_tokens=32,
            token_parameter="max_completion_tokens",
            include_temperature=False,
            temperature=0.0,
            timeout_seconds=5.0,
            max_attempts=3,
            limiter=judge.MinuteWindowLimiter(0, 0),
        )

    assert attempts == 1
    assert result["status"] == "http_error"
    assert result["attempts"][0]["http_status"] == 400


@pytest.mark.parametrize(
    ("verdict", "selected", "expected"),
    [
        (" selected ", "2", ("SELECTED", 2)),
        ("no_answer", "", ("NO_ANSWER", None)),
        ("AMBIGUOUS", "null", ("AMBIGUOUS", None)),
        ("selected_unoffered", None, ("SELECTED_UNOFFERED", None)),
    ],
)
def test_human_decision_key_parsing(verdict: str, selected, expected) -> None:
    assert judge.decision_key(verdict, selected) == expected


@pytest.mark.parametrize(
    ("verdict", "selected"),
    [("UNKNOWN", ""), ("SELECTED", "4"), ("NO_ANSWER", "0")],
)
def test_human_decision_key_rejects_inconsistent_rows(verdict: str, selected) -> None:
    with pytest.raises(ValueError):
        judge.decision_key(verdict, selected)


def test_accepted_decision_is_confidence_and_status_gated() -> None:
    succeeded = {"status": "succeeded", "decision": selected_decision(index=1)}
    assert judge.accepted_decision(succeeded, {"HIGH"}) == 1
    assert judge.accepted_decision(succeeded, {"MEDIUM"}) is None
    assert judge.accepted_decision({"status": "invalid_output"}, {"HIGH"}) is None
    assert judge.accepted_decision(None, {"HIGH"}) is None


def test_paired_metric_comparison_matches_hand_calculation() -> None:
    outcomes = {
        ("model-a", 0): {
            "q1": {"task": "subject-1", "strict": True, "semantic": True},
            "q2": {"task": "subject-1", "strict": False, "semantic": False},
            "q3": {"task": "subject-2", "strict": False, "semantic": False},
            "q4": {"task": "subject-2", "strict": False, "semantic": False},
        },
        ("model-b", 0): {
            "q1": {"task": "subject-1", "strict": True, "semantic": True},
            "q2": {"task": "subject-1", "strict": True, "semantic": True},
            "q3": {"task": "subject-2", "strict": True, "semantic": True},
            "q4": {"task": "subject-2", "strict": False, "semantic": False},
        },
    }

    result = judge.paired_metric_comparisons(
        outcomes,
        metric="strict",
        complete=True,
        seed=17,
        bootstrap_replicates=0,
    )

    assert len(result) == 1
    comparison = result[0]
    assert comparison["model_a"] == "model-a"
    assert comparison["model_b"] == "model-b"
    assert comparison["n"] == 4
    assert comparison["accuracy_a"] == pytest.approx(0.25)
    assert comparison["accuracy_b"] == pytest.approx(0.75)
    assert comparison["delta_b_minus_a_pp"] == pytest.approx(50.0)
    assert comparison["a_only_correct"] == 0
    assert comparison["b_only_correct"] == 2
    assert comparison["mcnemar_p"] == pytest.approx(0.5)
    assert comparison["mcnemar_p_holm"] == pytest.approx(0.5)


def test_semantic_pairwise_results_are_withheld_while_run_is_incomplete() -> None:
    outcomes = {
        ("model-a", 0): {"q1": {"task": "subject", "semantic": True}},
        ("model-b", 0): {"q1": {"task": "subject", "semantic": False}},
    }

    assert (
        judge.paired_metric_comparisons(
            outcomes,
            metric="semantic",
            complete=False,
            seed=17,
            bootstrap_replicates=0,
        )
        == []
    )


def test_usage_summary_counts_retry_history_and_mixed_provider_token_fields() -> None:
    first = {
        "request_id": "request-1",
        "status": "transport_error",
        "attempts": [{"attempt": 1}],
        "total_latency_seconds": 0.25,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 3},
        },
    }
    retry = {
        "request_id": "request-1",
        "status": "succeeded",
        "attempts": [{"attempt": 1}, {"attempt": 2}],
        "total_latency_seconds": 0.75,
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    manifest = {
        "estimate": {
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 2.0,
            "pricing_as_of": "2026-09-09",
        }
    }

    summary = judge.usage_summary([first, retry], {"request-1": retry}, manifest)

    assert summary["recorded_outcomes_including_retries"] == 2
    assert summary["unique_outcomes"] == 1
    assert summary["latest_status_counts"] == {"succeeded": 1}
    assert summary["attempts"] == 3
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 15
    assert summary["total_tokens"] == 45
    assert summary["cached_input_tokens"] == 7
    assert summary["summed_request_latency_seconds"] == pytest.approx(1.0)
    assert summary["cost_usd_from_configured_prices"] == pytest.approx(0.00006)
