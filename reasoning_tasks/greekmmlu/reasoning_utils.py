"""Reasoning prompts and strict final-answer extraction for GreekMMLU.

The original tasks stay untouched. Subject names and gold-label validation are
reused from their helpers; only generated answers use the reasoning protocol.
"""

import importlib.util
import os
from pathlib import Path
import re


_ORIGINAL_HELPERS = (
    Path(__file__).resolve().parents[2]
    / "lm-evaluation-harness/lm_eval/tasks/greekmmlu/utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "greekmmlu_original_helpers", _ORIGINAL_HELPERS
)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)

LABELS = _original.LABELS
subjects_gr = _original.subjects_gr
doc_to_target = _original.doc_to_target

LABEL_MAP = {
    **{label: label for label in LABELS},
    **{label.lower(): label for label in LABELS},
    **dict(zip("ABCD", LABELS)),
    **dict(zip("abcd", LABELS)),
}
INVALID = "[invalid]"
REQUIRE_THINKING_CLOSE_ENV = "GREEKMMLU_REASONING_REQUIRE_THINKING_CLOSE"
THINKING_PAIRS = {
    "<think>": "</think>",
    "<ifm|think>": "</ifm|think>",
    "<ifm|think_fast>": "</ifm|think_fast>",
    "<ifm|think_faster>": "</ifm|think_faster>",
    # Accept older pipe-delimited spellings defensively as well.
    "<|ifm|think|>": "<|ifm|/think|>",
    "<|ifm|think>": "<|ifm|/think>",
}
THINKING_TAG_RE = re.compile(
    "|".join(re.escape(tag) for pair in THINKING_PAIRS.items() for tag in pair)
)
TERMINAL_STOP_RE = re.compile(
    r"(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|ifm\|endoftext\|>|<\|ifm\|im_end\|>|</s>)[ \t\r\n]*\Z"
)

# A later question must never supply the answer for the current question.
NEW_QUESTION_RE = re.compile(
    r"(?im)^[ \t]*(?:Ερώτηση|Νέα ερώτηση|Επόμενη ερώτηση)[ \t]*:"
)
# Require an explicit final-answer marker on the last nonempty line, not merely
# any intermediate box. Allow common Markdown/LaTeX wrappers around that line.
FINAL_ANSWER_RE = re.compile(
    r"(?im)^[ \t]*(?:\*\*)?Τελική[ \t]+απάντηση(?:\*\*)?[ \t]*:"
    r"[ \t]*(?:\*\*)?[ \t]*(?:\$+)?\\boxed[ \t]*\{[ \t]*"
    r"([ΑΒΓΔαβγδABCDabcd])[ \t]*[.)]?[ \t]*\}"
    r"(?:\$+)?(?:\*\*)?[ \t]*[.!]?[ \t]*\Z"
)


def doc_to_text(doc):
    """Ask for an explanation followed by a final box, without reading gold."""
    choices = doc["choices"]
    if not 2 <= len(choices) <= len(LABELS):
        raise ValueError(f"GreekMMLU expects 2–4 choices, got {len(choices)}")
    subject = subjects_gr.get(doc["subject"], doc["subject"])
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(LABELS, choices)
    )
    allowed_labels = ", ".join(LABELS[: len(choices)])
    return (
        f"Αυτή είναι μια ερώτηση {subject}.\n\n"
        f"Ερώτηση: {doc['question']}\n{choices_text}\n\n"
        "Λύσε την ερώτηση και εξήγησε τον συλλογισμό σου στα ελληνικά. "
        "Στη συνέχεια, ολοκλήρωσε την απάντησή σου με μία ξεχωριστή τελική "
        "γραμμή της μορφής «Τελική απάντηση: \\boxed{γράμμα}», "
        f"αντικαθιστώντας το «γράμμα» με ένα από τα {allowed_labels}. "
        "Το πλαίσιο πρέπει να περιέχει μόνο το γράμμα της επιλογής που "
        "θεωρείς σωστή. Μη γράψεις άλλη ερώτηση ή κείμενο μετά την τελική γραμμή."
        "\n\nΑπάντηση:"
    )


def doc_to_fewshot_target(doc):
    """Dev examples demonstrate the answer format, without invented reasoning."""
    return f"Τελική απάντηση: \\boxed{{{doc_to_target(doc)}}}"


def _thinking_state(response):
    """Return final close position and whether an explicitly opened span is left."""
    pending = []
    close_end = None
    for match in THINKING_TAG_RE.finditer(response):
        token = match.group(0)
        if token in THINKING_PAIRS:
            pending.append(THINKING_PAIRS[token])
        else:
            if pending:
                if pending[-1] != token:
                    return close_end, True
                pending.pop()
            close_end = match.end()
    return close_end, bool(pending)


def reasoning_is_closed(response):
    """Whether native thinking was closed, including a span opened in the prompt."""
    if not isinstance(response, str):
        return False
    close_end, pending = _thinking_state(response)
    return close_end is not None and not pending


def _first_response(response, native_preopened):
    """Stop at the first question marker outside a native thinking span."""
    pending = []
    inherited_thinking = native_preopened
    events = sorted(
        [(match.start(), "thinking", match) for match in THINKING_TAG_RE.finditer(response)]
        + [(match.start(), "question", match) for match in NEW_QUESTION_RE.finditer(response)]
    )
    for position, kind, match in events:
        if kind == "question":
            if not pending and not inherited_thinking:
                return response[:position].strip()
            continue
        token = match.group(0)
        if token in THINKING_PAIRS:
            pending.append(THINKING_PAIRS[token])
        else:
            if pending and pending[-1] == token:
                pending.pop()
            inherited_thinking = False
    return response.strip()


def extract_final_answer(response, num_choices):
    """Return one normalized label or INVALID; never mutate the raw response."""
    if not isinstance(response, str):
        return INVALID
    require_close = os.environ.get(REQUIRE_THINKING_CLOSE_ENV, "0") == "1"
    current_response = _first_response(response, native_preopened=require_close)
    close_end, pending = _thinking_state(current_response)
    if pending:
        return INVALID
    # Native chat templates may open thinking in the prompt, so the raw generated
    # completion need not contain the opening tag. The runner sets this flag per
    # model and records it in the run manifest: 1 for native thinking, 0 for Base.
    if require_close and close_end is None:
        return INVALID
    if close_end is not None:
        current_response = current_response[close_end:].strip()
    while TERMINAL_STOP_RE.search(current_response):
        current_response = TERMINAL_STOP_RE.sub("", current_response).strip()
    match = FINAL_ANSWER_RE.search(current_response)
    if match is None:
        return INVALID
    label = LABEL_MAP.get(match.group(1), INVALID)
    return label if label in LABELS[:num_choices] else INVALID


def extract_final_answers(resps, docs):
    """Harness custom-filter interface; raw `resps` remain available for audit."""
    return [
        [extract_final_answer(response, len(doc["choices"])) for response in batch]
        for batch, doc in zip(resps, docs)
    ]
