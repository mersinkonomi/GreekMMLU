# GreekMMLU reasoning task overrides

Pass `--include_path reasoning_tasks --tasks greekmmlu` from the project root.
The 45 subject tasks and five groups retain their original names and result
keys. Their YAML wrappers reuse the existing dataset selections and group
membership; the reasoning template overrides only the evaluation protocol.
The original direct-answer task files are not modified.

The prompt asks for a Greek explanation and a separate final line such as
`Τελική απάντηση: \boxed{Β}`. It lists only the available 2–4 choice letters
and never reads the current question's gold answer. Five-shot runs use the
same five dev examples with gold final-answer lines; those examples contain
no invented reasoning.

Defaults are `generate_until`, up to 32,768 generated tokens, sampling with
temperature 1.0 and top-p 0.95. The runner may apply recorded model-specific
sampling settings. Generation stops on model end markers, not on question
quotations inside reasoning, intermediate boxes, or a closing answer brace.

The `boxed-extract` filter requires a marked final-answer line at the end of
the current response. It normalizes Greek/Latin uppercase/lowercase letters,
validates the label against the number of choices, ignores intermediate boxes,
and never scores an answer from a later generated question after the thinking
span. Question quotations inside native thinking are allowed. It strips known
trailing control tokens only from its extraction copy. Any `[invalid]` answer
receives zero `exact_match`; raw generated responses are left untouched.

For native thinking models, the runner must set
`GREEKMMLU_REASONING_REQUIRE_THINKING_CLOSE=1` and record that setting in the
manifest. A thinking span opened by the model's chat template must then have
a generated closing tag before the final-answer line can be scored. Use `0`
for Base models with ordinary untagged completion prompts. Explicit unclosed
thinking spans are invalid regardless of this flag. Both Qwen and K2 native
thinking delimiters are supported.

Run the regression checks in the evaluation Python environment:

```bash
PYTHONPATH=lm-evaluation-harness python reasoning_tasks/test_reasoning_tasks.py
```

The tests include actual TaskManager override loading and cached Mathematics
dataset requests for 0-shot and 5-shot, plus prompt-leakage, option-count,
thinking-boundary, regex, raw-response preservation, and group-config checks.
Set the normal offline Hugging Face cache variables when running offline.
