# LLM-assisted evaluation of the GreekMMLU generative runs

This runbook applies to the completed `boxed_20260908_v1` results and the
implemented `greekmmlu_judge.py` command-line tool.

The official generative result remains the gold-based strict boxed exact-match
score. The closed-source LLM produces a separate semantic-intent sensitivity
score by extracting an answer only from responses that are not canonical after
the current-answer boundary is applied. It never decides correctness.

## Evaluation contract

Keep these measurements separate:

1. **Likelihood accuracy** (`acc,none`) is the historical probability-based
   GreekMMLU result.
2. **Strict generation accuracy** (`exact_match,boxed-extract`) is the primary
   generative result produced by the evaluation harness.
3. **Semantic-intent accuracy** is the secondary result produced by
   `greekmmlu_judge.py`. It uses boundary-aware deterministic parsing where
   possible and blinded LLM answer extraction otherwise. Gold comparison is
   always local.

Do not average the three measurements into one score. They use different
inference and scoring protocols.

The semantic judge must not:

- receive `target`, `doc.answer`, or any other gold-answer field;
- receive the candidate checkpoint name, shot condition, or model score;
- solve the multiple-choice question;
- complete unfinished reasoning;
- infer the answer the candidate would probably have selected; or
- replace or modify the saved strict score.

Reasoning that ends without an explicit selection is `NO_ANSWER`. Otherwise,
the evaluation would measure the judge's knowledge rather than the candidate's
answer.

## Exact routing policy

The source run contains 12 model/shot systems and 16,632 questions per system,
for 199,584 observations in total.

The semantic pipeline first divides each raw generation into:

- the **current-answer segment**; and
- a discarded continuation beginning at the first generated blank-line plus
  `Αυτό είναι μια ερώτηση` boundary.

Only the current-answer segment is classified. It is canonical only when the
whole segment, apart from surrounding whitespace, matches exactly one offered
boxed label in Greek or Latin form. For example, `\boxed{Α}` and `\boxed{A}`
are canonical for an item that offers option A. Explanations, several boxes,
unboxed text, malformed boxes, and out-of-range boxes are noncanonical.

This policy routes **14,020 noncanonical observations** to semantic
adjudication. This is intentionally larger than the invalid-extraction count in
the original HTML report: the original extractor searched the full generation
and used its last matching box, whereas this secondary analysis enforces a
current-answer boundary and a canonical full match.

For this frozen source run, those observations reduce to **13,980 unique
noncanonical requests** because identical visible question/response payloads
are adjudicated once.

By default, `prepare` also chooses 50 canonical audit controls from every
model/shot system, balanced between strictly correct and incorrect examples
where possible. With 12 systems, that is **600 audit observations**. Identical
question/response payloads share a request ID, so the number of unique API
requests can be lower than 14,580. With the documented seed and current prompt,
the expected total is **14,575 unique API requests**; the prepared manifest and
`status` output remain authoritative.

Canonical controls diagnose whether the judge can recover a clean answer. They
do not replace deterministic scoring and are not required to finalize the
secondary score.

## Paths and generated artifacts

Source artifacts are read from:

```text
results/generative/boxed_20260908_v1/
```

For a judge run named `boxed_20260908_v1_intent_v1`, `prepare` creates:

```text
results/judge/boxed_20260908_v1_intent_v1/
├── manifest.json
├── normalized_items.jsonl
└── requests.jsonl
```

Later commands add `judgments.jsonl`, `run_metadata.json`, pilot files,
`calibration_summary.json`, aggregate files, and the HTML report. Source sample
files are never rewritten.

## Environment and dependency check

Run from the project directory with the existing Python environment:

```bash
cd "/shared/home/mersin.konomi/greekmmlu generate"
PYTHON=/shared/home/mersin.konomi/miniconda3/envs/greekllm311/bin/python

"$PYTHON" -m pip install -r requirements-judge.txt
"$PYTHON" greekmmlu_judge.py self-test
```

`self-test` makes no API request. The API runner uses an OpenAI-compatible Chat
Completions HTTP endpoint through `httpx`; it does not require the OpenAI SDK.

Keep the API credential only in an environment variable:

```bash
export JUDGE_API_KEY='replace-with-the-secret-at-runtime'
```

Do not place the secret in TOML, committed job scripts, manifests, logs, URLs,
or cached results, and do not echo it.

## Configuration

Create a local `judge_config.toml`. Provider-specific values below are
placeholders:

```toml
[source]
run_id = "boxed_20260908_v1"

[output]
judge_run_id = "boxed_20260908_v1_intent_v1"

[judge]
transport = "openai_chat"
base_url = "https://PROVIDER-BASE-URL/v1"
chat_completions_path = "/chat/completions"
model = "PINNED-JUDGE-MODEL-VERSION"
api_key_env = "JUDGE_API_KEY"

structured_mode = "json_schema"
temperature = 0
omit_temperature = false
max_output_tokens = 180
token_parameter = "max_tokens"

timeout_seconds = 90
max_attempts = 6
concurrency = 4
requests_per_minute = 60
tokens_per_minute = 100000

[adjudication]
audit_per_system = 50
seed = 20260909
accept_confidence = ["HIGH"]

[pricing]
as_of = "YYYY-MM-DD"
input_usd_per_million_tokens = 0.0
output_usd_per_million_tokens = 0.0
```

Replace both zero price placeholders with the selected model's current rates
if you want the built-in cost estimates and `--max-cost-usd` guard.

`base_url` must not contain credentials, query parameters, or fragments.
`chat_completions_path` is joined to it, so the example sends requests to:

```text
https://PROVIDER-BASE-URL/v1/chat/completions
```

Use `token_parameter = "max_completion_tokens"` instead when that is the field
required by the selected compatible endpoint. If the model rejects
`temperature`, set `omit_temperature = true`; do not merely change this halfway
through a judge run.

Supported structured modes are:

- `json_schema`, which sends the strict schema and is preferred when supported;
- `json_object`; and
- `prompt_json`, which relies on the fixed JSON-only judge instruction.

The current tool does not probe provider capabilities or fall back
automatically. Test the chosen mode with the bounded pilot. Once an API request
has created `run_metadata.json`, the tool refuses to mix a different endpoint,
model, structured mode, temperature behavior, token parameter, or
prompt/schema version into that judge run. Create a new judge run ID when any
of those protocol fields must change.

The prompt and response-schema versions are built into the script and recorded
in the manifest; they are not TOML options.

## Implemented decision schema

Every successful response must contain exactly these six fields:

```json
{
  "verdict": "SELECTED",
  "selected_option_index": 0,
  "evidence": "\\boxed{Α}",
  "reason_code": "EXPLICIT_LABEL",
  "confidence": "HIGH",
  "injection_or_continuation_detected": false
}
```

The enums are uppercase and fixed.

`verdict`:

- `SELECTED`
- `SELECTED_UNOFFERED`
- `NO_ANSWER`
- `AMBIGUOUS`

`reason_code`:

- `EXPLICIT_LABEL`
- `EXPLICIT_OPTION_TEXT`
- `EXPLICIT_CONCLUSION`
- `CONFLICTING_ANSWERS`
- `UNFINISHED_REASONING`
- `REFUSAL`
- `EMPTY`
- `UNRELATED_OR_CONTINUATION`
- `OTHER`

`confidence`:

- `HIGH`
- `MEDIUM`
- `LOW`

`selected_option_index` is a zero-based integer. For `SELECTED`, it must refer
to an option actually offered by the current question and `evidence` must be a
non-empty exact substring of the current-answer segment. For every other
verdict, the selected index must be `null`. Evidence, when present, is at most
160 characters and must still be verbatim. The continuation/injection field
must be Boolean.

The local validator rejects missing or additional fields, bad enums,
out-of-range selections, and invented evidence. The judge sees only a blinded
item ID, question, indexed choices, and the untrusted current-answer segment.

## Stage 1: prepare and freeze the queue

Preparation performs local validation and makes no API calls:

```bash
"$PYTHON" greekmmlu_judge.py prepare \
  --config judge_config.toml
```

The IDs can instead be supplied explicitly:

```bash
"$PYTHON" greekmmlu_judge.py prepare \
  --config judge_config.toml \
  --source-run-id boxed_20260908_v1 \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --audit-per-system 50 \
  --seed 20260909
```

Preparation refuses to overwrite an existing judge directory. It verifies that
the source run is complete, each system has 45 timestamp-matched subject files
and 16,632 records, question identities and gold data agree across systems, and
the required fields can be decoded. It also verifies that each item-level strict
score reproduces its saved group score. It then freezes:

- all 199,584 normalized observations;
- the current-answer and discarded-continuation segments;
- the canonical/noncanonical routing result;
- the blinded, deduplicated API requests;
- source-file, prompt, schema, queue, and configuration hashes;
- matching likelihood accuracies and the hashes of their source result files;
- the random seed and audit-control selection; and
- a rough character-based token estimate.

Inspect the result:

```bash
"$PYTHON" greekmmlu_judge.py status \
  --judge-run-id boxed_20260908_v1_intent_v1
```

Confirm that it reports 14,020 LLM-adjudication observations and 600 audit
observations under the default policy. It separately prints never-attempted
requests and retry-eligible transient/transport outcomes. Record the printed
unique-request count for the production cost guard.

The token estimate assumes roughly two characters per token and is deliberately
rough. If dated nonzero rates are filled in under `[pricing]`, `prepare` also
records and prints a planning estimate based on that rough input count and the
maximum output-token allowance for one attempt per request.

## Stage 2: create and run a pilot

Create a reproducible 300-request pilot:

```bash
"$PYTHON" greekmmlu_judge.py pilot \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --sample-size 300 \
  --seed 20260909
```

The command makes no API calls. It writes both:

```text
results/judge/boxed_20260908_v1_intent_v1/pilot_20260909_300.json
results/judge/boxed_20260908_v1_intent_v1/human_calibration_20260909_300.csv
```

The JSON file is the frozen API request list. The CSV is a blind human
annotation template. Pilot selection round-robins across model, shot, category,
and adjudication/audit kind. Inspect the selected mix before judging. Those
model strata remain local and are not exposed in the human CSV or API payload.

Run only the frozen pilot. `--max-requests` is mandatory on every `run`
invocation and caps the number of unique queued items selected in that
invocation:

```bash
"$PYTHON" greekmmlu_judge.py run \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --request-list results/judge/boxed_20260908_v1_intent_v1/pilot_20260909_300.json \
  --max-requests 300
```

When dated nonzero rates are frozen in `[pricing]`, add a dollar guard as well:

```bash
"$PYTHON" greekmmlu_judge.py run \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --request-list results/judge/boxed_20260908_v1_intent_v1/pilot_20260909_300.json \
  --max-requests 300 \
  --max-cost-usd 5
```

The run is rejected before submission if its planning estimate exceeds the
specified dollar threshold. The estimate is only as current as the price and
date you recorded in the configuration, and retries can create additional HTTP
attempts, so this is not a guaranteed provider billing cap.

If the endpoint rejects the request shape, stop. Choose the correct
`structured_mode`, `token_parameter`, or temperature behavior and prepare a new
judge run rather than mixing protocols in the current one.

## Stage 3: human annotation and calibration

Before anyone annotates, make one copy of the blind CSV for each annotator:

```bash
cp results/judge/boxed_20260908_v1_intent_v1/human_calibration_20260909_300.csv \
  annotator_a.csv
cp results/judge/boxed_20260908_v1_intent_v1/human_calibration_20260909_300.csv \
  annotator_b.csv
```

Use at least two Greek-fluent annotators. They should extract only the
candidate's committed answer, without solving the question or consulting the
gold label. Fill:

- `human_verdict` with one of the four uppercase verdicts;
- `human_selected_option_index` with a zero-based index only for `SELECTED`;
- `human_confidence`; and
- optional `notes`.

Leave the selected index empty for `SELECTED_UNOFFERED`, `NO_ANSWER`, and
`AMBIGUOUS`.

Compare the completed blind annotations with each other and with the pilot
judge results:

```bash
"$PYTHON" greekmmlu_judge.py calibrate \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --annotations annotator_a.csv annotator_b.csv
```

`calibrate` prints and saves:

- human exact agreement on rows completed by all supplied annotators;
- judge-versus-human exact agreement on the unanimous-human subset; and
- precision of the judge's `SELECTED` decisions whose confidence is accepted by
  the frozen manifest on that subset.

The output file is:

```text
results/judge/boxed_20260908_v1_intent_v1/calibration_summary.json
```

The command does not resolve human disagreements and does not automatically
block production. Review disagreement cases with a third annotator. As a
project-level go/no-go rule, pre-register a threshold before examining model
rankings. One reasonable target is at least 95% exact judge/human agreement and
98% precision among accepted selections (HIGH under the recommended config),
with the sample sizes reported alongside the percentages. This threshold is a
recommended policy, not an enforced CLI feature.

If calibration is inadequate, do not run the production queue. Revise the
protocol, use a new judge run ID, and repeat the pilot, or use human-only
adjudication.

## Stage 4: run or resume production

Check the number of unique requests, recorded outcomes, and outcome statuses:

```bash
"$PYTHON" greekmmlu_judge.py status \
  --judge-run-id boxed_20260908_v1_intent_v1
```

Then set `--max-requests` explicitly to the number you authorize for that
invocation. For example, to process at most 2,000 pending requests:

```bash
"$PYTHON" greekmmlu_judge.py run \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --max-requests 2000
```

Repeat the command after checking `status` until `Eligible in a normal run
invocation` is zero. Then inspect the status breakdown. That number
includes never-attempted requests plus retryable exhausted-transient and
transport outcomes. Recorded terminal API/schema errors are excluded from that
number, but can still make the final semantic report incomplete. Resolve those
required failures with a diagnosed, bounded `--retry-errors` invocation. Resume
is automatic: successful request IDs already present in `judgments.jsonl` are
not submitted again. There is no `--resume` option.

Throughput settings may be reduced without changing the evaluation protocol:

```bash
"$PYTHON" greekmmlu_judge.py run \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --max-requests 500 \
  --concurrency 2 \
  --requests-per-minute 30 \
  --tokens-per-minute 50000
```

The runner:

- applies shared request-per-minute and estimated-token-per-minute windows;
- retries transport errors and HTTP 408, 429, 500, 502, 503, and 504;
- respects a numeric `Retry-After` value when present, otherwise using capped
  exponential backoff with jitter;
- writes each result to the append-only `judgments.jsonl` and flushes it to
  disk;
- records API status, attempts, latency, returned model, response ID, and any
  provider usage object; and
- stores the first run's protocol fields in `run_metadata.json` to prevent
  incompatible results from being mixed.

HTTP errors, refusals, invalid output, and client errors are terminal by
default. Transport errors remain eligible on a later invocation. After
diagnosing a terminal error and confirming that retrying it is appropriate, use
an explicit bounded retry:

```bash
"$PYTHON" greekmmlu_judge.py run \
  --judge-run-id boxed_20260908_v1_intent_v1 \
  --max-requests 100 \
  --retry-errors
```

`--retry-errors` makes all non-success outcomes eligible, so do not use it in an
unbounded loop for authentication failures, unsupported schemas, or systematic
refusals. Invalid JSON is recorded immediately; the current implementation does
not issue a hidden formatting-repair call.

## Acceptance and denominator rules

Acceptance is deterministic after schema validation.

For a canonical current-answer segment, the secondary pipeline uses its boxed
option directly and makes no adjudication request, except when the item is a
clean audit control.

For a routed noncanonical response, an LLM result is accepted only when:

1. its API status is `succeeded`;
2. its `verdict` is `SELECTED`;
3. its `confidence` is in the manifest's frozen `accept_confidence` list;
4. `selected_option_index` is an offered zero-based option index; and
5. non-empty `evidence` is a verbatim substring of the current-answer segment.

The schema validator enforces the last two conditions. Under the recommended
configuration, only `HIGH` confidence is accepted. `reason_code` and
`injection_or_continuation_detected` are retained for audit but do not currently
change automatic acceptance; review suspicious accepted decisions manually.

| Outcome | Secondary treatment |
|---|---|
| Canonical current answer | Deterministic selected index |
| Successful accepted `SELECTED` | Compare selected index with hidden gold locally |
| `SELECTED_UNOFFERED` | Candidate unresolved; incorrect in the 16,632-item denominator |
| `NO_ANSWER` | Candidate unresolved; incorrect in the denominator |
| `AMBIGUOUS` | Candidate unresolved; incorrect in the denominator |
| `SELECTED` below accepted confidence | Candidate unresolved; incorrect in the denominator |
| Missing request, API error, refusal, or invalid schema | Processing failure; semantic score remains incomplete |

Candidate failures and processing failures are deliberately different. A
schema-valid `NO_ANSWER` is evidence about the candidate and counts as
incorrect. A network failure is not evidence about the candidate, so the final
semantic score is withheld until every required noncanonical request has a
schema-valid successful result. A provisional lower bound remains available in
the machine-readable aggregate.

The strict harness score is never recomputed or overwritten. The semantic
pipeline may record both `rescued` cases, where it makes a strict failure
correct, and `reversed` cases, where boundary-aware interpretation makes a
strict success incorrect.

## Stage 5: report

Generate the default report and machine-readable outputs:

```bash
"$PYTHON" greekmmlu_judge.py report \
  --judge-run-id boxed_20260908_v1_intent_v1
```

The command writes:

```text
results/judge/boxed_20260908_v1_intent_v1/
├── aggregate.json
├── aggregate.csv
├── per_subject.csv
├── per_category.csv
├── pairwise.csv
├── adjudications.jsonl
└── report.html
```

Use `--output PATH` only when another HTML location is needed. The JSON/CSV and
adjudication files remain in the judge-run directory.

The current report shows, per model and shot condition:

- likelihood accuracy when a matching complete historical result is found;
- the original strict generation accuracy;
- final semantic-intent accuracy, or a provisional lower bound while partial;
- semantic-minus-strict change;
- canonical-format and generated-continuation rates;
- routed, rescued, reversed, unresolved, and processing-failure counts; and
- exact agreement on completed clean audit controls.

The HTML and `aggregate.json` also include the calibration summary when it is
present, provider status/token/latency totals, cost calculated from the frozen
configured prices, subject-macro accuracy, and every within-shot pairwise model
comparison. `pairwise.csv` reports B-minus-A accuracy differences, 5,000
bootstrap replicates that resample questions within each subject, 95% percentile
intervals, exact paired McNemar tests, and Holm-adjusted p-values. The bootstrap
seed is derived deterministically from the frozen run seed, metric, shot, and
model pair. Semantic comparisons are withheld while any required judge request
is missing or failed; strict comparisons remain available.

Interpret the report in this order:

1. Use strict generation as the primary generative ranking.
2. Use semantic intent to test whether boundary/formatting behavior materially
   changes that ranking.
3. Keep likelihood scores alongside, not blended with, the generative metrics.
4. If a winner appears only under poorly calibrated or aggressive semantic
   recovery, report the comparison as inconclusive.

## Privacy, governance, and cost control

Before the first pilot request:

- confirm which GreekMMLU split generated the source artifacts;
- confirm that its license and project policy allow transmission to the chosen
  service;
- never send private leaderboard questions without explicit authorization;
- review the provider's retention, model-training, deletion, and regional
  processing terms;
- verify whether a documented no-storage setting is available; the current
  client does not invent or assume one; and
- obtain any required institutional approval for external processing.

The API payload contains the Greek question, choices, and candidate's
current-answer text. It excludes gold answers and local model metadata. Inspect
`requests.jsonl` before authorizing external transmission.

Estimate cost using the unique request count from `status`, the rough estimate
in `manifest.json`, and current provider pricing:

```text
estimated cost =
    input_tokens / 1,000,000 * provider_input_price
  + output_tokens / 1,000,000 * provider_output_price
```

The two-characters-per-token estimate is only a rough heuristic, not a
provider-tokenizer result or a guaranteed upper bound. Use actual pilot usage
from `judgments.jsonl` or the provider dashboard to refine the estimate.
`--max-requests` limits unique queued items per invocation; automatic retries
can produce more HTTP attempts. `--max-cost-usd` adds a dollar-denominated
preflight planning threshold when the config contains nonzero dated input or
output rates. It is not a provider billing cap and does not budget retry
attempts. The current runner does not implement a provider batch API.

## Separate future track: explanation quality

The existing `boxed_20260908_v1` protocol explicitly requested only one boxed
label and allowed at most 64 generated tokens. It cannot validly measure Greek
prose, reasoning quality, or explanation quality. The semantic-intent judge
must not be repurposed to assign those scores.

Create a separate experiment if explanation quality is important for choosing
a fine-tuning checkpoint:

1. Freeze a new prompt that requests a selected option and a short Greek
   explanation.
2. Select a stratified subset across subject groups and education levels.
3. Have qualified Greek-speaking reviewers create or verify reference
   rationales and supporting evidence; a gold option alone is not a gold
   explanation.
4. Generate all candidate outputs under a new run ID with an identical,
   sufficient output budget and documented prompt/template protocol.
5. Score the selected option deterministically against the gold index.
6. Judge explanation correctness, evidence consistency, relevance,
   completeness, Greek grammar and idiomaticity, clarity, and register as a
   separate rubric.
7. Blind model identities, randomize pairwise A/B order, repeat comparisons
   with order reversed, allow ties, and route order-sensitive decisions to a
   second judge or human.
8. Calibrate the explanation judge against Greek-speaking subject-matter
   reviewers, especially for law, medicine, and engineering.
9. Report explanation quality conditional on answer correctness as well as
   overall, so fluent explanations of wrong answers cannot dominate.

The current `greekmmlu_judge.py` does not implement this explanation track. It
requires a new generation protocol, prompt/schema, cache namespace, calibration
set, and report. Do not merge it with likelihood accuracy, strict boxed exact
match, or semantic-intent extraction.

## Reproducibility checklist

- [ ] `self-test` passes in the intended environment.
- [ ] `prepare` validates all 12 systems and creates a new judge-run directory.
- [ ] The manifest reports 199,584 source observations, 14,020 adjudication
      observations, and 600 default audit observations.
- [ ] `requests.jsonl` was reviewed before external transmission.
- [ ] No API payload contains a gold answer or candidate model identity.
- [ ] The pilot request list and blind human CSV were generated and preserved.
- [ ] At least two Greek-fluent annotators completed the calibration subset.
- [ ] `calibrate` results meet the pre-registered project threshold.
- [ ] Every `run` invocation has an explicit, reviewed `--max-requests` value.
- [ ] Provider/model, schema mode, token field, and temperature behavior remain
      fixed within the judge run.
- [ ] Required processing errors are resolved before treating semantic scores
      as final.
- [ ] The strict score remains primary and unchanged.
- [ ] Privacy, license, retention, and cost checks are documented.
- [ ] Any explanation-quality evaluation is a separate experiment.
