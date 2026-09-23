# Topic-Constrained AI Climbing Coach (IFB220 Assignment 2)

A multi-turn AI assistant that only converses about a single configured
topic (climbing, by default), enforced through several independent,
complementary guardrail layers rather than a single mechanism. Built
against the IFB220 Developer API Portal (GPT-4.1-mini for chat,
Ada-002 for embeddings).

> **Before you submit:** this project was built and tested in an
> environment with no network route to the real IFB220 portal. Sections
> below marked with a note like this need you to actually run them
> against the real API with your own key and record what happens. See
> `docs/TESTING.md` section 4 for the exact checklist, and
> `docs/AI_USAGE.md` for what you personally need to add to the AI-usage
> reflection.

## Contents

- [What this does](#what-this-does)
- [Setup and running it](#setup-and-running-it)
- [Changing the topic](#changing-the-topic-config-only-no-code-changes)
- [Architecture](#architecture)
- [Guardrail layers, and why each one exists](#guardrail-layers-and-why-each-one-exists)
- [Context window management](#context-window-management)
- [Monitoring, logging, and cost](#monitoring-logging-and-cost)
- [Error handling](#error-handling)
- [Testing](#testing)
- [Use of AI tools](#use-of-ai-tools)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)

## What this does

You run `main.py`, get a command-line chat loop, and can ask the
assistant anything about sport and trad climbing -- technique, training,
gear, safety, grading systems, route strategy, and so on. Anything
outside that scope is politely declined, and the assistant resists
attempts to argue, role-play, or inject its way around that restriction.

## Setup and running it

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: fill in API_KEY and AZURE_OPENAI_BASE_URL from the
# IFB220 Developer API Portal dashboard

python main.py
```

In-chat commands: `/usage` (session API usage so far), `/reset` (clear
history, keep the topic), `/quit`.

Run the automated tests (no API key needed -- these are fully mocked):

```bash
pytest -v
```

Run the adversarial prompt suite (works either with real credentials, or
falls back to an offline stand-in with a clear warning label -- see
`docs/TESTING.md`):

```bash
python -m tests.run_adversarial_suite
```

## Changing the topic (config only, no code changes)

Point `TOPIC_CONFIG` in `.env` (or the environment) at a different file
under `topics/`:

```bash
TOPIC_CONFIG=topics/motor_vehicles.json python main.py
TOPIC_CONFIG=topics/cinematography.json python main.py
```

Both of those files ship as worked examples, alongside the default
`topics/climbing.json`. To create a genuinely new topic, copy
`topics/climbing.json` and fill in:

| Field | Purpose |
|---|---|
| `persona_description` | Who the assistant is ("an expert climbing coach...") |
| `in_scope_summary` | Everything the assistant is allowed to discuss |
| `out_of_scope_note` | Explicit reminder of what's excluded |
| `refusal_message` | Exact wording used to decline off-topic requests |
| `injection_refusal_message` | Exact wording used to decline injection attempts |
| `similarity_threshold` | Cosine-similarity cutoff for the embedding relevance check (0.74 worked well for climbing; see note below on tuning) |
| `anchor_phrases` | 10-20 representative in-topic questions, used to build the embedding centroid that the relevance check compares against |

No file under `src/` needs to change. This was verified directly: all
three shipped topic configs load and build a valid system prompt through
the exact same code path (see `docs/TESTING.md`).

## Architecture

```
 raw user input
       |
       v
 [Layer 0] sanitizer            -- strip control chars, normalise unicode,
       |                           cap length, flag suspicious encodings
       v
 [Layer 1] injection detector   -- order-agnostic pattern matching for
       |                           override/jailbreak/extraction attempts
       |                           (refuses here, NO API call made)
       v
 [Layer 2] topic relevance      -- embed input with Ada-002, cosine-compare
       |    (input)                to a cached per-topic centroid
       |                           (refuses here, NO chat-completion call made)
       v
 [Layer 3] hardened system      -- topic + anti-override rules injected
       |    prompt + chat call     as the system message, then GPT-4.1-mini
       |                           is called with the managed conversation
       |                           context
       v
 [Layer 4] topic relevance      -- embed the MODEL'S OWN response, cosine-
       |    (output)                compare to the same centroid; catches
       |                           drift/bypass that got past Layers 0-3
       v
 final response, or refusal, returned to the user
       |
       +--> audit log (every layer's decision, every turn)
       +--> usage log (tokens, calls, calls avoided by early refusal)
       +--> conversation context updated (sliding window + rolling summary)
```

Code layout mirrors this directly:

- `src/guardrails/sanitizer.py` -- Layer 0
- `src/guardrails/injection_detector.py` -- Layer 1
- `src/guardrails/topic_relevance.py` -- Layers 2 and 4 (same class, used twice)
- `src/prompt_builder.py` + `src/api_client.py` -- Layer 3
- `src/pipeline.py` -- orchestrates all five layers in `GuardedChatSession.handle_message()`
- `src/context_manager.py` -- the context window strategy feeding Layer 3
- `src/usage_monitor.py`, `src/audit_logger.py` -- the two logs

## Guardrail layers, and why each one exists

**The core design principle**: don't rely on the model to enforce its
own restrictions. A system prompt telling the model "only discuss
climbing" is Layer 3 here, not the whole system -- prompt-based
restrictions can be argued with, role-played around, or eroded over a
long conversation. Layers 0, 1, 2, and 4 are all *programmatic* checks
that don't depend on the model choosing to comply, so even a successful
manipulation of the model at Layer 3 still has to get past Layer 4
before the user sees it.

| Layer | Catches | Why it's a separate layer, not folded into another one |
|---|---|---|
| 0. Sanitizer | Oversized input, control characters, confusable-unicode tricks, suspiciously encoded blobs | Cleans input BEFORE pattern matching, so Layer 1 can't be dodged by formatting tricks alone. Never refuses by itself -- only normalises. |
| 1. Injection detector | "Ignore previous instructions", fake `[SYSTEM]`/`###ADMIN###` messages, authority claims, persona/jailbreak attempts (DAN-style), prompt extraction attempts | Cheap and instant -- catches the most common, well-known attack family *before spending an API call*, independent of whether the model would have resisted it anyway |
| 2. Topic relevance (input) | Off-topic requests, including ones dressed up in on-topic language | A keyword/regex allow-list is trivially beaten by paraphrase in both directions. Embedding-based similarity to a centroid built from representative in-topic sentences is far more robust to paraphrase, and is the reason Ada-002 is used at all |
| 3. Hardened system prompt | Reinforces scope and anti-override rules at the model level | The model-facing layer -- necessary but not sufficient on its own |
| 4. Topic relevance (output) | Multi-turn topic drift, or a jailbreak that got past Layers 1-3 | The safety net: even if the model *was* talked into answering off-topic, the reply itself won't score as similar to the topic centroid, so it's swapped for the refusal message before the user ever sees it |

Layers 1 and 2 both end the turn *before* calling the chat model when
they fire -- this is a deliberate cost/safety property, not just an
optimisation: the majority of adversarial or off-topic probes never
generate a chat-completion call at all, which shows up directly in the
usage totals (`calls_skipped_by_guardrails` in `UsageTotals`) and is
required reading for the "monitor usage" part of the brief.

### Why embeddings specifically, not just keywords

A denylist/allowlist of keywords is defeated by paraphrase in both
directions: "what should I do about the thing on my fingers after
gripping small holds all day?" contains no climbing keyword but is
clearly in-scope, while a user can wrap an off-topic request in
climbing-flavoured words to slip past a keyword filter. Comparing the
*meaning* of the input to a centroid built from representative on-topic
sentences (via Ada-002 cosine similarity) is much more robust to
paraphrase in both directions -- which is also presumably why the
assignment brief specifically requires an embedding-based component.

### Threshold tuning

`similarity_threshold` (0.74 for the shipped climbing config) is a
judgement call, not a derived constant. Too high and genuine but
loosely-worded on-topic questions get refused (rubric explicitly
penalises this: "on-topic and safe prompts are almost never
restricted"); too low and borderline off-topic requests slip through.
**This needs tuning against the real embedding model** -- the offline
demo in this repo uses a much lower threshold because its crude
hashed-bag-of-words stand-in produces systematically lower similarity
scores than real Ada-002 embeddings (see `docs/TESTING.md`). Run the
adversarial suite live, look at the actual scores in the audit log for
borderline cases, and adjust `similarity_threshold` in the topic config
accordingly before submission.

## Context window management

`src/context_manager.py`: the most recent `MAX_CONTEXT_TURNS` (default
6) conversation turns are kept verbatim; anything older is folded into a
single rolling summary line prepended to the context, so:

- the conversation never silently exceeds the model's context window,
- token usage per turn stays roughly bounded even in a very long
  conversation, and
- the assistant still "remembers" what was discussed earlier, just at
  lower fidelity than the last few turns.

Token counts are estimated via `tiktoken` when available, with a
graceful `len(text)//4` fallback if it isn't (see the `tiktoken` bug in
`docs/TESTING.md` -- this fallback path was not optional, it was
required by a real failure encountered during testing). Rollover
triggers on *either* turn count or estimated token budget, whichever is
hit first, so one unusually long turn can't silently blow the budget.

The summariser is pluggable (`Summarizer` protocol in
`context_manager.py`): the default is a free, deterministic heuristic
(concatenates the folded-away user turns into one sentence); an optional
LLM-based summariser can be enabled with `SUMMARIZE_WITH_LLM=true` for
higher-fidelity summaries at the cost of an extra API call each time the
window rolls over.

## Monitoring, logging, and cost

Two separate logs, deliberately kept apart:

- **`logs/usage.jsonl`** -- one line per API call (chat or embedding)
  with token counts, plus entries for every chat call *avoided* by an
  early guardrail refusal. `UsageMonitor.summary()` prints a running
  total (calls, tokens, estimated cost, calls avoided) and is available
  in-chat via `/usage`.
- **`logs/audit.jsonl`** -- one structured record per user turn, with
  every guardrail layer's decision (sanitizer flags, injection match,
  both relevance scores, final action, latency). This is the file a
  security reviewer would want: "did the guardrails work, and on what
  did they trigger." Deliberately includes the actual (sanitised) user
  text -- for a coaching assistant with no expected PII, this is far
  more useful for auditing than a redacted placeholder. A production
  deployment handling sensitive data would hash or truncate this field;
  documented here as a conscious trade-off, not an oversight.
- **`logs/errors.log`** -- conventional rotating text log (stdlib
  `logging`, `RotatingFileHandler`) for operational issues (API errors,
  timeouts), separate from the security-focused audit trail.

Cost estimation (`UsageTotals.estimated_cost()`) uses per-1k-token rates
read from `.env` (`CHAT_COST_PER_1K_PROMPT` etc.), defaulted to 0 in
`.env.example` since actual portal pricing wasn't available while
building this -- **fill these in from the portal's real pricing before
relying on the cost figures**.

## Error handling

`src/api_client.py` wraps every call with retries (exponential backoff,
capped) for timeouts, connection errors, rate limits (429), and 5xx
responses; a 4xx that isn't a rate limit is not retried and surfaces
immediately. Every failure mode collapses to one of three typed
exceptions (`ApiTimeoutError`, `ApiRateLimitError`, `ApiError`) so the
pipeline only needs to catch `ApiError` once, log it to `errors.log`,
and return a single generic, non-crashing message to the user
(`GENERIC_ERROR_MESSAGE` in `pipeline.py`) -- verified in
`test_chat_api_failure_returns_generic_error_and_does_not_crash`.
Missing configuration (no `API_KEY`, bad `TOPIC_CONFIG` path) fails
fast at startup with a clear message and a non-zero exit code rather
than a raw traceback -- verified by actually running `main.py` with
missing/bad env vars (see `docs/TESTING.md`).

## Testing

Full methodology, results, two real bugs found and fixed, and the
checklist of what still needs a live API key: **[`docs/TESTING.md`](docs/TESTING.md)**.
Short version: 35 unit/integration tests pass (fully mocked, no network
needed), plus a 24-prompt adversarial suite across 9 attack/control
categories, with 24/24 correct on the deterministic layers.

## Use of AI tools

Full disclosure, verification steps actually taken, and what still needs
your own personal input before submission: **[`docs/AI_USAGE.md`](docs/AI_USAGE.md)**.

## Project structure

```
main.py                        CLI entry point
src/
  config.py                    env-var-driven settings
  topic.py                     loads topics/*.json
  prompt_builder.py            builds the hardened system prompt (Layer 3)
  api_client.py                IFB220 portal wrapper, retries/error types
  context_manager.py           sliding window + rolling summary
  usage_monitor.py             token/cost tracking -> logs/usage.jsonl
  audit_logger.py              per-turn guardrail decisions -> logs/audit.jsonl
  pipeline.py                  orchestrates all 5 layers
  guardrails/
    sanitizer.py                Layer 0
    injection_detector.py       Layer 1
    topic_relevance.py          Layers 2 and 4
topics/
  climbing.json                 default topic
  motor_vehicles.json           worked example of a topic swap
  cinematography.json           worked example of a topic swap
tests/
  test_*.py                     unit/integration tests (pytest)
  adversarial_prompts.json      the required adversarial evaluation set
  run_adversarial_suite.py      runs it (live or offline stand-in)
docs/
  TESTING.md
  AI_USAGE.md
.env.example
requirements.txt
```

## Known limitations

- The injection detector is a heuristic pattern matcher, not a
  classifier -- it will not catch every possible phrasing, which is
  exactly why Layer 4 (output relevance) exists as a backstop.
- `similarity_threshold` needs live tuning (see above) -- the shipped
  value is a reasonable starting point, not a validated constant.
- The heuristic conversation summariser is intentionally simple
  (concatenation, not real summarisation); the LLM-based alternative
  exists but costs an extra API call per rollover.
- Cost figures are placeholders until real portal pricing is entered in
  `.env`.
- `run_adversarial_suite.py`'s offline stand-in mode is a wiring smoke
  test only -- it is explicitly not representative of real embedding
  accuracy (see `docs/TESTING.md`).
