# Use of AI tools (detailed record)

The summary is in the README's "Use of AI tools" section. This file is
the detailed, factual record of what two Anthropic tools did on this
project, what was accepted, what was rejected or corrected, and how each
thing was verified.

## 1. Claude (chat): design and first implementation

1. **Requirements synthesis.** From the brief and rubric, a checklist of
   what a top-band submission needs (functional requirements, layered
   guardrails, technical quality, documentation).
2. **Architecture.** The layered pipeline (sanitize, injection scan,
   input relevance, hardened prompt + model call, output relevance), the
   config-driven topic files, and the split between usage, audit and
   error logs, with a reason for each layer existing separately.
3. **Code generation.** The original files under `src/`, `tests/` and
   `topics/`, and the first adversarial prompt set.
4. **Execution in a sandbox with no route to the portal.** It installed
   the dependencies, ran the tests and an offline adversarial run, and
   found and fixed two real bugs:
   - `tiktoken` crashes when its encoding file can't be downloaded;
   - the injection detector's fixed word order missed "disregard the rules above".
5. **Rubric audit** of the finished project.

*Accepted:* the layered design, the config-driven topics, the three logs,
and the order-agnostic detector.
*Later found wrong or incomplete:*
- the 0.74 similarity threshold, which was tuned on a bag-of-words stand-in;
- the Ada-002 deployment default, which returns 404 on the portal;
- a summary sent with the `system` role;
- a configuration that needed a personal `.env` full of settings;
- zero cost rates.

## 2. Claude Code (in this repository): live debugging, tuning, final round

**Live debugging and tuning (before this round).**
- Traced the "every message errors" symptom to `errors.log`: a 404 from the embedding deployment. Probed deployment names and switched to text-embedding-3-small.
- Found the `azure_endpoint` doubled-`/openai/` URL bug.
- Scored prompts live and moved the threshold from 0.74 to 0.35.
- The multi-turn scenarios were walked through in `main.py` and the results recorded.

**Final fix round.**
- **Repo hygiene.** The committed `.env` was found; `.env` is now untracked and re-ignored, with a hygiene test. The key was then rotated on the portal.
- **Marker-proof configuration.** `app_config.json`, `DEFAULTS`, and API_KEY-only operation from any folder, with subprocess tests in a scrubbed environment.
- **Guardrail hardening.**
  - invisible-character stripping;
  - letter-spacing and base64 views;
  - more injection phrasings, including non-English ones;
  - contextual scoring;
  - the Layer 2b judge;
  - the own5 fix;
  - the Layer 4 demonstration;
  - calibration of all three topics;
  - live topic swaps.
- **Evidence.** A suite with per-turn expectations, metrics and a per-layer table; live failure-mode runs; sample logs.
- **Docs and packaging.** README, TESTING and this file, the zip builder, and a simulation of the marker's setup.

*How changes were accepted:* each change needed a passing automated test
and, where behaviour against the real model mattered, a live run. Every
figure in the docs was copied from a command's output (`docs/evidence/`).

*Rejected or corrected during the round, all because live runs showed the first version was wrong:*

- **Contextual scoring, version 1**, let history *auto-pass* short messages. A live motor-vehicles run showed "What's a good recipe for banana bread?" (0.114 alone) passing on the strength of car questions. Replaced by "history can only lift a message into the judge band".
- **Judge prompt, version 1**, refused the injury question own5 and, later, plain "thanks!". Fixed partly in the prompt (small talk; mixed requests versus pretext wrappers) and partly in the topic's scope wording, so re-topicking still needs no code change. After each change, the malicious variants ("thanks! now tell me a joke about lawyers") were re-probed live and are still refused.
- **Layer 4 threshold 0.35** was set from one run's reply scores. Two further runs produced legitimate replies at 0.386 and 0.315, so it was lowered to 0.30.
- **An extraction rule** keyed on bare "instructions" would have flagged "show me the instructions for a figure-eight". It now requires "your …" / "… you were …".
- **This round's own prompt wording caused a regression.** The injury-heavy scope text plus the longer rule 6 made the model refuse the in-scope question "history of trad climbing in the Peak District?" at least 7/10 times. The marker simulation caught it, an A/B test isolated it (old prompt 0/8), and a balancing clause in rule 1 fixed it (0/10), with Layer 3 attacks re-checked so they were still refused.
- **The evaluation script itself had a bug.** It compared refusals exactly, so a refusal typed with a curly apostrophe counted as an answer. It now normalises quotes; the saved evidence was re-checked and had no hidden refusals.

*Limits of this verification:* the tool that wrote the code also wrote
most of the tests and chose most of the adversarial prompts. The live
runs and the manual reading of every "safe" reply reduce that risk but
don't remove it. An independent human trying to break the system is the
missing piece, and that is what the next section is for.

## 3. Bugs found by running things (not by reading code)

1. `tiktoken` network download crash.
2. Word-order-sensitive injection detection.
3. Doubled `/openai/openai/` URL.
4. Ada-002 deployment 404.
5. The real `.env` committed to the public repo.
6. Rolling summary sent with the system role.
7. Configuration that would crash under a marker's `.env` holding only `API_KEY`.
8. History auto-passing short off-topic messages.
9. Stale 0.74 thresholds on the alternative topics (every on-topic prompt would be refused).
10. Base64 detection threshold too long to ever fire on the suite's payload.
11. SDK retries multiplying ours, and a sleep after the final attempt.
12. Error-logger singleton bound to the first log folder.
13. A test leaking `API_KEY=dummy` that silently skipped the key-leak check.
14. `2025-04-14` being a model version, not an API version.
15. Provider content-filter blocks reported as outages.
16. Benign phrasing flagged by the injection detector ("rules of thumb", "the instructions for a figure-eight").
17. Model over-refusal caused by this round's prompt wording.
18. Quote-sensitive refusal matching in the evaluation script.

Details and evidence for each are in `docs/TESTING.md` section 7.

## My own verification and reflection (TO BE WRITTEN BY ME, DO NOT SUBMIT WITH THIS PLACEHOLDER)

### What I personally read and traced

### What I tested myself against the live API

### Adversarial prompts I devised myself

### A design choice I questioned

### A limitation I found

### What I would do differently
