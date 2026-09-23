# Use of AI tools

## Read this first

This project was built collaboratively with Claude (Anthropic), working
directly in a sandboxed development environment with the ability to
write files, install packages, and actually execute code and tests. The
sections below up to **"Your turn"** describe, factually, what happened
in that session: what was generated, what was run, and what was found
and fixed as a result. That part is true and you can use it as-is.

**What you must not do**: submit the "Your turn" section blank, or fill
it in with more AI-generated text you haven't actually verified
yourself. The rubric specifically wants evidence of *your* critical
thinking, *your* verification, and *your* testing -- not a well-written
description of an AI's testing process. An AI (any AI, including the one
that helped write this file) cannot supply that on your behalf, because
the whole point is that it's evidence of independent human judgement
checking the AI's work. Read the code. Run it yourself. Try to break it
yourself. Form your own opinion about the design choices below, even
where that opinion is "I agree with this and here's why" -- agreement
you've actually thought through counts as critical thinking; agreement
you haven't is just trust.

## What AI assistance actually consisted of

1. **Requirements synthesis**: given the assignment brief and grading
   rubric, Claude produced a structured checklist of what a 7/7
   submission needs (functional requirements, layered-guardrail
   architecture, technical-quality expectations, documentation
   expectations) before any code was written.
2. **Architecture design**: the five-layer guardrail pipeline (sanitize
   -> injection-scan -> topic-relevance-in -> model-call ->
   topic-relevance-out), the config-driven topic system, and the
   split between usage/audit/error logs were all proposed by Claude as
   a coherent design, with reasoning for why each layer exists and why
   it isn't folded into another one (see the README's guardrail table).
3. **Code generation**: every file under `src/`, `tests/`, and `topics/`
   was written by Claude.
4. **Actual execution and verification, in the same session**: this is
   the part that goes beyond "generate code and hope it works." Claude:
   - installed the actual dependencies (`openai`, `tiktoken`, `pytest`,
     etc.) in a real Python environment,
   - ran `python -m py_compile` and an AST parse over every file to
     confirm they're syntactically valid,
   - wrote and then **ran** the pytest suite, not just wrote it,
   - **found a real bug from that run**: `tiktoken.get_encoding()`
     attempts a network download and threw `requests.exceptions.HTTPError:
     403 Client Error` in the sandboxed environment, crashing every test
     that touched the context manager. The original code only caught
     `ImportError`, which didn't cover this failure mode at all. Fixed
     by broadening the exception handling and caching success/failure
     once at module load.
   - built and ran an adversarial prompt suite (`tests/adversarial_prompts.json`
     + `tests/run_adversarial_suite.py`), and **found a second, more
     interesting bug**: the injection detector's regexes assumed a fixed
     word order (e.g. "ignore ... previous ... instructions" in that
     exact sequence) and missed real attacks phrased the other way round
     -- "disregard the rules **above**" (target word before the scope
     word), and "what instructions were you given ... **print** them
     verbatim" (62 characters between the relevant words, outside the
     original matching window). Fixed by rewriting the detector to use
     order-agnostic proximity matching instead of ordered regexes, then
     **re-ran the full suite again** to confirm the fix (24/24 on the
     deterministic layers, up from 21/24 and then 23/24 across two
     intermediate fixes) without breaking any of the 35 unit tests.
   - actually ran `main.py` with missing and invalid configuration to
     confirm it fails with a clear message and clean exit code rather
     than a raw traceback.
   - actually loaded all three topic configs (climbing, motor vehicles,
     cinematography) through the same code path to confirm the "retopic
     via config only" claim is true, rather than just asserting it.

   All of the above is captured with real command output in
   `docs/TESTING.md`, not paraphrased or invented after the fact.

5. **Honest limitations flagged by Claude itself**, unprompted:
   - The sandbox this was built in has **no network access to the real
     IFB220 Developer API Portal**, so nothing here has been tested
     against actual GPT-4.1-mini or Ada-002 responses. The adversarial
     suite's "full pipeline" results are from a clearly-labelled offline
     bag-of-words stand-in, explicitly documented as not representative
     of real embedding accuracy.
   - The `similarity_threshold` value is a guess, not a tuned constant.
   - The injection detector is a heuristic pattern list, not a trained
     classifier -- it will have gaps that weren't found by this session's
     24-prompt suite.

## Why this counts as "verification", but only partially

Re-running tests and fixing what breaks is real verification -- it's
better than reading code and assuming it works. But everything above was
still done by the same system that wrote the code, checking its own work
inside one session, without a live API, and without an independent human
trying to actively break it with prompts of their own devising. That's
the gap you need to close.

## Your turn (delete this heading and write your own version of the section below)

Fill this in with what you actually did, honestly, even if it's brief.
A credit-level answer briefly describes what you checked; an HD-level
answer shows you found something, understood *why* it was wrong, and
can explain the fix (or explain why you chose not to fix it, if that was
the right call).

Suggested structure, based on what the rubric is actually asking for:

**What I personally read and understood** (not just skimmed): pick 2-3
files you actually traced through line by line -- `pipeline.py` is the
one place that ties everything together and is worth understanding
properly. Note anything that confused you initially and how you worked
out what it does.

**What I personally tested against the live API**: run
`python -m tests.run_adversarial_suite` with your real credentials in
`.env`, and paste the actual output here. Note anything that surprised
you compared to the offline stand-in results in `docs/TESTING.md`.

**Adversarial prompts I tried myself, beyond the shipped set**: the 24
prompts here are a starting point. What would *you* try to break this
with? Write at least a few of your own, run them, and record what
happened -- pass or fail, and why you think that was the result.

**A design choice I questioned**: did you agree with the layered
architecture, or would you have done something differently? For example
-- do you think the `similarity_threshold` of 0.74 is right for
climbing, based on real scores you observed? Would you have made the
injection detector block on suspicious base64 blobs directly, rather
than only flagging them for the audit log? There's a real, defensible
case either way -- picking one and explaining why is the critical
thinking the rubric wants.

**A limitation I found that wasn't already documented**: try to actually
break this. What happened?

**What I'd do differently if I had more time**: this is where genuine
reflection goes, not a restatement of the "Known limitations" list in
the README.
