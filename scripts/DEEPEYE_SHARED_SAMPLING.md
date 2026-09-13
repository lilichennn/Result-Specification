# DeepEye shared sampling runtime (C4, 2026-09-13)

The native `scripts/deepeye_bird_interact_run.py` and RC
`scripts/rc_evaluation/deepeye/cli.py` production entrypoints now use one
`SamplingRuntime` per invocation. All selected questions enter the workflow;
each question retains Schema Linking → Generation → Revision → Selection
dependencies. RC defaults to its target stage and uses the frozen native prefix.
`--continue-downstream` explicitly runs/reuses downstream stages against the new
target output. No implicit `none`/control calls are made.

## Limits and budgets

The process-local aggregate defaults are `--request-limit 8000`,
`--request-workers 8000`, `--coordinator-workers 6000`, `--http-connections 8000`,
`--request-start-rate 50`, `--chat-timeout 660`, `--retry-delay 0`, and
`--pg-concurrency 10`. The coordinator pool is shared across all native branches
and questions. Nested coordinating work executes inline to avoid deadlock;
independent sample slots run on the separate request-worker pool. Native
constructor pools are drained and replaced before stage instrumentation. The
HTTP facade uses the runtime's shared asynchronous connection pool. SDK retries
are zero; C1/C2 own the single sample retry loop (four total attempts per slot).
No extra model gate or adaptive question throttle is installed.
`extractor_max_retries`/`--extractor-retries` remains a native constructor
compatibility field; it does not install another retry loop or change those four
sample attempts.

The normal environment model remains `qwen3.8-2.4t-a95b`; `--max-tokens 16384`,
temperature 0.6, and no thinking toggle. Schema Linking uses direct 4/reverse 4;
Generation uses DC 4/Skeleton 4/ICL 4; a triggered Revision checker requests 5
samples and the eight checkers retain their native order. Selection retains
evaluator 5, top-k 2, shortcut 0.6. SQL deduplication, PG behavior, native caches,
prompts, and the separately maintained `rc_prompt.txt` are preserved. Sample
completeness counts slots even when their SQL strings duplicate each other.

Native `prepare`, `run`, and `resume` accept the explicit runtime flags above.
RC `prepare` inherits the native source's frozen configuration and accepts
explicit runtime overrides; `run`/`resume` reconstruct that saved configuration.
Legacy `--workers`, `--inner-workers`, `--adaptive-concurrency`,
`--concurrency-*`, `--thinking-budget`, and native `--inherit-from` are rejected
by the new production CLI. Their historical record readers/controllers remain
available; they are not a route for mixing old and new sampling semantics.

## Commands (from `code/`)

```sh
.venv/bin/python scripts/deepeye_bird_interact_run.py prepare --run-dir /absolute/new-native-run --precompute-dir /absolute/frozen-precompute --few-shot-source /absolute/train.json
.venv/bin/python scripts/deepeye_bird_interact_run.py resume --run-dir /absolute/new-native-run --precompute-dir /absolute/frozen-precompute --few-shot-source /absolute/train.json
.venv/bin/python scripts/rc_evaluation/deepeye/cli.py prepare --source-run /absolute/new-native-run --run-dir /absolute/new-rc-run --target-stage sql_generation --condition rc --rc-lite /absolute/lite-contracts.json --rc-full /absolute/full-contracts.json
.venv/bin/python scripts/rc_evaluation/deepeye/cli.py run --run-dir /absolute/new-rc-run
.venv/bin/python scripts/rc_evaluation/deepeye/cli.py token-pairs --run-dir /absolute/new-rc-run
```

`prepare`, `inspect`, `export`, `samples`, `renew-samples`, and `token-pairs` are
offline. `run`/`resume` can make real model and PostgreSQL requests. The example
paths are placeholders; this implementation task did not run those experiments.

Both CLIs expose exact-group offline recovery inspection and explicit renewal:

```sh
.venv/bin/python scripts/deepeye_bird_interact_run.py samples --run-dir /absolute/run
.venv/bin/python scripts/deepeye_bird_interact_run.py renew-samples --run-dir /absolute/run --group-id EXACT_HASH_FROM_SAMPLES --decision 'Explicit operator reason for another four attempts'
```

Use the same subcommands on the RC CLI for an RC run. Renewal appends an unfinished
audit attempt and a `sampling_retry_authorized` event; only exhausted/fatal
unfinished indices gain four new attempts. Successes and ordinary remaining
budgets stay intact. It launches no work; a later explicit `resume` uses the normal
sampling path. Normal resume never renews exhausted samples automatically.

SIGINT/SIGTERM uses the recorder/runtime's same stop event. The first signal
rejects new/queued work and drains active requests; a second cancels active HTTP.
Paused stages remain unfinished. Authentication/configuration failures propagate
the original fatal error after cleanup. Local cancellation does not prove the
server stopped work or billing.

## Records and comparison

The manifest identifies `profile=shared-sampling-v1`,
`runtime.version=shared-sampling-runtime-v1`, `scheduler.mode=all_questions`, all
explicit limits, and source hashes. Stage payloads retain their sampling source
version. Changed source/configuration rejects reuse: prepare new native and RC
RunStores instead of mutating old records or importing legacy successful prefixes.

Runtime status separates limits, coordinator/sample/request occupancy and peaks,
and independent PG occupancy. `api_request` means a logical SDK invocation;
`request_dispatch` records admission, actual send/transfer where available,
queue/retry/pacing/service timing. A reservation stopped before dispatch may be
spent/uncertain and is not evidence of a sent request.

`token-pairs` compares the target's retained successful sample usage with the
native source snapshot, requiring complete groups, retained slot evidence,
successful stage status, complete usage, and proven RC participation. Failures,
unknown usage, legacy sampling metrics, missing stages, and native no-model
targets are excluded with reasons; excluded deltas are `null`, never fabricated
zero. Reasoning tokens remain a subset of completion tokens and are not added to
totals. Existing `observed_usage`/evaluation `_usage` keep their original role as
all-current-attempt reported/new-incurred usage, including failed retries; that
ledger is not the retained-sample comparison. SQL-result evaluation/comparison
remains separate and does not drive generation or sampling acceptance.

A zero-native-call target reuses its source and records RC not participating.
Restored samples with proven RC participation are still model-participating even
when no new calls occur. A called target missing RC fails explicitly.

## Validation boundary

C4 exercises both production entrypoints with fake SDK/PG boundaries, 605 queued
questions, low-cap nested scheduling, RC retries/context, retained usage,
downstream input changes, stop/fatal paths, and cleanup. C3's high-capacity evidence
was local fake/loopback capacity, not remote throughput. C5/C6 and live remote/PG
validation remain pending; no new remote capacity or experiment result is claimed.
