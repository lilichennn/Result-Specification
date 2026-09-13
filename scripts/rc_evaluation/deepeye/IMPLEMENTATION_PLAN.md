# DeepEye RC Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Follow TDD; use apply_patch; do not commit, start paid runs, or alter existing files outside your ownership.

**Goal:** Implement four-stage isolated RC interventions, reproducible reruns, append-only results and independent result metrics.
**Architecture:** Fixed RC prompt wrapper + independent stage controller over existing RunStore/native services + separate evaluator. One condition/target/repeat per run.
**Tech Stack:** Python 3.12, existing code/.venv, unittest, RunStore/psycopg/sqlglot.
**Spec:** DESIGN.md in this directory (approved conversation plus user's latest restrictions).

## Global Constraints

- Work in the user's requested new directory in the existing checkout; do not alter old adapter, baseline, entrypoints, .env, README, git state or dependencies.
- All remote calls are opt-in commands. Implementation tests block networking and do not use real providers.
- RC uses final Round2 six fields, static definitions and static reference request only; no extra question-specific hints. Zero-model successful baseline target is reused, never rerun.
- Independent task ownership below permits parallel work without overlapping files. Do not revert others' changes or spawn additional workers. Root coordinates integration/review.

## Task 1: RC loading and prompt injection

Own `contracts.py`, `injection.py`, `rc_prompt.txt`, `tests/test_contracts.py`, `tests/test_injection.py`.

Interfaces consumed: existing Round1RC/Round2RC dataclasses; tasks as `(variant, DataItem)`; existing PG PromptFactory installation.
Interfaces produced:

```python
load_contracts(paths: dict[str, Path], tasks: list[tuple]) -> dict[str, dict]
# each dict: task_key, db_id, question, evidence, round1, round2,
# source_file, source_file_sha256, record_sha256
render_rc_block(contract: dict) -> str
rc_context(stage: str, task_key: str, contract: dict | None)  # context manager
install_rc_prompts()  # context manager, wraps current PromptFactory methods
count_rc_requests(events: list[dict], block: str) -> int
```

- [x] Write tests for duplicate/missing/misaligned/failed records and immutable prompt definition; observe failure before code.
- [x] Implement strict loading by split/instance_id, full Round2 and source hashes; no generate function calls.
- [x] Write prompt tests against real PromptFactory and PG wrapper for all stages, no-op none, nested context, thread isolation, complete block in actual API request shape, progressive formatting does not count as model usage.
- [x] Implement reversible wrapper over seven format methods; define fields faithfully, omit examples/task analyses. Run focused suite.

## Task 2: Independent metrics and reference evaluation

Own `comparison.py`, `evaluation.py`, `tests/test_comparison.py`, `tests/test_evaluation.py`.

Interfaces consumed: RunStore attempts containing native `payload.artifact`; source native run; source bindings fields unchanged. New experiment manifest specified by Task 3 below.
Interfaces produced:

```python
compare_results(predicted: dict, reference: dict) -> dict
# result dicts native result_type/result_cols/result_rows;
# return comparable, bag_equal, ordered_equal, reason
evaluate_run(run_dir: Path, output_dir: Path, reference_paths: dict[str, Path],
             *, env_file: Path, database_version: str,
             source_run_dir: Path | None = None, execute_fn=None) -> dict
# separate append-only evaluation RunStore; no conversation calls;
# execute_fn(task_key, database_id, sql) returns native-shaped dict for offline tests
compare_evaluations(left_dir: Path, right_dir: Path, output_dir: Path) -> dict
```

- [x] Tests first: duplicates, unordered/ordered difference, NULL/empty/shape, Decimal versus int, bool/string not number, nonfinite/unhandled types, missing reference and SQL failures.
- [x] Implement exact bag and ordered comparison without native frozenset hashes. Strict finite numeric equivalence preserving precision; don't silently stringify unsupported values.
- [x] Implement reference Query lookup by exact split/id, require one nonempty sol_sql, reject required preprocessing/cleanup for this first version. Save hashes, source identity, explicit database version, SQL execution records and unknown/error statuses without mutating input run.
- [x] Implement per-stage candidate pool/paired repair/selection/coverage diagnostics and paired reports. Keep fixed denominator and unknown states. Structural gold coverage is best effort, unknown on unresolved SQL.
- [x] Test with real temporary RunStores and synthetic executor; no PG/model requests.

## Task 3: Source snapshots, controller and CLI

Own `__init__.py`, `__main__.py`, `source.py`, `runner.py`, `cli.py`, `USAGE.md`, `tests/__init__.py`, `tests/test_source.py`, `tests/test_runner.py`, `tests/test_cli.py`. If clean separation needs one further small runtime helper, announce filename to root before adding.

Interfaces consumed: Task 1 functions above; Task 2 evaluation functions above; existing `prepare_inputs`, `build_effective_config`, `bounded_runner_factory`, `build_runtime_config`, `admission_context`, RunStore/TraceRecorder/resources/slots. Preserve exact baseline model and sampling budgets; never bypass bounded wrapper.

Experiment manifest contract:

```python
{
  'format': 'deepeye-rc-evaluation-run-v1',
  'target_stage': 'schema_linking|sql_generation|sql_revision|sql_selection',
  'condition': 'none|rc', 'repeat_id': '1', 'continue_downstream': False,
  'source_run': '/absolute/path', 'source_manifest_fingerprint': '...',
  'items': [  # retain baseline binding fields
     {'task_key': 'lite/example_1', 'database_id': 'example', ...}
  ],
  'effective_config': {...}, 'sources': {...}, 'contracts': {...},
  'source_checkpoints': {...}  # per task complete validated prefixes/payloads/hashes/API counts
}
```

Source snapshot payloads may be stored in separate immutable RunStore seed attempts instead of directly in manifest, provided content hashes and source attempt IDs bind every record. Communicate final format to Task 2. Stage output uses native stage names and `payload.artifact` unchanged; extra fields include `execution_origin` (`executed` or `reused_no_native_llm_call`), `rc_participation` with actual request count/reason, and source provenance. Reuse is never a new provider call. Use `RunStore.completed` and append retries; no updates to old records.

- [x] Write real-temp-RunStore tests: partial source cannot prove no-call; complete zero-model target reused without invoking factory; call-enabled target reruns from pristine same prefix; no source target/downstream leaks; gold empty; no condition mutation resume; partial failures preserve earlier rows.
- [x] Implement snapshot/controller with per-item independent slots and native resource cleanup. Target stage plus opt-in downstream only. For zero-call targets replay source result and record nonparticipation, no model client construction if whole run is replay-only.
- [x] Add CLI `prepare/run/resume/inspect/export/evaluate/compare`, explicit source/target/condition/RC files/input filters, familiar concurrency/budget configuration. `prepare` offline; all source/model/budget/input mismatches rejected before paid work. Bind new-directory production hashes without changing old hash functions.
- [x] Add offline CLI path tests for root cwd and code cwd, using code/.venv; evaluate accepts explicit gold paths and database-version label, never pipeline inputs.
- [x] Write usage in plain Chinese including zero-call reuse, baseline versus fresh none control, comparator limits and no automatic paid experiment.

## Integration and review

- [x] Run all new tests with networking blocked; verify original full suite and unchanged protected hashes.
- [x] Real-cache offline prepare for both conditions: all 10 completed Schema Linking tasks and 9 fully completed pipelines for the other stages, plus zero-call run/resume/export; no LLM/PG. The tenth pipeline remains monitored separately and is not treated as complete.
- [x] Independent review for spec compliance, contamination, lifecycle/context races, persistence and evaluation denominators; fix covering tests before declaring complete.
- [x] Record baseline monitoring/completion separately; retain all original runs. No commit or push in this task.
