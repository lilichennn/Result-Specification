# Task 1 implementation report

## Scope

Implemented only the assigned RC contract and injection surface:

- `contracts.py`
- `injection.py`
- `rc_prompt.txt`
- `tests/test_contracts.py`
- `tests/test_injection.py`

`contracts.py` also re-exports `render_rc_block` because the Task 3 controller imports that fixed plan API from the contracts module. The implementation remains in `injection.py`.

No existing baseline, adapter, entrypoint, environment, dependency, or Git file was edited. No provider or PostgreSQL call was made and no experiment was started.

## RED evidence

Initial missing-feature run:

```text
code/.venv/bin/python -m unittest \
  scripts.rc_evaluation.deepeye.tests.test_contracts \
  scripts.rc_evaluation.deepeye.tests.test_injection
```

Exit 1: both test modules failed to import because `contracts.py` and `injection.py` did not exist.

Controller compatibility regression:

```text
code/.venv/bin/python -m unittest \
  scripts.rc_evaluation.deepeye.tests.test_contracts.LoadContractsTest.test_contract_module_exposes_render_function_for_controller
```

Exit 1: `render_rc_block` was not exposed by `contracts.py`.

Actual-request scope regression:

```text
code/.venv/bin/python -m unittest \
  scripts.rc_evaluation.deepeye.tests.test_injection.CountRCRequestsTest.test_block_in_trace_metadata_but_not_messages_does_not_count
```

Exit 1: a block present only in trace metadata was incorrectly counted as model participation. The implementation was then restricted to `event.payload.kwargs.messages`.

## GREEN evidence

Fresh focused, PostgreSQL-wrapper lifecycle, and compilation verification:

```text
code/.venv/bin/python -m unittest \
  scripts.rc_evaluation.deepeye.tests.test_contracts \
  scripts.rc_evaluation.deepeye.tests.test_injection \
  tests.test_deepeye_postgres_prompts

code/.venv/bin/python -m py_compile \
  scripts/rc_evaluation/deepeye/contracts.py \
  scripts/rc_evaluation/deepeye/injection.py \
  scripts/rc_evaluation/deepeye/tests/test_contracts.py \
  scripts/rc_evaluation/deepeye/tests/test_injection.py
```

Exit 0: 23 tests ran and passed; compilation exited 0.

Covered behaviors include strict `(variant, instance_id)` joins; duplicate, missing, failed, malformed, and identity-mismatched records; normalized complete Round 1/Round 2 values; exact source and canonical record hashes; no generation calls; fixed prompt text containing only the definition, six generic meanings, final Round2 JSON, and generic consultation request; all seven real PromptFactory methods after PostgreSQL support; no-op controls; descriptor restoration; stage independence; nested and threaded ContextVar isolation; and model-participation counting only from complete blocks in SDK request messages.

## Integration concern for root

The latest combined Task 1 plus `test_integration` run reaches 17 passing Task 1 tests, then stops in Task 3 manifest preflight before prompt execution:

- `none`: `Experiment condition and bound contracts disagree`
- `rc`: `RC contract db_id differs from source input`

This is in the integration fixture/controller-owned validation path, not in the Task 1 prompt code. Root should update the manual integration manifest/RC fixture to the latest Task 3 contract and rerun the native fake-SDK integration test during coordinated review.
