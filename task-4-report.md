# Task 4 report: shared DeepEye preparation and RC execution

## Delivered interfaces

- Native: `prepare_run(store, tasks, *, selected_keys=None, checkpoints=None) -> PreparedRun`; `select_unfinished(store, tasks, *, prepared=None)`; `run_pipeline(..., runtime=None, prepared=None)`.
- RC: `prepare_experiment(store, checkpoints=None, *, item_keys=None) -> PreparedExperiment`; `unfinished_keys(store, *, prepared=None)`; `run_experiment(..., item_keys=None, prepared=None)`.
- RC production entry: `execute_run(store, environment, *, item_keys=None, prepared=None)` reuses one full `PreparedExperiment` for frozen checks, unfinished selection, preflight, and execution.
- Generic CLI mappings: repeated `prepare --rc PARTITION=PATH` and `evaluate --reference PARTITION=PATH`; legacy Lite/Full flags remain supported and duplicate partition mappings are rejected.

Both prepared objects bind the originating store, its database revision, the frozen manifest digest, restored per-item plans, indexed attempts, and the full-store verification verdict. A write after preparation makes the object stale.

## Behavior and efficiency

- The complete RunStore is verified and its attempts are indexed once per preparation.
- Only selected native/RC item objects are restored; checksum or lineage damage outside the selection still fails full validation.
- Fingerprints use one frozen manifest digest plus the item input and preceding stage chain under `manifest-digest-v2`.
- Generic native types and typed external IDs survive source snapshot/restore. RC joins use the explicit partition and the original typed ID.
- RC resumes strictly reload every bound external RC record, but reuse already restored prepared states rather than restoring source checkpoints a second time.
- Replay-only target stages do not construct native model/database resources. Call-enabled execution uses the shared workload config and backend context, so SQLite and BigQuery paths do not install PostgreSQL support.
- Native completion, retry, partial-sampling, independent target-stage execution, optional downstream continuation, RC participation, and token-trace semantics remain covered by the existing suites.

## TDD evidence

Initial focused run:

```text
Ran 84 tests in 2.932s
FAILED (failures=11, errors=1)
```

The failures exposed the old RC CLI environment signature, BI-only preparation/resource routing, duplicate preparation, checkout-relative CLI interpreter assumption, and the missing generic native-to-RC path. The new reference mapping test separately failed because `--reference` was not defined and argparse treated it as an ambiguous abbreviation.

Focused Task 4 green run:

```text
Ran 89 tests in 8.807s
OK
```

This includes real 5/50/500-item native and RC preparation tests. For RC, each size observes exactly one `verify`, one `validate_manifest`, one `attempts` scan, and one source restore per item across preparation, unfinished selection, and execution.

Full RC offline suite:

```text
Ran 105 tests in 8.497s
OK
```

No live model, database, or network experiment was started.
