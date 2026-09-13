# Task 2 implementation report

Implemented `comparison.py`, `evaluation.py`, `tests/test_comparison.py`, and `tests/test_evaluation.py`. No edits were made to the existing adapters, native DeepEye implementation, entrypoints, dependencies, environment, or baseline runs. No commits, model calls, PostgreSQL queries, or network requests were made.

## Delivered behavior

- `compare_results(predicted, reference)` reports both bag and ordered equality. It retains duplicate rows and column positions, ignores column aliases, and separates NULL, bool, text, and numbers. Finite numeric values use exact rational equality, including Decimal versus int without precision loss. Binary float 0.1 therefore differs from Decimal 0.1; no hidden tolerance is applied.
- Standard JSON objects and arrays, PG arrays, bytes, UUID, date, time, and datetime are supported with explicit typed representations. JSON object key order is ignored; nested array order is retained. Aware timestamps compare exact UTC microseconds; naive timestamps remain distinct. Aware time uses offset-adjusted microseconds without wrapping across a day. Native results do not include PG type OIDs, so JSON arrays and PG arrays share structural sequence semantics. Ranges/custom types, nonfinite numbers, malformed shapes and failed SQL remain unavailable instead of being stringified or counted equal.
- `evaluate_run(...)` reads an immutable inference snapshot and writes a separate append-only RunStore. A completed evaluation is reused only when the full evaluation identity matches. Changed inference snapshots, reference files, database-version declarations or executor identities reject reuse. Source checkpoints are checked against their payload/input hashes and, when the source store is supplied, original successful attempt identities. Source and inference stores remain read-only.
- References are loaded exclusively by the evaluator from exact split/instance-id Query records. Native `selected_database` is checked against the run binding. One nonempty `sol_sql` is required; duplicate references and nonempty preprocessing/cleanup fields, including `clean_up_sqls`, are rejected as explicit unavailable states. Reference file, record, SQL, database and operator-declared database-version identities are retained. The version label is not represented as an automatically detected database snapshot.
- Each SQL execution has start/result events, SQL hash, database identity/version, execution duration and native-shaped results. Identical SQL is cached within a task evaluation. Exceptions, timeouts and unsupported serialization are explicit unavailable results. The native executor uses the existing read-only PostgreSQL adapter and checks host/port/principal/TLS consistency; its actual connection and execution-policy identity is saved without credentials.
- Generation reports slots, unique SQL, successful executions, correct/unknown candidates and pool agreement. Revision pairs the original generation slots by index, retains missing slots as unknown, reports repair/harm/unchanged correctness and SQL unchanged status, and exposes extra output slots. Both candidate pools remain available in the report.
- Selection reports final agreement, conditional selection when the revised pool contains a correct candidate, unknown eligibility, shortlist retention and trace-supported branch decisions. Single-shortlist, fallback, pairwise and consistency-shortcut branches use native trace output and recorded threshold evidence, never native frozenset result hashes. Replayed current/downstream selection uses bound source trace with explicit provenance; unavailable trace yields unknown.
- Schema Linking reports retained table/column size and conservative reference SQL coverage. Ambiguous unqualified columns, stars, CTE/subquery lineage and unresolved schema-qualified tables remain unknown. USING and NATURAL joins report `implicit_join_columns_unresolved` instead of overlooking their implicit join columns; known table coverage and retained table/column counts remain available. RC population and Meta fields are never ground truth. Optional downstream selected SQL is reported only when present in the new run.
- Summary rates retain the fixed task denominator and explicit unknown states. Conditional Selection uses only positively established eligibility and separately reports unknown eligibility/outcomes. Current-run API usage includes failed/unfinished retries, uses the existing observed-usage validator, reports incomplete usage metadata, and excludes source/historical cumulative costs. Actual RC counts are reconstructed from current-run target-stage API request events using each task's manifest-bound complete RC block, including interrupted attempts with no finish payload and recovered retry history. Finish metadata is retained for provenance but is not the count source. Other tasks/stages, wrong contract blocks and source events are excluded; condition `none` always reports zero RC participation. Canonical RC runs must match their bound production hash (which includes the template) before evaluation; mismatches explicitly require the corresponding production version.
- `compare_evaluations(...)` writes another append-only RunStore and reports both bag and ordered correctness transitions. Pairing requires the same target, ordered item bindings, source identity/checkpoints, repeat, downstream setting, runtime/PG configuration, production-source hashes, database version, actual evaluator connection identity and reference hashes. Expected `none` versus `rc` contract differences are allowed.

## RED/GREEN evidence

Initial comparator/evaluator tests failed because the new modules were absent. The first evaluator attempt also exposed the required native `app` import path; rerunning with the baseline path confirmed the intended missing-module RED before implementation.

Subsequent observed RED → GREEN cycles covered:

1. Strict equality, NULL/empty/shape/duplicate boundaries, fixed denominators, separate RunStore persistence/resume and revision slot repair/harm → 10 tests passed.
2. Genuine native reference field names (`selected_database`, `clean_up_sqls`), nested API response usage, candidate pool and conditional summary metrics → 14 tests passed.
3. Trace-supported consistency shortcuts and replay provenance → 18 tests passed.
4. Structured JSON/array, typed temporal/binary/UUID values → 20 tests passed.
5. Source-payload tampering and pairing differences in PG identity/upstream/production sources → 22 tests passed.
6. Permitted RC contract difference, ordered paired metrics and real controller/native tagged-payload integration → 23 tests passed.
7. Incomplete API usage and unchanged-upstream source replay → 24 tests passed.
8. Actual evaluator connection identity and TLS mismatch rejection before SQL → 25 tests passed.
9. Final review P2: interrupted RC requests initially reported 0 instead of 1; recovered retry/source/wrong-stage/wrong-contract/none fixtures then verified event-derived counting → 27 tests passed.
10. Final review P2: USING, NATURAL and NATURAL LEFT JOIN fixtures initially reported available column coverage; all now remain unknown while retaining table/count diagnostics → 28 tests passed.
11. Adjacent template-integrity regression initially accepted an older bound production hash; canonical RC evaluation now rejects it before SQL/output creation → 29 tests passed.

Final verification command (from repository root):

```sh
PYTHONPATH=code:code/baselines/DeepEye-SQL code/.venv/bin/python -m unittest scripts.rc_evaluation.deepeye.tests.test_comparison scripts.rc_evaluation.deepeye.tests.test_evaluation
```

Latest result after final review fixes: `Ran 29 tests in 0.244s ... OK`. Each of the two P2 fixes and the adjacent template-integrity guard had an observed failing regression before its production fix. Evaluation tests block `socket.socket.connect` and use synthetic SQL results with real temporary RunStores. The default native execution context was inspected/tested without issuing SQL. Whole-suite integration and protected-file hash verification are owned by the coordinating agent.

## Explicit limits

This is strict result agreement, not official BIRD-Interact accuracy. It does not execute benchmark test cases or required setup/cleanup, detect remote database changes, resolve arbitrary SQL lineage, apply approximate numeric matching, or infer missing trace branches. It serializes evaluation execution within one invocation; the PG environment context is intended for CLI use. An unknown evaluation is a completed diagnostic record; to intentionally reexecute SQL after an external database change, use a fresh evaluation output directory and a new database-version declaration.
