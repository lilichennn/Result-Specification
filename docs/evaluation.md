# Evaluation and saved records

Inference records, SQL assessments, and summary tables serve different purposes. Keep the original run directory until all exports have been verified. A table alone does not contain enough information to reconstruct a run.

## SQL correctness

The shared execution comparison checks a predicted SQL query against its reference query on the same database. The number and positions of result columns must agree. Row order is ignored; duplicate rows retain their multiplicities. Column names are not used as the correctness criterion. A query that fails or times out is not a successful match. Keep execution failures separate from successful-but-different answers in subsequent analysis.

Database access is required when executing or re-evaluating SQL. Once the compact assessment files have been exported, their saved match flags, execution status, candidate identities, and counts can be analyzed offline. The saved execution timeout and database identity are part of the evaluation context; do not silently mix assessments made with different databases or limits.

## What to count

| Component | Recommended unit and denominator |
| --- | --- |
| Schema Linking | Compare selected columns/tables with the reference set. Micro precision/recall pools set counts across questions; macro column recall averages per-question recall over questions with a nonempty valid reference. Report excluded questions separately. |
| Generation | Count all actual generated candidates, and separately count questions with at least one matching candidate. Candidate match rate and question coverage answer different questions. |
| Revision | Compare the same incoming SQL before and after revision. Distinguish unique incoming SQL from repeated candidate slots and report whether RS actually reached a model request. |
| Selection | Compare the selected SQL on a fixed candidate pool. For the two-candidate, one-correct/one-wrong subset, report wrong-to-right and right-to-wrong transitions and its denominator. |

Stage-specific DeepEye runs normally restart only the selected stage from a native snapshot; downstream execution is optional. Do not present a stage-local candidate statistic as the accuracy of an end-to-end rerun.

DAIL has four logical modes: `native`, `rc_first`, `rc_second`, and `rc_both`. Their records can reference shared physical executions when the ordered retrieved-example IDs match. Count a mode's resolved outputs once, not every stored event. DIN stores native and RS-assisted stage outputs as distinct records. Its schema-filtering/linking comparison is an additional operation; running Generation or Revision does not implicitly run that comparison.

## Token accounting

Retain per-attempt usage and distinguish successful sampling usage from usage across all reported attempts. Failed requests may not report usage. Missing usage is not zero usage. Reasoning tokens may be included in completion tokens by the provider; subtract them only when an explicit reasoning-token field is available. Do not add reasoning tokens to a completion total that already includes them.

## Records, reruns, and exports

- DeepEye keeps an incremental `run.sqlite3` per run, with manifests, question progress, stage payloads, and model-call records. Campaigns additionally have a `campaign.sqlite3` that links the native and stage-specific jobs.
- DAIL stores incremental versioned question/round/mode records. Reused round executions are referenced rather than regenerated. A question rerun creates a replacement version; use the current-version export, not a concatenation of every historical version.
- DIN stores incremental records and versioned question/stage outputs. Export commands resolve the active records before building handoff files.

Use the supported exporters after a run has stopped or completed. Do not copy a live SQLite database without its write-ahead log or a consistent backup. Export into a new destination; verify its index and verification information before deleting any source files. Share the raw records when detailed prompt, retry, or token analysis is needed, and the compact assessment directory for routine offline accuracy analysis.

## Commands

All commands run from the repository root. Replace the example paths with the exact batch directories printed by the runner.

```bash
uv run python -m scripts.rc_evaluation.dail_sql.cli export \
  --batch outputs/dail_sql/batches/my-run \
  --output outputs/analysis/dail_sql

uv run python -m scripts.rc_evaluation.din_sql.cli export-handoff \
  --batch outputs/din_sql/batches/my-run \
  --output outputs/analysis/din_sql

uv run python -m scripts.analysis.deepeye.analyze --help
```

The DAIL and DIN exports can execute SQL and therefore need database access. DeepEye's `analyze` entry separates extraction and SQL evaluation; `scripts.analysis.deepeye.summarize` performs offline summarization afterwards. Inspect the command help before running. Other tools under `scripts/analysis/deepeye/` read the resulting assessment files. The bundled `data/reference/schema_linking_annotations.jsonl` contains reference labels, not prediction results or raw model-response logs.

See [Analysis commands](analysis.md) for the complete table-generation sequence and input requirements. Plotting programs under `scripts/analysis/` accept explicit input paths. They do not constitute an independent correctness evaluator. Keep the source tables and evaluation settings alongside generated figures.
