# Analysis commands

Run commands from the repository root with `uv run python -m`. Analysis consumes saved run records and assessments. Generated reports default to `outputs/analysis/`. No prediction results or precomputed summary tables are bundled.

## DeepEye preparation and assessment

The [analysis entry point](../scripts/analysis/deepeye/analyze.py) separates record extraction from SQL execution. The five supported groups are `bird_dev`, `spider_dev`, `spider_test`, `bird_interact_lite`, and `bird_interact_full`. These tools validate the complete group membership. The stage metric aggregators require all five groups; extraction, assessment, summarization, verification, and supplementary diagnostics also accept `--group` to process one group at a time.

```bash
# Campaign directories must contain campaign.sqlite3 and completed original/RS jobs.
uv run python -m scripts.analysis.deepeye.analyze extract \
  --campaign-root outputs/deepeye/campaigns \
  --campaign-pattern '{group}' \
  --analysis-dir outputs/analysis/deepeye

# Executes benchmark SQL. Omit --group to assess all five groups.
uv run python -m scripts.analysis.deepeye.analyze evaluate \
  --analysis-dir outputs/analysis/deepeye --group bird_dev --workers 4 \
  --env-file config/.env

# Run after all required group assessments have completed.
uv run python -m scripts.analysis.deepeye.summarize \
  --analysis-dir outputs/analysis/deepeye
```

Replace the campaign root and naming pattern with the directories created by your campaign commands. The pattern defaults to `{group}` and may include a run suffix. Campaign job paths, frozen manifest reference paths, and database paths must remain accessible. Extraction reads those sources and writes `GROUP/offline.json`; it does not relocate their provenance.

The `evaluate` action requires configured benchmark databases. PostgreSQL credentials default to `config/.env`; `--env-file` overrides that location. `--workers` controls question concurrency; `--query-workers` enables nested query concurrency for SQLite only. Assessment resumes its `GROUP/evaluation.sqlite3` cache. Extraction reuses an existing `offline.json`, and summarization reuses an existing per-group summary. Other JSON report writers refuse to overwrite an existing report, so use a fresh analysis destination for a new analysis.

## Stage metrics and audits

All commands below accept `--analysis-dir outputs/analysis/deepeye`. Unless noted otherwise, they read saved records without executing benchmark SQL or making model requests.

| Tool | Required inputs | Output |
| --- | --- | --- |
| [`analyze extract`](../scripts/analysis/deepeye/analyze.py) | Campaign ledgers, source RunStores, binding reference files | `GROUP/offline.json` |
| [`analyze evaluate`](../scripts/analysis/deepeye/analyze.py) | Offline snapshots and benchmark database access | `GROUP/evaluation.sqlite3` |
| [`summarize`](../scripts/analysis/deepeye/summarize.py) | Offline snapshots and assessment caches | `GROUP/summary.json`, `GROUP/changes.json`; all-group `summary_all.json` |
| [`generation_metrics`](../scripts/analysis/deepeye/generation_metrics.py) | Assessment caches and per-group summaries | `generation_metrics.json` |
| [`revision_metrics`](../scripts/analysis/deepeye/revision_metrics.py) | Offline snapshots, assessment caches, summaries, original RunStore events | `revision_metrics.json` |
| [`selection_metrics`](../scripts/analysis/deepeye/selection_metrics.py) | Offline snapshots, assessment caches, summaries, original RunStore events | `selection_metrics.json` |
| [`selection_active_mixed_metrics`](../scripts/analysis/deepeye/selection_active_mixed_metrics.py) | Selection metrics and assessment caches | JSONL on stdout |
| [`selection_prompt_audit`](../scripts/analysis/deepeye/selection_prompt_audit.py) | Selection metrics, offline snapshots, original RunStore events | `selection_prompt_audit.json` |
| [`verify`](../scripts/analysis/deepeye/verify.py) | Offline snapshots, cached query results, source successful-sample events | `GROUP/verification.json` |
| [`supplement`](../scripts/analysis/deepeye/supplement.py) | Offline snapshots and assessment caches | `GROUP/supplement.json`, `GROUP/sampling_sensitivity.json` |
| [`schema_linking_metrics`](../scripts/analysis/deepeye/schema_linking_metrics.py) | Offline snapshots | Conservative SQL-parser diagnostic `schema_linking_metrics.json` |
| [`sampling_exhaustion`](../scripts/analysis/deepeye/sampling_exhaustion.py) | Offline snapshots, campaign ledgers, original RunStores | New subdirectory containing summary, question, and exhausted-slot JSON files |

After summarization, run the stage metrics in this order:

```bash
uv run python -m scripts.analysis.deepeye.generation_metrics --analysis-dir outputs/analysis/deepeye
uv run python -m scripts.analysis.deepeye.revision_metrics --analysis-dir outputs/analysis/deepeye
uv run python -m scripts.analysis.deepeye.selection_metrics --analysis-dir outputs/analysis/deepeye
uv run python -m scripts.analysis.deepeye.selection_active_mixed_metrics --analysis-dir outputs/analysis/deepeye
```

Generation counts actual candidate slots, including repeated SQL. Unknown comparisons remain in the denominator without being treated as proven mismatches. Revision's `rc_involved_unique_units` describes normalized unique SQL units that actually participated in RS-augmented Revision; all-slot diagnostics are also retained. The active mixed-pair Selection report includes cases where RS participated and the RS variant's shortlist has exactly two candidates and one confirmed match. It verifies those rows against the saved assessment checksums.

Token summaries retain successful-sample usage, with separate complete and equal-budget cohorts. Exhaustion counts a fixed sampling slot whose final result remains unsuccessful after all four attempts; success on the fourth attempt does not count. This audit validates the configured Qwen3.8 2.4T model, four-attempt budget, timeout, and complete group membership. It accepts `--output-dir` for a new report directory and optional `--campaign-root`/`--campaign-pattern` overrides for campaign locations.

## Annotation-based column recall

Use [column_recall](../scripts/analysis/column_recall.py) for column macro recall against the [reference annotations](../data/reference/schema_linking_annotations.jsonl). The `schema_linking_metrics` tool above is a separate conservative SQL-parser diagnostic.

```bash
uv run python -m scripts.analysis.column_recall \
  --annotations data/reference/schema_linking_annotations.jsonl \
  --deepeye-analysis outputs/analysis/deepeye \
  --output outputs/analysis/deepeye_column_recall.json

uv run python -m scripts.analysis.column_recall \
  --annotations data/reference/schema_linking_annotations.jsonl \
  --linking-details outputs/analysis/din_sql/evaluation/linking_details.jsonl \
  --output outputs/analysis/din_column_recall.json
```

`--deepeye-analysis` reads all five `GROUP/offline.json` files. `--linking-details` accepts DIN or DAIL `evaluation/linking_details.jsonl`, whose records contain a string annotation `task_key`, a `group`, and `base`/`rc3` objects with `status` and normalized physical `columns` pairs. Raw `record_export/linking.jsonl` is a different format and is not accepted. Use the path produced by your exporter.

Reference JSONL contains unique `task_key` values, an annotation `status`, and `required_tables`/`required_columns` for resolved labels. The scorer uses the existing set-metric implementation and exact physical names. It averages per-question column recall over resolved annotations with nonempty required-column sets. Empty or failed predictions score zero for nonempty gold; non-resolved annotations and empty gold-column sets are excluded. Duplicate prediction identities or missing labels fail validation. Reports include eligible-question counts and unmatched-annotation coverage, including when supplied linking details cover only a subset.

## DAIL compact facts

The [offline facts exporter](../scripts/analysis/export_dail_facts.py) extracts the current sealed versions and four DAIL modes without executing benchmark SQL:

```bash
uv run python -m scripts.analysis.export_dail_facts \
  --batch outputs/dail_sql/batches/my-run \
  --output outputs/analysis/dail_facts
```

The batch must contain `manifest.json`, `current.sqlite3`, and `group-<hex>/run.sqlite3` stores. The exporter verifies event checksums and references, then creates a new `record_export/` containing `versions.jsonl`, `modes.jsonl`, `rounds.jsonl`, `candidates.jsonl`, `failed_questions.jsonl`, and `verification.json`. It does not score SQL against references; the verification record states `reference_sql_evaluation_complete: false`.

For a source package accepted by the DAIL Linking comparison, use the completed compact SQL export and the public handoff assembler. This copies the saved SQL evaluation without rerunning benchmark queries; it creates and verifies the current record facts, `inputs_manifest.json`, `evaluation/latest.json`, indexed file hashes, and a completion seal. Both version sets must match the current batch.

```bash
uv run python -m scripts.rc_evaluation.dail_sql.cli export \
  --batch outputs/dail_sql/batches/my-run --evaluation-profile deepeye

uv run python -m scripts.rc_evaluation.dail_sql.cli export-handoff \
  --batch outputs/dail_sql/batches/my-run \
  --compact outputs/dail_sql/batches/my-run/exports/compact/REPLACE_WITH_RETURNED_DIRECTORY \
  --output outputs/analysis/dail_source
```

Use the exact directory printed by `export` for `--compact`; `export-handoff` requires a new `--output` directory. A batch containing a selected subset can be exported when every selected question has a sealed current version and a completed compact evaluation. The standalone facts export above remains useful when SQL evaluation is intentionally absent, but it is not a Linking source package.
