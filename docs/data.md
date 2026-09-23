# Data layout

Reusable inputs live under `data/<group>/`. Generated experiment records belong under `outputs/`, not alongside inputs.

| Group | Questions | Database engine | Round-3 specifications |
| --- | ---: | --- | ---: |
| `bird_dev` | 1,534 | SQLite | 1,534 |
| `spider_dev` | 1,034 | SQLite | 1,034 |
| `spider_test` | 2,147 | SQLite | 2,147 |
| `bird_interact_lite` | 195 | PostgreSQL | 195 |
| `bird_interact_full` | 410 | PostgreSQL | 410 |

These five groups contain 5,320 questions, each with a successful Round-3 specification.

## Per-group files

- `<group>.json`: question records with `index`, `db_id`, `question`, and `evidence`.
- `meta/`: database metadata exposed to the methods and RS generation, organized by database.
- `filtered_meta.json`: saved RS-filtered metadata, keyed by the string form of the question's `index`. Each record contains `result` (a metadata array) and `status` (`success` and `reason`), matching the output format of `scripts/filter_meta.py`.
- `rc.json`: question-aligned RS records with `rc_round1`, `rc_round2`, `rc_round3`, and per-round status/error fields. These are the actual field names; see [Terminology](../README.md#terminology).
- `gold_sql.json`, `gold_sql_schema_linking.json`, and `gold_sql_preparation.json`: reference SQL/dependency information and preparation provenance, where available. The runners' configuration identifies the actual evaluation reference source.
- `data/reference/schema_linking_annotations.jsonl`: final model-assisted reference labels for the five main groups. It does not include annotation request logs or a run database.
- `data/reference/schema_filter_manifest.json`: provenance, model, per-group counts, file hashes, and the question/RS/metadata identities associated with `filtered_meta.json`.

Question identifiers are scoped by group. BIRD-Interact preserves `instance_id` as `index`; do not renumber it or join solely on row position.

## Reusable filtered schemas

The bundled `filtered_meta.json` files preserve the outputs of the DIN-SQL schema-filtering extension, using question, evidence, and `rc_round3` as guidance with `qwen3.8-2.4t-a95b`. They contain the selected metadata, not model responses, SQL predictions, accuracy measurements, or token logs. Original column attributes and surviving foreign-key references are retained.

There are 5,318 successful filters across 5,320 question records. BIRD-Interact Full questions `robot_fault_prediction_1` and `robot_fault_prediction_8` exhausted their original retry budgets: their records retain `success: false`, `reason: "retry_budget_exhausted"`, and `result: null`. A failed filter is not an empty successful schema and must not be silently substituted for one.

These are reusable intermediate artifacts, separate from the gold-SQL-based reference labels in `schema_linking_annotations.jsonl`. `meta/` remains the full schema. The default method configurations continue to use full schemas; merely having `filtered_meta.json` in a data directory does not switch a run to filtered inputs. Filter-aware consumers can read each successful record's `result` as a metadata array. To generate new filters, use `scripts/filter_meta.py`; it writes to ignored `outputs/schema_filter/` without overwriting these bundled snapshots.

`uv run python -m scripts.check_setup` checks the bundled filter identities, success/failure counts, file hashes, and associated question/RS/metadata hashes offline. Preserve the manifest when sharing the saved filters.

## Regenerating inputs

The bundled inputs can be used directly; preprocessing does not need to be repeated merely to run an experiment. Raw benchmark resources are still required for database execution and some retrieval preparation. The preprocessors accept explicit resource roots:

```bash
uv run python scripts/preprocess.py --dataset bird --split dev --dataset-root ../BIRD
uv run python scripts/preprocess.py --dataset spider --split dev --dataset-root ../Spider
uv run python scripts/preprocess.py --dataset spider --split test --dataset-root ../Spider
uv run python scripts/preprocess.py --dataset birdinteract --split lite \
  --dataset-root ../BIRD-Interact/BIRD-Interact-ADK
uv run python scripts/preprocess.py --dataset birdinteract --split full \
  --dataset-root ../BIRD-Interact/BIRD-Interact-ADK \
  --livesqlbench-root ../livesqlbench-base-full-v1
```

These commands write to `data/<group>/`. Use `--output-dir` for a separate scratch destination when inspecting a different benchmark version. Relative input paths are resolved from the current working directory. Without overrides, the preprocessing scripts look for benchmark directories in that working directory; the examples above explicitly use sibling directories.

BIRD-Interact preparation selects `Query` instances only, without expanding follow-up interactions. Full questions are joined back to LiveSQLBench by `instance_id`. Public metadata defines the schema exposed to the method; PostgreSQL execution still requires the actual databases.

## Resource roots

The example method configurations use sibling `../BIRD/` and `../Spider/` directories. DeepEye workload JSON paths are relative to the workload file; DAIL/DIN configuration paths use the repository-root conventions described in their guides. Update the configuration rather than moving or duplicating a large benchmark installation just to match an example.

Keep benchmark distribution and model-resource terms when redistributing inputs. Database dumps and pretrained weights are deliberately not copied into this repository.
