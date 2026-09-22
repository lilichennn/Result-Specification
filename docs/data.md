# Data layout

Reusable inputs live under `data/<group>/`. Generated experiment records belong under `outputs/`, not alongside inputs.

| Group | Questions | Database engine | Round-3 specifications |
| --- | ---: | --- | ---: |
| `bird_dev` | 1,534 | SQLite | 1,534 |
| `spider_dev` | 1,034 | SQLite | 1,034 |
| `spider_test` | 2,147 | SQLite | 2,147 |
| `bird_interact_lite` | 195 | PostgreSQL | 195 |
| `bird_interact_full` | 410 | PostgreSQL | 410 |
| `spider2_lite` | 280 | SQLite / BigQuery | 121 |

The first five groups contain 5,320 questions. Spider2 Lite is an additional preparation resource; the five-group experiment configurations do not include it. Its availability depends on the corresponding local/cloud resources, and Round-3 specifications are only present for the subset with supplied reference SQL.

## Per-group files

- `<group>.json`: question records with `index`, `db_id`, `question`, and `evidence`.
- `meta/`: database metadata exposed to the methods and RS generation, organized by database.
- `rc.json`: question-aligned records with `rc_round1`, `rc_round2`, `rc_round3`, and per-round status/error fields.
- `gold_sql.json`, `gold_sql_schema_linking.json`, and `gold_sql_preparation.json`: reference SQL/dependency information and preparation provenance, where available. The runners' configuration identifies the actual evaluation reference source.
- `data/reference/schema_linking_annotations.jsonl`: final model-assisted reference labels for the five main groups. It does not include annotation request logs or a run database.

Question identifiers are scoped by group. BIRD-Interact preserves `instance_id` as `index`; do not renumber it or join solely on row position.

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

For Spider2 Lite, inspect `uv run python scripts/preprocess_spider2.py --help`. Cloud credentials and database availability are separate from metadata preparation.

## Resource roots

The example method configurations use sibling `../BIRD/` and `../Spider/` directories. DeepEye workload JSON paths are relative to the workload file; DAIL/DIN configuration paths use the repository-root conventions described in their guides. Update the configuration rather than moving or duplicating a large benchmark installation just to match an example.

Keep benchmark distribution and model-resource terms when redistributing inputs. Database dumps and pretrained weights are deliberately not copied into this repository.
