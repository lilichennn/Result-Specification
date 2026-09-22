# Generating and using a Result Specification

The reusable implementation is in `result_contract/rc/`; command-line orchestration is in `scripts/generate_rc.py`. Prompts live next to the implementation. Paths and identifiers retain the RC abbreviation as explained in [Terminology](../README.md#terminology).

## Rounds and fields

| Round | Additional information | Output |
| --- | --- | --- |
| 1 | Question and evidence | `population`, `row_grain`, `column_role`, `derivation`, `filter_policy` |
| 2 | Public database metadata | Revised fields plus `meta_review` |
| 3 | Reference SQL, alongside the question, evidence, and Round-2 specification | Corrected six-field specification |

The five RS dimensions are **Population, Row Grain, Column Role, Derivation, and Filter Policy**, stored as `population`, `row_grain`, `column_role`, `derivation`, and `filter_policy`. The additional `meta_review` field records metadata-refinement notes. All six JSON fields contain text. The full generation instructions are in the versioned prompt files.

**Oracle RS** is constructed by checking and correcting RS against gold SQL. This correction step is named **Round 3** in the scripts; the supplied five-group experiment configurations explicitly use `rc_round3`. The correction step uses gold SQL, not solely the user question.

## Commands

Fill an alias in `config/.env` first. For example, `qwen38=API1` selects the model and credentials in `API1`, `API1_API_KEY`, and `API1_BASE_URL`. The client appends `/chat/completions` to that base URL. Method experiments use the separate `DASH_*` keys.

```bash
# Generate Round 1 and Round 2 for the question/metadata files in data/bird_dev/.
uv run python scripts/generate_rc.py --dataset_split bird_dev --llm qwen38

# Correct existing Round 2 records using the supplied reference file.
uv run python scripts/generate_rc.py --dataset_split bird_dev --llm qwen38 \
  --round3 --gold-file data/bird_dev/gold_sql_schema_linking.json
```

The commands update `data/<group>/rc.json`; preserve a copy before deliberately regenerating a released artifact. Logs go to `outputs/rc_generation/<group>/`. `--concurrency` sets the **initial** request concurrency, not a fixed ceiling: this generator increases concurrency dynamically. Review its settings and your provider quota before starting a large generation job.

`scripts/prepare_rc_gold.py` prepares reference records for BIRD-Interact. `scripts/extract_gold_schema_linking.py` extracts reference dependencies for BIRD/Spider. Both expose explicit resource-path options; inspect `--help` when working with a new raw-data layout. `--allow-partial-gold` on the Round-3 command restricts processing to available reference IDs and preserves the other records.

## Schema filtering

`scripts/filter_meta.py` runs the existing filtering procedure using the selected model alias. Its output defaults to `outputs/schema_filter/<group>/`, leaving the public metadata in `data/` unchanged. Filtering calls a model; it is not a local format conversion. Method-specific linking comparisons have their own orchestration in `scripts/rc_evaluation/`.

## Python use

The library exposes `generate_round1`, `generate_round2`, and `generate_round3`. Supply either an explicit `model_call` callback or the keyword-only `llm` alias. With neither, generation fails early rather than choosing a service implicitly. See each function's signature for round-specific inputs. A custom callback is useful for local fixtures without making network requests.
