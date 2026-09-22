# Running the method experiments

Run all commands from the repository root. First complete [resource setup](dependencies.md), fill `config/.env`, and run `uv run python -m scripts.check_setup`. The latter is offline; the experiment commands below can incur model charges and execute SQL.

Each runner freezes its inputs and settings in run records. Use a new run/batch directory when changing model, inputs, prompts, or sampling settings. Resume a compatible existing batch instead of editing its saved manifest.

## DeepEye-SQL

### Prepare reusable inputs

Each workload JSON binds the question set, public metadata, RS version, raw dataset root, and prepared snapshot. Prepare one group before running it:

```bash
uv run python -m scripts.deepeye_run prepare-native \
  --workload config/deepeye/spider_dev_rc3.json \
  --output cache/deepeye_shared_preparation/prepared/spider/dev.snapshot \
  --env-file config/.env
```

Preparation includes database sample-value embeddings, question keywords, preliminary SQL, and native dynamic example retrieval. Some steps call language/embedding models. Completed cache entries are reusable. Use the analogous workload and its configured snapshot path for each other group.

### Run a native/RS campaign

```bash
uv run python -m scripts.deepeye_campaign configure \
  --campaign-dir outputs/deepeye/campaigns/spider_dev \
  --workload config/deepeye/spider_dev_rc3.json \
  --env-file config/.env --rc-version 3 --tail-fraction 0.8

uv run python -m scripts.deepeye_campaign run \
  --campaign-dir outputs/deepeye/campaigns/spider_dev

uv run python -m scripts.deepeye_campaign status \
  --campaign-dir outputs/deepeye/campaigns/spider_dev
```

The campaign schedules native execution and separate RS-injected stage jobs. It allows later work to start after the configured completion fraction rather than waiting for every slow question. Stage comparisons reuse native inputs before the selected stage; they are not automatically a chained pipeline with RS at all four stages. Stages with no real model request record nonparticipation.

For a small scope, `configure --item` is repeatable and accepts the workload's canonical task keys. For finer budget/resource control, use `scripts.deepeye_run prepare/run/resume` and `scripts.rc_evaluation.deepeye prepare/run`; their `--help` lists individual stage budgets, request limits, and the optional `--continue-downstream` flag. Campaign configuration does not expose every lower-level option.

Unlike `prepare-native`, the lower-level `scripts.rc_evaluation.deepeye prepare` validates existing native records and creates a stage experiment without model or database calls. Each selected question must already have a successfully completed source target stage and its preceding stages. A stage experiment binds one target stage, condition (`none` or `rc`), and repeat; its model, sampling budgets, database execution settings, and input sources must match the frozen native run. RS runs use existing specification records rather than generating missing ones; the supplied workloads select Round 3.

`pause` and `resume` operate on the same campaign directory. Preserve its child run directories: the campaign ledger refers to them and analysis uses those records.

## DAIL-SQL

The adapter runs a two-round pipeline with four logical modes: native, RS in the first round, RS in the second round, and RS in both rounds. When the ordered example-ID list is identical, eligible second-round executions are shared and referenced by multiple modes.

The supplied settings use 9 examples, 5 independent `n=1` samples per round, temperature 0.6, and a 910-second request timeout. Each sample has at most 5 attempts. There is no explicit output-token cap by default. These scientific settings are validated by the adapter; resource limits are separate.

### Prepare retrieval resources

First prepare the DeepEye shared snapshots referenced by `config/dail_sql/experiment.json`; these provide common task/database inputs. Install DAIL's Java/CoreNLP/MPNet/NLTK resources and edit the resource manifest.

```bash
uv run python -m scripts.rc_evaluation.dail_sql.cli prepare \
  --config config/dail_sql/experiment.json \
  --resources config/dail_sql/resources.example.json \
  --output cache/dail_sql --encoder-device cpu
```

Use the **exact immutable directory printed by preparation** for `--prepared` below. Do not pass only its parent cache directory. Preparation can be restricted with `--groups spider_dev` while bringing up the first group.

```bash
uv run python -m scripts.rc_evaluation.dail_sql.cli smoke \
  --config config/dail_sql/experiment.json \
  --prepared cache/dail_sql/REPLACE_WITH_RETURNED_DIRECTORY \
  --env-file config/.env --groups spider_dev --per-group 1

uv run python -m scripts.rc_evaluation.dail_sql.cli run \
  --config config/dail_sql/experiment.json \
  --prepared cache/dail_sql/REPLACE_WITH_RETURNED_DIRECTORY \
  --env-file config/.env --batch-id my-run

uv run python -m scripts.rc_evaluation.dail_sql.cli status \
  --batch outputs/dail_sql/batches/my-run
```

The supplied group order is Spider dev, BIRD dev, BIRD-Interact Full, BIRD-Interact Lite, Spider test. `--request-limit` and `--sql-workers` override execution resources. `resume` continues the batch; `rerun --targets PATH` reruns selected whole questions using new record versions, without deleting unrelated questions. Inspect the command help and existing task identity format before constructing targets.

## DIN-SQL

DIN preparation binds questions, metadata, RS, reference SQL, and difficulty labels derived from the reference SQL. The default does not depend on historical experiment outputs. Files under `scripts/baseline_adapters/din_sql/stages/` are frozen template sources read by the adapter, not standalone execution entrypoints. SQLite-only preparation can be scoped before configuring PostgreSQL:

```bash
uv run python -m scripts.rc_evaluation.din_sql.cli prepare \
  --config config/din_sql/experiment.json --batch-id my-run \
  --groups spider_dev bird_dev spider_test

uv run python -m scripts.rc_evaluation.din_sql.cli run \
  --batch outputs/din_sql/batches/my-run --env-file config/.env

uv run python -m scripts.rc_evaluation.din_sql.cli status \
  --batch outputs/din_sql/batches/my-run
```

Omit `--groups` to prepare all five groups once PostgreSQL is available. Preparation itself may read databases, although it does not call a language model. Settings under `config/din_sql/experiment.json` control the retained runtime. `rerun --group NAME --ids ID ...` creates targeted replacement versions; `resume` preserves the current scope unless explicitly told to process all pending questions.

### Schema filtering followed by native Linking

DIN's additional linking comparison consumes a completed source batch and applies RS schema filtering before native Linking:

```bash
uv run python -m scripts.rc_evaluation.din_sql_linking.cli prepare \
  --source-batch outputs/din_sql/batches/my-run --batch-id my-linking
```

Use the returned batch path with that entrypoint's `run`, `status`, and `verify` commands. This comparison makes model requests; it is not just a comparison of two saved arrays. DAIL's analogous comparison reuses the filter handoff and performs DAIL's local Linking. Its explicit inputs are listed by:

```bash
uv run python -m scripts.rc_evaluation.dail_sql.linking_handoff --help
```

It requires the source DAIL compact export, source DAIL batch, completed DIN filter handoff, local retrieval resources, a work directory, and a fresh export destination. Neither main DAIL Generation nor main DIN Generation/Revision implicitly produces this extra linking comparison.

## Inspecting and exporting

Use method-specific `status`/`inspect` commands while running. After completion, follow [Evaluation](evaluation.md) for SQL comparison and compact exports. Experiment databases and large responses stay under ignored `outputs/`; reusable RS inputs remain under `data/`.
