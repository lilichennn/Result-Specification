# External dependencies and resources

## Python and upstream source

Install Python 3.12 dependencies with `uv sync --locked`. The root lockfile is the environment for the adapters; it is not necessary to install each upstream repository's legacy requirements into that same environment.

| Dependency | How it is provided |
| --- | --- |
| [DeepEye-SQL](https://github.com/HKUSTDial/DeepEye-SQL) | Vendored source under `baselines/DeepEye-SQL/`, including the local sampling/runtime integration. |
| [DAIL-SQL](https://github.com/BeachWang/DAIL-SQL) | Git submodule under `baselines/DAIL-SQL/`. |
| [DIN-SQL](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting) | Git submodule under `baselines/DIN-SQL/`; the adapters also retain four local stage source files used to extract templates. |

Run `git submodule update --init --recursive` after cloning without submodules. Git pins the two submodules to specific revisions. Their upstream files and licenses are independent of this repository's own output policy.

## Benchmark databases

Obtain benchmark databases and training examples from their original distributions. Prepared questions and specifications alone cannot execute SQL.

The supplied configurations expect the following sibling resource directories; paths can be changed in `config/`:

```text
parent/
  Result-Contract/                 This repository
  BIRD/data/
    dev/dev.json
    dev/dev_tables.json
    dev/dev_databases/
    train/train.json
    train/train_tables.json
    train/train_databases/
  Spider/data/
    dev.json
    test.json
    tables.json
    test_tables.json
    train_spider.json
    train_others.json
    database/
    test_database/
```

Use the actual files from your dataset release. In particular, Spider test may use separate database and schema files rather than the dev resources. BIRD-Interact uses externally hosted PostgreSQL databases, not SQLite copies of the public schema files.

[BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) preprocessing consumes the ADK metadata. Full question/reference preparation additionally uses [LiveSQLBench Base Full](https://huggingface.co/datasets/birdsql/livesqlbench-base-full-v1). Install the databases separately and configure a read-only evaluation role in `config/.env`. Public metadata controls what the method sees; the database account controls what the executor can access. These are separate boundaries.

## Model and embedding services

`config/.env.example` has two independent chat configuration groups:

- RS generation: aliases such as `qwen38=API1` and `API1_*`.
- Method experiments: `DASH_MODELS`, `DASH_API_KEY`, and `DASH_BASE_URL`.

Use a single model name for `DASH_MODELS`. Chat base URLs should be the provider's OpenAI-compatible base, without `/chat/completions`. The retained runners send thinking/sampling fields defined in their configuration; use a provider supporting those fields rather than assuming every endpoint accepts them.

DeepEye preparation additionally requires `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, and `EMBEDDING_BASE_URL`. Use the endpoint format supported by that embedding service; a native DashScope embedding endpoint and an OpenAI-compatible embedding base are not interchangeable strings. Cache identities include the embedding settings. Do not reuse a cache produced by a different model or dimension without verification.

Some DeepEye environment checks require nonempty PostgreSQL fields even for a SQLite workload. Fill the required configuration fields; actual database access depends on the selected workload. `scripts.check_setup` deliberately does not read `.env` or probe these services.

## DAIL retrieval resources

DAIL uses local resources rather than the DeepEye embedding API for its sentence retrieval model. Fill `config/dail_sql/resources.example.json` with your installation paths:

| Resource | Required content |
| --- | --- |
| Java JDK | `bin/java` and `bin/javac`; Java 11 is suitable for this setup. |
| Stanford CoreNLP 3.9.2 | Full CoreNLP directory and the required jars. |
| `sentence-transformers/all-mpnet-base-v2` | A complete local SentenceTransformer model snapshot, not only a weights file. |
| NLTK data | English stopwords at `corpora/stopwords/english`. |

On macOS with Homebrew, Java can be installed with `brew install openjdk@11`. Set `java.home` to the installed JDK home (the directory containing `bin/java`), not to the Homebrew executable itself. Obtain the CoreNLP/model/NLTK resources through their official distributions and keep them under ignored `resources/` or another existing resource directory.

The DAIL preparation command does not automatically download these files; the MPNet loader is local-only. With the documented root working directory, relative paths in the example resource manifest resolve correctly. Absolute paths are also supported. DAIL preparation additionally consumes the prepared dataset snapshots referenced in its experiment configuration.

## Parallel execution

The supplied runners retain high-concurrency experiment settings, including large request limits. These settings are not universal service recommendations. Inspect each method's CLI and configuration, start with a smoke run, and lower request/database worker limits to match your environment. Shared-account quotas and database capacity can be more restrictive than local CPU capacity.

Original prompts, sampling settings, retry policies, and metric definitions are preserved. Concurrency/resource configuration is separate from changing a method's candidate budget or model parameters.
