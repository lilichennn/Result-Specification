# Result Specifications for NL-to-SQL

**Separating result semantics from SQL implementation.**

A Result Specification (RS) is a structured set of semantic constraints on the intended result table, organized into five dimensions: **Population, Row Grain, Column Role, Derivation, and Filter Policy**. It specifies what a query should return without prescribing a particular SQL implementation. This repository provides RS generation, benchmark input preparation, and comparisons of the original and RS variants of **DeepEye-SQL, DAIL-SQL, and DIN-SQL**.

[Data](docs/data.md) · [RS generation](docs/rs.md) · [Experiments](docs/experiments.md) · [Evaluation](docs/evaluation.md) · [Analysis tools](docs/analysis.md) · [Dependencies](docs/dependencies.md)

## What is included

| Area | Contents |
| --- | --- |
| RS generation | Question-based generation, metadata review, and reference-SQL correction; prompts and reusable Python functions. |
| Method integration | Adapters, RS injection, concurrent execution, resumable records, and targeted reruns. |
| Reusable inputs | Questions, public metadata, generated RS, reference SQL where available, and schema-linking reference annotations. |
| Analysis | SQL execution comparison, stage-specific summary tables, and compact exports. |

The five evaluation groups are BIRD dev, Spider dev, Spider test, BIRD-Interact Lite, and BIRD-Interact Full. Database files, downloaded models, vector caches, and experiment outputs are not bundled.

## Terminology

The concept is **Result Specification (RS)**. The implementation retains the earlier **Result Contract (RC)** naming in paths, functions, command-line options, data fields, and saved records for compatibility. For example, `rc.json` stores RS records, and `rc_round3` stores the specification after gold-SQL correction. These names refer to the same concept. Prompt templates retain their original wording for reproducibility. Use the identifiers in commands and configuration examples exactly as written.

## Quick start

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run commands from the repository root.

```bash
git clone --recurse-submodules https://github.com/lilichennn/Result-Contract.git
cd Result-Contract
uv sync --locked
uv run python -m scripts.check_setup
```

For an existing clone, initialize dependencies with `git submodule update --init --recursive`. The setup check is offline: it validates bundled inputs and source dependencies without reading credentials, calling a model, or connecting to a database.

To run experiments, copy and fill the local configuration:

```bash
cp config/.env.example config/.env
```

Then follow the [dataset/resource setup](docs/dependencies.md) and [method-specific commands](docs/experiments.md). Preparation may itself call embedding or language-model services; it is not an offline setup check. Start with a small scope and inspect your provider limits before launching a full group.

## Repository map

```text
config/                 Environment template and method/workload settings
data/                   Questions, metadata, RS, and reference annotations
result_contract/        Reusable RS generation and preprocessing code
scripts/
  preprocess.py         Benchmark input preparation
  generate_rc.py        RS generation and reference-SQL correction
  filter_meta.py        Schema filtering
  baseline_adapters/    Method and database integration
  rc_evaluation/        Experiment runners, records, and exports
  analysis/             Evaluation summaries and tables
baselines/              Upstream method source and pinned submodules
docs/                   Setup, execution, and analysis guides
pyproject.toml          Python dependencies
uv.lock                 Locked dependency versions
```

Runtime directories are created as needed and ignored by Git: `resources/` for downloaded tools/models, `cache/` for reusable preparation, and `outputs/` for runs and analyses.

## Oracle RS and reproducibility

Oracle RS is constructed by checking and correcting the specification against gold SQL. The scripts name this correction step **Round 3** and store its output in `rc_round3`. The supplied experiment configurations select this field explicitly and do not silently fall back to Round 2.

Configuration, prompt snapshots, question identities, individual model attempts, and stage outputs are retained in experiment records. Reused computations remain traceable. See [evaluation and record handling](docs/evaluation.md) before combining reruns or calculating accuracy and token statistics.

## Attribution

This repository builds on the upstream methods and benchmarks linked in [Dependencies](docs/dependencies.md). Their licenses and usage conditions remain applicable. Third-party license files are retained with their source; benchmark and model resources must be obtained under their original terms.
