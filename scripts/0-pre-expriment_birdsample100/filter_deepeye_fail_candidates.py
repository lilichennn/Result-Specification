import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "sample_bird_dev_with_candidates_DeepEye.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sample_bird_dev_fail_candidates_DeepEye"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute DeepEye candidates and retain candidates whose result does not "
            "contain the gold result under column projection."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def resolve_database_path(sample: dict[str, Any]) -> Path:
    configured_path = Path(sample["db_path"])
    if configured_path.is_file():
        return configured_path

    db_id = sample["db_id"]
    original_dev_path = (
        PROJECT_ROOT
        / "BIRD1.0"
        / "data"
        / "dev_original"
        / "dev_20240627"
        / "dev_databases"
        / db_id
        / f"{db_id}.sqlite"
    )
    if original_dev_path.is_file():
        return original_dev_path

    raise FileNotFoundError(
        f"Database not found for {db_id}: {configured_path} or {original_dev_path}"
    )


def execute_sql(db_path: Path, sql: str, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + timeout
    timed_out = False

    def progress_handler() -> int:
        nonlocal timed_out
        if time.monotonic() >= deadline:
            timed_out = True
            return 1
        return 0

    try:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.set_progress_handler(progress_handler, 10_000)
            cursor = connection.execute(sql)
            if cursor.description is None:
                raise sqlite3.DatabaseError("SQL did not produce a result table")
            columns = [str(column[0]) for column in cursor.description]
            rows = [
                [json_safe_value(value) for value in row]
                for row in cursor.fetchall()
            ]

        return {
            "status": "succeeded",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except Exception as error:
        return {
            "status": "timeout" if timed_out else "error",
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }


def row_set(rows: list[list[Any]]) -> set[tuple[Any, ...]]:
    return {tuple(row) for row in rows}


def candidate_contains_gold(
    candidate_table: dict[str, Any],
    gold_rows: list[list[Any]],
    gold_column_count: int,
) -> bool:
    candidate_rows = candidate_table["rows"]
    candidate_column_count = len(candidate_table["columns"])
    if candidate_column_count < gold_column_count:
        return False
    if not gold_rows:
        return True

    gold_row_set = row_set(gold_rows)
    for candidate_column_indices in permutations(
        range(candidate_column_count), gold_column_count
    ):
        projected_candidate_rows = {
            tuple(row[index] for index in candidate_column_indices)
            for row in candidate_rows
        }
        if gold_row_set <= projected_candidate_rows:
            return True
    return False


def write_table(file, name: str, table: dict[str, Any], indent: int, comma: bool) -> None:
    prefix = "  " * indent
    inner_prefix = "  " * (indent + 1)
    row_prefix = "  " * (indent + 2)
    file.write(f'{prefix}"{name}": {{\n')
    if "status" in table:
        file.write(
            f'{inner_prefix}"status": '
            f'{json.dumps(table["status"], ensure_ascii=False)},\n'
        )
    file.write(
        f'{inner_prefix}"columns": '
        f'{json.dumps(table["columns"], ensure_ascii=False)},\n'
    )
    file.write(f'{inner_prefix}"rows": [\n')
    for row_index, row in enumerate(table["rows"]):
        row_comma = "," if row_index < len(table["rows"]) - 1 else ""
        file.write(f"{row_prefix}{json.dumps(row, ensure_ascii=False)}{row_comma}\n")
    file.write(f"{inner_prefix}],\n")
    row_count_comma = "," if "elapsed_ms" in table else ""
    file.write(f'{inner_prefix}"row_count": {table["row_count"]}{row_count_comma}\n')
    if "elapsed_ms" in table:
        file.write(f'{inner_prefix}"elapsed_ms": {table["elapsed_ms"]}\n')
    closing_comma = "," if comma else ""
    file.write(f"{prefix}}}{closing_comma}\n")


def write_sample_json(output_path: Path, sample: dict[str, Any]) -> None:
    scalar_keys = [
        "index",
        "db_id",
        "db_path",
        "complexity",
        "tier",
        "question",
        "evidence",
        "gold_sql",
    ]
    with output_path.open("w", encoding="utf-8") as file:
        file.write("{\n")
        for key in scalar_keys:
            file.write(
                f'  "{key}": {json.dumps(sample[key], ensure_ascii=False)},\n'
            )

        file.write('  "fail_candidates": [\n')
        for failure_index, failure in enumerate(sample["fail_candidates"]):
            file.write("    {\n")
            file.write(f'      "candidate_num": {failure["candidate_num"]},\n')
            file.write('      "failure_type": "gold_not_contained",\n')
            file.write(
                '      "candidate_sql": '
                f'{json.dumps(failure["candidate_sql"], ensure_ascii=False)},\n'
            )
            write_table(
                file,
                "candidate_table",
                failure["candidate_table"],
                indent=3,
                comma=False,
            )
            failure_comma = (
                "," if failure_index < len(sample["fail_candidates"]) - 1 else ""
            )
            file.write(f"    }}{failure_comma}\n")
        file.write("  ],\n")

        write_table(file, "gold_table", sample["gold_table"], indent=1, comma=True)
        file.write(
            '  "sql_features": '
            f'{json.dumps(sample["sql_features"], ensure_ascii=False)}\n'
        )
        file.write("}\n")


def evaluate_candidate(
    sample_position: int,
    candidate_position: int,
    db_path: str,
    candidate_sql: str,
    gold_rows: list[list[Any]],
    gold_column_count: int,
    timeout: float,
) -> tuple[int, int, str, dict[str, Any] | None]:
    candidate_table = execute_sql(Path(db_path), candidate_sql, timeout)

    if candidate_table["status"] != "succeeded":
        return sample_position, candidate_position, candidate_table["status"], None

    matches = candidate_contains_gold(candidate_table, gold_rows, gold_column_count)
    if matches:
        return sample_position, candidate_position, "matched", None

    return sample_position, candidate_position, "gold_not_contained", {
        "candidate_sql": candidate_sql,
        "candidate_table": candidate_table,
    }


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    with input_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    tasks = []
    resolved_db_paths = []
    for sample_position, sample in enumerate(samples):
        resolved_db_path = resolve_database_path(sample)
        resolved_db_paths.append(str(resolved_db_path))
        for candidate_position, candidate_sql in enumerate(sample["candidate_sqls"]):
            tasks.append(
                (
                    sample_position,
                    candidate_position,
                    str(resolved_db_path),
                    candidate_sql,
                    sample["gold_table"]["rows"],
                    len(sample["gold_table"]["columns"]),
                    args.timeout,
                )
            )

    mismatches_by_sample: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    samples_with_any_failure = set()
    outcome_counts = {
        "matched": 0,
        "gold_not_contained": 0,
        "error": 0,
        "timeout": 0,
    }
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_candidate, *task) for task in tasks]
        for future in as_completed(futures):
            sample_position, candidate_position, outcome, mismatch = future.result()
            completed += 1
            outcome_counts[outcome] += 1
            if outcome != "matched":
                samples_with_any_failure.add(sample_position)
            if mismatch is not None:
                mismatches_by_sample.setdefault(sample_position, []).append(
                    (candidate_position, mismatch)
                )
            if completed % 25 == 0 or completed == len(tasks):
                print(f"Executed candidates: {completed}/{len(tasks)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("[0-9]*_[0-9]*.json", "index_[0-9]*.json"):
        for old_output_path in output_dir.glob(pattern):
            old_output_path.unlink()

    written_files = 0
    samples_with_mismatches = 0
    for sample_position, sample in enumerate(samples):
        indexed_mismatches = mismatches_by_sample.get(sample_position, [])
        if not indexed_mismatches:
            continue

        samples_with_mismatches += 1
        indexed_mismatches.sort(key=lambda item: item[0])
        fail_candidates = []
        for candidate_position, mismatch in indexed_mismatches:
            candidate_number = candidate_position + 1
            fail_candidates.append(
                {
                    "candidate_num": candidate_number,
                    "candidate_sql": mismatch["candidate_sql"],
                    "candidate_table": mismatch["candidate_table"],
                }
            )

        output_item = {
            "index": sample["index"],
            "db_id": sample["db_id"],
            "db_path": resolved_db_paths[sample_position],
            "complexity": sample["complexity"],
            "tier": sample["tier"],
            "question": sample["question"],
            "evidence": sample["evidence"],
            "gold_sql": sample["gold_sql"],
            "fail_candidates": fail_candidates,
            "gold_table": sample["gold_table"],
            "sql_features": sample["sql_features"],
        }
        output_path = output_dir / f"index_{sample['index']}.json"
        write_sample_json(output_path, output_item)
        written_files += 1

    print(f"Matched candidates: {outcome_counts['matched']}")
    print(f"Gold-not-contained candidates written: {outcome_counts['gold_not_contained']}")
    print(
        "Execution failures excluded: "
        f"{outcome_counts['error']} errors, {outcome_counts['timeout']} timeouts"
    )
    print(f"Samples containing gold-not-contained candidates: {samples_with_mismatches}")
    print(
        "Samples containing a custom mismatch or execution failure: "
        f"{len(samples_with_any_failure)}"
    )
    print(f"JSON files written: {written_files}")
    print(output_dir)


if __name__ == "__main__":
    main()
