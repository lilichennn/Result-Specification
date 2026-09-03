import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLES_PATH = SCRIPT_DIR / "sample_bird_dev.json"
DEEPEYE_ITEMS_PATH = (
    SCRIPT_DIR
    / "baseline_repro"
    / "DeepEye-SQL"
    / "sample_bird_dev"
    / "run"
    / "sql_generation.snapshot.data"
    / "items.jsonl"
)
OUTPUT_PATH = SCRIPT_DIR / "sample_bird_dev_with_candidates_DeepEye.json"


def deduplicate_sqls(sqls: list[str]) -> list[str]:
    unique_sqls = []
    seen = set()
    for sql in sqls:
        normalized_sql = " ".join(sql.split()).strip().lower()
        if normalized_sql in seen:
            continue
        seen.add(normalized_sql)
        unique_sqls.append(sql)
    return unique_sqls


def write_formatted_samples(samples: list[dict]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write("[\n")
        for sample_index, sample in enumerate(samples):
            file.write("  {\n")
            file.write(f'    "index": {json.dumps(sample["index"])},\n')
            file.write(
                f'    "db_id": {json.dumps(sample["db_id"], ensure_ascii=False)},\n'
            )
            file.write(
                f'    "db_path": {json.dumps(sample["db_path"], ensure_ascii=False)},\n'
            )
            file.write(f'    "complexity": {json.dumps(sample["complexity"])},\n')
            file.write(
                f'    "tier": {json.dumps(sample["tier"], ensure_ascii=False)},\n'
            )
            file.write(
                f'    "question": {json.dumps(sample["question"], ensure_ascii=False)},\n'
            )
            file.write(
                f'    "evidence": {json.dumps(sample["evidence"], ensure_ascii=False)},\n'
            )
            file.write(
                f'    "gold_sql": {json.dumps(sample["gold_sql"], ensure_ascii=False)},\n'
            )

            file.write('    "candidate_sqls": [\n')
            for candidate_index, sql in enumerate(sample["candidate_sqls"]):
                comma = "," if candidate_index < len(sample["candidate_sqls"]) - 1 else ""
                file.write(f"      {json.dumps(sql, ensure_ascii=False)}{comma}\n")
            file.write("    ],\n")

            gold_table = sample["gold_table"]
            file.write('    "gold_table": {\n')
            file.write(
                f'      "columns": {json.dumps(gold_table["columns"], ensure_ascii=False)},\n'
            )
            file.write('      "rows": [\n')
            for row_index, row in enumerate(gold_table["rows"]):
                comma = "," if row_index < len(gold_table["rows"]) - 1 else ""
                file.write(f"        {json.dumps(row, ensure_ascii=False)}{comma}\n")
            file.write("      ],\n")
            file.write(f'      "row_count": {json.dumps(gold_table["row_count"])}\n')
            file.write("    },\n")
            file.write(
                f'    "sql_features": {json.dumps(sample["sql_features"], ensure_ascii=False)}\n'
            )
            comma = "," if sample_index < len(samples) - 1 else ""
            file.write(f"  }}{comma}\n")
        file.write("]\n")


def main() -> None:
    with SAMPLES_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    candidates_by_question_id = {}
    with DEEPEYE_ITEMS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            question_id = item["input"]["question_id"]
            if question_id in candidates_by_question_id:
                raise ValueError(f"Duplicate DeepEye question_id: {question_id}")

            candidates_by_question_id[question_id] = {
                "database_id": item["input"]["database_id"],
                "candidate_sqls": deduplicate_sqls(
                    item["pipeline_artifacts"]["sql_generation"]["sql_candidates"]
                ),
            }

    merged_samples = []
    for sample in samples:
        question_id = sample["index"]
        if question_id not in candidates_by_question_id:
            raise KeyError(f"No DeepEye candidates for sample index {question_id}")

        deepeye_item = candidates_by_question_id[question_id]
        if sample["db_id"] != deepeye_item["database_id"]:
            raise ValueError(
                f"Database mismatch for sample index {question_id}: "
                f"{sample['db_id']} != {deepeye_item['database_id']}"
            )

        merged_sample = {
            "index": sample["index"],
            "db_id": sample["db_id"],
            "db_path": sample["db_path"],
            "complexity": sample["complexity"],
            "tier": sample["tier"],
            "question": sample["question"],
            "evidence": sample["evidence"],
            "gold_sql": sample["gold_sql"],
            "candidate_sqls": deepeye_item["candidate_sqls"],
            "gold_table": sample["gold_table"],
            "sql_features": sample["sql_features"],
        }
        merged_samples.append(merged_sample)

    write_formatted_samples(merged_samples)

    total_candidates = sum(len(sample["candidate_sqls"]) for sample in merged_samples)
    print(f"Saved {len(merged_samples)} samples and {total_candidates} candidates")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
