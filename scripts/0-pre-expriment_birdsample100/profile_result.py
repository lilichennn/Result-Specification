"""从 sample_bird.json 的 gold_table 抽 result_profile,写进 sample_profile.json。

result_profile 是 refRC 的确定性部分:不调模型,纯粹描述"执行结果长什么样"。
"""

import json
from pathlib import Path

from result_contract.rc.schema import declared_type_map, introspect_schema
from result_contract.rc.sql_analysis import extract_top_level_sql_hints, result_profile

HERE = Path(__file__).resolve().parent
IN = HERE / "sample_bird.json"
OUT = HERE / "profile.json"

# 对齐 cvpr_workspace/configs/stage005_reference_v2.json
MAX_EXAMPLES = 3


def declared_types_for(db_path, cache={}):
    """每个库的 列名 -> 声明类型 映射,按库缓存(100 条样本只有 60 个库)。"""
    if db_path not in cache:
        cache[db_path] = declared_type_map(introspect_schema(Path(db_path)))
    return cache[db_path]


if __name__ == "__main__":
    data = json.loads(IN.read_text())

    for i, inst in enumerate(data, 1):
        table = inst["gold_table"]
        declared = declared_types_for(inst["db_path"])
        inst["result_profile"] = result_profile(
            columns=table["columns"],
            rows=table["rows"],
            declared_types=declared,
            max_examples=MAX_EXAMPLES,
        )

        types = [c["type"] for c in inst["result_profile"]["columns"]]
        print(f"[{i:3d}/{len(data)}] {inst['db_id']:<28} {table['row_count']:>5} rows  {types}")

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\n写入 {OUT}")
