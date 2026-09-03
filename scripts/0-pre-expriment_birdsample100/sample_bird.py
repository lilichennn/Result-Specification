"""从 BIRD 1.0 train 抽一个跨难度、跨库的样本,执行 gold SQL,写进 sample_bird.json。"""

import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from result_contract.harness.bird import classify_sql_features

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "BIRD1.0/data/dev/dev_20240627/dev.json"
DB_ROOT = ROOT / "BIRD1.0/data/dev/dev_20240627/dev_databases"
OUT = Path(__file__).resolve().parent / "sample_bird_dev.json"
TIMEOUT = 60

# 难度按 SQL 复杂度分数切,分位点见 train 全量分布(38% / 47% / 15%)
TIERS = {"easy": lambda s: s <= 2, "medium": lambda s: 3 <= s <= 4, "hard": lambda s: s >= 5}


def load(limit=None):
    """返回 instance 列表,每条含 db_id / question / evidence / gold_sql / db_path / tier。"""
    raw = json.loads(TRAIN_JSON.read_text())
    if limit:
        raw = raw[:limit]
    out = []
    for i, item in enumerate(raw):
        features, score = classify_sql_features(item["SQL"])
        tier = next(name for name, test in TIERS.items() if test(score))
        out.append(
            {
                "index": i,
                "db_id": item["db_id"],
                "question": item["question"],
                "evidence": item["evidence"],
                "gold_sql": item["SQL"],
                "db_path": str(DB_ROOT / item["db_id"] / f"{item['db_id']}.sqlite"),
                "sql_features": list(features),
                "complexity": score,
                "tier": tier,
            }
        )
    return out


def sample(n=100, seed=42):
    """每个难度层取 n/3,层内按库轮转,尽量铺满不同的库。"""
    rng = random.Random(seed)
    data = load()
    picked = []

    for k, tier in enumerate(TIERS):
        quota = n // len(TIERS) + (1 if k < n % len(TIERS) else 0)

        by_db = defaultdict(list)
        for inst in data:
            if inst["tier"] == tier:
                by_db[inst["db_id"]].append(inst)
        for pool in by_db.values():
            rng.shuffle(pool)

        dbs = sorted(by_db)
        rng.shuffle(dbs)
        # 轮转:先每个库拿一条,不够再拿第二条,以此类推
        while quota > 0:
            for db_id in dbs:
                if not by_db[db_id]:
                    continue
                picked.append(by_db[db_id].pop())
                quota -= 1
                if quota == 0:
                    break

    picked.sort(key=lambda x: x["index"])
    return picked


def run_sql(db_path, sql, timeout=TIMEOUT):
    """只读执行,返回 {columns, rows, row_count} 或 {error}。"""
    deadline = time.monotonic() + timeout
    try:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            cur = conn.execute(sql)
            columns = [c[0] for c in cur.description]
            rows = [[v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in r] for r in cur.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def build(n=100, seed=42):
    """抽样 -> 执行 gold SQL -> gold SQL 跑不通的随机换一条补上,直到 n 条全部可执行。"""
    rng = random.Random(seed)
    pool = load()
    picked = sample(n, seed)
    used = {x["index"] for x in picked}
    done = []

    for i, inst in enumerate(picked, 1):
        while True:
            inst["gold_table"] = run_sql(inst["db_path"], inst["gold_sql"])
            if "error" not in inst["gold_table"]:
                print(f"[{i:3d}/{n}] {inst['db_id']:<28} {inst['gold_table']['row_count']} rows")
                done.append(inst)
                break
            print(f"[{i:3d}/{n}] {inst['db_id']:<28} 剔除:{inst['gold_table']['error']}")
            # 随机补一条没用过的(补位不分难度)
            while True:
                cand = pool[rng.randrange(len(pool))]
                if cand["index"] not in used:
                    used.add(cand["index"])
                    inst = dict(cand)
                    break

    done.sort(key=lambda x: x["index"])
    return done


if __name__ == "__main__":
    data = build(100)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    empty = [x for x in data if x["gold_table"]["row_count"] == 0]
    print(f"\n写入 {OUT}")
    print(f"{len(data)} 条,覆盖 {len(set(x['db_id'] for x in data))} 个库,空结果 {len(empty)} 条")
    print("难度分布:", dict(Counter(x["tier"] for x in data)))
