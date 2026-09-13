"""Deterministic, gold-independent prompt selection from a read-only historical run."""
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys
from .core import fingerprint


def prepare_workload(source, per_cell=5):
    c = sqlite3.connect(Path(source).resolve().as_uri()+"?mode=ro", uri=True)
    stages = ("schema_linking", "sql_generation", "sql_revision", "sql_selection")
    groups, seen = defaultdict(list), set()
    for event, item, stage, raw in c.execute("SELECT e.event_id,a.item_key,a.stage,e.payload_json FROM events e JOIN attempts a USING(attempt_id) WHERE e.kind='api_request' ORDER BY e.event_id"):
        if stage not in stages:
            continue
        p = json.loads(raw)
        messages = p["kwargs"]["messages"]
        branch = p.get("branch_path", [])
        key = (item, stage, tuple(branch))
        if key in seen:
            continue
        seen.add(key)
        # Selection is based only on inputs; no success, latency or correctness filter.
        groups[stage].append({"source_event_id": event, "item_key": item, "stage": stage,
            "branch_path": branch, "messages": messages, "prompt_sha256": fingerprint(messages),
            "input_characters": sum(len(m.get("content") or "") for m in messages),
            "component_call_id": p.get("component_call_id"), "source_run": str(Path(source).resolve())})
    selected, bins = [], []
    for stage in stages:
        data = sorted(groups[stage], key=lambda x: (x["input_characters"], x["source_event_id"]))
        if len(data) < 3:
            raise ValueError(f"not enough source prompts for {stage}")
        for k in range(3):
            cell = data[len(data)*k//3:len(data)*(k+1)//3]
            bins.append({"stage": stage, "length_bin": k, "available": len(cell),
                         "min_characters": cell[0]["input_characters"], "max_characters": cell[-1]["input_characters"]})
            for j in range(per_cell):
                row = dict(cell[min(len(cell)-1, int((j+.5)*len(cell)/per_cell))])
                row["stratum"] = stage+f"/length_{k}"
                row["sample_order"] = j
                selected.append(row)
    ids = {x["component_call_id"] for x in selected}
    inputs = {}
    # One streaming scan, not a large JSON join/copy of all historical responses.
    for raw, in c.execute("SELECT payload_json FROM events WHERE kind='component_start'"):
        p = json.loads(raw)
        if p.get("component_call_id") in ids:
            inputs[p["component_call_id"]] = p.get("inputs", {})
    c.close()
    for row in selected:
        branch = ".".join(row["branch_path"])
        row["parser"] = ("direct" if row["stage"] == "schema_linking" and ".direct" in branch
                         else "reversed" if row["stage"] == "schema_linking" else
                         "selection" if row["stage"] == "sql_selection" else "sql")
        row["parser_kwargs"] = inputs.get(row["component_call_id"], {}).get("parser_kwargs") or {}
        row["fix_end_token"] = inputs.get(row["component_call_id"], {}).get("fix_end_token", False)
        if row["parser"] in {"direct", "reversed"} and not row["parser_kwargs"].get("database_schema"):
            raise ValueError(f"missing historical schema parser context at event {row['source_event_id']}")
    selected.sort(key=lambda x: (x["sample_order"], x["stratum"]))
    return {"selection": "first input per item/stage/branch; stage-length tertiles; evenly spaced input-only samples",
            "bins": bins, "workload": selected, "sha256": fingerprint(selected),
            "unique_prompts": len({x["prompt_sha256"] for x in selected})}


def parse_response(item, content):
    if item.get("fix_end_token") and content and not content.endswith("</result>"):
        content += "</result>"
    if not content:
        return None
    if item["parser"] == "sql":
        # Exact acceptance rule of BaseSQLGenerator/BaseChecker: result tag, nonempty,
        # optional ```sql fence. No execution or correctness test.
        m = re.search(r"<result>(.*?)</result>", content, re.DOTALL)
        if not m:
            return None
        result = m.group(1).strip()
        if result.startswith("```sql") and result.endswith("```"):
            result = result[6:-3].strip()
        return result or None
    if item["parser"] == "selection":
        m = re.search(r"<result>(.*?)</result>", content, re.DOTALL)
        result = m.group(1).strip().upper() if m else None
        return result if result in {"A", "B", "TIE"} else None
    # Real native schema parsing also checks column/table names; no DB connection.
    baseline = Path(__file__).resolve().parents[2]/"baselines/DeepEye-SQL"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    schema = item["parser_kwargs"]["database_schema"]
    if item["parser"] == "direct":
        from app.pipeline.schema_linking.linkers.direct_linker import DirectLinker
        return DirectLinker()._parse_llm_response(content, schema)
    if item["parser"] == "reversed":
        from app.pipeline.schema_linking.linkers.reversed_linker import ReversedLinker
        # Avoid __init__ log noise; these parsing methods need no mutable instance state.
        linker = object.__new__(ReversedLinker)
        sql = linker._parse_llm_response(content)
        return linker._extract_tables_and_columns(sql, schema) if sql else None
    raise ValueError("unknown parser")
