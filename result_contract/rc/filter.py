"""Select schema using either RC or question/evidence, then crop original metadata.

This module performs no file I/O. The caller supplies the model adapter and
handles dataset iteration and persistence.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .rc_round1 import ModelCall
from .rc_round2 import Round2RC


FILTER_INSTRUCTION = """Filter the database schema by removing tables and columns that are clearly unrelated to the supplied guidance.
The filtered schema will be used by a downstream text-to-SQL method that performs its own schema linking. Preserve plausible candidates and their connecting information for that method to resolve. Do not attempt to produce a minimal set of columns or commit to a single SQL implementation.

Rules:
1. Read all supplied guidance together: entity scope, eligibility conditions, output roles, row grain, computations, selection policies, and any metadata clarifications or unresolved interpretations. Use whichever of these are present; do not assume missing requirements.
2. Remove a column only when its meaning and entity ownership make it clearly unrelated to the query. Lack of an explicit mention is not sufficient reason to remove it. Retain plausible columns for output, conditions, grouping, aggregation, ordering, and entity identification.
3. Do not force an abstract output role into one concrete column. When names, identifiers, codes, or descriptions remain plausible ways to express the requested entity or value, retain those candidates. Narrow them only when the guidance and metadata clearly exclude alternatives.
4. Preserve alternatives explicitly left unresolved by the guidance. Use semantic clarifications to exclude incompatible candidates, but do not treat a mentioned column as an exhaustive list of all relevant columns.
5. Explore ref_key relationships among the candidate entities and tables, including paths through intermediate tables. Retain the foreign-key columns, their referenced columns, and intermediate tables needed for plausible query relationships. If the relationship is unresolved, preserve plausible alternative paths. A foreign-key connection alone does not make every neighboring table or its unrelated attributes relevant.
6. Before returning the schema, check that removing any remaining candidate would not prematurely decide an unresolved output interpretation, condition, computation, or relationship. Also check that retained plausible relationships have both endpoints and their intermediate join columns available.
7. A row count does not require every attribute of the counted table. Preserve plausible identity and relationship columns where they may be needed for counting entities or joining tables. A table may have an empty columns list when only its rows are relevant and no concrete column is a plausible requirement.
8. Return the retained tables and columns using only exact names from the schema. For each column, use original_column_name when that field is present; otherwise use column_name. Preserve plausible candidates without adding unrelated columns merely to keep an entire table. Do not invent or rewrite names. There is no fixed retention quota.

Return only JSON in this format:
{"tables": [{"name": "table_name", "columns": ["column_name"]}]}
"""


def build_filter_messages(
    metadata: Sequence[Mapping[str, Any]],
    *,
    rc: Round2RC | Mapping[str, Any] | None = None,
    question: str | None = None,
    evidence: str = "",
) -> list[dict[str, str]]:
    """Build selection instructions from RC, question/evidence, or both."""
    _catalog(metadata)
    if question is not None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        if not isinstance(evidence, str):
            raise ValueError("evidence must be text")
        guidance: dict[str, Any] = {
            "question": question.strip(),
            "evidence": evidence,
        }
    elif evidence:
        raise ValueError("evidence requires a question")
    else:
        guidance = {}
    if rc is not None:
        value = rc if isinstance(rc, Round2RC) else Round2RC.from_value(rc)
        guidance["result_contract"] = value.to_dict()  # Includes meta_review.
    if not guidance:
        raise ValueError("Provide RC, question/evidence, or both")
    return [
        {"role": "system", "content": FILTER_INSTRUCTION},
        {
            "role": "user",
            "content": "Guidance:\n"
            + json.dumps(guidance, ensure_ascii=False, indent=2)
            + "\n\nDatabase schema:\n"
            + json.dumps(list(metadata), ensure_ascii=False, indent=2),
        },
    ]


def parse_filter_response(raw_response: str) -> dict[str, Any]:
    """Parse a JSON selection; schema-name validation occurs during cropping."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("model response must be non-empty text")
    content = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1)
    selection = json.loads(content)
    if not isinstance(selection, dict) or set(selection) != {"tables"}:
        raise ValueError("selection must contain only the tables field")
    if not isinstance(selection["tables"], list):
        raise ValueError("tables must be a list")
    for table in selection["tables"]:
        if not isinstance(table, dict) or set(table) != {"name", "columns"}:
            raise ValueError("each selected table must contain name and columns")
        if not isinstance(table["name"], str):
            raise ValueError("selected table name must be text")
        if not isinstance(table["columns"], list) or not all(
            isinstance(name, str) for name in table["columns"]
        ):
            raise ValueError("selected columns must be a list of names")
    return selection


def apply_filter(
    metadata: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Preserve original order/attributes; retain only FKs with both ends selected.

    ref_key follows the preprocessor format: 'table.column; table.column'.
    Invalid selections raise errors instead of silently changing the selection.
    """
    catalog = _catalog(metadata)
    selection = parse_filter_response(json.dumps(selection, ensure_ascii=False))
    selected: dict[str, set[str]] = {}
    for table in selection["tables"]:
        name, columns = table["name"], table["columns"]
        if name not in catalog:
            raise ValueError(f"unknown table: {name!r}")
        if name in selected or len(columns) != len(set(columns)):
            raise ValueError(f"duplicate table or columns: {name!r}")
        unknown = set(columns) - catalog[name]
        if unknown:
            raise ValueError(f"unknown columns in {name!r}: {sorted(unknown)}")
        selected[name] = set(columns)

    endpoints = {f"{table}.{column}" for table, columns in selected.items() for column in columns}
    filtered = []
    for table in metadata:
        name = table["table_name"]
        if name not in selected:
            continue
        kept_table = deepcopy(dict(table))
        kept_table["columns"] = []
        for column in table["columns"]:
            if column.get("original_column_name", column.get("column_name")) not in selected[name]:
                continue
            kept_column = deepcopy(dict(column))
            if kept_column.get("ref_key"):
                refs = kept_column["ref_key"].split(";")
                if not all(ref.strip() in endpoints for ref in refs):
                    kept_column["ref_key"] = "; ".join(
                        ref.strip() for ref in refs if ref.strip() in endpoints
                    )
            kept_table["columns"].append(kept_column)
        filtered.append(kept_table)
    return filtered


def filter_schema(
    metadata: Sequence[Mapping[str, Any]],
    model_call: ModelCall,
    *,
    rc: Round2RC | Mapping[str, Any] | None = None,
    question: str | None = None,
    evidence: str = "",
) -> list[dict[str, Any]]:
    """Make one model call and return cropped metadata; propagate failures."""
    messages = build_filter_messages(metadata, rc=rc, question=question, evidence=evidence)
    selection = parse_filter_response(model_call(messages))
    return apply_filter(metadata, selection)


def _catalog(metadata: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    if isinstance(metadata, (str, bytes)) or not isinstance(metadata, Sequence) or not metadata:
        raise ValueError("metadata must be a non-empty sequence of tables")
    catalog: dict[str, set[str]] = {}
    for table in metadata:
        if not isinstance(table, Mapping):
            raise ValueError("metadata tables must be objects")
        name, columns = table.get("table_name"), table.get("columns")
        if not isinstance(name, str) or not name or name in catalog:
            raise ValueError("metadata table names must be non-empty and unique")
        if not isinstance(columns, list):
            raise ValueError(f"columns must be a list in {name!r}")
        names: set[str] = set()
        for column in columns:
            if not isinstance(column, Mapping):
                raise ValueError("metadata columns must be objects")
            column_name = column.get("original_column_name", column.get("column_name"))
            if not isinstance(column_name, str) or not column_name or column_name in names:
                raise ValueError(f"column names must be non-empty and unique in {name!r}")
            if not isinstance(column.get("ref_key", ""), str):
                raise ValueError("ref_key must be text")
            names.add(column_name)
        catalog[name] = names
    return catalog
