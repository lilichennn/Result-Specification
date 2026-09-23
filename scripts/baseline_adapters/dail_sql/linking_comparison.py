"""Pure schema projection and physical-name views of native DAIL linking.

Cropping never rebuilds labels from filter metadata: the original schema is
authoritative for tokenization, types and SQLite identifiers. Linking itself
continues to run through :func:`native.link_question`.
"""

from copy import deepcopy
import re


def _index(value, size, label, *, minimum=0):
    if type(value) is not int or not minimum <= value < size:
        raise ValueError(f"Invalid {label} index: {value!r}")
    return value


def _schema_names(schema):
    if not isinstance(schema, dict):
        raise ValueError("Native schema must be an object")
    tables = schema.get("table_names_original")
    columns = schema.get("column_names_original")
    if not isinstance(tables, list) or not isinstance(columns, list) or not columns:
        raise ValueError("Native schema requires original table and column arrays")
    if any(not isinstance(name, str) or not name.strip() for name in tables):
        raise ValueError("Native table names must be nonempty strings")
    if columns[0] != [-1, "*"]:
        raise ValueError("Native schema must retain wildcard column 0")
    for column_id, column in enumerate(columns):
        if (not isinstance(column, list) or len(column) != 2
                or not isinstance(column[1], str) or not column[1].strip()):
            raise ValueError(f"Invalid native column: {column_id}")
        if column_id:
            _index(column[0], len(tables), "column table")
    return tables, columns


def _resolve_name(name, candidates, label):
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Invalid selected {label} name: {name!r}")
    exact = [index for index, original in candidates if original == name]
    matches = exact or [index for index, original in candidates if original.strip() == name.strip()]
    if len(matches) != 1:
        reason = "Ambiguous" if matches else "Unknown"
        raise ValueError(f"{reason} selected {label}: {name!r}")
    return matches[0]


def crop_schema(full_schema: dict, filtered_metadata: list) -> dict:
    """Project selected physical names onto native indices and return mappings.

    Metadata uses ``table_name`` and a ``columns`` list of column objects, with
    ``original_column_name`` (or ``column_name`` as a fallback). Exact names
    win; surrounding-whitespace fallback is allowed only for a unique match.
    Selection ordering never changes native ordering. Empty selections and
    tables without selected columns are valid, and wildcard 0 is implicit.

    All native per-table/per-column arrays retain their original values;
    column table IDs, primary/foreign keys and unresolved-reference source IDs
    are remapped. A resolved FK survives only when both ends are selected.
    Unresolved reference strings remain diagnostics and are never inferred.
    Inputs are not mutated. Unknown, duplicate or invalid selections raise
    ``ValueError``, as do invalid native indices or misaligned detail arrays.
    """
    tables, columns = _schema_names(full_schema)
    if not isinstance(filtered_metadata, list):
        raise ValueError("Filtered metadata must be a list")
    for field in ("table_names", "column_names", "column_types"):
        if not isinstance(full_schema.get(field), list):
            raise ValueError(f"Native schema requires {field}")
    for field, values in full_schema.items():
        if isinstance(values, list) and field.startswith(("table_", "column_")):
            size = len(tables) if field.startswith("table_") else len(columns)
            if len(values) != size:
                raise ValueError(f"Misaligned native schema array: {field}")
    for index, column in enumerate(full_schema["column_names"]):
        if (not isinstance(column, list) or len(column) != 2
                or column[0] != columns[index][0] or not isinstance(column[1], str)):
            raise ValueError(f"Invalid native column label: {index}")
    if full_schema["column_names"][0] != [-1, "*"]:
        raise ValueError("Native schema labels must retain wildcard column 0")

    selected_tables, selected_columns = set(), {0}
    for table in filtered_metadata:
        if not isinstance(table, dict) or not isinstance(table.get("columns"), list):
            raise ValueError("Selected table must contain a columns list")
        table_id = _resolve_name(table.get("table_name"), list(enumerate(tables)), "table")
        if table_id in selected_tables:
            raise ValueError(f"Duplicate selected table: {table['table_name']!r}")
        selected_tables.add(table_id)
        candidates = [(index, name) for index, (parent, name) in enumerate(columns) if parent == table_id]
        for column in table["columns"]:
            if not isinstance(column, dict):
                raise ValueError("Selected column must be an object")
            name = column.get("original_column_name") or column.get("column_name")
            column_id = _resolve_name(name, candidates, "column")
            if column_id in selected_columns:
                raise ValueError(f"Duplicate selected column: {table['table_name']!r}.{name!r}")
            selected_columns.add(column_id)

    table_local_to_full = sorted(selected_tables)
    column_local_to_full = sorted(selected_columns)
    table_full_to_local = {full: local for local, full in enumerate(table_local_to_full)}
    column_full_to_local = {full: local for local, full in enumerate(column_local_to_full)}
    schema = deepcopy(full_schema)
    for field, values in full_schema.items():
        if isinstance(values, list) and field.startswith(("table_", "column_")):
            indices = table_local_to_full if field.startswith("table_") else column_local_to_full
            schema[field] = [deepcopy(values[index]) for index in indices]
    for field in ("column_names_original", "column_names"):
        for column in schema[field][1:]:
            column[0] = table_full_to_local[column[0]]

    if "primary_keys" in schema:
        if not isinstance(schema["primary_keys"], list):
            raise ValueError("Native primary keys must be a list")
        for index in schema["primary_keys"]:
            _index(index, len(columns), "primary key", minimum=1)
        schema["primary_keys"] = [column_full_to_local[index] for index in schema["primary_keys"]
                                  if index in selected_columns]
    if "foreign_keys" in schema:
        if not isinstance(schema["foreign_keys"], list):
            raise ValueError("Native foreign keys must be a list")
        kept = []
        for pair in schema["foreign_keys"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Native foreign key must contain two column indices")
            for index in pair:
                _index(index, len(columns), "foreign key", minimum=1)
            if all(index in selected_columns for index in pair):
                kept.append([column_full_to_local[index] for index in pair])
        schema["foreign_keys"] = kept
    if "unresolved_foreign_keys" in schema:
        if not isinstance(schema["unresolved_foreign_keys"], list):
            raise ValueError("Unresolved foreign keys must be a list")
        kept = []
        for reference in schema["unresolved_foreign_keys"]:
            if not isinstance(reference, dict):
                raise ValueError("Unresolved foreign key must be an object")
            index = _index(reference.get("column_id"), len(columns), "unresolved foreign key", minimum=1)
            if index in selected_columns:
                reference["column_id"] = column_full_to_local[index]
                kept.append(reference)
        schema["unresolved_foreign_keys"] = kept
    return {"schema": schema, "table_local_to_full": table_local_to_full,
            "column_local_to_full": column_local_to_full}


def normalized_linking(linking: dict, schema: dict) -> dict:
    """Union native SC/CV matches into sorted physical table and column names.

    Names are stripped only for the evaluator view; wildcard 0 is excluded.
    A linked column always contributes its parent table. Indices refer to the
    supplied schema, which must therefore be the cropped schema for cropped
    native results. Question bounds are checked when question tokens exist.
    """
    tables, columns = _schema_names(schema)
    tables = [name.strip() for name in tables]
    physical_columns = [(parent, name.strip()) for parent, name in columns]
    if len(tables) != len(set(tables)) or len(physical_columns) != len(set(physical_columns)):
        raise ValueError("Ambiguous physical names after whitespace normalization")
    if not isinstance(linking, dict):
        raise ValueError("Native linking must be an object")
    question = linking.get("question")
    if "question" in linking and not isinstance(question, list):
        raise ValueError("Native question tokens must be a list")

    table_ids, column_ids = set(), set()
    for channel, name, size, selected in (
        ("sc_link", "q_tab_match", len(tables), table_ids),
        ("sc_link", "q_col_match", len(columns), column_ids),
        ("cv_link", "num_date_match", len(columns), column_ids),
        ("cv_link", "cell_match", len(columns), column_ids),
    ):
        matches = linking.get(channel, {})
        if not isinstance(matches, dict) or not isinstance(matches.get(name, {}), dict):
            raise ValueError(f"Native {channel}.{name} must be an object")
        for key in matches.get(name, {}):
            if not isinstance(key, str) or re.fullmatch(r"[0-9]+,[0-9]+", key) is None:
                raise ValueError(f"Invalid native match index: {key!r}")
            question_id, schema_id = map(int, key.split(","))
            if question is not None:
                _index(question_id, len(question), "question")
            selected.add(_index(schema_id, size, name))
    column_ids.discard(0)
    table_ids.update(columns[index][0] for index in column_ids)
    return {"tables": sorted(tables[index] for index in table_ids),
            "columns": sorted([[tables[columns[index][0]], columns[index][1].strip()] for index in column_ids])}
