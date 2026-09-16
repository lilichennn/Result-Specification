"""DAIL SQL representation / QA layout, using only the bound public schema.

Template source: DAIL 2061f681, prompt/PromptReprTemplate.py:SQLPrompt,
ExampleFormatTemplate.py:QuestionSqlExampleStyle and PromptICLTemplate.py.
Physical sqlite_master DDL is replaced by public metadata DDL at this boundary.
"""
import re

import sqlglot
from sqlglot import exp
import sqlparse

from . import native


PG_INSTRUCTION = "/* Use PostgreSQL syntax for the target query. */"


def _question(row):
    # Matches data_preprocess.py's BIRD question + ' ' + evidence, then strip.
    return (row["question"] + " " + (row.get("evidence") or "")).strip()


def _quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _schema_ddl(schema):
    tables, columns = schema["table_names_original"], schema["column_names_original"]
    types = schema.get("column_types_original", schema["column_types"])
    definitions = []
    for table_id, table in enumerate(tables):
        body = [f"  {_quote(name)} {types[index]}" for index, (owner, name) in enumerate(columns) if owner == table_id]
        keys = [columns[index][1] for index in schema.get("primary_keys", []) if columns[index][0] == table_id]
        if keys:
            body.append("  PRIMARY KEY (" + ", ".join(map(_quote, keys)) + ")")
        for source, target in schema.get("foreign_keys", []):
            if not (0 <= source < len(columns) and 0 <= target < len(columns)):
                raise ValueError("Invalid public foreign key index")
            owner, name = columns[source]
            target_owner, target_name = columns[target]
            if owner == table_id and 0 <= target_owner < len(tables):
                body.append(f"  FOREIGN KEY ({_quote(name)}) REFERENCES {_quote(tables[target_owner])} ({_quote(target_name)})")
        definitions.append(f"CREATE TABLE {_quote(table)} (\n" + ",\n".join(body) + "\n);")
    return "\n\n".join(definitions)


def build_prompt(task: dict, examples: list[dict]) -> list[dict]:
    """One user message, exactly nine ordered QA examples, same in both rounds."""
    if len(examples) != 9:
        raise ValueError("DAIL requires exactly nine ordered examples")
    dialect = task["database"]["dialect"]
    if dialect not in ("sqlite", "postgresql"):
        raise ValueError("Unsupported SQL dialect")
    components = [f"/* Answer the following: {_question(example)} */\n{example['sql']}" for example in examples]
    target = ["/* Given the following database schema: */\n" + _schema_ddl(task["schema"])]
    if dialect == "postgresql":
        target.append(PG_INSTRUCTION)
    target.append(f"/* Answer the following: {_question(task)} */\nSELECT ")
    components.append("\n\n".join(target))
    return [{"role": "user", "content": "/* Some SQL examples are provided based on similar problems: */\n" + "\n\n".join(components)}]


def extract_sql(raw_text: str, dialect: str) -> dict:
    """Extract a single choice; never synthesize a replacement model response.

    Full SELECT/WITH SQL is retained even when invalid, so database errors are
    distinct from extraction errors. Bare SELECT continuations must parse as a
    query before receiving the native prefix. Every changed string is recorded.
    """
    if not isinstance(raw_text, str) or dialect not in ("sqlite", "postgresql"):
        raise ValueError("Expected raw text and a supported SQL dialect")
    result = {"raw_text": raw_text, "candidate_sql": None, "transformations": [], "extraction_error": None}
    sql = raw_text

    def change(name, value):
        nonlocal sql
        if value != sql:
            result["transformations"].append({"name": name, "before": sql, "after": value})
            sql = value

    change("strip_whitespace", sql.strip())
    if sql.startswith("```"):
        fenced = re.fullmatch(r"```(?:sql|postgresql|postgres|sqlite)?\s*\n(.*?)\n```", sql, re.I | re.S)
        if not fenced or any("```" in value for kind, value in sqlparse.lexer.tokenize(fenced.group(1))
                             if kind not in sqlparse.tokens.Literal.String and kind not in sqlparse.tokens.Comment):
            result["extraction_error"] = "Expected exactly one SQL code block"
            return result
        change("remove_markdown_fence", fenced.group(1))
        change("strip_whitespace", sql.strip())
    if dialect == "sqlite":
        change("native.whitespace_cleanup", " ".join(sql.replace("\n", " ").split()))
        change("native.process_duplication", native.process_duplication(sql))
    else:
        # PG JSON/quoted values may literally contain /*. Cut only a real
        # trailing example comment; preserve strings and casts byte-for-byte.
        offset = 0
        for token, value in sqlparse.lexer.tokenize(sql):
            if token in sqlparse.tokens.Comment.Multiline and re.match(r"/\*\s*(Answer the following:|Some SQL examples are provided)", value):
                change("postgresql.trailing_example_comment", sql[:offset])
                break
            offset += len(value)
    first_token = next((value.upper() for token, value in sqlparse.lexer.tokenize(sql)
                        if token not in sqlparse.tokens.Whitespace and token not in sqlparse.tokens.Comment), None)
    if not sql.strip():
        result["extraction_error"] = "Empty SQL choice"
    elif first_token in ("SELECT", "WITH"):
        result["candidate_sql"] = sql
    else:
        if dialect == "postgresql":
            try:
                native.parse_postgres_query(sql)
            except (sqlglot.errors.SqlglotError, ValueError):
                pass
            else:
                result["candidate_sql"] = sql
                return result
        proposed = "SELECT " + sql
        try:
            # Validate SQLite continuations after its native operator repair,
            # without applying that repair to the retained candidate text.
            # Actual changes and their evidence still belong to vote processing.
            validation_sql = native.postprocess(proposed) if dialect == "sqlite" else proposed
            if dialect == "postgresql":
                native.parse_postgres_query(validation_sql)
            else:
                parsed = sqlglot.parse(validation_sql, read="sqlite", error_level=sqlglot.ErrorLevel.RAISE)
                if len(parsed) != 1 or not isinstance(parsed[0], exp.Query):
                    raise ValueError("Expected one query")
        except (sqlglot.errors.SqlglotError, ValueError):
            result["extraction_error"] = "Choice is neither SQL nor a valid SELECT continuation"
        else:
            change("native.select_continuation_prefix", proposed)
            result["candidate_sql"] = sql
    return result
