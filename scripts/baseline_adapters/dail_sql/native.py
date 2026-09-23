"""Narrow boundary to trusted DAIL pure functions, without legacy model imports.

The allowlist executes only the named top-level function definitions. Imports,
classes and module statements are never executed; globals are real dependencies.
Source and function hashes are recorded in preparation identity. No sys.modules
replacement is used. The PostgreSQL namespace alone retains structural CTE
AS/opening-parenthesis tokens in native alias removal. Source locations refer
to the fixed local DAIL checkout; the adapter fingerprint covers that repair.
"""

import ast
import collections
import csv
from functools import lru_cache
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import re
import random
import string
from types import SimpleNamespace
from typing import Tuple, List, Set
import warnings

from sql_metadata import Parser
import sqlparse


SOURCE_ROOT = Path(__file__).resolve().parents[3] / "baselines/DAIL-SQL"
ALLOWLIST = {
    "utils/utils.py": ("sql_normalization", "sql2skeleton", "isNegativeInt", "isFloat", "jaccard_similarity"),
    "utils/linking_utils/spider_match_utils.py": ("compute_schema_linking", "compute_cell_value_linking", "match_shift"),
    "utils/linking_utils/application.py": ("mask_question_with_schema_linking",),
}
VOTE_SOURCE = "utils/post_process.py"
VOTE_FUNCTIONS = ("process_duplication", "permute_tuple", "unorder_row", "quick_rej", "multiset_eq",
                  "get_constraint_permutation", "result_eq", "replace_cur_year", "postprocess", "remove_distinct")


def _parse_trusted(source):
    # Pinned upstream YEAR regex uses a non-raw '\\s' string. Its value is
    # unchanged; suppress only this Python 3.12 parser warning at the boundary.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)
        return ast.parse(source)


class SkeletonError(ValueError):
    """SQL could not be parsed into a deterministic retrieval skeleton."""


@lru_cache(maxsize=1)
def source_fingerprints() -> dict:
    result = {}
    for name, functions in {**ALLOWLIST, VOTE_SOURCE: VOTE_FUNCTIONS}.items():
        source = (SOURCE_ROOT / name).read_text(encoding="utf-8")
        nodes = {node.name: node for node in _parse_trusted(source).body if isinstance(node, ast.FunctionDef)}
        if not set(functions) <= nodes.keys():
            raise RuntimeError(f"DAIL native definitions missing: {name}")
        result[name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "functions": {
            function: {"line": nodes[function].lineno, "end_line": nodes[function].end_lineno,
                       "sha256": hashlib.sha256(ast.get_source_segment(source, nodes[function]).encode()).hexdigest()}
            for function in functions}}
    return result


def _preserve_cte_delimiters(definitions):
    """Guard the two native alias-removal skips, without copying its algorithm.

    Native normalization drops AS and its next token. For CTEs that token is
    '(', not an alias; keeping both prevents an unmatched closing parenthesis.
    Patch only the known AST conditions in this isolated function namespace;
    fail closed if upstream changes them. SQLite uses the untouched functions.
    """
    normalizer = next(node for node in definitions if node.name == "sql_normalization")
    remover = next(node for node in normalizer.body
                   if isinstance(node, ast.FunctionDef) and node.name == "remove_table_alias")
    guards = {
        ast.dump(ast.parse("s[i] == 'as'", mode="eval").body):
            "i + 1 >= len(s) or s[i + 1] != '('",
        ast.dump(ast.parse("i > 0 and s[i-1] == 'as'", mode="eval").body):
            "s[i] != '('",
    }
    matches = collections.Counter()
    for node in ast.walk(remover):
        if isinstance(node, ast.If) and (key := ast.dump(node.test)) in guards:
            guard = ast.parse(guards[key], mode="eval").body
            node.test = ast.copy_location(ast.BoolOp(op=ast.And(), values=[node.test, guard]), node.test)
            matches[key] += 1
    if matches != collections.Counter({key: 1 for key in guards}):
        raise RuntimeError("DAIL alias-removal conditions changed; review PostgreSQL CTE compatibility")


@lru_cache(maxsize=4)
def _functions(stopwords_path=None, *, preserve_cte=False):
    namespace = {"re": re, "collections": collections, "Parser": Parser,
                 "PUNKS": set(string.punctuation)}
    if stopwords_path is not None:
        # Explicit CorpusReader root: never consult remote/download machinery.
        from nltk.corpus.reader import WordListCorpusReader
        import nltk.data
        corpus = Path(stopwords_path) / "corpora/stopwords"
        if not (corpus / "english").is_file():
            raise RuntimeError("Local NLTK English stopwords are missing")
        if str(Path(stopwords_path)) not in nltk.data.path:
            nltk.data.path.append(str(Path(stopwords_path)))
        namespace["STOPWORDS"] = set(WordListCorpusReader(str(corpus), ["english"]).words())
    for flag, value in {"CELL_EXACT_MATCH_FLAG": "EXACTMATCH", "CELL_PARTIAL_MATCH_FLAG": "PARTIALMATCH",
                        "COL_PARTIAL_MATCH_FLAG": "CPM", "COL_EXACT_MATCH_FLAG": "CEM",
                        "TAB_PARTIAL_MATCH_FLAG": "TPM", "TAB_EXACT_MATCH_FLAG": "TEM"}.items():
        namespace[flag] = value
    source_fingerprints()
    for name, names in ALLOWLIST.items():
        tree = ast.parse((SOURCE_ROOT / name).read_text(encoding="utf-8"))
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if preserve_cte and name == "utils/utils.py":
            _preserve_cte_delimiters(definitions)
        module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
        exec(compile(module, str(SOURCE_ROOT / name), "exec"), namespace)
    return namespace


@lru_cache(maxsize=1)
def _vote_code():
    tree = _parse_trusted((SOURCE_ROOT / VOTE_SOURCE).read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in VOTE_FUNCTIONS]
    if {node.name for node in definitions} != set(VOTE_FUNCTIONS):
        raise RuntimeError("DAIL native vote definitions missing")
    return compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE_ROOT / VOTE_SOURCE), "exec")


def _vote_functions(rng):
    # Each invocation binds the unchanged native random.choice call to its
    # caller-owned RNG. No mutation of cached globals or process random state.
    namespace = {"Tuple": Tuple, "List": List, "Set": Set, "defaultdict": collections.defaultdict,
                 "product": product, "random": rng, "sqlparse": sqlparse, "re": re}
    exec(_vote_code(), namespace)
    return namespace


def result_eq(result1, result2, order_matters=False, *, rng=None):
    """Fixed DAIL result comparator; independent evaluation does not use this."""
    return _vote_functions(rng if rng is not None else random.Random(0))["result_eq"](result1, result2, order_matters)


def process_duplication(sql):
    return _vote_functions(None)["process_duplication"](sql)


def postprocess(sql):
    return _vote_functions(None)["postprocess"](sql)


def remove_distinct(sql):
    return _vote_functions(None)["remove_distinct"](sql)


def replace_cur_year(sql):
    return _vote_functions(None)["replace_cur_year"](sql)


def mask_question(row: dict) -> str:
    return _functions()["mask_question_with_schema_linking"]([row], "<mask>", "<unk>")[0]


def skeleton_similarity(left: str, right: str) -> float:
    return _functions()["jaccard_similarity"](left, right)


def parse_postgres_query(sql: str):
    """One query, ignoring empty/comment-only statements, not real statements."""
    import sqlglot
    from sqlglot import exp
    statements = [node for node in sqlglot.parse(sql, read="postgres", error_level=sqlglot.ErrorLevel.RAISE)
                  if node is not None and not isinstance(node, exp.Semicolon)]
    if len(statements) != 1 or not isinstance(statements[0], (exp.Query, exp.Values)):
        raise ValueError("Expected one query")
    return statements[0]


def sql_skeleton(sql: str, schema: dict, dialect: str) -> str:
    if dialect not in {"sqlite", "postgresql"}:
        raise SkeletonError("Unsupported SQL dialect")
    try:
        if not sql.strip():
            raise ValueError("Empty SQL")
        if dialect == "postgresql":
            # Parse PostgreSQL first. Canonical unquoted identifiers are then fed
            # through DAIL's unchanged masking/collapse rules; no SQL execution.
            from sqlglot import exp
            tree = parse_postgres_query(sql)
            tables = {name.lower() for name in schema["table_names_original"]}
            columns = {name.lower() for _, name in schema["column_names_original"]}
            # Match parsed public identifiers before unquoting; spaces and
            # PostgreSQL schema qualifiers cannot be flattened as SQL words.
            for column in tree.find_all(exp.Column):
                if column.name.lower() in columns:
                    column.set("this", exp.to_identifier("_"))
                    for part in ("table", "db", "catalog"):
                        column.set(part, None)
            for table in tree.find_all(exp.Table):
                if table.name.lower() in tables:
                    table.set("this", exp.to_identifier("_"))
                    for part in ("db", "catalog", "alias"):
                        table.set(part, None)
            # The native normalizer globally replaces double quotes and splits
            # aliases by spaces. Give only atomic names to that legacy code;
            # preserve repeated references and avoid collisions with real names.
            identifiers = list(tree.find_all(exp.Identifier))
            occupied = {node.name.lower() for node in identifiers}
            names = {}
            for node in identifiers:
                if node.args.get("quoted") and node.name not in names:
                    index = len(names)
                    while (safe := f"dail_pg_identifier_{index}") in occupied:
                        index += 1
                    names[node.name] = safe
                    occupied.add(safe)
            for node in identifiers:
                key = node.name if node.args.get("quoted") else node.name.lower()
                if key in names:
                    node.set("this", names[key])
                node.set("quoted", False)
            sql = tree.sql(dialect="postgres", comments=False)
            # Mask rendered string tokens, including JSON path keys that the
            # PostgreSQL AST does not represent as Literal nodes. Do this before
            # native double-quote replacement can expose literal text as SQL.
            sql = "".join("'_'" if kind in sqlparse.tokens.Literal.String.Single else value
                          for kind, value in sqlparse.lexer.tokenize(sql))
            if isinstance(tree, exp.Values):
                # sql-metadata requires a SELECT root; this equivalent wrapper
                # is used only for retrieval, never for execution or storage.
                sql = f"SELECT * FROM ({sql}) AS dail_pg_values"
        return _functions(preserve_cte=True)["sql2skeleton"](sql, schema) if dialect == "postgresql" else _functions()["sql2skeleton"](sql, schema)
    except Exception as exc:
        raise SkeletonError(f"Unable to parse {dialect} SQL skeleton") from exc


def load_public_schema(directory: Path) -> dict:
    """Only public column/type fields; never read sample cells into the schema."""
    schema = {"table_names_original": [], "table_names": [],
              "column_names_original": [[-1, "*"]], "column_names": [[-1, "*"]],
              "column_types": ["text"], "column_types_original": ["text"],
              "column_descriptions": [""], "column_ref_keys": [""],
              "primary_keys": [], "foreign_keys": [], "unresolved_foreign_keys": []}
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise ValueError("Public Meta CSVs missing")
    pending_foreign_keys = []
    for table_id, path in enumerate(files):
        schema["table_names_original"].append(path.stem)
        schema["table_names"].append(path.stem.replace("_", " "))
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            import chardet
            encoding = chardet.detect(raw)["encoding"]
            if not encoding:
                raise ValueError(f"Cannot identify public Meta encoding: {path}")
            text = raw.decode(encoding)
        with io.StringIO(text, newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                name = row.get("original_column_name") or row.get("column_name")
                if not name:
                    raise ValueError(f"Unrecognized public Meta column: {path}")
                dtype = (row.get("column_type") or row.get("data_format") or "text").lower()
                kind = "number" if any(t in dtype for t in ("int", "real", "number", "numeric", "double", "float", "decimal")) else "time" if any(t in dtype for t in ("date", "time")) else "text"
                schema["column_names_original"].append([table_id, name])
                schema["column_names"].append([table_id, (row.get("column_name") or name).replace("_", " ")])
                schema["column_types"].append(kind)
                schema["column_types_original"].append(row.get("column_type") or row.get("data_format") or "text")
                schema["column_descriptions"].append(row.get("column_description") or "")
                schema["column_ref_keys"].append(row.get("ref_key") or "")
                column_id = len(schema["column_names_original"]) - 1
                if (row.get("primary_key") or "").strip().lower() in {"true", "1", "yes"}:
                    schema["primary_keys"].append(column_id)
                for foreign_key in json.loads(row.get("foreign_keys") or "[]"):
                    if not isinstance(foreign_key, list) or len(foreign_key) != 2:
                        raise ValueError("Public Meta foreign key must contain table and column")
                    pending_foreign_keys.append((column_id, ".".join(foreign_key)))
    qualified = {f"{schema['table_names_original'][table]}.{name}".lower(): index
                 for index, (table, name) in enumerate(schema["column_names_original"]) if table >= 0}
    for index, ref in [*enumerate(schema["column_ref_keys"]), *pending_foreign_keys]:
        for ref in filter(None, (part.strip() for part in ref.split(";"))):
            if ref.lower() not in qualified:
                schema["unresolved_foreign_keys"].append({"column_id": index, "reference": ref})
                continue
            pair = [index, qualified[ref.lower()]]
            if pair not in schema["foreign_keys"]:
                schema["foreign_keys"].append(pair)
    return schema


def tokenize_schema(schema: dict, tokenizer) -> dict:
    return {"columns": [tokenizer.tokenize(name) for _, name in schema["column_names"]],
            "tables": [tokenizer.tokenize(name) for name in schema["table_names"]]}


def link_question(question: str, schema: dict, tokenizer, *, compute_cv_link: bool,
                  stopwords_path: Path, connection=None, tokenized_schema=None) -> dict:
    """Effective SpiderEncoderV2Preproc linking with include_table_name=False.

    Schema tokens can be reused per database. A caller-owned read-only SQLite
    connection is required only for CV; native LIKE and numeric matching remain.
    """
    functions = _functions(str(Path(stopwords_path).resolve()))
    lemmas, copying = tokenizer.tokenize_for_copying(question)
    tokens = tokenized_schema or tokenize_schema(schema, tokenizer)
    sc_link = functions["compute_schema_linking"](lemmas, tokens["columns"], tokens["tables"])
    cv_link = {"num_date_match": {}, "cell_match": {}}
    if compute_cv_link:
        if connection is None:
            raise ValueError("CV linking requires a read-only SQLite connection")
        tables = [SimpleNamespace(orig_name=name) for name in schema["table_names_original"]]
        columns = [SimpleNamespace(orig_name=name, type=kind, table=tables[table_id] if table_id >= 0 else None)
                   for (table_id, name), kind in zip(schema["column_names_original"], schema["column_types"])]
        cv_link = functions["compute_cell_value_linking"](lemmas, SimpleNamespace(columns=columns, connection=connection))
    return {"question": lemmas, "question_for_copying": copying, "sc_link": sc_link, "cv_link": cv_link}
