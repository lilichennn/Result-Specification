"""Reversible, process-local PostgreSQL interfaces for unmodified DeepEye.

Install before constructing runners, and undo after all runner threads finish.
Only one installation at a time is supported. Non-PG behavior delegates to the
original callables. No source files, native configuration or databases are edited.
"""
from functools import wraps
import inspect

from . import backend_hooks
from .postgres_prompt_template import postgres_template

_installed = False


def _prompt_wrapper(original, template, format_example):
    signature = inspect.signature(original)
    names = {"database_schema": "DATABASE_SCHEMA", "question": "QUESTION", "hint": "HINT",
             "sql": "QUERY", "execution_result": "RESULT", "suggestions": "SUGGESTIONS",
             "query_a": "QUERY_A", "query_b": "QUERY_B", "result_a": "RESULT_A", "result_b": "RESULT_B"}

    @wraps(original)
    def format_prompt(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("db_type") != "postgresql":
            return original(*args, **kwargs)
        values = {target: bound.arguments[source] for source, target in names.items() if source in bound.arguments}
        if "few_shot_examples" in bound.arguments:
            values["FEW_SHOT_EXAMPLES"] = "\n".join(
                format_example(i + 1, example)
                for i, example in enumerate(bound.arguments["few_shot_examples"])
            )
        return postgres_template(template).format(**values)
    return format_prompt


def install_postgres_support():
    """Install external wrappers and return an idempotent restoration callable."""
    global _installed
    if _installed:
        raise RuntimeError("DeepEye PostgreSQL support is already installed")
    from app import db_utils
    from app.db_utils import execution, schema
    from app.pipeline import utils
    from app.pipeline.value_retrieval import value_retrieval
    from app.pipeline.schema_linking.schema_linking import SchemaLinkingRunner
    from app.pipeline.sql_revision.checkers.time_checker import TimeChecker
    from app.prompt import factory, prompt_template
    from app.services import execution_service, schema_service

    changes = []
    missing = object()
    undone = False

    def patch(owner, name, value):
        changes.append((owner, name, vars(owner).get(name, missing)))
        setattr(owner, name, value)

    def undo():
        global _installed
        nonlocal undone
        if undone:
            return
        undone = True
        while changes:
            owner, name, original = changes.pop()
            if original is missing:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        execution_service.reset_execution_service()
        schema_service.reset_schema_service()
        _installed = False

    execution_service.reset_execution_service()
    schema_service.reset_schema_service()
    try:
        for name, make in (
            ("execute_sql_for_data_item", backend_hooks.make_execute_sql_for_data_item),
            ("measure_execution_time_for_data_item", backend_hooks.make_measure_execution_time_for_data_item),
        ):
            wrapper = make(getattr(execution, name))
            for owner in (execution, db_utils, execution_service):
                patch(owner, name, wrapper)
        for name, make in (("_build_result_key", backend_hooks.make_result_cache_key),
                           ("_build_time_key", backend_hooks.make_time_cache_key)):
            patch(execution_service.ExecutionService, name,
                  staticmethod(make(getattr(execution_service.ExecutionService, name))))
        result_hash = backend_hooks.make_execution_result_hash(utils.get_execution_result_hash)
        for owner in (utils, execution_service):
            patch(owner, "get_execution_result_hash", result_hash)

        original_predicate = value_retrieval._is_spider2_item
        patch(value_retrieval, "_is_spider2_item", lambda item:
              False if getattr(item, "db_type", None) == "postgresql" else original_predicate(item))

        for stem in ("direct_linking", "skeleton_sql_generation", "dc_sql_generation", "icl_sql_generation",
                     "execution_checker", "common_checker", "br_pair_selection"):
            name = f"format_{stem}_prompt"
            patch(factory.PromptFactory, name, staticmethod(_prompt_wrapper(
                getattr(factory.PromptFactory, name), getattr(prompt_template, f"{stem.upper()}_PROMPT"),
                factory._format_few_shot_example)))

        native_profile = schema.get_database_schema_profile

        @wraps(native_profile)
        def profile(database_schema_dict, *args, **kwargs):
            result = native_profile(database_schema_dict, *args, **kwargs)
            if database_schema_dict.get("db_type") == "postgresql":
                result = result.replace("Schema:\n", "Database Type: POSTGRESQL\nSchema:\n", 1)
            return result

        for owner in (schema, db_utils, schema_service):
            patch(owner, "get_database_schema_profile", profile)
        native_time_check = TimeChecker.check_and_revise

        @wraps(native_time_check)
        def time_check(self, sql, data_item, llm, sampling_budget=1):
            if getattr(data_item, "db_type", None) == "postgresql":
                return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return native_time_check(self, sql, data_item, llm, sampling_budget)

        patch(TimeChecker, "check_and_revise", time_check)
        native_recall = SchemaLinkingRunner._eval_schema_linking_recall

        @wraps(native_recall)
        def recall(self, data_item):
            if getattr(data_item, "db_type", None) == "postgresql" and not data_item.gold_sql:
                for name in ("direct", "reversed", "value", "final"):
                    setattr(data_item, f"{name}_linking_recall", None)
                return
            return native_recall(self, data_item)

        patch(SchemaLinkingRunner, "_eval_schema_linking_recall", recall)
        _installed = True
        return undo
    except BaseException:
        undo()
        raise
