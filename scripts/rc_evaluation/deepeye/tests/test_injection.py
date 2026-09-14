from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
from unittest import TestCase

CODE_ROOT = Path(__file__).resolve().parents[4]
DEEPEYE_ROOT = CODE_ROOT / "baselines" / "DeepEye-SQL"
if str(DEEPEYE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPEYE_ROOT))

from app.prompt.factory import PromptFactory
from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
from scripts.rc_evaluation.deepeye.injection import (
    count_rc_requests,
    install_rc_prompts,
    rc_context,
    render_rc_block,
)


ROUND2 = {
    "population": "eligible entities",
    "row_grain": "one eligible entity",
    "column_role": "the requested values",
    "derivation": "the requested computation",
    "filter_policy": "the requested relative selection",
    "meta_review": "the accepted metadata contribution",
}


def _contract(task_key: str = "lite/item_1", suffix: str = "") -> dict:
    return {
        "task_key": task_key,
        "db_id": "secret_database",
        "question": f"QUESTION_SENTINEL{suffix}",
        "evidence": f"EVIDENCE_SENTINEL{suffix}",
        "round1": {
            "population": f"old population{suffix}",
            "row_grain": f"old row{suffix}",
            "column_role": f"old column{suffix}",
            "derivation": f"old derivation{suffix}",
            "filter_policy": f"old filter{suffix}",
        },
        "round2": {key: f"{value}{suffix}" for key, value in ROUND2.items()},
    }


def _expected_block(contract: dict) -> str:
    round2 = json.dumps(contract["round2"], ensure_ascii=False, indent=2)
    return """<result_contract>
Definition: A Result Contract (RC) is the minimal set of semantic commitments that every correct result table must satisfy. It records determined meanings and unresolved parts; it does not provide the query answer.

Field meanings:
- population: The complete semantic set of eligible entities, events, or records. Put every Boolean membership requirement here, including equality, ranges, thresholds, existence conditions, and comparisons involving a derived metric; preserve their AND/OR scope. If a final aggregate counts, sums, or averages qualifying items, population must state every condition controlling which items contribute. If the question uses a separate reference population to define a comparison, name the target and reference populations separately. A qualifier of a requested metric is not automatically a membership condition on the entity.
- row_grain: What one result row semantically represents after any requested derivation: for example one school, one transaction, one group, or one global summary. Express semantic multiplicity, not SQL machinery. Do not mention DISTINCT, keys, joins, or deduplication operations.
- column_role: Only the semantic roles of values that must appear in the returned table, including stated optionality. Do not include hidden filtering or ranking measures as output columns. Do not replace a requested entity with a guessed physical identifier unless the input explicitly requires that identifier.
- derivation: The semantic computation, comparison, aggregation, or set relation required to define an output value or row grain. State operands and aggregation domains only when the input determines them. Describe the required semantic metric without deciding whether it is stored directly or computed at query time. Do not put top-k, extremum retention, final ordering, or other result selection here. Use exactly "none" when no derivation is required.
- filter_policy: Relative choice or cardinality control applied after the eligible population is defined, such as argmax or argmin, top-k, first or latest, final ordering, or a limit. Equality, range, threshold, existence, and other true-or-false eligibility conditions belong in population even when evaluating them requires a derived metric. If relative selection is requested but its ranking criterion is not determined, state precisely what is unspecified instead of inventing it. Use exactly "none" when no relative selection or output limit is requested.
- meta_review: the metadata contribution accepted into the final contract, or that no relevant refinement was found.

Final RC:
""" + round2 + """

When completing the current stage task, consult this RC.
</result_contract>"""


class RenderRCBlockTest(TestCase):
    def test_native_report_is_not_labeled_as_an_rc2_experiment(self):
        from scripts.rc_evaluation.deepeye.injection import rc_labels
        self.assertEqual(rc_labels({'format': 'deepeye-run-v2'}),
                         {'rc_version': None, 'gold_corrected': False})

    def test_explicit_final_value_and_frozen_template_override_live_definition(self):
        contract = {**_contract(), 'rc_version': 3,
                    'final_rc': {**ROUND2, 'population': 'unique-round-three'}}
        block = render_rc_block(contract, prompt_template='frozen-definition\nFinal RC:\n<<FINAL_RC>>')
        self.assertTrue(block.startswith('frozen-definition\nFinal RC:'))
        self.assertIn('unique-round-three', block)
        self.assertNotIn('eligible entities', block)

    def test_explicit_version_cannot_render_legacy_round2_without_final_value(self):
        with self.assertRaises(ValueError):
            render_rc_block({**_contract(), 'rc_version': 3})

    def test_fixed_definition_contains_only_six_meanings_final_rc_and_request(self) -> None:
        contract = _contract()

        block = render_rc_block(contract)

        self.assertEqual(block, _expected_block(contract))
        self.assertNotIn("QUESTION_SENTINEL", block)
        self.assertNotIn("EVIDENCE_SENTINEL", block)
        self.assertNotIn("old population", block)
        self.assertNotIn("SQL example", block)
        self.assertNotIn("gold", block.casefold())


class PromptInstallationTest(TestCase):
    def _seven_calls(self, db_type: str | None = None):
        return {
            "direct": lambda: PromptFactory.format_direct_linking_prompt("S", "Q", "E", db_type),
            "skeleton": lambda: PromptFactory.format_skeleton_sql_generation_prompt("S", "Q", "E", db_type),
            "dc": lambda: PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E", db_type),
            "icl": lambda: PromptFactory.format_icl_sql_generation_prompt(
                [{"question": "Example", "sql": "SELECT 1"}], "S", "Q", "E", db_type
            ),
            "execution": lambda: PromptFactory.format_execution_checker_prompt(
                "S", "Q", "E", "SELECT 1", "R", db_type
            ),
            "common": lambda: PromptFactory.format_common_checker_prompt(
                "S", "Q", "E", "SELECT 1", "Advice", db_type
            ),
            "selection": lambda: PromptFactory.format_br_pair_selection_prompt(
                "S", "Q", "E", "SELECT 1", "R1", "SELECT 2", "R2", db_type
            ),
        }

    def test_no_context_and_explicit_none_leave_native_prompts_byte_identical(self) -> None:
        calls = self._seven_calls()
        originals = {name: call() for name, call in calls.items()}

        with install_rc_prompts():
            self.assertEqual({name: call() for name, call in calls.items()}, originals)
            with rc_context("sql_generation", "lite/item_1", None):
                self.assertEqual({name: call() for name, call in calls.items()}, originals)

    def test_all_seven_current_pg_wrappers_receive_the_complete_block(self) -> None:
        undo_postgres = install_postgres_support()
        names = tuple(f"format_{stem}_prompt" for stem in (
            "direct_linking",
            "skeleton_sql_generation",
            "dc_sql_generation",
            "icl_sql_generation",
            "execution_checker",
            "common_checker",
            "br_pair_selection",
        ))
        originals = {name: vars(PromptFactory)[name] for name in names}
        contract = _contract()
        block = _expected_block(contract)
        try:
            calls = self._seven_calls("postgresql")
            without_rc = {name: call() for name, call in calls.items()}
            with install_rc_prompts():
                with rc_context("sql_generation", "lite/item_1", contract):
                    injected = {name: call() for name, call in calls.items()}
            for name in calls:
                with self.subTest(name=name):
                    self.assertEqual(injected[name], f"{without_rc[name]}\n\n{block}")
                    self.assertIn("PostgreSQL", injected[name])
            self.assertEqual({name: vars(PromptFactory)[name] for name in names}, originals)
        finally:
            undo_postgres()

    def test_actual_stage_context_not_prompt_name_controls_injection(self) -> None:
        contract = _contract()
        block = _expected_block(contract)

        with install_rc_prompts():
            for stage in ("schema_linking", "sql_generation", "sql_revision", "sql_selection"):
                with self.subTest(stage=stage):
                    with rc_context(stage, "lite/item_1", contract):
                        # Reversed schema linking deliberately reuses this generation formatter.
                        prompt = PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E")
                    self.assertTrue(prompt.endswith(block))

    def test_nested_context_restores_outer_contract_and_none_suppresses_it(self) -> None:
        outer = _contract(suffix=" outer")
        inner = _contract("lite/item_2", suffix=" inner")
        with install_rc_prompts():
            with rc_context("sql_generation", "lite/item_1", outer):
                outer_before = PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E")
                with rc_context("sql_generation", "lite/item_2", inner):
                    inner_prompt = PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E")
                with rc_context("sql_generation", "lite/item_1", None):
                    none_prompt = PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E")
                outer_after = PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E")

        self.assertTrue(outer_before.endswith(_expected_block(outer)))
        self.assertTrue(inner_prompt.endswith(_expected_block(inner)))
        self.assertNotIn("<result_contract>", none_prompt)
        self.assertEqual(outer_after, outer_before)

    def test_contextvars_keep_concurrent_tasks_isolated(self) -> None:
        barrier = threading.Barrier(2)

        def format_for(contract: dict) -> str:
            with rc_context("sql_revision", contract["task_key"], contract):
                barrier.wait(timeout=5)
                return PromptFactory.format_common_checker_prompt(
                    "S", "Q", "E", "SELECT 1", "Advice"
                )

        first = _contract("lite/item_1", " first")
        second = _contract("full/item_2", " second")
        with install_rc_prompts(), ThreadPoolExecutor(max_workers=2) as executor:
            prompts = list(executor.map(format_for, (first, second)))

        self.assertIn(_expected_block(first), prompts[0])
        self.assertNotIn(_expected_block(second), prompts[0])
        self.assertIn(_expected_block(second), prompts[1])
        self.assertNotIn(_expected_block(first), prompts[1])


class CountRCRequestsTest(TestCase):
    def test_counts_only_api_requests_containing_one_complete_block(self) -> None:
        block = _expected_block(_contract())
        events = [
            {"kind": "prompt_formatted", "payload": {"prompt": block}},
            {"kind": "api_request", "payload": {"kwargs": {"messages": [
                {"role": "user", "content": f"native prompt\n\n{block}"}
            ]}}},
            {"kind": "api_request", "payload": {"kwargs": {"messages": [
                {"role": "user", "content": block[:-1]}
            ]}}},
            {"kind": "api_response", "payload": {"response": {"echo": block}}},
            {"kind": "api_request", "payload": {"args": [], "kwargs": {
                "messages": [{"role": "user", "content": [{"type": "text", "text": block}]}]
            }}},
        ]

        self.assertEqual(count_rc_requests(events, block), 2)

    def test_block_in_trace_metadata_but_not_messages_does_not_count(self) -> None:
        block = _expected_block(_contract())
        events = [{
            "kind": "api_request",
            "payload": {
                "branch_path": [block],
                "component_call_id": block,
                "kwargs": {"messages": [{"role": "user", "content": "native only"}]},
            },
        }]

        self.assertEqual(count_rc_requests(events, block), 0)

    def test_progressive_prompt_formatting_without_api_request_counts_zero(self) -> None:
        contract = _contract()
        block = _expected_block(contract)
        with install_rc_prompts(), rc_context("schema_linking", "lite/item_1", contract):
            formatted = [
                PromptFactory.format_direct_linking_prompt("S", "Q", "E"),
                PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E"),
                PromptFactory.format_dc_sql_generation_prompt("S2", "Q", "E"),
            ]

        events = [{"kind": "prompt_formatted", "payload": {"prompt": prompt}} for prompt in formatted]
        self.assertEqual(count_rc_requests(events, block), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
