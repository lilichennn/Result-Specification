from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/DeepEye-SQL"))

from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_usage import observed_usage


class ObservedUsageTests(unittest.TestCase):
    def test_duplicate_sample_results_are_rejected_instead_of_double_counted(self):
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {}) as store:
            attempt = store.begin_attempt('q', 'stage', 'fp')
            store.append_event(attempt, 'sampling_group_start', {'group_id': 'g', 'target_n': 1})
            for _ in range(2):
                store.append_event(attempt, 'sample_result', {'group_id': 'g', 'sample_index': 0,
                    'succeeded': True, 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}})
            with self.assertRaisesRegex(ValueError, 'duplicate sample'):
                observed_usage(store)

    def test_started_group_without_terminal_is_incomplete(self):
        from scripts.baseline_adapters.deepeye.run_usage import sampling_completeness
        self.assertFalse(sampling_completeness([{'kind': 'sampling_group_start',
            'payload': {'group_id': 'g', 'target_n': 5}}])['complete'])

    def test_sampling_ledger_separates_effective_reported_and_unknown(self):
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from app.llm_extractor import LLMExtractor
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {}) as store:
            llm, calls = llm_fixture([response()] * 3 + [response('bad', 7)] + [response()] * 2)
            recorder = TraceRecorder(store)
            cleanup = recorder.instrument_runner(SimpleNamespace(_llm=llm, _checkers=[]), 'sql_revision')
            try:
                attempt = store.begin_attempt('q', 'stage', 'fp')
                with recorder.context(attempt):
                    LLMExtractor().extract_with_retry(llm, [], parse, n=5)
                usage = observed_usage(store)
                self.assertEqual(usage['reported_tokens']['total_tokens'], 157)
                self.assertEqual(usage['effective_sampling']['known_tokens']['total_tokens'], 150)
                self.assertTrue(usage['effective_sampling']['usage_complete'])
                self.assertIsNone(usage['effective_sampling']['reasoning_tokens'])
                self.assertEqual(usage['effective_sampling']['retained_samples'], 5)
            finally:
                cleanup()

    def test_error_response_usage_and_missing_success_usage_are_not_zero(self):
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from app.llm_extractor import LLMExtractor
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from types import SimpleNamespace
        import httpx
        from openai import InternalServerError
        error = InternalServerError('fixture', response=httpx.Response(500,
            request=httpx.Request('POST', 'https://invalid.test')), body={'usage': {
                'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}})
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {}) as store:
            llm, calls = llm_fixture([error, response(tokens=None), response(reasoning=12)])
            recorder = TraceRecorder(store)
            cleanup = recorder.instrument_runner(SimpleNamespace(_llm=llm, _checkers=[]), 'sql_revision')
            try:
                attempt = store.begin_attempt('q', 'stage', 'fp')
                with recorder.context(attempt):
                    LLMExtractor().extract_with_retry(llm, [], parse, n=2)
                usage = observed_usage(store)
                self.assertEqual(usage['reported_tokens']['total_tokens'], 37)
                self.assertEqual(usage['unknown_usage_attempts'], 1)
                self.assertEqual(usage['known_reported_reasoning_tokens'], 12)
                self.assertIsNone(usage['reported_reasoning_tokens'])
                effective = usage['effective_sampling']
                self.assertEqual(effective['known_tokens']['total_tokens'], 30)
                self.assertEqual(effective['retained_samples'], 2)
                self.assertEqual(effective['samples_missing_usage'], 1)
                self.assertFalse(effective['usage_complete'])
                self.assertEqual(effective['known_reasoning_tokens'], 12)
                self.assertIsNone(effective['reasoning_tokens'])
            finally:
                cleanup()

    def test_exhausted_group_reports_120_effective_and_four_unknown_errors(self):
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from app.llm_extractor import LLMExtractor
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from types import SimpleNamespace
        import httpx
        from openai import APIConnectionError
        error = APIConnectionError(request=httpx.Request('POST', 'https://invalid.test'))
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {}) as store:
            llm, calls = llm_fixture([response()] * 3 + [error] * 4 + [response()])
            recorder = TraceRecorder(store)
            cleanup = recorder.instrument_runner(SimpleNamespace(_llm=llm, _checkers=[]), 'sql_revision')
            try:
                attempt = store.begin_attempt('q', 'stage', 'fp')
                with recorder.context(attempt):
                    LLMExtractor().extract_with_retry(llm, [], parse, n=5)
                usage = observed_usage(store)
                self.assertEqual((usage['requests'], usage['unknown_usage_attempts']), (8, 4))
                self.assertEqual(usage['effective_sampling']['known_tokens']['total_tokens'], 120)
                self.assertEqual(usage['reported_tokens']['total_tokens'], 120)
                self.assertFalse(usage['sampling']['complete'])
                self.assertFalse(usage['usage_complete'])
            finally:
                cleanup()

    def test_counts_reported_usage_across_success_failed_and_interrupted_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                failed = store.begin_attempt("failed", "stage", "fp")
                store.append_event(failed, "api_request", {"call_id": "reported"})
                store.append_event(failed, "api_response", {
                    "call_id": "reported",
                    "response": {"usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    }},
                })
                store.finish_attempt(failed, "failed", {})

                succeeded = store.begin_attempt("succeeded", "stage", "fp")
                store.append_event(succeeded, "api_request", {"call_id": "errored"})
                store.append_event(succeeded, "api_error", {"call_id": "errored"})
                store.append_event(succeeded, "api_request", {"call_id": "missing"})
                store.append_event(succeeded, "api_response", {
                    "call_id": "missing",
                    "response": {"id": "no-usage"},
                })
                store.finish_attempt(succeeded, "succeeded", {})

                interrupted = store.begin_attempt("interrupted", "stage", "fp")
                store.append_event(interrupted, "api_request", {"call_id": "in-flight"})
                store.append_event(interrupted, "sql_execute", {"sql": "large irrelevant output"})

                self.assertEqual(observed_usage(store), {
                    "reported_tokens": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                    "requests": 4,
                    "responses": 2,
                    "errors": 1,
                    "unanswered_requests": 1,
                    "responses_missing_usage": 1,
                    "usage_complete": False,
                    "semantics": "reported_tokens_only_not_provider_bill",
                })

    def test_complete_usage_requires_one_reported_response_per_request(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                attempt = store.begin_attempt("q", "stage", "fp")
                store.append_event(attempt, "api_request", {"call_id": "one"})
                store.append_event(attempt, "api_response", {
                    "call_id": "one",
                    "response": {"usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 4,
                        "total_tokens": 11,
                    }},
                })
                self.assertTrue(observed_usage(store)["usage_complete"])

    def test_duplicate_or_unpaired_call_ids_are_rejected(self):
        cases = (
            (("api_request", "api_request"), "duplicate request"),
            (("api_request", "api_response", "api_response"), "duplicate response"),
            (("api_response",), "response without request"),
            (("api_request", "api_response", "api_error"), "two terminal events"),
        )
        for kinds, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                with RunStore.create(Path(temp) / "run", {}) as store:
                    attempt = store.begin_attempt("q", "stage", "fp")
                    for kind in kinds:
                        payload = {"call_id": "same"}
                        if kind == "api_response":
                            payload["response"] = {"usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            }}
                        store.append_event(attempt, kind, payload)
                    with self.assertRaises(ValueError):
                        observed_usage(store)


if __name__ == "__main__":
    unittest.main()
