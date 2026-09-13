from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/DeepEye-SQL"))

from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_usage import observed_usage


class ObservedUsageTests(unittest.TestCase):
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
