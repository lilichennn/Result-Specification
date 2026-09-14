"""Offline adaptive API checks using the real SDK and an in-memory HTTP transport."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import openai

from scripts.baseline_adapters.deepeye.embedding_service import EmbeddingLimits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/DeepEye-SQL"))
MODULE = "scripts.baseline_adapters.deepeye.precompute_api"
try:
    api_module = importlib.import_module(MODULE)
except ModuleNotFoundError as exc:
    if exc.name != MODULE:
        raise
    api_module = None

VALUES = {
    "DASH_BASE_URL": "https://offline.invalid/v1", "DASH_API_KEY": "sk-test-secret-chat",
    "DASH_MODELS": "test-chat", "EMBEDDING_BASE_URL": "https://offline.invalid/v1",
    "EMBEDDING_API_KEY": "sk-test-secret-embed", "EMBEDDING_MODEL": "test-embed",
}


def reply(body, status=200, request_id="request-123"):
    return httpx.Response(status, content=json.dumps(body).encode(), headers={
        "content-type": "application/json", "x-request-id": request_id,
    })


def chat_reply(content="answer", tokens=(3, 2, 5)):
    body = {"id": "chat-1", "object": "chat.completion", "created": 1,
            "model": "test-chat", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}]}
    if tokens is not None:
        body["usage"] = dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), tokens))
    return reply(body)


def embed_reply(vectors, indices=None):
    return reply({"object": "list", "model": "test-embed", "data": [
        {"object": "embedding", "index": i, "embedding": vector}
        for i, vector in zip(range(len(vectors)) if indices is None else indices, vectors)
    ], "usage": {"prompt_tokens": 4, "total_tokens": 4}})


class AdaptiveAPITests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(api_module, "Adaptive precompute API is missing")

    @contextmanager
    def api(self, handler, **kwargs):
        self.assertIsNotNone(api_module, "Adaptive precompute API is missing")
        original_client = httpx.Client
        self.pool_limits = []
        pool_limits = self.pool_limits

        class OfflineClient(original_client):
            def __init__(self, *args, **options):
                pool_limits.append(options["limits"])
                options["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **options)

        with tempfile.TemporaryDirectory() as directory:
            self.output = Path(directory)
            kwargs.setdefault('embedding_limits', EmbeddingLimits(dimension=2, retry_delay=0))
            with patch.object(api_module.httpx, "Client", OfflineClient):
                client = api_module.AdaptiveAPI(VALUES, self.output, **kwargs)
            try:
                yield client
            finally:
                client.close()

    def events(self):
        return [json.loads(line) for line in (self.output / "api_calls.jsonl").read_text().splitlines()]

    def test_request_parameters_ordering_and_audit_are_preserved(self):
        requests = []

        def handler(request):
            requests.append((str(request.url), json.loads(request.content), request.extensions))
            return (embed_reply([[3., 4.], [1., 2.]], [1, 0])
                    if request.url.path.endswith("embeddings") else chat_reply("  original body\n"))

        with self.api(handler) as api:
            texts = ["first", "second"]
            messages = [{"role": "user", "content": "private prompt"}]
            self.assertEqual(api.embed(texts), [[1., 2.], [3., 4.]])
            self.assertEqual(api.chat(messages), "  original body\n")
            self.assertEqual(requests[0][1]["encoding_format"], "float")
            self.assertEqual(requests[0][2]["timeout"]["read"], 60)
            self.assertEqual(requests[1][1]["temperature"], 0.6)
            self.assertEqual(requests[1][1]["max_tokens"], 2048)
            self.assertEqual(requests[1][1]["thinking_budget"], 1024)
            self.assertEqual(requests[1][2]["timeout"]["read"], 300)
            self.assertEqual([limit.max_connections for limit in self.pool_limits], [600, 64])
            self.assertEqual([limit.max_keepalive_connections for limit in self.pool_limits], [600, 64])
            events = self.events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["total_tokens"], 5)
            self.assertEqual(events[0]["request_id"], "request-123")
            self.assertEqual(events[0]["input_hash"], hashlib.sha256(json.dumps(
                messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest())
            self.assertNotIn("private prompt", json.dumps(events))
            self.assertEqual(api.summary()["config"]["temperature"], 0.6)
            embedding_events = [json.loads(line) for line in
                (self.output / 'embedding_calls.jsonl').read_text().splitlines()]
            self.assertEqual(sum(event['kind'] == 'embedding_cache' for event in embedding_events), 1)
            embedding_events = [event for event in embedding_events if event['kind'] == 'embedding_request']
            self.assertEqual(len(embedding_events), 1)
            self.assertEqual(embedding_events[0]['usage']['total_tokens'], 4)

    def test_empty_embedding_input_makes_no_request(self):
        with self.api(lambda _: self.fail("Empty input must not call HTTP")) as api:
            self.assertEqual(api.embed([]), [])
            self.assertEqual(api.summary()["attempts"], 0)

    def test_full_chat_admission_does_not_block_shared_embedding_admission(self):
        chat_entered, release_chat = threading.Event(), threading.Event()

        def handler(request):
            if request.url.path.endswith('embeddings'):
                return embed_reply([[1., 2.]])
            chat_entered.set()
            release_chat.wait(timeout=5)
            return chat_reply()

        with self.api(handler, initial_concurrency=1, max_concurrency=1) as api:
            with ThreadPoolExecutor(max_workers=2) as pool:
                chat = pool.submit(api.chat, [])
                try:
                    self.assertTrue(chat_entered.wait(timeout=3))
                    embedding = pool.submit(api.embed, ['independent'])
                    self.assertEqual(embedding.result(timeout=3), [[1., 2.]])
                    self.assertEqual(api.summary()['in_flight'], 1)
                finally:
                    release_chat.set()
                self.assertEqual(chat.result(timeout=3), 'answer')

    def test_malformed_embeddings_are_rejected_before_return(self):
        cases = [([[1., 2.]], [0]), ([[1., 2.], [3., 4.]], [0, 0]),
                 ([[1., 2.], [3., 4.]], [0, 2]), ([[0., 0.], [3., 4.]], [0, 1]),
                 ([[float("nan"), 1.], [3., 4.]], [0, 1]),
                 ([[float("inf"), 1.], [3., 4.]], [0, 1]),
                 ([[1.], [3., 4.]], [0, 1])]
        for vectors, indices in cases:
            with self.subTest(vectors=vectors, indices=indices):
                with self.api(lambda _: embed_reply(vectors, indices)) as api:
                    with self.assertRaises(ValueError):
                        api.embed(["a", "b"])
                    self.assertEqual(api.summary()["in_flight"], 0)

    def test_embedding_dimension_cannot_change_between_batches(self):
        responses = iter([embed_reply([[1., 2.]]), embed_reply([[1., 2., 3.]])])
        with self.api(lambda _: next(responses)) as api:
            self.assertEqual(api.embed(["a"]), [[1., 2.]])
            with self.assertRaises(ValueError):
                api.embed(["b"])

    def test_auth_and_regular_bad_request_are_not_retried(self):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                with self.api(lambda _: reply({"error": {"message": VALUES["DASH_API_KEY"]}}, status,
                                              VALUES["DASH_API_KEY"])) as api:
                    with self.assertRaises(openai.APIStatusError):
                        api.chat([{"role": "user", "content": "x"}])
                    self.assertEqual(api.summary()["attempts"], 1)
                    self.assertNotIn(VALUES["DASH_API_KEY"], json.dumps(self.events()))

    def test_transient_errors_and_empty_content_retry_up_to_four_times(self):
        for failure in (429, 500, "network", "empty"):
            with self.subTest(failure=failure):
                def handler(request):
                    if failure == "network":
                        raise httpx.ConnectError("network down", request=request)
                    if failure == "empty":
                        return chat_reply(" ")
                    return reply({"error": {"message": "transient"}}, failure)

                with self.api(handler) as api, patch.object(api_module, "sleep") as delay:
                    with self.assertRaises(Exception):
                        api.chat([{"role": "user", "content": "x"}])
                    self.assertEqual(api.summary()["attempts"], 5)
                    self.assertEqual(len(self.events()), 5)
                    self.assertEqual(len(delay.call_args_list), 4)
                    self.assertTrue(all(0 <= call.args[0] <= 30 for call in delay.call_args_list))
                    self.assertEqual(api.summary()["in_flight"], 0)

    def test_success_after_retry_has_separate_cost_records(self):
        responses = iter([chat_reply(""), chat_reply("recovered")])
        with self.api(lambda _: next(responses)) as api, patch.object(api_module, "sleep"):
            self.assertEqual(api.chat([]), "recovered")
            events = self.events()
            self.assertEqual([event["status"] for event in events], ["error", "ok"])
            self.assertEqual([event["http_status"] for event in events], [200, 200])
            self.assertEqual([event["total_tokens"] for event in events], [5, 5])
            self.assertEqual(events[0]["input_hash"], events[1]["input_hash"])
            cost = api.usage_for(events[0]["input_hash"])
            self.assertEqual(cost["attempts"], 2)
            self.assertEqual(cost["prompt_tokens"], 6)
            self.assertEqual(cost["completion_tokens"], 4)
            self.assertEqual(cost["total_tokens"], 10)
            self.assertEqual(cost["usage_missing"], 0)

    def test_missing_usage_is_reported_without_inventing_token_counts(self):
        response = {"id": "chat-2", "object": "chat.completion", "created": 1,
                    "model": "test-chat", "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"}}]}
        with self.api(lambda _: reply(response)) as api:
            api.chat([])
            cost = api.usage_for(self.events()[0]["input_hash"])
            self.assertIsNone(cost["total_tokens"])
            self.assertIsNone(cost["prompt_tokens"])
            self.assertIsNone(cost["completion_tokens"])
            self.assertEqual(cost["attempts"], 1)
            self.assertEqual(cost["usage_missing"], 1)

    def test_identical_concurrent_prompts_keep_independent_usage(self):
        entered = threading.Barrier(2)
        sequence_lock = threading.Lock()
        sequence = [0]

        def handler(request):
            with sequence_lock:
                position = sequence[0]
                sequence[0] += 1
            entered.wait(timeout=3)
            return chat_reply("first", (3, 2, 5)) if position == 0 else chat_reply("second", (11, 7, 18))

        with self.api(handler, initial_concurrency=2) as api:
            self.assertTrue(callable(getattr(api, "chat_with_usage", None)),
                            "Independent logical-call cost interface is missing")
            messages = [{"role": "user", "content": "same prompt"}]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(api.chat_with_usage, messages) for _ in range(2)]
                results = [future.result(timeout=5) for future in futures]
            results = {result["content"]: result for result in results}
            first, second = results["first"], results["second"]
            self.assertEqual(first["input_hash"], second["input_hash"])
            self.assertNotEqual(first["call_id"], second["call_id"])
            for result, tokens in ((first, (3, 2, 5)), (second, (11, 7, 18))):
                self.assertEqual(tuple(result["usage"][field] for field in
                    ("prompt_tokens", "completion_tokens", "total_tokens")), tokens)
                self.assertEqual(result["usage"]["attempts"], 1)
                self.assertEqual(result["usage"]["usage_missing"], 0)
                self.assertTrue(result["usage"]["usage_complete"])
                self.assertGreaterEqual(result["usage"]["elapsed_seconds"], 0)
                matching = [event for event in self.events() if event["call_id"] == result["call_id"]]
                self.assertEqual(len(matching), 1)
                self.assertEqual(matching[0]["total_tokens"], tokens[2])

    def test_logical_call_cost_includes_its_retries_but_not_previous_identical_calls(self):
        responses = iter([chat_reply("earlier", (20, 10, 30)),
                          chat_reply("", (3, 2, 5)), chat_reply("recovered", (11, 7, 18))])
        with self.api(lambda _: next(responses)) as api, patch.object(api_module, "sleep"):
            self.assertTrue(callable(getattr(api, "chat_with_usage", None)),
                            "Independent logical-call cost interface is missing")
            earlier = api.chat_with_usage([])
            result = api.chat_with_usage([])
            self.assertEqual(result["content"], "recovered")
            self.assertEqual(result["usage"]["prompt_tokens"], 14)
            self.assertEqual(result["usage"]["completion_tokens"], 9)
            self.assertEqual(result["usage"]["total_tokens"], 23)
            self.assertEqual(result["usage"]["attempts"], 2)
            self.assertEqual(result["usage"]["usage_missing"], 0)
            self.assertNotEqual(earlier["call_id"], result["call_id"])
            events = self.events()
            self.assertEqual([event["call_id"] for event in events],
                             [earlier["call_id"], result["call_id"], result["call_id"]])
            self.assertEqual(api.usage_for(result["input_hash"])["total_tokens"], 53)

    def test_logical_call_with_missing_usage_keeps_none_fields(self):
        with self.api(lambda _: chat_reply(tokens=None)) as api:
            self.assertTrue(callable(getattr(api, "chat_with_usage", None)),
                            "Independent logical-call cost interface is missing")
            result = api.chat_with_usage([])
            self.assertEqual(result["content"], "answer")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.assertIsNone(result["usage"][field])
            self.assertEqual(result["usage"]["attempts"], 1)
            self.assertEqual(result["usage"]["usage_missing"], 1)
            self.assertFalse(result["usage"]["usage_complete"])

    def test_logical_call_retains_reported_retry_subtotal_and_marks_it_incomplete(self):
        responses = iter([chat_reply("", tokens=None), chat_reply("recovered")])
        with self.api(lambda _: next(responses)) as api, patch.object(api_module, "sleep"):
            self.assertTrue(callable(getattr(api, "chat_with_usage", None)),
                            "Independent logical-call cost interface is missing")
            result = api.chat_with_usage([])
            self.assertEqual(result["content"], "recovered")
            self.assertEqual(result["usage"]["prompt_tokens"], 3)
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertEqual(result["usage"]["total_tokens"], 5)
            self.assertEqual(result["usage"]["attempts"], 2)
            self.assertEqual(result["usage"]["usage_missing"], 1)
            self.assertFalse(result["usage"]["usage_complete"])

    def test_virtual_clock_increases_only_after_a_stable_window(self):
        now = [0.]
        with patch.object(api_module, "monotonic", side_effect=lambda: now[0]):
            with self.api(lambda _: chat_reply(), max_concurrency=250) as api:
                for _ in range(50):
                    api.chat([])
                self.assertEqual(api.summary()["concurrency"], 200)
                now[0] = 60.
                api.chat([])
                self.assertEqual(api.summary()["concurrency"], 250)
                now[0] = 120.
                for _ in range(50):
                    api.chat([])
                self.assertEqual(api.summary()["concurrency"], 250)

    def test_virtual_clock_failures_lower_shared_limit_with_cooldown_and_floor(self):
        now = [0.]
        with patch.object(api_module, "monotonic", side_effect=lambda: now[0]):
            with self.api(lambda _: reply({"error": {"message": "denied"}}, 403),
                          initial_concurrency=100, min_concurrency=1) as api:
                def fail_call():
                    with self.assertRaises(openai.APIStatusError):
                        api.chat([])

                for _ in range(4):
                    fail_call()
                self.assertEqual(api.summary()["concurrency"], 100)
                fail_call()
                self.assertEqual(api.summary()["concurrency"], 50)
                for _ in range(5):
                    fail_call()
                self.assertEqual(api.summary()["concurrency"], 50)
                now[0] = 15.
                fail_call()
                self.assertEqual(api.summary()["concurrency"], 1)

    def test_old_failures_cannot_trigger_another_decrease_without_five_new_failures(self):
        now, failing = [0.], [True]
        with patch.object(api_module, "monotonic", side_effect=lambda: now[0]):
            with self.api(lambda _: reply({"error": {"message": "denied"}}, 403)
                          if failing[0] else chat_reply()) as api:
                def fail_call():
                    with self.assertRaises(openai.APIStatusError):
                        api.chat([])

                for _ in range(5):
                    fail_call()
                self.assertEqual(api.summary()["concurrency"], 150)
                now[0] = 15.
                failing[0] = False
                api.chat([])
                self.assertEqual(api.summary()["concurrency"], 150)
                failing[0] = True
                for _ in range(4):
                    fail_call()
                self.assertEqual(api.summary()["concurrency"], 150)
                fail_call()
                self.assertEqual(api.summary()["concurrency"], 100)

    def test_less_than_ten_percent_failures_does_not_reduce_concurrency(self):
        fail = [False]
        with self.api(lambda _: reply({"error": {"message": "denied"}}, 403)
                      if fail[0] else chat_reply()) as api:
            for _ in range(50):
                api.chat([])
            fail[0] = True
            for _ in range(5):
                with self.assertRaises(openai.APIStatusError):
                    api.chat([])
            self.assertEqual(api.summary()["concurrency"], 200)

    def test_real_chat_threads_share_limit_and_release_slots_after_errors(self):
        entered = threading.Condition()
        release = threading.Event()
        state = {"active": 0, "peak": 0, "calls": 0}

        def handler(request):
            with entered:
                state["active"] += 1
                state["calls"] += 1
                sequence = state["calls"]
                state["peak"] = max(state["peak"], state["active"])
                entered.notify_all()
            release.wait(timeout=5)
            with entered:
                state["active"] -= 1
            if sequence == 1:
                return reply({"error": {"message": "denied"}}, 403)
            return embed_reply([[1., 2.]]) if request.url.path.endswith("embeddings") else chat_reply()

        with self.api(handler, initial_concurrency=2, max_concurrency=2) as api:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(api.chat, []) for _ in range(8)]
                try:
                    with entered:
                        self.assertTrue(entered.wait_for(lambda: state["calls"] >= 2, timeout=3))
                    self.assertEqual(api.summary()["in_flight"], 2)
                finally:
                    release.set()
                failures = 0
                for future in futures:
                    try:
                        future.result(timeout=5)
                    except openai.PermissionDeniedError:
                        failures += 1
            self.assertEqual(failures, 1)
            self.assertEqual(state["peak"], 2)
            self.assertEqual(len(self.events()), 8)
            self.assertEqual(api.summary()["in_flight"], 0)

    def test_context_close_prevents_new_http_requests(self):
        with self.api(lambda _: chat_reply()) as api:
            with api:
                self.assertEqual(api.chat([]), "answer")
            with self.assertRaises(RuntimeError):
                api.chat([])
            self.assertEqual(api.summary()["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
