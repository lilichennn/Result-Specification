"""Offline tests for optional native DeepEye embedding hooks."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from openai import APITimeoutError
from tenacity import wait_none


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines/DeepEye-SQL"
sys.path.insert(0, str(BASELINE_ROOT))


def _embedding_config(store_root: Path | None = None):
    return SimpleNamespace(
        embedding_model_name_or_path="offline-fixture",
        api_type="openai",
        use_qwen3_embedding=False,
        local_files_only=True,
        normalize_embeddings=False,
        base_url="https://example.test/v1",
        api_key="unused",
        embedding_device="cpu",
        store_root_path=str(store_root) if store_root is not None else "unused",
        max_value_length=100,
        lower_meta_data=True,
        build_backend="local_index",
    )


def _fixed_embeddings(texts):
    return [[float(len(text)), 1.0] for text in texts]


class EmbeddingHookTests(unittest.TestCase):
    def test_managed_keyword_embeddings_skip_blank_native_fallback_terms(self):
        from app.pipeline.value_retrieval.utils import embed_keywords
        calls = []
        def embedding(texts):
            calls.append(texts)
            return [[1., 2.] for _ in texts]
        embedding.manages_retries = True
        words = ['', 'singers', ' ']
        self.assertEqual(embed_keywords(words, embedding, 20), [[1., 2.]])
        self.assertEqual(calls, [['singers']])
        self.assertEqual(words, ['', 'singers', ' '], 'keep the native keyword record unchanged')
        embedding.manages_retries = False
        calls.clear()
        self.assertEqual(len(embed_keywords(words, embedding, 20)), 3)
        self.assertEqual(calls, [words])

    def test_index_builder_uses_injected_function_without_native_factory(self):
        from app.few_shot.index_builder import build_few_shot_index

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.json").write_text(json.dumps([
                {"question_id": 1, "db_id": "a", "question": "one", "SQL": "select 1"},
                {"question_id": 2, "db_id": "b", "question": "three", "SQL": "select 22"},
            ]))
            output = root / "index"
            with patch("app.few_shot.index_builder.get_embedding_function",
                       side_effect=AssertionError("native embedding factory must stay lazy")):
                result = build_few_shot_index(
                    dataset_type="bird",
                    root_path=root,
                    save_path=output,
                    embedding_config=_embedding_config(),
                    embedding_batch_size=1,
                    skip_mask_llm=True,
                    embedding_function=_fixed_embeddings,
                )

            self.assertFalse(result.skipped)
            question_matrix = np.load(output / "question_embeddings.npy")
            self.assertEqual(question_matrix.shape, (2, 2))
            np.testing.assert_allclose(np.linalg.norm(question_matrix, axis=1), [1.0, 1.0])

    def test_index_missing_shards_use_ordered_map_overlap_and_resume(self):
        from app.few_shot.index_builder import _embed_texts

        texts = ["0", "1", "2", "3", "4", "5"]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            calls = []

            def interrupted(batch):
                calls.append(tuple(batch))
                if batch[0] == "2":
                    raise RuntimeError("interrupted")
                return [[float(int(text) + 1), 1.0] for text in batch]

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                _embed_texts(
                    texts=texts,
                    embedding_function=interrupted,
                    embedding_batch_size=2,
                    label="fixture",
                    progress_log_interval=100,
                    checkpoint_dir=checkpoint,
                    checkpoint_key="stable",
                )
            self.assertEqual(calls, [("0", "1"), ("2", "3")])
            self.assertTrue((checkpoint / "00000000_00000002.npy").is_file())

            barrier = threading.Barrier(2)
            resumed_calls = []
            finished = []
            call_lock = threading.Lock()

            def resumed(batch):
                with call_lock:
                    resumed_calls.append(tuple(batch))
                barrier.wait(timeout=2)
                if batch[0] == "2":
                    time.sleep(0.05)
                result = [[float(int(text) + 1), 1.0] for text in batch]
                with call_lock:
                    finished.append(tuple(batch))
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                matrix = _embed_texts(
                    texts=texts,
                    embedding_function=resumed,
                    embedding_batch_size=2,
                    label="fixture",
                    progress_log_interval=100,
                    checkpoint_dir=checkpoint,
                    checkpoint_key="stable",
                    embedding_map=executor.map,
                )

            self.assertCountEqual(resumed_calls, [("2", "3"), ("4", "5")])
            self.assertEqual(finished[0], ("4", "5"))
            raw = np.asarray([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]], dtype=np.float32)
            expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            np.testing.assert_allclose(matrix, expected)
            self.assertEqual(len(list(checkpoint.glob("*.npy"))), 3)

            serial_matrix = _embed_texts(
                texts=texts,
                embedding_function=lambda batch: [[float(int(text) + 1), 1.0] for text in batch],
                embedding_batch_size=2,
                label="serial fixture",
                progress_log_interval=100,
                checkpoint_dir=Path(temporary) / "serial-checkpoint",
                checkpoint_key="stable",
            )
            np.testing.assert_allclose(matrix, serial_matrix)

    def test_retriever_injection_is_used_and_default_factory_is_preserved(self):
        from app.few_shot.retriever import FewShotIndex, FewShotRetriever

        index = FewShotIndex(
            index_path="unused",
            records=[{"example_id": "one", "question": "q", "sql": "s"}],
            question_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            sql_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
            manifest={},
        )
        with patch("app.few_shot.retriever.get_embedding_function",
                   side_effect=AssertionError("factory must not be called")):
            retriever = FewShotRetriever(
                index=index,
                embedding_config=_embedding_config(),
                embedding_batch_size=1,
                embedding_function=_fixed_embeddings,
            )
            embedded = retriever._embed_texts(["a", "bbb"])
        np.testing.assert_allclose(embedded, [[1.0, 1.0], [3.0, 1.0]])

        with patch("app.few_shot.retriever.get_embedding_function", return_value=_fixed_embeddings) as factory:
            default_retriever = FewShotRetriever(index=index, embedding_config=_embedding_config())
            default_retriever._embed_texts(["a"])
        factory.assert_called_once()

    def test_value_retrieval_runner_accepts_injected_function_without_factory(self):
        from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SimpleNamespace(
                value_retrieval_config=SimpleNamespace(
                    save_path=str(root / "value.snapshot"),
                    llm=object(),
                    backend="local_index",
                    local_index_device="cpu",
                ),
                dataset_config=SimpleNamespace(
                    type="bird", save_path=str(root / "input.snapshot"), max_value_example_length=50,
                ),
                vector_database_config=_embedding_config(root / "vectors"),
                llm_extractor_config=SimpleNamespace(max_retry=2),
                run_config=SimpleNamespace(
                    parallelism=1, embedding_batch_size=2, progress_log_interval=10, checkpoint_interval=10,
                ),
            )
            with patch("app.pipeline.value_retrieval.value_retrieval.ArtifactStore", return_value=object()), \
                 patch("app.pipeline.value_retrieval.value_retrieval.load_stage_dataset", return_value=([], "fixture")), \
                 patch("app.pipeline.value_retrieval.value_retrieval.configure_schema_service"), \
                 patch("app.pipeline.value_retrieval.value_retrieval.LLM", return_value=object()), \
                 patch("app.pipeline.value_retrieval.value_retrieval.get_embedding_function",
                       side_effect=AssertionError("factory must not be called")):
                runner = ValueRetrievalRunner.from_config(config, embedding_function=_fixed_embeddings)
            try:
                self.assertIs(runner._embedding_function, _fixed_embeddings)
            finally:
                runner._thread_pool_executor.shutdown()
                runner._column_query_executor.shutdown()

    def test_vector_runner_injection_bypasses_factory_and_default_uses_it(self):
        from runner.create_vector_db_parallel import make_vector_db_for_db_path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "fixture.sqlite"
            with sqlite3.connect(database) as connection:
                connection.executescript("CREATE TABLE words(value TEXT); INSERT INTO words VALUES ('a'), ('bbb');")

            injected_config = _embedding_config(root / "injected")
            with patch("runner.create_vector_db_parallel._get_worker_embedding_function",
                       side_effect=AssertionError("factory must not be called")):
                injected_result = make_vector_db_for_db_path(
                    str(database),
                    injected_config,
                    parallelism=1,
                    embedding_batch_size=2,
                    work_semaphore=threading.BoundedSemaphore(1),
                    progress_log_interval=10,
                    embedding_function=_fixed_embeddings,
                )
            self.assertTrue(injected_result)

            default_config = _embedding_config(root / "default")
            with patch("runner.create_vector_db_parallel._get_worker_embedding_function",
                       return_value=_fixed_embeddings) as factory:
                default_result = make_vector_db_for_db_path(
                    str(database),
                    default_config,
                    parallelism=1,
                    embedding_batch_size=2,
                    work_semaphore=threading.BoundedSemaphore(1),
                    progress_log_interval=10,
                )
            self.assertTrue(default_result)
            factory.assert_called_once_with(default_config)

    def test_managed_embedding_failure_bypasses_outer_retry(self):
        from app.pipeline.value_retrieval import utils

        timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))

        class ManagedFailure:
            manages_retries = True

            def __init__(self):
                self.calls = 0

            def __call__(self, batch):
                self.calls += 1
                raise timeout

        class UnmanagedTransient:
            def __init__(self):
                self.calls = 0

            def __call__(self, batch):
                self.calls += 1
                if self.calls == 1:
                    raise timeout
                return [[1.0, 0.0] for _ in batch]

        original_wait = utils.embed_keywords.retry.wait
        utils.embed_keywords.retry.wait = wait_none()
        self.addCleanup(setattr, utils.embed_keywords.retry, "wait", original_wait)

        managed = ManagedFailure()
        with self.assertRaises(APITimeoutError):
            utils.embed_keywords(["one"], managed, embedding_batch_size=1)
        self.assertEqual(managed.calls, 1)

        unmanaged = UnmanagedTransient()
        self.assertEqual(utils.embed_keywords(["one"], unmanaged, embedding_batch_size=1), [[1.0, 0.0]])
        self.assertEqual(unmanaged.calls, 2)


if __name__ == "__main__":
    unittest.main()
