"""Content-addressed, validated float32 embedding cache (no service credentials)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading

import numpy as np


def fingerprint(data) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def atomic_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_vectors(values, count: int, dimension: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype='<f4')
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1:
        raise ValueError('Embedding response count or shape is invalid')
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError('Embedding dimension changed within one model namespace')
    if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError('Embedding vectors must be finite and nonzero')
    return array


class VectorCache:
    """Thread-safe SQLite storage; submit disjoint batches to avoid duplicate in-flight calls."""

    def __init__(self, path: Path, namespace: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = fingerprint(namespace)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self._connection.execute('PRAGMA journal_mode=WAL')
        self._connection.execute('PRAGMA synchronous=FULL')
        self._connection.executescript('''
            CREATE TABLE IF NOT EXISTS namespaces (id TEXT PRIMARY KEY, config TEXT NOT NULL, dimension INTEGER);
            CREATE TABLE IF NOT EXISTS vectors (
                namespace TEXT NOT NULL, text_hash TEXT NOT NULL, text TEXT NOT NULL,
                dimension INTEGER NOT NULL, vector BLOB NOT NULL, checksum TEXT NOT NULL,
                PRIMARY KEY(namespace, text_hash));
        ''')
        with self._connection:
            self._connection.execute('INSERT OR IGNORE INTO namespaces(id, config) VALUES (?, ?)',
                                     (self.namespace, json.dumps(namespace, sort_keys=True)))

    def close(self):
        with self._lock:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def dimension(self):
        with self._lock:
            return self._connection.execute('SELECT dimension FROM namespaces WHERE id=?', (self.namespace,)).fetchone()[0]

    def get(self, text: str):
        if not isinstance(text, str) or not text:
            raise ValueError('Embedding input must be nonempty text')
        with self._lock:
            row = self._connection.execute(
                'SELECT text, dimension, vector, checksum FROM vectors WHERE namespace=? AND text_hash=?',
                (self.namespace, fingerprint(text))).fetchone()
        if row is None:
            return None
        stored_text, dimension, blob, checksum = row
        if stored_text != text or len(blob) != dimension * 4 or hashlib.sha256(blob).hexdigest() != checksum:
            raise ValueError('Corrupted embedding cache entry')
        vector = np.frombuffer(blob, dtype='<f4').copy()
        validate_vectors(vector.reshape(1, -1), 1, self.dimension)
        return vector

    def embed(self, texts, embed_batch, batch_size=20) -> np.ndarray:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Embedding batch size must be positive')
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension or 0), dtype='<f4')
        unique = list(dict.fromkeys(texts))
        found = {text: self.get(text) for text in unique}
        missing = [text for text in unique if found[text] is None]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            array = validate_vectors(embed_batch(batch), len(batch), self.dimension)
            with self._lock, self._connection:
                # Another batch can establish the dimension during the service call.
                validate_vectors(array, len(batch), self.dimension)
                self._connection.execute('UPDATE namespaces SET dimension=? WHERE id=?', (array.shape[1], self.namespace))
                for text, vector in zip(batch, array):
                    blob = vector.tobytes()
                    self._connection.execute('INSERT OR IGNORE INTO vectors VALUES (?, ?, ?, ?, ?, ?)',
                                             (self.namespace, fingerprint(text), text, len(vector), blob, hashlib.sha256(blob).hexdigest()))
                    found[text] = vector
        return np.stack([found[text] for text in texts])

    def read(self, texts):
        def unavailable(_):
            raise ValueError('Required embedding is absent; run precomputation first')
        return self.embed(texts, unavailable)

    def count(self):
        with self._lock:
            return self._connection.execute('SELECT count(*) FROM vectors WHERE namespace=?', (self.namespace,)).fetchone()[0]
