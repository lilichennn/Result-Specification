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
    """Thread-safe float32 storage; request deduplication belongs to the service."""

    def __init__(self, path: Path, namespace: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = fingerprint(namespace)
        self.namespace_config = dict(namespace)
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
        return self.get_many([text])[text]

    def get_many(self, texts):
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError('Embedding input must be text')
        hashes = {fingerprint(text): text for text in dict.fromkeys(texts)}
        found = dict.fromkeys(texts)
        with self._lock:
            dimension = self.dimension
            keys = list(hashes)
            for offset in range(0, len(keys), 500):
                batch = keys[offset:offset + 500]
                placeholders = ','.join('?' for _ in batch)
                rows = self._connection.execute(
                    f'SELECT text_hash, text, dimension, vector, checksum FROM vectors '
                    f'WHERE namespace=? AND text_hash IN ({placeholders})', (self.namespace, *batch))
                for key, stored_text, size, blob, checksum in rows:
                    if (stored_text != hashes[key] or len(blob) != size * 4
                            or hashlib.sha256(blob).hexdigest() != checksum):
                        raise ValueError('Corrupted embedding cache entry')
                    vector = np.frombuffer(blob, dtype='<f4').copy()
                    validate_vectors(vector.reshape(1, -1), 1, dimension)
                    found[stored_text] = vector
        return found

    def put_many(self, texts, vectors):
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError('Embedding input must be text')
        if not texts:
            return
        with self._lock, self._connection:
            array = validate_vectors(vectors, len(texts), self.dimension)
            rows = []
            for text, vector in zip(texts, array):
                blob = vector.tobytes()
                rows.append((self.namespace, fingerprint(text), text, len(vector), blob,
                             hashlib.sha256(blob).hexdigest()))
            self._connection.execute('UPDATE namespaces SET dimension=? WHERE id=?',
                                     (array.shape[1], self.namespace))
            self._connection.executemany('INSERT OR IGNORE INTO vectors VALUES (?, ?, ?, ?, ?, ?)', rows)

    def embed(self, texts, embed_batch, batch_size=20) -> np.ndarray:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Embedding batch size must be positive')
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension or 0), dtype='<f4')
        unique = list(dict.fromkeys(texts))
        found = self.get_many(unique)
        missing = [text for text in unique if found[text] is None]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            array = validate_vectors(embed_batch(batch), len(batch), self.dimension)
            self.put_many(batch, array)
            found.update(zip(batch, array))
        return np.stack([found[text] for text in texts])

    def read(self, texts):
        def unavailable(_):
            raise ValueError('Required embedding is absent; run precomputation first')
        return self.embed(texts, unavailable)

    def count(self):
        with self._lock:
            return self._connection.execute('SELECT count(*) FROM vectors WHERE namespace=?', (self.namespace,)).fetchone()[0]
