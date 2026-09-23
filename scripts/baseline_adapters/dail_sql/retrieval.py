"""DAIL mask-vector distance and second-round multiset skeleton filtering."""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from .native import skeleton_similarity


def distance_order(train_vectors, target_vector) -> list[int]:
    train = np.asarray(train_vectors)
    target = np.asarray(target_vector).reshape(1, -1)
    if train.ndim != 2 or not np.isfinite(train).all() or not np.isfinite(target).all():
        raise ValueError("Expected finite, two-dimensional training vectors")
    # Same distance implementation as ExampleSelectorTemplate.py, stable ties.
    return np.argsort(euclidean_distances(target, train)[0], kind="stable").tolist()


def choose_examples(distance_ids: list[str], *, k: int, qualified_ids: set[str] | None) -> list[str]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    if qualified_ids is None:
        return distance_ids[:k]
    preferred = [i for i in distance_ids if i in qualified_ids]
    remaining = [i for i in distance_ids if i not in qualified_ids]
    return (preferred + remaining)[:k]


def qualified_examples(skeletons: dict[str, str], target_skeleton: str) -> set[str]:
    return {key for key, skeleton in skeletons.items()
            if skeleton_similarity(skeleton, target_skeleton) >= 0.85}
