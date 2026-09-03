"""Benchmark-specific data preprocessing."""

from .bird import preprocess_bird
from .spider import preprocess_spider

__all__ = ["preprocess_bird", "preprocess_spider"]
