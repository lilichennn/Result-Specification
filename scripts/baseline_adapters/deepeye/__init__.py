"""BIRD-Interact integration surfaces for the unmodified DeepEye baseline."""

from importlib import import_module

__all__ = [
    "BirdInteractDataItem",
    "BirdInteractDataset",
    "BirdInteractDatasetConfig",
    "build_postgres_value_index",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".postgres_index" if name == "build_postgres_value_index" else ".dataset"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
