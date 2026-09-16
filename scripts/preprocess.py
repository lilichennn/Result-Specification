from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from result_contract.data_preprocess import (
    preprocess_bird,
    preprocess_bird_interact,
    preprocess_spider,
)


DEFAULT_DATASET_ROOTS = {
    "bird": Path("BIRD"),
    "birdinteract": Path("BIRD-Interact") / "BIRD-Interact-ADK",
    "spider": Path("Spider"),
}
DEFAULT_LIVESQLBENCH_ROOT = Path("livesqlbench-base-full-v1")
SUPPORTED_SPLITS = {
    "bird": {"dev"},
    "spider": {"dev", "test"},
    "birdinteract": {"lite", "full"},
}


def preprocess_dataset(
    dataset: str,
    split: str,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    livesqlbench_root: str | Path | None = None,
) -> dict[str, Any]:
    """Preprocess a split; relative input paths are resolved from the working directory."""
    dataset = dataset.lower()
    split = split.lower()
    _validate_dataset_split(dataset, split)
    if livesqlbench_root is not None and (dataset, split) != ("birdinteract", "full"):
        raise ValueError("--livesqlbench-root is only supported for birdinteract/full")

    resolved_dataset_root = Path(
        dataset_root if dataset_root is not None else DEFAULT_DATASET_ROOTS[dataset]
    )
    # Preserve the existing BIRD-Interact workspace names used by RC generation.
    output_dataset = "bird_interact" if dataset == "birdinteract" else dataset
    resolved_output_dir = Path(
        output_dir
        if output_dir is not None
        else SCRIPT_DIR / f"{output_dataset}_{split}" / "preprocessed_data"
    )

    if dataset == "bird":
        summary = preprocess_bird(
            bird_root=resolved_dataset_root,
            split=split,
            output_dir=resolved_output_dir,
        )
    elif dataset == "birdinteract":
        summary = preprocess_bird_interact(
            interact_root=resolved_dataset_root,
            variant=split,
            output_dir=resolved_output_dir,
            livesqlbench_root=(
                Path(livesqlbench_root)
                if livesqlbench_root is not None
                else DEFAULT_LIVESQLBENCH_ROOT
            ) if split == "full" else None,
        )
    elif dataset == "spider":
        summary = preprocess_spider(
            spider_root=resolved_dataset_root,
            split=split,
            output_dir=resolved_output_dir,
        )
    else:
        raise ValueError(f"No preprocessor implemented for dataset: {dataset!r}")

    return {"dataset": dataset, "split": split, **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a benchmark split for RC generation."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(sorted(SUPPORTED_SPLITS)),
        help="Dataset to preprocess. choose from [bird, spider, birdinteract]",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("dev", "test", "lite", "full"),
        help="Dataset split to preprocess.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Override the dataset root; relative paths use the working directory. "
            "Defaults: bird=BIRD, spider=Spider, "
            "birdinteract=BIRD-Interact/BIRD-Interact-ADK."
        ),
    )
    parser.add_argument(
        "--livesqlbench-root",
        type=Path,
        default=None,
        help=(
            "For birdinteract/full only: LiveSQLBench root, default "
            "livesqlbench-base-full-v1 relative to the working directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the output directory; defaults to "
            "<this script's directory>/<dataset>_<split>/preprocessed_data "
            "(birdinteract uses bird_interact_<split>). "
            "An explicit relative path uses the working directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preprocess_dataset(
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        livesqlbench_root=args.livesqlbench_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate_dataset_split(dataset: str, split: str) -> None:
    if dataset not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}; expected one of "
            f"{sorted(SUPPORTED_SPLITS)}"
        )
    if split not in SUPPORTED_SPLITS[dataset]:
        raise ValueError(
            f"Unsupported split {split!r} for {dataset!r}; expected one of "
            f"{sorted(SUPPORTED_SPLITS[dataset])}"
        )


if __name__ == "__main__":
    main()
